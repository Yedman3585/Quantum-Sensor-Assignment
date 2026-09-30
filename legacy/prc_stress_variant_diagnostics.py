import argparse
import json
import logging
import os
import time
import warnings
from datetime import datetime

import numpy as np

warnings.filterwarnings("ignore")

UNCOVERED_PENALTY = 15.0
OVERLOAD_PENALTY = 8.0


def configure_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return logging.getLogger(__name__)


class PrcStressVariantDiagnostic:
    def __init__(
        self,
        variant="base",
        n_cameras=20000,
        n_servers=800,
        batch_size=80,
        max_servers_per_batch=20,
        random_seed=42,
        capacity_scale=1.0,
        log_root="logs_prc_stress_variant_diagnostics",
        log_every=20,
        coverage_stop_threshold=0.995,
        guard_weight=45.0,
        conflict_weight=180.0,
        anneal_reads=4,
        anneal_sweeps=80,
    ):
        self.variant = variant
        self.use_guard = "guard" in variant
        self.use_conflict = "conflict" in variant
        self.use_decoder = "decoder" in variant
        self.n_cameras = n_cameras
        self.n_servers = n_servers
        self.batch_size = batch_size
        self.max_servers_per_batch = max_servers_per_batch
        self.random_seed = random_seed
        self.capacity_scale = capacity_scale
        self.log_root = log_root
        self.log_every = log_every
        self.coverage_stop_threshold = coverage_stop_threshold
        self.guard_weight = guard_weight
        self.conflict_weight = conflict_weight
        self.anneal_reads = anneal_reads
        self.anneal_sweeps = anneal_sweeps
        self.rng = np.random.default_rng(random_seed + 1701)

        self.priority = None
        self.weight_mbps = None
        self.load_gflops = None
        self.camera_x = None
        self.camera_y = None
        self.server_x = None
        self.server_y = None
        self.base_initial_capacity = None
        self.initial_capacity = None
        self.remaining_capacity = None
        self.cost_matrix = None
        self.assignment_matrix = None
        self.assigned_cameras = set()

        self.total_load = 0.0
        self.base_total_capacity = 0.0
        self.total_capacity = 0.0
        self.utilization_percent = 0.0

        variant_dir = f"logs_prc_{variant}_b{batch_size}_s{str(capacity_scale).replace('.', 'p')}_m{max_servers_per_batch}"
        self.log_dir = os.path.join(log_root, variant_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.progress_log = os.path.join(self.log_dir, f"progress_{self.run_id}.jsonl")
        self.summary_log = os.path.join(self.log_dir, f"summary_{self.run_id}.json")
        self.logger = configure_logging(os.path.join(self.log_dir, f"run_{self.run_id}.log"))

        self.processed_batches = 0
        self.successful_batches = 0
        self.failed_batches = 0
        self.total_solver_time = 0.0
        self.total_capacity_rejected_raw = 0
        self.total_zero_selection_raw = 0
        self.total_multi_selection_raw = 0
        self.total_raw_selected_variables = 0
        self.total_candidate_pairs = 0
        self.total_feasible_candidate_pairs = 0
        self.selection_rows = []

    def generate_realistic_data(self):
        self.logger.info(
            "PRC diagnostic variant=%s cameras=%d servers=%d capacity_scale=%.3f batch=%d candidates=%d",
            self.variant,
            self.n_cameras,
            self.n_servers,
            self.capacity_scale,
            self.batch_size,
            self.max_servers_per_batch,
        )
        np.random.seed(self.random_seed)

        self.priority = np.random.choice([3, 2, 1], size=self.n_cameras, p=[0.15, 0.25, 0.6])
        self.weight_mbps = np.zeros(self.n_cameras)
        self.load_gflops = np.zeros(self.n_cameras)

        high = self.priority == 3
        medium = self.priority == 2
        low = self.priority == 1

        self.weight_mbps[high] = np.random.uniform(4, 7, np.sum(high))
        self.load_gflops[high] = np.random.uniform(8, 15, np.sum(high))
        self.weight_mbps[medium] = np.random.uniform(2, 4, np.sum(medium))
        self.load_gflops[medium] = np.random.uniform(4, 8, np.sum(medium))
        self.weight_mbps[low] = np.random.uniform(0.4, 0.8, np.sum(low))
        self.load_gflops[low] = np.random.uniform(1, 3, np.sum(low))

        self.camera_x = np.random.uniform(0, 1000, self.n_cameras)
        self.camera_y = np.random.uniform(0, 1000, self.n_cameras)
        self.server_x = np.random.uniform(0, 1000, self.n_servers)
        self.server_y = np.random.uniform(0, 1000, self.n_servers)

        server_types = np.random.choice([3, 2, 1], size=self.n_servers, p=[0.1, 0.3, 0.6])
        capacities = np.zeros(self.n_servers)
        capacities[server_types == 3] = np.random.uniform(800, 1000, np.sum(server_types == 3))
        capacities[server_types == 2] = np.random.uniform(400, 800, np.sum(server_types == 2))
        capacities[server_types == 1] = np.random.uniform(200, 400, np.sum(server_types == 1))
        self.base_initial_capacity = capacities.copy()
        self.initial_capacity = capacities * self.capacity_scale
        self.remaining_capacity = self.initial_capacity.copy()

        self._build_cost_matrix()

        self.total_load = float(self.load_gflops.sum())
        self.base_total_capacity = float(self.base_initial_capacity.sum())
        self.total_capacity = float(self.initial_capacity.sum())
        self.utilization_percent = self.total_load / self.total_capacity * 100.0
        self.logger.info(
            "total load %.1f, base capacity %.1f, scaled capacity %.1f, utilization %.1f%%",
            self.total_load,
            self.base_total_capacity,
            self.total_capacity,
            self.utilization_percent,
        )

    def _build_cost_matrix(self):
        distances = np.hypot(
            self.camera_x[:, None] - self.server_x[None, :],
            self.camera_y[:, None] - self.server_y[None, :],
        )
        distances_norm = distances / (np.max(distances) + 1e-12)
        priorities_norm = (3 - self.priority) / 2.0
        loads_norm = self.load_gflops / (np.max(self.load_gflops) + 1e-12)
        capacities_inv = 1.0 / (self.initial_capacity + 1e-9)
        capacities_norm = capacities_inv / (np.max(capacities_inv) + 1e-12)
        self.cost_matrix = (
            0.40 * distances_norm
            + 0.35 * loads_norm[:, None]
            + 0.20 * priorities_norm[:, None]
            + 0.05 * capacities_norm[None, :]
        )
        self.cost_matrix = (self.cost_matrix - np.min(self.cost_matrix)) / (
            np.max(self.cost_matrix) - np.min(self.cost_matrix) + 1e-9
        )

    def run(self):
        total_start = time.time()
        self.assignment_matrix = np.zeros((self.n_cameras, self.n_servers), dtype=np.int8)
        self.remaining_capacity = self.initial_capacity.copy()
        self.assigned_cameras = set()
        priority_scores = self.priority * self.load_gflops
        sorted_indices = np.argsort(-priority_scores)
        total_batches = int(np.ceil(self.n_cameras / self.batch_size))

        for batch_idx in range(total_batches):
            if len(self.assigned_cameras) / self.n_cameras > self.coverage_stop_threshold:
                self.logger.info("%.2f%% coverage threshold reached, completing", self.coverage_stop_threshold * 100.0)
                break

            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, self.n_cameras)
            batch_indices = sorted_indices[start_idx:end_idx]
            if len(batch_indices) == 0:
                continue

            top_servers, selection_stats = self.select_prc_servers(batch_indices)
            self.processed_batches += 1
            self.total_candidate_pairs += selection_stats["candidate_pairs"]
            self.total_feasible_candidate_pairs += selection_stats["feasible_candidate_pairs"]
            self.selection_rows.append(selection_stats)

            solver_start = time.time()
            linear = self.build_prc_linear(batch_indices, top_servers)
            choices = self.solve_assignment_level(linear, batch_indices, top_servers)
            batch_solution, raw_metrics = self.decode_choices(choices, linear, batch_indices, top_servers)
            solver_time = time.time() - solver_start
            self.total_solver_time += solver_time

            batch_assigned, assignments = self.commit_batch(batch_solution, batch_indices, top_servers)
            batch_failed = batch_assigned < max(1, int(0.5 * len(batch_indices)))
            if batch_failed:
                self.failed_batches += 1
            else:
                self.successful_batches += 1

            self.total_capacity_rejected_raw += raw_metrics["capacity_rejected_raw"]
            self.total_zero_selection_raw += raw_metrics["zero_selection_raw"]
            self.total_multi_selection_raw += raw_metrics["multi_selection_raw"]
            self.total_raw_selected_variables += raw_metrics["raw_selected_variables"]

            coverage = len(self.assigned_cameras) / self.n_cameras * 100.0
            success_rate = self.successful_batches / self.processed_batches * 100.0
            self.log_progress(
                batch_idx,
                batch_assigned,
                coverage,
                success_rate,
                solver_time,
                selection_stats,
                raw_metrics,
                assignments,
            )
            if (batch_idx + 1) % self.log_every == 0:
                self.logger.info(
                    "batch %d: assigned %d, coverage %.2f%%, success %.1f%%, rejected %d",
                    batch_idx + 1,
                    batch_assigned,
                    coverage,
                    success_rate,
                    self.total_capacity_rejected_raw,
                )

        total_time = time.time() - total_start
        quality = self.calculate_quality(self.assignment_matrix)
        summary = self.build_summary(total_time, quality)
        with open(self.summary_log, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        self.print_summary(summary)
        return summary

    def select_prc_servers(self, batch_indices):
        batch_loads = self.load_gflops[batch_indices]
        min_load = float(np.min(batch_loads))
        avg_load = float(np.mean(batch_loads))
        valid_servers = np.where(self.remaining_capacity >= avg_load * 0.7)[0]
        if len(valid_servers) == 0:
            valid_servers = np.where(self.remaining_capacity >= min_load)[0]
        if len(valid_servers) == 0:
            return np.array([], dtype=int), self.selection_stats(batch_indices, np.array([], dtype=int))

        mean_cost = np.mean(self.cost_matrix[batch_indices, :], axis=0)
        capacity_score = self.remaining_capacity[valid_servers] / (
            np.max(self.remaining_capacity[valid_servers]) + 1e-12
        )
        feasible_counts = np.sum(batch_loads[:, None] <= self.remaining_capacity[valid_servers][None, :], axis=0)
        feasibility_score = feasible_counts / max(1, len(batch_indices))
        cost_score = 1.0 - mean_cost[valid_servers]
        score = capacity_score * 0.3 + cost_score * 0.5 + feasibility_score * 0.2
        top_indices = np.argsort(-score)[: self.max_servers_per_batch]
        top_servers = valid_servers[top_indices].astype(int)
        return top_servers, self.selection_stats(batch_indices, top_servers)

    def selection_stats(self, batch_indices, top_servers):
        batch_loads = self.load_gflops[batch_indices]
        feasible_pairs = (
            int(np.sum(batch_loads[:, None] <= self.remaining_capacity[top_servers][None, :]))
            if len(top_servers)
            else 0
        )
        feasible_server_count = int(np.sum(np.any(batch_loads[:, None] <= self.remaining_capacity[None, :], axis=0)))
        return {
            "candidate_servers": int(len(top_servers)),
            "candidate_pairs": int(len(batch_indices) * len(top_servers)),
            "feasible_candidate_pairs": feasible_pairs,
            "selection_feasible_server_count": feasible_server_count,
        }

    def build_prc_linear(self, batch_indices, top_servers):
        linear = np.full((len(batch_indices), len(top_servers)), 100.0, dtype=float)
        if len(top_servers) == 0:
            return linear
        for i, cam_idx in enumerate(batch_indices):
            load = float(self.load_gflops[cam_idx])
            priority_weight = float(4 - self.priority[cam_idx])
            for j, server_idx in enumerate(top_servers):
                residual = float(self.remaining_capacity[server_idx])
                cost = float(self.cost_matrix[cam_idx, server_idx])
                if load <= residual:
                    value = -25.0 * (1.0 - cost) * priority_weight
                    if self.use_guard:
                        value += self.guard_weight * load / (residual + 1e-12)
                    linear[i, j] = value
        return linear

    def solve_assignment_level(self, linear, batch_indices, top_servers):
        n_batch, n_top = linear.shape
        if n_top == 0:
            return np.full(n_batch, -1, dtype=int)

        initial = np.argmin(linear, axis=1).astype(int)
        initial[np.min(linear, axis=1) >= 99.0] = -1
        if not self.use_conflict:
            return initial

        best_choices = initial.copy()
        best_energy = self.assignment_energy(best_choices, linear, batch_indices, top_servers)
        loads = self.load_gflops[batch_indices]
        residual = self.remaining_capacity[top_servers]
        residual_sq = np.maximum(residual, 1e-9) ** 2

        for _ in range(max(1, self.anneal_reads)):
            choices = initial.copy()
            server_loads = self.local_server_loads(choices, loads, n_top)
            current_energy = self.assignment_energy(choices, linear, batch_indices, top_servers)
            temperature = 4.0

            for sweep in range(max(1, self.anneal_sweeps)):
                for i in self.rng.permutation(n_batch):
                    old = int(choices[i])
                    best_delta = 0.0
                    best_new = old
                    for new in range(n_top):
                        if new == old or linear[i, new] >= 99.0:
                            continue
                        delta = linear[i, new] - (linear[i, old] if old >= 0 else 0.0)
                        if old >= 0:
                            delta -= self.conflict_weight * loads[i] * max(0.0, server_loads[old] - loads[i]) / residual_sq[old]
                        delta += self.conflict_weight * loads[i] * server_loads[new] / residual_sq[new]
                        if delta < best_delta:
                            best_delta = delta
                            best_new = new

                    accept = best_new != old and (
                        best_delta < 0.0 or self.rng.random() < np.exp(-best_delta / max(temperature, 1e-9))
                    )
                    if accept:
                        if old >= 0:
                            server_loads[old] -= loads[i]
                        choices[i] = best_new
                        server_loads[best_new] += loads[i]
                        current_energy += best_delta
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_choices = choices.copy()
                temperature *= 0.94
        return best_choices

    def assignment_energy(self, choices, linear, batch_indices, top_servers):
        energy = 0.0
        loads = self.load_gflops[batch_indices]
        server_loads = np.zeros(len(top_servers), dtype=float)
        for i, choice in enumerate(choices):
            if choice < 0:
                energy += UNCOVERED_PENALTY
                continue
            energy += float(linear[i, choice])
            server_loads[choice] += loads[i]
        if self.use_conflict and len(top_servers):
            residual_sq = np.maximum(self.remaining_capacity[top_servers], 1e-9) ** 2
            energy += 0.5 * self.conflict_weight * float(np.sum((server_loads ** 2) / residual_sq))
        return energy

    @staticmethod
    def local_server_loads(choices, loads, n_top):
        server_loads = np.zeros(n_top, dtype=float)
        for i, choice in enumerate(choices):
            if choice >= 0:
                server_loads[choice] += loads[i]
        return server_loads

    def decode_choices(self, choices, linear, batch_indices, top_servers):
        if self.use_decoder:
            return self.decode_with_capacity_alternatives(choices, linear, batch_indices, top_servers)
        return self.decode_like_prc(choices, batch_indices, top_servers)

    def decode_like_prc(self, choices, batch_indices, top_servers):
        assignment = np.zeros((len(batch_indices), len(top_servers)), dtype=np.int8)
        candidates = []
        raw_counts = np.zeros(len(batch_indices), dtype=int)
        for i, choice in enumerate(choices):
            if choice < 0:
                continue
            server_idx = int(top_servers[choice])
            cam_idx = int(batch_indices[i])
            cost = float(self.cost_matrix[cam_idx, server_idx])
            priority = float(self.priority[cam_idx])
            load = float(self.load_gflops[cam_idx])
            residual = float(self.remaining_capacity[server_idx])
            capacity_utilization = 1.0 - (load / residual) if residual > 0 else 0.0
            score = priority * 0.4 + (1.0 - cost) * 0.4 + capacity_utilization * 0.2
            raw_counts[i] += 1
            candidates.append((i, choice, server_idx, score))

        candidates.sort(key=lambda item: -item[3])
        used_cameras = set()
        server_loads = {int(server_idx): 0.0 for server_idx in top_servers}
        capacity_rejected = 0
        for i, choice, server_idx, _score in candidates:
            if i in used_cameras:
                continue
            cam_idx = int(batch_indices[i])
            load = float(self.load_gflops[cam_idx])
            if server_loads[server_idx] + load <= self.remaining_capacity[server_idx]:
                assignment[i, choice] = 1
                used_cameras.add(i)
                server_loads[server_idx] += load
            else:
                capacity_rejected += 1
        return assignment, {
            "raw_selected_variables": int(np.sum(raw_counts)),
            "zero_selection_raw": int(np.sum(raw_counts == 0)),
            "multi_selection_raw": 0,
            "capacity_rejected_raw": int(capacity_rejected),
            "residual_blind_rejected_assignments": int(capacity_rejected),
        }

    def decode_with_capacity_alternatives(self, choices, linear, batch_indices, top_servers):
        assignment = np.zeros((len(batch_indices), len(top_servers)), dtype=np.int8)
        raw_selected = int(np.sum(choices >= 0))
        server_loads = np.zeros(len(top_servers), dtype=float)
        order = np.argsort(-(self.priority[batch_indices] * self.load_gflops[batch_indices]))
        capacity_rejected = 0

        for i in order:
            cam_idx = int(batch_indices[i])
            load = float(self.load_gflops[cam_idx])
            candidate_order = np.argsort(linear[i])
            chosen = -1
            for j in candidate_order:
                if linear[i, j] >= 99.0:
                    continue
                server_idx = int(top_servers[j])
                if server_loads[j] + load <= self.remaining_capacity[server_idx]:
                    chosen = int(j)
                    break
            if chosen >= 0:
                assignment[i, chosen] = 1
                server_loads[chosen] += load
            elif choices[i] >= 0:
                capacity_rejected += 1

        return assignment, {
            "raw_selected_variables": raw_selected,
            "zero_selection_raw": int(len(batch_indices) - raw_selected),
            "multi_selection_raw": 0,
            "capacity_rejected_raw": int(capacity_rejected),
            "residual_blind_rejected_assignments": int(capacity_rejected),
        }

    def commit_batch(self, batch_solution, batch_indices, top_servers):
        assignments = []
        batch_assigned = 0
        for i, cam_idx in enumerate(batch_indices):
            selected = np.where(batch_solution[i] == 1)[0]
            if len(selected) == 0:
                continue
            server_idx = int(top_servers[int(selected[0])])
            cam_load = float(self.load_gflops[cam_idx])
            if self.remaining_capacity[server_idx] >= cam_load and int(cam_idx) not in self.assigned_cameras:
                self.assignment_matrix[cam_idx, server_idx] = 1
                self.remaining_capacity[server_idx] -= cam_load
                self.assigned_cameras.add(int(cam_idx))
                batch_assigned += 1
                assignments.append({"cam_id": int(cam_idx), "server_id": int(server_idx)})
            else:
                self.total_capacity_rejected_raw += 1
        return batch_assigned, assignments

    def calculate_quality(self, assignment):
        assigned_mask = np.any(assignment, axis=1)
        assigned_indices = np.where(assigned_mask)[0]
        total_cost = 0.0
        if len(assigned_indices) > 0:
            assigned_servers = np.argmax(assignment[assigned_indices], axis=1)
            priority_weight = 4 - self.priority[assigned_indices]
            total_cost = float(np.sum(self.cost_matrix[assigned_indices, assigned_servers] * priority_weight))
        uncovered = int(self.n_cameras - np.sum(assigned_mask))
        uncovered_penalty = float(uncovered * UNCOVERED_PENALTY)
        server_loads = assignment.T.astype(float).dot(self.load_gflops)
        overload = np.maximum(0.0, server_loads - self.initial_capacity)
        overload_penalty = float(np.sum(overload * OVERLOAD_PENALTY))
        return {
            "assignment_cost": total_cost,
            "uncovered_cameras": uncovered,
            "uncovered_penalty": uncovered_penalty,
            "overload_penalty": overload_penalty,
            "objective_value": total_cost + uncovered_penalty + overload_penalty,
            "covered_cameras": int(np.sum(assigned_mask)),
            "coverage_percent": float(np.sum(assigned_mask) / self.n_cameras * 100.0),
            "overloaded_servers": int(np.sum(overload > 1e-9)),
        }

    def build_summary(self, total_time, quality):
        server_loads = self.assignment_matrix.T.astype(float).dot(self.load_gflops)
        residual = self.initial_capacity - server_loads
        residual_positive = residual[residual > 1e-9]
        summary = {
            "run_id": self.run_id,
            "formulation": f"PRC-Diagnostic-{self.variant}",
            "solver": "assignment-level-diagnostic",
            "variant": self.variant,
            "n_cameras": self.n_cameras,
            "n_servers": self.n_servers,
            "batch_size": self.batch_size,
            "max_servers_per_batch": self.max_servers_per_batch,
            "random_seed": self.random_seed,
            "capacity_scale": float(self.capacity_scale),
            "base_total_capacity": float(self.base_total_capacity),
            "total_capacity": float(self.total_capacity),
            "total_load": float(self.total_load),
            "utilization_percent": float(self.utilization_percent),
            "guard_weight": float(self.guard_weight),
            "conflict_weight": float(self.conflict_weight),
            "anneal_reads": int(self.anneal_reads),
            "anneal_sweeps": int(self.anneal_sweeps),
            "total_time_sec": float(total_time),
            "solver_time_sec": float(self.total_solver_time),
            "throughput_cam_per_sec": float(quality["covered_cameras"] / total_time) if total_time else 0.0,
            "processed_batches": int(self.processed_batches),
            "successful_batches": int(self.successful_batches),
            "failed_batches": int(self.failed_batches),
            "solver_success_rate_percent": float(self.successful_batches / self.processed_batches * 100.0)
            if self.processed_batches
            else 0.0,
            "fallback_count": 0,
            "avg_feasible_candidate_pairs_per_batch": float(
                self.total_feasible_candidate_pairs / self.processed_batches if self.processed_batches else 0.0
            ),
            "capacity_rejected_raw_assignments": int(self.total_capacity_rejected_raw),
            "zero_selection_raw": int(self.total_zero_selection_raw),
            "multi_selection_raw": int(self.total_multi_selection_raw),
            "raw_selected_variables": int(self.total_raw_selected_variables),
            "used_capacity": float(np.sum(server_loads)),
            "unused_capacity": float(np.sum(np.maximum(0.0, residual))),
            "used_capacity_percent": float(np.sum(server_loads) / self.total_capacity * 100.0) if self.total_capacity else 0.0,
            "active_servers": int(np.sum(server_loads > 1e-9)),
            "residual_capacity_min": float(np.min(residual)) if len(residual) else 0.0,
            "residual_capacity_mean": float(np.mean(residual)) if len(residual) else 0.0,
            "residual_capacity_std": float(np.std(residual)) if len(residual) else 0.0,
            "residual_capacity_cv_positive": float(np.std(residual_positive) / np.mean(residual_positive))
            if len(residual_positive) and np.mean(residual_positive) > 0
            else 0.0,
            "avg_candidate_servers": float(
                np.mean([row["candidate_servers"] for row in self.selection_rows]) if self.selection_rows else 0.0
            ),
            "avg_candidate_pairs": float(
                np.mean([row["candidate_pairs"] for row in self.selection_rows]) if self.selection_rows else 0.0
            ),
            "avg_selection_feasible_server_count": float(
                np.mean([row["selection_feasible_server_count"] for row in self.selection_rows])
                if self.selection_rows
                else 0.0
            ),
            **quality,
        }
        return summary

    def log_progress(
        self,
        batch_idx,
        batch_assigned,
        coverage,
        success_rate,
        solver_time,
        selection_stats,
        raw_metrics,
        assignments,
    ):
        log_entry = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "formulation": f"PRC-Diagnostic-{self.variant}",
            "solver": "assignment-level-diagnostic",
            "variant": self.variant,
            "capacity_scale": float(self.capacity_scale),
            "utilization_percent": float(self.utilization_percent),
            "batch_idx": int(batch_idx),
            "batch_assigned": int(batch_assigned),
            "coverage_percent": float(coverage),
            "solver_success_rate": float(success_rate),
            "batch_failed": bool(batch_assigned < max(1, int(0.5 * self.batch_size))),
            "fallback_used": False,
            "qubo_time_sec": 0.0,
            "annealing_time_sec": float(solver_time),
            "solver_time_sec": float(solver_time),
            "energy": None,
            "best_energy": None,
            "assignments": assignments,
            **selection_stats,
            **raw_metrics,
        }
        with open(self.progress_log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(log_entry) + "\n")

    @staticmethod
    def print_summary(summary):
        print("\n" + "=" * 90)
        print(f"{summary['formulation']} RESULTS")
        print("=" * 90)
        print(
            f"Capacity scale: {summary['capacity_scale']:.3f} | "
            f"Utilization: {summary['utilization_percent']:.2f}%"
        )
        print(f"Coverage: {summary['covered_cameras']}/{summary['n_cameras']} ({summary['coverage_percent']:.2f}%)")
        print(f"Objective: {summary['objective_value']:.3f}")
        print(f"Assignment cost: {summary['assignment_cost']:.3f}")
        print(f"Uncovered penalty: {summary['uncovered_penalty']:.3f}")
        print(f"Overload penalty: {summary['overload_penalty']:.3f}")
        print(f"Total time: {summary['total_time_sec']:.3f}s")
        print(f"Rejected raw assignments: {summary['capacity_rejected_raw_assignments']}")
        print(f"Failed batches: {summary['failed_batches']}")
        print("=" * 90)


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnostic PRC-QUBO stress variants without external QUBO solvers")
    parser.add_argument(
        "--variant",
        choices=[
            "base",
            "guard",
            "conflict",
            "guard-conflict",
            "decoder",
            "guard-decoder",
            "conflict-decoder",
            "guard-conflict-decoder",
        ],
        default="base",
    )
    parser.add_argument("--n-cameras", type=int, default=20000)
    parser.add_argument("--n-servers", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--max-servers-per-batch", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--capacity-scale", type=float, default=1.0)
    parser.add_argument("--log-root", default="logs_prc_stress_variant_diagnostics")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--coverage-stop-threshold", type=float, default=0.995)
    parser.add_argument("--guard-weight", type=float, default=45.0)
    parser.add_argument("--conflict-weight", type=float, default=180.0)
    parser.add_argument("--anneal-reads", type=int, default=4)
    parser.add_argument("--anneal-sweeps", type=int, default=80)
    return parser.parse_args()


def main():
    args = parse_args()
    experiment = PrcStressVariantDiagnostic(
        variant=args.variant,
        n_cameras=args.n_cameras,
        n_servers=args.n_servers,
        batch_size=args.batch_size,
        max_servers_per_batch=args.max_servers_per_batch,
        random_seed=args.seed,
        capacity_scale=args.capacity_scale,
        log_root=args.log_root,
        log_every=args.log_every,
        coverage_stop_threshold=args.coverage_stop_threshold,
        guard_weight=args.guard_weight,
        conflict_weight=args.conflict_weight,
        anneal_reads=args.anneal_reads,
        anneal_sweeps=args.anneal_sweeps,
    )
    experiment.generate_realistic_data()
    experiment.run()


if __name__ == "__main__":
    main()
