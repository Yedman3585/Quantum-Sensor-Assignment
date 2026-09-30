import argparse
import json
import logging
import os
import time
import warnings
from datetime import datetime

import numpy as np

warnings.filterwarnings("ignore")

FORMULATION = "AO-QUBO"
SOLVER = "SA"
LOG_FILE = "ao_qubo_sa.log"
LOG_DIR = "logs_ao_qubo_sa"

LAMBDA_ASSIGN = 15.0
LAMBDA_CAPACITY = 25.0

try:
    from neal import SimulatedAnnealingSampler

    HAVE_NEAL = True
except ImportError:
    HAVE_NEAL = False

try:
    import openjij as oj

    HAVE_OPENJIJ = True
except ImportError:
    HAVE_OPENJIJ = False


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class BaselineQuboExperiment:
    def __init__(
        self,
        n_cameras=20000,
        n_servers=800,
        batch_size=80,
        max_servers_per_batch=20,
        random_seed=42,
        num_reads=150,
        num_sweeps=1000,
        trotter=8,
        log_every=20,
        final_opt=False,
    ):
        self.n_cameras = n_cameras
        self.n_servers = n_servers
        self.batch_size = batch_size
        self.max_servers_per_batch = max_servers_per_batch
        self.random_seed = random_seed
        self.num_reads = num_reads
        self.num_sweeps = num_sweeps
        self.trotter = trotter
        self.log_every = log_every
        self.final_opt = final_opt

        self.priority = None
        self.weight_mbps = None
        self.load_gflops = None
        self.camera_x = None
        self.camera_y = None
        self.server_x = None
        self.server_y = None
        self.initial_capacity = None
        self.remaining_capacity = None
        self.cost_matrix = None
        self.assignment_matrix = None
        self.assigned_cameras = set()

        os.makedirs(LOG_DIR, exist_ok=True)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.progress_log = os.path.join(LOG_DIR, f"progress_{self.run_id}.jsonl")
        self.summary_log = os.path.join(LOG_DIR, f"summary_{self.run_id}.json")

        self.total_qubo_time = 0.0
        self.total_solver_time = 0.0
        self.successful_batches = 0
        self.failed_batches = 0
        self.processed_batches = 0
        self.fallback_count = 0
        self.total_capacity_rejected_raw = 0
        self.total_zero_selection_raw = 0
        self.total_multi_selection_raw = 0
        self.total_raw_selected_variables = 0
        self.total_feasible_candidate_pairs = 0
        self.qubo_stat_rows = []

    def generate_realistic_data(self):
        logger.info(
            "%s + %s: data generation for %d cameras and %d servers",
            FORMULATION,
            SOLVER,
            self.n_cameras,
            self.n_servers,
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
        self.initial_capacity = np.zeros(self.n_servers)
        self.initial_capacity[server_types == 3] = np.random.uniform(800, 1000, np.sum(server_types == 3))
        self.initial_capacity[server_types == 2] = np.random.uniform(400, 800, np.sum(server_types == 2))
        self.initial_capacity[server_types == 1] = np.random.uniform(200, 400, np.sum(server_types == 1))
        self.remaining_capacity = self.initial_capacity.copy()

        self._build_cost_matrix()

        total_load = float(self.load_gflops.sum())
        total_capacity = float(self.initial_capacity.sum())
        utilization = total_load / total_capacity * 100.0
        logger.info(
            "total load %.1f, total capacity %.1f, utilization %.1f%%",
            total_load,
            total_capacity,
            utilization,
        )
        return utilization

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
        min_cost = np.min(self.cost_matrix)
        max_cost = np.max(self.cost_matrix)
        self.cost_matrix = (self.cost_matrix - min_cost) / (max_cost - min_cost + 1e-9)

    def select_static_servers(self, batch_indices):
        mean_cost = np.mean(self.cost_matrix[batch_indices, :], axis=0)
        order = np.lexsort((-self.initial_capacity, mean_cost))
        return order[: self.max_servers_per_batch]

    def run(self):
        self._ensure_solver_available()

        total_start = time.time()
        self.assignment_matrix = np.zeros((self.n_cameras, self.n_servers), dtype=np.int8)
        self.remaining_capacity = self.initial_capacity.copy()
        self.assigned_cameras = set()

        priority_scores = self.priority * self.load_gflops
        sorted_indices = np.argsort(-priority_scores)
        total_batches = int(np.ceil(self.n_cameras / self.batch_size))

        logger.info("%s + %s processing %d batches", FORMULATION, SOLVER, total_batches)

        for batch_idx in range(total_batches):
            if len(self.assigned_cameras) / self.n_cameras > 0.995:
                logger.info("99.5%% coverage achieved, completing")
                break

            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, self.n_cameras)
            batch_indices = sorted_indices[start_idx:end_idx]
            if len(batch_indices) == 0:
                continue

            top_servers = self.select_static_servers(batch_indices)
            self.processed_batches += 1

            qubo_start = time.time()
            q, qubo_stats = self.build_qubo(batch_indices, top_servers)
            qubo_time = time.time() - qubo_start
            self.total_qubo_time += qubo_time

            response, solver_time, solver_ok, failed_reason = self.solve_qubo(q)
            self.total_solver_time += solver_time

            if response is None:
                batch_solution = np.zeros((len(batch_indices), len(top_servers)), dtype=np.int8)
                raw_metrics = self.empty_raw_metrics(len(batch_indices))
                energy = None
                batch_failed = True
            else:
                energy = float(response.first.energy)
                batch_solution, raw_metrics = self.decode_solution(response, batch_indices, top_servers)
                valid_decoded = int(np.sum(np.any(batch_solution, axis=1)))
                batch_failed = (not solver_ok) or valid_decoded < max(1, int(0.5 * len(batch_indices)))
                if batch_failed and not failed_reason:
                    failed_reason = "weak_decoded_solution"

            batch_assigned, assignments_in_batch = self.commit_batch(batch_solution, batch_indices, top_servers)

            if batch_failed:
                self.failed_batches += 1
            else:
                self.successful_batches += 1

            self.total_capacity_rejected_raw += raw_metrics["capacity_rejected_raw"]
            self.total_zero_selection_raw += raw_metrics["zero_selection_raw"]
            self.total_multi_selection_raw += raw_metrics["multi_selection_raw"]
            self.total_raw_selected_variables += raw_metrics["raw_selected_variables"]
            self.total_feasible_candidate_pairs += qubo_stats["feasible_candidate_pairs"]
            self.qubo_stat_rows.append(qubo_stats)

            coverage = len(self.assigned_cameras) / self.n_cameras * 100.0
            success_rate = self.successful_batches / self.processed_batches * 100.0

            self.log_progress(
                batch_idx=batch_idx,
                batch_assigned=batch_assigned,
                coverage=coverage,
                success_rate=success_rate,
                qubo_time=qubo_time,
                solver_time=solver_time,
                energy=energy,
                qubo_stats=qubo_stats,
                raw_metrics=raw_metrics,
                assignments=assignments_in_batch,
                batch_failed=batch_failed,
                failed_reason=failed_reason,
            )

            if (batch_idx + 1) % self.log_every == 0:
                logger.info(
                    "batch %d: assigned %d, coverage %.2f%%, success %.1f%%, failed %d",
                    batch_idx + 1,
                    batch_assigned,
                    coverage,
                    success_rate,
                    self.failed_batches,
                )

        if self.final_opt:
            self.optimize_final_solution()

        total_time = time.time() - total_start
        quality = self.calculate_quality(self.assignment_matrix)
        summary = self.build_summary(total_time, quality)
        self.write_summary(summary)
        self.print_summary(summary)
        return summary

    def _ensure_solver_available(self):
        if SOLVER == "SA" and not HAVE_NEAL:
            raise RuntimeError("dwave-neal is not available; SA baseline cannot run")
        if SOLVER == "SQA" and not HAVE_OPENJIJ:
            raise RuntimeError("openjij is not available; SQA baseline cannot run")

    def build_qubo(self, batch_indices, top_servers):
        if FORMULATION == "AO-QUBO":
            return self.build_ao_qubo(batch_indices, top_servers)
        if FORMULATION == "Static-QCP-QUBO":
            return self.build_static_qcp_qubo(batch_indices, top_servers)
        raise ValueError(f"Unknown formulation: {FORMULATION}")

    def build_ao_qubo(self, batch_indices, top_servers):
        n_batch = len(batch_indices)
        n_servers_batch = len(top_servers)
        q = {}

        for i, cam_idx in enumerate(batch_indices):
            for j, server_idx in enumerate(top_servers):
                var = self.var_name(i, server_idx)
                q[(var, var)] = float(self.cost_matrix[cam_idx, server_idx] - LAMBDA_ASSIGN)

        for i in range(n_batch):
            for j1 in range(n_servers_batch):
                var1 = self.var_name(i, top_servers[j1])
                for j2 in range(j1 + 1, n_servers_batch):
                    var2 = self.var_name(i, top_servers[j2])
                    self.add_q(q, var1, var2, 2.0 * LAMBDA_ASSIGN)

        return q, self.collect_qubo_stats(q, batch_indices, top_servers)

    def build_static_qcp_qubo(self, batch_indices, top_servers):
        n_batch = len(batch_indices)
        n_servers_batch = len(top_servers)
        q = {}

        capacity_scale = float(np.max(self.initial_capacity) + 1e-12)
        load_norm = self.load_gflops[batch_indices] / capacity_scale
        capacity_norm = self.initial_capacity[top_servers] / capacity_scale

        for i, cam_idx in enumerate(batch_indices):
            for j, server_idx in enumerate(top_servers):
                var = self.var_name(i, server_idx)
                base = float(self.cost_matrix[cam_idx, server_idx] - LAMBDA_ASSIGN)
                cap_linear = LAMBDA_CAPACITY * (load_norm[i] ** 2 - 2.0 * capacity_norm[j] * load_norm[i])
                q[(var, var)] = base + float(cap_linear)

        for i in range(n_batch):
            for j1 in range(n_servers_batch):
                var1 = self.var_name(i, top_servers[j1])
                for j2 in range(j1 + 1, n_servers_batch):
                    var2 = self.var_name(i, top_servers[j2])
                    self.add_q(q, var1, var2, 2.0 * LAMBDA_ASSIGN)

        for j, server_idx in enumerate(top_servers):
            for i1 in range(n_batch):
                var1 = self.var_name(i1, server_idx)
                for i2 in range(i1 + 1, n_batch):
                    var2 = self.var_name(i2, server_idx)
                    coeff = 2.0 * LAMBDA_CAPACITY * load_norm[i1] * load_norm[i2]
                    self.add_q(q, var1, var2, float(coeff))

        return q, self.collect_qubo_stats(q, batch_indices, top_servers)

    @staticmethod
    def add_q(q, var1, var2, value):
        key = (var1, var2) if var1 <= var2 else (var2, var1)
        q[key] = q.get(key, 0.0) + float(value)

    @staticmethod
    def var_name(local_camera_idx, server_idx):
        return f"x_{local_camera_idx}_{int(server_idx)}"

    def collect_qubo_stats(self, q, batch_indices, top_servers):
        n_variables = int(len(batch_indices) * len(top_servers))
        linear_count = sum(1 for a, b in q if a == b)
        quadratic_count = len(q) - linear_count
        possible_terms = n_variables + n_variables * (n_variables - 1) / 2.0
        coeffs = np.fromiter(q.values(), dtype=float) if q else np.array([0.0])

        feasible_pairs = int(
            np.sum(self.load_gflops[batch_indices][:, None] <= self.remaining_capacity[top_servers][None, :])
        )

        return {
            "qubo_variables": n_variables,
            "linear_coefficient_count": int(linear_count),
            "quadratic_coefficient_count": int(quadratic_count),
            "qubo_coefficient_count": int(len(q)),
            "qubo_density": float(len(q) / possible_terms) if possible_terms else 0.0,
            "coefficient_min": float(np.min(coeffs)),
            "coefficient_max": float(np.max(coeffs)),
            "coefficient_range": float(np.max(coeffs) - np.min(coeffs)),
            "feasible_candidate_pairs": feasible_pairs,
        }

    def solve_qubo(self, q):
        if not q:
            return None, 0.0, False, "empty_qubo"

        start = time.time()
        try:
            if SOLVER == "SA":
                sampler = SimulatedAnnealingSampler()
                response = sampler.sample_qubo(q, num_reads=self.num_reads)
            else:
                sampler = self.make_sqa_sampler()
                response = sampler.sample_qubo(q)
            elapsed = time.time() - start
            return response, elapsed, True, ""
        except Exception as exc:
            elapsed = time.time() - start
            logger.warning("%s solver failed without fallback: %s", SOLVER, exc)
            return None, elapsed, False, str(exc)

    def make_sqa_sampler(self):
        attempts = [
            {"num_reads": min(10, self.num_reads), "num_sweeps": self.num_sweeps, "trotter": self.trotter},
            {"num_sweeps": self.num_sweeps, "trotter": self.trotter},
            {"trotter": self.trotter},
            {},
        ]
        last_error = None
        for kwargs in attempts:
            try:
                return oj.SQASampler(**kwargs)
            except TypeError as exc:
                last_error = exc
        raise last_error

    def decode_solution(self, response, batch_indices, top_servers):
        n_batch = len(batch_indices)
        assignment = np.zeros((n_batch, len(top_servers)), dtype=np.int8)
        sample = response.first.sample

        raw_counts = np.zeros(n_batch, dtype=int)
        selected_by_camera = [[] for _ in range(n_batch)]
        for i, cam_idx in enumerate(batch_indices):
            for j, server_idx in enumerate(top_servers):
                var = self.var_name(i, server_idx)
                if sample.get(var, 0) == 1:
                    raw_counts[i] += 1
                    selected_by_camera[i].append((j, int(server_idx), float(self.cost_matrix[cam_idx, server_idx])))

        local_server_loads = {int(server_idx): 0.0 for server_idx in top_servers}
        capacity_rejected = 0

        for i, selected in enumerate(selected_by_camera):
            if not selected:
                continue
            j, server_idx, _cost = min(selected, key=lambda item: item[2])
            cam_idx = int(batch_indices[i])
            cam_load = float(self.load_gflops[cam_idx])
            if local_server_loads[server_idx] + cam_load <= self.remaining_capacity[server_idx]:
                assignment[i, j] = 1
                local_server_loads[server_idx] += cam_load
            else:
                capacity_rejected += 1

        raw_metrics = {
            "raw_selected_variables": int(np.sum(raw_counts)),
            "zero_selection_raw": int(np.sum(raw_counts == 0)),
            "multi_selection_raw": int(np.sum(raw_counts > 1)),
            "capacity_rejected_raw": int(capacity_rejected),
            "residual_blind_rejected_assignments": int(capacity_rejected),
        }
        return assignment, raw_metrics

    @staticmethod
    def empty_raw_metrics(n_batch):
        return {
            "raw_selected_variables": 0,
            "zero_selection_raw": int(n_batch),
            "multi_selection_raw": 0,
            "capacity_rejected_raw": 0,
            "residual_blind_rejected_assignments": 0,
        }

    def commit_batch(self, batch_solution, batch_indices, top_servers):
        assignments = []
        batch_assigned = 0
        for i, cam_idx in enumerate(batch_indices):
            for j, server_idx in enumerate(top_servers):
                if batch_solution[i, j] != 1:
                    continue
                cam_load = float(self.load_gflops[cam_idx])
                if self.remaining_capacity[server_idx] >= cam_load and cam_idx not in self.assigned_cameras:
                    self.assignment_matrix[cam_idx, server_idx] = 1
                    self.remaining_capacity[server_idx] -= cam_load
                    self.assigned_cameras.add(int(cam_idx))
                    batch_assigned += 1
                    assignments.append({"cam_id": int(cam_idx), "server_id": int(server_idx)})
                break
        return batch_assigned, assignments

    def optimize_final_solution(self):
        logger.info("final local reassignment optimization")
        for iteration in range(3):
            improvements = 0
            for cam_idx in range(self.n_cameras):
                if not np.any(self.assignment_matrix[cam_idx]):
                    continue
                current_server = int(np.argmax(self.assignment_matrix[cam_idx]))
                current_cost = float(self.cost_matrix[cam_idx, current_server])
                cam_load = float(self.load_gflops[cam_idx])
                best_server = current_server
                best_cost = current_cost

                for server_idx in range(self.n_servers):
                    if server_idx == current_server:
                        continue
                    new_cost = float(self.cost_matrix[cam_idx, server_idx])
                    if new_cost < best_cost * 0.98 and self.remaining_capacity[server_idx] >= cam_load:
                        best_server = server_idx
                        best_cost = new_cost

                if best_server != current_server:
                    self.assignment_matrix[cam_idx, current_server] = 0
                    self.assignment_matrix[cam_idx, best_server] = 1
                    self.remaining_capacity[current_server] += cam_load
                    self.remaining_capacity[best_server] -= cam_load
                    improvements += 1

            logger.info("final optimization iteration %d: %d improvements", iteration + 1, improvements)
            if improvements == 0:
                break

    def calculate_quality(self, assignment):
        assigned_mask = np.any(assignment, axis=1)
        assigned_indices = np.where(assigned_mask)[0]

        total_cost = 0.0
        if len(assigned_indices) > 0:
            assigned_servers = np.argmax(assignment[assigned_indices], axis=1)
            priority_weight = 4 - self.priority[assigned_indices]
            total_cost = float(
                np.sum(self.cost_matrix[assigned_indices, assigned_servers] * priority_weight)
            )

        uncovered = int(self.n_cameras - np.sum(assigned_mask))
        uncovered_penalty = float(uncovered * 15.0)
        server_loads = assignment.T.astype(float).dot(self.load_gflops)
        overload = np.maximum(0.0, server_loads - self.initial_capacity)
        overload_penalty = float(np.sum(overload * 8.0))
        objective = total_cost + uncovered_penalty + overload_penalty

        return {
            "assignment_cost": total_cost,
            "uncovered_cameras": uncovered,
            "uncovered_penalty": uncovered_penalty,
            "overload_penalty": overload_penalty,
            "objective_value": objective,
            "covered_cameras": int(np.sum(assigned_mask)),
            "coverage_percent": float(np.sum(assigned_mask) / self.n_cameras * 100.0),
        }

    def build_summary(self, total_time, quality):
        avg_feasible_pairs = (
            self.total_feasible_candidate_pairs / self.processed_batches if self.processed_batches else 0.0
        )
        avg_qubo_stats = self.average_qubo_stats()
        summary = {
            "run_id": self.run_id,
            "formulation": FORMULATION,
            "solver": SOLVER,
            "n_cameras": self.n_cameras,
            "n_servers": self.n_servers,
            "batch_size": self.batch_size,
            "max_servers_per_batch": self.max_servers_per_batch,
            "random_seed": self.random_seed,
            "total_time_sec": float(total_time),
            "qubo_time_sec": float(self.total_qubo_time),
            "solver_time_sec": float(self.total_solver_time),
            "throughput_cam_per_sec": float(quality["covered_cameras"] / total_time) if total_time > 0 else 0.0,
            "processed_batches": int(self.processed_batches),
            "successful_batches": int(self.successful_batches),
            "failed_batches": int(self.failed_batches),
            "solver_success_rate_percent": float(
                self.successful_batches / self.processed_batches * 100.0
            )
            if self.processed_batches
            else 0.0,
            "fallback_count": int(self.fallback_count),
            "avg_feasible_candidate_pairs_per_batch": float(avg_feasible_pairs),
            "capacity_rejected_raw_assignments": int(self.total_capacity_rejected_raw),
            "zero_selection_raw": int(self.total_zero_selection_raw),
            "multi_selection_raw": int(self.total_multi_selection_raw),
            "raw_selected_variables": int(self.total_raw_selected_variables),
            **quality,
            **avg_qubo_stats,
        }
        return summary

    def average_qubo_stats(self):
        if not self.qubo_stat_rows:
            return {}
        keys = [
            "qubo_variables",
            "linear_coefficient_count",
            "quadratic_coefficient_count",
            "qubo_coefficient_count",
            "qubo_density",
            "coefficient_min",
            "coefficient_max",
            "coefficient_range",
        ]
        averaged = {}
        for key in keys:
            averaged[f"avg_{key}"] = float(np.mean([row[key] for row in self.qubo_stat_rows]))
        return averaged

    def log_progress(
        self,
        batch_idx,
        batch_assigned,
        coverage,
        success_rate,
        qubo_time,
        solver_time,
        energy,
        qubo_stats,
        raw_metrics,
        assignments,
        batch_failed,
        failed_reason,
    ):
        log_entry = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "formulation": FORMULATION,
            "solver": SOLVER,
            "batch_idx": int(batch_idx),
            "batch_assigned": int(batch_assigned),
            "coverage_percent": float(coverage),
            "solver_success_rate": float(success_rate),
            "batch_failed": bool(batch_failed),
            "failed_reason": failed_reason,
            "fallback_used": False,
            "qubo_time_sec": float(qubo_time),
            "annealing_time_sec": float(solver_time),
            "energy": float(energy) if energy is not None else None,
            "best_energy": float(energy) if energy is not None else None,
            "assignments": assignments,
            **qubo_stats,
            **raw_metrics,
        }
        with open(self.progress_log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(log_entry) + "\n")

    def write_summary(self, summary):
        with open(self.summary_log, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)

    @staticmethod
    def print_summary(summary):
        print("\n" + "=" * 90)
        print(f"{summary['formulation']} + {summary['solver']} RESULTS")
        print("=" * 90)
        print(f"Coverage: {summary['covered_cameras']}/{summary['n_cameras']} ({summary['coverage_percent']:.2f}%)")
        print(f"Objective: {summary['objective_value']:.3f}")
        print(f"Assignment cost: {summary['assignment_cost']:.3f}")
        print(f"Uncovered penalty: {summary['uncovered_penalty']:.3f}")
        print(f"Overload penalty: {summary['overload_penalty']:.3f}")
        print(f"Total time: {summary['total_time_sec']:.3f}s")
        print(f"Throughput: {summary['throughput_cam_per_sec']:.3f} cameras/s")
        print(f"Successful batches: {summary['successful_batches']}/{summary['processed_batches']}")
        print(f"Failed batches: {summary['failed_batches']}")
        print(f"Fallback count: {summary['fallback_count']}")
        print(f"Capacity-rejected raw assignments: {summary['capacity_rejected_raw_assignments']}")
        print("=" * 90)


def parse_args():
    parser = argparse.ArgumentParser(description=f"{FORMULATION} + {SOLVER} baseline QUBO experiment")
    parser.add_argument("--n-cameras", type=int, default=20000)
    parser.add_argument("--n-servers", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--max-servers-per-batch", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-reads", type=int, default=150)
    parser.add_argument("--num-sweeps", type=int, default=1000)
    parser.add_argument("--trotter", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--final-opt", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    experiment = BaselineQuboExperiment(
        n_cameras=args.n_cameras,
        n_servers=args.n_servers,
        batch_size=args.batch_size,
        max_servers_per_batch=args.max_servers_per_batch,
        random_seed=args.seed,
        num_reads=args.num_reads,
        num_sweeps=args.num_sweeps,
        trotter=args.trotter,
        log_every=args.log_every,
        final_opt=args.final_opt,
    )
    experiment.generate_realistic_data()
    experiment.run()


if __name__ == "__main__":
    main()
