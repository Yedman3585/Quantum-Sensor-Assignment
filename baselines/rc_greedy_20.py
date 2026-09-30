import argparse
import json
import logging
import os
import time
import warnings
from datetime import datetime

import numpy as np

warnings.filterwarnings("ignore")

METHOD = "RC-Greedy-20"
SOLVER = "Deterministic Greedy"
LOG_FILE = "rc_greedy_20.log"
LOG_DIR = "logs_rc_greedy_20"

UNCOVERED_PENALTY = 15.0
OVERLOAD_PENALTY = 8.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class ResidualCapacityGreedyExperiment:
    def __init__(
        self,
        n_cameras=20000,
        n_servers=800,
        batch_size=80,
        max_servers_per_batch=20,
        random_seed=42,
        log_every=20,
        final_opt=False,
        coverage_stop_threshold=0.995,
        cost_weight=0.70,
        waste_weight=0.20,
        residual_weight=0.10,
    ):
        self.n_cameras = n_cameras
        self.n_servers = n_servers
        self.batch_size = batch_size
        self.max_servers_per_batch = max_servers_per_batch
        self.random_seed = random_seed
        self.log_every = log_every
        self.final_opt = final_opt
        self.coverage_stop_threshold = coverage_stop_threshold
        self.cost_weight = cost_weight
        self.waste_weight = waste_weight
        self.residual_weight = residual_weight

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

        self.processed_batches = 0
        self.successful_batches = 0
        self.failed_batches = 0
        self.fallback_count = 0
        self.total_candidate_pairs = 0
        self.total_feasible_candidate_pairs = 0
        self.total_no_feasible_candidate = 0
        self.total_capacity_rejected = 0
        self.total_solver_time = 0.0
        self.batch_stat_rows = []

    def generate_realistic_data(self):
        logger.info("%s: data generation for %d cameras and %d servers", METHOD, self.n_cameras, self.n_servers)
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
        logger.info("total load %.1f, total capacity %.1f, utilization %.1f%%", total_load, total_capacity, utilization)
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

    def run(self):
        total_start = time.time()
        self.assignment_matrix = np.zeros((self.n_cameras, self.n_servers), dtype=np.int8)
        self.remaining_capacity = self.initial_capacity.copy()
        self.assigned_cameras = set()

        priority_scores = self.priority * self.load_gflops
        sorted_indices = np.argsort(-priority_scores)
        total_batches = int(np.ceil(self.n_cameras / self.batch_size))
        logger.info("%s processing %d batches", METHOD, total_batches)

        for batch_idx in range(total_batches):
            if len(self.assigned_cameras) / self.n_cameras > self.coverage_stop_threshold:
                logger.info("%.2f%% coverage threshold reached, completing", self.coverage_stop_threshold * 100.0)
                break

            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, self.n_cameras)
            batch_indices = sorted_indices[start_idx:end_idx]
            if len(batch_indices) == 0:
                continue

            top_servers, selection_stats = self.select_residual_servers(batch_indices)
            self.processed_batches += 1

            solver_start = time.time()
            batch_solution, raw_metrics = self.solve_batch_greedy(batch_indices, top_servers)
            solver_time = time.time() - solver_start
            self.total_solver_time += solver_time

            batch_assigned, assignments_in_batch = self.commit_batch(batch_solution, batch_indices, top_servers)
            batch_failed = batch_assigned < max(1, int(0.5 * len(batch_indices)))
            if batch_failed:
                self.failed_batches += 1
            else:
                self.successful_batches += 1

            self.total_candidate_pairs += selection_stats["candidate_pairs"]
            self.total_feasible_candidate_pairs += selection_stats["feasible_candidate_pairs"]
            self.total_no_feasible_candidate += raw_metrics["zero_selection_raw"]
            self.total_capacity_rejected += raw_metrics["capacity_rejected_raw"]
            self.batch_stat_rows.append(selection_stats)

            coverage = len(self.assigned_cameras) / self.n_cameras * 100.0
            success_rate = self.successful_batches / self.processed_batches * 100.0
            self.log_progress(
                batch_idx=batch_idx,
                batch_assigned=batch_assigned,
                coverage=coverage,
                success_rate=success_rate,
                solver_time=solver_time,
                batch_failed=batch_failed,
                selection_stats=selection_stats,
                raw_metrics=raw_metrics,
                assignments=assignments_in_batch,
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

    def select_residual_servers(self, batch_indices):
        batch_loads = self.load_gflops[batch_indices]
        candidate_pairs = int(len(batch_indices) * self.n_servers)
        feasible_counts = np.sum(batch_loads[:, None] <= self.remaining_capacity[None, :], axis=0)
        feasible_mask = feasible_counts > 0

        if not np.any(feasible_mask):
            return np.array([], dtype=int), {
                "candidate_servers": 0,
                "candidate_pairs": candidate_pairs,
                "feasible_candidate_pairs": 0,
                "selection_feasible_server_count": 0,
            }

        mean_cost = np.mean(self.cost_matrix[batch_indices, :], axis=0)
        residual_norm = self.remaining_capacity / (np.max(self.remaining_capacity) + 1e-12)
        feasible_ratio = feasible_counts / max(1, len(batch_indices))

        score = 0.55 * (1.0 - mean_cost) + 0.25 * feasible_ratio + 0.20 * residual_norm
        invalid_penalty = np.where(feasible_mask, 0.0, 1e9)
        order = np.lexsort((-self.remaining_capacity, mean_cost, -(score - invalid_penalty)))
        top_servers = order[: self.max_servers_per_batch]

        feasible_pairs = int(np.sum(batch_loads[:, None] <= self.remaining_capacity[top_servers][None, :]))
        return top_servers.astype(int), {
            "candidate_servers": int(len(top_servers)),
            "candidate_pairs": int(len(batch_indices) * len(top_servers)),
            "feasible_candidate_pairs": feasible_pairs,
            "selection_feasible_server_count": int(np.sum(feasible_mask)),
        }

    def solve_batch_greedy(self, batch_indices, top_servers):
        n_batch = len(batch_indices)
        assignment = np.zeros((n_batch, len(top_servers)), dtype=np.int8)
        if len(top_servers) == 0:
            return assignment, {
                "raw_selected_variables": 0,
                "zero_selection_raw": int(n_batch),
                "multi_selection_raw": 0,
                "capacity_rejected_raw": 0,
                "residual_blind_rejected_assignments": 0,
            }

        local_remaining = self.remaining_capacity[top_servers].copy()
        local_initial = self.initial_capacity[top_servers]
        local_order = np.argsort(-(self.priority[batch_indices] * self.load_gflops[batch_indices]))
        zero_selection = 0

        for local_i in local_order:
            cam_idx = int(batch_indices[local_i])
            load = float(self.load_gflops[cam_idx])
            feasible_local = np.where(local_remaining >= load)[0]
            if len(feasible_local) == 0:
                zero_selection += 1
                continue

            costs = self.cost_matrix[cam_idx, top_servers[feasible_local]]
            remaining_after = local_remaining[feasible_local] - load
            waste = remaining_after / (local_initial[feasible_local] + 1e-12)
            residual_ratio = remaining_after / (np.max(local_remaining) + 1e-12)
            score = self.cost_weight * costs + self.waste_weight * waste - self.residual_weight * residual_ratio
            best_pos = int(feasible_local[int(np.argmin(score))])
            assignment[local_i, best_pos] = 1
            local_remaining[best_pos] -= load

        raw_selected = int(np.sum(assignment))
        return assignment, {
            "raw_selected_variables": raw_selected,
            "zero_selection_raw": int(zero_selection),
            "multi_selection_raw": 0,
            "capacity_rejected_raw": 0,
            "residual_blind_rejected_assignments": 0,
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
                self.total_capacity_rejected += 1
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

                feasible_servers = np.where(self.remaining_capacity >= cam_load)[0]
                for server_idx in feasible_servers:
                    new_cost = float(self.cost_matrix[cam_idx, server_idx])
                    if new_cost < best_cost * 0.98:
                        best_server = int(server_idx)
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
            total_cost = float(np.sum(self.cost_matrix[assigned_indices, assigned_servers] * priority_weight))

        uncovered = int(self.n_cameras - np.sum(assigned_mask))
        uncovered_penalty = float(uncovered * UNCOVERED_PENALTY)
        server_loads = assignment.T.astype(float).dot(self.load_gflops)
        overload = np.maximum(0.0, server_loads - self.initial_capacity)
        overload_penalty = float(np.sum(overload * OVERLOAD_PENALTY))
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
        avg_stats = self.average_batch_stats()
        summary = {
            "run_id": self.run_id,
            "formulation": METHOD,
            "solver": SOLVER,
            "n_cameras": self.n_cameras,
            "n_servers": self.n_servers,
            "batch_size": self.batch_size,
            "max_servers_per_batch": self.max_servers_per_batch,
            "random_seed": self.random_seed,
            "final_opt": bool(self.final_opt),
            "coverage_stop_threshold": float(self.coverage_stop_threshold),
            "cost_weight": float(self.cost_weight),
            "waste_weight": float(self.waste_weight),
            "residual_weight": float(self.residual_weight),
            "total_time_sec": float(total_time),
            "qubo_time_sec": 0.0,
            "solver_time_sec": float(self.total_solver_time),
            "throughput_cam_per_sec": float(quality["covered_cameras"] / total_time) if total_time > 0 else 0.0,
            "processed_batches": int(self.processed_batches),
            "successful_batches": int(self.successful_batches),
            "failed_batches": int(self.failed_batches),
            "solver_success_rate_percent": float(self.successful_batches / self.processed_batches * 100.0)
            if self.processed_batches
            else 0.0,
            "fallback_count": int(self.fallback_count),
            "avg_feasible_candidate_pairs_per_batch": float(
                self.total_feasible_candidate_pairs / self.processed_batches if self.processed_batches else 0.0
            ),
            "capacity_rejected_raw_assignments": int(self.total_capacity_rejected),
            "zero_selection_raw": int(self.total_no_feasible_candidate),
            "multi_selection_raw": 0,
            "raw_selected_variables": int(quality["covered_cameras"]),
            "avg_qubo_variables": 0.0,
            "avg_linear_coefficient_count": 0.0,
            "avg_quadratic_coefficient_count": 0.0,
            "avg_qubo_coefficient_count": 0.0,
            "avg_qubo_density": 0.0,
            "avg_coefficient_min": 0.0,
            "avg_coefficient_max": 0.0,
            "avg_coefficient_range": 0.0,
            **quality,
            **avg_stats,
        }
        return summary

    def average_batch_stats(self):
        if not self.batch_stat_rows:
            return {}
        return {
            "avg_candidate_servers": float(np.mean([row["candidate_servers"] for row in self.batch_stat_rows])),
            "avg_candidate_pairs": float(np.mean([row["candidate_pairs"] for row in self.batch_stat_rows])),
            "avg_selection_feasible_server_count": float(
                np.mean([row["selection_feasible_server_count"] for row in self.batch_stat_rows])
            ),
        }

    def log_progress(
        self,
        batch_idx,
        batch_assigned,
        coverage,
        success_rate,
        solver_time,
        batch_failed,
        selection_stats,
        raw_metrics,
        assignments,
    ):
        log_entry = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "formulation": METHOD,
            "solver": SOLVER,
            "batch_idx": int(batch_idx),
            "batch_assigned": int(batch_assigned),
            "coverage_percent": float(coverage),
            "solver_success_rate": float(success_rate),
            "batch_failed": bool(batch_failed),
            "failed_reason": "weak_batch" if batch_failed else "",
            "fallback_used": False,
            "qubo_time_sec": 0.0,
            "annealing_time_sec": 0.0,
            "solver_time_sec": float(solver_time),
            "energy": None,
            "best_energy": None,
            "assignments": assignments,
            **selection_stats,
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
        print(f"{summary['formulation']} RESULTS")
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
        print(f"No-feasible-candidate selections: {summary['zero_selection_raw']}")
        print("=" * 90)


def parse_args():
    parser = argparse.ArgumentParser(description="Residual-capacity greedy full-scale baseline")
    parser.add_argument("--n-cameras", type=int, default=20000)
    parser.add_argument("--n-servers", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--max-servers-per-batch", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--final-opt", action="store_true")
    parser.add_argument("--coverage-stop-threshold", type=float, default=0.995)
    parser.add_argument("--cost-weight", type=float, default=0.70)
    parser.add_argument("--waste-weight", type=float, default=0.20)
    parser.add_argument("--residual-weight", type=float, default=0.10)
    return parser.parse_args()


def main():
    args = parse_args()
    experiment = ResidualCapacityGreedyExperiment(
        n_cameras=args.n_cameras,
        n_servers=args.n_servers,
        batch_size=args.batch_size,
        max_servers_per_batch=args.max_servers_per_batch,
        random_seed=args.seed,
        log_every=args.log_every,
        final_opt=args.final_opt,
        coverage_stop_threshold=args.coverage_stop_threshold,
        cost_weight=args.cost_weight,
        waste_weight=args.waste_weight,
        residual_weight=args.residual_weight,
    )
    experiment.generate_realistic_data()
    experiment.run()


if __name__ == "__main__":
    main()
