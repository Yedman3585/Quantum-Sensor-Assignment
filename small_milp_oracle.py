import argparse
import json
import logging
import os
import time
import warnings
from datetime import datetime

import numpy as np

warnings.filterwarnings("ignore")

METHOD = "Small-MILP-Oracle"
SOLVER = "HiGHS MILP"
LOG_FILE = "small_milp_oracle.log"
LOG_DIR = "logs_small_milp_oracle"

UNCOVERED_PENALTY = 15.0
OVERLOAD_PENALTY = 8.0

try:
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix

    HAVE_SCIPY_MILP = True
except ImportError:
    HAVE_SCIPY_MILP = False


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class SmallMilpOracleExperiment:
    def __init__(
        self,
        n_cameras=200,
        n_servers=40,
        random_seed=42,
        time_limit=300.0,
        mip_rel_gap=0.0,
        backtracking_limit=18,
    ):
        self.n_cameras = n_cameras
        self.n_servers = n_servers
        self.random_seed = random_seed
        self.time_limit = time_limit
        self.mip_rel_gap = mip_rel_gap
        self.backtracking_limit = backtracking_limit

        self.priority = None
        self.weight_mbps = None
        self.load_gflops = None
        self.camera_x = None
        self.camera_y = None
        self.server_x = None
        self.server_y = None
        self.initial_capacity = None
        self.cost_matrix = None
        self.assignment_matrix = None

        os.makedirs(LOG_DIR, exist_ok=True)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.progress_log = os.path.join(LOG_DIR, f"progress_{self.run_id}.jsonl")
        self.summary_log = os.path.join(LOG_DIR, f"summary_{self.run_id}.json")

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
        if HAVE_SCIPY_MILP:
            assignment, solver_info = self.solve_with_scipy_milp()
        elif self.n_cameras <= self.backtracking_limit:
            assignment, solver_info = self.solve_with_backtracking()
        else:
            raise RuntimeError(
                "scipy.optimize.milp is not available and the instance is too large for built-in backtracking. "
                "Install scipy with HiGHS support or run with --n-cameras <= backtracking limit."
            )

        total_time = time.time() - total_start
        self.assignment_matrix = assignment
        quality = self.calculate_quality(assignment)
        summary = self.build_summary(total_time, quality, solver_info)
        self.write_progress(summary)
        self.write_summary(summary)
        self.print_summary(summary)
        return summary

    def solve_with_scipy_milp(self):
        n_vars = self.n_cameras * self.n_servers
        priority_weight = 4 - self.priority
        objective = np.zeros(n_vars)
        for i in range(self.n_cameras):
            start = i * self.n_servers
            end = start + self.n_servers
            objective[start:end] = self.cost_matrix[i, :] * priority_weight[i] - UNCOVERED_PENALTY

        n_constraints = self.n_cameras + self.n_servers
        constraints = lil_matrix((n_constraints, n_vars), dtype=float)
        lower = np.zeros(n_constraints, dtype=float)
        upper = np.zeros(n_constraints, dtype=float)

        for i in range(self.n_cameras):
            start = i * self.n_servers
            end = start + self.n_servers
            constraints[i, start:end] = 1.0
            upper[i] = 1.0

        for j in range(self.n_servers):
            row = self.n_cameras + j
            for i in range(self.n_cameras):
                constraints[row, i * self.n_servers + j] = self.load_gflops[i]
            upper[row] = self.initial_capacity[j]

        integrality = np.ones(n_vars, dtype=int)
        bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
        linear_constraint = LinearConstraint(constraints.tocsr(), lb=lower, ub=upper)

        start = time.time()
        result = milp(
            c=objective,
            integrality=integrality,
            bounds=bounds,
            constraints=linear_constraint,
            options={
                "time_limit": self.time_limit,
                "mip_rel_gap": self.mip_rel_gap,
                "disp": False,
            },
        )
        solver_time = time.time() - start

        if result.x is None:
            assignment = np.zeros((self.n_cameras, self.n_servers), dtype=np.int8)
        else:
            assignment = (result.x.reshape(self.n_cameras, self.n_servers) > 0.5).astype(np.int8)

        return assignment, {
            "oracle_backend": "scipy.optimize.milp",
            "milp_status": int(result.status),
            "milp_success": bool(result.success),
            "milp_message": str(result.message),
            "milp_objective_raw": float(result.fun) if result.fun is not None else None,
            "milp_gap": float(getattr(result, "mip_gap", 0.0) or 0.0),
            "milp_dual_bound": float(getattr(result, "mip_dual_bound", 0.0) or 0.0),
            "solver_time_sec": float(solver_time),
            "variables": int(n_vars),
            "constraints": int(n_constraints),
        }

    def solve_with_backtracking(self):
        order = np.argsort(-(self.priority * self.load_gflops))
        priority_weight = 4 - self.priority
        remaining_capacity = self.initial_capacity.copy()
        assignment = np.zeros((self.n_cameras, self.n_servers), dtype=np.int8)
        best_assignment = assignment.copy()
        best_objective = float("inf")

        min_assign_cost = np.min(self.cost_matrix * priority_weight[:, None], axis=1)
        optimistic = np.minimum(min_assign_cost, UNCOVERED_PENALTY)

        def dfs(pos, current_cost):
            nonlocal best_objective, best_assignment
            if pos == len(order):
                if current_cost < best_objective:
                    best_objective = current_cost
                    best_assignment = assignment.copy()
                return

            lower_bound = current_cost + float(np.sum(optimistic[order[pos:]]))
            if lower_bound >= best_objective:
                return

            cam_idx = int(order[pos])
            load = float(self.load_gflops[cam_idx])
            server_order = np.argsort(self.cost_matrix[cam_idx, :])

            for server_idx in server_order:
                server_idx = int(server_idx)
                if remaining_capacity[server_idx] < load:
                    continue
                remaining_capacity[server_idx] -= load
                assignment[cam_idx, server_idx] = 1
                dfs(pos + 1, current_cost + float(self.cost_matrix[cam_idx, server_idx] * priority_weight[cam_idx]))
                assignment[cam_idx, server_idx] = 0
                remaining_capacity[server_idx] += load

            dfs(pos + 1, current_cost + UNCOVERED_PENALTY)

        start = time.time()
        dfs(0, 0.0)
        solver_time = time.time() - start
        return best_assignment, {
            "oracle_backend": "internal_backtracking",
            "milp_status": 0,
            "milp_success": True,
            "milp_message": "Exact backtracking completed",
            "milp_objective_raw": float(best_objective),
            "milp_gap": 0.0,
            "milp_dual_bound": float(best_objective),
            "solver_time_sec": float(solver_time),
            "variables": int(self.n_cameras * self.n_servers),
            "constraints": int(self.n_cameras + self.n_servers),
        }

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

    def build_summary(self, total_time, quality, solver_info):
        summary = {
            "run_id": self.run_id,
            "formulation": METHOD,
            "solver": SOLVER if solver_info["oracle_backend"] == "scipy.optimize.milp" else "Exact Backtracking",
            "n_cameras": self.n_cameras,
            "n_servers": self.n_servers,
            "batch_size": self.n_cameras,
            "max_servers_per_batch": self.n_servers,
            "random_seed": self.random_seed,
            "total_time_sec": float(total_time),
            "qubo_time_sec": 0.0,
            "throughput_cam_per_sec": float(quality["covered_cameras"] / total_time) if total_time > 0 else 0.0,
            "processed_batches": 1,
            "successful_batches": 1 if quality["covered_cameras"] > 0 else 0,
            "failed_batches": 0 if quality["covered_cameras"] > 0 else 1,
            "solver_success_rate_percent": 100.0 if quality["covered_cameras"] > 0 else 0.0,
            "fallback_count": 0,
            "avg_feasible_candidate_pairs_per_batch": int(self.n_cameras * self.n_servers),
            "capacity_rejected_raw_assignments": 0,
            "zero_selection_raw": int(quality["uncovered_cameras"]),
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
            **solver_info,
        }
        return summary

    def write_progress(self, summary):
        entry = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "formulation": summary["formulation"],
            "solver": summary["solver"],
            "batch_idx": 0,
            "batch_assigned": int(summary["covered_cameras"]),
            "coverage_percent": float(summary["coverage_percent"]),
            "solver_success_rate": float(summary["solver_success_rate_percent"]),
            "batch_failed": bool(summary["failed_batches"] > 0),
            "failed_reason": "" if summary["failed_batches"] == 0 else "oracle_failed",
            "fallback_used": False,
            "qubo_time_sec": 0.0,
            "annealing_time_sec": 0.0,
            "solver_time_sec": float(summary["solver_time_sec"]),
            "energy": None,
            "best_energy": None,
            "assignments": [],
        }
        with open(self.progress_log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

    def write_summary(self, summary):
        with open(self.summary_log, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)

    @staticmethod
    def print_summary(summary):
        print("\n" + "=" * 90)
        print(f"{summary['formulation']} RESULTS")
        print("=" * 90)
        print(f"Backend: {summary['solver']} ({summary['oracle_backend']})")
        print(f"Coverage: {summary['covered_cameras']}/{summary['n_cameras']} ({summary['coverage_percent']:.2f}%)")
        print(f"Objective: {summary['objective_value']:.3f}")
        print(f"Assignment cost: {summary['assignment_cost']:.3f}")
        print(f"Uncovered penalty: {summary['uncovered_penalty']:.3f}")
        print(f"Overload penalty: {summary['overload_penalty']:.3f}")
        print(f"Total time: {summary['total_time_sec']:.3f}s")
        print(f"Solver success: {summary['milp_success']}")
        print("=" * 90)


def parse_args():
    parser = argparse.ArgumentParser(description="Small-instance MILP oracle for assignment-quality validation")
    parser.add_argument("--n-cameras", type=int, default=200)
    parser.add_argument("--n-servers", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--mip-rel-gap", type=float, default=0.0)
    parser.add_argument("--backtracking-limit", type=int, default=18)
    return parser.parse_args()


def main():
    args = parse_args()
    experiment = SmallMilpOracleExperiment(
        n_cameras=args.n_cameras,
        n_servers=args.n_servers,
        random_seed=args.seed,
        time_limit=args.time_limit,
        mip_rel_gap=args.mip_rel_gap,
        backtracking_limit=args.backtracking_limit,
    )
    experiment.generate_realistic_data()
    experiment.run()


if __name__ == "__main__":
    main()
