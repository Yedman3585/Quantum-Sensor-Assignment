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

METHOD_LABELS = {
    "grasp": ("GRASP-20", "Greedy Randomized Adaptive Search"),
    "brkga": ("BRKGA-20", "Biased Random-Key Genetic Algorithm"),
    "tabu": ("Tabu-20", "Tabu Search Light"),
}

logger = logging.getLogger(__name__)


def configure_logging(method):
    method_label, _ = METHOD_LABELS[method]
    log_file = f"{method_label.lower().replace('-', '_')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


class ClassicalMetaheuristicExperiment:
    def __init__(
        self,
        method="grasp",
        n_cameras=20000,
        n_servers=800,
        batch_size=80,
        max_servers_per_batch=20,
        random_seed=42,
        capacity_scale=1.0,
        log_root=None,
        log_every=20,
        final_opt=False,
        coverage_stop_threshold=0.995,
        cost_weight=0.70,
        waste_weight=0.20,
        residual_weight=0.10,
        grasp_iterations=12,
        grasp_alpha=0.35,
        grasp_local_passes=1,
        population_size=16,
        generations=6,
        elite_fraction=0.25,
        mutant_fraction=0.15,
        crossover_bias=0.70,
        key_weight=0.08,
        order_noise=0.05,
        tabu_iterations=20,
        tabu_tenure=7,
    ):
        self.method = method
        self.method_label, self.solver_label = METHOD_LABELS[method]
        self.n_cameras = n_cameras
        self.n_servers = n_servers
        self.batch_size = batch_size
        self.max_servers_per_batch = max_servers_per_batch
        self.random_seed = random_seed
        self.capacity_scale = capacity_scale
        self.log_root = log_root
        self.log_every = log_every
        self.final_opt = final_opt
        self.coverage_stop_threshold = coverage_stop_threshold
        self.cost_weight = cost_weight
        self.waste_weight = waste_weight
        self.residual_weight = residual_weight
        self.grasp_iterations = grasp_iterations
        self.grasp_alpha = grasp_alpha
        self.grasp_local_passes = grasp_local_passes
        self.population_size = population_size
        self.generations = generations
        self.elite_fraction = elite_fraction
        self.mutant_fraction = mutant_fraction
        self.crossover_bias = crossover_bias
        self.key_weight = key_weight
        self.order_noise = order_noise
        self.tabu_iterations = tabu_iterations
        self.tabu_tenure = tabu_tenure

        self.rng = np.random.default_rng(random_seed + self._method_offset())

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
        self.total_load = 0.0
        self.total_capacity = 0.0
        self.utilization = 0.0

        log_dir_name = f"logs_{self.method_label.lower().replace('-', '_')}"
        self.log_dir = os.path.join(log_root, log_dir_name) if log_root else log_dir_name
        os.makedirs(self.log_dir, exist_ok=True)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.progress_log = os.path.join(self.log_dir, f"progress_{self.run_id}.jsonl")
        self.summary_log = os.path.join(self.log_dir, f"summary_{self.run_id}.json")

        self.processed_batches = 0
        self.successful_batches = 0
        self.failed_batches = 0
        self.fallback_count = 0
        self.total_candidate_pairs = 0
        self.total_feasible_candidate_pairs = 0
        self.total_zero_selection = 0
        self.total_multi_selection = 0
        self.total_capacity_rejected = 0
        self.total_solver_time = 0.0
        self.batch_stat_rows = []

    def _method_offset(self):
        return {"grasp": 101, "brkga": 202, "tabu": 303}[self.method]

    def generate_realistic_data(self):
        logger.info(
            "%s: data generation for %d cameras and %d servers",
            self.method_label,
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
        self.initial_capacity *= float(self.capacity_scale)
        self.remaining_capacity = self.initial_capacity.copy()

        self._build_cost_matrix()

        self.total_load = float(self.load_gflops.sum())
        self.total_capacity = float(self.initial_capacity.sum())
        self.utilization = self.total_load / self.total_capacity * 100.0
        logger.info(
            "total load %.1f, total capacity %.1f, utilization %.1f%%",
            self.total_load,
            self.total_capacity,
            self.utilization,
        )
        return self.utilization

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
        logger.info("%s processing %d batches", self.method_label, total_batches)

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
            batch_solution, raw_metrics = self.solve_batch(batch_indices, top_servers)
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
            self.total_zero_selection += raw_metrics["zero_selection_raw"]
            self.total_multi_selection += raw_metrics["multi_selection_raw"]
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

    def solve_batch(self, batch_indices, top_servers):
        if self.method == "grasp":
            return self.solve_batch_grasp(batch_indices, top_servers)
        if self.method == "brkga":
            return self.solve_batch_brkga(batch_indices, top_servers)
        if self.method == "tabu":
            return self.solve_batch_tabu(batch_indices, top_servers)
        raise ValueError(f"Unknown method: {self.method}")

    def empty_assignment_result(self, n_batch, n_top):
        return np.zeros((n_batch, n_top), dtype=np.int8), {
            "raw_selected_variables": 0,
            "zero_selection_raw": int(n_batch),
            "multi_selection_raw": 0,
            "capacity_rejected_raw": 0,
            "residual_blind_rejected_assignments": 0,
        }

    def greedy_construct(self, batch_indices, top_servers, randomize=False, grasp_alpha=0.0, genes=None):
        n_batch = len(batch_indices)
        n_top = len(top_servers)
        assignment = np.zeros((n_batch, n_top), dtype=np.int8)
        if n_top == 0:
            return assignment, int(n_batch)

        local_remaining = self.remaining_capacity[top_servers].copy()
        local_initial = self.initial_capacity[top_servers]
        priority_scores = self.priority[batch_indices] * self.load_gflops[batch_indices]

        if genes is not None:
            order_noise = self.order_noise * (genes[:, 0] - 0.5) * (np.max(priority_scores) + 1e-12)
            local_order = np.argsort(-(priority_scores + order_noise))
        else:
            local_order = np.argsort(-priority_scores)
            if randomize:
                jitter = self.rng.normal(0.0, 0.03, size=n_batch) * (np.max(priority_scores) + 1e-12)
                local_order = np.argsort(-(priority_scores + jitter))

        zero_selection = 0
        for local_i in local_order:
            cam_idx = int(batch_indices[local_i])
            load = float(self.load_gflops[cam_idx])
            feasible_local = np.where(local_remaining >= load)[0]
            if len(feasible_local) == 0:
                zero_selection += 1
                continue

            scores = self.local_candidate_scores(cam_idx, feasible_local, top_servers, local_remaining, local_initial, load)
            if genes is not None:
                key_bias = genes[local_i, 1:][feasible_local]
                scores = scores - self.key_weight * key_bias
                chosen_local = int(feasible_local[int(np.argmin(scores))])
            elif randomize:
                min_score = float(np.min(scores))
                max_score = float(np.max(scores))
                cutoff = min_score + grasp_alpha * (max_score - min_score + 1e-12)
                rcl = feasible_local[scores <= cutoff]
                chosen_local = int(self.rng.choice(rcl))
            else:
                chosen_local = int(feasible_local[int(np.argmin(scores))])

            assignment[local_i, chosen_local] = 1
            local_remaining[chosen_local] -= load

        return assignment, int(zero_selection)

    def local_candidate_scores(self, cam_idx, feasible_local, top_servers, local_remaining, local_initial, load):
        costs = self.cost_matrix[cam_idx, top_servers[feasible_local]]
        remaining_after = local_remaining[feasible_local] - load
        waste = remaining_after / (local_initial[feasible_local] + 1e-12)
        residual_ratio = remaining_after / (np.max(local_remaining) + 1e-12)
        return self.cost_weight * costs + self.waste_weight * waste - self.residual_weight * residual_ratio

    def solve_batch_grasp(self, batch_indices, top_servers):
        n_batch = len(batch_indices)
        n_top = len(top_servers)
        if n_top == 0:
            return self.empty_assignment_result(n_batch, n_top)

        best_assignment = None
        best_fitness = float("inf")
        best_zero = n_batch

        iterations = max(1, int(self.grasp_iterations))
        for _ in range(iterations):
            assignment, zero_selection = self.greedy_construct(
                batch_indices,
                top_servers,
                randomize=True,
                grasp_alpha=self.grasp_alpha,
            )
            assignment = self.improve_local_assignment(assignment, batch_indices, top_servers, passes=self.grasp_local_passes)
            fitness = self.local_assignment_fitness(assignment, batch_indices, top_servers)
            if fitness < best_fitness:
                best_assignment = assignment
                best_fitness = fitness
                best_zero = zero_selection

        raw_selected = int(np.sum(best_assignment))
        return best_assignment, {
            "raw_selected_variables": raw_selected,
            "zero_selection_raw": int(max(0, n_batch - raw_selected) if best_zero is None else best_zero),
            "multi_selection_raw": 0,
            "capacity_rejected_raw": 0,
            "residual_blind_rejected_assignments": 0,
        }

    def solve_batch_brkga(self, batch_indices, top_servers):
        n_batch = len(batch_indices)
        n_top = len(top_servers)
        if n_top == 0:
            return self.empty_assignment_result(n_batch, n_top)

        population_size = max(4, int(self.population_size))
        generations = max(1, int(self.generations))
        gene_shape = (n_batch, n_top + 1)
        population = self.rng.random((population_size, *gene_shape))

        best_assignment = None
        best_fitness = float("inf")
        best_zero = n_batch

        for _ in range(generations):
            fitness_rows = []
            for idx in range(population_size):
                assignment, zero_selection = self.greedy_construct(batch_indices, top_servers, genes=population[idx])
                fitness = self.local_assignment_fitness(assignment, batch_indices, top_servers)
                fitness_rows.append((fitness, idx, zero_selection, assignment))
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_assignment = assignment.copy()
                    best_zero = zero_selection

            fitness_rows.sort(key=lambda row: row[0])
            elite_count = max(1, int(round(self.elite_fraction * population_size)))
            mutant_count = max(1, int(round(self.mutant_fraction * population_size)))
            elite_indices = [row[1] for row in fitness_rows[:elite_count]]
            nonelite_indices = [row[1] for row in fitness_rows[elite_count:]] or elite_indices

            next_population = [population[idx].copy() for idx in elite_indices]
            while len(next_population) < population_size - mutant_count:
                elite_parent = population[int(self.rng.choice(elite_indices))]
                other_parent = population[int(self.rng.choice(nonelite_indices))]
                mask = self.rng.random(gene_shape) < self.crossover_bias
                child = np.where(mask, elite_parent, other_parent)
                next_population.append(child)

            while len(next_population) < population_size:
                next_population.append(self.rng.random(gene_shape))

            population = np.asarray(next_population)

        best_assignment = self.improve_local_assignment(best_assignment, batch_indices, top_servers, passes=1)
        raw_selected = int(np.sum(best_assignment))
        return best_assignment, {
            "raw_selected_variables": raw_selected,
            "zero_selection_raw": int(max(0, n_batch - raw_selected) if best_zero is None else best_zero),
            "multi_selection_raw": 0,
            "capacity_rejected_raw": 0,
            "residual_blind_rejected_assignments": 0,
        }

    def solve_batch_tabu(self, batch_indices, top_servers):
        n_batch = len(batch_indices)
        n_top = len(top_servers)
        if n_top == 0:
            return self.empty_assignment_result(n_batch, n_top)

        assignment, _ = self.greedy_construct(batch_indices, top_servers, randomize=False)
        assignment = self.tabu_improve_assignment(assignment, batch_indices, top_servers)
        raw_selected = int(np.sum(assignment))
        return assignment, {
            "raw_selected_variables": raw_selected,
            "zero_selection_raw": int(max(0, n_batch - raw_selected)),
            "multi_selection_raw": 0,
            "capacity_rejected_raw": 0,
            "residual_blind_rejected_assignments": 0,
        }

    def local_assignment_fitness(self, assignment, batch_indices, top_servers):
        assigned_mask = np.any(assignment, axis=1)
        assigned_local = np.where(assigned_mask)[0]
        uncovered = len(batch_indices) - len(assigned_local)
        total_cost = 0.0
        if len(assigned_local) > 0:
            selected_local_servers = np.argmax(assignment[assigned_local], axis=1)
            cameras = batch_indices[assigned_local]
            servers = top_servers[selected_local_servers]
            priority_weight = 4 - self.priority[cameras]
            total_cost = float(np.sum(self.cost_matrix[cameras, servers] * priority_weight))
        return total_cost + UNCOVERED_PENALTY * uncovered

    def local_remaining_after_assignment(self, assignment, batch_indices, top_servers):
        local_remaining = self.remaining_capacity[top_servers].copy()
        for local_i in range(len(batch_indices)):
            selected = np.where(assignment[local_i] == 1)[0]
            if len(selected) == 0:
                continue
            local_remaining[int(selected[0])] -= float(self.load_gflops[int(batch_indices[local_i])])
        return local_remaining

    def improve_local_assignment(self, assignment, batch_indices, top_servers, passes=1):
        if assignment is None or len(top_servers) == 0 or passes <= 0:
            return assignment

        improved = assignment.copy()
        local_remaining = self.local_remaining_after_assignment(improved, batch_indices, top_servers)
        local_initial = self.initial_capacity[top_servers]

        for _ in range(passes):
            changes = 0
            for local_i, cam_idx in enumerate(batch_indices):
                selected = np.where(improved[local_i] == 1)[0]
                if len(selected) == 0:
                    continue
                current_local = int(selected[0])
                load = float(self.load_gflops[int(cam_idx)])
                local_remaining[current_local] += load
                feasible_local = np.where(local_remaining >= load)[0]
                if len(feasible_local) == 0:
                    local_remaining[current_local] -= load
                    continue

                scores = self.local_candidate_scores(
                    int(cam_idx),
                    feasible_local,
                    top_servers,
                    local_remaining,
                    local_initial,
                    load,
                )
                best_local = int(feasible_local[int(np.argmin(scores))])
                current_score = float(
                    self.local_candidate_scores(
                        int(cam_idx),
                        np.array([current_local]),
                        top_servers,
                        local_remaining,
                        local_initial,
                        load,
                    )[0]
                )
                best_score = float(np.min(scores))
                if best_local != current_local and best_score < current_score * 0.995:
                    improved[local_i, current_local] = 0
                    improved[local_i, best_local] = 1
                    local_remaining[best_local] -= load
                    changes += 1
                else:
                    local_remaining[current_local] -= load

            if changes == 0:
                break
        return improved

    def tabu_improve_assignment(self, assignment, batch_indices, top_servers):
        improved = assignment.copy()
        n_batch = len(batch_indices)
        n_top = len(top_servers)
        local_remaining = self.local_remaining_after_assignment(improved, batch_indices, top_servers)
        tabu_until = {}
        best_fitness = self.local_assignment_fitness(improved, batch_indices, top_servers)
        current_fitness = best_fitness
        best_seen = improved.copy()

        for iteration in range(max(1, int(self.tabu_iterations))):
            best_move = None
            best_delta = 0.0

            for local_i in range(n_batch):
                cam_idx = int(batch_indices[local_i])
                load = float(self.load_gflops[cam_idx])
                priority_weight = 4 - self.priority[cam_idx]
                selected = np.where(improved[local_i] == 1)[0]
                current_local = int(selected[0]) if len(selected) else None

                if current_local is not None:
                    local_remaining[current_local] += load
                    current_cost = float(self.cost_matrix[cam_idx, top_servers[current_local]] * priority_weight)
                else:
                    current_cost = UNCOVERED_PENALTY

                feasible_local = np.where(local_remaining >= load)[0]
                for target_local in feasible_local:
                    target_local = int(target_local)
                    if target_local == current_local:
                        continue
                    new_cost = float(self.cost_matrix[cam_idx, top_servers[target_local]] * priority_weight)
                    delta = new_cost - current_cost
                    move_key = (local_i, target_local)
                    tabu_active = tabu_until.get(move_key, -1) > iteration
                    aspiration = current_fitness + delta < best_fitness
                    if tabu_active and not aspiration:
                        continue
                    if best_move is None or delta < best_delta:
                        best_delta = delta
                        best_move = (local_i, current_local, target_local, load)

                if current_local is not None:
                    local_remaining[current_local] -= load

            if best_move is None or best_delta >= -1e-9:
                break

            local_i, current_local, target_local, load = best_move
            if current_local is not None:
                improved[local_i, current_local] = 0
                local_remaining[current_local] += load
                tabu_until[(local_i, current_local)] = iteration + int(self.tabu_tenure)

            improved[local_i, target_local] = 1
            local_remaining[target_local] -= load
            tabu_until[(local_i, target_local)] = iteration + int(self.tabu_tenure)

            current_fitness += best_delta
            if current_fitness < best_fitness:
                best_fitness = current_fitness
                best_seen = improved.copy()

        return best_seen

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
            "formulation": self.method_label,
            "solver": self.solver_label,
            "method": self.method,
            "n_cameras": self.n_cameras,
            "n_servers": self.n_servers,
            "batch_size": self.batch_size,
            "max_servers_per_batch": self.max_servers_per_batch,
            "random_seed": self.random_seed,
            "capacity_scale": float(self.capacity_scale),
            "total_load": float(self.total_load),
            "total_capacity": float(self.total_capacity),
            "utilization_percent": float(self.utilization),
            "final_opt": bool(self.final_opt),
            "coverage_stop_threshold": float(self.coverage_stop_threshold),
            "cost_weight": float(self.cost_weight),
            "waste_weight": float(self.waste_weight),
            "residual_weight": float(self.residual_weight),
            "grasp_iterations": int(self.grasp_iterations),
            "grasp_alpha": float(self.grasp_alpha),
            "grasp_local_passes": int(self.grasp_local_passes),
            "population_size": int(self.population_size),
            "generations": int(self.generations),
            "elite_fraction": float(self.elite_fraction),
            "mutant_fraction": float(self.mutant_fraction),
            "crossover_bias": float(self.crossover_bias),
            "key_weight": float(self.key_weight),
            "order_noise": float(self.order_noise),
            "tabu_iterations": int(self.tabu_iterations),
            "tabu_tenure": int(self.tabu_tenure),
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
            "zero_selection_raw": int(self.total_zero_selection),
            "multi_selection_raw": int(self.total_multi_selection),
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
            "formulation": self.method_label,
            "solver": self.solver_label,
            "method": self.method,
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
    parser = argparse.ArgumentParser(description="Classical residual-capacity metaheuristic baselines")
    parser.add_argument("--method", choices=sorted(METHOD_LABELS), default="grasp")
    parser.add_argument("--n-cameras", type=int, default=20000)
    parser.add_argument("--n-servers", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--max-servers-per-batch", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--capacity-scale", type=float, default=1.0)
    parser.add_argument("--log-root", default=None)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--final-opt", action="store_true")
    parser.add_argument("--coverage-stop-threshold", type=float, default=0.995)
    parser.add_argument("--cost-weight", type=float, default=0.70)
    parser.add_argument("--waste-weight", type=float, default=0.20)
    parser.add_argument("--residual-weight", type=float, default=0.10)
    parser.add_argument("--grasp-iterations", type=int, default=12)
    parser.add_argument("--grasp-alpha", type=float, default=0.35)
    parser.add_argument("--grasp-local-passes", type=int, default=1)
    parser.add_argument("--population-size", type=int, default=16)
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--elite-fraction", type=float, default=0.25)
    parser.add_argument("--mutant-fraction", type=float, default=0.15)
    parser.add_argument("--crossover-bias", type=float, default=0.70)
    parser.add_argument("--key-weight", type=float, default=0.08)
    parser.add_argument("--order-noise", type=float, default=0.05)
    parser.add_argument("--tabu-iterations", type=int, default=20)
    parser.add_argument("--tabu-tenure", type=int, default=7)
    return parser.parse_args()


def main():
    args = parse_args()
    configure_logging(args.method)
    experiment = ClassicalMetaheuristicExperiment(
        method=args.method,
        n_cameras=args.n_cameras,
        n_servers=args.n_servers,
        batch_size=args.batch_size,
        max_servers_per_batch=args.max_servers_per_batch,
        random_seed=args.seed,
        capacity_scale=args.capacity_scale,
        log_root=args.log_root,
        log_every=args.log_every,
        final_opt=args.final_opt,
        coverage_stop_threshold=args.coverage_stop_threshold,
        cost_weight=args.cost_weight,
        waste_weight=args.waste_weight,
        residual_weight=args.residual_weight,
        grasp_iterations=args.grasp_iterations,
        grasp_alpha=args.grasp_alpha,
        grasp_local_passes=args.grasp_local_passes,
        population_size=args.population_size,
        generations=args.generations,
        elite_fraction=args.elite_fraction,
        mutant_fraction=args.mutant_fraction,
        crossover_bias=args.crossover_bias,
        key_weight=args.key_weight,
        order_noise=args.order_noise,
        tabu_iterations=args.tabu_iterations,
        tabu_tenure=args.tabu_tenure,
    )
    experiment.generate_realistic_data()
    experiment.run()


if __name__ == "__main__":
    main()
