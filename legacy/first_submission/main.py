import numpy as np
import pandas as pd
import time
import logging
from scipy.spatial import distance_matrix
import warnings
import os
import json
from datetime import datetime

warnings.filterwarnings('ignore')

try:
    import dimod
    from neal import SimulatedAnnealingSampler
    HAVE_DIMOD = True
except ImportError:
    HAVE_DIMOD = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('classical_optimization.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CameraScheduler:
    assignment_matrix: None

    def __init__(self, n_cameras=20000, n_servers=800,
                 batch_size=80, max_servers_per_batch=20,
                 random_seed=42):

        logger.info(f"Initializing cameras: {n_cameras} and servers: {n_servers}")

        self.n_cameras = n_cameras
        self.n_servers = n_servers
        self.batch_size = batch_size
        self.max_servers_per_batch = max_servers_per_batch

        self.random_seed = random_seed
        np.random.seed(self.random_seed)

        self.cameras_df = None
        self.servers_df = None
        self.cost_matrix = None
        self.assignment_matrix = None
        self.remaining_capacity = None
        self.assigned_cameras = set()

        self.log_dir = "logs"
        self.snapshot_dir = os.path.join(self.log_dir, "snapshots")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.snapshot_dir, exist_ok=True)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.progress_log = os.path.join(self.log_dir, f"progress_{self.run_id}.jsonl")
        self.snapshot_every = 50

    def log_progress(self, batch_idx, batch_assigned, coverage, success_rate,
                     energy=None, best_energy=None, qubo_time=0, annealing_time=0,
                     assignments=None):
        """Log progress to JSONL"""
        log_entry = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "batch_idx": int(batch_idx),
            "batch_assigned": int(batch_assigned),
            "coverage_percent": float(coverage),
            "qubo_success_rate": float(success_rate),
            "qubo_time_sec": float(qubo_time),
            "annealing_time_sec": float(annealing_time),
            "energy": float(energy) if energy is not None else None,
            "best_energy": float(best_energy) if best_energy is not None else None,
            "assignments": assignments or []
        }
        with open(self.progress_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

    def generate_data(self):
        logger.info("=== Data Generation ===")

        priorities = np.random.choice([3, 2, 1], size=self.n_cameras, p=[0.15, 0.25, 0.6])
        weights = np.zeros(self.n_cameras)
        loads = np.zeros(self.n_cameras)

        mask_p1 = priorities == 3
        mask_p2 = priorities == 2
        mask_p3 = priorities == 1

        weights[mask_p1] = np.random.uniform(4, 7, np.sum(mask_p1))
        loads[mask_p1] = np.random.uniform(8, 15, np.sum(mask_p1))

        weights[mask_p2] = np.random.uniform(2, 4, np.sum(mask_p2))
        loads[mask_p2] = np.random.uniform(4, 8, np.sum(mask_p2))

        weights[mask_p3] = np.random.uniform(0.4, 0.8, np.sum(mask_p3))
        loads[mask_p3] = np.random.uniform(1, 3, np.sum(mask_p3))

        camera_x = np.random.uniform(0, 1000, self.n_cameras)
        camera_y = np.random.uniform(0, 1000, self.n_cameras)

        self.cameras_df = pd.DataFrame({
            'camera_id': range(self.n_cameras),
            'priority': priorities,
            'weight_Mbps': weights,
            'load_GFLOPS': loads,
            'x': camera_x,
            'y': camera_y
        })

        server_x = np.random.uniform(0, 1000, self.n_servers)
        server_y = np.random.uniform(0, 1000, self.n_servers)

        server_types = np.random.choice([3, 2, 1], size=self.n_servers, p=[0.1, 0.3, 0.6])
        capacities = np.zeros(self.n_servers)

        capacities[server_types == 3] = np.random.uniform(800, 1000, np.sum(server_types == 3))
        capacities[server_types == 2] = np.random.uniform(400, 800, np.sum(server_types == 2))
        capacities[server_types == 1] = np.random.uniform(200, 400, np.sum(server_types == 1))

        self.servers_df = pd.DataFrame({
            'server_id': range(self.n_servers),
            'capacity_GFLOPS': capacities,
            'x': server_x,
            'y': server_y
        })

        self.build_cost_matrix()
        self.remaining_capacity = capacities.copy()

        total_load = loads.sum()
        total_capacity = capacities.sum()
        utilization = total_load / total_capacity * 100
        logger.info(
            f"Total load: {total_load:.1f}, capacity: {total_capacity:.1f}, utilization: {utilization:.1f}%")

        return utilization

    def build_cost_matrix(self):
        cam_coords = self.cameras_df[['x', 'y']].values
        srv_coords = self.servers_df[['x', 'y']].values

        distances = distance_matrix(cam_coords, srv_coords)
        distances_norm = distances / np.max(distances)

        priorities_norm = (3 - self.cameras_df['priority'].values) / 2.0
        loads_norm = self.cameras_df['load_GFLOPS'].values / np.max(self.cameras_df['load_GFLOPS'])

        capacities_inv = 1 / (self.servers_df['capacity_GFLOPS'].values + 1e-9)
        capacities_norm = capacities_inv / np.max(capacities_inv)

        self.cost_matrix = (
                0.40 * distances_norm +
                0.35 * loads_norm[:, None] +
                0.20 * priorities_norm[:, None] +
                0.05 * capacities_norm[None, :]
        )

        self.cost_matrix = (self.cost_matrix - np.min(self.cost_matrix)) / (
                np.max(self.cost_matrix) - np.min(self.cost_matrix) + 1e-9)

    def solve_annealing(self, num_reads=200):
        if not HAVE_DIMOD:
            logger.error("Dimod unavailable")
            return None, 0, 0, 0, 0

        logger.info("=== Classical Annealing ===")
        total_start = time.time()

        self.assignment_matrix = np.zeros((self.n_cameras, self.n_servers), dtype=int)
        self.remaining_capacity = self.servers_df['capacity_GFLOPS'].values.copy()
        self.assigned_cameras = set()

        priority_scores = self.cameras_df['priority'].values * self.cameras_df['load_GFLOPS'].values
        sorted_indices = np.argsort(-priority_scores)
        total_batches = (self.n_cameras + self.batch_size - 1) // self.batch_size

        total_qubo_time = 0
        total_annealing_time = 0
        annealing_success_count = 0
        total_assigned = 0

        logger.info(f"Processing {total_batches} batches...")

        self.current_batch_idx = 0

        for batch_idx in range(total_batches):
            self.current_batch_idx = batch_idx

            if len(self.assigned_cameras) / self.n_cameras > 0.995:
                logger.info("99.5% coverage achieved, completing")
                break

            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, self.n_cameras)
            batch_indices = sorted_indices[start_idx:end_idx]

            available_indices = [idx for idx in batch_indices if idx not in self.assigned_cameras]
            if not available_indices:
                continue

            top_servers = self.select_servers(available_indices)
            if len(top_servers) == 0:
                continue

            batch_complexity = self.calculate_batch_complexity(available_indices, top_servers)
            adaptive_reads = max(num_reads, int(num_reads * (1 + batch_complexity)))

            batch_solution, qubo_time, annealing_time, annealing_used = self.solve_batch(
                available_indices, top_servers, adaptive_reads
            )

            total_qubo_time += qubo_time
            total_annealing_time += annealing_time
            if annealing_used:
                annealing_success_count += 1

            batch_solution = self.post_process(batch_solution, available_indices, top_servers)

            assignments_in_batch = []
            batch_assigned = 0

            for i, cam_idx in enumerate(available_indices):
                assigned = False
                for j_local, j_global in enumerate(top_servers):
                    if batch_solution[i, j_local] == 1 and not assigned:
                        cam_load = self.cameras_df.iloc[cam_idx]['load_GFLOPS']
                        if self.remaining_capacity[j_global] >= cam_load:
                            self.assignment_matrix[cam_idx, j_global] = 1
                            self.remaining_capacity[j_global] -= cam_load
                            self.assigned_cameras.add(cam_idx)
                            batch_assigned += 1
                            assigned = True
                            assignments_in_batch.append({
                                "cam_id": int(cam_idx),
                                "server_id": int(j_global)
                            })

            total_assigned += batch_assigned

            current_coverage = len(self.assigned_cameras) / self.n_cameras * 100
            success_rate = (annealing_success_count / (batch_idx + 1)) * 100 if batch_idx > 0 else 0

            self.log_progress(
                batch_idx=batch_idx,
                batch_assigned=batch_assigned,
                coverage=current_coverage,
                success_rate=success_rate,
                annealing_time=annealing_time,
                assignments=assignments_in_batch
            )

            if (batch_idx + 1) % self.snapshot_every == 0:
                snapshot_path = os.path.join(self.snapshot_dir, f"snapshot_{self.run_id}_batch_{batch_idx + 1}.npz")
                np.savez_compressed(
                    snapshot_path,
                    assignment=self.assignment_matrix,
                    cameras=self.cameras_df.values,
                    servers=self.servers_df.values,
                    remaining_capacity=self.remaining_capacity
                )
                logger.info(f"SNAPSHOT: {snapshot_path}")

            if (batch_idx + 1) % 20 == 0:
                logger.info(
                    f"Batch {batch_idx + 1}: {batch_assigned} cameras, "
                    f"coverage: {current_coverage:.1f}%, QUBO success: {success_rate:.1f}%"
                )

        total_time = time.time() - total_start
        self.optimize_final()
        objective = self.calculate_objective(self.assignment_matrix)

        self.analyze_performance(
            total_qubo_time, total_annealing_time, total_time,
            total_assigned, annealing_success_count, total_batches
        )

        return self.assignment_matrix, objective, total_time, total_qubo_time, total_annealing_time

    def calculate_batch_complexity(self, batch_indices, top_servers):
        batch_loads = self.cameras_df.iloc[batch_indices]['load_GFLOPS'].values
        total_load = batch_loads.sum()
        available_capacity = self.remaining_capacity[top_servers].sum()

        complexity = min(1.0, total_load / (available_capacity + 1e-9))
        return complexity

    def select_servers(self, batch_indices):
        batch_loads = self.cameras_df.iloc[batch_indices]['load_GFLOPS'].values
        min_load = np.min(batch_loads)
        avg_load = np.mean(batch_loads)

        valid_servers = np.where(self.remaining_capacity >= avg_load * 0.7)[0]

        if len(valid_servers) == 0:
            valid_servers = np.where(self.remaining_capacity >= min_load)[0]

        if len(valid_servers) == 0:
            return np.array([])

        scores = []
        for server_idx in valid_servers:
            server_capacity = self.remaining_capacity[server_idx]

            avg_cost = np.mean(self.cost_matrix[batch_indices, server_idx])

            feasible_cameras = np.sum(batch_loads <= server_capacity)

            capacity_score = server_capacity / np.max(self.remaining_capacity[valid_servers])
            cost_score = 1 - avg_cost
            feasibility_score = feasible_cameras / len(batch_indices)

            total_score = capacity_score * 0.3 + cost_score * 0.5 + feasibility_score * 0.2
            scores.append(total_score)

        scores = np.array(scores)
        top_indices = np.argsort(-scores)[:self.max_servers_per_batch]

        return valid_servers[top_indices]

    def solve_batch(self, batch_indices, top_servers, num_reads):
        n_batch = len(batch_indices)
        n_servers_batch = len(top_servers)

        qubo_start = time.time()
        Q = self.build_qubo(batch_indices, top_servers)
        qubo_time = time.time() - qubo_start

        if not Q:
            solution = self.solve_batch_greedy(batch_indices, top_servers)
            self.log_progress(
                batch_idx=self.current_batch_idx,
                batch_assigned=0,
                coverage=len(self.assigned_cameras) / self.n_cameras * 100,
                success_rate=0,
                qubo_time=qubo_time,
                annealing_time=0
            )
            return solution, qubo_time, 0, False

        try:
            annealing_start = time.time()
            sampler = SimulatedAnnealingSampler()
            response = sampler.sample_qubo(Q, num_reads=num_reads)
            annealing_time = time.time() - annealing_start

            best_energy = response.first.energy

            assignment = self.decode_solution(response, n_batch, top_servers, batch_indices)

            assignments = []
            for i, cam_idx in enumerate(batch_indices):
                for j, srv_idx in enumerate(top_servers):
                    if assignment[i, j] == 1:
                        assignments.append({"cam_id": int(cam_idx), "server_id": int(srv_idx)})

            valid_assignments = len(assignments)
            if valid_assignments >= n_batch * 0.5:
                self.log_progress(
                    batch_idx=self.current_batch_idx,
                    batch_assigned=valid_assignments,
                    coverage=len(self.assigned_cameras) / self.n_cameras * 100,
                    success_rate=100.0,
                    energy=best_energy,
                    best_energy=best_energy,
                    qubo_time=qubo_time,
                    annealing_time=annealing_time,
                    assignments=assignments
                )
                return assignment, qubo_time, annealing_time, True
            else:
                solution = self.solve_batch_greedy(batch_indices, top_servers)
                self.log_progress(
                    batch_idx=self.current_batch_idx,
                    batch_assigned=0,
                    coverage=len(self.assigned_cameras) / self.n_cameras * 100,
                    success_rate=0,
                    qubo_time=qubo_time,
                    annealing_time=annealing_time
                )
                return solution, qubo_time, annealing_time, False

        except Exception as e:
            logger.warning(f"Annealing error: {e}")
            solution = self.solve_batch_greedy(batch_indices, top_servers)
            self.log_progress(
                batch_idx=self.current_batch_idx,
                batch_assigned=0,
                coverage=len(self.assigned_cameras) / self.n_cameras * 100,
                success_rate=0,
                qubo_time=qubo_time,
                annealing_time=0
            )
            return solution, qubo_time, 0, False

    def build_qubo(self, batch_indices, top_servers):
        n_batch = len(batch_indices)
        n_servers_batch = len(top_servers)

        batch_costs = self.cost_matrix[batch_indices][:, top_servers]
        batch_loads = self.cameras_df.iloc[batch_indices]['load_GFLOPS'].values
        batch_priorities = self.cameras_df.iloc[batch_indices]['priority'].values

        Q = {}

        for i in range(n_batch):
            priority_weight = (4 - batch_priorities[i])
            for j in range(n_servers_batch):
                var_name = f"x_{i}_{top_servers[j]}"
                cost = batch_costs[i, j]
                load = batch_loads[i]
                server_capacity = self.remaining_capacity[top_servers[j]]

                if load <= server_capacity:
                    reward = -25.0 * (1.0 - cost) * priority_weight
                    Q[(var_name, var_name)] = reward
                else:
                    penalty = 100.0
                    Q[(var_name, var_name)] = penalty

        for i in range(n_batch):
            for j1 in range(n_servers_batch):
                var1 = f"x_{i}_{top_servers[j1]}"
                for j2 in range(j1 + 1, n_servers_batch):
                    var2 = f"x_{i}_{top_servers[j2]}"
                    Q[(var1, var2)] = 15.0

        return Q

    def decode_solution(self, response, n_batch, top_servers, batch_indices):
        assignment = np.zeros((n_batch, len(top_servers)), dtype=int)
        best_solution = response.first.sample

        candidates = []
        for i in range(n_batch):
            for j, server_idx in enumerate(top_servers):
                var_name = f"x_{i}_{server_idx}"
                if best_solution.get(var_name, 0) == 1:
                    cost = self.cost_matrix[batch_indices[i], server_idx]
                    priority = self.cameras_df.iloc[batch_indices[i]]['priority']
                    load = self.cameras_df.iloc[batch_indices[i]]['load_GFLOPS']
                    server_capacity = self.remaining_capacity[server_idx]

                    capacity_utilization = 1.0 - (load / server_capacity) if server_capacity > 0 else 0
                    score = priority * 0.4 + (1 - cost) * 0.4 + capacity_utilization * 0.2
                    candidates.append((i, j, score, cost, priority))

        candidates.sort(key=lambda x: -x[2])

        used_cameras = set()
        server_loads = {server_idx: 0 for server_idx in top_servers}

        for i, j, score, cost, priority in candidates:
            if i not in used_cameras:
                server_idx = top_servers[j]
                cam_idx = batch_indices[i]
                cam_load = self.cameras_df.iloc[cam_idx]['load_GFLOPS']

                if server_loads[server_idx] + cam_load <= self.remaining_capacity[server_idx]:
                    assignment[i, j] = 1
                    used_cameras.add(i)
                    server_loads[server_idx] += cam_load

        return assignment

    def post_process(self, assignment, batch_indices, top_servers):
        improved = assignment.copy()
        n_batch = len(batch_indices)

        for i in range(n_batch):
            current_server = None
            current_cost = float('inf')

            for j in range(len(top_servers)):
                if assignment[i, j] == 1:
                    current_server = j
                    current_cost = self.cost_matrix[batch_indices[i], top_servers[j]]
                    break

            if current_server is not None:
                for j in range(len(top_servers)):
                    if j != current_server:
                        new_cost = self.cost_matrix[batch_indices[i], top_servers[j]]
                        cam_load = self.cameras_df.iloc[batch_indices[i]]['load_GFLOPS']

                        if (new_cost < current_cost * 0.95 and
                                self.remaining_capacity[top_servers[j]] >= cam_load):
                            improved[i, current_server] = 0
                            improved[i, j] = 1
                            break

        return improved

    def optimize_final(self):
        logger.info("Final optimization...")

        for iteration in range(3):
            improvements = 0
            for i in range(self.n_cameras):
                current_server = np.argmax(self.assignment_matrix[i]) if np.any(self.assignment_matrix[i]) else -1
                if current_server == -1:
                    continue

                current_cost = self.cost_matrix[i, current_server]
                current_load = self.cameras_df.iloc[i]['load_GFLOPS']

                best_server = current_server
                best_cost = current_cost

                for j in range(self.n_servers):
                    if j != current_server:
                        new_cost = self.cost_matrix[i, j]
                        if (new_cost < best_cost * 0.98 and
                                self.remaining_capacity[j] >= current_load):

                            if self.can_reassign(i, current_server, j):
                                best_server = j
                                best_cost = new_cost

                if best_server != current_server:
                    self.assignment_matrix[i, current_server] = 0
                    self.assignment_matrix[i, best_server] = 1
                    self.remaining_capacity[current_server] += current_load
                    self.remaining_capacity[best_server] -= current_load
                    improvements += 1

            if improvements == 0:
                break

            logger.info(f"Iteration {iteration + 1}: {improvements} improvements")

    def can_reassign(self, cam_idx, from_server, to_server):
        cam_load = self.cameras_df.iloc[cam_idx]['load_GFLOPS']
        return self.remaining_capacity[to_server] >= cam_load

    def solve_batch_greedy(self, batch_indices, top_servers):
        n_batch = len(batch_indices)
        assignment = np.zeros((n_batch, len(top_servers)), dtype=int)

        remaining_capacity = self.remaining_capacity[top_servers].copy()

        batch_data = self.cameras_df.iloc[batch_indices]
        priority_scores = batch_data['priority'].values * batch_data['load_GFLOPS'].values
        sorted_indices = np.argsort(-priority_scores)

        for pos in sorted_indices:
            cam_idx = batch_indices[pos]
            cam_load = self.cameras_df.iloc[cam_idx]['load_GFLOPS']
            cam_priority = self.cameras_df.iloc[cam_idx]['priority']

            best_server = -1
            best_score = -float('inf')

            for j, server_idx in enumerate(top_servers):
                if remaining_capacity[j] >= cam_load:
                    cost = self.cost_matrix[cam_idx, server_idx]
                    capacity_ratio = remaining_capacity[j] / self.servers_df.iloc[server_idx]['capacity_GFLOPS']

                    score = (cam_priority * 0.5) + ((1 - cost) * 0.3) + (capacity_ratio * 0.2)

                    if score > best_score:
                        best_score = score
                        best_server = j

            if best_server != -1:
                assignment[pos, best_server] = 1
                remaining_capacity[best_server] -= cam_load

        return assignment

    def solve_greedy(self, timeout_seconds=180):
        logger.info("=== Greedy Algorithm ===")
        start_time = time.time()
        last_progress_time = start_time

        assignment = np.zeros((self.n_cameras, self.n_servers), dtype=int)
        remaining_capacity = self.servers_df['capacity_GFLOPS'].values.copy()

        priority_scores = self.cameras_df['priority'].values * self.cameras_df['load_GFLOPS'].values
        sorted_indices = np.argsort(-priority_scores)

        processed = 0
        for idx, cam_idx in enumerate(sorted_indices):
            current_time = time.time()

            if current_time - last_progress_time > 30:
                logger.warning(
                    f"Greedy algorithm stalled - no progress for 30 seconds. Processed {idx}/{self.n_cameras} cameras")
                break

            if current_time - start_time > timeout_seconds:
                logger.warning(
                    f"Greedy algorithm timeout after {timeout_seconds} sec. Processed {idx}/{self.n_cameras} cameras")
                break

            cam_load = self.cameras_df.iloc[cam_idx]['load_GFLOPS']
            cam_priority = self.cameras_df.iloc[cam_idx]['priority']

            best_server = -1
            best_score = -float('inf')

            for j in range(self.n_servers):
                if remaining_capacity[j] >= cam_load:
                    cost = self.cost_matrix[cam_idx, j]
                    capacity_ratio = remaining_capacity[j] / self.servers_df.iloc[j]['capacity_GFLOPS']

                    score = (cam_priority * 0.6) + ((1 - cost) * 0.4) + (capacity_ratio * 0.1)

                    if score > best_score:
                        best_score = score
                        best_server = j

            if best_server != -1:
                assignment[cam_idx, best_server] = 1
                remaining_capacity[best_server] -= cam_load
                processed += 1
                last_progress_time = current_time

            if idx % 1000 == 0:
                logger.info(f"Greedy progress: {idx}/{self.n_cameras} cameras processed")

        time_taken = time.time() - start_time
        objective = self.calculate_objective(assignment)
        covered = np.sum(np.any(assignment, axis=1))

        logger.info(
            f"Greedy algorithm: {time_taken:.3f}sec, {covered} cameras (processed {processed}/{self.n_cameras})")
        return assignment, objective, time_taken

    def calculate_objective(self, assignment):
        total_cost = 0
        uncovered_penalty = 0
        overload_penalty = 0

        assigned_mask = np.any(assignment, axis=1)
        assigned_indices = np.where(assigned_mask)[0]

        for i in assigned_indices:
            j = np.argmax(assignment[i])
            priority = self.cameras_df.iloc[i]['priority']
            priority_weight = (4 - priority)
            total_cost += self.cost_matrix[i, j] * priority_weight

        uncovered = self.n_cameras - np.sum(assigned_mask)
        uncovered_penalty = uncovered * 15.0

        for j in range(self.n_servers):
            server_load = np.sum(self.cameras_df['load_GFLOPS'].values * assignment[:, j])
            overload = max(0, server_load - self.servers_df.iloc[j]['capacity_GFLOPS'])
            overload_penalty += overload * 8.0

        total_objective = total_cost + uncovered_penalty + overload_penalty

        logger.info(
            f"Objective: cost={total_cost:.1f}, uncovered={uncovered_penalty:.1f}, overload={overload_penalty:.1f}")

        return total_objective

    # def analyze_performance(self, total_qubo_time, total_annealing_time, total_time,
    #                         total_assigned, annealing_success, total_batches):
    #     logger.info("\n" + "=" * 50)
    #     logger.info("Performance Analysis")
    #     logger.info("=" * 50)
    #
    #     coverage = total_assigned / self.n_cameras * 100
    #     annealing_efficiency = (total_annealing_time / total_time * 100) if total_time > 0 else 0
    #
    #     logger.info(
    #         f"Successful QUBO batches: {annealing_success}/{total_batches} ({annealing_success / total_batches * 100:.1f}%)")
    #     logger.info(f"Coverage: {total_assigned}/{self.n_cameras} ({coverage:.1f}%)")
    #     logger.info(f"Total time: {total_time:.2f}s")
    #     logger.info(f"QUBO time: {total_qubo_time:.2f}s ({total_qubo_time / total_time * 100:.1f}%)")
    #     logger.info(f"Annealing time: {total_annealing_time:.2f}s ({annealing_efficiency:.1f}%)")

    def analyze_performance(self, total_qubo_time, total_annealing_time, total_time,
                            total_assigned, annealing_success, total_batches):
        logger.info("\n" + "=" * 50)
        logger.info("Performance Analysis")
        logger.info("=" * 50)

        coverage = total_assigned / self.n_cameras * 100
        annealing_efficiency = (total_annealing_time / total_time * 100) if total_time > 0 else 0
        qubo_efficiency = (total_qubo_time / total_time * 100) if total_time > 0 else 0

        # НОВАЯ МЕТРИКА ПРОИЗВОДИТЕЛЬНОСТИ (как в квантовом)
        throughput = total_assigned / total_time if total_time > 0 else 0

        logger.info(
            f"Successful QUBO batches: {annealing_success}/{total_batches} ({annealing_success / total_batches * 100:.1f}%)")
        logger.info(f"Coverage: {total_assigned}/{self.n_cameras} ({coverage:.1f}%)")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"QUBO time: {total_qubo_time:.2f}s ({qubo_efficiency:.1f}%)")
        logger.info(f"Annealing time: {total_annealing_time:.2f}s ({annealing_efficiency:.1f}%)")
        logger.info(f"Throughput: {throughput:.2f} cameras/second")  # ← ДОБАВИТЬ

    def run_comparison(self):
        logger.info("=== Algorithm Comparison ===")

        results = {}

        logger.info("\n1. Greedy algorithm with timeout...")
        try:
            greedy_assignment, greedy_objective, greedy_time = self.solve_greedy(timeout_seconds=180)
            results['Greedy'] = {
                'assignment': greedy_assignment,
                'objective': greedy_objective,
                'time': greedy_time,
                'covered': np.sum(np.any(greedy_assignment, axis=1))
            }
            logger.info(
                f"Greedy completed: {greedy_time:.2f}s, coverage: {results['Greedy']['covered']}/{self.n_cameras}")
        except Exception as e:
            logger.error(f"Greedy algorithm failed: {e}")

            results['Greedy'] = {
                'assignment': np.zeros((self.n_cameras, self.n_servers), dtype=int),
                'objective': float('inf'),
                'time': 0,
                'covered': 0
            }
            logger.info("Using fallback solution for greedy algorithm")

        if HAVE_DIMOD:
            logger.info("\n2. Classical annealing...")
            try:
                annealing_assignment, annealing_objective, annealing_time, annealing_qubo, annealing_time_actual = \
                    self.solve_annealing(num_reads=150)

                results['Annealing'] = {
                    'assignment': annealing_assignment,
                    'objective': annealing_objective,
                    'time': annealing_time,
                    'qubo_time': annealing_qubo,
                    'annealing_time': annealing_time_actual,
                    'covered': np.sum(np.any(annealing_assignment, axis=1))
                }
            except Exception as e:
                logger.error(f"Classical annealing failed: {e}")
                results['Annealing'] = {
                    'assignment': np.zeros((self.n_cameras, self.n_servers), dtype=int),
                    'objective': float('inf'),
                    'time': 0,
                    'qubo_time': 0,
                    'annealing_time': 0,
                    'covered': 0
                }

        self.print_results(results)
        return results

    def print_results(self, results):
        print("\n" + "=" * 100)
        print("== RESULTS ==")
        print("=" * 100)
        print(f"{'Method':<15} {'Time (s)':<10} {'Objective':<12} {'Cameras':<10} {'Time%':<10} {'Throughput':<12}")
        print("-" * 100)

        base_objective = results['Greedy']['objective'] if results['Greedy']['objective'] != float('inf') else 0

        for method, data in results.items():
            if data['objective'] == float('inf'):
                print(f"{method:<15} {'FAILED':<10} {'-':<12} {'-':<10} {'-':<10} {'-':<12}")
                continue

            if 'annealing_time' in data:
                time_efficiency = (data['annealing_time'] / data['time'] * 100) if data['time'] > 0 else 0
                throughput = data['covered'] / data['time'] if data['time'] > 0 else 0
                improvement = ((base_objective - data['objective']) / base_objective * 100) if base_objective > 0 else 0
                improvement_str = f"{improvement:+.1f}%" if improvement != 0 else "0.0%"

                print(
                    f"{method:<15} {data['time']:<10.2f} {data['objective']:<12.1f} {data['covered']:<10} "
                    f"{time_efficiency:<10.1f} {throughput:<12.2f}"
                )
            else:
                if method == 'Greedy':
                    throughput = data['covered'] / data['time'] if data['time'] > 0 else 0
                    ideal_cost_estimate = data['covered'] * 0.1
                    max_possible_improvement = max(0,
                                                   (data['objective'] - ideal_cost_estimate) / data['objective'] * 100)
                    improvement_str = f"IDEAL: {max_possible_improvement:.1f}%"
                else:
                    throughput = 0
                    improvement_str = "N/A"

                print(
                    f"{method:<15} {data['time']:<10.2f} {data['objective']:<12.1f} {data['covered']:<10} "
                    f"{'N/A':<10} {throughput:<12.2f}"
                )

        print("=" * 100)
        print("Time% = Annealing time / Total time (higher means more time spent on annealing)")
        print("Throughput = Cameras per second (higher is better)")
        print("=" * 100)


def main():
    logger.info("=== START FOR 20000 CAMERAS ===")

    scheduler = CameraScheduler(
        n_cameras=20000,
        n_servers=800,
        batch_size=80,
        max_servers_per_batch=20
    )

    utilization = scheduler.generate_data()
    logger.info(f"Capacity utilization: {utilization:.1f}%")
    results = scheduler.run_comparison()
    logger.info("=== EXPERIMENT COMPLETED ===")
    return results


if __name__ == "__main__":
    results = main()





