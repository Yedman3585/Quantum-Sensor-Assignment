

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
    import openjij as oj

    print("OpenJij 0.11.6")
    HAVE_OPENJIJ = True

    try:
        test_sampler = oj.SQASampler(num_reads=1, num_sweeps=100, trotter=4)
        OPENJIJ_WORKING_PARAMS = ['num_reads', 'num_sweeps', 'trotter']
        print("working: num_reads, num_sweeps, trotter")
    except TypeError as e:
        if "beta" in str(e):
            OPENJIJ_WORKING_PARAMS = ['num_sweeps', 'trotter']
            print("working: num_sweeps, trotter (beta баг)")
        else:
            OPENJIJ_WORKING_PARAMS = ['minimal']
            print("minimal constructor")
except ImportError:
    HAVE_OPENJIJ = False
    OPENJIJ_WORKING_PARAMS = []
    print("OpenJij not found")

try:
    from neal import SimulatedAnnealingSampler

    HAVE_NEAL = True
    print("Neal available for fallback")
except ImportError:
    HAVE_NEAL = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('quantum_openjij_windows.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class WindowsOpenJijScheduler:
    def __init__(self, n_cameras=20000, n_servers=800, batch_size=80, max_servers_per_batch=20, random_seed=42):
        logger.info(f"initialization OpenJij 0.11.6 windows: {n_cameras} cameras, {n_servers} servers")

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

        self.log_dir = "logs_openjij_windows"
        self.snapshot_dir = os.path.join(self.log_dir, "snapshots")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.snapshot_dir, exist_ok=True)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.progress_log = os.path.join(self.log_dir, f"progress_{self.run_id}.jsonl")
        self.snapshot_every = 50
        self.current_batch_idx = 0
        self.performance_history = []  # Для хранения истории производительности

    # def _log_progress(self, batch_idx, batch_assigned, coverage, success_rate,
    #                   energy=None, best_energy=None, qubo_time=0, quantum_time=0,
    #                   assignments=None):
    #
    #     log_entry = {
    #         "run_id": self.run_id,
    #         "timestamp": datetime.now().isoformat(),
    #         "batch_idx": int(batch_idx),
    #         "batch_assigned": int(batch_assigned),
    #         "coverage_percent": float(coverage),
    #         "qubo_success_rate": float(success_rate),
    #         "qubo_time_sec": float(qubo_time),
    #         "quantum_time_sec": float(quantum_time),
    #         "energy": float(energy) if energy is not None else None,
    #         "best_energy": float(best_energy) if best_energy is not None else None,
    #         "assignments": assignments or [],
    #         "qubo_variables": len(assignments) * 20 if assignments else 0,
    #         "batch_size": len(assignments) if assignments else 0
    #     }
    #     with open(self.progress_log, "a", encoding="utf-8") as f:
    #         f.write(json.dumps(log_entry) + "\n")

    def _log_progress(self, batch_idx, batch_assigned, coverage, success_rate,
                      energy=None, best_energy=None, qubo_time=0, quantum_time=0,
                      assignments=None):

        log_entry = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "batch_idx": int(batch_idx),
            "batch_assigned": int(batch_assigned),
            "coverage_percent": float(coverage),
            "qubo_success_rate": float(success_rate),
            "qubo_time_sec": float(qubo_time),
            "annealing_time_sec": float(quantum_time),  # ← ИЗМЕНИТЬ: было quantum_time_sec
            "energy": float(energy) if energy is not None else None,
            "best_energy": float(best_energy) if best_energy is not None else None,
            "assignments": assignments or [],
            "qubo_variables": len(assignments) * 20 if assignments else 0,
            "batch_size": len(assignments) if assignments else 0
        }
        with open(self.progress_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")



    def generate_realistic_data(self):
        logger.info("===Data Generation===")

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

        self._build_cost_matrix()
        self.remaining_capacity = capacities.copy()

        total_load = loads.sum()
        total_capacity = capacities.sum()
        utilization = total_load / total_capacity * 100
        logger.info(
            f"total load: {total_load:.1f}, capacity: {total_capacity:.1f}, usage: {utilization:.1f}%")

        return utilization

    def _build_cost_matrix(self):
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

    def solve_with_quantum_optimized(self, num_reads=200):
        if not HAVE_OPENJIJ and not HAVE_NEAL:
            logger.error("no available solvers")
            return None, 0, 0, 0, 0

        logger.info("Quantum annealing OpenJij 0.11.6")
        total_start = time.time()

        self.assignment_matrix = np.zeros((self.n_cameras, self.n_servers), dtype=int)
        self.remaining_capacity = self.servers_df['capacity_GFLOPS'].values.copy()
        self.assigned_cameras = set()

        priority_scores = self.cameras_df['priority'].values * self.cameras_df['load_GFLOPS'].values
        sorted_indices = np.argsort(-priority_scores)
        total_batches = (self.n_cameras + self.batch_size - 1) // self.batch_size

        total_qubo_time = 0
        total_quantum_time = 0
        quantum_success_count = 0
        total_assigned = 0
        processed_batches = 0

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

            top_servers = self._select_servers_optimized(available_indices)
            if len(top_servers) == 0:
                continue

            processed_batches += 1

            batch_complexity = self._calculate_batch_complexity(available_indices, top_servers)
            adaptive_reads = max(num_reads, int(num_reads * (1 + batch_complexity)))

            batch_solution, qubo_time, quantum_time, quantum_used, batch_energy = self._solve_batch_sqa_windows_reliable(
                available_indices, top_servers, adaptive_reads
            )

            total_qubo_time += qubo_time
            total_quantum_time += quantum_time
            if quantum_used:
                quantum_success_count += 1

            batch_solution = self._post_process_solution(batch_solution, available_indices, top_servers)

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
            success_rate = (quantum_success_count / (batch_idx + 1)) * 100 if batch_idx > 0 else 0

            self._log_progress(
                batch_idx=batch_idx,
                batch_assigned=batch_assigned,
                coverage=current_coverage,
                success_rate=success_rate,
                energy=batch_energy,
                best_energy=batch_energy,
                qubo_time=qubo_time,
                quantum_time=quantum_time,
                assignments=assignments_in_batch
            )

            if (batch_idx + 1) % 20 == 0:
                logger.info(
                    f"Batch {batch_idx + 1}: {batch_assigned} cameras, "
                    f"coverage: {current_coverage:.1f}%, success SQA: {success_rate:.1f}%"
                )

        total_time = time.time() - total_start

        logger.info(f"Processed batches: {processed_batches}/{total_batches}")
        if processed_batches > 0:
            logger.info(f"Average QUBO time per batch: {total_qubo_time / processed_batches:.3f}s")

        self._optimize_final_solution()
        objective = self._calculate_optimized_objective(self.assignment_matrix)

        self._analyze_optimized_performance(
            total_qubo_time, total_quantum_time, total_time,
            total_assigned, quantum_success_count, total_batches,
            objective_value=objective
        )

        # Сохраняем производительность для истории
        performance_record = {
            'run_id': self.run_id,
            'timestamp': datetime.now(),
            'total_time': total_time,
            'annealing_time': total_quantum_time,
            'coverage': total_assigned,
            'objective': objective,
            'cameras_per_second': total_assigned / total_time if total_time > 0 else 0,
            'annealing_efficiency': (total_quantum_time / total_time * 100) if total_time > 0 else 0
        }
        self.performance_history.append(performance_record)

        return self.assignment_matrix, objective, total_time, total_qubo_time, total_quantum_time

    def _solve_batch_sqa_windows_reliable(self, batch_indices, top_servers, num_reads=100):
        n_batch = len(batch_indices)
        n_servers_batch = len(top_servers)

        qubo_start = time.time()
        Q = self._build_optimized_qubo(batch_indices, top_servers)
        qubo_time = time.time() - qubo_start

        if self.current_batch_idx % 50 == 0:
            logger.debug(f"QUBO batch {self.current_batch_idx}: {qubo_time:.3f}s, "
                         f"variables: {n_batch}×{n_servers_batch} = {n_batch * n_servers_batch}")

        if not Q:
            return self._solve_batch_greedy_optimized(batch_indices, top_servers), qubo_time, 0, False, 0.0

        if not HAVE_OPENJIJ:
            return self._solve_batch_with_neal_or_greedy(Q, batch_indices, top_servers, qubo_time)

        try:
            quantum_start = time.time()
            all_responses = []
            successful_reads = 0

            for attempt in range(max(1, num_reads // 10)):
                try:
                    if 'num_reads' in OPENJIJ_WORKING_PARAMS and 'num_sweeps' in OPENJIJ_WORKING_PARAMS:
                        sampler = oj.SQASampler(
                            num_reads=min(10, num_reads),
                            num_sweeps=1000,
                            trotter=8
                        )
                        logger.debug("SQA num_reads, num_sweeps, trotter")

                    elif 'num_sweeps' in OPENJIJ_WORKING_PARAMS:
                        sampler = oj.SQASampler(num_sweeps=1000, trotter=8)
                        logger.debug("SQA num_sweeps, trotter")

                    elif 'trotter' in OPENJIJ_WORKING_PARAMS:
                        sampler = oj.SQASampler(trotter=8)
                        logger.debug("SQA trotter")

                    else:
                        sampler = oj.SQASampler()
                        logger.debug("SQA default")

                    response = sampler.sample_qubo(Q)
                    all_responses.append(response)
                    successful_reads += 1

                    if len(all_responses) >= 3:
                        break

                except Exception as e:
                    logger.debug(f"attempt SQA {attempt + 1} failed: {e}")
                    continue

            if not all_responses:
                raise ValueError("SQA didn't return any solution")

            response = min(all_responses, key=lambda r: r.first.energy)
            quantum_time = time.time() - quantum_start
            best_energy = response.first.energy

            assignment = self._decode_optimized_solution(response, n_batch, top_servers, batch_indices)
            valid_assignments = np.sum([np.any(assignment[i]) for i in range(n_batch)])

            if valid_assignments >= n_batch * 0.5:
                logger.info(f"SQA success: {valid_assignments}/{n_batch} cameras, energy {best_energy:.1f}")
                return assignment, qubo_time, quantum_time, True, best_energy
            else:
                logger.warning(f"SQA weak result ({valid_assignments}/{n_batch}) → fallback")
                raise ValueError("bad solution SQA")

        except Exception as e:
            logger.warning(f"SQA failed: {e} → Neal/Greedy")
            quantum_time = time.time() - quantum_start if 'quantum_time' in locals() else 0
            return self._solve_batch_with_neal_or_greedy(Q, batch_indices, top_servers, qubo_time)

    def _solve_batch_with_neal_or_greedy(self, Q, batch_indices, top_servers, qubo_time):
        if HAVE_NEAL:
            try:
                quantum_start = time.time()
                sampler = SimulatedAnnealingSampler()
                response = sampler.sample_qubo(Q, num_reads=50)
                quantum_time = time.time() - quantum_start
                best_energy = response.first.energy

                assignment = self._decode_optimized_solution(
                    response, len(batch_indices), top_servers, batch_indices
                )
                logger.info(f"success fallback Neal SA, energy {best_energy:.1f}")
                return assignment, qubo_time, quantum_time, True, best_energy
            except Exception as e:
                logger.warning(f"Neal failed: {e}")

        logger.info("greedy fallback")
        greedy_solution = self._solve_batch_greedy_optimized(batch_indices, top_servers)
        return greedy_solution, qubo_time, 0, False, 0.0

    def _calculate_batch_complexity(self, batch_indices, top_servers):
        batch_loads = self.cameras_df.iloc[batch_indices]['load_GFLOPS'].values
        total_load = batch_loads.sum()
        available_capacity = self.remaining_capacity[top_servers].sum()

        complexity = min(1.0, total_load / (available_capacity + 1e-9))
        return complexity

    def _select_servers_optimized(self, batch_indices):
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

    def _build_optimized_qubo(self, batch_indices, top_servers):
        n_batch = len(batch_indices)
        n_servers_batch = len(top_servers)

        if n_batch == 0 or n_servers_batch == 0:
            logger.warning(f"Empty QUBO: batch_indices={len(batch_indices)}, top_servers={len(top_servers)}")
            return {}

        batch_costs = self.cost_matrix[batch_indices][:, top_servers]
        batch_loads = self.cameras_df.iloc[batch_indices]['load_GFLOPS'].values
        batch_priorities = self.cameras_df.iloc[batch_indices]['priority'].values

        Q = {}

        valid_assignments = 0
        invalid_assignments = 0

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
                    valid_assignments += 1
                else:
                    penalty = 100.0
                    Q[(var_name, var_name)] = penalty
                    invalid_assignments += 1

        for i in range(n_batch):
            for j1 in range(n_servers_batch):
                var1 = f"x_{i}_{top_servers[j1]}"
                for j2 in range(j1 + 1, n_servers_batch):
                    var2 = f"x_{i}_{top_servers[j2]}"
                    Q[(var1, var2)] = 15.0

        if self.current_batch_idx % 50 == 0:
            logger.debug(f"QUBO stats: {len(Q)} coefficients, "
                         f"valid/invalid: {valid_assignments}/{invalid_assignments}")

        return Q

    def _decode_optimized_solution(self, response, n_batch, top_servers, batch_indices):
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

    def _post_process_solution(self, assignment, batch_indices, top_servers):
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

    def _optimize_final_solution(self):
        logger.info("final optimization...")

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

                            if self._can_reassign(i, current_server, j):
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

            logger.info(f"iteration {iteration + 1}: {improvements} improvements")

    def _can_reassign(self, cam_idx, from_server, to_server):
        cam_load = self.cameras_df.iloc[cam_idx]['load_GFLOPS']
        return self.remaining_capacity[to_server] >= cam_load

    def _solve_batch_greedy_optimized(self, batch_indices, top_servers):
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

    def _calculate_optimized_objective(self, assignment):
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
            f"Objective(target functions): cost={total_cost:.1f}, uncovered={uncovered_penalty:.1f}, overload={overload_penalty:.1f}")

        return total_objective

    # def _analyze_optimized_performance(self, total_qubo_time, total_quantum_time, total_time,
    #                                    total_assigned, quantum_success, total_batches,
    #                                    objective_value=None):
    #     logger.info("\n" + "=" * 50)
    #     logger.info("Analyze of optimized results")
    #     logger.info("=" * 50)
    #
    #     coverage = total_assigned / self.n_cameras * 100
    #
    #     quantum_efficiency_old = (total_quantum_time / total_time * 100) if total_time > 0 else 0
    #     qubo_efficiency = (total_qubo_time / total_time * 100) if total_time > 0 else 0
    #
    #     
    #     cameras_per_second = total_assigned / total_time if total_time > 0 else 0
    #
    #     # Комплексная метрика 
    #     quality_time_metric = 0
    #     if objective_value is not None and total_time > 0 and objective_value > 0:
    #         quality_time_metric = total_assigned / (total_time * objective_value)
    #
    #     logger.info(
    #         f"Success QUBO batches: {quantum_success}/{total_batches} ({quantum_success / total_batches * 100:.1f}%)")
    #     logger.info(f"Coverage: {total_assigned}/{self.n_cameras} ({coverage:.1f}%)")
    #     logger.info(f"Total time: {total_time:.2f}s")
    #
    #     # Старая метрика эффективности
    #     logger.info(f"QUBO building time: {total_qubo_time:.2f}s ({qubo_efficiency:.1f}%)")
    #     logger.info(f"Annealing time: {total_quantum_time:.2f}s ({quantum_efficiency_old:.1f}%)")
    #
    #     # Новая метрика производительности
    #     logger.info(f"Performance (cameras/second): {cameras_per_second:.2f}")
    #
    #     if quality_time_metric > 0:
    #         logger.info(f"Quality-Time metric: {quality_time_metric:.6f} (higher is better)")
    #
    #     logger.info(f"Other operations time: {total_time - total_qubo_time - total_quantum_time:.2f}s")
    #
    #     if total_batches > 0:
    #         avg_qubo_time = total_qubo_time / total_batches
    #         avg_quantum_time = total_quantum_time / total_batches
    #         logger.info(f"Average QUBO time per batch: {avg_qubo_time:.3f}s")
    #         logger.info(f"Average annealing time per batch: {avg_quantum_time:.3f}s")
    #
    #     # Сравнение с предыдущими запусками
    #     if len(self.performance_history) > 0:
    #         logger.info(f"\n=== PERFORMANCE COMPARISON ===")
    #         prev_run = self.performance_history[-1]
    #         if 'cameras_per_second' in prev_run:
    #             prev_cps = prev_run['cameras_per_second']
    #             if prev_cps > 0:
    #                 speedup = cameras_per_second / prev_cps
    #                 logger.info(f"Speedup vs previous run: {speedup:.2f}x")

    def _analyze_optimized_performance(self, total_qubo_time, total_quantum_time, total_time,
                                       total_assigned, quantum_success, total_batches,
                                       objective_value=None):
        logger.info("\n" + "=" * 50)
        logger.info("Performance Analysis")
        logger.info("=" * 50)

        coverage = total_assigned / self.n_cameras * 100
        annealing_efficiency = (total_quantum_time / total_time * 100) if total_time > 0 else 0
        qubo_efficiency = (total_qubo_time / total_time * 100) if total_time > 0 else 0

        throughput = total_assigned / total_time if total_time > 0 else 0

        logger.info(
            f"Successful QUBO batches: {quantum_success}/{total_batches} ({quantum_success / total_batches * 100:.1f}%)")
        logger.info(f"Coverage: {total_assigned}/{self.n_cameras} ({coverage:.1f}%)")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"QUBO building time: {total_qubo_time:.2f}s ({qubo_efficiency:.1f}%)")
        logger.info(f"Annealing time: {total_quantum_time:.2f}s ({annealing_efficiency:.1f}%)")
        logger.info(f"Throughput: {throughput:.2f} cameras/second")  # ← ДОБАВИТЬ
        logger.info(f"Other operations time: {total_time - total_qubo_time - total_quantum_time:.2f}s")

        if total_batches > 0:
            avg_qubo_time = total_qubo_time / total_batches
            avg_quantum_time = total_quantum_time / total_batches
            logger.info(f"Average QUBO time per batch: {avg_qubo_time:.3f}s")
            logger.info(f"Average annealing time per batch: {avg_quantum_time:.3f}s")


    # def _analyze_optimized_performance(self, total_qubo_time, total_quantum_time, total_time,
    #                                    total_assigned, quantum_success, total_batches,
    #                                    objective_value=None):
    #     logger.info("\n" + "=" * 50)
    #     logger.info("Performance Analysis")
    #     logger.info("=" * 50)
    #
    #     coverage = total_assigned / self.n_cameras * 100
    #
    #     # МЕТРИКИ ЭФФЕКТИВНОСТИ (как в классическом)
    #     annealing_efficiency = (total_quantum_time / total_time * 100) if total_time > 0 else 0
    #     qubo_efficiency = (total_qubo_time / total_time * 100) if total_time > 0 else 0
    #
    #     # МЕТРИКА ПРОИЗВОДИТЕЛЬНОСТИ (новая)
    #     throughput = total_assigned / total_time if total_time > 0 else 0
    #
    #     logger.info(
    #         f"Successful QUBO batches: {quantum_success}/{total_batches} ({quantum_success / total_batches * 100:.1f}%)")
    #     logger.info(f"Coverage: {total_assigned}/{self.n_cameras} ({coverage:.1f}%)")
    #     logger.info(f"Total time: {total_time:.2f}s")
    #     logger.info(f"QUBO building time: {total_qubo_time:.2f}s ({qubo_efficiency:.1f}%)")
    #     logger.info(f"Annealing time: {total_quantum_time:.2f}s ({annealing_efficiency:.1f}%)")
    #     logger.info(f"Throughput: {throughput:.2f} cameras/second")  # ← НОВАЯ МЕТРИКА
    #     logger.info(f"Other operations time: {total_time - total_qubo_time - total_quantum_time:.2f}s")
    #
    #     if total_batches > 0:
    #         avg_qubo_time = total_qubo_time / total_batches
    #         avg_quantum_time = total_quantum_time / total_batches
    #         logger.info(f"Average QUBO time per batch: {avg_qubo_time:.3f}s")
    #         logger.info(f"Average annealing time per batch: {avg_quantum_time:.3f}s")



    def run_optimized_comparison(self):
        logger.info("=== OpenJij SQA ===")

        results = {}

        logger.info("\n1. Quantum Annealing OpenJij 0.11.6...")

        # Сбрасываем состояния для чистого запуска
        self.assignment_matrix = np.zeros((self.n_cameras, self.n_servers), dtype=int)
        self.remaining_capacity = self.servers_df['capacity_GFLOPS'].values.copy()
        self.assigned_cameras = set()

        try:
            # Выполняем оптимизацию
            quantum_assignment, quantum_objective, quantum_time, quantum_qubo, quantum_annealing = \
                self.solve_with_quantum_optimized(num_reads=150)

            # Вычисляем покрытие
            covered = np.sum(np.any(quantum_assignment, axis=1))

            # Сохраняем результаты
            results['OpenJij-SQA-Windows'] = {
                'assignment': quantum_assignment,
                'objective': quantum_objective,
                'time': quantum_time,
                'qubo_time': quantum_qubo,
                'quantum_time': quantum_annealing,
                'covered': covered,
                'cameras_per_second': covered / quantum_time if quantum_time > 0 else 0,
                'success': True
            }

            logger.info(f"✓ Successfully completed quantum annealing")
            logger.info(f"  • Cameras covered: {covered}/{self.n_cameras} ({covered / self.n_cameras * 100:.1f}%)")
            logger.info(f"  • Total time: {quantum_time:.2f}s")
            logger.info(f"  • Objective value: {quantum_objective:.1f}")

        except Exception as e:
            logger.error(f"OpenJij annealing failed: {e}")
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")

            # Создаем минимальный результат для отображения
            covered = len(self.assigned_cameras) if hasattr(self, 'assigned_cameras') else 0
            quantum_time = self.current_batch_idx * 1.15 if hasattr(self, 'current_batch_idx') else 0

            # Используем текущую матрицу назначений если она существует
            if hasattr(self, 'assignment_matrix'):
                assignment_matrix = self.assignment_matrix
                objective_value = self._calculate_optimized_objective(self.assignment_matrix)
            else:
                assignment_matrix = np.zeros((self.n_cameras, self.n_servers), dtype=int)
                objective_value = float('inf')

            results['OpenJij-SQA-Windows'] = {
                'assignment': assignment_matrix,
                'objective': objective_value,
                'time': quantum_time,
                'qubo_time': 0,
                'quantum_time': 0,
                'covered': covered,
                'cameras_per_second': covered / quantum_time if quantum_time > 0 else 0,
                'success': False
            }

        self._print_optimized_results(results)
        return results

    def _print_optimized_results(self, results):
        print("\n" + "=" * 100)
        print("=" * 100)
        print(f"{'Method':<25} {'Time (s)':<12} {'Target':<15} {'Cameras':<12} {'Eff(%)':<8} {'Cam/s':<12}")
        print("-" * 100)

        for method, data in results.items():
            # Проверяем успешность выполнения
            if 'success' in data and not data['success']:
                print(f"{method:<25} {'FAILED':<12} {'-':<15} {'-':<12} {'-':<8} {'-':<12}")
                continue

            # Проверяем минимальные требования для отображения
            if (data['objective'] == float('inf') or
                    data['time'] == 0 or
                    'assignment' not in data or
                    data['assignment'] is None):
                print(f"{method:<25} {'FAILED':<12} {'-':<15} {'-':<12} {'-':<8} {'-':<12}")
                continue

            # Вычисляем значения если их нет
            if 'covered' not in data or data['covered'] == 0:
                if isinstance(data['assignment'], np.ndarray):
                    data['covered'] = np.sum(np.any(data['assignment'], axis=1))
                else:
                    data['covered'] = 0

            if 'cameras_per_second' not in data or data['cameras_per_second'] == 0:
                data['cameras_per_second'] = data['covered'] / data['time'] if data['time'] > 0 else 0

            efficiency_old = (data['quantum_time'] / data['time'] * 100) if data['time'] > 0 else 0

            print(
                f"{method:<25} {data['time']:<12.2f} {data['objective']:<15.1f} "
                f"{data['covered']:<12} {efficiency_old:<8.1f} {data['cameras_per_second']:<12.2f}"
            )

        print("=" * 100)
        print("Eff(%) = Annealing time / Total time (higher means more time spent on annealing)")
        print("Cam/s = Cameras per second (higher is better)")
        print("=" * 100)

    def print_performance_history(self):
        if not self.performance_history:
            logger.info("No performance history available.")
            return

        logger.info("\n" + "=" * 60)
        logger.info("PERFORMANCE HISTORY")
        logger.info("=" * 60)
        logger.info(f"{'Run ID':<15} {'Time (s)':<10} {'Cameras':<10} {'Cam/s':<12} {'Eff(%)':<8} {'Objective':<10}")

        for i, record in enumerate(self.performance_history):
            logger.info(
                f"{record.get('run_id', f'Run_{i + 1}'):<15} "
                f"{record['total_time']:<10.2f} "
                f"{record['coverage']:<10} "
                f"{record['cameras_per_second']:<12.2f} "
                f"{record['annealing_efficiency']:<8.1f} "
                f"{record['objective']:<10.1f}"
            )


def main():
    logger.info("=== OPENJIJ 0.11.6 WINDOWS 20000 cameras ===")

    scheduler = WindowsOpenJijScheduler(
        n_cameras=20000,
        n_servers=800,
        batch_size=80,
        max_servers_per_batch=20
    )

    utilization = scheduler.generate_realistic_data()
    logger.info(f"Capacity: {utilization:.1f}%")
    results = scheduler.run_optimized_comparison()
    scheduler.print_performance_history()

    return results, scheduler


if __name__ == "__main__":
    results, scheduler = main()
