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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('classical_greedy_windows.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class WindowsGreedyScheduler:
    def __init__(self, n_cameras=20000, n_servers=800, batch_size=80, max_servers_per_batch=20, random_seed=42):
        logger.info(f"initialization Greedy windows: {n_cameras} cameras, {n_servers} servers")

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

        self.log_dir = "logs_greedy_windows"
        self.snapshot_dir = os.path.join(self.log_dir, "snapshots")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.snapshot_dir, exist_ok=True)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.progress_log = os.path.join(self.log_dir, f"progress_{self.run_id}.jsonl")
        self.snapshot_every = 50
        self.current_batch_idx = 0
        self.performance_history = []

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
            "annealing_time_sec": float(quantum_time),
            "energy": float(energy) if energy is not None else None,
            "best_energy": float(best_energy) if best_energy is not None else None,
            "assignments": assignments or [],
            "qubo_variables": len(assignments) * 20 if assignments else 0,  # Аналогия, но для greedy variables=0
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

    def solve_with_greedy(self):
        logger.info("Classical Greedy Algorithm")
        total_start = time.time()

        self.assignment_matrix = np.zeros((self.n_cameras, self.n_servers), dtype=int)
        self.remaining_capacity = self.servers_df['capacity_GFLOPS'].values.copy()
        self.assigned_cameras = set()

        priority_scores = self.cameras_df['priority'].values * self.cameras_df['load_GFLOPS'].values
        sorted_indices = np.argsort(-priority_scores)
        total_batches = (self.n_cameras + self.batch_size - 1) // self.batch_size

        total_qubo_time = 0  # Для аналогии, но в greedy = 0
        total_greedy_time = 0  # Аналог quantum_time
        greedy_success_count = 0  # Всегда success для greedy
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

            top_servers = self._select_servers(available_indices)  # Упрощенная версия без optimized
            if len(top_servers) == 0:
                continue

            processed_batches += 1

            batch_solution, greedy_time, greedy_used, batch_energy = self._solve_batch_greedy(
                available_indices, top_servers
            )

            total_qubo_time += 0  # Нет QUBO
            total_greedy_time += greedy_time
            if greedy_used:
                greedy_success_count += 1

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
            success_rate = (greedy_success_count / (batch_idx + 1)) * 100 if batch_idx > 0 else 0

            self._log_progress(
                batch_idx=batch_idx,
                batch_assigned=batch_assigned,
                coverage=current_coverage,
                success_rate=success_rate,
                energy=batch_energy,
                best_energy=batch_energy,
                qubo_time=0,
                quantum_time=greedy_time,
                assignments=assignments_in_batch
            )

            if (batch_idx + 1) % 20 == 0:
                logger.info(
                    f"Batch {batch_idx + 1}: {batch_assigned} cameras, "
                    f"coverage: {current_coverage:.1f}%, success Greedy: {success_rate:.1f}%"
                )

        total_time = time.time() - total_start

        logger.info(f"Processed batches: {processed_batches}/{total_batches}")
        if processed_batches > 0:
            logger.info(f"Average Greedy time per batch: {total_greedy_time / processed_batches:.3f}s")

        self._optimize_final_solution()
        objective = self._calculate_optimized_objective(self.assignment_matrix)

        self._analyze_optimized_performance(
            total_qubo_time, total_greedy_time, total_time,
            total_assigned, greedy_success_count, total_batches,
            objective_value=objective
        )

        performance_record = {
            'run_id': self.run_id,
            'timestamp': datetime.now(),
            'total_time': total_time,
            'annealing_time': total_greedy_time,
            'coverage': total_assigned,
            'objective': objective,
            'cameras_per_second': total_assigned / total_time if total_time > 0 else 0,
            'annealing_efficiency': (total_greedy_time / total_time * 100) if total_time > 0 else 0
        }
        self.performance_history.append(performance_record)

        return self.assignment_matrix, objective, total_time, total_qubo_time, total_greedy_time

    def _solve_batch_greedy(self, batch_indices, top_servers):
        greedy_start = time.time()
        assignment = self._solve_batch_greedy_classic(batch_indices, top_servers)
        greedy_time = time.time() - greedy_start
        return assignment, greedy_time, True, 0.0

    def _select_servers(self, batch_indices):
        batch_loads = self.cameras_df.iloc[batch_indices]['load_GFLOPS'].values
        min_load = np.min(batch_loads)
        valid_servers = np.where(self.remaining_capacity >= min_load)[0]
        if len(valid_servers) == 0:
            return np.array([])
        return valid_servers[:self.max_servers_per_batch]

    def _solve_batch_greedy_classic(self, batch_indices, top_servers):
        n_batch = len(batch_indices)
        assignment = np.zeros((n_batch, len(top_servers)), dtype=int)

        remaining_capacity = self.remaining_capacity[top_servers].copy()

        batch_data = self.cameras_df.iloc[batch_indices]
        priority_scores = batch_data['priority'].values * batch_data['load_GFLOPS'].values
        sorted_indices = np.argsort(-priority_scores)

        for pos in sorted_indices:
            cam_idx = batch_indices[pos]
            cam_load = self.cameras_df.iloc[cam_idx]['load_GFLOPS']

            best_server = -1
            best_cost = float('inf')

            for j, server_idx in enumerate(top_servers):
                if remaining_capacity[j] >= cam_load:
                    cost = self.cost_matrix[cam_idx, server_idx]
                    if cost < best_cost:
                        best_cost = cost
                        best_server = j

            if best_server != -1:
                assignment[pos, best_server] = 1
                remaining_capacity[best_server] -= cam_load

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

    def _analyze_optimized_performance(self, total_qubo_time, total_greedy_time, total_time,
                                       total_assigned, greedy_success, total_batches,
                                       objective_value=None):
        logger.info("\n" + "=" * 50)
        logger.info("Performance Analysis")
        logger.info("=" * 50)

        coverage = total_assigned / self.n_cameras * 100

        annealing_efficiency = (total_greedy_time / total_time * 100) if total_time > 0 else 0
        qubo_efficiency = (total_qubo_time / total_time * 100) if total_time > 0 else 0

        throughput = total_assigned / total_time if total_time > 0 else 0

        logger.info(
            f"Successful Greedy batches: {greedy_success}/{total_batches} ({greedy_success / total_batches * 100:.1f}%)")
        logger.info(f"Coverage: {total_assigned}/{self.n_cameras} ({coverage:.1f}%)")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"QUBO building time: {total_qubo_time:.2f}s ({qubo_efficiency:.1f}%)")  # 0%
        logger.info(f"Greedy time: {total_greedy_time:.2f}s ({annealing_efficiency:.1f}%)")
        logger.info(f"Throughput: {throughput:.2f} cameras/second")
        logger.info(f"Other operations time: {total_time - total_qubo_time - total_greedy_time:.2f}s")

        if total_batches > 0:
            avg_qubo_time = total_qubo_time / total_batches
            avg_greedy_time = total_greedy_time / total_batches
            logger.info(f"Average QUBO time per batch: {avg_qubo_time:.3f}s")
            logger.info(f"Average greedy time per batch: {avg_greedy_time:.3f}s")

    def run_optimized_comparison(self):
        logger.info("=== Classical Greedy ===")

        results = {}

        logger.info("\n1. Classical Greedy Algorithm...")

        self.assignment_matrix = np.zeros((self.n_cameras, self.n_servers), dtype=int)
        self.remaining_capacity = self.servers_df['capacity_GFLOPS'].values.copy()
        self.assigned_cameras = set()

        try:
            greedy_assignment, greedy_objective, greedy_time, greedy_qubo, greedy_annealing = \
                self.solve_with_greedy()

            covered = np.sum(np.any(greedy_assignment, axis=1))
            results['Greedy-Windows'] = {
                'assignment': greedy_assignment,
                'objective': greedy_objective,
                'time': greedy_time,
                'qubo_time': greedy_qubo,
                'quantum_time': greedy_annealing,  # Аналогия
                'covered': covered,
                'cameras_per_second': covered / greedy_time if greedy_time > 0 else 0,
                'success': True
            }

            logger.info(f"✓ Successfully completed greedy algorithm")
            logger.info(f"  • Cameras covered: {covered}/{self.n_cameras} ({covered / self.n_cameras * 100:.1f}%)")
            logger.info(f"  • Total time: {greedy_time:.2f}s")
            logger.info(f"  • Objective value: {greedy_objective:.1f}")

        except Exception as e:
            logger.error(f"Greedy failed: {e}")
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            covered = len(self.assigned_cameras) if hasattr(self, 'assigned_cameras') else 0
            greedy_time = self.current_batch_idx * 1.15 if hasattr(self, 'current_batch_idx') else 0

            if hasattr(self, 'assignment_matrix'):
                assignment_matrix = self.assignment_matrix
                objective_value = self._calculate_optimized_objective(self.assignment_matrix)
            else:
                assignment_matrix = np.zeros((self.n_cameras, self.n_servers), dtype=int)
                objective_value = float('inf')

            results['Greedy-Windows'] = {
                'assignment': assignment_matrix,
                'objective': objective_value,
                'time': greedy_time,
                'qubo_time': 0,
                'quantum_time': 0,
                'covered': covered,
                'cameras_per_second': covered / greedy_time if greedy_time > 0 else 0,
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
            if 'success' in data and not data['success']:
                print(f"{method:<25} {'FAILED':<12} {'-':<15} {'-':<12} {'-':<8} {'-':<12}")
                continue

            if (data['objective'] == float('inf') or
                    data['time'] == 0 or
                    'assignment' not in data or
                    data['assignment'] is None):
                print(f"{method:<25} {'FAILED':<12} {'-':<15} {'-':<12} {'-':<8} {'-':<12}")
                continue
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
        print("Eff(%) = Greedy time / Total time (higher means more time spent on greedy)")
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
    logger.info("===  GREEDY  ===")

    scheduler = WindowsGreedyScheduler(
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






# C:\Users\Yedman\PycharmProjects\Qannealing\.venv\Scripts\python.exe C:\Users\Yedman\PycharmProjects\QAnnealing\greedy.py
# 2025-12-31 14:23:44,365 - INFO - === CLASSICAL GREEDY WINDOWS 20000 cameras ===
# 2025-12-31 14:23:44,365 - INFO - initialization Greedy windows: 20000 cameras, 800 servers
# 2025-12-31 14:23:44,366 - INFO - ===Data Generation===
# 2025-12-31 14:23:45,021 - INFO - total load: 88453.0, capacity: 372165.7, usage: 23.8%
# 2025-12-31 14:23:45,021 - INFO - Capacity: 23.8%
# 2025-12-31 14:23:45,021 - INFO - === Classical Greedy ===
# 2025-12-31 14:23:45,022 - INFO -
# 1. Classical Greedy Algorithm...
# 2025-12-31 14:23:45,022 - INFO - Classical Greedy Algorithm
# 2025-12-31 14:23:45,022 - INFO - Processing 250 batches...
# 2025-12-31 14:23:46,523 - INFO - Batch 20: 68 cameras, coverage: 7.5%, success Greedy: 100.0%
# 2025-12-31 14:23:47,971 - INFO - Batch 40: 68 cameras, coverage: 14.7%, success Greedy: 100.0%
# 2025-12-31 14:23:49,457 - INFO - Batch 60: 77 cameras, coverage: 22.3%, success Greedy: 100.0%
# 2025-12-31 14:23:50,948 - INFO - Batch 80: 55 cameras, coverage: 29.7%, success Greedy: 100.0%
# 2025-12-31 14:23:52,446 - INFO - Batch 100: 74 cameras, coverage: 37.1%, success Greedy: 100.0%
# 2025-12-31 14:23:53,929 - INFO - Batch 120: 80 cameras, coverage: 44.6%, success Greedy: 100.0%
# 2025-12-31 14:23:55,478 - INFO - Batch 140: 79 cameras, coverage: 52.4%, success Greedy: 100.0%
# 2025-12-31 14:23:57,046 - INFO - Batch 160: 80 cameras, coverage: 60.3%, success Greedy: 100.0%
# 2025-12-31 14:23:58,604 - INFO - Batch 180: 77 cameras, coverage: 68.2%, success Greedy: 100.0%
# 2025-12-31 14:24:00,162 - INFO - Batch 200: 80 cameras, coverage: 76.1%, success Greedy: 100.0%
# 2025-12-31 14:24:01,894 - INFO - Batch 220: 73 cameras, coverage: 83.9%, success Greedy: 100.0%
# 2025-12-31 14:24:03,463 - INFO - Batch 240: 80 cameras, coverage: 91.8%, success Greedy: 100.0%
# 2025-12-31 14:24:04,294 - INFO - Processed batches: 250/250
# 2025-12-31 14:24:04,294 - INFO - Average Greedy time per batch: 0.005s
# 2025-12-31 14:24:04,294 - INFO - final optimization...
# 2025-12-31 14:24:13,137 - INFO - iteration 1: 18465 improvements
# 2025-12-31 14:24:18,937 - INFO - iteration 2: 74 improvements
# 2025-12-31 14:24:26,127 - INFO - Objective(target functions): cost=5714.1, uncovered=12870.0, overload=0.0
# 2025-12-31 14:24:26,127 - INFO -
# ==================================================
# 2025-12-31 14:24:26,127 - INFO - Performance Analysis
# 2025-12-31 14:24:26,127 - INFO - ==================================================
# 2025-12-31 14:24:26,127 - INFO - Successful Greedy batches: 250/250 (100.0%)
# 2025-12-31 14:24:26,127 - INFO - Coverage: 19142/20000 (95.7%)
# 2025-12-31 14:24:26,127 - INFO - Total time: 19.27s
# 2025-12-31 14:24:26,127 - INFO - QUBO building time: 0.00s (0.0%)
# 2025-12-31 14:24:26,127 - INFO - Greedy time: 1.29s (6.7%)
# 2025-12-31 14:24:26,127 - INFO - Throughput: 993.21 cameras/second
# 2025-12-31 14:24:26,127 - INFO - Other operations time: 17.99s
# 2025-12-31 14:24:26,127 - INFO - Average QUBO time per batch: 0.000s
# 2025-12-31 14:24:26,127 - INFO - Average greedy time per batch: 0.005s
#
# ====================================================================================================
# ====================================================================================================
# Method                    Time (s)     Target          Cameras      Eff(%)   Cam/s
# ----------------------------------------------------------------------------------------------------
# Greedy-Windows            19.27        18584.1         19142        6.7      993.21
# ====================================================================================================
# Eff(%) = Greedy time / Total time (higher means more time spent on greedy)
# Cam/s = Cameras per second (higher is better)
# ====================================================================================================
# 2025-12-31 14:24:26,142 - INFO - ✓ Successfully completed greedy algorithm
# 2025-12-31 14:24:26,142 - INFO -   • Cameras covered: 19142/20000 (95.7%)
# 2025-12-31 14:24:26,142 - INFO -   • Total time: 19.27s
# 2025-12-31 14:24:26,142 - INFO -   • Objective value: 18584.1
# 2025-12-31 14:24:26,142 - INFO -
# ============================================================
# 2025-12-31 14:24:26,142 - INFO - PERFORMANCE HISTORY
# 2025-12-31 14:24:26,142 - INFO - ============================================================
# 2025-12-31 14:24:26,142 - INFO - Run ID          Time (s)   Cameras    Cam/s        Eff(%)   Objective
# 2025-12-31 14:24:26,142 - INFO - 20251231_142344 19.27      19142      993.21       6.7      18584.1
#
# Process finished with exit code 0
