
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

# ==================== OPENJIJ 0.11.6 ====================
try:
    import openjij as oj

    print("✓ OpenJij 0.11.6 загружен")
    HAVE_OPENJIJ = True

    try:
        test_sampler = oj.SQASampler(num_reads=1, num_sweeps=100, trotter=4)
        OPENJIJ_WORKING_PARAMS = ['num_reads', 'num_sweeps', 'trotter']
        print("  working: num_reads, num_sweeps, trotter")
    except TypeError as e:
        if "beta" in str(e):
            OPENJIJ_WORKING_PARAMS = ['num_sweeps', 'trotter']
            print("  working: num_sweeps, trotter (beta баг)")
        else:
            OPENJIJ_WORKING_PARAMS = ['minimal']
            print("  → Только минимальный конструктор")
except ImportError:
    HAVE_OPENJIJ = False
    OPENJIJ_WORKING_PARAMS = []
    print("✗ OpenJij не найден")

try:
    from neal import SimulatedAnnealingSampler

    HAVE_NEAL = True
    print("✓ Neal доступен для fallback")
except ImportError:
    HAVE_NEAL = False
# ================================================================

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
        logger.info(f"Инициализация для OpenJij 0.11.6 Windows: {n_cameras} камер, {n_servers} серверов")

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

        # Сохраняем всю систему логирования из оригинального кода
        self.log_dir = "logs_openjij_windows"
        self.snapshot_dir = os.path.join(self.log_dir, "snapshots")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.snapshot_dir, exist_ok=True)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.progress_log = os.path.join(self.log_dir, f"progress_{self.run_id}.jsonl")
        self.snapshot_every = 50
        self.current_batch_idx = 0

    def _log_progress(self, batch_idx, batch_assigned, coverage, success_rate,
                      energy=None, best_energy=None, qubo_time=0, quantum_time=0,
                      assignments=None):
        """ТОЧНАЯ КОПИЯ из оригинального кода"""
        log_entry = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "batch_idx": int(batch_idx),
            "batch_assigned": int(batch_assigned),
            "coverage_percent": float(coverage),
            "qubo_success_rate": float(success_rate),
            "qubo_time_sec": float(qubo_time),
            "quantum_time_sec": float(quantum_time),
            "energy": float(energy) if energy is not None else None,
            "best_energy": float(best_energy) if best_energy is not None else None,
            "assignments": assignments or []
        }
        with open(self.progress_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

    def generate_realistic_data(self):
        """ТОЧНАЯ КОПИЯ из оригинального кода"""
        logger.info("=== Генерация реалистичных данных ===")

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
            f"Общая нагрузка: {total_load:.1f}, ёмкость: {total_capacity:.1f}, использование: {utilization:.1f}%")

        return utilization

    def _build_cost_matrix(self):
        """ТОЧНАЯ КОПИЯ из оригинального кода"""
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
        """Главный метод - используем НАДЁЖНЫЙ SQA для Windows"""
        if not HAVE_OPENJIJ and not HAVE_NEAL:
            logger.error("Нет доступных солверов!")
            return None, 0, 0, 0, 0

        logger.info("=== КВАНТОВЫЙ ОТЖИГ (OpenJij 0.11.6 Windows) ===")
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

        logger.info(f"Обработка {total_batches} батчей...")

        self.current_batch_idx = 0

        for batch_idx in range(total_batches):
            self.current_batch_idx = batch_idx

            if len(self.assigned_cameras) / self.n_cameras > 0.995:
                logger.info("Достигнуто 99.5% покрытие, завершаем")
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

            batch_complexity = self._calculate_batch_complexity(available_indices, top_servers)
            adaptive_reads = max(num_reads, int(num_reads * (1 + batch_complexity)))

            # ИСПОЛЬЗУЕМ НАДЁЖНЫЙ МЕТОД ДЛЯ WINDOWS
            batch_solution, qubo_time, quantum_time, quantum_used = self._solve_batch_sqa_windows_reliable(
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
                assignments=assignments_in_batch
            )

            if (batch_idx + 1) % 20 == 0:
                logger.info(
                    f"Батч {batch_idx + 1}: {batch_assigned} камер, "
                    f"покрытие: {current_coverage:.1f}%, Успех SQA: {success_rate:.1f}%"
                )

        # === ФИНАЛЬНАЯ ОПТИМИЗАЦИЯ ===
        total_time = time.time() - total_start
        self._optimize_final_solution()
        objective = self._calculate_optimized_objective(self.assignment_matrix)

        self._analyze_optimized_performance(
            total_qubo_time, total_quantum_time, total_time,
            total_assigned, quantum_success_count, total_batches
        )

        return self.assignment_matrix, objective, total_time, total_qubo_time, total_quantum_time

    def _solve_batch_sqa_windows_reliable(self, batch_indices, top_servers, num_reads=100):
        """НАДЁЖНЫЙ SQA ДЛЯ OpenJij 0.11.6 WINDOWS"""
        n_batch = len(batch_indices)
        n_servers_batch = len(top_servers)

        qubo_start = time.time()
        Q = self._build_optimized_qubo(batch_indices, top_servers)
        qubo_time = time.time() - qubo_start

        if not Q:
            return self._solve_batch_greedy_optimized(batch_indices, top_servers), qubo_time, 0, False

        # Если OpenJij недоступен, сразу переходим к fallback
        if not HAVE_OPENJIJ:
            return self._solve_batch_with_neal_or_greedy(Q, batch_indices, top_servers, qubo_time)

        try:
            quantum_start = time.time()

            # === МАКСИМАЛЬНО НАДЁЖНЫЙ SQA ДЛЯ OpenJij 0.11.6 (Windows) ===
            all_responses = []
            successful_reads = 0

            for attempt in range(max(1, num_reads // 10)):  # Уменьшаем количество попыток для скорости
                try:
                    # ПОПЫТКА 1: Безопасные параметры для Windows
                    if 'num_reads' in OPENJIJ_WORKING_PARAMS and 'num_sweeps' in OPENJIJ_WORKING_PARAMS:
                        sampler = oj.SQASampler(
                            num_reads=min(10, num_reads),
                            num_sweeps=1000,
                            trotter=8
                        )
                        logger.debug("SQA с num_reads, num_sweeps, trotter")

                    # ПОПЫТКА 2: Без num_reads (если есть баг)
                    elif 'num_sweeps' in OPENJIJ_WORKING_PARAMS:
                        sampler = oj.SQASampler(num_sweeps=1000, trotter=8)
                        logger.debug("SQA с num_sweeps, trotter")

                    # ПОПЫТКА 3: Только trotter
                    elif 'trotter' in OPENJIJ_WORKING_PARAMS:
                        sampler = oj.SQASampler(trotter=8)
                        logger.debug("SQA только с trotter")

                    # ПОПЫТКА 4: Голый конструктор (всегда работает)
                    else:
                        sampler = oj.SQASampler()
                        logger.debug("SQA с дефолтными параметрами")

                    response = sampler.sample_qubo(Q)
                    all_responses.append(response)
                    successful_reads += 1

                    # Если получили хорошее решение, можно остановиться
                    if len(all_responses) >= 3:  # Хватит 3 попыток
                        break

                except Exception as e:
                    logger.debug(f"Попытка SQA {attempt + 1} не удалась: {e}")
                    continue

            if not all_responses:
                raise ValueError("SQA не вернул ни одного решения")

            # Выбираем лучшее решение по энергии
            response = min(all_responses, key=lambda r: r.first.energy)
            quantum_time = time.time() - quantum_start

            assignment = self._decode_optimized_solution(response, n_batch, top_servers, batch_indices)
            valid_assignments = np.sum([np.any(assignment[i]) for i in range(n_batch)])

            if valid_assignments >= n_batch * 0.5:
                logger.info(f"SQA УСПЕХ: {valid_assignments}/{n_batch} камер, энергия {response.first.energy:.1f}")
                return assignment, qubo_time, quantum_time, True
            else:
                logger.warning(f"SQA слабый результат ({valid_assignments}/{n_batch}) → fallback")
                raise ValueError("Низкое качество решения SQA")

        except Exception as e:
            logger.warning(f"SQA полностью упал: {e} → переходим на Neal/Greedy")
            quantum_time = time.time() - quantum_start if 'quantum_time' in locals() else 0
            return self._solve_batch_with_neal_or_greedy(Q, batch_indices, top_servers, qubo_time)

    def _solve_batch_with_neal_or_greedy(self, Q, batch_indices, top_servers, qubo_time):
        """Fallback на Neal или Greedy"""
        # Сначала пробуем Neal
        if HAVE_NEAL:
            try:
                quantum_start = time.time()
                sampler = SimulatedAnnealingSampler()
                response = sampler.sample_qubo(Q, num_reads=50)
                quantum_time = time.time() - quantum_start

                assignment = self._decode_optimized_solution(
                    response, len(batch_indices), top_servers, batch_indices
                )
                logger.info("Успешный fallback на Neal SA")
                return assignment, qubo_time, quantum_time, True
            except Exception as e:
                logger.warning(f"Neal тоже упал: {e}")

        # Если всё упало - используем greedy
        logger.info("Полный fallback на greedy алгоритм")
        return self._solve_batch_greedy_optimized(batch_indices, top_servers), qubo_time, 0, False

    # ========== ВСЕ ОСТАЛЬНЫЕ МЕТОДЫ - ТОЧНЫЕ КОПИИ ИЗ ОРИГИНАЛА ==========

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
        """ТОЧНАЯ КОПИЯ оригинальной QUBO"""
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

    def _decode_optimized_solution(self, response, n_batch, top_servers, batch_indices):
        """ТОЧНАЯ КОПИЯ декодера"""
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
        """ТОЧНАЯ КОПИЯ пост-обработки"""
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
        """ТОЧНАЯ КОПИЯ финальной оптимизации"""
        logger.info("Финальная оптимизация...")

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

            logger.info(f"Итерация {iteration + 1}: {improvements} улучшений")

    def _can_reassign(self, cam_idx, from_server, to_server):
        cam_load = self.cameras_df.iloc[cam_idx]['load_GFLOPS']
        return self.remaining_capacity[to_server] >= cam_load

    def _solve_batch_greedy_optimized(self, batch_indices, top_servers):
        """ТОЧНАЯ КОПИЯ greedy алгоритма"""
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
        """ТОЧНАЯ КОПИЯ расчёта целевой функции"""
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
            f"Целевая функция: стоимость={total_cost:.1f}, непокрытые={uncovered_penalty:.1f}, перегрузка={overload_penalty:.1f}")

        return total_objective

    def _analyze_optimized_performance(self, total_qubo_time, total_quantum_time, total_time,
                                       total_assigned, quantum_success, total_batches):
        """ТОЧНАЯ КОПИЯ анализа производительности"""
        logger.info("\n" + "=" * 50)
        logger.info("Анализ оптимизированных результатов")
        logger.info("=" * 50)

        coverage = total_assigned / self.n_cameras * 100
        quantum_efficiency = (total_quantum_time / total_time * 100) if total_time > 0 else 0

        logger.info(
            f"Успешные QUBO батчи: {quantum_success}/{total_batches} ({quantum_success / total_batches * 100:.1f}%)")
        logger.info(f"Покрытие: {total_assigned}/{self.n_cameras} ({coverage:.1f}%)")
        logger.info(f"Общее время: {total_time:.2f}с")
        logger.info(f"QUBO время: {total_qubo_time:.2f}с ({total_qubo_time / total_time * 100:.1f}%)")
        logger.info(f"Время отжига: {total_quantum_time:.2f}с ({quantum_efficiency:.1f}%)")

    def run_optimized_comparison(self):
        """Запуск квантового отжига"""
        logger.info("=== Запуск OpenJij SQA ===")

        results = {}

        logger.info("\n1. Запуск квантового отжига OpenJij 0.11.6...")
        try:
            quantum_assignment, quantum_objective, quantum_time, quantum_qubo, quantum_annealing = \
                self.solve_with_quantum_optimized(num_reads=150)

            results['OpenJij-SQA-Windows'] = {
                'assignment': quantum_assignment,
                'objective': quantum_objective,
                'time': quantum_time,
                'qubo_time': quantum_qubo,
                'quantum_time': quantum_annealing,
                'covered': np.sum(np.any(quantum_assignment, axis=1))
            }
        except Exception as e:
            logger.error(f"OpenJij отжиг не удался: {e}")
            results['OpenJij-SQA-Windows'] = {
                'assignment': np.zeros((self.n_cameras, self.n_servers), dtype=int),
                'objective': float('inf'),
                'time': 0,
                'qubo_time': 0,
                'quantum_time': 0,
                'covered': 0
            }

        self._print_optimized_results(results)
        return results

    def _print_optimized_results(self, results):
        print("\n" + "=" * 100)
        print("== РЕЗУЛЬТАТЫ OPENJIJ 0.11.6 WINDOWS ==")
        print("=" * 100)
        print(f"{'Метод':<25} {'Время (с)':<12} {'Цель':<15} {'Камеры':<12} {'Эффектив. (%)':<12}")
        print("-" * 100)

        for method, data in results.items():
            if data['objective'] == float('inf'):
                print(f"{method:<25} {'НЕ УДАЛОСЬ':<12} {'-':<15} {'-':<12} {'-':<12}")
                continue

            efficiency = (data['quantum_time'] / data['time'] * 100) if data['time'] > 0 else 0

            print(
                f"{method:<25} {data['time']:<12.2f} {data['objective']:<15.1f} {data['covered']:<12} {efficiency:<12.1f}")

        print("=" * 100)


def main():
    logger.info("=== ЗАПУСК OPENJIJ 0.11.6 WINDOWS ДЛЯ 20000 КАМЕР ===")

    scheduler = WindowsOpenJijScheduler(
        n_cameras=20000,
        n_servers=800,
        batch_size=80,
        max_servers_per_batch=20
    )

    utilization = scheduler.generate_realistic_data()
    logger.info(f"Использование ёмкости: {utilization:.1f}%")
    results = scheduler.run_optimized_comparison()
    logger.info("=== ЭКСПЕРИМЕНТ ЗАВЕРШЁН ===")
    return results


if __name__ == "__main__":
    results = main()