import argparse
import json
import os
import time
from datetime import datetime

import numpy as np

from capacity_stress_experiment import (
    CapacityStressExperiment,
    LAMBDA_ASSIGN,
    configure_logger,
)


LOG_ROOT = "logs_prc_qubo_conflict"


class SimpleFirst:
    def __init__(self, sample, energy):
        self.sample = sample
        self.energy = energy


class SimpleResponse:
    def __init__(self, sample, energy):
        self.first = SimpleFirst(sample, energy)


class PRCQuboConflictStressExperiment(CapacityStressExperiment):
    """Stress-aware PRC-QUBO variant kept separate from the original pipeline.

    The parent class supplies the same data generator, cost model, capacity
    scaling, logging schema, QUBO solving interface, and evaluation objective.
    This class only changes the PRC batch Hamiltonian and decoding stage.
    """

    def __init__(
        self,
        solver="SQA",
        n_cameras=20000,
        n_servers=800,
        batch_size=80,
        max_servers_per_batch=20,
        random_seed=42,
        capacity_scale=1.0,
        num_reads=150,
        num_sweeps=1000,
        trotter=8,
        log_every=20,
        final_opt=False,
        coverage_stop_threshold=0.995,
        log_root=LOG_ROOT,
        conflict_weight=180.0,
        guard_weight=45.0,
        use_guard=True,
        use_conflict=True,
        conflict_mode="all",
        conflict_margin=1.0,
        capacity_aware_decoder=True,
        decoder_capacity_weight=8.0,
        decoder_alternative_penalty=0.75,
        method_label=None,
    ):
        super().__init__(
            formulation="PRC-QUBO",
            solver=solver,
            n_cameras=n_cameras,
            n_servers=n_servers,
            batch_size=batch_size,
            max_servers_per_batch=max_servers_per_batch,
            random_seed=random_seed,
            capacity_scale=capacity_scale,
            num_reads=num_reads,
            num_sweeps=num_sweeps,
            trotter=trotter,
            log_every=log_every,
            final_opt=final_opt,
            coverage_stop_threshold=coverage_stop_threshold,
            log_root=log_root,
        )
        self.use_conflict = bool(use_conflict)
        self.capacity_aware_decoder = bool(capacity_aware_decoder)
        if method_label:
            self.report_formulation = method_label
        elif self.use_conflict and self.capacity_aware_decoder:
            self.report_formulation = "PRC-QUBO-C"
        elif self.use_conflict:
            self.report_formulation = "PRC-QUBO-C-no-decoder"
        elif self.capacity_aware_decoder:
            self.report_formulation = "PRC-QUBO-D"
        else:
            self.report_formulation = "PRC-QUBO-guard"
        self.conflict_weight = float(conflict_weight)
        self.guard_weight = float(guard_weight)
        self.use_guard = bool(use_guard)
        self.conflict_mode = conflict_mode if self.use_conflict else "disabled"
        self.conflict_margin = float(conflict_margin)
        self.decoder_capacity_weight = float(decoder_capacity_weight)
        self.decoder_alternative_penalty = float(decoder_alternative_penalty)

        safe_label = self.report_formulation.lower().replace("-", "_")
        safe_name = f"{safe_label}_{self.solver.lower()}_s{str(capacity_scale).replace('.', 'p')}"
        self.log_dir = os.path.join(log_root, safe_name)
        os.makedirs(self.log_dir, exist_ok=True)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.progress_log = os.path.join(self.log_dir, f"progress_{self.run_id}.jsonl")
        self.summary_log = os.path.join(self.log_dir, f"summary_{self.run_id}.json")
        self.log_file = os.path.join(self.log_dir, f"run_{self.run_id}.log")
        self.logger = configure_logger(self.log_file)

        self.conflict_term_rows = []
        self.decoder_alternative_assignments = 0
        self.decoder_zero_selection_rescues = 0
        self.decoder_capacity_rescues = 0
        self._last_linear = None
        self._last_batch_indices = None
        self._last_top_servers = None

    def _ensure_solver_available(self):
        if self.solver == "DIAGNOSTIC":
            return
        return super()._ensure_solver_available()

    def build_prc_qubo(self, batch_indices, top_servers):
        q = {}
        n_batch = len(batch_indices)
        n_top = len(top_servers)
        linear = np.full((n_batch, n_top), 100.0, dtype=float)
        conflict_terms = 0

        for i, cam_idx in enumerate(batch_indices):
            load = float(self.load_gflops[cam_idx])
            priority_weight = float(4 - self.priority[cam_idx])
            for j, server_idx in enumerate(top_servers):
                var = self.var_name(i, server_idx)
                residual = float(self.remaining_capacity[server_idx])
                cost = float(self.cost_matrix[cam_idx, server_idx])
                if load <= residual:
                    coeff = -25.0 * (1.0 - cost) * priority_weight
                    if self.use_guard:
                        coeff += self.guard_weight * load / (residual + 1e-12)
                else:
                    coeff = 100.0
                linear[i, j] = coeff
                q[(var, var)] = float(coeff)

        self.add_one_hot_terms(q, n_batch, top_servers, LAMBDA_ASSIGN)

        if self.use_conflict and self.conflict_weight > 0:
            loads = self.load_gflops[batch_indices].astype(float)
            for j, server_idx in enumerate(top_servers):
                residual = float(max(self.remaining_capacity[server_idx], 1e-9))
                residual_sq = residual * residual
                for i1 in range(n_batch):
                    if loads[i1] > residual:
                        continue
                    var1 = self.var_name(i1, server_idx)
                    for i2 in range(i1 + 1, n_batch):
                        if loads[i2] > residual:
                            continue
                        if self.conflict_mode == "risky" and (loads[i1] + loads[i2]) <= self.conflict_margin * residual:
                            continue
                        var2 = self.var_name(i2, server_idx)
                        coeff = self.conflict_weight * loads[i1] * loads[i2] / residual_sq
                        self.add_q(q, var1, var2, float(coeff))
                        conflict_terms += 1

        self._last_linear = linear
        self._last_batch_indices = np.asarray(batch_indices, dtype=int)
        self._last_top_servers = np.asarray(top_servers, dtype=int)

        stats = self.collect_qubo_stats(q, batch_indices, top_servers)
        stats["capacity_conflict_quadratic_count"] = int(conflict_terms)
        stats["capacity_conflict_weight"] = float(self.conflict_weight)
        self.conflict_term_rows.append(conflict_terms)
        return q, stats

    def solve_qubo(self, q):
        if self.solver != "DIAGNOSTIC":
            return super().solve_qubo(q)
        if not q:
            return None, 0.0, False, "empty_qubo"

        start = time.time()
        sample = {}
        n_batch = 0 if self._last_batch_indices is None else len(self._last_batch_indices)
        top_servers = [] if self._last_top_servers is None else list(self._last_top_servers)
        for i in range(n_batch):
            best_var = None
            best_coeff = float("inf")
            for server_idx in top_servers:
                var = self.var_name(i, server_idx)
                coeff = float(q.get((var, var), 100.0))
                sample[var] = 0
                if coeff < best_coeff:
                    best_coeff = coeff
                    best_var = var
            if best_var is not None and best_coeff < 99.0:
                sample[best_var] = 1

        energy = self.evaluate_qubo_energy(q, sample)
        return SimpleResponse(sample, energy), time.time() - start, True, "diagnostic_argmin"

    @staticmethod
    def evaluate_qubo_energy(q, sample):
        energy = 0.0
        for (var1, var2), coeff in q.items():
            energy += float(coeff) * sample.get(var1, 0) * sample.get(var2, 0)
        return float(energy)

    def decode_prc_solution(self, response, batch_indices, top_servers):
        if not self.capacity_aware_decoder:
            return super().decode_prc_solution(response, batch_indices, top_servers)
        return self.decode_prc_capacity_aware(response, batch_indices, top_servers)

    def decode_prc_capacity_aware(self, response, batch_indices, top_servers):
        n_batch = len(batch_indices)
        n_top = len(top_servers)
        assignment = np.zeros((n_batch, n_top), dtype=np.int8)
        sample = response.first.sample
        raw_counts = np.zeros(n_batch, dtype=int)
        selected_by_camera = [set() for _ in range(n_batch)]

        for i, _cam_idx in enumerate(batch_indices):
            for j, server_idx in enumerate(top_servers):
                var = self.var_name(i, server_idx)
                if sample.get(var, 0) == 1:
                    raw_counts[i] += 1
                    selected_by_camera[i].add(j)

        linear = self._last_linear
        if linear is None or linear.shape != (n_batch, n_top):
            linear = self.rebuild_linear_for_decoder(batch_indices, top_servers)

        local_loads = np.zeros(n_top, dtype=float)
        order = np.argsort(-(self.priority[batch_indices] * self.load_gflops[batch_indices]))
        capacity_rejected = 0
        alternative_assignments = 0
        zero_selection_rescues = 0
        capacity_rescues = 0

        for i in order:
            cam_idx = int(batch_indices[i])
            load = float(self.load_gflops[cam_idx])
            selected = selected_by_camera[i]
            ranked = self.decoder_candidate_order(i, load, linear, local_loads, top_servers, selected)

            chosen = -1
            selected_was_blocked = False
            for j in ranked:
                if linear[i, j] >= 99.0:
                    continue
                server_idx = int(top_servers[j])
                if local_loads[j] + load <= self.remaining_capacity[server_idx]:
                    chosen = int(j)
                    break
                if j in selected:
                    selected_was_blocked = True

            if chosen >= 0:
                assignment[i, chosen] = 1
                local_loads[chosen] += load
                if chosen not in selected:
                    alternative_assignments += 1
                    if raw_counts[i] == 0:
                        zero_selection_rescues += 1
                    elif selected_was_blocked:
                        capacity_rescues += 1
            elif raw_counts[i] > 0:
                capacity_rejected += 1

        self.decoder_alternative_assignments += int(alternative_assignments)
        self.decoder_zero_selection_rescues += int(zero_selection_rescues)
        self.decoder_capacity_rescues += int(capacity_rescues)

        return assignment, {
            "raw_selected_variables": int(np.sum(raw_counts)),
            "zero_selection_raw": int(np.sum(raw_counts == 0)),
            "multi_selection_raw": int(np.sum(raw_counts > 1)),
            "capacity_rejected_raw": int(capacity_rejected),
            "residual_blind_rejected_assignments": int(capacity_rejected),
            "decoder_alternative_assignments": int(alternative_assignments),
            "decoder_zero_selection_rescues": int(zero_selection_rescues),
            "decoder_capacity_rescues": int(capacity_rescues),
        }

    def rebuild_linear_for_decoder(self, batch_indices, top_servers):
        linear = np.full((len(batch_indices), len(top_servers)), 100.0, dtype=float)
        for i, cam_idx in enumerate(batch_indices):
            load = float(self.load_gflops[cam_idx])
            priority_weight = float(4 - self.priority[cam_idx])
            for j, server_idx in enumerate(top_servers):
                residual = float(self.remaining_capacity[server_idx])
                cost = float(self.cost_matrix[cam_idx, server_idx])
                if load <= residual:
                    coeff = -25.0 * (1.0 - cost) * priority_weight
                    if self.use_guard:
                        coeff += self.guard_weight * load / (residual + 1e-12)
                    linear[i, j] = coeff
        return linear

    def decoder_candidate_order(self, local_i, load, linear, local_loads, top_servers, selected):
        scores = []
        for j, server_idx in enumerate(top_servers):
            if linear[local_i, j] >= 99.0:
                continue
            residual = float(max(self.remaining_capacity[server_idx], 1e-9))
            projected_ratio = (local_loads[j] + load) / residual
            alternative_penalty = 0.0 if j in selected else self.decoder_alternative_penalty
            score = float(linear[local_i, j]) + self.decoder_capacity_weight * projected_ratio + alternative_penalty
            scores.append((score, j))
        scores.sort(key=lambda item: item[0])
        return [j for _score, j in scores]

    def post_process_batch(self, assignment, batch_indices, top_servers):
        return assignment

    def build_summary(self, total_time, quality):
        summary = super().build_summary(total_time, quality)
        summary["formulation"] = self.report_formulation
        summary["base_formulation"] = "PRC-QUBO"
        summary["capacity_conflict_enabled"] = bool(self.use_conflict)
        summary["capacity_conflict_weight"] = float(self.conflict_weight)
        summary["capacity_guard_enabled"] = bool(self.use_guard)
        summary["capacity_guard_weight"] = float(self.guard_weight if self.use_guard else 0.0)
        summary["capacity_conflict_mode"] = self.conflict_mode
        summary["capacity_conflict_margin"] = float(self.conflict_margin)
        summary["capacity_aware_decoder"] = bool(self.capacity_aware_decoder)
        summary["decoder_capacity_weight"] = float(self.decoder_capacity_weight)
        summary["decoder_alternative_penalty"] = float(self.decoder_alternative_penalty)
        summary["decoder_alternative_assignments"] = int(self.decoder_alternative_assignments)
        summary["decoder_zero_selection_rescues"] = int(self.decoder_zero_selection_rescues)
        summary["decoder_capacity_rescues"] = int(self.decoder_capacity_rescues)
        summary["avg_capacity_conflict_quadratic_count"] = float(
            np.mean(self.conflict_term_rows) if self.conflict_term_rows else 0.0
        )
        return summary

    def log_progress(self, *args, **kwargs):
        old_formulation = self.formulation
        self.formulation = self.report_formulation
        try:
            return super().log_progress(*args, **kwargs)
        finally:
            self.formulation = old_formulation


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stress-aware PRC-QUBO-C with intra-batch capacity-conflict couplings"
    )
    parser.add_argument("--solver", choices=["SQA", "SA", "DIAGNOSTIC"], default="SQA")
    parser.add_argument("--n-cameras", type=int, default=20000)
    parser.add_argument("--n-servers", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--max-servers-per-batch", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--capacity-scale", type=float, default=1.0)
    parser.add_argument("--num-reads", type=int, default=150)
    parser.add_argument("--num-sweeps", type=int, default=1000)
    parser.add_argument("--trotter", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--final-opt", action="store_true")
    parser.add_argument("--coverage-stop-threshold", type=float, default=0.995)
    parser.add_argument("--log-root", default=LOG_ROOT)
    parser.add_argument("--conflict-weight", type=float, default=180.0)
    parser.add_argument("--guard-weight", type=float, default=45.0)
    parser.add_argument("--no-guard", action="store_true")
    parser.add_argument("--no-conflict", action="store_true")
    parser.add_argument("--conflict-mode", choices=["all", "risky"], default="all")
    parser.add_argument("--conflict-margin", type=float, default=1.0)
    parser.add_argument("--plain-decoder", action="store_true")
    parser.add_argument("--decoder-capacity-weight", type=float, default=8.0)
    parser.add_argument("--decoder-alternative-penalty", type=float, default=0.75)
    parser.add_argument("--method-label", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    experiment = PRCQuboConflictStressExperiment(
        solver=args.solver,
        n_cameras=args.n_cameras,
        n_servers=args.n_servers,
        batch_size=args.batch_size,
        max_servers_per_batch=args.max_servers_per_batch,
        random_seed=args.seed,
        capacity_scale=args.capacity_scale,
        num_reads=args.num_reads,
        num_sweeps=args.num_sweeps,
        trotter=args.trotter,
        log_every=args.log_every,
        final_opt=args.final_opt,
        coverage_stop_threshold=args.coverage_stop_threshold,
        log_root=args.log_root,
        conflict_weight=args.conflict_weight,
        guard_weight=args.guard_weight,
        use_guard=not args.no_guard,
        use_conflict=not args.no_conflict,
        conflict_mode=args.conflict_mode,
        conflict_margin=args.conflict_margin,
        capacity_aware_decoder=not args.plain_decoder,
        decoder_capacity_weight=args.decoder_capacity_weight,
        decoder_alternative_penalty=args.decoder_alternative_penalty,
        method_label=args.method_label,
    )
    experiment.generate_realistic_data()
    experiment.run()


if __name__ == "__main__":
    main()
