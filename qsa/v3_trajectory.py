"""End-to-end sequential run: every method builds its OWN trajectory over all batches.

Methods: exact (batch MILP), greedy, regret, sqa_v3, sa_v3 (QUBO-v3: reduction + domain wall +
adaptive capacity prices). Output: total objective (same formula as calculate_quality), coverage,
time, and QUBO statistics.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from batch_quality_benchmark import Instance, build_batch, greedy_fill, solve_exact, solve_greedy, solve_regret  # noqa: E402
from qubo_v3_lab import build_v3, decode_v3, raw_choice, run  # noqa: E402


def solve_v3(b, solver, a, seed):
    q, info, fixed, dinfo = build_v3(b, True, True, lam_onehot=a.lam_onehot, lam_cap=a.lam_cap)
    stats = {"vars": info["vars"], "fixed": info["fixed_cams"], "qubo_calls": 0, "viol": 0}
    if not q:
        return greedy_fill(b, fixed.copy()), stats
    srv_w, srv_mu = np.ones(b.m), np.zeros(b.m)
    best_x, best_o = None, np.inf
    for rnd in range(a.adaptive_rounds):
        if rnd > 0:
            q, _, fixed, dinfo = build_v3(b, True, True, lam_onehot=a.lam_onehot, lam_cap=a.lam_cap,
                                          srv_w=srv_w, srv_mu=srv_mu)
        samples = run(q, solver, a.reads, a.sweeps, a.trotter, seed=seed + 31 * rnd,
                      gamma=a.gamma if solver == "SQA" else None, beta=a.beta if solver == "SQA" else None)
        stats["qubo_calls"] += 1
        rover = None
        for smp in samples:
            x, v = decode_v3(b, smp, fixed, dinfo, True)
            o = b.objective(x)
            if o < best_o:
                best_x, best_o, stats["viol"] = x, o, v
                raw = raw_choice(b, smp, fixed, dinfo, True)
                rover = np.bincount(b.ps[raw], weights=b.l[b.pi[raw]], minlength=b.m) / np.maximum(b.R, 1e-9) - 1.0
        if rover is None or np.all(rover <= 1e-9):
            break
        vs = rover > 1e-9
        srv_mu[vs] += a.mu_step * (1.0 + rover[vs])
        srv_w[vs] *= a.w_growth
    return best_x, stats


def lagrange_prices(inst, iters=300):
    """Capacity prices u_j >= 0 from the Lagrangian relaxation of the full problem."""
    C, l, K = inst.wcost, inst.load, inst.cap
    u = np.zeros(len(K)); best, best_u = -np.inf, u.copy()
    for it in range(iters):
        R = C + np.outer(l, u)
        j = R.argmin(1); r = R[np.arange(len(l)), j]; take = r < 15.0
        val = np.where(take, r, 15.0).sum() - u @ K
        if val > best:
            best, best_u = val, u.copy()
        g = np.bincount(j[take], weights=l[take], minlength=len(K)) - K
        u = np.maximum(0, u + g / (np.linalg.norm(g) + 1e-9) * np.sqrt(len(K)) * 0.05 / (1 + it / 50) / np.mean(l) * 0.5)
    return best_u, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["exact", "greedy", "regret", "sqa_v3", "sa_v3"])
    ap.add_argument("--capacity-scale", type=float, default=0.25)
    ap.add_argument("--window", default="percam")
    ap.add_argument("--k-percam", type=int, default=5)
    ap.add_argument("--m-shared", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reads", type=int, default=10)
    ap.add_argument("--sweeps", type=int, default=2000)
    ap.add_argument("--trotter", type=int, default=8)
    ap.add_argument("--beta", type=float, default=20.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--lam-onehot", type=float, default=0.45)
    ap.add_argument("--lam-cap", type=float, default=1.0)
    ap.add_argument("--adaptive-rounds", type=int, default=3)
    ap.add_argument("--mu-step", type=float, default=0.5)
    ap.add_argument("--w-growth", type=float, default=2.0)
    ap.add_argument("--price", action="store_true", help="add Lagrangian capacity prices u_j*l_i to decision costs")
    ap.add_argument("--price-scale", type=float, default=1.0)
    ap.add_argument("--price-window", action="store_true", help="also choose per-camera candidates by priced cost")
    ap.add_argument("--out", default="logs_qubo_v3_trajectory")
    a = ap.parse_args()

    inst = Instance(20000, 800, a.seed, a.capacity_scale)
    order = np.argsort(-(inst.priority * inst.load))
    residual = inst.cap.copy()
    u = None
    if a.price:
        u, lb = lagrange_prices(inst)
        u = u * a.price_scale
        print(f"prices: lower bound {lb:.1f}, priced servers {int(np.sum(u > 0))}", flush=True)
    total, covered, t_all = 0.0, 0, time.time()
    agg = {"qubo_batches": 0, "qubo_calls": 0, "vars_sum": 0, "fixed_sum": 0, "viol_sum": 0}
    for t in range(250):
        cams = order[t * 80:(t + 1) * 80]
        sel = (lambda c: inst.wcost[c] + u * inst.load[c]) if (u is not None and a.price_window) else None
        b = build_batch(inst, cams, residual.copy(), a.window, a.m_shared, a.k_percam, inst.gen, sel_cost=sel)
        if u is not None:
            add = u[b.servers[b.ps]] * b.l[b.pi]
            b.val = b.val + add
            b.dec = b.pcost + add
        if a.method == "exact":
            x, _ = solve_exact(b)
        elif a.method == "greedy":
            x, _ = solve_greedy(b)
        elif a.method == "regret":
            x, _ = solve_regret(b)
        else:
            x, st = solve_v3(b, "SQA" if a.method == "sqa_v3" else "SA", a, seed=a.seed * 1000 + t)
            if st["qubo_calls"]:
                agg["qubo_batches"] += 1
            agg["qubo_calls"] += st["qubo_calls"]; agg["vars_sum"] += st["vars"]
            agg["fixed_sum"] += st["fixed"]; agg["viol_sum"] += st["viol"]
        assert b.feasible(x)
        total += b.objective(x)
        covered += int(np.count_nonzero(x))
        residual[b.servers] -= np.bincount(b.ps[x], weights=b.l[b.pi[x]], minlength=b.m)
        if t % 50 == 0:
            print(f"[{a.method}] batch {t} objective so far {total:.1f} elapsed {time.time()-t_all:.0f}s", flush=True)
    assert np.all(residual >= -1e-6)
    res = {"method": a.method, "price": bool(a.price), "price_scale": a.price_scale, "price_window": bool(a.price_window), "k_percam": a.k_percam, "window": a.window, "capacity_scale": a.capacity_scale, "seed": a.seed,
           "utilization_percent": inst.utilization, "objective": total, "covered": covered,
           "coverage_percent": covered / 200.0, "time_sec": time.time() - t_all, **agg, "params": vars(a)}
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"traj_{a.method}{'_priced' if a.price else ''}_{a.window}_s{a.capacity_scale:g}_seed{a.seed}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    json.dump(res, open(path, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "params"}))


if __name__ == "__main__":
    main()
