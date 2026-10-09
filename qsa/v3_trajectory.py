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
from batch_quality_benchmark import (Instance, build_batch, greedy_fill, local_search, solve_exact,  # noqa: E402
                                     solve_greedy, solve_regret)
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
            if a.polish:
                x = local_search(b, x)
            o = float(b.val[x].sum())
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


def decision_obj(b, x):
    """Batch decision objective: the (possibly price-adjusted) values every solver optimises."""
    return float(b.val[x].sum())


def solve_ils(b, time_budget, rng):
    """Iterated local search on one batch: regret start, then (destroy 20% of the batch,
    greedy repair, shift/swap local search) until the time budget is spent. Classical
    equal-time counterpart of SQA-v3. Optimises the same price-adjusted values as all other solvers."""
    t0 = time.time()
    x, _ = solve_regret(b)
    x = local_search(b, x)
    best, best_o = x.copy(), decision_obj(b, x)
    cur, cur_o = x.copy(), best_o
    while time.time() - t0 < time_budget:
        y = cur.copy()
        assigned = np.where(y)[0]
        if len(assigned) == 0:
            break
        k = max(1, int(0.2 * len(assigned)))
        y[rng.choice(assigned, size=k, replace=False)] = False
        order = rng.permutation(b.n)
        y = local_search(b, greedy_fill(b, y, order=order))
        o = decision_obj(b, y)
        if o <= cur_o + 1e-9 or rng.random() < 0.05:
            cur, cur_o = y, o
        if o < best_o - 1e-9:
            best, best_o = y.copy(), o
    return best


def lagrange_prices(inst, iters=300, cams=None, cap=None, top=0):
    """Capacity prices u_j >= 0 from the Lagrangian relaxation of the full problem, or (QUBO-v3.1)
    of the residual problem: the cameras not yet assigned and the residual capacities.
    top > 0: each camera only considers its `top` cheapest servers (sparse, ~20x faster; used for re-pricing)."""
    C, l, K = inst.wcost, inst.load, inst.cap
    if cams is not None:
        C, l, K = C[cams], l[cams], np.maximum(cap, 0.0)
    cols = None
    if top and top < C.shape[1]:
        if not hasattr(inst, "_topcols") or inst._topcols.shape[1] != top:
            inst._topcols = np.argpartition(inst.wcost, top - 1, axis=1)[:, :top]
        cols = inst._topcols if cams is None else inst._topcols[cams]
        C = np.take_along_axis(C, cols, axis=1)
    return _subgradient(C, l, K, cols, iters)


def _subgradient(C, l, K, cols, iters):
    u = np.zeros(len(K)); best, best_u = -np.inf, u.copy()
    rows = np.arange(len(l))
    for it in range(iters):
        R = C + l[:, None] * (u if cols is None else u[cols])
        jj = R.argmin(1); r = R[rows, jj]; take = r < 15.0
        j = jj if cols is None else cols[rows, jj]
        val = np.where(take, r, 15.0).sum() - u @ K
        if val > best:
            best, best_u = val, u.copy()
        g = np.bincount(j[take], weights=l[take], minlength=len(K)) - K
        u = np.maximum(0, u + g / (np.linalg.norm(g) + 1e-9) * np.sqrt(len(K)) * 0.05 / (1 + it / 50) / np.mean(l) * 0.5)
    return best_u, best


class Forecast:
    """Inexact information for re-pricing (forecast experiment).
    mode 'oracle': true remaining cameras (default QUBO-v3.1).
    mode 'twin'  : remaining cameras replaced by an independent draw from the same distribution
                   (seed + 10007), costed against the real servers with the real normalisation.
    mode 'noise' : true remaining cameras, loads multiplied by a fixed per-camera factor 1+e,
                   e ~ U(-level, level).
    bias        : all forecast loads multiplied by (1 + bias) (systematic demand error)."""
    def __init__(self, inst, mode="oracle", level=0.0, bias=0.0, seed=0, top=40):
        self.inst, self.mode, self.level, self.bias, self.top = inst, mode, level, bias, top
        g = inst.gen
        n = len(inst.load)
        if mode == "twin":
            rs = np.random.RandomState(seed + 10007)
            pr = rs.choice([3, 2, 1], size=n, p=[0.15, 0.25, 0.6])
            ld = np.zeros(n)
            for p_, lo, hi in ((3, 8, 15), (2, 4, 8), (1, 1, 3)):
                m = pr == p_; ld[m] = rs.uniform(lo, hi, m.sum())
            x = rs.uniform(0, 1000, n); y = rs.uniform(0, 1000, n)
            d_real = np.hypot(g.camera_x[:, None] - g.server_x[None, :], g.camera_y[:, None] - g.server_y[None, :])
            d = np.hypot(x[:, None] - g.server_x[None, :], y[:, None] - g.server_y[None, :])
            cinv = 1.0 / (g.initial_capacity + 1e-9)
            def raw(dd, prr, ll):
                return (0.40 * dd / (d_real.max() + 1e-12) + 0.35 * (ll / (g.load_gflops.max() + 1e-12))[:, None]
                        + 0.20 * ((3 - prr) / 2.0)[:, None] + 0.05 * (cinv / (cinv.max() + 1e-12))[None, :])
            r_real = raw(d_real, g.priority, g.load_gflops)
            lo_, hi_ = r_real.min(), r_real.max()
            cost = np.clip((raw(d, pr, ld) - lo_) / (hi_ - lo_ + 1e-9), 0, 1)
            self.wcost = cost * (pr if np.array_equal(inst.w, inst.priority.astype(float)) else 4 - pr)[:, None]
            self.load = ld
            self.order = np.argsort(-(pr * ld))
        else:
            self.wcost = inst.wcost
            f = np.ones(n)
            if mode == "noise" and level > 0:
                f = 1.0 + np.random.default_rng(seed + 20011).uniform(-level, level, n)
            self.load = inst.load * f
            self.order = None
        self.load = self.load * (1.0 + bias)
        self.cols = np.argpartition(self.wcost, top - 1, axis=1)[:, :top] if top else None

    def prices(self, true_remaining, t_start, residual, iters):
        cams = true_remaining if self.order is None else self.order[t_start:]
        C = self.wcost[cams]; l = self.load[cams]
        cols = None
        if self.cols is not None:
            cols = self.cols[cams]; C = np.take_along_axis(C, cols, axis=1)
        return _subgradient(C, l, np.maximum(residual, 0.0), cols, iters)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["exact", "greedy", "regret", "ils", "sqa_v3", "sa_v3"])
    ap.add_argument("--capacity-scale", type=float, default=0.25)
    ap.add_argument("--target-util", type=float, default=None, help="percent; overrides --capacity-scale")
    ap.add_argument("--n-cameras", type=int, default=20000)
    ap.add_argument("--n-servers", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=80)
    ap.add_argument("--ils-time", type=float, default=0.8, help="seconds per batch for the ILS method")
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
    ap.add_argument("--reprice", type=int, default=0,
                    help="QUBO-v3.1: recompute Lagrangian prices on the remaining cameras and residual capacities every T batches (0 = static prices)")
    ap.add_argument("--reprice-iters", type=int, default=300)
    ap.add_argument("--reprice-top", type=int, default=40, help="cheapest servers per camera considered when re-pricing (0 = all)")
    ap.add_argument("--k-slack", type=int, default=0, help="QUBO-v3.1: extra candidates per camera from the servers with most residual capacity")
    ap.add_argument("--trace", action="store_true", help="store per-batch objective, QUBO size and assignment")
    ap.add_argument("--polish", action="store_true", help="shift/swap local search on every decoded QUBO sample")
    ap.add_argument("--out", default="logs_qubo_v3_trajectory")
    ap.add_argument("--weight", default="inv", choices=["inv", "prio"],
                    help="priority weight in the objective: inv w_i = 4 - p_i (default, as in all campaigns), prio w_i = p_i")
    ap.add_argument("--forecast", default="oracle", choices=["oracle", "twin", "noise"],
                    help="information used for re-pricing (forecast experiment)")
    ap.add_argument("--forecast-level", type=float, default=0.0, help="noise mode: per-camera load error U(-level, level)")
    ap.add_argument("--forecast-bias", type=float, default=0.0, help="systematic relative error of forecast loads")
    a = ap.parse_args()

    inst = Instance(a.n_cameras, a.n_servers, a.seed, a.capacity_scale, a.target_util)
    if a.weight == "prio":
        inst.w = inst.priority.astype(float)
        inst.wcost = inst.cost * inst.w[:, None]
    nb = int(np.ceil(a.n_cameras / a.batch_size))
    rng = np.random.default_rng(a.seed)
    order = np.argsort(-(inst.priority * inst.load))
    residual = inst.cap.copy()
    u = None
    if a.price:
        u, lb = lagrange_prices(inst)
        u = u * a.price_scale
        print(f"prices: lower bound {lb:.1f}, priced servers {int(np.sum(u > 0))}", flush=True)
    fc = None
    if a.forecast != "oracle" or a.forecast_bias != 0.0:
        fc = Forecast(inst, a.forecast, a.forecast_level, a.forecast_bias, a.seed, a.reprice_top)
    total, covered, t_all = 0.0, 0, time.time()
    trace = []
    agg = {"qubo_batches": 0, "qubo_calls": 0, "vars_sum": 0, "fixed_sum": 0, "viol_sum": 0}
    for t in range(nb):
        cams = order[t * a.batch_size:(t + 1) * a.batch_size]
        if a.price and a.reprice > 0 and t > 0 and t % a.reprice == 0:
            if fc is None:
                u, _ = lagrange_prices(inst, iters=a.reprice_iters, cams=order[t * a.batch_size:], cap=residual, top=a.reprice_top)
            else:
                u, _ = fc.prices(order[t * a.batch_size:], t * a.batch_size, residual, a.reprice_iters)
            u = u * a.price_scale
        sel = (lambda c: inst.wcost[c] + u * inst.load[c]) if (u is not None and a.price_window) else None
        b = build_batch(inst, cams, residual.copy(), a.window, a.m_shared, a.k_percam, inst.gen, sel_cost=sel, k_slack=a.k_slack)
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
        elif a.method == "ils":
            x = solve_ils(b, a.ils_time, rng)
        else:
            x, st = solve_v3(b, "SQA" if a.method == "sqa_v3" else "SA", a, seed=a.seed * 1000 + t)
            if st["qubo_calls"]:
                agg["qubo_batches"] += 1
            agg["qubo_calls"] += st["qubo_calls"]; agg["vars_sum"] += st["vars"]
            agg["fixed_sum"] += st["fixed"]; agg["viol_sum"] += st["viol"]
        assert b.feasible(x)
        total += b.objective(x)
        covered += int(np.count_nonzero(x))
        if a.trace:
            assigned = np.full(len(cams), -1)
            assigned[b.pi[x]] = b.servers[b.ps[x]]
            trace.append({"batch": t, "objective": b.objective(x), "covered": int(np.count_nonzero(x)),
                          "qubo_vars": int(st["vars"]) if a.method in ("sqa_v3", "sa_v3") else None,
                          "fixed": int(st["fixed"]) if a.method in ("sqa_v3", "sa_v3") else None,
                          "cams": [int(c) for c in cams], "servers": [int(v) for v in assigned]})
        residual[b.servers] -= np.bincount(b.ps[x], weights=b.l[b.pi[x]], minlength=b.m)
        if t % 50 == 0:
            print(f"[{a.method}] batch {t} objective so far {total:.1f} elapsed {time.time()-t_all:.0f}s", flush=True)
    assert np.all(residual >= -1e-6)
    res = {"method": a.method, "price": bool(a.price), "reprice": a.reprice, "reprice_top": a.reprice_top, "k_slack": a.k_slack, "polish": a.polish, "weight": a.weight, "forecast": a.forecast, "forecast_level": a.forecast_level, "forecast_bias": a.forecast_bias, "price_scale": a.price_scale, "price_window": bool(a.price_window), "k_percam": a.k_percam, "window": a.window, "capacity_scale": inst.capacity_scale, "target_util": a.target_util, "n_cameras": a.n_cameras, "n_servers": a.n_servers, "seed": a.seed,
           "utilization_percent": inst.utilization, "objective": total, "covered": covered,
           "coverage_percent": covered / a.n_cameras * 100.0, "time_sec": time.time() - t_all, **agg, "params": vars(a), **({"trace": trace} if a.trace else {})}
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"traj_{a.method}{'_priced' if a.price else ''}{('_rp%d' % a.reprice) if a.reprice else ''}{'_ls' if a.polish else ''}{'_wprio' if a.weight == 'prio' else ''}{('_ks%d' % a.k_slack) if a.k_slack else ''}{('_fc-%s%g%+g' % (a.forecast, a.forecast_level, a.forecast_bias)) if (a.forecast != 'oracle' or a.forecast_bias) else ''}_{a.window}_{a.n_cameras}x{a.n_servers}_{('u%g' % a.target_util) if a.target_util else ('s%g' % a.capacity_scale)}_seed{a.seed}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    json.dump(res, open(path, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("params", "trace")}))


if __name__ == "__main__":
    main()
