"""Online-allocation baselines for the sequential batch protocol (same batches, candidate sets and evaluation
as qsa/v3_trajectory.py; the batch subproblem is always solved exactly with HiGHS).

  --policy lp         bid prices = exact LP duals of the residual problem, re-solved every --reprice batches
                      (re-solving / bid-price control; the Lagrangian of v3.1 approximates the same duals)
  --policy dmd        dual mirror descent (Balseiro, Lu & Mirrokni 2023): prices updated after every batch from
                      the deviation of consumption from the target rate K_j / T; needs no demand information
  --policy lookahead  rolling-horizon MILP: batch t is optimised jointly with the next --lookahead batches
                      (true future sensors), only batch t is committed; no prices
"""
import argparse, json, os, sys, time
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from batch_quality_benchmark import P_UNCOVERED, Instance, build_batch, solve_exact  # noqa: E402
from v3_trajectory import lagrange_prices  # noqa: E402


def lp_prices(inst, cams, cap, top=40, bias=0.0):
    """Exact capacity duals of the LP relaxation of the residual problem (each sensor: its `top` cheapest servers);
    bias > 0 scales the remaining demand by (1 + bias), as the demand margin of v3.1."""
    C = inst.wcost[cams]; l = inst.load[cams] * (1.0 + bias); n, m = len(cams), len(cap)
    cols = np.argpartition(C, top - 1, axis=1)[:, :top]
    c = np.concatenate([np.take_along_axis(C, cols, axis=1).ravel(), np.full(n, P_UNCOVERED)])
    rows = np.repeat(np.arange(n), top)
    nx = n * top
    A_eq = sp.hstack([sp.csr_matrix((np.ones(nx), (rows, np.arange(nx))), shape=(n, nx)), sp.identity(n, format="csr")]).tocsr()
    A_ub = sp.hstack([sp.csr_matrix((np.repeat(l, top), (cols.ravel(), np.arange(nx))), shape=(m, nx)), sp.csr_matrix((m, n))]).tocsr()
    res = linprog(c, A_ub=A_ub, b_ub=np.maximum(cap, 0.0), A_eq=A_eq, b_eq=np.ones(n), bounds=(0, None), method="highs")
    assert res.status == 0, res.message
    return np.maximum(-res.ineqlin.marginals, 0.0), res.fun - 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, choices=["lp", "dmd", "lookahead"])
    ap.add_argument("--target-util", type=float, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-cameras", type=int, default=20000)
    ap.add_argument("--n-servers", type=int, default=800)
    ap.add_argument("--layout", default="uniform", choices=["uniform", "clustered"])
    ap.add_argument("--batch-size", type=int, default=80)
    ap.add_argument("--k-percam", type=int, default=5)
    ap.add_argument("--k-slack", type=int, default=2)
    ap.add_argument("--reprice", type=int, default=5)
    ap.add_argument("--eta", type=float, default=1.0, help="dmd: step size, in units of the mean initial Lagrangian price")
    ap.add_argument("--dmd-init", default="zero", choices=["zero", "lagrange"])
    ap.add_argument("--dmd-rate", default="uniform", choices=["uniform", "load"],
                    help="target consumption per batch: K_j/T (uniform) or residual capacity times the batch's share of the remaining load")
    ap.add_argument("--lookahead", type=int, default=2)
    ap.add_argument("--weight", default="inv", choices=["inv", "prio"], help="priority weight: inv w_i = 4 - p_i (default), prio w_i = p_i")
    ap.add_argument("--forecast-bias", type=float, default=0.0, help="lp: demand margin applied to the remaining loads")
    ap.add_argument("--out", default="results/v3/online_baselines")
    a = ap.parse_args()

    inst = Instance(a.n_cameras, a.n_servers, a.seed, 0.25, a.target_util, layout=a.layout)
    if a.weight == "prio":
        inst.w = inst.priority.astype(float); inst.wcost = inst.cost * inst.w[:, None]
    B = a.batch_size; nb = int(np.ceil(a.n_cameras / B))
    order = np.argsort(-(inst.priority * inst.load))
    residual = inst.cap.copy()
    u = None; extra = {}
    t0 = time.time(); t_prices = 0.0
    if a.policy in ("lp", "dmd"):
        tp = time.time()
        u0, lb = lagrange_prices(inst)
        scale = float(np.mean(u0[u0 > 0])) if np.any(u0 > 0) else 1e-3
        if a.policy == "lp":
            u, _ = lp_prices(inst, order, residual, bias=a.forecast_bias)
        else:
            u = u0.copy() if a.dmd_init == "lagrange" else np.zeros(a.n_servers)
            rho = inst.cap / nb                       # target consumption per batch
            step = a.eta * scale / np.mean(rho)       # dimensionless eta -> price units per GFLOP of deviation
            extra = {"dmd_step": step, "price_scale_ref": scale}
        t_prices += time.time() - tp
    total, covered = 0.0, 0
    for t in range(nb):
        cams = order[t * B:(t + 1) * B]
        if a.policy == "lp" and t > 0 and t % a.reprice == 0:
            tp = time.time(); u, _ = lp_prices(inst, order[t * B:], residual, bias=a.forecast_bias); t_prices += time.time() - tp
        if a.policy == "lookahead":
            ucams = order[t * B:(t + 1 + a.lookahead) * B]
            b = build_batch(inst, ucams, residual.copy(), "percam", 20, a.k_percam, inst.gen, k_slack=a.k_slack)
            x, _ = solve_exact(b)
            x = x & (b.pi < len(cams))                # commit only the current batch
            obj = float(b.pcost[x].sum() + P_UNCOVERED * (len(cams) - np.count_nonzero(x)))
        else:
            sel = lambda c: inst.wcost[c] + u * inst.load[c]
            b = build_batch(inst, cams, residual.copy(), "percam", 20, a.k_percam, inst.gen, sel_cost=sel, k_slack=a.k_slack)
            add = u[b.servers[b.ps]] * b.l[b.pi]
            b.val = b.val + add; b.dec = b.pcost + add
            x, _ = solve_exact(b)
            obj = b.objective(x)
        assert b.feasible(x)
        cons = np.bincount(b.ps[x], weights=b.l[b.pi[x]], minlength=b.m)
        residual[b.servers] -= cons
        total += obj; covered += int(np.count_nonzero(x))
        if a.policy == "dmd":
            g = np.zeros(a.n_servers); g[b.servers] = cons
            if a.dmd_rate == "load":
                rem = inst.load[order[t * B:]].sum()
                rho = (residual + g) * inst.load[cams].sum() / max(rem, 1e-9)
            u = np.maximum(0.0, u + step * (g - rho))
        if t % 50 == 0:
            print(f"[{a.policy}] batch {t} objective so far {total:.1f} elapsed {time.time()-t0:.0f}s", flush=True)
    assert np.all(residual >= -1e-6)
    res = {"method": "exact", "policy": a.policy, "target_util": a.target_util, "seed": a.seed, "layout": a.layout,
           "n_cameras": a.n_cameras, "n_servers": a.n_servers, "batch_size": B, "reprice": a.reprice,
           "eta": a.eta if a.policy == "dmd" else None, "dmd_init": a.dmd_init if a.policy == "dmd" else None, "dmd_rate": a.dmd_rate if a.policy == "dmd" else None,
           "lookahead": a.lookahead if a.policy == "lookahead" else None, "forecast_bias": a.forecast_bias, "weight": a.weight,
           "objective": total, "covered": covered, "time_sec": time.time() - t0, "price_time_sec": t_prices, **extra,
           "params": vars(a)}
    os.makedirs(a.out, exist_ok=True)
    tag = {"lp": f"lp_rp{a.reprice}" + (f"_b{a.forecast_bias:+g}" if a.forecast_bias else ""), "dmd": f"dmd_eta{a.eta:g}_{a.dmd_init}_{a.dmd_rate}", "lookahead": f"la{a.lookahead}"}[a.policy]
    path = os.path.join(a.out, f"ob_{tag}{'_wprio' if a.weight == 'prio' else ''}{'_clu' if a.layout == 'clustered' else ''}_{a.n_cameras}x{a.n_servers}_u{a.target_util:g}_seed{a.seed}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    json.dump(res, open(path, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "params"}))


if __name__ == "__main__":
    main()
