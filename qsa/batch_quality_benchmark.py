"""Batch-level quality benchmark: exact MILP vs classical heuristics vs QUBO (SQA / SA).

Every method solves the *same* batch subproblems. The trajectory (residual capacities
between batches) is driven by a single reference method (default: the exact batch
MILP), so differences between methods are measured on identical inputs.

Batch subproblem (identical objective for all methods):
    min  sum_{(i,j)} w_i c_ij x_ij + P * #uncovered(i)
    s.t. sum_j x_ij <= 1                       (each camera at most one server)
         sum_i l_i x_ij <= R_j^(t)             (residual capacity)
where w_i = 4 - priority_i and P = 15, exactly as in calculate_quality().

Candidate windows:
    shared  : the original 80 x M window (select_prc_servers, M servers shared by the batch)
    percam  : every camera gets its own K cheapest residual-feasible servers

QUBO-v2 (fixes found in the audit):
    * linear term = scaled true objective (w_i c_ij - P), no ad-hoc reward;
    * one-hot "at most one" penalty A * x_p x_q with A > max|linear| so the ground state is one-hot;
    * capacity via unbalanced penalisation (Montanez-Barrera et al., 2022), no slack variables:
          -lam1 * h_j + lam2 * h_j^2,   h_j = 1 - sum_i (l_i / R_j) x_ij
    * all sampler parameters passed explicitly and logged; seeded.
Decoding: best read after the same deterministic repair (drop duplicates / overload, greedy fill).
"""

import argparse
import json
import os
import sys
import tempfile
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from capacity_stress_experiment import CapacityStressExperiment  # noqa: E402

P_UNCOVERED = 15.0


# --------------------------------------------------------------------------- instance

class Instance:
    def __init__(self, n_cameras, n_servers, seed, capacity_scale):
        gen = CapacityStressExperiment(
            "PRC-QUBO", "SQA", n_cameras=n_cameras, n_servers=n_servers, random_seed=seed,
            capacity_scale=capacity_scale, log_root=tempfile.mkdtemp(prefix="bqb_"),
        )
        gen.generate_realistic_data()
        self.gen = gen
        self.w = (4 - gen.priority).astype(float)
        self.cost = gen.cost_matrix
        self.wcost = gen.cost_matrix * self.w[:, None]
        self.load = gen.load_gflops.astype(float)
        self.cap = gen.initial_capacity.astype(float)
        self.priority = gen.priority
        self.utilization = gen.utilization_percent


class Batch:
    """Candidate pairs of one batch. Local camera index i, local server index s."""

    def __init__(self, cams, servers, pi, ps, residual, inst):
        self.cams = np.asarray(cams, int)
        self.servers = np.asarray(servers, int)
        self.pi = np.asarray(pi, int)
        self.ps = np.asarray(ps, int)
        self.R = np.asarray(residual, float)
        self.l = inst.load[self.cams]
        self.val = inst.wcost[self.cams[self.pi], self.servers[self.ps]] - P_UNCOVERED  # negative
        self.pcost = inst.wcost[self.cams[self.pi], self.servers[self.ps]]
        self.prio = inst.priority[self.cams]
        self.n = len(self.cams)
        self.m = len(self.servers)
        self.npairs = len(self.pi)
        self.pairs_of_cam = [np.where(self.pi == i)[0] for i in range(self.n)]
        self.pairs_of_srv = [np.where(self.ps == s)[0] for s in range(self.m)]

    def objective(self, x):
        """x: bool array over pairs; must be feasible."""
        return float(self.pcost[x].sum() + P_UNCOVERED * (self.n - np.count_nonzero(x)))

    def feasible(self, x):
        per_cam = np.bincount(self.pi[x], minlength=self.n)
        per_srv = np.bincount(self.ps[x], weights=self.l[self.pi[x]], minlength=self.m)
        return bool(per_cam.max(initial=0) <= 1 and np.all(per_srv <= self.R + 1e-9))


def build_batch(inst, cams, residual, window, m_shared, k_percam, gen, sel_cost=None):
    if window == "shared":
        gen.remaining_capacity = residual
        top, _ = gen.select_prc_servers(cams)
        servers = np.asarray(top, int)
        pi, ps = [], []
        for i, c in enumerate(cams):
            ok = inst.load[c] <= residual[servers]
            for s in np.where(ok)[0]:
                pi.append(i)
                ps.append(s)
        return Batch(cams, servers, pi, ps, residual[servers], inst)
    # per-camera window
    chosen = {}
    per_cam = []
    for c in cams:
        feas = np.where(residual >= inst.load[c])[0]
        if len(feas) == 0:
            per_cam.append([])
            continue
        k = min(k_percam, len(feas))
        crow = (inst.wcost[c] if sel_cost is None else sel_cost(c))
        best = feas[np.argpartition(crow[feas], k - 1)[:k]]
        per_cam.append(best)
        for s in best:
            chosen.setdefault(int(s), len(chosen))
    servers = np.array(sorted(chosen, key=chosen.get), int)
    pi, ps = [], []
    for i, best in enumerate(per_cam):
        for s in best:
            pi.append(i)
            ps.append(chosen[int(s)])
    return Batch(cams, servers, pi, ps, residual[servers], inst)


# --------------------------------------------------------------------------- classical solvers

def solve_exact(b, time_limit=60.0):
    npair = b.npairs
    rows, cols, vals = [], [], []
    for i in range(b.n):
        for p in b.pairs_of_cam[i]:
            rows.append(i); cols.append(p); vals.append(1.0)
    for s in range(b.m):
        for p in b.pairs_of_srv[s]:
            rows.append(b.n + s); cols.append(p); vals.append(b.l[b.pi[p]])
    A = coo_matrix((vals, (rows, cols)), shape=(b.n + b.m, npair)).tocsr()
    ub = np.concatenate([np.ones(b.n), b.R])
    res = milp(c=b.val, constraints=[LinearConstraint(A, -np.inf, ub)], integrality=np.ones(npair),
               bounds=Bounds(0, 1), options={"time_limit": time_limit, "mip_rel_gap": 1e-9})
    x = np.zeros(npair, bool) if res.x is None else res.x > 0.5
    return x, {"status": int(res.status), "mip_gap": float(getattr(res, "mip_gap", np.nan) or 0.0)}


def greedy_fill(b, x, order=None):
    """Insert unassigned cameras (in the given order) into their cheapest feasible pair."""
    used = np.bincount(b.ps[x], weights=b.l[b.pi[x]], minlength=b.m).astype(float)
    assigned = np.zeros(b.n, bool)
    assigned[b.pi[x]] = True
    if order is None:
        order = np.argsort(-(b.prio * b.l))
    for i in order:
        if assigned[i]:
            continue
        best, bestv = -1, np.inf
        for p in b.pairs_of_cam[i]:
            s = b.ps[p]
            if used[s] + b.l[i] <= b.R[s] + 1e-9 and b.val[p] < bestv:
                best, bestv = p, b.val[p]
        if best >= 0 and bestv < 0:
            x[best] = True
            used[b.ps[best]] += b.l[i]
            assigned[i] = True
    return x


def solve_greedy(b):
    return greedy_fill(b, np.zeros(b.npairs, bool)), {}


def solve_regret(b):
    x = np.zeros(b.npairs, bool)
    used = np.zeros(b.m)
    free = set(range(b.n))
    while free:
        best_i, best_p, best_reg = -1, -1, -np.inf
        for i in free:
            opts = [(b.val[p], p) for p in b.pairs_of_cam[i] if used[b.ps[p]] + b.l[i] <= b.R[b.ps[p]] + 1e-9]
            if not opts:
                continue
            opts.sort()
            reg = (opts[1][0] if len(opts) > 1 else 0.0) - opts[0][0]
            if reg > best_reg:
                best_i, best_p, best_reg = i, opts[0][1], reg
        if best_i < 0:
            break
        x[best_p] = True
        used[b.ps[best_p]] += b.l[best_i]
        free.discard(best_i)
    return x, {}


def local_search(b, x, max_rounds=50):
    """Best-improvement shift + swap with capacity feasibility."""
    x = x.copy()
    used = np.bincount(b.ps[x], weights=b.l[b.pi[x]], minlength=b.m).astype(float)
    cur = -np.ones(b.n, int)
    cur[b.pi[x]] = np.where(x)[0]
    for _ in range(max_rounds):
        improved = False
        for i in range(b.n):  # shift (incl. insert of uncovered)
            p0 = cur[i]
            v0 = b.val[p0] if p0 >= 0 else 0.0
            for p in b.pairs_of_cam[i]:
                if p == p0:
                    continue
                s = b.ps[p]
                extra = b.l[i] if (p0 < 0 or b.ps[p0] != s) else 0.0
                if b.val[p] < v0 - 1e-12 and used[s] + extra <= b.R[s] + 1e-9:
                    if p0 >= 0:
                        x[p0] = False; used[b.ps[p0]] -= b.l[i]
                    x[p] = True; used[s] += b.l[i]; cur[i] = p; v0 = b.val[p]; p0 = p
                    improved = True
        for i in range(b.n):  # swap servers of two assigned cameras
            pi_ = cur[i]
            if pi_ < 0:
                continue
            si = b.ps[pi_]
            for k in range(i + 1, b.n):
                pk = cur[k]
                if pk < 0 or b.ps[pk] == si:
                    continue
                sk = b.ps[pk]
                qi = [p for p in b.pairs_of_cam[i] if b.ps[p] == sk]
                qk = [p for p in b.pairs_of_cam[k] if b.ps[p] == si]
                if not qi or not qk:
                    continue
                qi, qk = qi[0], qk[0]
                delta = b.val[qi] + b.val[qk] - b.val[pi_] - b.val[pk]
                if delta < -1e-12 and used[sk] - b.l[k] + b.l[i] <= b.R[sk] + 1e-9 and \
                        used[si] - b.l[i] + b.l[k] <= b.R[si] + 1e-9:
                    x[[pi_, pk]] = False; x[[qi, qk]] = True
                    used[sk] += b.l[i] - b.l[k]; used[si] += b.l[k] - b.l[i]
                    cur[i], cur[k] = qi, qk
                    pi_, si = qi, sk
                    improved = True
        if not improved:
            break
    return x


def solve_greedy_ls(b):
    x, _ = solve_regret(b)
    return local_search(b, x), {}


# --------------------------------------------------------------------------- QUBO-v2

def build_qubo_v2(b, lam1, lam2, onehot_factor=2.0):
    """Returns (linear dict, quadratic dict, scale). Variables are pair indices."""
    scale = 1.0 / P_UNCOVERED
    lin = b.val * scale                                  # in [-1, ~-0.8]
    A = onehot_factor * float(np.max(np.abs(lin)))       # > max |linear|  -> ground state one-hot
    Q = {}
    L = {int(p): float(lin[p]) for p in range(b.npairs)}
    for i in range(b.n):
        ps = b.pairs_of_cam[i]
        for a in range(len(ps)):
            for c in range(a + 1, len(ps)):
                Q[(int(ps[a]), int(ps[c]))] = A
    for s in range(b.m):
        ps = b.pairs_of_srv[s]
        if len(ps) == 0:
            continue
        a = b.l[b.pi[ps]] / max(b.R[s], 1e-9)            # a_i = l_i / R_j
        if a.sum() <= 1.0 + 1e-12:                         # server can never overflow -> no term
            continue
        for u, p in enumerate(ps):
            L[int(p)] += lam1 * a[u] - 2.0 * lam2 * a[u] + lam2 * a[u] * a[u]
        for u in range(len(ps)):
            for v in range(u + 1, len(ps)):
                key = (int(ps[u]), int(ps[v]))
                Q[key] = Q.get(key, 0.0) + 2.0 * lam2 * a[u] * a[v]
    q = {(p, p): v for p, v in L.items()}
    q.update(Q)
    return q


def repair(b, x):
    x = x.copy()
    for i in range(b.n):                                  # at most one server per camera
        sel = [p for p in b.pairs_of_cam[i] if x[p]]
        if len(sel) > 1:
            keep = min(sel, key=lambda p: b.val[p])
            for p in sel:
                x[p] = p == keep
    for s in range(b.m):                                  # capacity: drop least valuable per unit load
        sel = [p for p in b.pairs_of_srv[s] if x[p]]
        load = sum(b.l[b.pi[p]] for p in sel)
        if load > b.R[s] + 1e-9:
            sel.sort(key=lambda p: -b.val[p] / b.l[b.pi[p]])  # least negative value per load first
            for p in sel:
                if load <= b.R[s] + 1e-9:
                    break
                x[p] = False
                load -= b.l[b.pi[p]]
    return greedy_fill(b, x)


def sample_matrix(response, npairs):
    rec = response.record
    samples = np.asarray(rec.sample)
    labels = list(response.variables)
    X = np.zeros((samples.shape[0], npairs), bool)
    idx = np.array([int(v) for v in labels])
    X[:, idx] = samples > 0
    return X


def solve_qubo(b, solver, params, lam1, lam2, seed, polish=False):
    q = build_qubo_v2(b, lam1, lam2)
    t0 = time.time()
    if solver == "SQA":
        import openjij as oj
        resp = oj.SQASampler().sample_qubo(q, num_reads=params["reads"], num_sweeps=params["sweeps"],
                                           trotter=params["trotter"], seed=seed)
    else:
        import neal
        resp = neal.SimulatedAnnealingSampler().sample_qubo(q, num_reads=params["reads"],
                                                            num_sweeps=params["sweeps"], seed=seed)
    t_sample = time.time() - t0
    X = sample_matrix(resp, b.npairs)
    best, bestv, raw_feas = None, np.inf, 0
    for r in range(X.shape[0]):
        raw_feas += int(b.feasible(X[r]) and np.bincount(b.pi[X[r]], minlength=b.n).min() >= 1)
        xr = repair(b, X[r])
        if polish:
            xr = local_search(b, xr)
        v = b.objective(xr)
        if v < bestv:
            best, bestv = xr, v
    return best, {"sample_time": t_sample, "qubo_terms": len(q), "raw_full_feasible_reads": raw_feas,
                  "reads": int(X.shape[0])}


# --------------------------------------------------------------------------- driver

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capacity-scale", type=float, default=1.0)
    ap.add_argument("--n-cameras", type=int, default=20000)
    ap.add_argument("--n-servers", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=80)
    ap.add_argument("--window", choices=["shared", "percam"], default="shared")
    ap.add_argument("--m-shared", type=int, default=20)
    ap.add_argument("--k-percam", type=int, default=5)
    ap.add_argument("--eval-every", type=int, default=10, help="run QUBO solvers on every k-th batch")
    ap.add_argument("--max-batches", type=int, default=0)
    ap.add_argument("--methods", default="exact,greedy,regret,greedy_ls,sqa,sa")
    ap.add_argument("--reads", type=int, default=10)
    ap.add_argument("--sweeps", type=int, default=1000)
    ap.add_argument("--trotter", type=int, default=8)
    ap.add_argument("--lam1", type=float, default=0.5)
    ap.add_argument("--lam2", type=float, default=2.0)
    ap.add_argument("--polish", action="store_true", help="apply the same local search after QUBO decoding")
    ap.add_argument("--trajectory", default="exact")
    ap.add_argument("--out", default="logs_batch_quality")
    args = ap.parse_args()

    inst = Instance(args.n_cameras, args.n_servers, args.seed, args.capacity_scale)
    gen = inst.gen
    gen.max_servers_per_batch = args.m_shared
    order = np.argsort(-(inst.priority * inst.load))
    nb = int(np.ceil(len(order) / args.batch_size))
    if args.max_batches:
        nb = min(nb, args.max_batches)
    methods = args.methods.split(",")
    residual = inst.cap.copy()
    rows = []
    total = {m: 0.0 for m in methods}
    covered = 0
    run_id = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.out, exist_ok=True)
    tag = f"{args.window}_s{args.capacity_scale:g}"
    t_all = time.time()
    for t in range(nb):
        cams = order[t * args.batch_size:(t + 1) * args.batch_size]
        b = build_batch(inst, cams, residual.copy(), args.window, args.m_shared, args.k_percam, gen)
        evaluate = (t % args.eval_every == 0)
        row = {"batch": t, "npairs": b.npairs, "nservers": b.m, "evaluated": evaluate}
        sols = {}
        for m in methods:
            if m in ("sqa", "sa") and not evaluate:
                continue
            t0 = time.time()
            if m == "exact":
                x, info = solve_exact(b)
            elif m == "greedy":
                x, info = solve_greedy(b)
            elif m == "regret":
                x, info = solve_regret(b)
            elif m == "greedy_ls":
                x, info = solve_greedy_ls(b)
            elif m in ("sqa", "sa"):
                params = {"reads": args.reads, "sweeps": args.sweeps, "trotter": args.trotter}
                x, info = solve_qubo(b, m.upper(), params, args.lam1, args.lam2,
                                     seed=args.seed * 1000 + t, polish=args.polish)
            else:
                raise ValueError(m)
            dt = time.time() - t0
            assert b.feasible(x), (m, t)
            sols[m] = x
            row[m] = {"obj": b.objective(x), "time": dt, "covered": int(np.count_nonzero(x)), **info}
        # commit trajectory
        xt = sols[args.trajectory]
        used = np.bincount(b.ps[xt], weights=b.l[b.pi[xt]], minlength=b.m)
        residual[b.servers] -= used
        covered += int(np.count_nonzero(xt))
        for m in sols:
            if m in ("exact", "greedy", "regret", "greedy_ls"):
                total[m] += row[m]["obj"]
        rows.append(row)
        if t % 25 == 0:
            msg = " ".join(f"{m}={row[m]['obj']:.2f}" for m in methods if m in row)
            print(f"[{tag}] batch {t}/{nb} {msg} elapsed={time.time()-t_all:.0f}s", flush=True)

    ev = [r for r in rows if r["evaluated"] and "exact" in r]
    summary = {
        "run_id": run_id, "window": args.window, "capacity_scale": args.capacity_scale,
        "utilization_percent": inst.utilization, "batches": nb, "trajectory": args.trajectory,
        "trajectory_objective": float(sum(r[args.trajectory]["obj"] for r in rows)),
        "trajectory_covered": covered, "params": vars(args), "methods": {},
    }
    for m in methods:
        rs = [r for r in ev if m in r]
        if not rs:
            continue
        gaps = np.array([(r[m]["obj"] - r["exact"]["obj"]) / r["exact"]["obj"] * 100 for r in rs])
        summary["methods"][m] = {
            "n_eval_batches": len(rs),
            "mean_gap_pct": float(gaps.mean()), "median_gap_pct": float(np.median(gaps)),
            "max_gap_pct": float(gaps.max()), "optimal_batches": int(np.sum(gaps < 1e-6)),
            "mean_time_s": float(np.mean([r[m]["time"] for r in rs])),
            "sum_obj_eval": float(sum(r[m]["obj"] for r in rs)),
        }
    path = os.path.join(args.out, f"summary_{tag}_{run_id}.json")
    with open(path, "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=1, default=float)
    print(json.dumps(summary["methods"], indent=1))
    print(f"trajectory objective ({args.trajectory}) = {summary['trajectory_objective']:.1f}, covered={covered}")
    print("saved", path)


if __name__ == "__main__":
    main()
