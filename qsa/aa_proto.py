"""Anti-affinity extension: cameras with overlapping fields of view should not share a server
(a server failure would blind the same area twice). Objective = F + beta * #co-located overlapping pairs.

Batch-level benchmark on common states: the trajectory is driven by the exact (linearised MILP) solution;
on every batch all methods solve the same subproblem and are scored with the full quadratic objective.
"""
import argparse, json, os, sys, time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from batch_quality_benchmark import Instance, build_batch, greedy_fill, P_UNCOVERED
from qubo_v3_lab import build_v3, decode_v3, run
from v3_trajectory import lagrange_prices


def quad_pairs(b, nbr):
    """In-batch pair-index pairs (p, q) on the same server whose cameras overlap."""
    loc = {c: i for i, c in enumerate(b.cams)}
    out = []
    for i, c in enumerate(b.cams):
        for c2 in nbr[c]:
            k = loc.get(c2)
            if k is None or k <= i:
                continue
            for p in b.pairs_of_cam[i]:
                for q in b.pairs_of_cam[k]:
                    if b.ps[p] == b.ps[q]:
                        out.append((p, q))
    return out


def full_obj(b, x, qp, beta):
    """Batch decision objective: priced costs (incl. linear anti-affinity), uncovered penalty, pairwise term."""
    return float(b.val[x].sum() + P_UNCOVERED * b.n + beta * sum(1 for p, q in qp if x[p] and x[q]))


def solve_exact_aa(b, qp, beta, tl):
    n0 = b.npairs; ny = len(qp)
    rows, cols, vals, lb, ub = [], [], [], [], []
    r = 0
    for i in range(b.n):
        for p in b.pairs_of_cam[i]:
            rows.append(r); cols.append(p); vals.append(1.0)
        lb.append(-np.inf); ub.append(1.0); r += 1
    for s in range(b.m):
        for p in b.pairs_of_srv[s]:
            rows.append(r); cols.append(p); vals.append(b.l[b.pi[p]])
        lb.append(-np.inf); ub.append(b.R[s]); r += 1
    for k, (p, q) in enumerate(qp):          # x_p + x_q - y_k <= 1
        rows += [r, r, r]; cols += [p, q, n0 + k]; vals += [1.0, 1.0, -1.0]
        lb.append(-np.inf); ub.append(1.0); r += 1
    A = coo_matrix((vals, (rows, cols)), shape=(r, n0 + ny)).tocsr()
    c = np.concatenate([b.val, np.full(ny, beta)])
    integ = np.concatenate([np.ones(n0), np.zeros(ny)])
    t = time.time()
    res = milp(c=c, constraints=[LinearConstraint(A, lb, ub)], integrality=integ, bounds=Bounds(0, 1),
               options={"time_limit": tl, "mip_rel_gap": 1e-9})
    x = np.zeros(n0, bool) if res.x is None else res.x[:n0] > 0.5
    lbnd = getattr(res, "mip_dual_bound", None)
    return x, {"time": time.time() - t, "status": int(res.status), "dual_bound": None if lbnd is None else float(lbnd) + b.n * P_UNCOVERED}


def greedy_aa(b, qp, beta, order=None, x=None):
    nb = [[] for _ in range(b.npairs)]
    for p, q in qp:
        nb[p].append(q); nb[q].append(p)
    x = np.zeros(b.npairs, bool) if x is None else x.copy()
    used = np.bincount(b.ps[x], weights=b.l[b.pi[x]], minlength=b.m).astype(float)
    done = np.zeros(b.n, bool); done[b.pi[x]] = True
    for i in (np.argsort(-(b.prio * b.l)) if order is None else order):
        if done[i]:
            continue
        best, bv = -1, 0.0
        for p in b.pairs_of_cam[i]:
            s = b.ps[p]
            if used[s] + b.l[i] > b.R[s] + 1e-9:
                continue
            v = b.val[p] + beta * sum(x[q] for q in nb[p])
            if v < bv:
                best, bv = p, v
        if best >= 0:
            x[best] = True; used[b.ps[best]] += b.l[i]; done[i] = True
    return x


def ls_aa(b, qp, beta, x, rounds=30):
    nb = [[] for _ in range(b.npairs)]
    for p, q in qp:
        nb[p].append(q); nb[q].append(p)
    x = x.copy()
    used = np.bincount(b.ps[x], weights=b.l[b.pi[x]], minlength=b.m).astype(float)
    cur = -np.ones(b.n, int); cur[b.pi[x]] = np.where(x)[0]
    def cost(p):
        return b.val[p] + beta * sum(x[q] for q in nb[p])
    for _ in range(rounds):
        imp = False
        for i in range(b.n):
            p0 = cur[i]
            if p0 >= 0:
                x[p0] = False
            v0 = cost(p0) if p0 >= 0 else 0.0
            bp, bv = p0, v0
            for p in b.pairs_of_cam[i]:
                s = b.ps[p]
                extra = b.l[i] if (p0 < 0 or b.ps[p0] != s) else 0.0
                if used[s] + extra > b.R[s] + 1e-9:
                    continue
                v = cost(p)
                if v < bv - 1e-9:
                    bp, bv = p, v
            if bp != p0:
                if p0 >= 0:
                    used[b.ps[p0]] -= b.l[i]
                used[b.ps[bp]] += b.l[i]; cur[i] = bp; imp = True
            if cur[i] >= 0:
                x[cur[i]] = True
        if not imp:
            break
    return x


def ils_aa(b, qp, beta, budget, rng):
    t = time.time()
    x = ls_aa(b, qp, beta, greedy_aa(b, qp, beta)); bo = full_obj(b, x, qp, beta); best = x.copy(); cur, co = x, bo
    while time.time() - t < budget:
        y = cur.copy(); a = np.where(y)[0]
        if len(a) == 0:
            break
        y[rng.choice(a, size=max(1, int(0.2 * len(a))), replace=False)] = False
        y = ls_aa(b, qp, beta, greedy_aa(b, qp, beta, order=rng.permutation(b.n), x=y))
        o = full_obj(b, y, qp, beta)
        if o <= co + 1e-9 or rng.random() < 0.05:
            cur, co = y, o
        if o < bo - 1e-9:
            best, bo = y.copy(), o
    return best


def solve_qubo_aa(b, qp, beta, solver, seed, reads=10, sweeps=2000, polish=False):
    quad = [(p, q, beta) for p, q in qp]
    q, info, fixed, dinfo = build_v3(b, True, True, quad=quad)
    if not q:
        return greedy_fill(b, fixed.copy()), info
    if solver == "SQA":   # fixed-temperature SQA schedule: bring coefficients to the scale it was tuned for
        sc = 0.7 / max(np.percentile(np.abs(list(q.values())), 95), 1e-9)
        q = {k: v * sc for k, v in q.items()}
    best, bo = None, np.inf
    for smp in run(q, solver, reads, sweeps, 8, seed=seed, gamma=1.0 if solver == "SQA" else None, beta=20.0 if solver == "SQA" else None):
        x, _ = decode_v3(b, smp, fixed, dinfo, True)
        if polish:
            x = ls_aa(b, qp, beta, x)
        o = full_obj(b, x, qp, beta)
        if o < bo:
            best, bo = x, o
    return best, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000); ap.add_argument("--m", type=int, default=200)
    ap.add_argument("--util", type=float, default=95); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--radius", type=float, default=16.0); ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=80); ap.add_argument("--max-batches", type=int, default=20)
    ap.add_argument("--first-batch", type=int, default=20)
    ap.add_argument("--tl", type=float, default=30.0); ap.add_argument("--ils", type=float, default=2.0)
    ap.add_argument("--methods", default="greedy,greedy_ls,ils,sa,sqa,sqa_ls")
    ap.add_argument("--out", default="aa")
    a = ap.parse_args()
    inst = Instance(a.n, a.m, a.seed, 0.25, a.util); g = inst.gen
    xy = np.c_[g.camera_x, g.camera_y]
    nbr = cKDTree(xy).query_ball_point(xy, a.radius)
    nbr = [set(v) - {i} for i, v in enumerate(nbr)]
    side = np.sqrt(1e6 * a.bs / a.n)                     # spatial tiles of ~bs cameras (district roll-out)
    ty, tx = np.floor(g.camera_y / side), np.floor(g.camera_x / side)
    tx = np.where(ty % 2 == 1, -tx, tx)                  # serpentine order of tiles
    order = np.lexsort((-(inst.priority * inst.load), tx, ty))
    res = inst.cap.copy(); assigned = -np.ones(a.n, int)
    u, _ = lagrange_prices(inst)
    rng = np.random.default_rng(a.seed); rows = []
    nbt = int(np.ceil(a.n / a.bs))
    for t in range(min(nbt, a.first_batch + a.max_batches)):
        cams = order[t * a.bs:(t + 1) * a.bs]
        if t > 0 and t % 5 == 0:
            u, _ = lagrange_prices(inst, cams=order[t * a.bs:], cap=res)
        sel = lambda c: inst.wcost[c] + u * inst.load[c]
        b = build_batch(inst, cams, res.copy(), "percam", 20, 5, g, sel_cost=sel, k_slack=2)
        # linear anti-affinity with already placed neighbours
        lin = np.array([a.beta * sum(1 for c2 in nbr[b.cams[b.pi[p]]] if assigned[c2] == b.servers[b.ps[p]]) for p in range(b.npairs)])
        add = u[b.servers[b.ps]] * b.l[b.pi]
        b.pcost = b.pcost + lin; b.val = b.val + lin + add; b.dec = b.pcost + add
        qp = quad_pairs(b, nbr)
        xe, st = solve_exact_aa(b, qp, a.beta, a.tl)
        if t >= a.first_batch and qp:
            row = {"batch": t, "quad_pairs": len(qp), "pairs": b.npairs, "exact": full_obj(b, xe, qp, a.beta),
                   "exact_time": st["time"], "exact_status": st["status"], "exact_bound": st["dual_bound"]}
            for m in a.methods.split(","):
                t0 = time.time()
                if m == "greedy":
                    x = greedy_aa(b, qp, a.beta)
                elif m == "greedy_ls":
                    x = ls_aa(b, qp, a.beta, greedy_aa(b, qp, a.beta))
                elif m == "ils":
                    x = ils_aa(b, qp, a.beta, a.ils, rng)
                else:
                    x, info = solve_qubo_aa(b, qp, a.beta, "SQA" if m.startswith("sqa") else "SA", a.seed * 1000 + t, polish=m.endswith("_ls"))
                    row["qubo_vars"] = info["vars"]
                assert b.feasible(x)
                row[m] = full_obj(b, x, qp, a.beta); row[m + "_time"] = time.time() - t0
            rows.append(row)
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
        # advance with the exact solution
        res[b.servers] -= np.bincount(b.ps[xe], weights=b.l[b.pi[xe]], minlength=b.m)
        assigned[b.cams[b.pi[xe]]] = b.servers[b.ps[xe]]
    os.makedirs(a.out, exist_ok=True)
    json.dump({"args": vars(a), "rows": rows}, open(os.path.join(a.out, f"aa_r{a.radius:g}_b{a.beta:g}_u{a.util:g}_seed{a.seed}_{time.strftime('%H%M%S')}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
