"""QUBO formulation lab on hard batch subproblems.

Replays the exact-MILP trajectory, collects batches where the priority greedy is >= min_gap
worse than the batch optimum, and compares QUBO formulations / samplers on exactly those
batches against exact, greedy, regret and regret+local-search.

Formulations (all with an explicit "unassigned" dummy per camera, cost P):
  F1  exactly-one (normalised costs) + unbalanced capacity penalty (no slack)
  F2  exactly-one (normalised costs) + exact capacity equality with binary slack variables
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from batch_quality_benchmark import (P_UNCOVERED, Instance, build_batch, greedy_fill, local_search,  # noqa: E402
                                     solve_exact, solve_greedy, solve_regret)


def build_qubo(b, form, lam_onehot=1.5, lam_cap=1.0, lam1=0.2, slack_bits=6):
    """Variables: 0..npairs-1 = pairs, npairs..npairs+n-1 = dummy 'unassigned' per camera, then slacks."""
    n, npairs = b.n, b.npairs
    cost_pair = b.pcost.copy()
    cmin = np.array([cost_pair[b.pairs_of_cam[i]].min() if len(b.pairs_of_cam[i]) else P_UNCOVERED
                     for i in range(n)])
    gaps = []
    for i in range(n):
        cs = np.sort(cost_pair[b.pairs_of_cam[i]])
        if len(cs) > 1:
            gaps.append(cs[1] - cs[0])
    unit = max(float(np.median(gaps)) if gaps else 1.0, 1e-3)      # typical decision difference -> 1
    lin_pair = (cost_pair - cmin[b.pi]) / unit
    lin_dummy = (P_UNCOVERED - cmin) / unit
    # one-hot weight: just above the largest cost difference that matters (cap at dummy cost)
    A = lam_onehot * float(np.percentile(np.concatenate([lin_pair, np.minimum(lin_dummy, lin_pair.max())]), 95))
    A = max(A, 1.0)
    q = {}

    def add(u, v, c):
        key = (u, v) if u <= v else (v, u)
        q[key] = q.get(key, 0.0) + float(c)

    # exactly-one over {pairs of camera i} U {dummy_i}: A (sum x - 1)^2
    for i in range(n):
        vs = [int(p) for p in b.pairs_of_cam[i]] + [npairs + i]
        lins = [lin_pair[p] for p in b.pairs_of_cam[i]] + [lin_dummy[i]]
        for v, c in zip(vs, lins):
            add(v, v, c - A)
        for a in range(len(vs)):
            for c in range(a + 1, len(vs)):
                add(vs[a], vs[c], 2 * A)
    nvar = npairs + n
    cap_servers = 0
    for s in range(b.m):
        ps = b.pairs_of_srv[s]
        if len(ps) == 0:
            continue
        a = b.l[b.pi[ps]] / max(b.R[s], 1e-9)
        if a.sum() <= 1.0 + 1e-12:
            continue
        cap_servers += 1
        W = lam_cap * A
        if form == "F1":            # unbalanced: -lam1*h + h^2, h = 1 - sum a x   (scaled by W)
            for u, p in enumerate(ps):
                add(int(p), int(p), W * (lam1 * a[u] - 2 * a[u] + a[u] ** 2))
            for u in range(len(ps)):
                for v in range(u + 1, len(ps)):
                    add(int(ps[u]), int(ps[v]), W * 2 * a[u] * a[v])
        else:                       # F2: W (sum a x + sum b s - 1)^2
            bits = [2.0 ** k for k in range(slack_bits)]
            bsc = np.array(bits) / sum(bits)          # slack in [0,1]
            sv = list(range(nvar, nvar + slack_bits))
            nvar += slack_bits
            coefs = list(a) + list(bsc)
            vars_ = [int(p) for p in ps] + sv
            for u in range(len(vars_)):
                add(vars_[u], vars_[u], W * (coefs[u] ** 2 - 2 * coefs[u]))
                for v in range(u + 1, len(vars_)):
                    add(vars_[u], vars_[v], W * 2 * coefs[u] * coefs[v])
    return q, {"vars": nvar, "terms": len(q), "cap_servers": cap_servers, "A": A, "unit": unit}


def decode(b, sample_row):
    x = sample_row[: b.npairs].astype(bool).copy()
    for i in range(b.n):
        sel = [p for p in b.pairs_of_cam[i] if x[p]]
        if len(sel) > 1:
            keep = min(sel, key=lambda p: b.val[p])
            for p in sel:
                x[p] = p == keep
    for s in range(b.m):
        sel = [p for p in b.pairs_of_srv[s] if x[p]]
        load = sum(b.l[b.pi[p]] for p in sel)
        if load > b.R[s] + 1e-9:
            sel.sort(key=lambda p: -b.val[p] / b.l[b.pi[p]])
            for p in sel:
                if load <= b.R[s] + 1e-9:
                    break
                x[p] = False
                load -= b.l[b.pi[p]]
    return greedy_fill(b, x)


def run_sampler(q, nvar, solver, reads, sweeps, trotter, seed, init=None):
    if solver == "SQA":
        import openjij as oj
        kw = dict(num_reads=reads, num_sweeps=sweeps, trotter=trotter, seed=seed)
        if init is not None:
            kw["initial_state"] = {v: int(init[v]) for v in range(nvar)}
        r = oj.SQASampler().sample_qubo(q, **kw)
    else:
        import neal
        kw = dict(num_reads=reads, num_sweeps=sweeps, seed=seed)
        if init is not None:
            kw["initial_states"] = (np.tile(init.astype(np.int8), (reads, 1)), list(range(nvar)))
            kw["initial_states_generator"] = "tile"
        r = neal.SimulatedAnnealingSampler().sample_qubo(q, **kw)
    S = np.asarray(r.record.sample)
    lab = [int(v) for v in r.variables]
    X = np.zeros((S.shape[0], nvar), np.int8)
    X[:, lab] = S
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capacity-scale", type=float, default=0.25)
    ap.add_argument("--window", default="shared")
    ap.add_argument("--k-percam", type=int, default=5)
    ap.add_argument("--min-gap", type=float, default=0.01)
    ap.add_argument("--max-hard", type=int, default=12)
    ap.add_argument("--configs", default="F1:SA,F1:SQA,F2:SA,F2:SQA")
    ap.add_argument("--reads", type=int, default=10)
    ap.add_argument("--sweeps", type=int, default=2000)
    ap.add_argument("--trotter", type=int, default=8)
    ap.add_argument("--lam-cap", type=float, default=1.0)
    ap.add_argument("--lam-onehot", type=float, default=1.5)
    ap.add_argument("--warm", action="store_true", help="also run warm-started variants from the greedy solution")
    ap.add_argument("--out", default="logs_qubo_lab")
    args = ap.parse_args()

    inst = Instance(20000, 800, 42, args.capacity_scale)
    gen = inst.gen
    order = np.argsort(-(inst.priority * inst.load))
    residual = inst.cap.copy()
    hard = []
    for t in range(250):
        cams = order[t * 80:(t + 1) * 80]
        b = build_batch(inst, cams, residual.copy(), args.window, 20, args.k_percam, gen)
        xe, _ = solve_exact(b)
        xg, _ = solve_greedy(b)
        oe, og = b.objective(xe), b.objective(xg)
        if (og - oe) / oe >= args.min_gap and len(hard) < args.max_hard:
            hard.append((t, b, xe, xg))
        used = np.bincount(b.ps[xe], weights=b.l[b.pi[xe]], minlength=b.m)
        residual[b.servers] -= used
    print(f"collected {len(hard)} hard batches", flush=True)

    rows = []
    for t, b, xe, xg in hard:
        oe = b.objective(xe)
        row = {"batch": t, "exact": oe, "greedy": b.objective(xg)}
        t0 = time.time(); xr, _ = solve_regret(b); row["regret"] = b.objective(xr)
        row["regret_ls"] = b.objective(local_search(b, xr)); row["t_regret_ls"] = time.time() - t0
        for cfg in args.configs.split(","):
            form, solver = cfg.split(":")
            q, info = build_qubo(b, form, lam_onehot=args.lam_onehot, lam_cap=args.lam_cap)
            variants = [("cold", None)]
            if args.warm:
                init = np.zeros(info["vars"], np.int8)
                init[: b.npairs] = xg
                cov = np.zeros(b.n, bool); cov[b.pi[xg]] = True
                init[b.npairs: b.npairs + b.n] = ~cov
                variants.append(("warm", init))
            for vname, init in variants:
                t0 = time.time()
                X = run_sampler(q, info["vars"], solver, args.reads, args.sweeps, args.trotter, seed=1000 + t, init=init)
                ts = time.time() - t0
                objs, objs_ls, raw_onehot = [], [], 0
                for r in range(X.shape[0]):
                    xs = X[r, : b.npairs].astype(bool)
                    dummy = X[r, b.npairs: b.npairs + b.n].astype(bool)
                    per = np.bincount(b.pi[xs], minlength=b.n) + dummy
                    raw_onehot += int(np.all(per == 1))
                    xd = decode(b, X[r])
                    objs.append(b.objective(xd))
                    objs_ls.append(b.objective(local_search(b, xd)))
                key = f"{form}:{solver}:{vname}"
                row[key] = min(objs)
                row[key + "+ls"] = min(objs_ls)
                row[key + ":time"] = ts
                row[key + ":onehot_reads"] = raw_onehot / X.shape[0]
                row[key + ":vars"] = info["vars"]
        rows.append(row)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)

    keys = [k for k in rows[0] if k not in ("batch",) and ":time" not in k and ":onehot" not in k
            and ":vars" not in k and not k.startswith("t_")]
    summary = {}
    for k in keys:
        g = np.array([(r[k] - r["exact"]) / r["exact"] * 100 for r in rows])
        summary[k] = {"mean_gap_pct": float(g.mean()), "max_gap_pct": float(g.max()),
                      "optimal": int(np.sum(g < 1e-6)), "beats_greedy": int(sum(r[k] < r["greedy"] - 1e-9 for r in rows)),
                      "beats_regret_ls": int(sum(r[k] < r["regret_ls"] - 1e-9 for r in rows))}
        tk = k.replace("+ls", "") + ":time"
        if tk in rows[0]:
            summary[k]["mean_time_s"] = float(np.mean([r[tk] for r in rows]))
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"lab_{args.window}_s{args.capacity_scale:g}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    json.dump({"args": vars(args), "summary": summary, "rows": rows}, open(path, "w"), indent=1)
    for k, v in summary.items():
        print(f"{k:28s} gap {v['mean_gap_pct']:7.3f}% max {v['max_gap_pct']:7.2f}% opt {v['optimal']:2d} "
              f">greedy {v['beats_greedy']:2d} >regLS {v['beats_regret_ls']:2d} t {v.get('mean_time_s', float('nan')):.2f}s")
    print("saved", path)


if __name__ == "__main__":
    main()
