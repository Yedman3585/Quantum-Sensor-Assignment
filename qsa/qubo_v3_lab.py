"""QUBO-v3 lab: improvements aimed at *pure* annealing quality (no classical local search).

Components (each can be switched on/off for ablation):
  R   exact problem reduction: a camera whose cheapest candidate server can never overflow
      (sum of loads of all candidates for that server <= residual) is fixed to it. This is
      provably optimal-preserving; such servers also need no capacity term.
  DW  domain-wall encoding of the per-camera choice (Chancellor 2019). Options sorted by cost
      (dummy 'unassigned' last); k options -> k-1 spins; every single flip moves the wall by
      one position, so single-flip dynamics never leaves the one-hot manifold.
      Without DW: standard exactly-one penalty A (sum x - 1)^2.
  Capacity: unbalanced penalisation W[(lam1-2) S + S^2], S = sum_i (l_i/R_j) x_ij (no slack).

Decoding: DW -> wall position (always a valid choice); one-hot -> cheapest selected / dummy.
Then the SAME minimal feasibility repair for every variant (drop overload, greedy fill of freed
cameras). We report the pre-repair capacity violation rate so repair is not hiding quality.
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from batch_quality_benchmark import (P_UNCOVERED, Instance, build_batch, greedy_fill,  # noqa: E402
                                     solve_exact, solve_greedy, solve_regret, local_search)


class Expr:
    """Affine expression over binary variables: const + sum coef*var."""

    def __init__(self, const=0.0, terms=None):
        self.c = float(const)
        self.t = dict(terms or {})

    def __add__(self, o):
        t = dict(self.t)
        for k, v in o.t.items():
            t[k] = t.get(k, 0.0) + v
        return Expr(self.c + o.c, t)

    def scale(self, s):
        return Expr(self.c * s, {k: v * s for k, v in self.t.items()})


class QB:
    def __init__(self):
        self.q = defaultdict(float)
        self.n = 0

    def new(self):
        self.n += 1
        return self.n - 1

    def lin(self, e, w):            # add w * e (constant dropped)
        for k, v in e.t.items():
            self.q[(k, k)] += w * v

    def prod(self, e1, e2, w):      # add w * e1 * e2 (constant dropped)
        for k, v in e1.t.items():
            self.q[(k, k)] += w * v * e2.c
        for k, v in e2.t.items():
            self.q[(k, k)] += w * v * e1.c
        for k1, v1 in e1.t.items():
            for k2, v2 in e2.t.items():
                if k1 == k2:
                    self.q[(k1, k1)] += w * v1 * v2          # x^2 = x
                else:
                    key = (k1, k2) if k1 < k2 else (k2, k1)
                    self.q[key] += w * v1 * v2


def reduce_batch(b, nofix=None):
    """Returns fixed pairs (bool over pairs) and the set of 'safe' servers (cannot overflow).
    nofix: optional bool mask over cameras that must stay free (e.g. cameras with pairwise terms)."""
    total = np.bincount(b.ps, weights=b.l[b.pi], minlength=b.m)
    safe = total <= b.R + 1e-9
    fixed = np.zeros(b.npairs, bool)
    fixed_cam = np.zeros(b.n, bool)
    for i in range(b.n):
        ps = b.pairs_of_cam[i]
        if len(ps) == 0 or (nofix is not None and nofix[i]):
            continue
        p = ps[np.argmin(b.val[ps])]
        if safe[b.ps[p]]:
            fixed[p] = True
            fixed_cam[i] = True
    return fixed, fixed_cam, safe


def build_v3(b, use_dw, use_red, lam_onehot=0.45, lam_cap=1.0, lam1=0.2, lam_dw=None, cap_mode="unbalanced", slack_bits=6, srv_w=None, srv_mu=None, quad=None):
    """quad: optional list of (pair_p, pair_q, weight) pairwise costs w * x_p * x_q (same units as costs)."""
    nofix = None
    if quad:
        nofix = np.zeros(b.n, bool)
        for p_, q_, _ in quad:
            nofix[b.pi[p_]] = True; nofix[b.pi[q_]] = True
    fixed, fixed_cam, safe = reduce_batch(b, nofix) if use_red else (np.zeros(b.npairs, bool), np.zeros(b.n, bool),
                                                              np.zeros(b.m, bool))
    # cost normalisation (same as v2)
    dec = getattr(b, "dec", b.pcost)          # decision cost (may include capacity prices)
    cmin = np.array([dec[ps].min() if len(ps) else P_UNCOVERED for ps in b.pairs_of_cam])
    gaps = [np.diff(np.sort(dec[ps]))[0] for ps in b.pairs_of_cam if len(ps) > 1]
    unit = max(float(np.median(gaps)) if gaps else 1.0, 1e-3)
    lin_pair = (dec - cmin[b.pi]) / unit
    lin_dummy = (P_UNCOVERED - cmin) / unit
    A = max(lam_onehot * float(np.percentile(np.concatenate([lin_pair, np.minimum(lin_dummy, lin_pair.max())]), 95)), 1.0)
    Wdw = A if lam_dw is None else lam_dw * A
    # pairwise terms on domain-wall expressions are unbounded below off the valid manifold
    # (x = z_m - z_{m+1} can be -1), so each camera's wall penalty must dominate its pairwise weight
    qdeg = np.zeros(b.n)
    if quad:
        for p_, q_, w_ in quad:
            qdeg[b.pi[p_]] += abs(w_) / unit; qdeg[b.pi[q_]] += abs(w_) / unit
    qb = QB()
    xexpr = {}          # pair -> Expr
    decode_info = []    # per free camera: (i, options list (pair or -1 dummy), vars)
    for i in range(b.n):
        if fixed_cam[i]:
            continue
        ps = list(b.pairs_of_cam[i])
        opts = sorted(ps, key=lambda p: lin_pair[p]) + [-1]
        costs = [lin_pair[p] for p in opts[:-1]] + [lin_dummy[i]]
        k = len(opts)
        if use_dw:
            z = [qb.new() for _ in range(k - 1)]          # z_1..z_{k-1}; z_0 = 1, z_k = 0
            zexpr = [Expr(1.0)] + [Expr(0.0, {v: 1.0}) for v in z] + [Expr(0.0)]
            for m in range(k):
                xm = zexpr[m] + zexpr[m + 1].scale(-1.0)
                if opts[m] >= 0:
                    xexpr[opts[m]] = xm
                qb.lin(xm, costs[m])
            Wi = Wdw + 2.0 * qdeg[i]
            for m in range(1, k - 1):                     # z_{m+1} <= z_m : Wdw * z_{m+1}(1 - z_m)
                qb.q[(z[m], z[m])] += Wi
                a_, c_ = sorted((z[m - 1], z[m]))
                qb.q[(a_, c_)] -= Wi
            decode_info.append((i, opts, z))
        else:
            xs = [qb.new() for _ in range(k)]
            for m in range(k):
                e = Expr(0.0, {xs[m]: 1.0})
                if opts[m] >= 0:
                    xexpr[opts[m]] = e
                qb.lin(e, costs[m])
            s = Expr(-1.0, {v: 1.0 for v in xs})
            qb.prod(s, s, A)
            decode_info.append((i, opts, xs))
    if quad:
        for p_, q_, w_ in quad:
            qb.prod(xexpr[p_], xexpr[q_], w_ / unit)
    cap_servers = 0
    for s in range(b.m):
        if use_red and safe[s]:
            continue
        ps = [p for p in b.pairs_of_srv[s] if p in xexpr]
        if not ps:
            continue
        a = b.l[b.pi[ps]] / max(b.R[s], 1e-9)
        fixed_load = sum(b.l[b.pi[p]] for p in b.pairs_of_srv[s] if fixed[p]) / max(b.R[s], 1e-9)
        if a.sum() + fixed_load <= 1.0 + 1e-12:
            continue
        cap_servers += 1
        S = Expr(fixed_load)
        for u, p in enumerate(ps):
            S = S + xexpr[p].scale(a[u])
        W = lam_cap * A * (1.0 if srv_w is None else float(srv_w[s]))
        if srv_mu is not None and srv_mu[s] > 0:
            qb.lin(S, A * float(srv_mu[s]))          # Lagrange price on server load
        if cap_mode == "slack":
            # exact inequality S <= 1 via S + sigma - 1 = 0, sigma = sum_k b_k s_k in [0, 1 - fixed_load]
            head = max(1.0 - fixed_load, 0.0)
            bits = np.array([2.0 ** k for k in range(slack_bits)])
            bsc = bits / bits.sum() * head
            T = S + Expr(-1.0)
            for bk in bsc:
                T = T + Expr(0.0, {qb.new(): float(bk)})
            qb.prod(T, T, W)
        else:
            qb.lin(S, W * (lam1 - 2.0))
            qb.prod(S, S, W)
    q = {k: v for k, v in qb.q.items() if abs(v) > 1e-12}
    info = {"vars": qb.n, "terms": len(q), "free_cams": len(decode_info), "fixed_cams": int(fixed_cam.sum()),
            "cap_servers": cap_servers}
    return q, info, fixed, decode_info


def decode_v3(b, sample, fixed, decode_info, use_dw):
    x = fixed.copy()
    for i, opts, vs in decode_info:
        if use_dw:
            m = 0
            while m < len(vs) and sample.get(vs[m], 0) == 1:
                m += 1
            if opts[m] >= 0:
                x[opts[m]] = True
        else:
            sel = [opts[m] for m, v in enumerate(vs) if sample.get(v, 0) == 1 and opts[m] >= 0]
            if sel:
                x[min(sel, key=lambda p: b.val[p])] = True
    # capacity repair
    violated = 0
    for s in range(b.m):
        sel = [p for p in b.pairs_of_srv[s] if x[p]]
        load = sum(b.l[b.pi[p]] for p in sel)
        if load > b.R[s] + 1e-9:
            violated += 1
            sel.sort(key=lambda p: -b.val[p] / b.l[b.pi[p]])
            for p in sel:
                if load <= b.R[s] + 1e-9:
                    break
                if fixed[p]:
                    continue
                x[p] = False
                load -= b.l[b.pi[p]]
    return greedy_fill(b, x), violated


def raw_choice(b, sample, fixed, decode_info, use_dw):
    """Decoded choice BEFORE capacity repair (for measuring overflow)."""
    x = fixed.copy()
    for i, opts, vs in decode_info:
        if use_dw:
            m = 0
            while m < len(vs) and sample.get(vs[m], 0) == 1:
                m += 1
            if opts[m] >= 0:
                x[opts[m]] = True
        else:
            sel = [opts[m] for m, v in enumerate(vs) if sample.get(v, 0) == 1 and opts[m] >= 0]
            if sel:
                x[min(sel, key=lambda p: b.val[p])] = True
    return x


def run(q, solver, reads, sweeps, trotter, seed, gamma=None, beta=None):
    if solver == "SQA":
        import openjij as oj
        kw = dict(num_reads=reads, num_sweeps=sweeps, trotter=trotter, seed=seed)
        if gamma is not None:
            kw["gamma"] = gamma
        if beta is not None:
            kw["beta"] = beta
        r = oj.SQASampler().sample_qubo(q, **kw)
    else:
        import neal
        r = neal.SimulatedAnnealingSampler().sample_qubo(q, num_reads=reads, num_sweeps=sweeps, seed=seed)
    labels = list(r.variables)
    return [dict(zip(labels, row)) for row in np.asarray(r.record.sample)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capacity-scale", type=float, default=0.25)
    ap.add_argument("--target-util", type=float, default=None)
    ap.add_argument("--instance-seed", type=int, default=42)
    ap.add_argument("--window", default="shared")
    ap.add_argument("--k-percam", type=int, default=5)
    ap.add_argument("--min-gap", type=float, default=0.01)
    ap.add_argument("--max-hard", type=int, default=12)
    ap.add_argument("--variants", default="base,R,DW,R+DW")
    ap.add_argument("--solvers", default="SQA")
    ap.add_argument("--reads", type=int, default=10)
    ap.add_argument("--sweeps", type=int, default=2000)
    ap.add_argument("--trotter", type=int, default=8)
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--lam-onehot", type=float, default=0.45)
    ap.add_argument("--lam-cap", type=float, default=1.0)
    ap.add_argument("--lam-dw", type=float, default=None)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--cap-mode", choices=["unbalanced", "slack"], default="unbalanced")
    ap.add_argument("--slack-bits", type=int, default=6)
    ap.add_argument("--adaptive-rounds", type=int, default=1, help=">1: re-sample with prices/weights raised on overflowing servers")
    ap.add_argument("--mu-step", type=float, default=0.5)
    ap.add_argument("--w-growth", type=float, default=2.0)
    ap.add_argument("--out", default="logs_qubo_v3")
    args = ap.parse_args()

    inst = Instance(20000, 800, args.instance_seed, args.capacity_scale, args.target_util)
    order = np.argsort(-(inst.priority * inst.load))
    residual = inst.cap.copy()
    hard = []
    for t in range(250):
        cams = order[t * 80:(t + 1) * 80]
        b = build_batch(inst, cams, residual.copy(), args.window, 20, args.k_percam, inst.gen)
        xe, _ = solve_exact(b)
        xg, _ = solve_greedy(b)
        if (b.objective(xg) - b.objective(xe)) / b.objective(xe) >= args.min_gap and len(hard) < args.max_hard:
            hard.append((t, b, xe, xg))
        residual[b.servers] -= np.bincount(b.ps[xe], weights=b.l[b.pi[xe]], minlength=b.m)
    print(f"collected {len(hard)} hard batches ({args.window}, scale {args.capacity_scale})", flush=True)

    rows = []
    for t, b, xe, xg in hard:
        oe = b.objective(xe)
        xr, _ = solve_regret(b)
        row = {"batch": t, "exact": oe, "greedy": b.objective(xg), "regret": b.objective(xr),
               "regret_ls": b.objective(local_search(b, xr))}
        for var in args.variants.split(","):
            use_red = "R" in var.split("+")
            use_dw = "DW" in var.split("+")
            q, info, fixed, dinfo = build_v3(b, use_dw, use_red, lam_onehot=args.lam_onehot,
                                             lam_cap=args.lam_cap, lam_dw=args.lam_dw,
                                             cap_mode=args.cap_mode, slack_bits=args.slack_bits)
            for solver in args.solvers.split(","):
                key = f"{var}:{solver}"
                best, viol, tt = [], [], 0.0
                for sd in range(args.seeds):
                    if not q:          # everything fixed by reduction
                        x = greedy_fill(b, fixed.copy()); best.append(b.objective(x)); viol.append(0)
                        continue
                    srv_w = np.ones(b.m); srv_mu = np.zeros(b.m)
                    objs, vs = [], []
                    qq, dd = q, dinfo
                    for rnd in range(args.adaptive_rounds):
                        if rnd > 0:
                            qq, _, fixed, dd = build_v3(b, use_dw, use_red, lam_onehot=args.lam_onehot,
                                                        lam_cap=args.lam_cap, lam_dw=args.lam_dw,
                                                        cap_mode=args.cap_mode, slack_bits=args.slack_bits,
                                                        srv_w=srv_w, srv_mu=srv_mu)
                        t0 = time.time()
                        samples = run(qq, solver, args.reads, args.sweeps, args.trotter, seed=7919 * sd + 31 * rnd + t,
                                      gamma=args.gamma, beta=args.beta)
                        tt += time.time() - t0
                        rbest, rover = np.inf, None
                        for smp in samples:
                            x, v = decode_v3(b, smp, fixed, dd, use_dw)
                            assert b.feasible(x)
                            o = b.objective(x); objs.append(o); vs.append(v)
                            if o < rbest:
                                rbest = o
                                raw = raw_choice(b, smp, fixed, dd, use_dw)
                                rover = np.bincount(b.ps[raw], weights=b.l[b.pi[raw]], minlength=b.m) / np.maximum(b.R, 1e-9) - 1.0
                        if rover is None or np.all(rover <= 1e-9):
                            break
                        viol_s = rover > 1e-9
                        srv_mu[viol_s] += args.mu_step * (1.0 + rover[viol_s])
                        srv_w[viol_s] *= args.w_growth
                    k = int(np.argmin(objs)); best.append(objs[k]); viol.append(vs[k])
                row[key] = float(np.mean(best))
                row[key + ":time"] = tt / max(1, args.seeds)
                row[key + ":viol_servers"] = float(np.mean(viol))
                row[key + ":vars"] = info["vars"]
                row[key + ":fixed"] = info["fixed_cams"]
        rows.append(row)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)

    summary = {}
    for k in rows[0]:
        if k == "batch" or k.count(":") > 1:
            continue
        g = np.array([(r[k] - r["exact"]) / r["exact"] * 100 for r in rows])
        summary[k] = {"mean_gap_pct": float(g.mean()), "median_gap_pct": float(np.median(g)), "max_gap_pct": float(g.max()),
                      "optimal": int(np.sum(g < 1e-6)),
                      "beats_greedy": int(sum(r[k] < r["greedy"] - 1e-9 for r in rows)),
                      "ties_or_beats_regret": int(sum(r[k] <= r["regret"] + 1e-9 for r in rows))}
        if k + ":time" in rows[0]:
            summary[k].update({"mean_time_s": float(np.mean([r[k + ":time"] for r in rows])),
                               "mean_vars": float(np.mean([r[k + ":vars"] for r in rows])),
                               "mean_fixed": float(np.mean([r[k + ":fixed"] for r in rows])),
                               "mean_viol_servers": float(np.mean([r[k + ":viol_servers"] for r in rows]))})
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"v3_{args.window}_{('u%g' % args.target_util) if args.target_util else ('s%g' % args.capacity_scale)}_seed{args.instance_seed}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    json.dump({"args": vars(args), "summary": summary, "rows": rows}, open(path, "w"), indent=1)
    for k, v in summary.items():
        extra = (f" t {v['mean_time_s']:.2f}s vars {v['mean_vars']:.0f} fixed {v['mean_fixed']:.0f} viol {v['mean_viol_servers']:.2f}"
                 if "mean_time_s" in v else "")
        print(f"{k:16s} gap {v['mean_gap_pct']:7.3f}% med {v['median_gap_pct']:6.3f}% max {v['max_gap_pct']:7.2f}% "
              f"opt {v['optimal']:2d}/{len(rows)} >greedy {v['beats_greedy']:2d} <=regret {v['ties_or_beats_regret']:2d}{extra}")
    print("saved", path)


if __name__ == "__main__":
    main()
