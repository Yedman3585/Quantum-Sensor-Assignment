"""Offline reference solution with the full assignment (for the service metrics).

Same algorithm as the offline global solution used in the paper: Lagrangian prices by
subgradient (400 iterations) on the whole instance, regret-ordered priced greedy, then
shift local search on the true weighted cost. The offline solver sees all cameras at once,
so it is a reference, not an online policy.

Usage: python qsa/offline_global.py OUT_DIR [--weight prio] SEED UTIL [SEED UTIL ...]
Writes OUT_DIR/offline[_wprio]_u{UTIL}_seed{SEED}.json with lb, objective and assign (server or -1).
"""
import json, os, sys, time
import numpy as np
from batch_quality_benchmark import Instance

P = 15.0  # penalty per uncovered camera


def solve(seed, util, n=20000, m=800, iters=400, layout="uniform", weight="inv"):
    inst = Instance(n, m, seed, 1.0, util, layout=layout)
    if weight == "prio":
        inst.w = inst.priority.astype(float); inst.wcost = inst.cost * inst.w[:, None]
    C, l, K = inst.wcost, inst.load, inst.cap
    u = np.zeros(len(K)); best, bu = -np.inf, u.copy()
    for it in range(iters):
        R = C + np.outer(l, u); j = R.argmin(1); r = R[np.arange(len(l)), j]; take = r < P
        val = np.where(take, r, P).sum() - u @ K
        if val > best: best, bu = val, u.copy()
        g = np.bincount(j[take], weights=l[take], minlength=len(K)) - K
        u = np.maximum(0, u + g / (np.linalg.norm(g) + 1e-9) * np.sqrt(len(K)) * 0.05 / (1 + it / 50) / np.mean(l) * 0.5)
    R = C + np.outer(l, bu); part = np.partition(R, 1, axis=1); regret = part[:, 1] - part[:, 0]
    rem = K.copy(); a = -np.ones(len(l), int)
    for i in np.argsort(-regret):
        feas = rem >= l[i]
        if not feas.any(): continue
        jj = int(np.where(feas, R[i], np.inf).argmin()); a[i] = jj; rem[jj] -= l[i]
    for _ in range(10):
        moved = 0
        for i in range(len(l)):
            if a[i] < 0: continue
            cand = np.where(rem >= l[i], C[i], np.inf); jj = int(cand.argmin())
            if cand[jj] < C[i, a[i]] - 1e-12:
                rem[a[i]] += l[i]; rem[jj] -= l[i]; a[i] = jj; moved += 1
        if moved == 0: break
    cost = np.where(a >= 0, C[np.arange(len(l)), np.maximum(a, 0)], P)
    return {"seed": seed, "util": util, "layout": layout, "weight": weight, "lb": float(best), "objective": float(cost.sum()),
            "covered": int((a >= 0).sum()), "assign": a.tolist()}


if __name__ == "__main__":
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    args = sys.argv[2:]; weight = "inv"
    if args[:1] == ["--weight"]:
        weight, args = args[1], args[2:]
    for s, u in zip(args[::2], args[1::2]):
        t = time.time(); r = solve(int(s), float(u), weight=weight); r["time_sec"] = time.time() - t
        tag = "_wprio" if weight == "prio" else ""
        with open(os.path.join(out, "offline%s_u%g_seed%d.json" % (tag, float(u), int(s))), "w") as f:
            json.dump(r, f)
        print(s, u, "lb %.2f offline %.2f covered %d  %.0fs" % (r["lb"], r["objective"], r["covered"], r["time_sec"]), flush=True)
