"""Aggregate campaign 2: target utilisation, equal-time ILS, shared-window baseline, ablation, scaling.

Paired Wilcoxon tests of SQA-v3 against each method, Holm-corrected within each table,
with the matched-pairs rank-biserial correlation as effect size.
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

root = sys.argv[1] if len(sys.argv) > 1 else "c2"
METHODS = ["greedy", "regret", "exact", "ils", "sa_v3", "sqa_v3"]
NAMES = {"greedy": "greedy", "regret": "regret", "exact": "exact batch MILP", "ils": "ILS (equal time)",
         "sa_v3": "SA-v3", "sqa_v3": "SQA-v3"}


def load_traj(sub):
    out = defaultdict(dict)
    for f in glob.glob(os.path.join(root, sub, "*.json")):
        d = json.load(open(f))
        key = (d.get("n_cameras", 20000), d.get("target_util"), d["seed"])
        out[key][d["method"]] = d
    return out


def load_bounds(sub):
    out = {}
    for f in glob.glob(os.path.join(root, sub, "*.json")):
        for d in json.load(open(f)):
            out[(d.get("n_cameras", 20000), d.get("target_util"), d["seed"])] = d
    return out


def holm(ps):
    idx = np.argsort(ps)
    adj = np.empty(len(ps))
    running = 0.0
    for r, i in enumerate(idx):
        running = max(running, (len(ps) - r) * ps[i])
        adj[i] = min(1.0, running)
    return adj


def rank_biserial(d):
    d = d[np.abs(d) > 1e-9]
    if len(d) == 0:
        return 0.0
    from scipy.stats import rankdata
    r = rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def table(runs, bounds, n_cam, util, lines, summary, label):
    keys = sorted(k for k in runs if k[0] == n_cam and k[1] == util and "sqa_v3" in runs[k])
    if not keys:
        return
    lines.append(f"\n### {label}: {n_cam:,} cameras, utilisation {util:g}%, seeds n={len(keys)}\n")
    lines.append("| Method | Objective mean ± sd | Gap to lower bound | Uncovered (total) | Time, s |")
    lines.append("|---|---|---|---|---|")
    for m in METHODS + ["global"]:
        if m == "global":
            ks = [k for k in keys if k in bounds]
            if not ks:
                continue
            obj = [bounds[k]["feasible_objective"] for k in ks]
            gap = [(bounds[k]["feasible_objective"] - bounds[k]["lagrangian_lower_bound"]) / bounds[k]["lagrangian_lower_bound"] * 100 for k in ks]
            unc = sum(n_cam - bounds[k]["covered"] for k in ks)
            tim = [bounds[k]["time_sec"] for k in ks]
            name = "global solution (offline)"
        else:
            ks = [k for k in keys if m in runs[k]]
            if not ks:
                continue
            obj = [runs[k][m]["objective"] for k in ks]
            gap = [(runs[k][m]["objective"] - bounds[k]["lagrangian_lower_bound"]) / bounds[k]["lagrangian_lower_bound"] * 100
                   for k in ks if k in bounds]
            unc = sum(n_cam - runs[k][m]["covered"] for k in ks)
            tim = [runs[k][m]["time_sec"] for k in ks]
            name = NAMES[m]
        sd = np.std(obj, ddof=1) if len(obj) > 1 else 0.0
        lines.append(f"| {name} | {np.mean(obj):,.1f} ± {sd:,.1f} | {np.mean(gap):.2f}% | {unc} | {np.mean(tim):.1f} |")
        summary.append({"table": label, "n_cameras": n_cam, "util": util, "method": m, "n": len(obj),
                        "mean": float(np.mean(obj)), "sd": float(sd), "gap_pct": float(np.mean(gap)) if gap else None,
                        "uncovered": int(unc), "time": float(np.mean(tim))})
    comps, ps = [], []
    for m in ["greedy", "regret", "exact", "ils", "sa_v3"]:
        ks = [k for k in keys if m in runs[k]]
        if len(ks) < 2:
            continue
        d = np.array([runs[k]["sqa_v3"]["objective"] - runs[k][m]["objective"] for k in ks])
        try:
            p = wilcoxon(d).pvalue if np.any(np.abs(d) > 1e-9) and len(d) >= 5 else 1.0
        except ValueError:
            p = 1.0
        comps.append((m, d)); ps.append(p)
    if comps:
        adj = holm(np.array(ps))
        lines.append("")
        lines.append("| SQA-v3 vs | mean diff | better / tie / worse | Wilcoxon p | Holm p | rank-biserial r |")
        lines.append("|---|---|---|---|---|---|")
        for (m, d), p, pa in zip(comps, ps, adj):
            b, t, w = int(np.sum(d < -1e-6)), int(np.sum(np.abs(d) <= 1e-6)), int(np.sum(d > 1e-6))
            lines.append(f"| {NAMES[m]} | {d.mean():+.2f} | {b} / {t} / {w} | {p:.3g} | {pa:.3g} | {rank_biserial(d):+.2f} |")
            summary.append({"table": label, "n_cameras": n_cam, "util": util, "vs": m, "mean_diff": float(d.mean()),
                            "better": b, "tie": t, "worse": w, "p": float(p), "p_holm": float(pa),
                            "r_rb": rank_biserial(d)})


def main():
    lines, summary = ["# Campaign 2 summary", "",
                      "Paired tests: SQA-v3 objective minus method (negative = SQA-v3 better); Holm correction within each table; "
                      "rank-biserial r < 0 favours SQA-v3."], []
    priced, shared = load_traj("priced"), load_traj("shared")
    bounds = load_bounds("global")
    for u in [75, 90, 95, 98]:
        table(priced, bounds, 20000, u, lines, summary, "QUBO-v3 architecture (per-camera window + prices)")
    # shared-window baseline (no prices)
    lines.append("\n## Shared 80×20 window, no prices (first-submission architecture)\n")
    lines.append("| Utilisation | exact batch MILP | greedy | regret | gap of exact to lower bound |")
    lines.append("|---|---|---|---|---|")
    for u in [75, 90, 95, 98]:
        ks = sorted(k for k in shared if k[1] == u)
        if not ks:
            continue
        row = []
        for m in ["exact", "greedy", "regret"]:
            row.append(np.mean([shared[k][m]["objective"] for k in ks if m in shared[k]]))
        gap = np.mean([(shared[k]["exact"]["objective"] - bounds[k]["lagrangian_lower_bound"]) / bounds[k]["lagrangian_lower_bound"] * 100
                       for k in ks if "exact" in shared[k] and k in bounds])
        lines.append(f"| {u}% | {row[0]:,.1f} | {row[1]:,.1f} | {row[2]:,.1f} | {gap:.1f}% |")
        summary.append({"table": "shared", "util": u, "exact": row[0], "greedy": row[1], "regret": row[2], "gap_exact_pct": gap})
    # scaling
    scale, sbounds = load_traj("scale"), load_bounds("scale_global")
    sb = dict(bounds); sb.update(sbounds)
    for n in [5000, 50000]:
        table(scale, sb, n, 95, lines, summary, "Scaling")
    # ablation
    lines.append("\n## Ablation on hard batches (95% utilisation, per-camera window)\n")
    abl = defaultdict(list)
    for sub in ["ablation", "ablation_adaptive", "ablation_beta20"]:
        for f in glob.glob(os.path.join(root, sub, "*.json")):
            d = json.load(open(f))
            for r in d["rows"]:
                for k, v in r.items():
                    if k in ("batch",) or k.count(":") > 1 or not isinstance(v, (int, float)):
                        continue
                    name = k if sub == "ablation" else f"{k} [{sub.split('_', 1)[1]}]"
                    if k in ("exact", "greedy", "regret", "regret_ls") and sub != "ablation":
                        continue
                    abl[name].append((v - r["exact"]) / r["exact"] * 100)
    lines.append("| Variant | batches | mean gap to batch optimum | median | optimal batches |")
    lines.append("|---|---|---|---|---|")
    for k, g in abl.items():
        g = np.array(g)
        lines.append(f"| {k} | {len(g)} | {g.mean():.3f}% | {np.median(g):.3f}% | {int(np.sum(g < 1e-6))} |")
        summary.append({"table": "ablation", "variant": k, "n": len(g), "mean_gap": float(g.mean()), "median_gap": float(np.median(g)),
                        "optimal": int(np.sum(g < 1e-6))})
    text = "\n".join(lines)
    open(os.path.join(root, "c2_summary.md"), "w").write(text)
    json.dump(summary, open(os.path.join(root, "c2_summary.json"), "w"), indent=1)
    print(text)


if __name__ == "__main__":
    main()
