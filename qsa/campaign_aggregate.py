"""Aggregate the multi-seed campaign (v3_trajectory + global_bound_check outputs)."""
import glob
import json
import os
import sys

import numpy as np
from scipy.stats import wilcoxon

root = sys.argv[1] if len(sys.argv) > 1 else "campaign"
rows = []
for arch in ("plain", "priced"):
    for f in glob.glob(os.path.join(root, arch, "*.json")):
        d = json.load(open(f))
        rows.append({"arch": arch, "method": d["method"], "scale": round(d["capacity_scale"], 2), "seed": d["seed"],
                     "util": d["utilization_percent"], "obj": d["objective"], "covered": d["covered"],
                     "time": d["time_sec"]})
bounds = {}
for f in glob.glob(os.path.join(root, "global", "*.json")):
    for d in json.load(open(f)):
        bounds[(round(d["capacity_scale"], 2), d["seed"])] = d

methods = ["greedy", "regret", "exact", "sa_v3", "sqa_v3"]
out = {"table": [], "paired": []}
lines = []
for arch in ("plain", "priced"):
    for scale in (1.0, 0.33, 0.25):
        sub = [r for r in rows if r["arch"] == arch and r["scale"] == scale]
        seeds = sorted({r["seed"] for r in sub if r["method"] == "sqa_v3" and r["util"] < 100.0})
        excluded = sorted({r["seed"] for r in sub if r["util"] >= 100.0})
        if not seeds:
            continue
        by = {(r["method"], r["seed"]): r for r in sub}
        util = np.mean([by[("sqa_v3", s)]["util"] for s in seeds])
        lines.append(f"\n### {arch}, capacity scale {scale} (utilisation ~{util:.2f}%), seeds n={len(seeds)}"
                     + (f"; excluded (utilisation >= 100%, infeasible): {excluded}" if excluded else "") + "\n")
        lines.append("| Method | Objective mean ± sd | Gap to global LB, mean | Uncovered, total | Time, s mean |")
        lines.append("|---|---|---|---|---|")
        for m in methods + ["global_feasible"]:
            if m == "global_feasible":
                vals = [bounds[(scale, s)]["feasible_objective"] for s in seeds if (scale, s) in bounds]
                gaps = [(bounds[(scale, s)]["feasible_objective"] - bounds[(scale, s)]["lagrangian_lower_bound"])
                        / bounds[(scale, s)]["lagrangian_lower_bound"] * 100 for s in seeds if (scale, s) in bounds]
                times = [bounds[(scale, s)]["time_sec"] for s in seeds if (scale, s) in bounds]
                unc = sum(20000 - bounds[(scale, s)]["covered"] for s in seeds if (scale, s) in bounds)
            else:
                ss = [s for s in seeds if (m, s) in by]
                vals = [by[(m, s)]["obj"] for s in ss]
                gaps = [(by[(m, s)]["obj"] - bounds[(scale, s)]["lagrangian_lower_bound"])
                        / bounds[(scale, s)]["lagrangian_lower_bound"] * 100 for s in ss if (scale, s) in bounds]
                times = [by[(m, s)]["time"] for s in ss]
                unc = sum(20000 - by[(m, s)]["covered"] for s in ss)
            if not vals:
                continue
            lines.append(f"| {m} | {np.mean(vals):,.1f} ± {np.std(vals, ddof=1) if len(vals) > 1 else 0:,.1f} | "
                         f"{np.mean(gaps) if gaps else float('nan'):.2f}% | {unc} | {np.mean(times):.1f} |")
            out["table"].append({"arch": arch, "scale": scale, "method": m, "n": len(vals), "mean": float(np.mean(vals)),
                                 "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                                 "gap_lb_pct": float(np.mean(gaps)) if gaps else None, "uncovered": int(unc),
                                 "time": float(np.mean(times))})
        lines.append("")
        lines.append("| SQA-v3 vs | mean diff (SQA − other) | SQA better / tie / worse | Wilcoxon p |")
        lines.append("|---|---|---|---|")
        for m in ["greedy", "regret", "exact", "sa_v3"]:
            ss = [s for s in seeds if (m, s) in by]
            d = np.array([by[("sqa_v3", s)]["obj"] - by[(m, s)]["obj"] for s in ss])
            better, tie, worse = int(np.sum(d < -1e-6)), int(np.sum(np.abs(d) <= 1e-6)), int(np.sum(d > 1e-6))
            try:
                p = wilcoxon(d).pvalue if np.any(np.abs(d) > 1e-6) and len(d) >= 5 else float("nan")
            except ValueError:
                p = float("nan")
            lines.append(f"| {m} | {d.mean():+.2f} | {better} / {tie} / {worse} | {p:.3g} |")
            out["paired"].append({"arch": arch, "scale": scale, "vs": m, "n": len(d), "mean_diff": float(d.mean()),
                                  "better": better, "tie": tie, "worse": worse, "p": float(p)})
text = "\n".join(lines)
print(text)
json.dump(out, open(os.path.join(root, "campaign_summary.json"), "w"), indent=1)
open(os.path.join(root, "campaign_summary.md"), "w").write(text)
