import glob, json, collections, sys
import numpy as np
from scipy.stats import wilcoxon, rankdata
root = sys.argv[1] if len(sys.argv) > 1 else "results/v3/campaign3/v31"
c2 = sys.argv[2] if len(sys.argv) > 2 else "results/v3/campaign2"
R = collections.defaultdict(dict)
skipped = 0
for f in glob.glob(root + "/*.json"):
    d = json.load(open(f))
    if "reprice_top" not in d: skipped += 1; continue   # interrupted run with the slow pre-release code
    m = d["method"] + ("+LS" if d.get("polish") else "")
    R[(d["target_util"], d["seed"])][m] = d
S = collections.defaultdict(dict)
for f in glob.glob(c2 + "/priced/*.json"):
    d = json.load(open(f)); S[(d["target_util"], d["seed"])][d["method"]] = d
LB = {}
for f in glob.glob(c2 + "/global/*.json"):
    for d in json.load(open(f)): LB[(d["target_util"], d["seed"])] = d
def holm(ps):
    idx = np.argsort(ps); adj = np.empty(len(ps)); run = 0
    for r, i in enumerate(idx): run = max(run, (len(ps) - r) * ps[i]); adj[i] = min(1, run)
    return adj
def rb(d):
    d = d[np.abs(d) > 1e-9]
    if not len(d): return 0.0
    r = rankdata(np.abs(d)); return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())
M = ["greedy", "regret", "exact", "ils", "sa_v3", "sqa_v3", "sqa_v3+LS"]
out = {"skipped": skipped, "tables": []}
lines = [f"# Campaign 3 (QUBO-v3.1, 20,000 x 800, 10 seeds)  [skipped stale files: {skipped}]"]
for u in [75.0, 90.0, 95.0, 98.0]:
    keys = sorted(k for k in R if k[0] == u)
    lines.append(f"\n## {u:g}% utilisation, seeds {len(keys)}\n")
    lines.append("| Method | Objective mean ± sd | Gap to LB | Uncovered | Time, s | v3 static (same method) |")
    lines.append("|---|---|---|---|---|---|")
    tab = {"util": u, "rows": [], "tests": {}}
    for m in M:
        ks = [k for k in keys if m in R[k]]
        o = np.array([R[k][m]["objective"] for k in ks]); g = np.array([(R[k][m]["objective"] - LB[k]["lagrangian_lower_bound"]) / LB[k]["lagrangian_lower_bound"] * 100 for k in ks])
        unc = sum(R[k][m]["n_cameras"] - R[k][m]["covered"] for k in ks); t = np.mean([R[k][m]["time_sec"] for k in ks])
        st = [S[k][m]["objective"] for k in ks if m in S[k]]
        stx = f"{np.mean(st):,.1f}" if len(st) == len(ks) and m in ("greedy", "regret", "exact") else "–"
        lines.append(f"| {m} | {o.mean():,.1f} ± {o.std(ddof=1):,.1f} | {g.mean():.2f}% | {unc} | {t:.0f} | {stx} |")
        tab["rows"].append({"method": m, "n": len(ks), "mean": o.mean(), "sd": o.std(ddof=1), "gap": g.mean(), "uncovered": int(unc), "time": t, "static_mean": float(np.mean(st)) if stx != "–" else None})
    glob_o = np.mean([LB[k]["feasible_objective"] for k in keys]); glob_g = np.mean([(LB[k]["feasible_objective"] - LB[k]["lagrangian_lower_bound"]) / LB[k]["lagrangian_lower_bound"] * 100 for k in keys])
    lines.append(f"| offline global | {glob_o:,.1f} | {glob_g:.2f}% | 0 | | |")
    for ref in ["sqa_v3", "sqa_v3+LS"]:
        lines.append(f"\n{ref} vs | mean diff | better/tie/worse | Wilcoxon p | Holm p | r_rb")
        lines.append("|---|---|---|---|---|---|")
        comps = [m for m in M if m != ref]; ds, ps = [], []
        for m in comps:
            d = np.array([R[k][ref]["objective"] - R[k][m]["objective"] for k in keys])
            p = wilcoxon(d).pvalue if np.any(np.abs(d) > 1e-9) else 1.0
            ds.append(d); ps.append(p)
        adj = holm(np.array(ps))
        for m, d, p, pa in zip(comps, ds, ps, adj):
            b, w = int((d < -1e-6).sum()), int((d > 1e-6).sum())
            lines.append(f"| {m} | {d.mean():+.2f} | {b}/{len(d)-b-w}/{w} | {p:.3g} | {pa:.3g} | {rb(d):+.2f} |")
            tab["tests"][f"{ref}|{m}"] = {"diff": float(d.mean()), "better": b, "worse": w, "p": float(p), "p_holm": float(pa), "r": rb(d)}
    out["tables"].append(tab)
txt = "\n".join(lines); print(txt)
open("c3_summary.md", "w").write(txt); json.dump(out, open("c3_summary.json", "w"), indent=1, default=float)
