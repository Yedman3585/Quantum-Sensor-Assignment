import glob, json, collections, sys
import numpy as np
from scipy.stats import wilcoxon
W = sys.argv[1]; LBF = sys.argv[2:]
LB = {}
for f in LBF:
    for d in json.load(open(f)): LB[(d["target_util"], d["seed"])] = d["lb"]
def cond(d):
    if not d.get("reprice"): return "static prices"
    return "oracle" if d.get("forecast", "oracle") == "oracle" else "twin"
R = collections.defaultdict(dict); dup = []
for f in glob.glob(W + "/*.json"):
    d = json.load(open(f)); k = (d["method"], cond(d), d["target_util"])
    if d["seed"] in R[k]: dup.append(abs(R[k][d["seed"]][0] - d["objective"]))
    R[k][d["seed"]] = (d["objective"], d["n_cameras"] - d["covered"])
L = ["# Priority-weight check: w_i = p_i (instead of 4 - p_i), 20,000 x 800, seeds 42-51", "",
     f"Duplicate runs (PC vs Mac) compared: {len(dup)}, max |objective difference| = {max(dup) if dup else 0:.2e}", "",
     "Mean gap to the Lagrangian LB computed with w_i = p_i (%); [uncovered]; n = seeds", ""]
for m in ["exact", "greedy"]:
    L += [f"## {m}", "", "| re-pricing | 90% | 95% | 98% |", "|---|---|---|---|"]
    for c in ["static prices", "oracle", "twin"]:
        cells = []
        for u in [90.0, 95.0, 98.0]:
            g = R[(m, c, u)]; ss = sorted(s for s in g if (u, s) in LB)
            gg = [(g[s][0] - LB[(u, s)]) / LB[(u, s)] * 100 for s in ss]; unc = sum(g[s][1] for s in ss)
            cells.append(f"{np.mean(gg):.2f}" + (f" [{unc}]" if unc else "") + (f" n={len(ss)}" if len(ss) != 10 else ""))
        L.append(f"| {c} | " + " | ".join(cells) + " |")
    L.append("")
txt = "\n".join(L); print(txt); open("weight_summary.md", "w").write(txt + "\n")
