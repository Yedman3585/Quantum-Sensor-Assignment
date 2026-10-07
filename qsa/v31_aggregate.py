import glob, json, collections
import numpy as np
from scipy.stats import wilcoxon
rows = collections.defaultdict(dict)
def add(f, tag):
    d = json.load(open(f)); n = d["method"] + ("+LS" if d.get("polish") else "") + tag
    rows[(d["target_util"], d["seed"])][n] = (d["objective"], d["n_cameras"] - d["covered"], d["time_sec"])
for f in glob.glob("results/v3/v31_prototype/r4/*.json"):
    # in r4, SQA/SA/ILS ranked their candidates by unpriced cost; fixed and re-run in r5
    if json.load(open(f))["method"] in ("exact", "greedy", "regret"): add(f, " v3.1")
for f in glob.glob("results/v3/v31_prototype/r5/*.json"): add(f, " v3.1")
for f in glob.glob("results/v3/v31_prototype/r4_static/*.json"): add(f, " v3")
for f in glob.glob("results/v3/campaign2/scale/*5000x200*.json"):
    d = json.load(open(f))
    if d["method"] in ("sqa_v3", "sa_v3", "ils"): add(f, " v3")
lb = {}
for f in glob.glob("results/v3/campaign2/scale_global/*5000x200*.json") + glob.glob("results/v3/v31_prototype/glob98/*.json"):
    for d in json.load(open(f)): lb[(d["target_util"], d["seed"])] = d["lagrangian_lower_bound"]
order = ["greedy v3", "regret v3", "exact v3", "ils v3", "sa_v3 v3", "sqa_v3 v3", "greedy v3.1", "regret v3.1", "exact v3.1", "ils v3.1", "sa_v3 v3.1", "sqa_v3 v3.1", "sqa_v3+LS v3.1"]
for u in [95.0, 98.0]:
    keys = sorted(k for k in rows if k[0] == u)
    print(f"\n== util {u:g}%  seeds {[k[1] for k in keys]}")
    for n in order:
        ks = [k for k in keys if n in rows[k]]
        if not ks: continue
        o = np.array([rows[k][n][0] for k in ks]); unc = sum(rows[k][n][1] for k in ks)
        g = [ (rows[k][n][0]-lb[k])/lb[k]*100 for k in ks if k in lb]
        line = f"{n:16s} n={len(ks)} mean {o.mean():8.1f} gapLB {np.mean(g) if g else float('nan'):5.2f}%  uncov {unc:3d}  t {np.mean([rows[k][n][2] for k in ks]):6.1f}s"
        ref = "sqa_v3 v3.1"
        kk = [k for k in ks if ref in rows[k]]
        if n != ref and kk:
            d = np.array([rows[k][ref][0] - rows[k][n][0] for k in kk])
            p = wilcoxon(d).pvalue if len(d) >= 5 and np.any(d != 0) else float('nan')
            line += f" | SQA-v3.1 - this {d.mean():+7.1f} ({int((d<0).sum())}/{len(d)} better, p={p:.3g})"
        print(line)
