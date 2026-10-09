import glob, json, collections, sys
import numpy as np
from scipy.stats import wilcoxon, rankdata
FC = sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads/Quantum-Sensor-Assignment/results/v3/forecast"
C3 = sys.argv[2] if len(sys.argv) > 2 else "/home/claude/work/c3/x/results/v3/campaign3/v31"
C2 = sys.argv[3] if len(sys.argv) > 3 else "/mnt/user-data/uploads/Quantum-Sensor-Assignment/results/v3/campaign2"
LB = {}
for f in glob.glob(C2 + "/global/*.json"):
    for d in json.load(open(f)): LB[(d["target_util"], d["seed"])] = d["lagrangian_lower_bound"]
def cond(d):
    if not d.get("reprice"): return "static prices"
    f, l, b = d.get("forecast", "oracle"), d.get("forecast_level", 0), d.get("forecast_bias", 0)
    if f == "oracle" and b == 0: return "oracle"
    if f == "twin": return "twin" + ("" if b == 0 else f" {b:+.0%}")
    if f == "noise": return f"noise ±{l:.0%}"
    return f"bias {b:+.0%}"
R = collections.defaultdict(dict)   # (method, cond, util) -> seed -> objective
for root in (FC, C3):
    for f in glob.glob(root + "/*.json"):
        d = json.load(open(f))
        if root == C3 and "reprice_top" not in d: continue
        m = d["method"] + ("+LS" if d.get("polish") else "")
        R[(m, cond(d), d["target_util"])][d["seed"]] = (d["objective"], d["n_cameras"] - d["covered"], d["time_sec"])
def gap(o, u, s): return (o - LB[(u, s)]) / LB[(u, s)] * 100
def holm(ps):
    idx = np.argsort(ps); adj = np.empty(len(ps)); run = 0
    for r, i in enumerate(idx): run = max(run, (len(ps) - r) * ps[i]); adj[i] = min(1, run)
    return adj
U = [90.0, 95.0, 98.0]
C = ["static prices", "oracle", "noise ±30%", "bias -10%", "bias +10%", "twin", "twin +10%"]
L = ["# Forecast experiment, 20,000 x 800, 10 seeds (42-51), exact batch solver",
     "", "Mean gap to Lagrangian LB (%); [uncovered, sum]; Δ vs oracle in pp (wins/losses vs oracle; Holm p within column)", ""]
L.append("| information for re-pricing | " + " | ".join(f"{u:g}%" for u in U) + " |"); L.append("|---" * (len(U) + 1) + "|")
cells = {c: [] for c in C}
for u in U:
    o = R[("exact", "oracle", u)]; ps, rows = [], []
    for c in C:
        g = R[("exact", c, u)]; ss = sorted(set(g) & set(o))
        gg = np.array([gap(g[s][0], u, s) for s in ss]); og = np.array([gap(o[s][0], u, s) for s in ss])
        unc = sum(g[s][1] for s in ss); d = gg - og
        p = wilcoxon(gg, og).pvalue if c != "oracle" and np.any(np.abs(d) > 1e-9) else np.nan
        rows.append((c, gg.mean(), unc, d, len(ss), p)); 
        if c != "oracle": ps.append(p)
    adj = iter(holm(np.array(ps)))
    for c, m, unc, d, n, p in rows:
        s = f"{m:.2f}" + (f" [{unc}]" if unc else "") + (f" n={n}" if n != 10 else "")
        if c != "oracle":
            s += f" ({d.mean():+.2f}; {(d < -1e-9).sum()}/{(d > 1e-9).sum()}; p={next(adj):.3f})"
        cells[c].append(s)
for c in C: L.append(f"| {c} | " + " | ".join(cells[c]) + " |")
L += ["", "## SQA-v3.1+LS with the twin +10% forecast (95/98%)", "",
      "| util | SQA+LS twin+10% | exact twin+10% | SQA+LS oracle (campaign 3) | SQA+LS twin+10% vs exact twin+10% (W/L, p) | time SQA+LS (s) |", "|---|---|---|---|---|---|"]
for u in [95.0, 98.0]:
    a = R[("sqa_v3+LS", "twin +10%", u)]; b = R[("exact", "twin +10%", u)]; c = R[("sqa_v3+LS", "oracle", u)]
    ss = sorted(set(a) & set(b) & set(c))
    ga = np.array([gap(a[s][0], u, s) for s in ss]); gb = np.array([gap(b[s][0], u, s) for s in ss]); gc = np.array([gap(c[s][0], u, s) for s in ss])
    d = ga - gb; p = wilcoxon(ga, gb).pvalue
    L.append(f"| {u:g}% | {ga.mean():.2f} | {gb.mean():.2f} | {gc.mean():.2f} | {(d<0).sum()}/{(d>0).sum()}, p={p:.3f} | {np.mean([a[s][2] for s in ss]):.0f} |")
L += ["", "Run time (s, mean, 4 parallel jobs): " + ", ".join(f"{c} {np.mean([v[2] for v in R[('exact', c, 95.0)].values()]):.0f}" for c in C)]
txt = "\n".join(L); print(txt); open("fc20k_summary.md", "w").write(txt + "\n")
