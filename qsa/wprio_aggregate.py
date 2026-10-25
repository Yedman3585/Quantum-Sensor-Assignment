"""Tables for the priority-weight robustness study (w_i = p_i), all blocks.

Usage: python qsa/wprio_aggregate.py [results/v3] > results/v3/weight_prio_summary/wprio_full_summary.md
Gap = (objective - LB) / LB, LB = Lagrangian bound with w_i = p_i of the same instance (lb_prio_*.json).
Paired two-sided Wilcoxon over seeds; Holm correction within each column of a table; W/L = wins/losses of the row.
"""
import collections, glob, json, os, sys
import numpy as np
from scipy.stats import wilcoxon

V = sys.argv[1] if len(sys.argv) > 1 else "results/v3"
S = f"{V}/weight_prio_summary"
LB = {}
for f, lay in [("lb_prio_20k.json", "uniform"), ("lb_prio_75.json", "uniform"), ("lb_prio_val_100_109.json", "uniform"), ("lb_prio_clustered.json", "clustered")]:
    for d in json.load(open(f"{S}/{f}")):
        LB[(d.get("layout", lay), float(d["target_util"]), d["seed"])] = d["lb"]


def load(sub):
    out = {}
    for f in sorted(glob.glob(f"{V}/{sub}/*.json")):
        d = json.load(open(f))
        if d.get("weight", "inv") != "prio" or "objective" not in d or "target_util" not in d: continue
        d.setdefault("layout", "uniform")
        lb = LB.get((d["layout"], float(d["target_util"]), d["seed"]))
        if lb is None: continue
        d["gap"] = (d["objective"] - lb) / lb * 100
        out[key(d) + (float(d["target_util"]), d["seed"])] = d          # later file wins (duplicates are identical)
    return out


def key(d):
    if d.get("policy"):
        b = d.get("forecast_bias") or 0
        return (d["policy"] + ("+5%" if b else ""), "exact")
    pol = "static" if not d.get("reprice") else ("statistics" if d.get("forecast") == "twin" else "re-pricing")
    b = d.get("forecast_bias") or 0
    if b: pol += " +%g%%" % round(b * 100)
    m = d["method"] + ("+LS" if d.get("polish") else "")
    return (pol, m)


def cell(R, pol, m, u, ref=None):
    xs = {k[3]: v for k, v in R.items() if k[0] == pol and k[1] == m and k[2] == u}
    if not xs: return None, "–"
    g = np.mean([v["gap"] for v in xs.values()]); unc = sum(20000 - v["covered"] for v in xs.values() if v.get("n_cameras", 20000) == 20000)
    s = f"{g:.2f}" + (f" [{unc} unc.]" if unc else "")
    if ref is None: return None, s + f" (n={len(xs)})" if len(xs) != 10 else s
    rs = {k[3]: v for k, v in R.items() if (k[0], k[1], k[2]) == ref}
    common = sorted(set(xs) & set(rs))
    if len(common) < 3: return None, s
    dlt = np.array([xs[c]["objective"] - rs[c]["objective"] for c in common])
    p = wilcoxon(dlt).pvalue if np.any(dlt != 0) else 1.0
    return p, (s, np.mean(dlt), int((dlt < 0).sum()), int((dlt > 0).sum()), len(common))


def table(R, rows, utils, title, note=""):
    print(f"\n## {title}\n"); note and print(note + "\n")
    print("| Policy | Solver | " + " | ".join(f"{u:g}%" for u in utils) + " |"); print("|---|---|" + "---|" * len(utils))
    cells, ps = [], []
    for pol, m, ref in rows:
        line = []
        for u in utils:
            p, c = cell(R, pol, m, u, None if ref is None else (ref[0], ref[1], u))
            line.append((p, c));
            if p is not None: ps.append(p)
        cells.append((pol, m, line))
    # Holm within each column (utilisation), as in the other summaries
    adj = {}
    for j in range(len(utils)):
        col = [(i, line[j][0]) for i, (_, _, line) in enumerate(cells) if line[j][0] is not None]
        order = sorted(col, key=lambda t: t[1]); run = 0
        for r, (i, p) in enumerate(order):
            run = max(run, min(1.0, p * (len(order) - r))); adj[(i, j)] = run
    for i, (pol, m, line) in enumerate(cells):
        out = []
        for j, (p, c) in enumerate(line):
            if p is None: out.append(c); continue
            s, dm, w, l, n = c; out.append(f"{s} ({dm:+.1f}; {w}/{l}; p={adj[(i, j)]:.3f})")
        print(f"| {pol} | {m} | " + " | ".join(out) + " |")


U4, U3 = [75.0, 90.0, 95.0, 98.0], [90.0, 95.0, 98.0]
print("# Priority weight w_i = p_i: full robustness study (20,000 x 800)\n")
print("Gap to the Lagrangian lower bound (%) computed with w_i = p_i; [uncovered cameras, sum over seeds];")
print("in brackets: mean objective difference to the reference row, wins/losses of the row, Wilcoxon p (Holm within column).")

main = load("weight_prio")
table(main, [("static", "exact", None), ("re-pricing", "exact", None), ("re-pricing +5%", "exact", ("re-pricing", "exact")),
             ("statistics", "exact", None), ("statistics +5%", "exact", ("statistics", "exact")),
             ("static", "greedy", None), ("re-pricing", "greedy", None), ("re-pricing +5%", "greedy", ("re-pricing", "greedy")),
             ("statistics +5%", "greedy", ("statistics", "greedy"))], U4,
      "1. Main table (test seeds 42–51)", "Reference for the margin rows: the same information without margin.")

tun = load("weight_prio_tuning")
print("\n## 2. Margin tuning (validation seeds 100–109, exact batches)\n")
print("| Information, utilisation | " + " | ".join(f"+{b}%" for b in (0, 5, 10, 15, 20, 30)) + " | best |"); print("|---|" + "---|" * 7)
for inf in ("re-pricing", "statistics"):
    for u in U3:
        ms = []
        for b in (0, 5, 10, 15, 20, 30):
            pol = inf + (f" +{b}%" if b else "")
            v = [d["gap"] for k, d in tun.items() if k[0] == pol and k[2] == u]
            ms.append(np.mean(v) if v else np.nan)
        best = (0, 5, 10, 15, 20, 30)[int(np.nanargmin(ms))]
        print(f"| {'exact demand' if inf == 're-pricing' else 'demand statistics'}, {u:g}% | " + " | ".join(f"**{m:.2f}**" if b == best else f"{m:.2f}" for m, b in zip(ms, (0, 5, 10, 15, 20, 30))) + f" | +{best}% |")

sol = {**main, **load("weight_prio_solvers")}
table(sol, [("re-pricing +5%", "exact", None), ("re-pricing +5%", "greedy", ("re-pricing +5%", "exact")),
            ("re-pricing +5%", "sa_v3", ("re-pricing +5%", "exact")), ("re-pricing +5%", "sqa_v3", ("re-pricing +5%", "exact")),
            ("re-pricing +5%", "sqa_v3+LS", ("re-pricing +5%", "exact")), ("statistics +5%", "exact", None),
            ("statistics +5%", "sqa_v3+LS", ("statistics +5%", "exact"))], U3,
      "3. Batch solvers inside the final policy", "Reference: exact batch MILP with the same information and margin.")
tm = collections.defaultdict(list)
for k, d in sol.items():
    if k[0] in ("re-pricing +5%",) and k[2] == 95.0: tm[k[1]].append(d["time_sec"])
print("\nMean time per run at 95% (s; exact/greedy on the Mac, SA/SQA on ASUS, not comparable across machines): " +
      ", ".join(f"{m} {np.mean(v):.0f}" for m, v in tm.items()))

onl = {**main, **load("weight_prio_online")}
table(onl, [("re-pricing +5%", "exact", None), ("re-pricing", "exact", None), ("lp", "exact", ("re-pricing +5%", "exact")),
            ("lp+5%", "exact", ("re-pricing +5%", "exact")), ("dmd", "exact", ("re-pricing +5%", "exact")),
            ("lookahead", "exact", ("re-pricing +5%", "exact")), ("static", "exact", None)], U3,
      "4. Online-allocation baselines", "Reference: re-pricing + 5% (Lagrangian). lp = exact LP bid prices; dmd = dual mirror descent (eta 0.1, Lagrangian start); lookahead = 2-batch rolling horizon.")
tm = collections.defaultdict(list)
for k, d in onl.items():
    if k[2] == 95.0 and k[1] == "exact": tm[k[0]].append(d["time_sec"])
print("\nMean time per run at 95% (s): " + ", ".join(f"{m} {np.mean(v):.0f}" for m, v in sorted(tm.items())))

clu = load("weight_prio_clustered")
table(clu, [("static", "exact", ("re-pricing", "exact")), ("re-pricing", "exact", None), ("re-pricing +5%", "exact", ("re-pricing", "exact")),
            ("statistics +5%", "exact", ("re-pricing", "exact"))], U3,
      "5. Clustered urban layout", "Reference: clustered re-pricing without margin.")

print("\n## 6. Encoding ablation on hard batches (95%, re-pricing, priced batch objective)\n")
def abl(sub):
    per = collections.defaultdict(lambda: collections.defaultdict(list)); nb = 0
    for f in glob.glob(f"{V}/{sub}/*.json"):
        d = json.load(open(f)); s = d["args"]["instance_seed"]
        for r in d["rows"]:
            nb += sub.endswith("plain")
            for k, v in r.items():
                if k in ("batch", "exact") or k.count(":") > 1: continue
                per[k][s].append((v - r["exact"]) / r["exact"] * 100)
    return {k: np.mean([np.mean(v) for v in d.values()]) for k, d in per.items()}, nb
A, nb = abl("ablation_v31_wprio/plain"); B, _ = abl("ablation_v31_wprio/adaptive")
print(f"{nb} hard batches (greedy >= 0.5% above the batch optimum). Mean over seeds of the per-seed mean gap to the batch optimum.\n")
print("| Variant | SQA | SA |\n|---|---|---|")
for v, lab in [("base", "one-hot"), ("R", "+ reduction"), ("DW", "+ domain wall"), ("R+DW", "R + DW")]:
    print(f"| {lab} | {A.get(v + ':SQA', np.nan):.2f}% | {A.get(v + ':SA', np.nan):.2f}% |")
print(f"| R + DW + adaptive | {B.get('R+DW:SQA', np.nan):.2f}% | {B.get('R+DW:SA', np.nan):.2f}% |")
print(f"| greedy / regret / regret+LS | {A.get('greedy', np.nan):.2f}% / {A.get('regret', np.nan):.2f}% / {A.get('regret_ls', np.nan):.2f}% | |")
