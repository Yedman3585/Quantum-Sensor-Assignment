import glob, json, collections, sys
import numpy as np
from scipy.stats import wilcoxon
REPO = sys.argv[1]; V = REPO + "/results/v3"
G2 = "/mnt/user-data/uploads/Quantum-Sensor-Assignment/results/v3/campaign2/global"
LB = {}
for f in glob.glob(G2 + "/*.json"):
    for d in json.load(open(f)): LB[("uniform", 20000, d["target_util"], d["seed"])] = d["lagrangian_lower_bound"]
for d in json.load(open(V + "/clustered_summary/lb_clustered_20k.json")): LB[("clustered", 20000, d["target_util"], d["seed"])] = d["lb"]
for f in glob.glob(V + "/campaign2/scale_global/*.json"):
    for d in json.load(open(f)): LB[("uniform", d["n_cameras"], d["target_util"], d["seed"])] = d["lagrangian_lower_bound"]
def holm(ps):
    ps = np.asarray(ps, float); idx = np.argsort(ps); adj = np.empty(len(ps)); run = 0
    for r, i in enumerate(idx): run = max(run, (len(ps) - r) * ps[i]); adj[i] = min(1, run)
    return adj
R = collections.defaultdict(dict)   # key -> seed -> (obj, unc, time)
def add(key, d):
    R[key][d["seed"]] = (d["objective"], d["n_cameras"] - d["covered"], d["time_sec"])
def cond(d):
    if not d.get("reprice"): return "static"
    f = d.get("forecast", "oracle"); b = d.get("forecast_bias", 0) or 0; l = d.get("forecast_level", 0)
    base = {"oracle": "oracle", "twin": "twin", "noise": f"noise{l:g}"}[f]
    return base + (f"{b:+g}" if b else "")
for sub in ["campaign3/v31", "forecast", "margin_test"]:
    for f in glob.glob(f"{V}/{sub}/*.json"):
        d = json.load(open(f))
        if sub == "campaign3/v31" and "reprice_top" not in d: continue
        m = d["method"] + ("+LS" if d.get("polish") else "")
        add(("uniform", 20000, m, cond(d), d["target_util"]), d)
for f in glob.glob(f"{V}/clustered/*.json"):
    d = json.load(open(f)); m = d["method"] + ("+LS" if d.get("polish") else "")
    add(("clustered", 20000, m, cond(d), d["target_util"]), d)
for f in glob.glob(f"{V}/sensitivity/*.json"):
    d = json.load(open(f)); add(("uniform", 20000, "exact", f"B{d['batch_size']}_T{d['reprice']}", d["target_util"]), d)
for f in glob.glob(f"{V}/scale_v31/*.json"):
    d = json.load(open(f)); m = d["method"] + ("+LS" if d.get("polish") else "")
    add(("uniform", d["n_cameras"], m, "oracle", d["target_util"]), d)
for f in glob.glob(f"{V}/online_baselines/*.json"):
    d = json.load(open(f)); p = d["policy"]
    if p == "lp": tag = "lp" + (f"{d['forecast_bias']:+g}" if d.get("forecast_bias") else "")
    elif p == "dmd": tag = f"dmd_{d['eta']:g}_{d['dmd_init']}"
    else: tag = f"la{d['lookahead']}"
    add((d["layout"], d["n_cameras"], "exact", tag, d["target_util"]), d)
def gap(lay, n, u, s, o): return (o - LB[(lay, n, u, s)]) / LB[(lay, n, u, s)] * 100
def row(lay, n, m, c, u, ref=None):
    g = R[(lay, n, m, c, u)]
    if not g: return None
    ss = sorted(s for s in g if (lay, n, u, s) in LB)
    gg = np.array([gap(lay, n, u, s, g[s][0]) for s in ss]); unc = sum(g[s][1] for s in ss)
    out = {"gap": gg.mean(), "unc": unc, "n": len(ss), "t": np.mean([g[s][2] for s in ss])}
    if ref is not None:
        r = R[ref[:3] + (ref[3], u)] if len(ref) == 4 else R[ref]
        cs = [s for s in ss if s in r]
        rg = np.array([gap(lay, n, u, s, r[s][0]) for s in cs]); g2 = np.array([gap(lay, n, u, s, g[s][0]) for s in cs])
        dd = g2 - rg
        out.update({"d": dd.mean(), "w": int((dd < -1e-9).sum()), "l": int((dd > 1e-9).sum()), "nn": len(cs),
                    "p": wilcoxon(g2, rg).pvalue if np.any(np.abs(dd) > 1e-9) else 1.0})
    return out
def fmt(o, holm_p=None, sig=True):
    if o is None: return "--"
    s = f"{o['gap']:.2f}" + (f" [{o['unc']}]" if o["unc"] else "") + (f" (n={o['n']})" if o["n"] != 10 else "")
    if "d" in o:
        p = holm_p if holm_p is not None else o["p"]
        s += f" ({o['d']:+.2f}; {o['w']}/{o['nn']}; p={p:.3f})"
    return s
U = [90.0, 95.0, 98.0]
print("## Forecast table (exact, uniform 20k), Δ vs oracle, Holm within column")
C = ["static", "oracle", "noise0.3", "oracle-0.1", "oracle+0.05", "oracle+0.1", "twin", "twin+0.05", "twin+0.1"]
cells = {c: [] for c in C}
for u in U:
    rs = {c: row("uniform", 20000, "exact", c, u, ("uniform", 20000, "exact", "oracle")) if c != "oracle" else row("uniform", 20000, "exact", c, u) for c in C}
    keys = [c for c in C if c != "oracle"]; adj = dict(zip(keys, holm([rs[c]["p"] for c in keys])))
    for c in C: cells[c].append(fmt(rs[c], adj.get(c)))
for c in C: print(f"| {c} | " + " | ".join(cells[c]) + " |")
print("\n## SQA+LS twin+5% vs exact twin+5%")
for u in [95.0, 98.0]:
    print(u, fmt(row("uniform", 20000, "sqa_v3+LS", "twin+0.05", u, ("uniform", 20000, "exact", "twin+0.05"))), "| greedy", fmt(row("uniform", 20000, "greedy", "twin+0.05", u, ("uniform", 20000, "exact", "twin+0.05"))))
print("\n## Clustered (Δ vs clustered oracle exact)")
for m, c in [("exact", "static"), ("exact", "oracle"), ("greedy", "oracle"), ("sqa_v3+LS", "oracle"), ("exact", "twin+0.05"), ("greedy", "twin+0.05"), ("sqa_v3+LS", "twin+0.05")]:
    print(f"| {m} {c} | " + " | ".join(fmt(row("clustered", 20000, m, c, u, None if (m, c) == ("exact", "oracle") else ("clustered", 20000, "exact", "oracle"))) for u in U) + " |")
print("\n## Online baselines (Δ vs v3.1 oracle exact)")
for c in ["oracle", "oracle+0.05", "lp", "lp+0.05", "dmd_0.1_lagrange", "dmd_1_zero", "la2", "static"]:
    rr = [row("uniform", 20000, "exact", c, u, None if c == "oracle" else ("uniform", 20000, "exact", "oracle")) for u in U]
    print(f"| {c} | " + " | ".join(fmt(x) for x in rr) + f" | t95={rr[1]['t'] if rr[1] else float('nan'):.0f}s |")
print("\n## Sensitivity (Δ vs B80 T5 = campaign3 oracle)")
for c in ["B40_T5", "B160_T5", "B320_T5", "B80_T1", "B80_T10", "B80_T25"]:
    rr = [row("uniform", 20000, "exact", c, u, ("uniform", 20000, "exact", "oracle")) for u in [95.0, 98.0]]
    print(f"| {c} | " + " | ".join(fmt(x) for x in rr) + f" | t95={rr[0]['t'] if rr[0] else float('nan'):.0f}s |")
print("\n## 50k")
for m in ["exact", "greedy", "sqa_v3+LS"]:
    o = row("uniform", 50000, m, "oracle", 95.0); print(m, fmt(o), f"t={o['t']:.0f}" if o else "")
# ablation v3.1
print("\n## Ablation v3.1 (hard batches, priced objective)")
rows = collections.defaultdict(list); ref = collections.defaultdict(list); nb = 0; per = collections.defaultdict(list)
for sub in ["plain", "adaptive", "beta20"]:
    for f in glob.glob(f"{V}/ablation_v31/{sub}/*.json"):
        d = json.load(open(f))
        for r in d["rows"]:
            for k, v in r.items():
                if k in ("batch", "exact") or k.count(":") != 1 and k not in ("greedy", "regret", "regret_ls"): continue
                name = (sub + "|" + k) if ":" in k else k
                if ":" not in k and sub != "plain": continue
                rows[name].append((v - r["exact"]) / r["exact"] * 100)
        if sub == "plain": nb += len(d["rows"])
print("hard batches", nb)
for k, v in sorted(rows.items()):
    v = np.array(v); print(f"{k:28s} gap {v.mean():.3f}% opt {int((v < 1e-6).sum())}/{len(v)}")
