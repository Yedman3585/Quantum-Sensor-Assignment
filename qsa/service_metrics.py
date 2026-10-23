"""Service metrics from trajectory traces (runs with --trace) and offline solutions.

For every run (policy, seed, utilisation) it reports
  * gap to the Lagrangian lower bound and gap to the offline solution (price of sequentiality);
  * per priority class: coverage, mean distance to the server, mean unweighted cost,
    share on the cheapest server, mean cost rank of the server, and the class
    contribution to the gap to the offline solution in pp (who pays for sequential decisions);
  * server load at the end of the horizon: share of servers filled to >= 99 %, coefficient
    of variation of the utilisation, and the first batch at which servers holding half of the
    total capacity are >= 99 % full.

Usage: python qsa/service_metrics.py --traces DIR [DIR ...] --offline DIR --out DIR
Writes service_runs.csv (one row per run and class), service_summary.md (means over seeds) and
saturation_curves.json (share of capacity on saturated servers after every batch, per run).
"""
import argparse, glob, json, os
from collections import defaultdict
import numpy as np
from batch_quality_benchmark import Instance

P = 15.0
CLASSES = {3: "high", 2: "medium", 1: "low"}
POLICY_ORDER = ["static", "reprice", "reprice5", "twin5", "offline"]
POLICY_NAME = {"static": "Static prices", "reprice": "Re-pricing (exact info)",
               "reprice5": "Re-pricing + 5% (exact info)", "twin5": "Re-pricing + 5% (statistics only)",
               "offline": "Offline solution"}


def policy_of(d):
    if not d.get("reprice"):
        return "static"
    fc = d.get("forecast") or d["params"].get("forecast") or "oracle"
    bias = d.get("forecast_bias", d["params"].get("forecast_bias", 0.0)) or 0.0
    if fc == "oracle" and bias == 0:
        return "reprice"
    if fc == "oracle" and abs(bias - 0.05) < 1e-9:
        return "reprice5"
    if fc == "twin" and abs(bias - 0.05) < 1e-9:
        return "twin5"
    return None


def assign_from_trace(d, n):
    a = -np.ones(n, int); when = np.full(n, -1)
    for tr in d["trace"]:
        a[tr["cams"]] = tr["servers"]; when[tr["cams"]] = tr["batch"]
    return a, when


def metrics(inst, a, when, off_a, lb, f_off):
    n = len(a); idx = np.arange(n); cov = a >= 0; s = np.maximum(a, 0)
    wc = np.where(cov, inst.wcost[idx, s], P)
    F = wc.sum()
    wc_off = np.where(off_a >= 0, inst.wcost[idx, np.maximum(off_a, 0)], P)
    excess = wc - wc_off
    g = inst.gen
    dist = np.hypot(g.camera_x - g.server_x[s], g.camera_y - g.server_y[s])
    c = inst.cost[idx, s]
    rank = (inst.cost < c[:, None] - 1e-12).sum(1) + 1  # 1 = cheapest server of the camera
    loadj = np.bincount(a[cov], weights=inst.load[cov], minlength=len(inst.cap))
    util = loadj / inst.cap
    sat = util >= 0.99
    # share of total capacity on servers that are >= 99% full, after every batch;
    # t_half = first batch at which it reaches 50% (None if never)
    t_half, curve = None, None
    if when is not None and when.max() >= 0:
        run = np.zeros(len(inst.cap)); curve = []
        for t in range(int(when.max()) + 1):
            m = cov & (when == t)
            run += np.bincount(a[m], weights=inst.load[m], minlength=len(inst.cap))
            curve.append(float(inst.cap[run >= 0.99 * inst.cap].sum() / inst.cap.sum()))
            if t_half is None and curve[-1] >= 0.5:
                t_half = t
    common = {"F": F, "gap_lb": 100 * (F - lb) / lb, "gap_off": 100 * (F - f_off) / f_off,
              "uncovered": int((~cov).sum()), "sat99": 100 * sat.mean(),
              "util_cv": float(util.std() / util.mean()), "t_half_sat": t_half, "curve": curve}
    rows = []
    for p, name in CLASSES.items():
        m = inst.priority == p; mc = m & cov
        rows.append({**common, "class": name, "n": int(m.sum()),
                     "coverage": 100 * mc.sum() / m.sum(),
                     "dist": float(dist[mc].mean()), "cost": float(c[mc].mean()),
                     "cheapest": 100 * float((rank[mc] == 1).mean()), "rank": float(rank[mc].mean()),
                     "excess": float(excess[m].sum()), "excess_pp": 100 * float(excess[m].sum()) / f_off})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", nargs="+", required=True)
    ap.add_argument("--offline", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    runs = {}
    for d_ in a.traces:
        for f in sorted(glob.glob(os.path.join(d_, "traj_exact_*.json"))):
            d = json.load(open(f))
            if "trace" not in d or d.get("layout", "uniform") != "uniform":
                continue
            pol = policy_of(d)
            if pol:
                runs[(d.get("weight", "inv"), pol, d["seed"], round(d["target_util"]))] = f  # later files override duplicates
    rows, curves = [], {}
    keys = sorted({(wt, s, u) for (wt, _, s, u) in runs})
    for wt, s, u in keys:
        fo = os.path.join(a.offline, "offline%s_u%g_seed%d.json" % ("_wprio" if wt == "prio" else "", u, s))
        if not os.path.exists(fo):
            print("no offline solution for", wt, s, u); continue
        off = json.load(open(fo)); off_a = np.array(off["assign"])
        inst = Instance(20000, 800, s, 1.0, u)
        if wt == "prio":
            inst.w = inst.priority.astype(float); inst.wcost = inst.cost * inst.w[:, None]
        for pol in POLICY_ORDER:
            if pol == "offline":
                a_, w_ = off_a, None
            elif (wt, pol, s, u) in runs:
                a_, w_ = assign_from_trace(json.load(open(runs[(wt, pol, s, u)])), 20000)
            else:
                continue
            rs_ = metrics(inst, a_, w_, off_a, off["lb"], off["objective"])
            if rs_[0]["curve"] is not None:
                curves["%s|%s|%d|%g" % (wt, pol, s, u)] = rs_[0]["curve"]
            for r in rs_:
                r.pop("curve")
                rows.append({"weight": wt, "policy": pol, "seed": s, "util": u, **r})
        print("done", wt, s, u, flush=True)
    cols = ["weight", "policy", "seed", "util", "class", "n", "F", "gap_lb", "gap_off", "uncovered", "sat99", "util_cv",
            "t_half_sat", "coverage", "dist", "cost", "cheapest", "rank", "excess", "excess_pp"]
    with open(os.path.join(a.out, "service_runs.csv"), "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join("" if r[c] is None else (("%.6g" % r[c]) if isinstance(r[c], float) else str(r[c])) for c in cols) + "\n")
    # summary
    agg = defaultdict(list)
    for r in rows:
        agg[(r["weight"], r["util"], r["policy"], r["class"])].append(r)
    L = ["# Service metrics (uniform layout, 20k x 800, exact batches; means over seeds)\n",
         "Gap to offline = price of sequentiality. Class contribution: 100 * (class cost - offline class cost) / offline objective;",
         "the three classes add up to the gap to offline. Distances in plane units (1000 x 1000 plane).\n"]
    for wt, u in sorted({(r["weight"], r["util"]) for r in rows}):
        L.append(f"\n## Weight {'w = 4 - p (default)' if wt == 'inv' else 'w = p'}, utilisation {u}%\n")
        L.append("| Policy | seeds | Gap to LB, % | Gap to offline, % | Servers >= 99% full, % | CV of utilisation | Half capacity saturated at batch |")
        L.append("|---|---|---|---|---|---|---|")
        for pol in POLICY_ORDER:
            rs = agg.get((wt, u, pol, "high"))
            if not rs: continue
            th = [r["t_half_sat"] for r in rs if r["t_half_sat"] is not None]
            L.append(f"| {POLICY_NAME[pol]} | {len(rs)} | {np.mean([r['gap_lb'] for r in rs]):.2f} | {np.mean([r['gap_off'] for r in rs]):.2f} | "
                     f"{np.mean([r['sat99'] for r in rs]):.1f} | {np.mean([r['util_cv'] for r in rs]):.3f} | "
                     f"{('%.0f' % np.mean(th)) if th else 'never'} |")
        L.append("\n| Policy | Class | Coverage, % | Mean distance | Mean cost c_ij | On cheapest server, % | Mean server rank | Contribution to gap to offline, pp |")
        L.append("|---|---|---|---|---|---|---|---|")
        for pol in POLICY_ORDER:
            for cl in CLASSES.values():
                rs = agg.get((wt, u, pol, cl))
                if not rs: continue
                ex = "—" if pol == "offline" else "%+.2f" % np.mean([r["excess_pp"] for r in rs])
                L.append(f"| {POLICY_NAME[pol]} | {cl} | {np.mean([r['coverage'] for r in rs]):.2f} | {np.mean([r['dist'] for r in rs]):.1f} | "
                         f"{np.mean([r['cost'] for r in rs]):.4f} | {np.mean([r['cheapest'] for r in rs]):.1f} | {np.mean([r['rank'] for r in rs]):.2f} | {ex} |")
    json.dump(curves, open(os.path.join(a.out, "saturation_curves.json"), "w"))
    open(os.path.join(a.out, "service_summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
