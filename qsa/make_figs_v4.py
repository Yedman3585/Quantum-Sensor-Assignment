"""Figures for the price-coordinated decomposition manuscript (round 3, vector PDF + PNG).

Reads the archived run logs of the repository and writes one PDF/PNG per figure.
Usage (repository root):
    python qsa/make_figs_v4.py --repo . --out figures [which ...]
Figures: gap_util, margin, policy_time, sensitivity, ablation2, trajectory2, weight, service (service_wprio)
Style: 16 cm single column, 8.5 pt text; fixed entity colours (validated categorical order: blue, orange, aqua,
yellow, magenta, green, violet) plus markers / line styles so every series survives grey-scale print.
"""
import argparse, collections, glob, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"
INK, SEC, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8983", "#e6e5e0"
W = 6.3
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.3,
    "axes.edgecolor": MUTED, "axes.labelcolor": SEC, "xtick.color": SEC, "ytick.color": SEC,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.linewidth": 0.7, "lines.linewidth": 1.6,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "legend.frameon": False, "pdf.fonttype": 42,
})
# one colour / marker / line style per entity, identical in every figure
ENT = {
    "static":      dict(c=ORANGE,  m="v", ls="-",  lab="static prices"),
    "reprice":     dict(c=AQUA,    m="s", ls="--", lab="Lagrangian re-pricing"),
    "reprice5":    dict(c=BLUE,    m="o", ls="-",  lab="Lagrangian re-pricing + 5% margin"),
    "twin5":       dict(c=MAGENTA, m="D", ls="-.", lab="statistics-only forecast + 5% margin"),
    "lp":          dict(c=YELLOW,  m="^", ls=":",  lab="LP bid prices"),
    "lp5":         dict(c=YELLOW,  m="^", ls="-",  lab="LP bid prices + 5% margin"),
    "dmd":         dict(c=GREEN,   m="P", ls="-",  lab="dual mirror descent"),
    "rh":          dict(c=VIOLET,  m="X", ls="-",  lab="rolling horizon (2 batches)"),
    "offline":     dict(c=MUTED,   m="*", ls=":",  lab="offline global solution"),
}
U4 = [75.0, 90.0, 95.0, 98.0]


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    fig.savefig(f"{out}/{name}.pdf"); fig.savefig(f"{out}/{name}.png", dpi=200); plt.close(fig)
    print("wrote", name)


def panel(ax, letter, title=None):
    ax.text(-0.02, 1.04, letter, transform=ax.transAxes, fontsize=9.5, fontweight="bold", color=INK, ha="right", va="bottom")
    if title: ax.set_title(title, loc="left", color=INK)


def ci95(x):
    x = np.asarray(x, float)
    return 0.0 if len(x) < 2 else 2.262 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) == 10 else 1.96 * x.std(ddof=1) / np.sqrt(len(x))


class Data:
    def __init__(self, repo, val_lb=None):
        V = f"{repo}/results/v3"; self.V = V
        self.LB, self.OFF = {}, {}
        for f in glob.glob(f"{V}/campaign2/global/*.json") + glob.glob(f"{V}/campaign2/scale_global/*.json"):
            for d in json.load(open(f)):
                k = ("uniform", d.get("n_cameras", 20000), d["target_util"], d["seed"])
                self.LB[k] = d["lagrangian_lower_bound"]; self.OFF[k] = d["feasible_objective"]
        p = f"{V}/clustered_summary/lb_clustered_20k.json"
        if os.path.exists(p):
            for d in json.load(open(p)): self.LB[("clustered", 20000, d["target_util"], d["seed"])] = d["lb"]
        if val_lb:
            for f in val_lb:
                for d in json.load(open(f)): self.LB[("uniform", 20000, d["target_util"], d["seed"])] = d["lb"]

    def gap(self, d, obj=None):
        k = (d.get("layout", "uniform"), d["n_cameras"], d["target_util"], d["seed"])
        lb = self.LB.get(k)
        return None if lb is None else ((d["objective"] if obj is None else obj) - lb) / lb * 100

    def runs(self, sub, pred=lambda d: True):
        out = []
        for f in glob.glob(f"{self.V}/{sub}/*.json"):
            d = json.load(open(f))
            if "rows" in d: continue
            d.setdefault("layout", "uniform")
            if pred(d) and self.gap(d) is not None: out.append(d)
        return out


def cond(d):
    if not d.get("reprice"): return "static"
    b = d.get("forecast_bias") or 0; f = d.get("forecast", "oracle")
    if f == "oracle": return "reprice5" if abs(b - 0.05) < 1e-9 else ("reprice" if b == 0 else f"oracle{b:+g}")
    if f == "twin": return "twin5" if abs(b - 0.05) < 1e-9 else ("twin" if b == 0 else f"twin{b:+g}")
    return f"{f}{d.get('forecast_level', 0):g}{b:+g}"


def by(runs, key):
    g = collections.defaultdict(dict)
    for d in runs: g[key(d)][d["seed"]] = d
    return g


# ------------------------------------------------------------------ 1. gap vs utilisation
def fig_gap_util(D, out):
    exact = lambda d: d["method"] == "exact" and not d.get("polish")
    S = {}
    S["static"] = by(D.runs("campaign2/priced", lambda d: exact(d) and d.get("window", "percam") == "percam"), lambda d: d["target_util"])
    S["reprice"] = by(D.runs("campaign3/v31", lambda d: exact(d) and "reprice_top" in d), lambda d: d["target_util"])
    mt = D.runs("margin_test", exact)
    S["reprice5"] = by([d for d in mt if cond(d) == "reprice5"], lambda d: d["target_util"])
    S["twin5"] = by([d for d in mt if cond(d) == "twin5"], lambda d: d["target_util"])
    S["lp5"] = by(D.runs("online_baselines", lambda d: d.get("policy") == "lp" and abs((d.get("forecast_bias") or 0) - 0.05) < 1e-9), lambda d: d["target_util"])
    cl = D.runs("clustered", exact)
    C = {k: by([d for d in cl if cond(d) == k], lambda d: d["target_util"]) for k in ["static", "reprice", "reprice5", "twin5"]}
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.7), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, data, letter, title, show_off in [(axs[0], S, "a", "Uniform layout", True), (axs[1], C, "b", "Clustered urban layout", False)]:
        for k in ["static", "reprice", "lp5", "twin5", "reprice5"]:
            if k not in data: continue
            us = [u for u in U4 if u in data[k] and len(data[k][u]) >= 3]
            if not us: continue
            m = [np.mean([D.gap(x) for x in data[k][u].values()]) for u in us]
            e = [ci95([D.gap(x) for x in data[k][u].values()]) for u in us]
            st = ENT[k]
            ax.errorbar(us, m, yerr=e, color=st["c"], marker=st["m"], ls=st["ls"], ms=4.5, capsize=2.2, lw=1.5, label=st["lab"])
        if show_off:
            og = [np.mean([(D.OFF[("uniform", 20000, u, s)] - D.LB[("uniform", 20000, u, s)]) / D.LB[("uniform", 20000, u, s)] * 100
                           for s in range(42, 52) if ("uniform", 20000, u, s) in D.OFF]) for u in U4]
            st = ENT["offline"]; ax.plot(U4, og, color=st["c"], marker=st["m"], ls=st["ls"], ms=5, lw=1.2, label=st["lab"])
        ax.set_yscale("log"); ax.set_xticks(U4); ax.set_xlim(73, 99.5); ax.set_xlabel("Utilisation (%)")
        ax.set_yticks([0.3, 1, 3, 10, 20]); ax.set_yticklabels(["0.3", "1", "3", "10", "20"])
        panel(ax, letter, title)
    axs[0].set_ylabel("Gap to the Lagrangian bound (%, log)"); axs[0].set_ylim(0.2, 30)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.02), fontsize=7)
    save(fig, out, "fig_gap_util")


# ------------------------------------------------------------------ 2. margin selection and information value
def fig_margin(D, out):
    tun = D.runs("margin_tuning", lambda d: d["method"] == "exact")
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.9), gridspec_kw={"wspace": 0.62, "width_ratios": [1, 1.05]})
    ax = axs[0]; B = [0, 5, 10, 15, 20, 30]
    for fc, ent in [("oracle", "reprice5"), ("twin", "twin5")]:
        for u, alpha in [(90.0, 0.45), (95.0, 0.7), (98.0, 1.0)]:
            m = []
            for b in B:
                g = [D.gap(d) for d in tun if d.get("forecast", "oracle") == fc and abs((d.get("forecast_bias") or 0) * 100 - b) < 1e-6 and d["target_util"] == u]
                m.append(np.mean(g) if g else np.nan)
            st = ENT[ent]
            ax.plot(B, m, color=st["c"], alpha=alpha, marker=st["m"], ms=3.8, ls=st["ls"], lw=1.3,
                    label=f"{'exact demand' if fc == 'oracle' else 'statistics only'}, {u:g}%")
    ax.axvline(5, color=MUTED, ls=":", lw=0.9); ax.text(5.6, ax.get_ylim()[1] * 0.97 if False else 4.9, "selected", fontsize=7, color=SEC, va="top")
    ax.set_xlabel("Demand margin b (%)"); ax.set_ylabel("Gap to the bound (%)"); ax.set_xticks(B)
    ax.legend(fontsize=6.2, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.2)); panel(ax, "a", "Margin selection (seeds 100–109)")
    # b: test seeds, information levels at 95 and 98 %
    ax = axs[1]
    rows = [("static prices", "static"), ("exact demand", "reprice"), ("exact + 5%", "reprice5"),
            ("demand −10%", "oracle-0.1"), ("±30% load noise", "noise0.3+0"), ("statistics only", "twin"), ("statistics + 5%", "twin5")]
    allr = D.runs("forecast", lambda d: d["method"] == "exact") + D.runs("margin_test", lambda d: d["method"] == "exact") + \
        D.runs("campaign3/v31", lambda d: d["method"] == "exact" and "reprice_top" in d)
    G = collections.defaultdict(dict)
    for d in allr: G[(cond(d), d["target_util"])][d["seed"]] = D.gap(d)
    for j, (u, col, mk, off) in enumerate([(95.0, AQUA, "o", -0.17), (98.0, BLUE, "s", 0.17)]):
        for i, (lab, c) in enumerate(rows):
            v = list(G[(c, u)].values())
            if not v: continue
            y = len(rows) - 1 - i + off
            ax.scatter(v, [y] * len(v), s=9, color=col, alpha=0.35, lw=0, zorder=2)
            ax.scatter([np.mean(v)], [y], s=34, color=col, marker=mk, edgecolor="white", lw=0.8, zorder=3, label=f"{u:g}% (mean)" if i == 0 else None)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]]); ax.grid(axis="y", visible=False)
    ax.set_xscale("log"); ax.set_xticks([0.5, 1, 2, 5, 10, 20]); ax.set_xticklabels(["0.5", "1", "2", "5", "10", "20"])
    ax.set_xlabel("Gap to the bound (%, log)")
    ax.legend(loc="upper center", fontsize=6.8, ncol=2, bbox_to_anchor=(0.5, -0.2)); panel(ax, "b", "Test seeds 42–51")
    save(fig, out, "fig_margin")


# ------------------------------------------------------------------ 3. quality versus time (one machine)
def fig_policy_time(D, out):
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.7), gridspec_kw={"wspace": 0.55})
    ax = axs[0]; u = 95.0
    pts = []
    tim = D.runs("timing_asus", lambda d: d["target_util"] == u)
    for d in tim: pass
    def add(runs, ent, lab=None):
        if not runs: return
        g = [D.gap(d) for d in runs]; t = [d["time_sec"] for d in runs]
        st = ENT[ent]
        ax.errorbar(np.mean(t), np.mean(g), xerr=ci95(t), yerr=ci95(g), color=st["c"], marker=st["m"], ms=6, capsize=2, lw=1, label=lab or st["lab"], ls="none")
    add([d for d in tim if "policy" not in d and d["method"] == "exact" and not d.get("reprice")], "static")
    add(D.runs("campaign3/v31", lambda d: d["method"] == "exact" and "reprice_top" in d and d["target_util"] == u), "reprice")
    add([d for d in tim if "policy" not in d and d["method"] == "exact" and d.get("reprice")], "reprice5")
    add([d for d in tim if d.get("policy") == "lp"], "lp5")
    ob = D.runs("online_baselines", lambda d: d["target_util"] == u)
    add([d for d in ob if d.get("policy") == "dmd" and d.get("dmd_init") == "lagrange"], "dmd", "dual mirror descent (warm start)")
    add([d for d in ob if d.get("policy") == "lookahead"], "rh")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Time per instance (s, log)"); ax.set_ylabel("Gap to the bound (%, log)")
    ax.set_yticks([1, 2, 5, 10, 20, 40]); ax.set_yticklabels(["1", "2", "5", "10", "20", "40"])
    ax.legend(fontsize=6.2, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.2)); panel(ax, "a", "Coordination policies, 95%")
    # b: batch solvers in the final configuration: same quality, different time
    ax = axs[1]; tim = D.runs("timing_asus")
    meth = [("exact", "exact MILP"), ("greedy", "greedy"), ("regret", "regret"), ("sa_v3", "SA-v3.1"),
            ("ils", "ILS"), ("sqa_v3", "SQA-v3.1"), ("sqa_v3+LS", "SQA-v3.1+LS")]
    for j, (key, lab) in enumerate(meth):
        for u, col, off in [(95.0, AQUA, -0.19), (98.0, BLUE, 0.19)]:
            r = [d for d in tim if "policy" not in d and d.get("reprice") and d["target_util"] == u and d["method"] + ("+LS" if d.get("polish") else "") == key]
            if not r: continue
            t = np.mean([d["time_sec"] for d in r]); g = np.mean([D.gap(d) for d in r])
            y = len(meth) - 1 - j + off
            ax.barh(y, t, height=0.34, color=col, label=f"{u:g}%" if j == 0 else None)
            ax.text(t + 10, y, f"{g:.2f}%", va="center", fontsize=6.0, color=SEC)
    ax.set_yticks(range(len(meth))); ax.set_yticklabels([m[1] for m in meth[::-1]]); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Time per instance (s); labels: gap to the bound"); ax.set_xlim(0, 860)
    ax.legend(fontsize=6.8, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.2)); panel(ax, "b", "Batch solvers, re-pricing + 5% margin")
    fig.text(0.5, -0.27, "Same machine (Intel Core i7-11370H, four jobs in parallel); means ± 95% CI over seeds.", ha="center", fontsize=7, color=SEC)
    save(fig, out, "fig_policy_time")


# ------------------------------------------------------------------ 4. batch size / re-pricing period
def fig_sensitivity(D, out):
    base0 = {d["target_util"]: [] for d in []}
    def series(sub, u, var):
        r = D.runs(sub, lambda d: d["method"] == "exact" and d["target_util"] == u)
        g = collections.defaultdict(list)
        for d in r:
            B, T = d.get("batch_size") or d["params"].get("batch_size", 80), d.get("reprice")
            g[(B, T)].append(D.gap(d))
        return g
    ref0 = {u: [D.gap(d) for d in D.runs("campaign3/v31", lambda d: d["method"] == "exact" and "reprice_top" in d and d["target_util"] == u)] for u in (95.0, 98.0)}
    ref5 = {u: [D.gap(d) for d in D.runs("margin_test", lambda d: d["method"] == "exact" and cond(d) == "reprice5" and d["target_util"] == u)] for u in (95.0, 98.0)}
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.5), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, letter, var, xs, xl in [(axs[0], "a", "T", [1, 5, 10, 25], "Re-pricing period $T_p$ (batches), $B=80$"),
                                    (axs[1], "b", "B", [40, 80, 160, 320], "Batch size $B$ (sensors), $T_p=5$")]:
        for sub, ref, ent, mtag in [("sensitivity", ref0, "reprice", "no margin"), ("sensitivity_m05", ref5, "reprice5", "+5% margin")]:
            for u, alpha in [(95.0, 0.55), (98.0, 1.0)]:
                g = series(sub, u, var); ys = []
                for x in xs:
                    k = (80, x) if var == "T" else (x, 5)
                    v = ref[u] if k == (80, 5) else g.get(k, [])
                    ys.append(np.mean(v) if v else np.nan)
                st = ENT[ent]
                ax.plot(xs, ys, color=st["c"], alpha=alpha, marker=st["m"], ls=st["ls"], ms=4, label=f"{mtag}, {u:g}%")
        ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels([str(x) for x in xs]); ax.minorticks_off()
        ax.axvline(5 if var == "T" else 80, color=MUTED, ls=":", lw=0.9)
        ax.set_xlabel(xl); panel(ax, letter)
    axs[0].set_yscale("log"); axs[0].set_yticks([0.7, 1, 2, 4, 8]); axs[0].set_yticklabels(["0.7", "1", "2", "4", "8"])
    axs[0].set_ylabel("Gap to the bound (%, log)"); axs[0].legend(fontsize=6.6, loc="upper left")
    save(fig, out, "fig_sensitivity")


# ------------------------------------------------------------------ 5. ablation, static and rolling prices
def fig_ablation2(D, out):
    def collect(dirs):
        rows = collections.defaultdict(list)
        for sub, tag in dirs:
            for f in glob.glob(f"{D.V}/{sub}/*.json"):
                d = json.load(open(f))
                for r in d["rows"]:
                    for k, v in r.items():
                        if k in ("batch", "exact") or (":" in k and k.count(":") != 1): continue
                        if ":" not in k and tag != "plain": continue
                        rows[(tag + "|" + k) if ":" in k else k].append((v - r["exact"]) / r["exact"] * 100)
        return {k: np.mean(v) for k, v in rows.items()}, max(len(v) for v in rows.values())
    sets = [("campaign2/ablation", "plain"), ("campaign2/ablation_adaptive", "adaptive")]
    A0, n0 = collect(sets); A1, n1 = collect([("ablation_v31/plain", "plain"), ("ablation_v31/adaptive", "adaptive")])
    variants = [("plain|base", "one-hot"), ("plain|R", "+ reduction (R)"), ("plain|DW", "+ domain wall (DW)"), ("plain|R+DW", "R + DW"), ("adaptive|R+DW", "R + DW + adaptive")]
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.5), sharey=True, gridspec_kw={"wspace": 0.06})
    for ax, A, n, letter, title in [(axs[0], A0, n0, "a", f"Static prices ({n0} hard batches)"), (axs[1], A1, n1, "b", f"Rolling re-pricing ({n1} hard batches)")]:
        for i, (k, lab) in enumerate(variants):
            y = len(variants) - 1 - i
            q, s_ = A.get(k + ":SQA"), A.get(k + ":SA")
            if q is None: continue
            ax.plot([s_, q], [y, y], color=GRID, lw=2.2, zorder=1)
            ax.scatter(s_, y, s=34, facecolor="white", edgecolor=BLUE, lw=1.5, marker="s", zorder=3, label="SA" if i == 0 else None)
            ax.scatter(q, y, s=34, color=ORANGE, edgecolor="white", lw=0.8, marker="o", zorder=3, label="SQA" if i == 0 else None)
        for key, lab, ls in [("greedy", "greedy", ":"), ("regret_ls", "regret+LS", "--")]:
            if key in A:
                ax.axvline(A[key], color=MUTED, ls=ls, lw=1.0); ax.text(A[key] * 1.06, len(variants) - 0.55, lab, color=SEC, fontsize=6.8)
        ax.set_xscale("log"); ax.set_xlabel("Mean gap to the batch optimum (%, log)"); panel(ax, letter, title)
        tk = [0.01, 0.03, 0.1, 0.3, 1, 3, 10]; lo, hi = ax.get_xlim(); tk = [x for x in tk if lo <= x <= hi]
        ax.set_xticks(tk); ax.set_xticklabels([f"{x:g}" for x in tk]); ax.minorticks_off()
        ax.set_ylim(-0.6, len(variants) - 0.2)
    axs[0].set_yticks(range(len(variants))); axs[0].set_yticklabels([v[1] for v in variants[::-1]]); axs[0].grid(axis="y", visible=False); axs[1].grid(axis="y", visible=False)
    axs[1].legend(loc="lower left", fontsize=7)
    save(fig, out, "fig_ablation2")


# ------------------------------------------------------------------ 6. where the loss arises (seed 42 traces)
def fig_trajectory2(D, out, tr_old, tr_new):
    G = {o["util"]: o for o in json.load(open(f"{tr_old}/global_assign_seed42.json"))}
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.5), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, u, letter in [(axs[0], 95.0, "a"), (axs[1], 98.0, "b")]:
        gcost = np.array(G[u]["cost"]); lb = G[u]["lb"]; ends = []
        for pat, d_, ent in [("traj_exact_priced_percam_20000x800_u%g_seed42_*.json", tr_old, "static"),
                             ("traj_exact_priced_rp5_ks2_percam_20000x800_u%g_seed42_*.json", tr_old, "reprice"),
                             ("traj_exact_priced_rp5_ks2_fc-twin0+0.05_percam_20000x800_u%g_seed42_*.json", tr_new, "twin5"),
                             ("traj_exact_priced_rp5_ks2_fc-oracle0+0.05_percam_20000x800_u%g_seed42_*.json", tr_new, "reprice5")]:
            fs = sorted(glob.glob(f"{d_}/" + pat % u))
            if not fs: continue
            d = json.load(open(fs[-1])); acc = 0.0; ex = []
            for r in d["trace"]:
                acc += r["objective"] - gcost[r["cams"]].sum(); ex.append(acc / lb * 100)
            st = ENT[ent]
            ax.plot(np.arange(1, len(ex) + 1), ex, color=st["c"], ls=st["ls"], lw=1.4, label=st["lab"])
            ends.append((ex[-1], len(ex)))
        ys = sorted(ends); placed = []
        for v, n in ys:                      # keep end labels at least 0.9 units apart
            y = v if not placed else max(v, placed[-1] + 0.9); placed.append(y)
            ax.annotate(f"{v:.1f}%", (n, v), xytext=(n + 4, y), textcoords="data", fontsize=7, color=SEC, va="center")
        ax.axhline(0, color=MUTED, lw=0.8); ax.set_xlim(0, 280)
        ax.set_xlabel("Batch (80 sensors, priority order)"); panel(ax, letter, f"{u:g}% utilisation, seed 42")
    axs[0].set_ylabel("Cumulative excess over the offline\nsolution (% of lower bound)"); axs[0].legend(loc="upper left", fontsize=6.6)
    save(fig, out, "fig_trajectory2")


# ------------------------------------------------------------------ 7. priority weight (supplementary)
def fig_weight(D, out):
    p = f"{D.V}/weight_prio_summary/lb_prio_20k.json"
    LBp = {(d["target_util"], d["seed"]): d["lb"] for d in json.load(open(p))}
    gd = collections.defaultdict(dict)   # duplicated runs (two machines) are identical; keep one per seed
    for f in glob.glob(f"{D.V}/weight_prio/*.json"):
        d = json.load(open(f))
        if d["method"] != "exact": continue
        gd[(cond(d), d["target_util"])][d["seed"]] = (d["objective"] - LBp[(d["target_util"], d["seed"])]) / LBp[(d["target_util"], d["seed"])] * 100
    g = collections.defaultdict(list, {k: list(v.values()) for k, v in gd.items()})
    fig, ax = plt.subplots(figsize=(W * 0.55, 2.3))
    for k, ent in [("static", "static"), ("twin", "twin5"), ("reprice", "reprice")]:
        us = [u for u in (90.0, 95.0, 98.0) if g[(k, u)]]
        st = ENT[ent]; lab = "statistics-only forecast" if k == "twin" else st["lab"]
        ax.errorbar(us, [np.mean(g[(k, u)]) for u in us], yerr=[ci95(g[(k, u)]) for u in us], color=st["c"], marker=st["m"], ls=st["ls"], ms=4.5, capsize=2, label=lab)
    ax.set_xticks([90, 95, 98]); ax.set_xlabel("Utilisation (%)"); ax.set_ylabel("Gap to the bound (%)")
    ax.legend(fontsize=6.8); ax.set_title("Priority weight $w_i=p_i$ (exact batches)", loc="left", color=INK)
    save(fig, out, "fig_weight")


# ------------------------------------------------------------------ 8. service and load metrics
def fig_service(D, out, weight="inv"):
    S = f"{D.V}/service_summary"
    rows = list(__import__("csv").DictReader(open(f"{S}/service_runs.csv")))
    curves = json.load(open(f"{S}/saturation_curves.json"))
    pols = ["static", "reprice", "reprice5", "twin5", "offline"]
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.8), gridspec_kw={"width_ratios": [1.05, 1], "wspace": 0.32})
    ax = axs[0]; ylab = []; y = 0
    for u in (95.0, 98.0):
        for cl in ("high", "medium", "low"):
            for k, pol in enumerate(pols):
                v = [float(r["dist"]) for r in rows if r["weight"] == weight and r["policy"] == pol and float(r["util"]) == u and r["class"] == cl]
                if not v: continue
                st = ENT[pol]
                ax.plot(np.mean(v), y + (k - 2) * 0.13, marker=st["m"], color=st["c"], ms=5 if pol != "offline" else 7,
                        ls="none", mfc=st["c"] if pol != "reprice" else "white", label=st["lab"] if (u, cl) == (95.0, "high") else None)
            ylab.append((y, f"{cl}, {u:g}%")); y += 1
        y += 0.4
    ax.set_yticks([p for p, _ in ylab]); ax.set_yticklabels([t for _, t in ylab]); ax.invert_yaxis()
    ax.set_xlabel("Mean camera–server distance (plane units)"); ax.grid(axis="y", visible=False)
    panel(ax, "a", "Distance by priority class")
    ax = axs[1]
    for pol in ["static", "reprice", "reprice5", "twin5"]:
        cs = [np.array(v) for k, v in curves.items() if k.startswith(f"{weight}|{pol}|") and k.endswith("|98")]
        if not cs: continue
        n = min(len(c) for c in cs); m = np.mean([c[:n] for c in cs], 0) * 100
        st = ENT[pol]; ax.plot(np.arange(1, n + 1), m, color=st["c"], ls=st["ls"], lw=1.4)
    ax.axhline(50, color=MUTED, lw=0.7, ls=":")
    ax.set_xlim(0, 255); ax.set_ylim(0, 100); ax.set_xlabel("Batch (80 sensors, priority order)")
    ax.set_ylabel("Capacity on servers ≥ 99% full (%)")
    panel(ax, "b", "Saturation over the horizon, 98%")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.02), fontsize=6.8)
    save(fig, out, "fig_service" + ("" if weight == "inv" else "_wprio"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="."); ap.add_argument("--out", default="figures")
    ap.add_argument("--trace-old", default="results/v3/figtrace"); ap.add_argument("--trace-new", default="results/v3/figtrace")
    ap.add_argument("--val-lb", default="results/v3/margin_tuning_summary/lb_val_100_109.json", help="comma-separated LB json files of the validation seeds")
    ap.add_argument("which", nargs="*")
    a = ap.parse_args()
    D = Data(a.repo, [f if os.path.isabs(f) else os.path.join(a.repo, f) for f in a.val_lb.split(",") if f])
    todo = a.which or ["gap_util", "margin", "policy_time", "sensitivity", "ablation2", "trajectory2", "weight", "service"]
    for w in todo:
        if w == "trajectory2": fig_trajectory2(D, a.out, a.trace_old, a.trace_new)
        elif w == "service_wprio": fig_service(D, a.out, "prio")
        else: globals()[f"fig_{w}"](D, a.out)
