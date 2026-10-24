"""All figures of the manuscript, generated from the archived run logs (vector PDF + 300 dpi PNG).

Usage (repository root):
    python qsa/make_figures.py --repo . --out figures            # every figure
    python qsa/make_figures.py --repo . --out figures gap_util   # selected figures

Main text:      fig1_method, fig2_gap_util, fig3_mechanism, fig4_margin, fig5_policy_time, fig6_ablation
Supplementary:  figS1_sensitivity, figS2_mechanism_u95, figS3_weight, figS4_service_wprio

Style (Nature / Scientific Reports figure guidelines): 183 mm (double column) or 89 mm (single column) wide,
Arial-compatible sans-serif 6-7 pt, no titles inside the figure (panel letters only; descriptions go in the
caption), fonts embedded. Colour encodes the coordination policy and is identical in every figure;
utilisation is encoded by marker fill (95 % hollow, 98 % filled), never by colour. Error bars are
95 % t-intervals over seeds.
"""
import argparse, collections, csv, glob, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

MM = 1 / 25.4
W2, W1 = 183 * MM, 89 * MM                      # double / single column width
BLUE, ORANGE, AQUA, AMBER, MAGENTA, GREEN, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#c98c00", "#e87ba4", "#008300", "#4a3aa7"
INK, SEC, MUTED, GRID = "#1a1a1a", "#4d4d4d", "#8a8983", "#e8e7e2"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "mathtext.fontset": "custom", "mathtext.rm": "sans", "mathtext.it": "sans:italic", "mathtext.bf": "sans:bold", "mathtext.cal": "sans", "mathtext.sf": "sans",
    "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.edgecolor": SEC, "axes.labelcolor": INK, "xtick.color": SEC, "ytick.color": SEC, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.axisbelow": True, "axes.linewidth": 0.6, "lines.linewidth": 1.2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "lines.markersize": 4, "errorbar.capsize": 1.8, "legend.frameon": False, "legend.handlelength": 2.2,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42, "ps.fonttype": 42,
})
# one colour / marker / line style per coordination policy, identical in every figure and in the text
ENT = {
    "static":   dict(c=ORANGE,  m="v", ls="-",  lab="Static prices"),
    "reprice":  dict(c=AQUA,    m="s", ls="--", lab="Re-pricing"),
    "reprice5": dict(c=BLUE,    m="o", ls="-",  lab="Re-pricing + 5%"),
    "twin5":    dict(c=MAGENTA, m="D", ls="-.", lab="Statistics + 5%"),
    "lp":       dict(c=AMBER,   m="^", ls=":",  lab="LP prices"),
    "lp5":      dict(c=AMBER,   m="^", ls="-",  lab="LP prices + 5%"),
    "dmd":      dict(c=GREEN,   m="P", ls="-",  lab="Dual mirror descent"),
    "rh":       dict(c=VIOLET,  m="X", ls="-",  lab="Rolling horizon"),
    "offline":  dict(c=MUTED,   m="*", ls=":",  lab="Offline solution"),
}
U4 = [75.0, 90.0, 95.0, 98.0]
YLAB = "Gap to lower bound (%)"
LOGTICKS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]


# ------------------------------------------------------------------ helpers
def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    fig.savefig(f"{out}/{name}.pdf"); fig.savefig(f"{out}/{name}.png", dpi=300); plt.close(fig)
    print("wrote", name)


def letters(fig, axs, dx=0.0):
    """Bold panel letters at the top-left corner of each panel's tight bounding box."""
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    for ax, L in zip(axs, "abcdefgh"):
        tb = ax.get_tightbbox(r).transformed(fig.transFigure.inverted())
        top = ax.get_position().y1
        fig.text(tb.x0 + dx, top + 0.015, L, fontsize=8, fontweight="bold", ha="left", va="bottom")


def tag(ax, text, loc="upper left"):
    x, ha = (0.02, "left") if "left" in loc else (0.98, "right")
    y, va = (0.98, "top") if "upper" in loc else (0.03, "bottom")
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=6.5, color=SEC)


def logaxis(ax, lo, hi, axis="y"):
    t = [v for v in LOGTICKS if lo <= v <= hi]
    if axis == "y":
        ax.set_yscale("log"); ax.set_ylim(lo, hi); ax.set_yticks(t); ax.set_yticklabels([f"{v:g}" for v in t])
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    else:
        ax.set_xscale("log"); ax.set_xlim(lo, hi); ax.set_xticks(t); ax.set_xticklabels([f"{v:g}" for v in t])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def ci95(x):
    x = np.asarray(x, float); n = len(x)
    if n < 2: return 0.0
    t = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36, 9: 2.31, 10: 2.26}.get(n, 1.96)
    return t * x.std(ddof=1) / np.sqrt(n)


def mfc(ent, u):
    """marker face: hollow for 95 % (and 90 %), filled for 98 %"""
    return ENT[ent]["c"] if u >= 98 else "white"


def handle(ent, **kw):
    st = ENT[ent]
    return Line2D([], [], color=st["c"], marker=st["m"], ls=st["ls"], label=st["lab"], **kw)


def util_handles():
    return [Line2D([], [], color=SEC, marker="o", ls="none", mfc="white", label="95% utilisation"),
            Line2D([], [], color=SEC, marker="o", ls="none", mfc=SEC, label="98% utilisation")]


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
        for f in val_lb or []:
            for d in json.load(open(f)): self.LB[("uniform", 20000, d["target_util"], d["seed"])] = d["lb"]

    def gap(self, d, obj=None):
        lb = self.LB.get((d.get("layout", "uniform"), d["n_cameras"], d["target_util"], d["seed"]))
        return None if lb is None else ((d["objective"] if obj is None else obj) - lb) / lb * 100

    def runs(self, sub, pred=lambda d: True):
        out = []
        for f in glob.glob(f"{self.V}/{sub}/*.json"):
            d = json.load(open(f))
            if "rows" in d: continue
            d.setdefault("layout", "uniform")
            if pred(d) and self.gap(d) is not None: out.append(d)
        return out

    def service(self):
        S = f"{self.V}/service_summary"
        return list(csv.DictReader(open(f"{S}/service_runs.csv"))), json.load(open(f"{S}/saturation_curves.json"))


def cond(d):
    if not d.get("reprice"): return "static"
    b = d.get("forecast_bias") or 0; f = d.get("forecast", "oracle")
    if f == "oracle": return "reprice5" if abs(b - 0.05) < 1e-9 else ("reprice" if b == 0 else f"oracle{b:+g}")
    if f == "twin": return "twin5" if abs(b - 0.05) < 1e-9 else ("twin" if b == 0 else f"twin{b:+g}")
    return f"{f}{d.get('forecast_level', 0):g}{b:+g}"


def by(runs, key):
    g = collections.defaultdict(dict)
    for d in runs: g[key(d)][d["seed"]] = d          # one run per seed (duplicates from two machines are identical)
    return g


# ------------------------------------------------------------------ Fig. 1: method schematic
def fig_method(D, out):
    fig = plt.figure(figsize=(W2, 72 * MM)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 183); ax.set_ylim(0, 72); ax.axis("off")
    TINT, NEUT = "#eaf2fc", "#f4f3ef"

    def band(x0, y0, x1, y1, fc, title, top=True):
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=2", fc=fc, ec="none"))
        ax.text(x0 + 2.5, y1 - 2.2 if top else y0 + 2.0, title, fontsize=7, fontweight="bold", va="top" if top else "bottom", color=INK)

    def box(x0, y0, x1, y1, head, body, ec=SEC, lw=0.7, fc="white"):
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec=ec, lw=lw))
        ax.text((x0 + x1) / 2, y1 - 2.0, head, ha="center", va="top", fontsize=6.8, fontweight="bold", color=INK)
        ax.text((x0 + x1) / 2, y1 - 6.0, body, ha="center", va="top", fontsize=6.0, color=SEC, linespacing=1.35)

    def arrow(xy0, xy1, color=SEC, lw=0.8, ls="-"):
        ax.annotate("", xy=xy1, xytext=xy0, arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls, mutation_scale=7, shrinkA=0, shrinkB=0))

    def line(pts, color=SEC, lw=0.8, ls="-"):
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=lw, ls=ls, solid_capstyle="butt")

    band(3, 41, 176, 66, TINT, r"Coordination layer — every $T_p=5$ batches")
    band(3, 2, 176, 37, NEUT, r"Batch layer — every batch of $B=80$ sensors; decisions are final", top=False)
    # coordination layer
    box(8, 44, 55, 60, "Remaining demand",
        "exact (known sensors) or statistics only\n(independent draw from the same generator)\n" + r"loads $\times(1+b)$, demand margin $b=5\%$", ec=BLUE)
    box(68, 44, 115, 60, "Residual Lagrangian prices",
        "maximise the Lagrangian bound of the\nresidual problem over $u\\geq0$:\n300 subgradient steps, 40 servers/sensor", ec=BLUE)
    box(128, 44, 171, 60, r"Capacity prices $u_j$",
        "decision cost\n" + r"$\hat c_{ij}=w_ic_{ij}+u_j\,l_i$" + "\nused by every batch solver", ec=BLUE)
    arrow((55, 52), (68, 52), BLUE); arrow((115, 52), (128, 52), BLUE)
    # batch layer
    box(7, 12.5, 34, 30, "Sensor stream", "ordered by priority\n" + r"$\times$ load, $p_i l_i$" + "\n250 batches at 20k")
    box(40, 12.5, 73, 30, "Candidate windows", "5 cheapest + 2 headroom\nservers per sensor\n" + r"under $\hat c_{ij}$")
    box(79, 12.5, 108, 30, "Exact reduction", "uncontested sensors fixed\n(Proposition 2); median\n2 of 80 sensors stay free")
    box(114, 12.5, 147, 30, "Batch solver", "QUBO-v3.1: domain wall,\nslack-free capacity, SQA/SA\n+ local search; or MILP, greedy", ec=INK, lw=1.0)
    box(153, 12.5, 172, 30, "Commit", "assign the batch,\nupdate residual\ncapacities $R$")
    for x0, x1 in [(34, 40), (73, 79), (108, 114), (147, 153)]:
        arrow((x0, 21), (x1, 21))
    # prices feed the windows and the solver
    line([(149.5, 44), (149.5, 39.5), (56.5, 39.5)], BLUE); arrow((56.5, 39.5), (56.5, 30), BLUE); arrow((130.5, 39.5), (130.5, 30), BLUE)
    ax.text(103, 40.2, r"prices $u$", fontsize=6.0, color=BLUE, ha="center", va="bottom")
    # residual capacities feed the next re-pricing
    line([(172, 21), (179, 21), (179, 69), (91.5, 69)], SEC, ls=(0, (3, 1.5))); arrow((91.5, 69), (91.5, 60), SEC)
    ax.text(135, 69.6, r"residual capacities $R$ and remaining sensors", fontsize=6.0, color=SEC, ha="center", va="bottom")
    save(fig, out, "fig1_method")


# ------------------------------------------------------------------ Fig. 2: gap vs utilisation
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
    fig, axs = plt.subplots(1, 2, figsize=(W2, 62 * MM), sharey=True, gridspec_kw={"wspace": 0.08, "width_ratios": [1.25, 1]})
    order = ["static", "reprice", "twin5", "lp5", "reprice5"]
    for ax, data, txt, xt, show_off in [(axs[0], S, "Uniform layout", U4, True), (axs[1], C, "Clustered urban layout", U4[1:], False)]:
        for k in order:
            if k not in data: continue
            us = [u for u in xt if u in data[k] and len(data[k][u]) >= 3]
            if not us: continue
            g = [[D.gap(x) for x in data[k][u].values()] for u in us]
            st = ENT[k]
            ax.errorbar(us, [np.mean(v) for v in g], yerr=[ci95(v) for v in g], color=st["c"], marker=st["m"], ls=st["ls"],
                        mfc=st["c"], mec=st["c"], lw=1.2, elinewidth=0.8, zorder=3)
        if show_off:
            og = [np.mean([(D.OFF[("uniform", 20000, u, s)] - D.LB[("uniform", 20000, u, s)]) / D.LB[("uniform", 20000, u, s)] * 100
                           for s in range(42, 52) if ("uniform", 20000, u, s) in D.OFF]) for u in U4]
            st = ENT["offline"]; ax.plot(U4, og, color=st["c"], marker=st["m"], ls=st["ls"], ms=5.5, lw=1.0, zorder=2)
        ax.set_xticks(xt); ax.set_xlim(xt[0] - (2.5 if len(xt) == 4 else 1.5), 99.3); ax.set_xlabel("Utilisation (%)")
        tag(ax, txt)
    logaxis(axs[0], 0.2, 30); axs[0].set_ylabel(YLAB)
    hs = [handle(k) for k in ["static", "reprice", "twin5", "lp5", "reprice5", "offline"]]
    fig.legend(handles=hs, loc="upper center", ncol=6, bbox_to_anchor=(0.5, -0.05), handlelength=2.6, columnspacing=1.2)
    letters(fig, axs)
    save(fig, out, "fig2_gap_util")


# ------------------------------------------------------------------ Fig. 3 / S2: where the loss arises
def fig_mechanism(D, out, u=98.0, tr="results/v3/figtrace"):
    rows, curves = D.service()
    G = {o["util"]: o for o in json.load(open(f"{tr}/global_assign_seed42.json"))}
    fig, axs = plt.subplots(1, 3, figsize=(W2, 58 * MM), gridspec_kw={"wspace": 0.42, "width_ratios": [1, 1, 0.95]})
    pols = ["static", "reprice", "twin5", "reprice5"]
    # a: cumulative excess over the offline solution, seed 42
    ax = axs[0]; gcost = np.array(G[u]["cost"]); lb = G[u]["lb"]
    pats = {"static": "traj_exact_priced_percam_20000x800_u%g_seed42_*.json",
            "reprice": "traj_exact_priced_rp5_ks2_percam_20000x800_u%g_seed42_*.json",
            "twin5": "traj_exact_priced_rp5_ks2_fc-twin0+0.05_percam_20000x800_u%g_seed42_*.json",
            "reprice5": "traj_exact_priced_rp5_ks2_fc-oracle0+0.05_percam_20000x800_u%g_seed42_*.json"}
    ends = []
    for k in pols:
        fs = sorted(glob.glob(f"{tr}/" + pats[k] % u))
        if not fs: continue
        acc, ex = 0.0, []
        for r in json.load(open(fs[-1]))["trace"]:
            acc += r["objective"] - gcost[r["cams"]].sum(); ex.append(acc / lb * 100)
        st = ENT[k]; ax.plot(np.arange(1, len(ex) + 1), ex, color=st["c"], ls=st["ls"], lw=1.2); ends.append((ex[-1], len(ex), st["c"]))
    placed = []
    for v, n, c in sorted(ends):
        y = v if not placed else max(v, placed[-1] + 1.3); placed.append(y)
        ax.text(n + 4, y, f"{v:.1f}", fontsize=6, color=c, va="center")
    ax.axhline(0, color=MUTED, lw=0.6); ax.set_xlim(0, 272); ax.set_yticks([0, 5, 10, 15]); ax.set_xticks([0, 50, 100, 150, 200, 250])
    ax.set_xlabel("Batch"); ax.set_ylabel("Cumulative excess over offline (%)"); tag(ax, "seed 42")
    # b: share of capacity on saturated servers, mean over seeds
    ax = axs[1]
    for k in pols:
        cs = [np.array(v) for key, v in curves.items() if key.startswith(f"inv|{k}|") and key.endswith(f"|{u:g}")]
        if not cs: continue
        n = min(len(c) for c in cs); st = ENT[k]
        ax.plot(np.arange(1, n + 1), np.mean([c[:n] for c in cs], 0) * 100, color=st["c"], ls=st["ls"], lw=1.2)
    ax.set_xlim(0, 255); ax.set_ylim(0, 100); ax.set_xticks([0, 50, 100, 150, 200, 250])
    ax.set_xlabel("Batch"); ax.set_ylabel("Capacity on full servers (%)"); tag(ax, "seeds 42–46")
    # c: mean distance by priority class
    ax = axs[2]; cls = ["high", "medium", "low"]
    for i, cl in enumerate(cls):
        for j, k in enumerate(pols + ["offline"]):
            v = [float(r["dist"]) for r in rows if r["weight"] == "inv" and r["policy"] == k and float(r["util"]) == u and r["class"] == cl]
            if not v: continue
            st = ENT[k]
            ax.plot(np.mean(v), i + (j - 2) * 0.12, marker=st["m"], color=st["c"], ls="none", ms=6.5 if k == "offline" else 4.5, mec=st["c"])
    ax.set_yticks(range(3)); ax.set_yticklabels([f"{c.capitalize()}\npriority" for c in cls]); ax.invert_yaxis(); ax.set_ylim(2.5, -0.5)
    ax.grid(axis="y", visible=False); ax.set_xlabel("Mean sensor–server distance"); tag(ax, "seeds 42–46", "lower right")
    hs = [handle(k) for k in pols + ["offline"]]
    fig.legend(handles=hs, loc="upper center", ncol=5, bbox_to_anchor=(0.5, -0.06), handlelength=2.6, columnspacing=1.4)
    letters(fig, axs)
    save(fig, out, "fig3_mechanism" if u == 98.0 else f"figS2_mechanism_u{u:g}")


# ------------------------------------------------------------------ Fig. 4: margin selection and value of information
def fig_margin(D, out):
    tun = D.runs("margin_tuning", lambda d: d["method"] == "exact")
    fig, axs = plt.subplots(1, 2, figsize=(W2, 66 * MM), gridspec_kw={"wspace": 0.55, "width_ratios": [1, 1.05]})
    ax = axs[0]; B = [0, 5, 10, 15, 20, 30]
    for fc, ent in [("oracle", "reprice5"), ("twin", "twin5")]:
        st = ENT[ent]; ends = []
        for u in (90.0, 95.0, 98.0):
            m = []
            for b in B:
                g = [D.gap(d) for d in tun if d.get("forecast", "oracle") == fc and abs((d.get("forecast_bias") or 0) * 100 - b) < 1e-6 and d["target_util"] == u]
                m.append(np.mean(g) if g else np.nan)
            a = {90.0: 0.35, 95.0: 0.65, 98.0: 1.0}[u]
            ax.plot(B, m, color=st["c"], alpha=a, marker=st["m"], ls=st["ls"], ms=3.2, mfc=st["c"] if u == 98 else "white", lw=1.1)
            ends.append([m[-1], u, a])
        ends.sort(); 
        for i in range(1, len(ends)): ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.21)
        for y, u, a in ends:
            ax.text(30.8, y, f"{u:g}%", fontsize=5.8, color=st["c"], alpha=max(a, 0.6), va="center")
    ax.axvline(5, color=MUTED, ls=":", lw=0.8); ax.text(5.5, 5.15, r"$b=5\%$", fontsize=6, color=SEC, va="top")
    ax.set_xlabel(r"Demand margin $b$ (%)"); ax.set_ylabel(YLAB); ax.set_xticks(B); ax.set_xlim(-1.5, 34); ax.set_ylim(0, 5.3)
    ax.legend(handles=[Line2D([], [], color=BLUE, marker="o", label="Exact demand"), Line2D([], [], color=MAGENTA, marker="D", ls="-.", label="Demand statistics")],
              loc="upper right", bbox_to_anchor=(0.97, 1.0)); tag(ax, "validation seeds 100–109", "lower right")
    # b: test seeds
    ax = axs[1]
    rows = [("Static prices", "static", ORANGE), ("Re-pricing", "reprice", AQUA), ("Re-pricing + 5%", "reprice5", BLUE),
            ("Demand −10%", "oracle-0.1", MUTED), ("Load noise ±30%", "noise0.3+0", MUTED),
            ("Demand statistics", "twin", MAGENTA), ("Statistics + 5%", "twin5", MAGENTA)]
    allr = D.runs("forecast", lambda d: d["method"] == "exact") + D.runs("margin_test", lambda d: d["method"] == "exact") + \
        D.runs("campaign3/v31", lambda d: d["method"] == "exact" and "reprice_top" in d)
    G = collections.defaultdict(dict)
    for d in allr: G[(cond(d), d["target_util"])][d["seed"]] = D.gap(d)
    for i, (lab, c, col) in enumerate(rows):
        for u, off in [(95.0, -0.16), (98.0, 0.16)]:
            v = list(G[(c, u)].values())
            if not v: continue
            y = i + off
            ax.scatter(v, [y] * len(v), s=6, color=col, alpha=0.35, lw=0, zorder=2)
            ax.scatter([np.mean(v)], [y], s=26, facecolor=col if u == 98 else "white", edgecolor=col, lw=1.0, zorder=3)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    logaxis(ax, 0.5, 30, "x"); ax.set_xlabel(YLAB); tag(ax, "test seeds 42–51", "lower right")
    ax.legend(handles=util_handles(), loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, handletextpad=0.2)
    letters(fig, axs)
    save(fig, out, "fig4_margin")


# ------------------------------------------------------------------ Fig. 5: quality and time (one machine)
def fig_policy_time(D, out):
    fig, axs = plt.subplots(1, 2, figsize=(W2, 62 * MM), gridspec_kw={"wspace": 0.45, "width_ratios": [1, 1.05]})
    ax = axs[0]; u = 95.0
    tim = D.runs("timing_asus", lambda d: d["target_util"] == u)
    ob = D.runs("online_baselines", lambda d: d["target_util"] == u)
    pts = [("static", [d for d in tim if "policy" not in d and d["method"] == "exact" and not d.get("reprice")], (0, 9)),
           ("reprice", D.runs("campaign3/v31", lambda d: d["method"] == "exact" and "reprice_top" in d and d["target_util"] == u), (-7, 5)),
           ("reprice5", [d for d in tim if "policy" not in d and d["method"] == "exact" and d.get("reprice")], (0, -11)),
           ("lp5", [d for d in tim if d.get("policy") == "lp"], (-9, 0)),
           ("dmd", [d for d in ob if d.get("policy") == "dmd" and d.get("dmd_init") == "lagrange"], (7, 0)),
           ("rh", [d for d in ob if d.get("policy") == "lookahead"], (0, 9))]
    for k, r, (dx, dy) in pts:
        if not r: continue
        g = [D.gap(d) for d in r]; t = [d["time_sec"] for d in r]; st = ENT[k]
        ax.errorbar(np.mean(t), np.mean(g), xerr=ci95(t), yerr=ci95(g), color=st["c"], marker=st["m"], ms=5, lw=0.8, ls="none", mec=st["c"])
        ax.annotate(st["lab"], (np.mean(t), np.mean(g)), xytext=(dx, dy), textcoords="offset points", fontsize=6,
                    color=st["c"], ha="left" if dx > 0 else ("right" if dx < 0 else "center"), va="center")
    ax.set_xscale("log"); ax.set_xlim(4, 6000); ax.set_xticks([10, 100, 1000]); ax.set_xticklabels(["10", "100", "1,000"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    logaxis(ax, 0.5, 50); ax.set_xlabel("Time per instance (s)"); ax.set_ylabel(YLAB); tag(ax, "95% utilisation", "upper right")
    # b: batch solvers inside the final policy
    ax = axs[1]; tim = D.runs("timing_asus")
    meth = [("exact", "Exact MILP"), ("greedy", "Greedy"), ("regret", "Regret"), ("sa_v3", "SA (QUBO-v3.1)"),
            ("ils", "ILS"), ("sqa_v3", "SQA (QUBO-v3.1)"), ("sqa_v3+LS", "SQA + local search")]
    for j, (key, lab) in enumerate(meth):
        for u, off in [(95.0, -0.19), (98.0, 0.19)]:
            r = [d for d in tim if "policy" not in d and d.get("reprice") and d["target_util"] == u and d["method"] + ("+LS" if d.get("polish") else "") == key]
            if not r: continue
            t = np.mean([d["time_sec"] for d in r]); g = np.mean([D.gap(d) for d in r])
            ax.barh(j + off, t, height=0.34, color=BLUE if u == 98 else "white", edgecolor=BLUE, lw=0.8)
            ax.text(t + 12, j + off, f"{g:.2f}%", va="center", fontsize=5.8, color=SEC)
    ax.set_yticks(range(len(meth))); ax.set_yticklabels([m[1] for m in meth]); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Time per instance (s)"); ax.set_xlim(0, 760)
    ax.legend(handles=[matplotlib.patches.Patch(fc="white", ec=BLUE, label="95% utilisation"), matplotlib.patches.Patch(fc=BLUE, ec=BLUE, label="98% utilisation")],
              loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=2, handlelength=1.4)
    letters(fig, axs)
    save(fig, out, "fig5_policy_time")


# ------------------------------------------------------------------ Fig. 6: encoding ablation on hard batches
def fig_ablation(D, out):
    def collect(dirs):
        rows = collections.defaultdict(list)
        for sub, tg in dirs:
            for f in glob.glob(f"{D.V}/{sub}/*.json"):
                for r in json.load(open(f))["rows"]:
                    for k, v in r.items():
                        if k in ("batch", "exact") or (":" in k and k.count(":") != 1): continue
                        if ":" not in k and tg != "plain": continue
                        rows[(tg + "|" + k) if ":" in k else k].append((v - r["exact"]) / r["exact"] * 100)
        return {k: np.mean(v) for k, v in rows.items()}, max(len(v) for v in rows.values())
    A0, n0 = collect([("campaign2/ablation", "plain"), ("campaign2/ablation_adaptive", "adaptive")])
    A1, n1 = collect([("ablation_v31/plain", "plain"), ("ablation_v31/adaptive", "adaptive")])
    variants = [("plain|base", "One-hot"), ("plain|R", "+ reduction (R)"), ("plain|DW", "+ domain wall (DW)"),
                ("plain|R+DW", "R + DW"), ("adaptive|R+DW", "R + DW + adaptive")]
    fig, axs = plt.subplots(1, 2, figsize=(W2, 56 * MM), sharey=True, sharex=True, gridspec_kw={"wspace": 0.12})
    for ax, A, txt in [(axs[0], A0, f"static prices, {n0} hard batches"), (axs[1], A1, f"re-pricing, {n1} hard batches")]:
        for key, lab, ls in [("greedy", "greedy", ":"), ("regret_ls", "regret + LS", "--")]:
            if key in A:
                ax.axvline(A[key], color=MUTED, ls=ls, lw=0.8, zorder=1)
                ax.text(A[key] * 1.08, -0.75, lab, color=SEC, fontsize=5.8, ha="left", va="bottom")
        for i, (k, lab) in enumerate(variants):
            q, s_ = A.get(k + ":SQA"), A.get(k + ":SA")
            if q is None: continue
            ax.plot([s_, q], [i, i], color=GRID, lw=2.4, zorder=2, solid_capstyle="round")
            ax.plot(s_, i, marker="s", ms=4.6, mfc="white", mec=SEC, mew=1.0, ls="none", zorder=3)
            ax.plot(q, i, marker="o", ms=4.6, color=INK, ls="none", zorder=3)
        ax.grid(axis="y", visible=False); tag(ax, txt, "lower right")
        ax.set_xlabel("Gap to batch optimum (%)")
    logaxis(axs[0], 0.007, 14, "x"); axs[0].set_xticks([0.01, 0.1, 1, 10]); axs[0].set_xticklabels(["0.01", "0.1", "1", "10"])
    axs[0].set_yticks(range(len(variants))); axs[0].set_yticklabels([v[1] for v in variants]); axs[0].set_ylim(len(variants) - 0.4, -0.95)
    fig.legend(handles=[Line2D([], [], marker="s", mfc="white", mec=SEC, ls="none", label="Simulated annealing (SA)"),
                        Line2D([], [], marker="o", color=INK, ls="none", label="Simulated quantum annealing (SQA)")],
               loc="upper center", ncol=2, bbox_to_anchor=(0.55, -0.07))
    letters(fig, axs)
    save(fig, out, "fig6_ablation")


# ------------------------------------------------------------------ Fig. S1: batch size and re-pricing period
def fig_sensitivity(D, out):
    def series(sub, u):
        g = collections.defaultdict(list)
        for d in D.runs(sub, lambda d: d["method"] == "exact" and d["target_util"] == u):
            g[(d.get("batch_size") or d["params"].get("batch_size", 80), d.get("reprice"))].append(D.gap(d))
        return g
    ref = {"reprice": {u: [D.gap(d) for d in D.runs("campaign3/v31", lambda d: d["method"] == "exact" and "reprice_top" in d and d["target_util"] == u)] for u in (95.0, 98.0)},
           "reprice5": {u: [D.gap(d) for d in D.runs("margin_test", lambda d: d["method"] == "exact" and cond(d) == "reprice5" and d["target_util"] == u)] for u in (95.0, 98.0)}}
    fig, axs = plt.subplots(1, 2, figsize=(W2, 56 * MM), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, var, xs, xl in [(axs[0], "T", [1, 5, 10, 25], r"Re-pricing period $T_p$ (batches); $B=80$"),
                            (axs[1], "B", [40, 80, 160, 320], r"Batch size $B$ (sensors); $T_p=5$")]:
        for sub, ent in [("sensitivity", "reprice"), ("sensitivity_m05", "reprice5")]:
            for u in (95.0, 98.0):
                g = series(sub, u); ys = []
                for x in xs:
                    k = (80, x) if var == "T" else (x, 5)
                    v = ref[ent][u] if k == (80, 5) else g.get(k, [])
                    ys.append(np.mean(v) if v else np.nan)
                st = ENT[ent]
                ax.plot(xs, ys, color=st["c"], marker=st["m"], ls=st["ls"], mfc=mfc(ent, u), mec=st["c"], alpha=1.0 if u == 98 else 0.7)
        ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels([str(x) for x in xs]); ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.axvline(5 if var == "T" else 80, color=MUTED, ls=":", lw=0.8); ax.set_xlabel(xl)
    logaxis(axs[0], 0.5, 10); axs[0].set_ylabel(YLAB)
    fig.legend(handles=[handle("reprice"), handle("reprice5")] + util_handles(), loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.07))
    letters(fig, axs)
    save(fig, out, "figS1_sensitivity")


# ------------------------------------------------------------------ Fig. S3: priority weight w = p
def fig_weight(D, out):
    LBp = {(d["target_util"], d["seed"]): d["lb"] for d in json.load(open(f"{D.V}/weight_prio_summary/lb_prio_20k.json"))}
    gd = collections.defaultdict(dict)
    for f in glob.glob(f"{D.V}/weight_prio/*.json"):
        d = json.load(open(f))
        if d["method"] != "exact": continue
        lb = LBp[(d["target_util"], d["seed"])]; gd[(cond(d), d["target_util"])][d["seed"]] = (d["objective"] - lb) / lb * 100
    fig, ax = plt.subplots(figsize=(W1, 55 * MM))
    for k, ent, lab in [("static", "static", None), ("reprice", "reprice", None), ("twin", "twin5", "Demand statistics")]:
        us = [u for u in (90.0, 95.0, 98.0) if gd.get((k, u))]
        st = ENT[ent]; v = [list(gd[(k, u)].values()) for u in us]
        ax.errorbar(us, [np.mean(x) for x in v], yerr=[ci95(x) for x in v], color=st["c"], marker=st["m"], ls=st["ls"], mec=st["c"], lw=1.1, label=lab or st["lab"])
    ax.set_xticks([90, 95, 98]); ax.set_xlim(88.5, 99.5); ax.set_xlabel("Utilisation (%)"); logaxis(ax, 0.5, 10); ax.set_ylabel(YLAB)
    ax.legend(loc="upper left"); tag(ax, r"weight $w_i=p_i$", "lower right")
    save(fig, out, "figS3_weight")


# ------------------------------------------------------------------ Fig. S4: service metrics under w = p
def fig_service_wprio(D, out):
    rows, curves = D.service()
    fig, axs = plt.subplots(1, 2, figsize=(W2, 56 * MM), gridspec_kw={"wspace": 0.4})
    pols = ["static", "reprice", "reprice5"]
    ax = axs[0]
    for k in pols:
        cs = [np.array(v) for key, v in curves.items() if key.startswith(f"prio|{k}|") and key.endswith("|98")]
        if not cs: continue
        n = min(len(c) for c in cs); st = ENT[k]
        ax.plot(np.arange(1, n + 1), np.mean([c[:n] for c in cs], 0) * 100, color=st["c"], ls=st["ls"])
    ax.set_xlim(0, 255); ax.set_ylim(0, 100); ax.set_xlabel("Batch"); ax.set_ylabel("Capacity on full servers (%)")
    ax = axs[1]; cls = ["high", "medium", "low"]
    for i, cl in enumerate(cls):
        for j, k in enumerate(pols + ["offline"]):
            v = [float(r["dist"]) for r in rows if r["weight"] == "prio" and r["policy"] == k and float(r["util"]) == 98.0 and r["class"] == cl]
            if not v: continue
            st = ENT[k]; ax.plot(np.mean(v), i + (j - 1.5) * 0.13, marker=st["m"], color=st["c"], ls="none", ms=6.5 if k == "offline" else 4.5)
    ax.set_yticks(range(3)); ax.set_yticklabels([f"{c.capitalize()}\npriority" for c in cls]); ax.set_ylim(2.5, -0.5)
    ax.grid(axis="y", visible=False); ax.set_xlabel("Mean sensor–server distance")
    fig.legend(handles=[handle(k) for k in pols + ["offline"]], loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.07))
    tag(axs[0], r"$w_i=p_i$, 98%, seeds 42–46"); letters(fig, axs)
    save(fig, out, "figS4_service_wprio")


FIGS = ["method", "gap_util", "mechanism", "margin", "policy_time", "ablation", "sensitivity", "mechanism95", "weight", "service_wprio"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="."); ap.add_argument("--out", default="figures")
    ap.add_argument("--val-lb", default="results/v3/margin_tuning_summary/lb_val_100_109.json", help="comma-separated LB files of the validation seeds")
    ap.add_argument("which", nargs="*", help="any of: " + ", ".join(FIGS))
    a = ap.parse_args()
    D = Data(a.repo, [f if os.path.isabs(f) else os.path.join(a.repo, f) for f in a.val_lb.split(",") if f])
    tr = os.path.join(a.repo, "results/v3/figtrace")
    for w in a.which or FIGS:
        if w == "mechanism": fig_mechanism(D, a.out, 98.0, tr)
        elif w == "mechanism95": fig_mechanism(D, a.out, 95.0, tr)
        else: globals()[f"fig_{w}"](D, a.out)
