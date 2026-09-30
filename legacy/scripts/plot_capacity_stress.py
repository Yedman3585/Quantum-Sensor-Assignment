import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FORMULATION_ORDER = ["AO-QUBO", "Static-QCP-QUBO", "PRC-QUBO", "RC-Greedy-20"]
PLOT_ORDER = ["Static-QCP-QUBO", "AO-QUBO", "PRC-QUBO", "RC-Greedy-20"]
COLORS = {
    "AO-QUBO": "#1557c0",
    "Static-QCP-QUBO": "#e8752a",
    "PRC-QUBO": "#65a83a",
    "RC-Greedy-20": "#7b4db3",
}
MARKERS = {
    "AO-QUBO": "o",
    "Static-QCP-QUBO": "s",
    "PRC-QUBO": "^",
    "RC-Greedy-20": "D",
}
LINESTYLES = {
    "AO-QUBO": (0, (1.0, 1.4)),
    "Static-QCP-QUBO": "-",
    "PRC-QUBO": "-",
    "RC-Greedy-20": ":",
}
LINEWIDTHS = {
    "AO-QUBO": 2.3,
    "Static-QCP-QUBO": 3.0,
    "PRC-QUBO": 2.4,
    "RC-Greedy-20": 2.2,
}
MARKER_SIZES = {
    "AO-QUBO": 8.0,
    "Static-QCP-QUBO": 7.2,
    "PRC-QUBO": 6.8,
    "RC-Greedy-20": 6.0,
}
ALPHAS = {
    "AO-QUBO": 0.95,
    "Static-QCP-QUBO": 1.0,
    "PRC-QUBO": 1.0,
    "RC-Greedy-20": 1.0,
}
ZORDERS = {
    "AO-QUBO": 4,
    "Static-QCP-QUBO": 3,
    "PRC-QUBO": 5,
    "RC-Greedy-20": 4,
}


def load_latest_summaries(log_root, solver, n_cameras, final_opt, formulations=None):
    allowed_formulations = set(formulations) if formulations else None
    selected = {}
    for path in Path(log_root).glob("**/summary_*.json"):
        with path.open("r", encoding="utf-8") as fh:
            row = json.load(fh)
        if int(row.get("n_cameras", -1)) != n_cameras:
            continue
        if bool(row.get("final_opt", False)) != final_opt:
            continue
        row_solver = row.get("solver")
        formulation = row.get("formulation")
        if allowed_formulations is not None and formulation not in allowed_formulations:
            continue
        if formulation == "RC-Greedy-20":
            solver_match = True
        else:
            solver_match = row_solver == solver
        if not solver_match:
            continue
        key = (formulation, float(row.get("capacity_scale", 0.0)))
        previous = selected.get(key)
        if previous is None or str(row.get("run_id", "")) > str(previous.get("run_id", "")):
            row["_summary_path"] = str(path)
            selected[key] = row
    return list(selected.values())


def write_csv(rows, csv_path):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "formulation",
        "solver",
        "capacity_scale",
        "utilization_percent",
        "capacity_surplus_ratio",
        "coverage_percent",
        "covered_cameras",
        "objective_value",
        "assignment_cost",
        "uncovered_penalty",
        "overload_penalty",
        "capacity_rejected_raw_assignments",
        "failed_batches",
        "avg_feasible_candidate_pairs_per_batch",
        "avg_qubo_coefficient_count",
        "total_time_sec",
        "throughput_cam_per_sec",
        "run_id",
        "_summary_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (FORMULATION_ORDER.index(item["formulation"]), item["capacity_scale"])):
            writer.writerow({field: row.get(field, "") for field in fields})


def plot(rows, output_path, solver):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=600)
    metrics = [
        ("coverage_percent", "Coverage (%)", "Coverage under tighter capacity"),
        ("objective_value", "Evaluation objective (log scale)", "Objective under tighter capacity"),
        ("capacity_rejected_raw_assignments", "Capacity-rejected raw assignments", "Rejected decoded assignments"),
        ("avg_feasible_candidate_pairs_per_batch", "Avg. feasible candidate pairs/batch", "Feasible candidate options"),
    ]
    for ax, (metric, ylabel, title) in zip(axes.flat, metrics):
        present_formulations = [name for name in PLOT_ORDER if any(row.get("formulation") == name for row in rows)]
        for formulation in present_formulations:
            series = [row for row in rows if row.get("formulation") == formulation]
            if not series:
                continue
            series = sorted(series, key=lambda row: row["utilization_percent"])
            x = [row["utilization_percent"] for row in series]
            y = [row[metric] for row in series]
            ax.plot(
                x,
                y,
                label=formulation,
                color=COLORS[formulation],
                marker=MARKERS[formulation],
                linestyle=LINESTYLES[formulation],
                linewidth=LINEWIDTHS[formulation],
                markersize=MARKER_SIZES[formulation],
                markerfacecolor="white" if formulation == "AO-QUBO" else COLORS[formulation],
                markeredgecolor=COLORS[formulation],
                markeredgewidth=1.7 if formulation == "AO-QUBO" else 0.8,
                alpha=ALPHAS[formulation],
                zorder=ZORDERS[formulation],
            )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Utilization (%)")
        ax.set_ylabel(ylabel)
        if metric == "objective_value":
            ax.set_yscale("log")
        ax.grid(True, alpha=0.28)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    labels = [name for name in FORMULATION_ORDER if name in by_label]
    handles = [by_label[name] for name in labels]
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.045), ncol=4, frameon=False)
    title_suffix = f"{solver} backend" if any(row.get("solver") == solver for row in rows) else "mixed backend"
    fig.suptitle(
        f"Capacity-stress QUBO formulation comparison on the 20,000 x 800 benchmark ({title_suffix})",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.012,
        "Capacities are uniformly scaled after data generation; raw QUBO energies are not compared across formulations.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=[0.02, 0.13, 1.0, 0.94])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot capacity-stress summaries from JSON logs")
    parser.add_argument("--log-root", default="logs_capacity_stress")
    parser.add_argument("--solver", choices=["SQA", "SA"], default="SQA")
    parser.add_argument("--n-cameras", type=int, default=20000)
    parser.add_argument("--final-opt", action="store_true")
    parser.add_argument(
        "--qubo-only",
        action="store_true",
        help="Plot only AO-QUBO, Static-QCP-QUBO, and PRC-QUBO.",
    )
    parser.add_argument("--csv", default="logs_capacity_stress/capacity_stress_sqa_latest.csv")
    parser.add_argument("--output", default="manuscript_revision/figures/capacity_stress_sqa_600dpi.png")
    args = parser.parse_args()

    formulations = ["AO-QUBO", "Static-QCP-QUBO", "PRC-QUBO"] if args.qubo_only else None
    rows = load_latest_summaries(args.log_root, args.solver, args.n_cameras, args.final_opt, formulations)
    if not rows:
        raise SystemExit("No matching capacity-stress summary logs found.")
    write_csv(rows, args.csv)
    plot(rows, args.output, args.solver)
    print(f"rows={len(rows)}")
    print(f"csv={args.csv}")
    print(f"figure={args.output}")


if __name__ == "__main__":
    main()
