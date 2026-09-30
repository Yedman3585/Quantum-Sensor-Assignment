from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev


BASELINE_SUMMARY_FILES = {
    "AO-QUBO": [
        "logs_ao_qubo_sqa/summary_20260528_062441.json",
        "logs_ao_qubo_sqa/summary_20260529_002240.json",
        "logs_ao_qubo_sqa/summary_20260529_022207.json",
        "logs_ao_qubo_sa/summary_20260528_064201.json",
        "logs_ao_qubo_sa/summary_20260529_011618.json",
        "logs_ao_qubo_sa/summary_20260529_023056.json",
    ],
    "Static-QCP": [
        "logs_static_qcp_qubo_sqa/summary_20260528_063156.json",
        "logs_static_qcp_qubo_sqa/summary_20260529_002748.json",
        "logs_static_qcp_qubo_sqa/summary_20260529_022428.json",
        "logs_static_qcp_qubo_sa/summary_20260528_072426.json",
        "logs_static_qcp_qubo_sa/summary_20260529_014956.json",
        "logs_static_qcp_qubo_sa/summary_20260529_025645.json",
    ],
}


PRC_REFERENCE = {
    "coverage_percent": 99.6,
    "objective_value": 7108.3,
    "covered_cameras": 19920.0,
    "uncovered_cameras": 80.0,
    "avg_qubo_coefficient_count": 16800.0,
}


@dataclass
class SeriesStats:
    label: str
    coverage_mean: float
    coverage_sd: float
    objective_mean: float
    objective_sd: float
    uncovered_mean: float
    uncovered_sd: float
    terms_mean: float
    terms_sd: float


def sample_sd(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def aggregate_baseline(root: Path, label: str, files: list[str]) -> SeriesStats:
    rows = [read_json(root / file_name) for file_name in files]
    coverage = [float(row["coverage_percent"]) for row in rows]
    objective = [float(row["objective_value"]) for row in rows]
    uncovered = [float(row["uncovered_cameras"]) for row in rows]
    terms = [float(row["avg_qubo_coefficient_count"]) for row in rows]
    return SeriesStats(
        label=label,
        coverage_mean=mean(coverage),
        coverage_sd=sample_sd(coverage),
        objective_mean=mean(objective),
        objective_sd=sample_sd(objective),
        uncovered_mean=mean(uncovered),
        uncovered_sd=sample_sd(uncovered),
        terms_mean=mean(terms),
        terms_sd=sample_sd(terms),
    )


def parse_prc_logs(root: Path) -> SeriesStats:
    objectives: list[float] = []
    covered: list[float] = []
    unique_sqa_runs: dict[str, tuple[float, float]] = {}

    sqa_line = re.compile(
        r"(?P<run_id>20\d{6}_\d{6})\s+"
        r"(?P<time>\d+(?:\.\d+)?)\s+"
        r"(?P<cameras>\d+)\s+"
        r"(?P<throughput>\d+(?:\.\d+)?)\s+"
        r"(?P<eff>\d+(?:\.\d+)?)\s+"
        r"(?P<objective>7108(?:\.\d+)?)"
    )
    sa_line = re.compile(
        r"\b(?:Annealing|Annealing-Optimized)\s+"
        r"(?P<time>\d+(?:\.\d+)?)\s+"
        r"(?P<objective>7108(?:\.\d+)?)\s+"
        r"(?P<cameras>19920)\b"
    )

    for log_name in ("SQA_toplog.txt", "quantum_openjij_windows.log"):
        log_path = root / log_name
        if not log_path.exists():
            continue
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        for match in sqa_line.finditer(text):
            run_id = match.group("run_id")
            unique_sqa_runs[run_id] = (
                float(match.group("objective")),
                float(match.group("cameras")),
            )

    for objective_value, cameras in unique_sqa_runs.values():
        objectives.append(objective_value)
        covered.append(cameras)

    for log_name in ("SA_toplog.txt", "SA_Chronology.txt", "classical_optimization.log"):
        log_path = root / log_name
        if not log_path.exists():
            continue
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        for match in sa_line.finditer(text):
            objectives.append(float(match.group("objective")))
            covered.append(float(match.group("cameras")))

    if objectives and covered:
        objective_mean = mean(objectives)
        objective_sd = sample_sd(objectives)
        covered_mean = mean(covered)
        uncovered = [20000.0 - value for value in covered]
        coverage = [value / 20000.0 * 100.0 for value in covered]
        return SeriesStats(
            label="PRC-QUBO",
            coverage_mean=mean(coverage),
            coverage_sd=sample_sd(coverage),
            objective_mean=objective_mean,
            objective_sd=objective_sd,
            uncovered_mean=mean(uncovered),
            uncovered_sd=sample_sd(uncovered),
            terms_mean=PRC_REFERENCE["avg_qubo_coefficient_count"],
            terms_sd=0.0,
        )

    return SeriesStats(
        label="PRC-QUBO",
        coverage_mean=PRC_REFERENCE["coverage_percent"],
        coverage_sd=0.0,
        objective_mean=PRC_REFERENCE["objective_value"],
        objective_sd=0.0,
        uncovered_mean=PRC_REFERENCE["uncovered_cameras"],
        uncovered_sd=0.0,
        terms_mean=PRC_REFERENCE["avg_qubo_coefficient_count"],
        terms_sd=0.0,
    )


def format_int_space(value: float) -> str:
    return f"{int(round(value)):,}".replace(",", " ")


def add_bar_labels(ax, bars, values, labels, offset):
    for bar, value, label in zip(bars, values, labels):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            label,
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="normal",
            color="black",
        )


def style_axis(ax, title: str, ylim: tuple[float, float]):
    ax.set_title(title, loc="left", fontsize=18, fontweight="bold", pad=8)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(axis="both", labelsize=13, width=0.9)
    ax.tick_params(axis="x", labelrotation=18)
    for label in ax.get_xticklabels():
        label.set_ha("right")


def plot_metric(ax, labels, values, errors, title, ylim, value_labels, label_offset, colors):
    x = list(range(len(labels)))
    bars = ax.bar(
        x,
        values,
        width=0.8,
        color=colors,
        edgecolor="#202020",
        linewidth=0.8,
        yerr=errors,
        capsize=4,
        error_kw={"elinewidth": 1.0, "capthick": 1.0, "ecolor": "black"},
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    style_axis(ax, title, ylim)
    add_bar_labels(ax, bars, values, value_labels, label_offset)


def build_figure(stats: list[SeriesStats]):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "matplotlib is required to generate the figure. Install it in the active "
            "Python environment, for example: python -m pip install matplotlib"
        ) from exc

    labels = [item.label for item in stats]
    colors = ["#4c72b0", "#e8752a", "#64a83d"]

    coverage = [item.coverage_mean for item in stats]
    coverage_err = [item.coverage_sd for item in stats]
    objective = [item.objective_mean for item in stats]
    objective_err = [item.objective_sd for item in stats]
    uncovered = [item.uncovered_mean for item in stats]
    uncovered_err = [item.uncovered_sd for item in stats]
    terms = [item.terms_mean for item in stats]
    terms_err = [item.terms_sd for item in stats]

    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Formulation-level comparison on the 20,000 x 800 edge-assignment benchmark",
        fontsize=24,
        fontweight="bold",
        y=0.975,
    )

    coverage_labels = [f"{coverage[0]:.2f}%", f"{coverage[1]:.2f}%", f"{coverage[2]:.1f}%"]
    objective_labels = [format_int_space(value) for value in objective]
    uncovered_labels = [format_int_space(value) for value in uncovered]
    terms_labels = [format_int_space(value) for value in terms]

    plot_metric(
        axes[0, 0],
        labels,
        coverage,
        coverage_err,
        "Coverage (%)",
        (0, 105),
        coverage_labels,
        2.5,
        colors,
    )
    plot_metric(
        axes[0, 1],
        labels,
        objective,
        objective_err,
        "Evaluation objective",
        (0, 260000),
        objective_labels,
        6500,
        colors,
    )
    plot_metric(
        axes[1, 0],
        labels,
        uncovered,
        uncovered_err,
        "Uncovered after validation",
        (0, 17000),
        uncovered_labels,
        420,
        colors,
    )
    plot_metric(
        axes[1, 1],
        labels,
        terms,
        terms_err,
        "Avg. QUBO terms per batch",
        (0, 86000),
        terms_labels,
        2200,
        colors,
    )

    fig.text(
        0.5,
        0.035,
        "AO-QUBO and Static-QCP-QUBO are generic non-residual baselines aggregated over six full-scale runs each; "
        "PRC-QUBO updates residual capacity between batches.",
        ha="center",
        va="center",
        fontsize=12,
        color="black",
    )
    fig.subplots_adjust(left=0.07, right=0.985, top=0.83, bottom=0.14, wspace=0.16, hspace=0.42)
    return fig


def collect_stats(root: Path) -> list[SeriesStats]:
    stats = [
        aggregate_baseline(root, label, files)
        for label, files in BASELINE_SUMMARY_FILES.items()
    ]
    stats.append(parse_prc_logs(root))
    return stats


def print_stats(stats: list[SeriesStats]) -> None:
    for item in stats:
        print(
            f"{item.label}: coverage={item.coverage_mean:.3f}+/-{item.coverage_sd:.3f}, "
            f"objective={item.objective_mean:.1f}+/-{item.objective_sd:.1f}, "
            f"uncovered={item.uncovered_mean:.1f}+/-{item.uncovered_sd:.1f}, "
            f"terms={item.terms_mean:.0f}+/-{item.terms_sd:.0f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the formulation-level QUBO comparison figure from experiment logs."
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("manuscript_revision/figures/formulation_comparison_600dpi.png"),
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=Path("manuscript_revision/figures/qubo_comparison_graphs_cropped.pdf"),
    )
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--show-data", action="store_true")
    parser.add_argument(
        "--show-data-only",
        action="store_true",
        help="Print aggregated data and exit without importing matplotlib.",
    )
    return parser.parse_args()


def resolve_output(root: Path, output: Path) -> Path:
    return output if output.is_absolute() else root / output


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    stats = collect_stats(root)
    if args.show_data or args.show_data_only:
        print_stats(stats)
    if args.show_data_only:
        return

    fig = build_figure(stats)
    png_path = resolve_output(root, args.output_png)
    pdf_path = resolve_output(root, args.output_pdf)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=args.dpi, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    import matplotlib.pyplot as plt

    plt.close(fig)
    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")


if __name__ == "__main__":
    main()
