import argparse
import csv
import json
from pathlib import Path


DEFAULT_ROOTS = [
    "logs_capacity_stress",
    "logs_prc_qubo_conflict",
    "logs_quality_stress_strong_baselines",
    "logs_article_revision",
]

QUBO_FORMULATIONS = {
    "AO-QUBO",
    "Static-QCP-QUBO",
    "PRC-QUBO",
    "PRC-QUBO-C",
    "PRC-QUBO-D",
    "PRC-QUBO-C-no-decoder",
}

CLASSICAL_FORMULATIONS = {
    "RC-Greedy-20",
    "Regret-Best-Fit-20",
    "Capacity-Priced-Greedy-20",
    "Lagrangian-Price-20",
    "GRASP-20",
    "BRKGA-20",
    "Tabu-20",
}


def load_json(path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def as_float(value, default=0.0):
    if value is None:
        return default
    return float(value)


def as_int(value, default=0):
    if value is None:
        return default
    return int(value)


def collect_rows(roots, include_diagnostic=False):
    seen = set()
    rows = []
    for root_name in roots:
        root = Path(root_name)
        if not root.exists():
            continue
        for path in root.rglob("summary_*.json"):
            if path in seen:
                continue
            seen.add(path)
            try:
                data = load_json(path)
            except (OSError, json.JSONDecodeError):
                continue

            if as_int(data.get("n_cameras")) != 20000:
                continue
            solver = str(data.get("solver", ""))
            if solver == "DIAGNOSTIC" and not include_diagnostic:
                continue

            formulation = str(data.get("formulation", ""))
            if formulation not in QUBO_FORMULATIONS and formulation not in CLASSICAL_FORMULATIONS:
                continue

            budget = as_int(data.get("max_servers_per_batch"))
            if budget and budget != 20:
                continue

            row = {
                "formulation": formulation,
                "solver": solver,
                "capacity_scale": as_float(data.get("capacity_scale")),
                "utilization_percent": as_float(data.get("utilization_percent")),
                "coverage_percent": as_float(data.get("coverage_percent")),
                "covered_cameras": as_int(data.get("covered_cameras")),
                "uncovered_cameras": as_int(data.get("uncovered_cameras")),
                "objective_value": as_float(data.get("objective_value")),
                "assignment_cost": as_float(data.get("assignment_cost")),
                "uncovered_penalty": as_float(data.get("uncovered_penalty")),
                "overload_penalty": as_float(data.get("overload_penalty")),
                "capacity_rejected_raw_assignments": as_int(data.get("capacity_rejected_raw_assignments")),
                "failed_batches": as_int(data.get("failed_batches")),
                "total_time_sec": as_float(data.get("total_time_sec")),
                "throughput_cam_per_sec": as_float(data.get("throughput_cam_per_sec")),
                "avg_qubo_variables": as_float(data.get("avg_qubo_variables")),
                "avg_qubo_coefficient_count": as_float(data.get("avg_qubo_coefficient_count")),
                "avg_capacity_conflict_quadratic_count": as_float(
                    data.get("avg_capacity_conflict_quadratic_count")
                ),
                "capacity_conflict_enabled": bool(data.get("capacity_conflict_enabled", False)),
                "capacity_aware_decoder": bool(data.get("capacity_aware_decoder", False)),
                "summary_path": str(path),
            }
            rows.append(row)
    rows.sort(key=lambda r: (r["capacity_scale"], r["formulation"], r["solver"], r["summary_path"]))
    return rows


def best_by_key(rows, predicate, key):
    grouped = {}
    for row in rows:
        if not predicate(row):
            continue
        grouped.setdefault(row["utilization_percent"], []).append(row)
    best = []
    for util, util_rows in grouped.items():
        util_rows.sort(key=key)
        best.append(util_rows[0])
    best.sort(key=lambda r: r["utilization_percent"])
    return best


def write_csv(path, rows):
    fieldnames = [
        "formulation",
        "solver",
        "capacity_scale",
        "utilization_percent",
        "coverage_percent",
        "covered_cameras",
        "uncovered_cameras",
        "objective_value",
        "assignment_cost",
        "capacity_rejected_raw_assignments",
        "failed_batches",
        "total_time_sec",
        "throughput_cam_per_sec",
        "avg_qubo_variables",
        "avg_qubo_coefficient_count",
        "avg_capacity_conflict_quadratic_count",
        "capacity_conflict_enabled",
        "capacity_aware_decoder",
        "summary_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def fmt(value, digits=2):
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    return str(value)


def markdown_table(rows, columns):
    lines = []
    lines.append("| " + " | ".join(label for label, _key in columns) + " |")
    lines.append("| " + " | ".join("---" for _label, _key in columns) + " |")
    for row in rows:
        values = []
        for _label, key in columns:
            value = key(row) if callable(key) else row.get(key, "")
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_markdown(path, rows):
    def prefer_article_revision(rows_to_filter):
        selected = {}
        for row in rows_to_filter:
            key = (row["formulation"], row["solver"], row["capacity_scale"])
            normalized_path = row["summary_path"].replace("\\", "/")
            score = (
                1 if normalized_path.startswith("logs_article_revision/") else 0,
                normalized_path,
            )
            if key not in selected or score > selected[key][0]:
                selected[key] = (score, row)
        return [item[1] for item in selected.values()]

    def latest_per_group(rows_to_filter, key_fn):
        selected = {}
        for row in rows_to_filter:
            key = key_fn(row)
            normalized_path = row["summary_path"].replace("\\", "/")
            score = (
                1 if normalized_path.startswith("logs_article_revision/") else 0,
                normalized_path,
            )
            if key not in selected or score > selected[key][0]:
                selected[key] = (score, row)
        return [item[1] for item in selected.values()]

    qubo_sqa = [
        r
        for r in rows
        if r["formulation"] in QUBO_FORMULATIONS
        and r["solver"] == "SQA"
        and not (r["formulation"] == "PRC-QUBO-D")
    ]
    qubo_sqa = latest_per_group(
        qubo_sqa,
        lambda r: (r["formulation"], r["solver"], r["capacity_scale"]),
    )
    qubo_sqa.sort(key=lambda r: (r["utilization_percent"], r["formulation"], r["solver"]))
    classical_best = best_by_key(
        rows,
        lambda r: r["formulation"] in CLASSICAL_FORMULATIONS,
        key=lambda r: (r["objective_value"], r["total_time_sec"]),
    )
    ablation = [
        r
        for r in rows
        if r["formulation"] in {"PRC-QUBO", "PRC-QUBO-D", "PRC-QUBO-C-no-decoder", "PRC-QUBO-C"}
        and r["solver"] == "SQA"
        and abs(r["capacity_scale"] - 0.25) < 1e-9
    ]
    ablation = latest_per_group(
        ablation,
        lambda r: (r["formulation"], r["solver"], r["capacity_scale"]),
    )
    ablation.sort(key=lambda r: (r["formulation"], r["objective_value"]))
    diagnostic_ablation = [
        r
        for r in rows
        if r["formulation"] in {"PRC-QUBO-D", "PRC-QUBO-C-no-decoder", "PRC-QUBO-C"}
        and r["solver"] == "DIAGNOSTIC"
        and abs(r["capacity_scale"] - 0.25) < 1e-9
    ]
    diagnostic_ablation = prefer_article_revision(diagnostic_ablation)
    diagnostic_ablation.sort(key=lambda r: (r["formulation"], r["objective_value"]))

    common_cols = [
        ("Util.", lambda r: fmt(r["utilization_percent"], 2) + "%"),
        ("Method", lambda r: f"{r['formulation']} + {r['solver']}"),
        ("Coverage", lambda r: fmt(r["coverage_percent"], 2) + "%"),
        ("Objective", lambda r: fmt(r["objective_value"], 2)),
        ("Assign. cost", lambda r: fmt(r["assignment_cost"], 2)),
        ("Rejected", lambda r: fmt(r["capacity_rejected_raw_assignments"], 0)),
        ("Time", lambda r: fmt(r["total_time_sec"], 2) + "s"),
    ]

    density_cols = common_cols + [
        ("QUBO terms", lambda r: fmt(r["avg_qubo_coefficient_count"], 0)),
        ("Conflict terms", lambda r: fmt(r["avg_capacity_conflict_quadratic_count"], 0)),
    ]

    content = [
        "# Article Revision Metric Summary",
        "",
        "Generated from selected `summary_*.json` files. Raw QUBO energies are intentionally not compared across formulations.",
        "",
        "## QUBO SQA Capacity-Stress Rows",
        "",
        markdown_table(qubo_sqa, density_cols),
        "",
        "## Best Classical Baseline Per Utilization",
        "",
        markdown_table(classical_best, common_cols),
        "",
        "## PRC-QUBO-C Ablation At 95.07% Utilization",
        "",
        markdown_table(ablation, density_cols),
        "",
        "## Diagnostic Ablation At 95.07% Utilization",
        "",
        "These rows are implementation diagnostics and must not be presented as OpenJij SQA results.",
        "",
        markdown_table(diagnostic_ablation, density_cols),
        "",
        "## Notes",
        "",
        "- `PRC-QUBO-C` is a stress-aware extension, not a silent replacement for `PRC-QUBO`.",
        "- `PRC-QUBO-D` denotes decoder-only ablation: PRC linear model plus capacity-aware decoder, without conflict couplings.",
        "- `PRC-QUBO-C-no-decoder` denotes conflict-coupled QUBO decoded with the original PRC decoder.",
        "- Markdown tables select the latest preferred summary per method/utilization. The CSV file retains every raw row for auditability.",
        "- The strongest currently logged classical baseline is usually `Regret-Best-Fit-20`; it is fast and feasible, so the paper should emphasize quality/runtime trade-off.",
        "",
    ]
    path.write_text("\n".join(content), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Build article-revision metric tables from summary JSON logs.")
    parser.add_argument("--output-dir", default="logs_article_revision")
    parser.add_argument("--include-diagnostic", action="store_true")
    parser.add_argument("--roots", nargs="*", default=DEFAULT_ROOTS)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(args.roots, include_diagnostic=args.include_diagnostic)
    csv_path = output_dir / "article_revision_summary.csv"
    md_path = output_dir / "article_revision_summary.md"
    write_csv(csv_path, rows)
    write_markdown(md_path, rows)

    manifest = {
        "row_count": len(rows),
        "include_diagnostic": bool(args.include_diagnostic),
        "roots": args.roots,
        "csv": str(csv_path),
        "markdown": str(md_path),
    }
    (output_dir / "article_revision_summary_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
