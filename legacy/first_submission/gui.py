import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import json
import glob
import numpy as np

def load_progress_logs(log_dir="logs"):
    files = sorted(glob.glob(f"{log_dir}/progress_*.jsonl"))
    if not files:
        print("No progress logs found!")
        return pd.DataFrame()

    data = []
    for file in files:
        run_id = file.split("_")[-1].replace(".jsonl", "")
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                entry["run_id"] = run_id
                data.append(entry)
    return pd.DataFrame(data)

def plot_dashboard():
    df = load_progress_logs()
    if df.empty:
        return

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Coverage Over Time",
            "QUBO Success Rate",
            "Energy Evolution",
            "Assignments per Batch"
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )

    runs = df['run_id'].unique()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, run in enumerate(runs):
        d = df[df['run_id'] == run]
        color = colors[i % len(colors)]

        fig.add_trace(go.Scatter(
            x=d['batch_idx'], y=d['coverage_percent'],
            mode='lines+markers', name=f"Coverage ({run})", line=dict(color=color)
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=d['batch_idx'], y=d['qubo_success_rate'],
            mode='lines', name=f"QUBO Success ({run})", line=dict(color=color, dash='dot')
        ), row=1, col=2)

        if 'energy' in d.columns and d['energy'].notna().any():
            fig.add_trace(go.Scatter(
                x=d['batch_idx'], y=d['energy'],
                mode='lines', name=f"Energy ({run})", line=dict(color=color)
            ), row=2, col=1)
        fig.add_trace(go.Bar(
            x=d['batch_idx'], y=d['batch_assigned'],
            name=f"Assigned/Batch ({run})", marker_color=color, opacity=0.6
        ), row=2, col=2)

    fig.update_layout(height=800, title_text="Quantum Scheduler Dashboard", showlegend=True)
    fig.update_xaxes(title_text="Batch Index")
    fig.update_yaxes(title_text="Coverage %", row=1, col=1)
    fig.update_yaxes(title_text="Success %", row=1, col=2)
    fig.update_yaxes(title_text="Energy", row=2, col=1)
    fig.update_yaxes(title_text="Cameras Assigned", row=2, col=2)

    fig.show()

if __name__ == "__main__":
    plot_dashboard()
