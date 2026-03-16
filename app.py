#
# from dash import Dash, dcc, html, Input, Output, callback
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import pandas as pd
# import json
# import glob
# import numpy as np
# from datetime import datetime
#
# app = Dash(__name__)
#
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# app = Dash(__name__, external_stylesheets=external_stylesheets)
#
#
# def load_all_runs():
#     files = sorted(glob.glob("logs/progress_*.jsonl"))
#     runs = {}
#     for file in files:
#         run_id = file.split("_")[-1].replace(".jsonl", "")
#         with open(file, "r", encoding="utf-8") as f:
#             data = [json.loads(line) for line in f]
#         df = pd.DataFrame(data)
#         df['batch_idx'] = df['batch_idx'].astype(int)
#         runs[run_id] = df
#     return runs
#
#
# runs_data = load_all_runs()
#
# app.layout = html.Div([
#     html.H1("Classical Annealing Scheduler (SA)",
#             style={'textAlign': 'center', 'color': '#1f77b4', 'fontsize': 28}),
#
#     dcc.Tabs(id="tabs", value='tab-coverage', children=[
#         dcc.Tab(label='Coverage & Comparison', value='tab-coverage'),
#         dcc.Tab(label='Annealing Success Rate', value='tab-success'),
#         dcc.Tab(label='Energy Evolution', value='tab-energy'),
#         dcc.Tab(label='3D Energy Landscape', value='tab-3d'),
#     ]),
#     html.Div(id='tabs-content')
# ])
#
#
# @callback(Output('tabs-content', 'children'), Input('tabs', 'value'))
# def render_content(tab):
#     if not runs_data:
#         return html.Div("No logs found in 'logs/' folder.",
#                         style={'color': 'red', 'textAlign': 'center', 'padding': '20px'})
#
#     if tab == 'tab-coverage':
#         fig = go.Figure()
#         for run_id, df in runs_data.items():
#             fig.add_trace(go.Scatter(
#                 x=df['batch_idx'], y=df['coverage_percent'],
#                 mode='lines+markers',
#                 name=f'Classical Annealing {run_id}',
#                 line=dict(width=3, color='#1f77b4')
#             ))
#
#             try:
#                 with open("classical_optimization.log", "r", encoding="utf-8") as f:
#                     log_text = f.read()
#                 import re
#                 match = re.search(rf"Greedy algorithm:.*?(\d+) cameras", log_text)
#                 if match:
#                     covered = int(match.group(1))
#                     fig.add_trace(go.Scatter(
#                         x=[0, max(df['batch_idx'])],
#                         y=[covered / 20000 * 100] * 2,
#                         mode='lines',
#                         name=f'Greedy Baseline',
#                         line=dict(dash='dash', color='red', width=2)
#                     ))
#             except:
#                 pass
#
#         fig.update_layout(
#             title="Camera Coverage: Classical Annealing vs Greedy Baseline",
#             xaxis_title="Batch Index",
#             yaxis_title="Coverage %",
#             hovermode='x unified',
#             template='plotly_white',
#             height=500
#         )
#         return dcc.Graph(figure=fig)
#
#     elif tab == 'tab-success':
#         fig = go.Figure()
#         colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
#         for idx, (run_id, df) in enumerate(runs_data.items()):
#             fig.add_trace(go.Scatter(
#                 x=df['batch_idx'],
#                 y=df['qubo_success_rate'],
#                 mode='lines',
#                 name=f'Run {run_id}',
#                 line=dict(width=3, color=colors[idx % len(colors)])
#             ))
#
#         fig.update_layout(
#             title="Classical Annealing Batch Success Rate",
#             xaxis_title="Batch Index",
#             yaxis_title="Success Rate %",
#             yaxis=dict(range=[0, 105]),
#             template='plotly_white',
#             height=500
#         )
#         return dcc.Graph(figure=fig)
#
#     elif tab == 'tab-energy':
#         fig = go.Figure()
#         colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
#
#         for idx, (run_id, df) in enumerate(runs_data.items()):
#             energy = df['energy'].dropna()
#             if not energy.empty:
#                 fig.add_trace(go.Scatter(
#                     x=df.loc[energy.index, 'batch_idx'],
#                     y=energy,
#                     mode='lines+markers',
#                     name=f'Run {run_id}',
#                     line=dict(width=2, color=colors[idx % len(colors)]),
#                     marker=dict(size=6)
#                 ))
#
#         fig.update_layout(
#             title="Classical Annealing Energy Evolution (Lower is Better)",
#             xaxis_title="Batch Index",
#             yaxis_title="Energy",
#             template='plotly_white',
#             height=500
#         )
#         return dcc.Graph(figure=fig)
#
#     elif tab == 'tab-3d':
#         fig = go.Figure()
#         colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
#         all_x, all_y, all_z = [], [], []
#
#         for idx, (run_id, df) in enumerate(runs_data.items()):
#             energy = df['energy'].dropna()
#             if energy.empty:
#                 continue
#             try:
#                 x = pd.to_numeric(df.loc[energy.index, 'batch_idx'], errors='coerce').dropna().values
#                 y = pd.to_numeric(df.loc[energy.index, 'coverage_percent'], errors='coerce').dropna().values
#                 z = pd.to_numeric(energy, errors='coerce').dropna().values
#
#                 if len(x) == 0:
#                     continue
#
#                 fig.add_trace(go.Scatter3d(
#                     x=x, y=y, z=z,
#                     mode='lines+markers',
#                     name=f'Classical Annealing {run_id}',
#                     line=dict(color=colors[idx % len(colors)], width=5),
#                     marker=dict(size=4, opacity=0.9),
#                     hovertemplate="<b>Batch:</b> %{x}<br><b>Coverage:</b> %{y:.1f}%<br><b>Energy:</b> %{z:.1f}"
#                 ))
#
#                 all_x.extend(x)
#                 all_y.extend(y)
#                 all_z.extend(z)
#             except:
#                 continue
#
#         if len(all_x) < 5:
#             fig.add_annotation(
#                 text="Not enough data for 3D visualization",
#                 xref="paper", yref="paper", x=0.5, y=0.5,
#                 showarrow=False, font=dict(size=16, color="gray")
#             )
#             fig.update_layout(
#                 height=700,
#                 scene=dict(
#                     xaxis_title="Batch Index",
#                     yaxis_title="Coverage %",
#                     zaxis_title="Energy"
#                 ),
#                 title="Classical Annealing 3D Energy Landscape"
#             )
#             return dcc.Graph(figure=fig)
#
#         all_x = np.array(all_x, dtype=float)
#         all_y = np.array(all_y, dtype=float)
#         all_z = np.array(all_z, dtype=float)
#
#         try:
#             df_points = pd.DataFrame({'x': all_x, 'y': all_y, 'z': all_z})
#             df_points = df_points.drop_duplicates(subset=['x', 'y']).round(6)
#
#             if len(df_points) < 3:
#                 raise ValueError("Too few points")
#
#             x_vals = df_points['x'].values
#             y_vals = df_points['y'].values
#             z_vals = df_points['z'].values
#
#             x_min, x_max = x_vals.min(), x_vals.max()
#             y_min, y_max = y_vals.min(), y_vals.max()
#
#             if x_max == x_min or y_max == y_min:
#                 raise ValueError("Zero range")
#
#             nx, ny = 50, 50
#             x_grid = np.linspace(x_min, x_max, nx)
#             y_grid = np.linspace(y_min, y_max, ny)
#             X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
#
#             Z_grid = np.full((ny, nx), np.nan)
#
#             for i in range(ny):
#                 for j in range(nx):
#                     px, py = X_grid[i, j], Y_grid[i, j]
#                     dists = np.hypot(x_vals - px, y_vals - py)
#                     if len(dists) > 0:
#                         idx = np.argmin(dists)
#                         Z_grid[i, j] = z_vals[idx]
#
#             mean_z = np.nanmean(Z_grid)
#             Z_grid = np.where(np.isnan(Z_grid), mean_z, Z_grid)
#
#             from scipy.ndimage import gaussian_filter
#             Z_grid = gaussian_filter(Z_grid, sigma=1.5)
#
#             fig.add_trace(go.Surface(
#                 x=X_grid, y=Y_grid, z=Z_grid,
#                 colorscale='Plasma',
#                 opacity=0.7,
#                 name='Energy Surface',
#                 showscale=False,
#                 lighting=dict(ambient=0.6, diffuse=0.9, specular=0.3),
#                 contours=dict(z=dict(show=True, color="white", width=1))
#             ))
#
#         except Exception as e:
#             print(f"[3D] Surface generation failed: {e}")
#
#         try:
#             min_idx = np.argmin(all_z)
#             fig.add_trace(go.Scatter3d(
#                 x=[all_x[min_idx]], y=[all_y[min_idx]], z=[all_z[min_idx]],
#                 mode='markers',
#                 marker=dict(size=14, color='gold', symbol='star',
#                             line=dict(width=3, color='yellow')),
#                 name='Optimal Solution'
#             ))
#         except:
#             pass
#
#         fig.update_layout(
#             title="Classical Annealing: 3D Energy Landscape",
#             scene=dict(
#                 xaxis_title="Batch Index",
#                 yaxis_title="Coverage %",
#                 zaxis_title="Energy",
#                 camera=dict(eye=dict(x=1.7, y=1.7, z=1.3)),
#                 aspectmode='manual',
#                 aspectratio=dict(x=1, y=1, z=0.7)
#             ),
#             height=800,
#             legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
#             margin=dict(l=0, r=0, t=60, b=0),
#             template='plotly_white'
#         )
#
#         return dcc.Graph(figure=fig, config={'scrollZoom': True, 'displayModeBar': True})
#
#
# if __name__ == '__main__':
#     print("Starting Classical Annealing Dashboard...")
#     print("Open: http://127.0.0.1:8050")
#     print("Make sure 'logs/progress_*.jsonl' files exist")
#     app.run(debug=False, port=8050)
#
#
#
#
#
#
#

















from dash import Dash, dcc, html, Input, Output, callback
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import json
import glob
import numpy as np
from datetime import datetime

app = Dash(__name__)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__, external_stylesheets=external_stylesheets)


def load_all_runs():
    files = sorted(glob.glob("logs/progress_*.jsonl"))
    runs = {}
    for file in files:
        run_id = file.split("_")[-1].replace(".jsonl", "")
        with open(file, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
        df = pd.DataFrame(data)
        df['batch_idx'] = df['batch_idx'].astype(int)
        runs[run_id] = df
    return runs


runs_data = load_all_runs()

app.layout = html.Div([
    html.H1("Classical Annealing Scheduler (SA)",
            style={'textAlign': 'center', 'color': '#1f77b4', 'fontsize': 28}),

    dcc.Tabs(id="tabs", value='tab-coverage', children=[
        dcc.Tab(label='Coverage & Comparison', value='tab-coverage'),
        dcc.Tab(label='Annealing Success Rate', value='tab-success'),
        dcc.Tab(label='Energy Evolution', value='tab-energy'),
        dcc.Tab(label='3D Energy Landscape', value='tab-3d'),
    ]),
    html.Div(id='tabs-content')
])


@callback(Output('tabs-content', 'children'), Input('tabs', 'value'))
def render_content(tab):
    if not runs_data:
        return html.Div("No logs found in 'logs/' folder.",
                        style={'color': 'red', 'textAlign': 'center', 'padding': '20px'})

    if tab == 'tab-coverage':
        fig = go.Figure()
        for run_id, df in runs_data.items():
            fig.add_trace(go.Scatter(
                x=df['batch_idx'], y=df['coverage_percent'],
                mode='lines+markers',
                name=f'Classical Annealing {run_id}',
                line=dict(width=3, color='#1f77b4')
            ))

            try:
                with open("classical_optimization.log", "r", encoding="utf-8") as f:
                    log_text = f.read()
                import re
                match = re.search(rf"Greedy algorithm:.*?(\d+) cameras", log_text)
                if match:
                    covered = int(match.group(1))
                    fig.add_trace(go.Scatter(
                        x=[0, max(df['batch_idx'])],
                        y=[covered / 20000 * 100] * 2,
                        mode='lines',
                        name=f'Greedy Baseline',
                        line=dict(dash='dash', color='red', width=2)
                    ))
            except:
                pass

        fig.update_layout(
            title="Camera Coverage: Classical Annealing vs Greedy Baseline",
            xaxis_title="Batch Index",
            yaxis_title="Coverage %",
            hovermode='x unified',
            template='plotly_white',
            height=500,
            xaxis=dict(tickfont=dict(size=16)),
            yaxis=dict(tickfont=dict(size=16))
        )
        return dcc.Graph(figure=fig)

    elif tab == 'tab-success':
        fig = go.Figure()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for idx, (run_id, df) in enumerate(runs_data.items()):
            fig.add_trace(go.Scatter(
                x=df['batch_idx'],
                y=df['qubo_success_rate'],
                mode='lines',
                name=f'Run {run_id}',
                line=dict(width=3, color=colors[idx % len(colors)])
            ))

        fig.update_layout(
            title="Classical Annealing Batch Success Rate",
            xaxis_title="Batch Index",
            yaxis_title="Success Rate %",
            yaxis=dict(range=[0, 105], tickfont=dict(size=16)),
            xaxis=dict(tickfont=dict(size=16)),
            template='plotly_white',
            height=500
        )
        return dcc.Graph(figure=fig)

    elif tab == 'tab-energy':
        fig = go.Figure()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        for idx, (run_id, df) in enumerate(runs_data.items()):
            energy = df['energy'].dropna()
            if not energy.empty:
                fig.add_trace(go.Scatter(
                    x=df.loc[energy.index, 'batch_idx'],
                    y=energy,
                    mode='lines+markers',
                    name=f'Run {run_id}',
                    line=dict(width=2, color=colors[idx % len(colors)]),
                    marker=dict(size=6)
                ))

        fig.update_layout(
            title="Classical Annealing Energy Evolution (Lower is Better)",
            xaxis_title="Batch Index",
            yaxis_title="Energy",
            template='plotly_white',
            height=500,
            xaxis=dict(tickfont=dict(size=16)),
            yaxis=dict(tickfont=dict(size=16))
        )
        return dcc.Graph(figure=fig)

    elif tab == 'tab-3d':
        fig = go.Figure()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        all_x, all_y, all_z = [], [], []

        for idx, (run_id, df) in enumerate(runs_data.items()):
            energy = df['energy'].dropna()
            if energy.empty:
                continue
            try:
                x = pd.to_numeric(df.loc[energy.index, 'batch_idx'], errors='coerce').dropna().values
                y = pd.to_numeric(df.loc[energy.index, 'coverage_percent'], errors='coerce').dropna().values
                z = pd.to_numeric(energy, errors='coerce').dropna().values

                if len(x) == 0:
                    continue

                fig.add_trace(go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='lines+markers',
                    name=f'Classical Annealing {run_id}',
                    line=dict(color=colors[idx % len(colors)], width=5),
                    marker=dict(size=4, opacity=0.9),
                    hovertemplate="<b>Batch:</b> %{x}<br><b>Coverage:</b> %{y:.1f}%<br><b>Energy:</b> %{z:.1f}"
                ))

                all_x.extend(x)
                all_y.extend(y)
                all_z.extend(z)
            except:
                continue

        if len(all_x) < 5:
            fig.add_annotation(
                text="Not enough data for 3D visualization",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=16, color="gray")
            )
            fig.update_layout(
                height=700,
                scene=dict(
                    xaxis_title="Batch Index",
                    yaxis_title="Coverage %",
                    zaxis_title="Energy"
                ),
                title="Classical Annealing 3D Energy Landscape"
            )
            return dcc.Graph(figure=fig)

        all_x = np.array(all_x, dtype=float)
        all_y = np.array(all_y, dtype=float)
        all_z = np.array(all_z, dtype=float)

        try:
            df_points = pd.DataFrame({'x': all_x, 'y': all_y, 'z': all_z})
            df_points = df_points.drop_duplicates(subset=['x', 'y']).round(6)

            if len(df_points) < 3:
                raise ValueError("Too few points")

            x_vals = df_points['x'].values
            y_vals = df_points['y'].values
            z_vals = df_points['z'].values

            x_min, x_max = x_vals.min(), x_vals.max()
            y_min, y_max = y_vals.min(), y_vals.max()

            if x_max == x_min or y_max == y_min:
                raise ValueError("Zero range")

            nx, ny = 50, 50
            x_grid = np.linspace(x_min, x_max, nx)
            y_grid = np.linspace(y_min, y_max, ny)
            X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

            Z_grid = np.full((ny, nx), np.nan)

            for i in range(ny):
                for j in range(nx):
                    px, py = X_grid[i, j], Y_grid[i, j]
                    dists = np.hypot(x_vals - px, y_vals - py)
                    if len(dists) > 0:
                        idx = np.argmin(dists)
                        Z_grid[i, j] = z_vals[idx]

            mean_z = np.nanmean(Z_grid)
            Z_grid = np.where(np.isnan(Z_grid), mean_z, Z_grid)

            from scipy.ndimage import gaussian_filter
            Z_grid = gaussian_filter(Z_grid, sigma=1.5)

            fig.add_trace(go.Surface(
                x=X_grid, y=Y_grid, z=Z_grid,
                colorscale='Plasma',
                opacity=0.7,
                name='Energy Surface',
                showscale=False,
                lighting=dict(ambient=0.6, diffuse=0.9, specular=0.3),
                contours=dict(z=dict(show=True, color="white", width=1))
            ))

        except Exception as e:
            print(f"[3D] Surface generation failed: {e}")

        try:
            min_idx = np.argmin(all_z)
            fig.add_trace(go.Scatter3d(
                x=[all_x[min_idx]], y=[all_y[min_idx]], z=[all_z[min_idx]],
                mode='markers',
                marker=dict(size=14, color='gold', symbol='star',
                            line=dict(width=3, color='yellow')),
                name='Optimal Solution'
            ))
        except:
            pass

        fig.update_layout(
            title="Classical Annealing: 3D Energy Landscape",
            scene=dict(
                xaxis_title="Batch Index",
                yaxis_title="Coverage %",
                zaxis_title="Energy",
                xaxis=dict(tickfont=dict(size=16)),
                yaxis=dict(tickfont=dict(size=16)),
                zaxis=dict(tickfont=dict(size=16)),
                camera=dict(eye=dict(x=1.7, y=1.7, z=1.3)),
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=0.7)
            ),
            height=800,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            margin=dict(l=0, r=0, t=60, b=0),
            template='plotly_white'
        )

        return dcc.Graph(figure=fig, config={'scrollZoom': True, 'displayModeBar': True})


if __name__ == '__main__':
    print("Starting Classical Annealing Dashboard...")
    print("Open: http://127.0.0.1:8050")
    print("Make sure 'logs/progress_*.jsonl' files exist")
    app.run(debug=False, port=8050)