# import sys
# import re
# import numpy as np
# import pandas as pd
# from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
#                              QPushButton, QLineEdit, QLabel, QTextEdit, QProgressBar, QTableWidget,
#                              QTableWidgetItem)
# from PyQt5.QtCore import Qt, QThread, pyqtSignal
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.figure import Figure
# import logging
# from main import OptimizedQuantumScheduler
#
# # Logging setup
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('quantum_optimized.log', encoding='utf-8'),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)
#
#
# class QTextEditLogger(logging.Handler):
#     def __init__(self, text_edit, progress_signal):
#         super().__init__()
#         self.text_edit = text_edit
#         self.progress_signal = progress_signal
#         self.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
#
#     def emit(self, record):
#         msg = self.format(record)
#         self.text_edit.append(msg)
#         # Match log format: "Batch X: Y cameras, coverage: Z%, QUBO Success: W%"
#         match = re.match(r'.*Batch (\d+): Assigned (\d+) cameras, Coverage: ([\d.]+)%, QUBO Success: ([\d.]+)%', msg)
#         if match:
#             batch_idx = int(match.group(1))
#             batch_assigned = int(match.group(2))
#             coverage = float(match.group(3))
#             success_qubo = float(match.group(4))
#             self.progress_signal.emit({
#                 "batch_idx": batch_idx,
#                 "batch_assigned": batch_assigned,
#                 "coverage": coverage,
#                 "success_qubo": success_qubo
#             })
#
#
# class WorkerThread(QThread):
#     progress = pyqtSignal(dict)
#     result = pyqtSignal(dict)
#     log = pyqtSignal(str)
#
#     def __init__(self, scheduler, method):
#         super().__init__()
#         self.scheduler = scheduler
#         self.method = method
#         self.is_running = True
#
#     def run(self):
#         try:
#             if self.method == "greedy":
#                 assignment, objective, time = self.scheduler.solve_with_greedy_optimized_with_timeout()
#                 covered = np.sum(np.any(assignment, axis=1))
#                 self.result.emit({
#                     "method": "Greedy",
#                     "assignment": assignment,
#                     "objective": objective,
#                     "time": time,
#                     "covered": covered
#                 })
#                 self.progress.emit({"coverage": covered / self.scheduler.n_cameras * 100, "batch_idx": 0})
#             elif self.method == "quantum":
#                 assignment, objective, time, qubo_time, quantum_time = self.scheduler.solve_with_quantum_optimized()
#                 covered = np.sum(np.any(assignment, axis=1))
#                 self.result.emit({
#                     "method": "Quantum",
#                     "assignment": assignment,
#                     "objective": objective,
#                     "time": time,
#                     "qubo_time": qubo_time,
#                     "quantum_time": quantum_time,
#                     "covered": covered
#                 })
#             self.progress.emit({"coverage": 100, "batch_idx": 0, "completed": True})
#         except Exception as e:
#             self.log.emit(f"Error: {str(e)}")
#
#
# class MatplotlibCanvas(FigureCanvas):
#     def __init__(self, parent=None, plot_type="progress"):
#         fig = Figure()
#         self.axes = fig.add_subplot(111)
#         super().__init__(fig)
#         self.setParent(parent)
#         self.plot_type = plot_type
#         self.coverages = []
#         self.success_qubo = []
#         self.batch_indices = []
#
#     def plot_assignments(self, cameras_df, servers_df, assignment):
#         self.axes.clear()
#         max_lines = 1000
#         if cameras_df is not None and servers_df is not None:
#             self.axes.scatter(cameras_df['x'], cameras_df['y'], c='blue', label='Cameras', s=10, alpha=0.5)
#             self.axes.scatter(servers_df['x'], servers_df['y'], c='red', marker='s', label='Servers', s=50)
#             for i in range(min(len(cameras_df), max_lines)):
#                 j = np.argmax(assignment[i]) if np.any(assignment[i]) else -1
#                 if j != -1:
#                     cam_x, cam_y = cameras_df.iloc[i][['x', 'y']]
#                     srv_x, srv_y = servers_df.iloc[j][['x', 'y']]
#                     self.axes.plot([cam_x, srv_x], [cam_y, srv_y], 'k-', alpha=0.1)
#             self.axes.legend()
#         self.axes.set_title("Camera-Server Assignments")
#         self.axes.set_xlabel("X Coordinate")
#         self.axes.set_ylabel("Y Coordinate")
#         self.draw()
#
#     def plot_quantum_progress(self, progress_data):
#         if "batch_idx" in progress_data:
#             self.coverages.append(progress_data["coverage"])
#             self.success_qubo.append(progress_data["success_qubo"])
#             self.batch_indices.append(progress_data["batch_idx"])
#
#         self.axes.clear()
#         if self.plot_type == "progress":
#             self.axes.plot(self.batch_indices, self.coverages, label="Coverage (%)", color="#1f77b4")
#             self.axes.set_xlabel("Batch Number")
#             self.axes.set_ylabel("Coverage (%)", color="#1f77b4")
#             self.axes.tick_params(axis='y', labelcolor="#1f77b4")
#             ax2 = self.axes.twinx()
#             ax2.plot(self.batch_indices, self.success_qubo, label="QUBO Success (%)", color="#ff7f0e")
#             ax2.set_ylabel("QUBO Success (%)", color="#ff7f0e")
#             ax2.tick_params(axis='y', labelcolor="#ff7f0e")
#             ax2.legend(loc="upper right")
#             self.axes.legend(loc="upper left")
#             self.axes.set_title("Coverage and QUBO Success")
#         self.draw()
#
#
# class MainWindow(QMainWindow):
#     progress_signal = pyqtSignal(dict)  # Define as class attribute
#
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Camera-Server Optimization")
#         self.setGeometry(100, 100, 1400, 800)
#
#         self.central_widget = QWidget()
#         self.setCentralWidget(self.central_widget)
#         self.layout = QHBoxLayout(self.central_widget)
#
#         self.control_panel = QWidget()
#         self.control_layout = QVBoxLayout(self.control_panel)
#         self.control_layout.setAlignment(Qt.AlignTop)
#
#         self.params = {
#             "n_cameras": QLineEdit("20000"),
#             "n_servers": QLineEdit("800"),
#             "batch_size": QLineEdit("80"),
#             "max_servers_per_batch": QLineEdit("20")
#         }
#         for param, edit in self.params.items():
#             self.control_layout.addWidget(QLabel(param.replace('_', ' ').title()))
#             edit.setFixedWidth(100)
#             self.control_layout.addWidget(edit)
#
#         self.generate_btn = QPushButton("Generate Data")
#         self.greedy_btn = QPushButton("Run Greedy")
#         self.quantum_btn = QPushButton("Run Quantum")
#         self.stop_btn = QPushButton("Stop")
#         self.save_btn = QPushButton("Save Results")
#         self.control_layout.addWidget(self.generate_btn)
#         self.control_layout.addWidget(self.greedy_btn)
#         self.control_layout.addWidget(self.quantum_btn)
#         self.control_layout.addWidget(self.stop_btn)
#         self.control_layout.addWidget(self.save_btn)
#
#         self.progress_bar = QProgressBar()
#         self.progress_bar.setMaximum(100)
#         self.control_layout.addWidget(QLabel("Progress"))
#         self.control_layout.addWidget(self.progress_bar)
#
#         self.log_text = QTextEdit()
#         self.log_text.setReadOnly(True)
#         self.control_layout.addWidget(QLabel("Logs"))
#         self.control_layout.addWidget(self.log_text)
#
#         self.log_handler = QTextEditLogger(self.log_text, self.progress_signal)
#         logging.getLogger().addHandler(self.log_handler)
#
#         self.visual_panel = QWidget()
#         self.visual_layout = QVBoxLayout(self.visual_panel)
#
#         self.assignment_canvas = MatplotlibCanvas(self, plot_type="assignments")
#         self.visual_layout.addWidget(QLabel("Assignments"))
#         self.visual_layout.addWidget(self.assignment_canvas)
#
#         self.progress_canvas = MatplotlibCanvas(self, plot_type="progress")
#         self.visual_layout.addWidget(QLabel("Coverage and QUBO Success"))
#         self.visual_layout.addWidget(self.progress_canvas)
#
#         self.result_table = QTableWidget()
#         self.result_table.setColumnCount(4)
#         self.result_table.setHorizontalHeaderLabels(["Method", "Time (s)", "Objective", "Covered"])
#         self.result_table.setMinimumHeight(100)
#         self.visual_layout.addWidget(self.result_table)
#
#         self.layout.addWidget(self.control_panel, 1)
#         self.layout.addWidget(self.visual_panel, 2)
#
#         self.scheduler = None
#         self.thread = None
#
#         self.generate_btn.clicked.connect(self.generate_data)
#         self.greedy_btn.clicked.connect(self.run_greedy)
#         self.quantum_btn.clicked.connect(self.run_quantum)
#         self.stop_btn.clicked.connect(self.stop)
#         self.save_btn.clicked.connect(self.save_results)
#         self.progress_signal.connect(self.update_progress)
#
#     def generate_data(self):
#         try:
#             params = {k: int(v.text()) for k, v in self.params.items()}
#             self.scheduler = OptimizedQuantumScheduler(**params)
#             utilization = self.scheduler.generate_realistic_data()
#             self.log_text.append(f"Data generated: {utilization:.1f}% capacity usage")
#             # Initialize assignment_matrix to prevent NoneType error
#             self.scheduler.assignment_matrix = np.zeros((params["n_cameras"], params["n_servers"]), dtype=int)
#             self.assignment_canvas.plot_assignments(
#                 self.scheduler.cameras_df, self.scheduler.servers_df,
#                 self.scheduler.assignment_matrix
#             )
#             self.progress_canvas.coverages = []
#             self.progress_canvas.success_qubo = []
#             self.progress_canvas.batch_indices = []
#             self.progress_canvas.plot_quantum_progress({"batch_idx": 0, "coverage": 0, "success_qubo": 0})
#         except Exception as e:
#             self.log_text.append(f"Error generating data: {str(e)}")
#
#     def run_greedy(self):
#         if self.scheduler is None:
#             self.log_text.append("Generate data first!")
#             return
#         self.progress_bar.setValue(0)
#         self.progress_canvas.coverages = []
#         self.progress_canvas.success_qubo = []
#         self.progress_canvas.batch_indices = []
#         self.thread = WorkerThread(self.scheduler, "greedy")
#         self.thread.progress.connect(self.update_progress)
#         self.thread.result.connect(self.display_results)
#         self.thread.log.connect(self.log_text.append)
#         self.thread.start()
#
#     def run_quantum(self):
#         if self.scheduler is None:
#             self.log_text.append("Generate data first!")
#             return
#         self.progress_bar.setValue(0)
#         self.progress_canvas.coverages = []
#         self.progress_canvas.success_qubo = []
#         self.progress_canvas.batch_indices = []
#         self.thread = WorkerThread(self.scheduler, "quantum")
#         self.thread.progress.connect(self.update_progress)
#         self.thread.result.connect(self.display_results)
#         self.thread.log.connect(self.log_text.append)
#         self.thread.start()
#
#     def update_progress(self, progress_data):
#         if "batch_idx" in progress_data:
#             value = int(progress_data["batch_idx"] / (self.scheduler.n_cameras / self.scheduler.batch_size) * 100)
#             self.progress_bar.setValue(value)
#             self.log_text.append(
#                 f"Batch {progress_data['batch_idx']}: {progress_data['batch_assigned']} cameras, "
#                 f"coverage: {progress_data['coverage']:.1f}%, QUBO success: {progress_data['success_qubo']:.1f}%"
#             )
#             self.progress_canvas.plot_quantum_progress(progress_data)
#         elif "completed" in progress_data:
#             self.progress_bar.setValue(100)
#             self.log_text.append("Completed")
#         else:
#             self.progress_bar.setValue(int(progress_data["coverage"]))
#             self.log_text.append(f"Coverage: {progress_data['coverage']:.1f}%")
#
#     def display_results(self, result):
#         self.scheduler.assignment_matrix = result["assignment"]
#         self.assignment_canvas.plot_assignments(
#             self.scheduler.cameras_df, self.scheduler.servers_df, result["assignment"]
#         )
#         row = self.result_table.rowCount()
#         self.result_table.insertRow(row)
#         self.result_table.setItem(row, 0, QTableWidgetItem(result["method"]))
#         self.result_table.setItem(row, 1, QTableWidgetItem(f"{result['time']:.2f}"))
#         self.result_table.setItem(row, 2, QTableWidgetItem(f"{result['objective']:.1f}"))
#         self.result_table.setItem(row, 3, QTableWidgetItem(f"{result['covered']}/{self.scheduler.n_cameras}"))
#         log_msg = f"{result['method']} completed: {result['time']:.2f}s, objective: {result['objective']:.1f}, "
#         log_msg += f"covered: {result['covered']}/{self.scheduler.n_cameras}"
#         if "qubo_time" in result:
#             log_msg += f", QUBO: {result['qubo_time']:.2f}s, Quantum: {result['quantum_time']:.2f}s"
#         self.log_text.append(log_msg)
#
#     def stop(self):
#         if self.thread:
#             self.thread.is_running = False
#             self.thread.terminate()
#             self.log_text.append("Computation stopped")
#
#     def save_results(self):
#         if self.scheduler and hasattr(self.scheduler, 'assignment_matrix'):
#             pd.DataFrame(self.scheduler.assignment_matrix).to_csv("assignments.csv")
#             self.log_text.append("Results saved to assignments.csv")
#         else:
#             self.log_text.append("No results to save!")
#
#
# def main_gui():
#     app = QApplication(sys.argv)
#     window = MainWindow()
#     window.show()
#     sys.exit(app.exec_())
#
#
# if __name__ == "__main__":
#     main_gui()





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
