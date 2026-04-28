[//]: # (# Quantum-Inspired Optimization for Sensor-to-Server Assignment in Smart-City Surveillance Systems)

[//]: # ()
[//]: # ([![Python 3.8+]&#40;https://img.shields.io/badge/python-3.8+-blue.svg&#41;]&#40;https://www.python.org/downloads/&#41;)

[//]: # ([![OpenJij]&#40;https://img.shields.io/badge/OpenJij-0.11.6-purple.svg&#41;]&#40;https://openjij.github.io/&#41;)

[//]: # ([![D-Wave Neal]&#40;https://img.shields.io/badge/D--Wave-Neal%20Sampler-orange.svg&#41;]&#40;https://dwave-neal-docs.readthedocs.io/en/latest/reference/sampler.html&#41;)

[//]: # (---)

[//]: # ()
[//]: # (## 📋 Table of Contents)

[//]: # (- [1. Introduction]&#40;#1-introduction&#41;)

[//]: # (- [2. Problem Formulation]&#40;#2-problem-formulation-and-input-data&#41;)

[//]: # (- [3. QUBO Formulation]&#40;#3-qubo-formulation&#41;)

[//]: # (- [4. Optimization Approaches]&#40;#4-optimization-approaches&#41;)

[//]: # (- [5. Results and Visualization]&#40;#5-results-and-visualization&#41;)

[//]: # (- [6. Getting Started]&#40;#6-getting-started&#41;)

[//]: # (- [7. Repository Structure]&#40;#7-repository-structure&#41;)

[//]: # (- [8. Citation]&#40;#8-citation&#41;)

[//]: # (- [9. License and Acknowledgments]&#40;#9-license-and-acknowledgments&#41;)

[//]: # ()
[//]: # (---)

[//]: # ()
[//]: # (## 1. Introduction)

[//]: # ()
[//]: # ()
[//]: # (**The main question of this work: Can simulated quantum annealing &#40;SQA&#41; with batch decomposition solve real-life camera-to-edge-server assignment problems with 20,000 cameras and 800 servers more effectively than classical optimization approaches, while maintaining practical runtime and full reproducibility using open-source tools?**)

[//]: # ()
[//]: # ()
[//]: # (The given work is completely experimentally reproducible*. Unlike proprietary quantum hardware solutions that may have variable access or performance characteristics, our framework leverages:)

[//]: # ()
[//]: # (- **OpenJij** — An open-source framework for the Ising model and QUBO problems, developed and maintained by Jij Inc.)

[//]: # (- **Fixed random seeds** — All experiments use NumPy seed 42 for deterministic data generation)

[//]: # (- **Structured JSONL logging** — Every batch's metrics are recorded for post-hoc analysis)

[//]: # (- **Version-controlled dependencies** — Exact library versions specified for environment recreation)

[//]: # ()
[//]: # (Thus, anyone can replicate our experiments by:)

[//]: # (1. Installing OpenJij 0.11.6 &#40;as used in this study&#41;)

[//]: # (2. Running the provided Python scripts with identical parameters)

[//]: # (3. Comparing results against our logs )

[//]: # ()
[//]: # (The OpenJij framework implements Simulated Quantum Annealing &#40;SQA&#41; through path-integral Monte Carlo with Trotter decomposition, making quantum-inspired optimization accessible on standard computing hardware without requiring access to physical quantum processors.)

[//]: # ()
[//]: # (### Key Topics)

[//]: # ()
[//]: # (| Topic | Description |)

[//]: # (|-------|-------------|)

[//]: # (| **Simulated Quantum Annealing &#40;SQA&#41;** | Quantum-inspired optimization using OpenJij's path-integral Monte Carlo with Trotter decomposition |)

[//]: # (| **Simulated Annealing &#40;SA&#41;** | Classical probabilistic metaheuristic implemented via D-Wave's Neal library |)

[//]: # (| **QUBO** | Quadratic Unconstrained Binary Optimization — NP-hard problem formulation |)

[//]: # (| **Smart City Sensors** | 20,000 cameras with heterogeneous priorities and computational loads |)

[//]: # (| **Distributed Data Processing** | Edge computing architecture with 800 servers of varying capacities |)

[//]: # ()
[//]: # ()
[//]: # (## 2. Problem Formulation and Input Data)

[//]: # ()
[//]: # (Let $C = \{1, 2, \dots, 20000\}$ denote the set of surveillance cameras and $S = \{1, 2, \dots, 800\}$ the set of edge servers.)

[//]: # ()
[//]: # (**Camera Parameters:**)

[//]: # (- $p_i \in \{1,2,3\}$ — priority level of camera $i$ &#40;3 = highest, 2 = middle, 1 = lowest&#41;)

[//]: # (- $w_i = 4 - p_i \in \{1,2,3\}$ — priority weight &#40;higher value → higher reward&#41;)

[//]: # (- $l_i > 0$ — computational demand of camera $i$ &#40;in GFLOPs&#41;)

[//]: # ()
[//]: # (**Server Parameters:**)

[//]: # (- $K_j > 0$ — total processing capacity of server $j$ &#40;in GFLOPs&#41;)

[//]: # ()
[//]: # (**Decision Variable:**)

[//]: # ($$)

[//]: # (x_{ij} = \begin{cases} )

[//]: # (1, & \text{if camera } i \text{ is assigned to server } j \\)

[//]: # (0, & \text{otherwise})

[//]: # (\end{cases})

[//]: # ($$)

[//]: # ()
[//]: # (**Constraints:**)

[//]: # ()
[//]: # (1. **Unique assignment constraint:** Each camera must be assigned to exactly one server)

[//]: # (   $$)

[//]: # (   \sum_{j=1}^{M} x_{ij} = 1, \quad \forall i = 1,\ldots,N)

[//]: # (   $$)

[//]: # ()
[//]: # (2. **Server capacity constraint:** The total computational load on each server must not exceed its processing capacity)

[//]: # (   $$)

[//]: # (   \sum_{i=1}^{N} l_i x_{ij} \leq K_j, \quad \forall j = 1,\ldots,M)

[//]: # (   $$)

[//]: # ()
[//]: # ()
[//]: # (The normalized assignment cost $c_{ij} \in [0,1]$ represents the cost of connecting camera $i$ to server $j$, incorporating four weighted factors:)

[//]: # ()
[//]: # ($$)

[//]: # (c_{ij} = 0.40 \cdot \frac{d_{ij}}{d_{\text{max}}} + 0.35 \cdot \frac{l_i}{l_{\text{max}}} + 0.20 \cdot \frac{3-p_i}{2} + 0.05 \cdot \frac{K_{\text{max}}/K_j}{&#40;K_{\text{max}}/K_{\text{min}}&#41;})

[//]: # ($$)

[//]: # ()
[//]: # (where:)

[//]: # (- $d_{ij}$ — Euclidean distance between camera $i$ and server $j$)

[//]: # (- $d_{\text{max}}$ — maximum distance across all camera-server pairs)

[//]: # (- $l_{\text{max}}$ — maximum computational load across all cameras)

[//]: # (- $K_{\text{max}}, K_{\text{min}}$ — maximum and minimum server capacities)

[//]: # ()
[//]: # (**Weight distribution:**)

[//]: # (- **40% Network latency** — prioritizes physical proximity to minimize transmission delays)

[//]: # (- **35% Computational load** — balances processing demands across servers)

[//]: # (- **20% Priority weighting** — ensures high-priority cameras receive better service)

[//]: # (- **5% Capacity utilization** — lightly favors underutilized servers for load balancing)

[//]: # ()
[//]: # (The resulting cost matrix is min-max normalized to $[0,1]$ to ensure numerical stability during the annealing process.)

[//]: # ()
[//]: # (### Data Generation)

[//]: # (To ensure a realistic evaluation of the scheduling algorithms, we generate synthetic data mimicking a large-scale video surveillance system with 20,000 cameras and 800 servers. The data generation process is identical for both Simulated Annealing &#40;SA&#41; and Simulated Quantum Annealing &#40;SQA&#41; methods, using **fixed random seed 42** for perfect reproducibility across all experiments.)

[//]: # ()
[//]: # (| Parameter | Distribution | Details |)

[//]: # (|-----------|--------------|---------|)

[//]: # (| **Camera Priorities** | $p_i = 3$ &#40;15%&#41;, $p_i = 2$ &#40;25%&#41;, $p_i = 1$ &#40;60%&#41; | High: pedestrian crossings<br>Medium: sidewalks<br>Low: roadways |)

[//]: # (| **Camera Load &#40;GFLOPs&#41;** | High: $l_i \sim \mathcal{U}[8,15]$<br>Medium: $l_i \sim \mathcal{U}[4,8]$<br>Low: $l_i \sim \mathcal{U}[1,3]$ | Computational demand for AI-based video analytics |)

[//]: # (| **Server Capacities &#40;GFLOPs&#41;** | High: $K_j \sim \mathcal{U}[800,1000]$ &#40;10%&#41;<br>Medium: $K_j \sim \mathcal{U}[400,800]$ &#40;30%&#41;<br>Standard: $K_j \sim \mathcal{U}[200,400]$ &#40;60%&#41; | Heterogeneous edge server infrastructure |)

[//]: # (| **Geographic Distribution** | $&#40;x,y&#41; \sim \mathcal{U}[0,1000]^2$ | Uniform random placement in $1000 \times 1000$ grid |)

[//]: # ()
[//]: # (**System-wide metrics:**)

[//]: # (- Total computational load: $\displaystyle \sum_{i=1}^{N} l_i \approx 88,453$ GFLOPs)

[//]: # (- Total server capacity: $\displaystyle \sum_{j=1}^{M} K_j \approx 372,166$ GFLOPs)

[//]: # (- System utilization: $\approx 23.8\%$)

[//]: # ()
[//]: # ()
[//]: # (![Simulated Surveillance Environment o]&#40;imgs_readme/map.jpg&#41;)

[//]: # ()
[//]: # (*The figure illustrates 3 edge servers with capacities 800–1000 GFLOPs &#40;high&#41;, 400–800 GFLOPs &#40;medium&#41;, and 200–400 GFLOPs &#40;standard&#41;, with 15 cameras distributed by priority: 15% high &#40;red: pedestrian crossings&#41;, 25% medium &#40;yellow: sidewalks&#41;, and 60% low &#40;blue: roadways&#41;. Computational loads &#40;$l_i$ in GFLOPs&#41; are annotated near cameras, following the uniform distributions described above.*)

[//]: # ()
[//]: # (---)

[//]: # ()
[//]: # (## 3. QUBO Formulation)

[//]: # ()
[//]: # (### From ILP to QUBO)

[//]: # ()
[//]: # (The constrained Integer Linear Programming &#40;ILP&#41; problem is transformed into an Quadratic Unconstrained Binary Optimization &#40;QUBO&#41; problem using a penalty-reward approach:)

[//]: # ()
[//]: # ($$)

[//]: # (H_{\text{QUBO}}&#40;\mathbf{x}&#41; = \underbrace{\sum_{i=1}^{N}\sum_{j=1}^{M} c_{ij} x_{ij}}_{\text{Primary cost}} + \lambda_1 \underbrace{\sum_{i=1}^{N} \left&#40;\sum_{j=1}^{M} x_{ij} - 1\right&#41;^2}_{\text{Assignment penalty}} + \lambda_2 \underbrace{\sum_{j=1}^{M} \max\left&#40;0, \sum_{i=1}^{N} l_i x_{ij} - K_j\right&#41;^2}_{\text{Capacity penalty}})

[//]: # ($$)

[//]: # ()
[//]: # (where $\lambda_1, \lambda_2 > 0$ are penalty coefficients chosen sufficiently large to enforce constraint satisfaction.)

[//]: # ()
[//]: # (### Batch-Level Hamiltonian)

[//]: # ()
[//]: # (Due to the huge size of the full problem &#40;16 million binary variables&#41;, we decompose it into tractable subproblems using an iterative batch strategy. For each batch of $n$ cameras $B$ and $m$ candidate servers $S$, the QUBO Hamiltonian is constructed as:)

[//]: # ()
[//]: # ($$)

[//]: # (H&#40;\mathbf{x}&#41; = \sum_{i \in B} \sum_{j \in S} Q_{ij}^{&#40;1&#41;} x_{ij} + \lambda \sum_{i \in B} \sum_{\substack{j,k \in S \\ j < k}} x_{ij} x_{ik})

[//]: # ($$)

[//]: # ()
[//]: # (where $\mathbf{x} = \{x_{ij} \in \{0,1\}\}$ represents binary assignment variables for the current batch.)

[//]: # ()
[//]: # (### Coefficient Definition)

[//]: # ()
[//]: # (The linear coefficients $Q_{ij}^{&#40;1&#41;}$ encode feasibility and objective preferences:)

[//]: # ()
[//]: # ($$)

[//]: # (Q_{ij}^{&#40;1&#41;} = \begin{cases} )

[//]: # (-\alpha \cdot w_i \cdot &#40;1 - c_{ij}&#41;, & \text{if } l_i \leq R_j \\)

[//]: # (+\beta, & \text{otherwise})

[//]: # (\end{cases})

[//]: # ($$)

[//]: # ()
[//]: # (where:)

[//]: # (- $w_i = 4 - p_i \in \{1,2,3\}$ — priority weight)

[//]: # (- $c_{ij} \in [0,1]$ — normalized connection cost)

[//]: # (- $l_i > 0$ — camera computational load)

[//]: # (- $R_j > 0$ — remaining server capacity at batch time)

[//]: # (- $\alpha, \beta, \lambda$ — experimentally tuned constants)

[//]: # ()
[//]: # (### QUBO Parameters)

[//]: # ()
[//]: # (| Parameter | Value | Description |)

[//]: # (|-----------|-------|-------------|)

[//]: # (| $\alpha$ | 25 | Reward coefficient for feasible assignments |)

[//]: # (| $\beta$ | 100 | Penalty for capacity violations |)

[//]: # (| $\lambda$ | 15 | Penalty for multiple assignments &#40;one-hot encoding&#41; |)

[//]: # ()
[//]: # (### Encoding Three Key Aspects)

[//]: # ()
[//]: # (This formulation encodes:)

[//]: # ()
[//]: # (1. **Primary objective** — via reward term $-\alpha w_i&#40;1-c_{ij}&#41;$ for feasible assignments:)

[//]: # (   - Higher priority cameras &#40;$w_i$ larger&#41; receive stronger negative coefficients)

[//]: # (   - Lower connection costs &#40;$c_{ij}$ smaller&#41; produce larger rewards)

[//]: # (   - The negative sign makes better assignments lower-energy states)

[//]: # ()
[//]: # (2. **Capacity constraints** — through large penalty $\beta$ for infeasible pairs where $l_i > R_j$:)

[//]: # (   - Any assignment violating server capacity gets a high positive contribution to energy)

[//]: # (   - The optimizer avoids these configurations during annealing)

[//]: # ()
[//]: # (3. **One-hot assignment** — via quadratic term $\lambda x_{ij}x_{ik}$:)

[//]: # (   - For a given camera $i$, assigning it to two different servers $j$ and $k$ creates a positive energy contribution)

[//]: # (   - This enforces $\sum_j x_{ij} \leq 1$ without hard constraints)

[//]: # (   - The coefficient $\lambda$ is tuned to balance against the objective terms)

[//]: # ()
[//]: # ()
[//]: # ()
[//]: # (## 4. Optimization Approaches)

[//]: # ()
[//]: # (### Architectural Overview)

[//]: # ()
[//]: # ()
[//]: # (![System Architecture]&#40;imgs_readme/architecture.jpg&#41;)

[//]: # ()
[//]: # ()
[//]: # (The architecture features **fully shared preprocessing stages** &#40;Data Initialization, Cost Matrix Construction, and QUBO Batch Decomposition&#41; before branching into the three solving approaches:)

[//]: # ()
[//]: # (1. **Simulated Quantum Annealing &#40;SQA&#41;** — OpenJij SQASampler)

[//]: # (2. **Classical Simulated Annealing &#40;SA&#41;** — D-Wave Neal)

[//]: # (3. **Greedy Baseline** — Deterministic heuristic)

[//]: # ()
[//]: # (### Batch Decomposition Strategy)

[//]: # ()
[//]: # (Due to the problem scale &#40;20,000 cameras × 800 servers&#41;, we implement an iterative batch strategy:)

[//]: # ()
[//]: # ($$)

[//]: # (\text{Total variables} = N \times M = 20,000 \times 800 = 16,000,000)

[//]: # ($$)

[//]: # ()
[//]: # (The batch decomposition reduces each subproblem to:)

[//]: # ()
[//]: # ($$)

[//]: # (\text{Batch variables} = B \times M = 80 \times 20 = 1,600)

[//]: # ($$)

[//]: # ()
[//]: # (**Algorithm:**)

[//]: # (1. **Step 1:** Initialize remaining server capacities $R_j = K_j$)

[//]: # (2. **Step 2:** Sort cameras by priority score = $p_i \times l_i$ &#40;higher first&#41;)

[//]: # (3. **Step 3:** While unassigned cameras remain:)

[//]: # (    * Select next batch of $B = 80$ highest-priority unassigned cameras)

[//]: # (    * Select $M = 20$ candidate servers with largest remaining capacity $R_j$)

[//]: # (    * Construct QUBO subproblem with current remaining capacities $R_j$)

[//]: # (    * Solve QUBO using SQA, SA, or Greedy)

[//]: # (    * Update assignments and reduce server capacities)

[//]: # (4. **Step 4:** Apply local optimization to improve assignments across batch boundaries)

[//]: # ()
[//]: # ()
[//]: # ()
[//]: # (### Simulated Quantum Annealing &#40;SQA&#41; with OpenJij)

[//]: # ()
[//]: # (**Library:** OpenJij 0.11.6 &#40;[official documentation]&#40;https://jij-inc.github.io/OpenJij/&#41;&#41;  )

[//]: # (**Sampler:** `SQASampler` &#40;Simulated Quantum Annealing&#41;  )

[//]: # (**Core Mechanism:** Path-Integral Monte Carlo with Trotter decomposition)

[//]: # ()
[//]: # (SQA transforms the $N$-variable QUBO into an extended $&#40;N \times P&#41;$-variable classical Ising model via the Suzuki-Trotter decomposition:)

[//]: # ()
[//]: # ($$)

[//]: # (H_{\text{SQA}} = -\frac{1}{P}\sum_{k=1}^{P} H_{\text{QUBO}}^{&#40;k&#41;} - J_{\perp} \sum_{k=1}^{P} \sum_{i=1}^{N} \sigma_{i,k} \sigma_{i,k+1})

[//]: # ($$)

[//]: # ()
[//]: # (where:)

[//]: # (- $P = 8$ — Trotter slices &#40;replicas&#41;)

[//]: # (- $H_{\text{QUBO}}^{&#40;k&#41;}$ — the $k$-th replica of the QUBO Hamiltonian)

[//]: # (- $\sigma_{i,k} \in \{\pm 1\}$ — classical Ising spins)

[//]: # (- $J_{\perp}$ — inter-replica coupling strength simulating quantum tunneling)

[//]: # ()
[//]: # (**Key Parameters:**)

[//]: # (- Trotter number &#40;$P$&#41;: 8 — balances simulation fidelity and computational cost)

[//]: # (- Monte Carlo sweeps: 1000 — annealing duration)

[//]: # (- Transverse field schedule: geometric decrease from $\Gamma_0$ to near-zero)

[//]: # ()
[//]: # (**OpenJij Implementation:**)

[//]: # (```python)

[//]: # (import openjij as oj)

[//]: # ()
[//]: # (sampler = oj.SQASampler&#40;)

[//]: # (    trotter=8,           # Number of replicas &#40;P&#41;)

[//]: # (    num_sweeps=1000,     # Monte Carlo steps)

[//]: # (    num_reads=10         # Parallel runs per batch)

[//]: # (&#41;)

[//]: # ()
[//]: # (response = sampler.sample_qubo&#40;Q&#41;  # Q is the QUBO matrix)

[//]: # (```)

[//]: # ()
[//]: # (### Classical Simulated Annealing &#40;SA&#41;)

[//]: # ()
[//]: # (* **Library:** D-Wave Neal &#40;`SimulatedAnnealingSampler`&#41;)

[//]: # (* **Core Mechanism:** Metropolis-Hastings with thermal fluctuations)

[//]: # ()
[//]: # (The algorithm interprets the QUBO objective as an energy function to be minimized:)

[//]: # ()
[//]: # ($$H&#40;\sigma&#41; = \sum_{i<j} J_{ij}\sigma_i\sigma_j + \sum_{i} h_i\sigma_i$$)

[//]: # ()
[//]: # (where binary variables $x_i \in \{0,1\}$ correspond to spins via $\sigma_i = 2x_i - 1$.)

[//]: # ()
[//]: # (**Dynamics:** The Metropolis-Hastings acceptance rule governs state transitions:)

[//]: # (* If $\Delta E < 0$: always accept &#40;downhill move&#41;.)

[//]: # (* Else: accept with probability $\exp&#40;-\Delta E / T&#41;$.)

[//]: # ()
[//]: # (**Key Parameters:**)

[//]: # (* **Number of reads:** 150 — independent annealing runs.)

[//]: # (* **Cooling schedule:** geometric with $\alpha = 0.995$.)

[//]: # (* **Initial temperature:** sufficiently high for broad exploration.)

[//]: # ()
[//]: # (---)

[//]: # ()
[//]: # (### Greedy Baseline)

[//]: # ()
[//]: # (* **Mechanism:** Deterministic assignment using priority-capacity scoring.)

[//]: # ()
[//]: # (For each camera &#40;processed in priority order&#41;, select the server maximizing:)

[//]: # ()
[//]: # ($$\text{score}_{ij} = w_i \cdot &#40;1 - c_{ij}&#41; \cdot \frac{R_j}{K_j}$$)

[//]: # ()
[//]: # (**Where:**)

[//]: # (* $w_i = 4 - p_i$ — priority weight.)

[//]: # (* $c_{ij}$ — connection cost.)

[//]: # (* $R_j$ — remaining capacity.)

[//]: # (* $K_j$ — total capacity.)

[//]: # ()
[//]: # (**Timeout:** 180 seconds — ensures practical runtime for comparison.)

[//]: # ()
[//]: # ()
[//]: # (## 5. Results and Visualization)

[//]: # ()
[//]: # (### Final Performance Comparison)

[//]: # ()
[//]: # (| Method | Objective Value | Coverage &#40;%&#41; | Total Time &#40;s&#41; | Core Time &#40;s&#41; | Eff. &#40;%&#41; | Throughput &#40;cam/s&#41; |)

[//]: # (| :--- | :---: | :---: | :---: | :---: | :---: | :---: |)

[//]: # (| **Simulated Quantum Annealing &#40;SQA&#41;** | 7,108.1 | 99.6 | 1,162.92 | 1,058.81 | 91.1% | 17.13 |)

[//]: # (| **Classical Simulated Annealing &#40;SA&#41;** | 7,108.1 | 99.6 | 5,082.57 | 4,985.86 | 98.1% | 3.92 |)

[//]: # (| **Optimized Greedy Algorithm** | 18,584.1 | 95.7 | 19.27 | 1.29 | 6.7% | 1,037.88 |)

[//]: # ()
[//]: # ()
[//]: # (### Visualization Dashboards)

[//]: # ()
[//]: # (The project includes two interactive Dash applications for visualizing optimization progress:)

[//]: # ()
[//]: # (* **Classical Annealing Dashboard** &#40;`app.py` — port 8050&#41;)

[//]: # (    * **Features:**)

[//]: # (        * Coverage progression vs. Greedy baseline.)

[//]: # (        * Batch success rate tracking.)

[//]: # (        * Energy minimization dynamics.)

[//]: # (        * 3D energy landscape with global minimum.)

[//]: # (* **Quantum Annealing Dashboard** &#40;`app_Q.py` — port 8051&#41;)

[//]: # (    * **Features:**)

[//]: # (        * Purple-themed visualizations for SQA.)

[//]: # (        * Quantum tunneling effect visualization.)

[//]: # (        * Dark theme 3D landscapes.)

[//]: # (        * Real-time batch monitoring.)

[//]: # ()
[//]: # (---)

[//]: # ()
[//]: # (### Logging Framework)

[//]: # ()
[//]: # (Both implementations use a three-tier logging architecture to ensure **maximal** data integrity and traceability:)

[//]: # ()
[//]: # (| Level | Format | Purpose |)

[//]: # (| :--- | :--- | :--- |)

[//]: # (| **Console** | Text &#40;`logging` module&#41; | Real-time monitoring |)

[//]: # (| **Structured** | JSON Lines &#40;`.jsonl`&#41; | Batch-level metrics |)

[//]: # (| **Snapshots** | NumPy &#40;`.npz`&#41; | Checkpointing & recovery |)

[//]: # ()
[//]: # ()
[//]: # (### Log Entry Structure &#40;JSON&#41;)

[//]: # ()
[//]: # (Each batch processed by the system generates a structured entry in the `.jsonl` log file. This ensures **maximum** transparency for post-run analysis:)

[//]: # ()
[//]: # (```json)

[//]: # ({)

[//]: # (  "run_id": "20251213_235835",)

[//]: # (  "timestamp": "2025-12-13T23:58:36.123456",)

[//]: # (  "batch_idx": 42,)

[//]: # (  "batch_assigned": 80,)

[//]: # (  "coverage_percent": 33.6,)

[//]: # (  "qubo_success_rate": 100.0,)

[//]: # (  "qubo_time_sec": 0.15,)

[//]: # (  "annealing_time_sec": 18.5,)

[//]: # (  "energy": -1567.8,)

[//]: # (  "assignments": [...])

[//]: # (})

[//]: # (```)

[//]: # ()
[//]: # (## 6. Getting Started)

[//]: # ()
[//]: # (### Prerequisites)

[//]: # (* **Python 3.8** or higher.)

[//]: # (* **pip** package manager.)

[//]: # (* &#40;Optional&#41; **CMake ≥ 3.22** for building from source.)

[//]: # (* &#40;Optional&#41; **C++17 compatible compiler** for development.)

[//]: # ()
[//]: # (### Installing OpenJij)

[//]: # (This project utilizes **OpenJij 0.11.6** for Simulated Quantum Annealing &#40;SQA&#41;. You can find more detailed information about the framework in the [OpenJij Documentation]&#40;OpenJij/README_OpenJij.md&#41;.)

[//]: # ()
[//]: # (#### Option 1: Quick Install via pip &#40;Recommended&#41;)

[//]: # (```bash)

[//]: # (# Install binary distribution &#40;fastest&#41;)

[//]: # (pip install openjij==0.11.6)

[//]: # ()
[//]: # (# Verify installation)

[//]: # (python -c "import openjij; print&#40;f'OpenJij version: {openjij.__version__}'&#41;")

[//]: # (```)

[//]: # ()
[//]: # (## 7. Repository Structure)

[//]: # ()
[//]: # (## 7. Project Structure)

[//]: # ()
[//]: # (```text)

[//]: # (QAnnealing/)

[//]: # (├── imgs_readme/                # Images and badges used in documentation)

[//]: # (├── logs/                       # Execution logs and checkpoints)

[//]: # (│   ├── snapshots/              # Batch-level metrics &#40;*.jsonl&#41; & recovery checkpoints &#40;*.npz&#41;)

[//]: # (│   ├── logs_adaptive/          # Logs for adaptive parameter runs)

[//]: # (│   ├── logs_diagnostic/        # Diagnostic and error logs)

[//]: # (│   ├── logs_final/             # Final summary logs for completed runs)

[//]: # (│   └── logs_greedy_windows/    # Logs specific to the greedy baseline)

[//]: # (├── OpenJij/                    # Cloned OpenJij framework repository)

[//]: # (│   ├── benchmark/              # OpenJij performance benchmarks)

[//]: # (│   ├── openjij/                # Core Python and C++ source code)

[//]: # (│   ├── tests/                  # C++ and Python test suites)

[//]: # (│   ├── CMakeLists.txt          # Build configuration)

[//]: # (│   └── README_OpenJIJ.md       # OpenJij documentation)

[//]: # (├── templates/                  # HTML templates for dashboards)

[//]: # (│   └── index.html)

[//]: # (├── *.png                       # Generated visualizations )

[//]: # (├── cost_matrix.npy             # Precomputed connection costs )

[//]: # (├── cost_matrix.txt             # Precomputed connection costs )

[//]: # (├── requirements.txt            # Python environment dependencies)

[//]: # (│)

[//]: # (# --- Core Application Scripts & Dashboards ---)

[//]: # (├── main.py                     # Classical Simulated Annealing implementation)

[//]: # (├── main_Q.py                   # Simulated Quantum Annealing &#40;OpenJij&#41; implementation)

[//]: # (├── app.py                      # Dashboard for classical annealing &#40;port 8050&#41;)

[//]: # (├── app_Q.py                    # Dashboard for quantum annealing &#40;port 8051&#41; )

[//]: # (├── SA_Chronology.txt           # Historical SA run records)

[//]: # (└── SQA_Chronology.txt          # Historical SQA run records)

[//]: # ()
[//]: # (```)

[//]: # ()
[//]: # (## 8 Citation)

[//]: # ()
[//]: # (If you use this code or the results of our research in your work, please cite it as follows:)

[//]: # ()
[//]: # (**BibTeX:**)

[//]: # (```bibtex)

[//]: # (@article{mussabayev2026quantum,)

[//]: # (  title={Quantum-Inspired Optimization for Camera-to-Server Assignment in Smart City Surveillance Systems},)

[//]: # (  author={Mussabayev, Y. and Bykov, A.},)

[//]: # (  year={2026},)

[//]: # (  url={https://github.com/Yedman3585/Quantum-Sensor-Assignment})

[//]: # (})

[//]: # (```)

[//]: # ()
[//]: # (## 9 Contributors and Notes)

[//]: # ()
[//]: # (This project is part of a scientific research paper currently submitted to the **MDPI Sensors** journal.)

[//]: # ()
[//]: # (The given research is supervised and supported by **[Artem Bykov]&#40;https://github.com/username111213&#41;** — Associate Professor, PhD.)

[//]: # ()
[//]: # ()




# Quantum-Inspired Optimization for Sensor-to-Server Assignment in Smart-City Surveillance Systems

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenJij](https://img.shields.io/badge/OpenJij-0.11.6-purple.svg)](https://openjij.github.io/)
[![D-Wave Neal](https://img.shields.io/badge/D--Wave-Neal%20Sampler-orange.svg)](https://dwave-neal-docs.readthedocs.io/en/latest/reference/sampler.html)

---

## Table of Contents
- [1. Introduction](#1-introduction)
- [2. Problem Formulation and Input Data](#2-problem-formulation-and-input-data)
- [3. QUBO Formulation](#3-qubo-formulation)
- [4. Optimization Approaches](#4-optimization-approaches)
- [5. Results and Visualization](#5-results-and-visualization)
- [6. Getting Started](#6-getting-started)
- [7. Repository Structure](#7-repository-structure)
- [8. Citation](#8-citation)
- [9. License and Acknowledgments](#9-license-and-acknowledgments)

---

## 1. Introduction

**Main research question:** Can simulated quantum annealing (SQA) with batch decomposition solve real-life camera-to-edge-server assignment problems with 20,000 cameras and 800 servers more effectively than classical optimization approaches, while maintaining practical runtime and reproducibility using open-source tools?

This repository accompanies a reproducible benchmark for large-scale camera-to-edge-server assignment in smart-city surveillance systems. The framework relies entirely on open-source software and standard hardware:

- **OpenJij 0.11.6** for simulated quantum annealing (SQA)
- **D-Wave Neal** for classical simulated annealing (SA)
- **Fixed NumPy seed 42** for deterministic problem generation
- **Structured JSONL logging** for batch-level monitoring and post-hoc analysis
- **Version-controlled code and dependencies** for transparent experiment recreation

The repository supports reproducible **problem generation, execution protocol, logging, and post-hoc evaluation**. Because the annealing solvers themselves are stochastic and hardware-dependent, exact runtime and trajectory replication across different machines should not be expected.

OpenJij implements SQA through path-integral Monte Carlo with Suzuki–Trotter decomposition, making quantum-inspired optimization accessible on conventional computing hardware without requiring physical quantum processors.

### Key Topics

| Topic | Description |
|---|---|
| **Simulated Quantum Annealing (SQA)** | Quantum-inspired optimization using OpenJij's path-integral Monte Carlo with Trotter decomposition |
| **Simulated Annealing (SA)** | Classical probabilistic metaheuristic implemented via D-Wave's Neal library |
| **QUBO** | Quadratic Unconstrained Binary Optimization formulation with batch decomposition |
| **Smart City Sensors** | 20,000 cameras with heterogeneous priorities, loads, and bandwidth demands |
| **Distributed Data Processing** | Edge computing architecture with 800 heterogeneous servers |

---

## 2. Problem Formulation and Input Data

Let $C = \{1,2,\dots,20000\}$ denote the set of surveillance cameras and $S = \{1,2,\dots,800\}$ the set of edge servers.

### Camera Parameters
- $p_i \in \{1,2,3\}$ — priority level of camera $i$ ($3$ = highest, $1$ = lowest)
- $l_i > 0$ — computational demand of camera $i$ (GFLOPS)
- $b_i > 0$ — bandwidth requirement of camera $i$ (Mbps)

Operational priority is enforced jointly through three complementary mechanisms:
1. batch ordering by descending priority score $p_i l_i$
2. the priority-dependent term embedded in the connection cost $c_{ij}$
3. downstream decoding, post-processing, and greedy repair rules

### Server Parameters
- $K_j > 0$ — total processing capacity of server $j$ (GFLOPS)

### Decision Variable

$$
 x_{ij} =
 \begin{cases}
 1, & \text{if camera } i \text{ is assigned to server } j \\
 0, & \text{otherwise}
 \end{cases}
$$

### Hard Constraints

1. **Unique assignment**
$$
\sum_{j=1}^{|S|} x_{ij} = 1, \qquad \forall i \in C
$$

2. **Server capacity**
$$
\sum_{i=1}^{|C|} l_i x_{ij} \le K_j, \qquad \forall j \in S
$$

### Connection Cost Matrix

The normalized assignment cost $c_{ij} \in [0,1]$ combines four weighted factors:

$$
 c_{ij} = 0.40\frac{d_{ij}}{d_{\max}} + 0.35\frac{l_i}{l_{\max}} + 0.20\frac{3-p_i}{2} + 0.05\frac{K_{\max}/K_j}{K_{\max}/K_{\min}}
$$

where:
- $d_{ij}$ — Euclidean distance between camera $i$ and server $j$
- $d_{\max}$ — maximum distance across all camera–server pairs
- $l_{\max}$ — maximum computational load across all cameras
- $K_{\max}, K_{\min}$ — maximum and minimum server capacities

**Weight distribution**
- **40% network latency** — prioritizes physical proximity
- **35% computational load** — balances processing demand
- **20% inverted priority term** — lowers cost for higher-priority cameras
- **5% inverse capacity term** — lightly favors underutilized servers

The resulting cost matrix is min–max normalized to $[0,1]$ for numerical stability during annealing.

### Data Generation

The benchmark simulates a large-scale urban surveillance system with 20,000 cameras and 800 servers. Both SA and SQA use identical synthetic instances generated with the same fixed seed.

| Parameter | Distribution | Details |
|---|---|---|
| **Camera priorities** | $p_i=3$ (15%), $p_i=2$ (25%), $p_i=1$ (60%) | High: pedestrian crossings; Medium: sidewalks; Low: roadways |
| **Bandwidth $b_i$ (Mbps)** | High: $\mathcal{U}[4,7]$; Medium: $\mathcal{U}[2,4]$; Low: $\mathcal{U}[0.4,0.8]$ | Models video-transmission requirements |
| **Load $l_i$ (GFLOPS)** | High: $\mathcal{U}[8,15]$; Medium: $\mathcal{U}[4,8]$; Low: $\mathcal{U}[1,3]$ | AI inference demand for video analytics |
| **Server capacity $K_j$ (GFLOPS)** | High: $\mathcal{U}[800,1000]$ (10%); Medium: $\mathcal{U}[400,800]$ (30%); Standard: $\mathcal{U}[200,400]$ (60%) | Heterogeneous edge server fleet |
| **Geographic distribution** | $(x,y) \sim \mathcal{U}[0,1000]^2$ | Uniform random placement in a $1000 \times 1000$ grid |

**System-wide metrics**
- Total computational load: $\sum_i l_i \approx 88{,}453$ GFLOPS
- Total server capacity: $\sum_j K_j \approx 372{,}165.7$ GFLOPS
- System utilization: $\approx 23.8\%$

![Simulated surveillance environment](imgs_readme/map.jpg)

*Representative scheduler schematic with heterogeneous cameras, bandwidth demands, loads, and example server assignments.*

---

## 3. QUBO Formulation

### 3.1 Theoretical ILP-to-QUBO Mapping

The full constrained problem can be written as:

$$
H_{\text{QUBO}}(\mathbf{x}) = \sum_{i,j} c_{ij}x_{ij}
+ \lambda_1 \sum_i \left(\sum_j x_{ij} - 1\right)^2
+ \lambda_2 \sum_j \max\!\left(0,\sum_i l_i x_{ij} - K_j\right)^2
$$

where $\lambda_1, \lambda_2 > 0$ are sufficiently large penalty coefficients.

### 3.2 Released Surrogate Batch Hamiltonian

Because the full city-scale formulation would involve

$$
20{,}000 \times 800 = 16{,}000{,}000
$$

binary assignment variables before quadratic couplings are formed, the released benchmark operates on iterative **80 × 20** surrogate subproblems with residual-capacity updates.

For each batch $B$ of cameras and candidate-server set $S_B$, the batch Hamiltonian is:

$$
H(\mathbf{x}) = \sum_{i\in B} \sum_{j\in S_B} Q^{(1)}_{(i,j)} x_{ij}
+ \lambda \sum_{i\in B} \sum_{\substack{j<k \\ j,k\in S_B}} x_{ij}x_{ik}
$$

with linear coefficients

$$
Q^{(1)}_{(i,j)} =
\begin{cases}
-\alpha\, r_i (1-c_{ij}), & \text{if } l_i \le R_j \\
+\beta, & \text{otherwise}
\end{cases}
$$

where:
- $r_i = 4 - p_i$ — inverse-priority coefficient used in the released surrogate batch QUBO
- $R_j$ — remaining capacity of server $j$ at the current batch
- $\alpha = 25$, $\beta = 100$, $\lambda = 15$

### 3.3 Interpretation of the Surrogate Objective

This batch Hamiltonian encodes:

1. **Feasible-assignment reward** via $-\alpha r_i(1-c_{ij})$
2. **Capacity discouragement** via the large penalty $\beta$ for infeasible pairs
3. **One-hot relaxation** via the quadratic penalty term enforcing $\sum_j x_{ij} \le 1$

Important nuance: because $r_i = 4-p_i$, the released surrogate batch reward uses an **inverse-priority coefficient**. Operational priority is therefore not carried by this coefficient alone. It is enforced jointly through:
- batch ordering by $p_i l_i$
- the inverted-priority term in the cost matrix
- downstream decoding, post-processing, and greedy repair rules

Thus, the released benchmark uses a **surrogate batch-QUBO**, not the full original ILP objective directly.

### 3.4 Core Parameters

| Parameter | Value | Description |
|---|---:|---|
| $\alpha$ | 25 | Feasible-assignment reward coefficient |
| $\beta$ | 100 | Penalty for infeasible assignments |
| $\lambda$ | 15 | Pairwise multiple-assignment penalty |
| $P$ | 8 | Trotter slices in SQA |
| $N_{\text{sweeps}}$ | 1000 | Monte Carlo sweeps in SQA |

---

## 4. Optimization Approaches

![System architecture](imgs_readme/architecture.jpg)

The workflow shares common preprocessing stages before branching into two solver-centered pipelines:
- **Simulated Quantum Annealing (SQA)** via OpenJij, with greedy fallback/repair when needed
- **Classical Simulated Annealing (SA)** via Neal, with a greedy baseline and fallback/repair role

### 4.1 Batch Decomposition Strategy

The full problem is decomposed into manageable subproblems:

$$
\text{Full variable count} = 20{,}000 \times 800 = 16{,}000{,}000
$$

$$
\text{Batch variable count} = 80 \times 20 = 1{,}600
$$

**Iterative procedure**
1. Initialize remaining capacities $R_j = K_j$
2. Sort cameras by priority score $p_i l_i$
3. While unassigned cameras remain:
   - select the next batch of $B=80$ highest-priority unassigned cameras
   - select $M=20$ candidate servers using a **composite ranking** based on remaining capacity, average assignment cost to the batch, and per-batch feasibility
   - build the surrogate batch QUBO using current $R_j$
   - solve with the selected pipeline
   - update assignments and residual capacities
4. Apply local post-processing across batch boundaries

### 4.2 Simulated Quantum Annealing (SQA)

**Library:** OpenJij 0.11.6  
**Sampler:** `SQASampler`  
**Mechanism:** path-integral Monte Carlo with Suzuki–Trotter decomposition

SQA maps the QUBO problem into an extended $(N\times P)$-spin classical Ising system:

$$
H_{\text{SQA}} = -\frac{1}{P}\sum_{k=1}^{P} H_{\text{QUBO}}^{(k)} - J_{\perp}\sum_{k=1}^{P}\sum_{i=1}^{N} \sigma_{i,k}\sigma_{i,k+1}
$$

where:
- $P=8$ — Trotter slices
- $H_{\text{QUBO}}^{(k)}$ — $k$-th replica of the QUBO Hamiltonian
- $\sigma_{i,k} \in \{\pm1\}$ — classical Ising spins
- $J_{\perp}$ — inter-replica coupling

**Released pipeline notes**
- targets `trotter = 8` and `num_sweeps = 1000` when supported by the installed OpenJij API
- applies an adaptive per-batch attempt/read budget in practice
- uses greedy only as a **fallback/repair mechanism**, not as an independent comparison solver within the SQA branch

### 4.3 Classical Simulated Annealing (SA)

**Library:** D-Wave Neal (`SimulatedAnnealingSampler`)  
**Mechanism:** thermal Metropolis–Hastings search

The Ising objective is:

$$
H(\sigma) = \sum_{i<j} J_{ij}\sigma_i\sigma_j + \sum_i h_i\sigma_i
$$

with spin mapping $\sigma_i = 2x_i - 1$.

**Released benchmark description**
- base read budget: 150
- geometric cooling factor: $\alpha = 0.995$
- runtime varies across runs due to stochasticity and system conditions

### 4.4 Greedy Baseline and Greedy Repair

The greedy heuristic plays two different roles in the released framework:

1. **Greedy baseline** — an explicit comparison method in the benchmark
2. **Greedy fallback/repair** — a safety mechanism inside the annealing pipelines when needed

In the SA architecture summary, the greedy baseline is described by the deterministic priority-capacity score

$$
\operatorname*{argmax}_j \left[b_i (1-c_{ij})\frac{R_j}{K_j}\right]
$$

which is used as a straightforward non-stochastic comparison path. In addition, greedy-style repair is used downstream to correct weak or incomplete batch assignments.

---

## 5. Results and Visualization

### 5.1 Final Performance Comparison

All final objective values below are **post-hoc evaluation scores**, not raw batch-Hamiltonian energies. They combine assignment cost, uncovered-camera penalty, and overload penalty.

| Method | Objective Value | Coverage (%) | Mean Time (s) | Mean Throughput (cam/s) |
|---|---:|---:|---:|---:|
| **Simulated Quantum Annealing (SQA)** | 7,108.3 | 99.6 | 299.61 | 66.46 |
| **Classical Simulated Annealing (SA)** | 7,108.1 | 99.6 | 2,779.26 | 7.22 |
| **Greedy baseline** | 225,163.4 | 25.33 | 180.01 | 28.14 |

### 5.2 Main Findings

- **Comparable solution quality:** SQA and SA both achieve 99.6% coverage.
- **Runtime advantage:** SQA is about **9.3× faster** than SA by mean runtime.
- **Throughput advantage:** SQA reaches **66.46 cameras/s**, versus **7.22 cameras/s** for SA.
- **Stability:** SQA shows very low runtime variation across runs.
- **Greedy limitations:** the greedy baseline is fast and predictable but leaves many cameras unassigned and produces much worse evaluation scores.

### 5.3 Logging and Visualization

Both implementations record structured JSONL logs for each batch. Logged fields include:
- `run_id`
- `batch_idx`
- `assignments`
- `energy`
- `best_energy`
- coverage and timing metrics

These records feed the repository’s dashboards and post-hoc visual analyses, including:
- coverage progression
- batch success rate
- energy evolution
- conceptual 3D energy-landscape views

### 5.4 Dashboard Applications

- **`app.py`** — dashboard for the classical pipeline
- **`app_Q.py`** — dashboard for the quantum-inspired pipeline

The dashboards visualize coverage, comparison metrics, success rate, energy evolution, and 3D energy-landscape navigation from the logged runs.

---

## 6. Getting Started

### Prerequisites
- Python 3.8+
- `pip`
- optional: CMake and a C++17 compiler for source builds

### Install dependencies

```bash
pip install -r requirements.txt
```

### Install OpenJij

```bash
pip install openjij==0.11.6
python -c "import openjij; print(openjij.__version__)"
```

### Run the experiments

```bash
python main.py      # classical SA pipeline
python main_Q.py    # SQA pipeline
```

### Launch dashboards

```bash
python app.py       # default port 8050
python app_Q.py     # default port 8051
```

---

## 7. Repository Structure

```text
Quantum-Sensor-Assignment/
├── imgs_readme/              # README figures
├── logs/                     # SA logs and checkpoints
├── logs_openjij_windows/     # SQA logs and checkpoints
├── templates/                # dashboard HTML templates
├── main.py                   # classical SA implementation
├── main_Q.py                 # OpenJij SQA implementation
├── app.py                    # SA dashboard
├── app_Q.py                  # SQA dashboard
├── requirements.txt          # Python dependencies
├── cost_matrix.npy           # cached cost matrix (if generated)
├── cost_matrix.txt           # optional exported cost matrix
└── README.md                 # repository documentation
```

---

## 8. Citation

If you use this repository or its benchmark results, please cite the associated paper and/or repository.

```bibtex
@misc{mussabayev2026quantum_sensor_assignment,
  title={Simulated quantum annealing for city-scale camera-to-edge server assignment using batch-decomposed QUBO with reproducible open-source benchmarking},
  author={Yedige Mussabayev, Artem Bykov, Evgeniy Lavrov, Diana Yermekova},
  year={2026},
  howpublished={GitHub repository},
  url={https://github.com/Yedman3585/Quantum-Sensor-Assignment}
}
```

---

## 9. License and Acknowledgments

This repository accompanies the scientific manuscript on batch-decomposed QUBO optimization for smart-city sensor-to-server assignment.

The research and software development were led by **Yedige Mussabayev**. Scientific supervision and methodological guidance were provided by **Artem Bykov**. Additional scientific review and recommendations were provided by **Evgeniy Lavrov**.
