# Quantum-Sensor-Assignment

QUBO decomposition for assigning video sensors (cameras) to capacitated edge servers, solved with simulated quantum annealing (OpenJij SQA) and benchmarked against classical heuristics, exact batch MILP and a global Lagrangian reference.

> Status: the accompanying manuscript is under revision. This repository contains the code and the raw experiment logs; the manuscript sources are not included until publication.

## Problem

`N` cameras (default 20,000) with priority `p_i`, load `l_i` (GFLOPS) and position are assigned to `M` servers (default 800) with capacity `K_j`. Cameras arrive in priority order in batches of 80 (online setting). The objective, identical for every method, is

```
sum_i (4 - p_i) * c_ij * x_ij  +  15 * (#uncovered cameras)      s.t.  sum_i l_i x_ij <= K_j
```

where `c_ij` is the normalised cost from `CapacityStressExperiment._build_cost_matrix()`. Server utilisation is controlled by uniformly scaling capacities.

## Method: QUBO-v3

Each batch is turned into a QUBO and sampled with SQA. The components, each evaluated in an ablation:

| Component | What it does |
|---|---|
| Per-camera window | each camera gets its K=5 cheapest residual-feasible servers (instead of 20 servers shared by the whole batch) |
| Capacity prices | Lagrangian prices `u_j` of the full problem, added as `u_j * l_i` to decision costs and used to choose the window |
| Exact reduction | a camera whose cheapest server cannot overflow is fixed before sampling (optimality-preserving); QUBO size drops from ~480 to ~100 variables |
| Domain-wall encoding | the per-camera choice is encoded as a domain wall (Chancellor, 2019), so single spin flips never break the one-server constraint |
| Adaptive capacity penalty | up to 3 resampling rounds with raised price/weight on overflowing servers |
| Tuned SQA | explicit `num_reads=10`, `num_sweeps=2000`, `trotter=8`, `beta=20`, seeded |

## Repository layout

```
qsa/                         current method and evaluation code
  capacity_stress_experiment.py   instance generator (seeded) + shared evaluation; also the original AO/Static/PRC QUBOs
  batch_quality_benchmark.py      batch subproblems, exact MILP (HiGHS), greedy, regret, local search
  qubo_formulation_lab.py         QUBO-v2 formulation experiments on hard batches
  qubo_v3_lab.py                  QUBO-v3 ablation on hard batches (gap to the batch optimum)
  v3_trajectory.py                end-to-end run of one method over all batches
  global_bound_check.py           global Lagrangian lower bound + global feasible solution (no batches)
  campaign_aggregate.py           tables and paired statistics for the multi-seed campaign
baselines/                   classical baselines used in the revision (GRASP, BRKGA, Tabu, regret best-fit, capacity-priced greedy, Lagrangian price, RC-greedy, small MILP oracle)
legacy/                      revision-stage formulations (AO-QUBO, Static-QCP-QUBO, PRC-QUBO, CC-PRC-QUBO) and plotting scripts; see legacy/README_legacy.md
  first_submission/          original pipeline of the first submission: main_Q.py (PRC-QUBO + OpenJij SQA), main.py (SA),
                             greedy.py (priority-capacity greedy), app.py / app_Q.py / gui.py (Dash dashboards over progress logs),
                             and the progress logs of the reported runs
results/
  v3/campaign/               10 seeds x 3 utilisation levels x 2 architectures x 5 methods + global reference
  v3/qubo_v3, v3/qubo_v3_trajectory, v3/qubo_lab, v3/batch_quality, v3/global_bound   development runs (seed 42)
  legacy/                    summaries and run logs of the first-submission experiments
```

## Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the main results

Run from the repository root. Each script writes JSON summaries to its `--out` directory.

```bash
# global reference: Lagrangian lower bound and a global feasible solution
python qsa/global_bound_check.py --scales 1.0,0.33,0.25 --seed 42 --out logs_global

# one end-to-end run (method: exact | greedy | regret | sa_v3 | sqa_v3)
python qsa/v3_trajectory.py --method sqa_v3 --capacity-scale 0.25 --seed 42 --price --price-window --out logs_v3

# QUBO-v3 ablation on the hardest batches
python qsa/qubo_v3_lab.py --window percam --min-gap 0.005 --max-hard 12 --variants base,R,DW,R+DW --solvers SQA,SA --adaptive-rounds 3 --beta 20

# rebuild the campaign tables from the archived logs
python qsa/campaign_aggregate.py results/v3/campaign
```

The full campaign is the job list in `results/v3/campaign/jobs.txt` (330 runs, about 2.5 CPU-hours).

## Main results (10 seeds, ~97% utilisation, seed 43 excluded as infeasible)

| Method | Objective, with prices | Gap to global lower bound |
|---|---|---|
| greedy | 7,164 | 19.5% |
| regret | 7,085 | 18.2% |
| exact batch MILP | 7,052 | 17.7% |
| SA-v3 | 7,045 | 17.6% |
| SQA-v3 | 7,058 | 17.8% |
| global solution (offline reference) | 6,180 | 3.2% |

SQA-v3 is significantly better than greedy (Wilcoxon p = 0.027), better than regret on average (p = 0.074) and statistically indistinguishable from exact batch optimisation and SA-v3. Its main cost is runtime (about 200 s per instance on CPU emulation versus seconds for classical methods). Full tables: `results/v3/campaign/campaign_summary.md`.

## Notes on earlier versions

The first submission reported PRC-QUBO results that should not be reused:

- the reported objective of about 7,108 includes a global local-search refinement (`final_opt`); the same refinement gives 7,108.3 for a plain greedy;
- OpenJij 0.11.x ignores constructor arguments, so earlier SQA runs used the sampler defaults (1 read, 4 Trotter slices). All current code passes parameters to `sample_qubo()`; `--sqa-legacy-defaults` reproduces the old behaviour;
- the 99.6% coverage figure comes from the 99.5% early-stop rule, not from the solver.

## Citation

```bibtex
@misc{mussabayev2026qsa,
  title  = {Quantum-Sensor-Assignment: QUBO decomposition for capacitated sensor-to-edge-server assignment},
  author = {Yedige Mussabayev},
  year   = {2026},
  url    = {https://github.com/Yedman3585/Quantum-Sensor-Assignment}
}
```

## License and acknowledgments

Released under the [MIT License](LICENSE).

The research and software development were led by Yedige Mussabayev. Scientific supervision and methodological guidance were provided by Artem Bykov. Additional scientific review and recommendations were provided by Evgeniy Lavrov.
