# Experiments: setup, commands, results

Every experiment in this repository is listed here with:

- its purpose;
- its job list and the folder where its data lives;
- the script that summarises it;
- its main result.

Method: [`METHODOLOGY.md`](METHODOLOGY.md). Metric definitions: [`METRICS.md`](METRICS.md). All gaps are to the Lagrangian lower bound (LB) of the same instance unless stated otherwise.

**Contents**

1. [Environment and running jobs](#1-environment-and-running-jobs)
2. [Overview of all experiments](#2-overview-of-all-experiments)
3. [Decomposition: shared window → static prices → re-pricing](#3-decomposition)
4. [Batch solvers under QUBO-v3.1 (campaign 3)](#4-batch-solvers-under-qubo-v31)
5. [Encoding ablation on hard batches](#5-encoding-ablation)
6. [Information for re-pricing and the demand margin](#6-information-and-demand-margin)
7. [Online-allocation baselines](#7-online-allocation-baselines)
8. [Clustered city layout](#8-clustered-city-layout)
9. [Sensitivity: batch size and re-pricing period](#9-sensitivity)
10. [Scale and priority weight](#10-scale-and-priority-weight)
11. [Run time](#11-run-time)
12. [Service and load metrics](#12-service-and-load-metrics)
13. [Anti-affinity extension](#13-anti-affinity-extension)
14. [Figures](#14-figures)

---

## 1. Environment and running jobs

**Requirements.** `requirements.txt`: Python 3.10, NumPy 1.24.3, SciPy 1.10.1 (HiGHS), OpenJij 0.11.6, dwave-neal 0.5.7, dimod 0.12.6. On Windows, with several Python versions installed, use `py -3.10`.

**One trajectory.** For example, the proposed policy with exact batches at 95% on seed 42:
```
python qsa/v3_trajectory.py --method exact --target-util 95 --seed 42 \
       --price --price-window --reprice 5 --k-slack 2 --forecast-bias 0.05 --out results/v3/example
```

Main options:

| Option | Meaning |
|---|---|
| `--method {greedy,regret,exact,ils,sa_v3,sqa_v3}` | batch solver |
| `--polish` | local search on the decoded QUBO reads |
| `--forecast {oracle,twin,noise}` | information used for re-pricing |
| `--forecast-bias b` | demand margin |
| `--layout {uniform,clustered}` | camera and server layout |
| `--weight {inv,prio}` | priority weight $w_i=4-p_i$ or $w_i=p_i$ |
| `--batch-size`, `--reprice` | batch size B and re-pricing period $T_p$ |
| `--n-cameras`, `--n-servers` | instance size |
| `--trace` | store the per-batch objective, QUBO size and assignment |

**Job lists.** A job list has one command per line, with `{py}` standing for the interpreter. The runner is resumable: finished jobs are recorded in `<jobfile>.done`.
```
python experiments/run_jobs.py experiments/<jobs>.txt --workers 4
```

**Machines.**

- Intel i7-11370H laptop ("ASUS"), 4 parallel jobs.
- Apple-silicon MacBook Air.
- 2-vCPU cloud machine.

Objective values agree to 1e−12 across machines. Run times are only compared within one machine.

---

## 2. Overview of all experiments

| # | Experiment | Scale, seeds | Job list | Data | Summary script |
|---|---|---|---|---|---|
| 1 | Campaign 1: first multi-seed study (capacity-scale protocol) | 20k, 9–10 seeds | – | `results/v3/campaign/` | `campaign_aggregate.py` |
| 2 | Campaign 2: QUBO-v3 (static prices), shared window, static ablation, scaling | 456 runs | `jobs_campaign2_local*.txt` | `results/v3/campaign2/` | `c2_aggregate.py` |
| 3 | v3.1 prototype: re-pricing and headroom candidates | 5k, 5 seeds | – | `results/v3/v31_prototype/` | `v31_aggregate.py` |
| 4 | **Campaign 3**: QUBO-v3.1, 7 batch solvers, 75–98% | 20k, 42–51, 280 runs | `jobs_campaign3_{a,b}.txt` | `results/v3/campaign3/` | `c3_aggregate.py` |
| 5 | Forecast probe | 5k | – | `results/v3/forecast_probe_5k/` | `fc5k_summary.md` |
| 6 | **Forecast experiment** (oracle, twin, noise, bias ±10%) | 20k, 42–51 | `jobs_forecast_{a,b}.txt` | `results/v3/forecast/` | `fc20k_aggregate.py` |
| 7 | **Margin tuning** (0–30%) | 20k, **100–109** | `jobs_margin_tuning.txt` | `results/v3/margin_tuning/` | `r2_tables.py` |
| 8 | **Margin +5% on the test seeds** | 20k, 42–51 | `jobs_test_margin05.txt`, `jobs_asus_r3.txt` | `results/v3/margin_test/` | `r2_tables.py` |
| 9 | v3.1 ablation on hard batches | 57 batches | `jobs_ablation_v31.txt` | `results/v3/ablation_v31/` | `qubo_v3_lab.py` |
| 10 | Clustered layout | 20k, 42–51 | `jobs_clustered.txt` | `results/v3/clustered/` | `r2_tables.py` |
| 11 | Online baselines: LP, LP+5%, DMD, rolling horizon | 20k, 42–51 | `jobs_online_*.txt` | `results/v3/online_baselines/` | `r2_tables.py` |
| 12 | Sensitivity B, $T_p$ without margin / with +5% | 20k, 42–51 | `jobs_sensitivity.txt`, `jobs_mac_r3.txt` | `results/v3/sensitivity{,_m05}/` | `r2_tables.py` |
| 13 | Scale 50k × 2k, v3.1 without margin / with +5% | 42–44 | `jobs_scale50k_v31.txt`, `jobs_mac_r3.txt` | `results/v3/scale_v31{,_m05}/` | `r2_tables.py` |
| 14 | Priority weight $w_i=p_i$ | 20k, 42–51 | `jobs_weight_c.txt` | `results/v3/weight_prio/` | `weight_aggregate.py` |
| 15 | Timing series (single machine) | 20k, 42–46 | `jobs_timing_asus.txt` | `results/v3/timing_asus/` | – |
| 16 | Service and load metrics (both weights, with offline assignments) | 20k, 42–46 | `jobs_service_traces.txt` | `results/v3/svctrace/` | `service_metrics.py` |
| 17 | Anti-affinity probe | 5k | – | `results/v3/anti_affinity_probe/` | – |

---

## 3. Decomposition

Exact batch solver, 20k × 800, seeds 42–51 (campaigns 2 and 3, forecast experiment, margin test).

| Decomposition | 75% | 90% | 95% | 98% |
|---|---|---|---|---|
| Shared 80×20 window (first generation) | 102.1% | 94.3% | 90.0% | 86.8% |
| Per-camera window, static prices (QUBO-v3) | 4.33% | 8.55% | 11.88% | 16.64% |
| + headroom candidates, re-pricing every 5 batches (QUBO-v3.1) | 0.45% | 1.03% | 1.64% | 3.35% |
| **+ demand margin 5%** | **0.29%** | **0.61%** | **0.89%** | **1.16%** |
| Offline solution (reference) | 0.29% | 0.57% | 0.74% | 1.03% |

QUBO-v3.1 covers every camera. With the margin, the sequential policy is within 0–0.15 pp of the offline solution, which sees all cameras and may revise everything.

---

## 4. Batch solvers under QUBO-v3.1

**Campaign 3** uses exact information without a margin. Values are mean objective (gap to LB); no method leaves a camera uncovered. Times are on ASUS with 4 parallel jobs.

| Method | 75% | 90% | 95% | 98% | Time (s) |
|---|---|---|---|---|---|
| Greedy | 5,921.6 (0.46%) | 6,006.4 (1.05%) | 6,074.3 (1.74%) | 6,196.5 (3.43%) | 113 |
| Regret | 5,921.1 (0.45%) | 6,005.4 (1.03%) | 6,068.4 (1.64%) | 6,196.5 (3.42%) | 121 |
| Exact batch MILP | 5,921.0 (0.45%) | 6,005.4 (1.03%) | 6,068.4 (1.64%) | 6,192.2 (3.35%) | 109 |
| ILS | 5,921.0 (0.45%) | 6,005.6 (1.03%) | 6,067.2 (1.62%) | 6,192.1 (3.35%) | 319 |
| SA-v3.1 | 5,923.8 (0.50%) | 6,009.1 (1.09%) | 6,072.6 (1.72%) | 6,195.7 (3.41%) | 112 |
| SQA-v3.1 | 5,924.2 (0.50%) | 6,012.3 (1.15%) | 6,075.3 (1.76%) | 6,198.6 (3.46%) | 308 |
| SQA-v3.1+LS | 5,921.3 (0.46%) | 6,005.7 (1.04%) | 6,067.6 (1.63%) | 6,189.3 (3.31%) | 397 |

Paired tests: Holm-corrected Wilcoxon $p$; W/L counts the wins and losses of the SQA variant.

| Comparison | 75% | 90% | 95% | 98% |
|---|---|---|---|---|
| SQA+LS vs greedy | −0.3 (7/3) 0.49 | −0.7 (5/5) 1.0 | **−6.7 (10/0) 0.012** | −7.2 (8/2) 0.12 |
| SQA+LS vs exact | +0.2 (1/9) 0.26 | +0.3 (3/7) 1.0 | −0.8 (7/3) 1.0 | −2.9 (7/3) 0.55 |
| SQA vs exact | +3.1 (0/10) 0.012 | +6.9 (1/9) 0.012 | +6.9 (0/10) 0.012 | +6.4 (3/7) 0.65 |
| SQA vs SA | +0.4 (4/6) 0.38 | +3.1 (0/10) 0.012 | +2.6 (3/7) 0.65 | +2.9 (3/7) 1.0 |

**With the +5% margin** (exact information, test seeds):

| Method | 90% | 95% | 98% |
|---|---|---|---|
| Exact batch MILP | 0.61 | 0.89 | 1.16 |
| Greedy | 0.61 | 0.91 | 1.18 |
| SQA-v3.1+LS | 0.61 | 0.89 | 1.17 (n.s. vs exact, p ≥ 0.19) |
| SA-v3.1 | 0.62 | 0.92 | 1.19 (behind exact at 95/98, p = 0.002) |

**Reading.**

- All solvers are within about 0.1 pp of each other. The coordination layer, not the batch solver, determines quality.
- Pure SQA is the weakest by a small margin and no better than SA.
- With local search, SQA is indistinguishable from exact batch optimisation.

---

## 5. Encoding ablation

Hard batches are those where greedy is at least 0.5% worse than the batch MILP. Gaps are to the exact batch optimum on the priced batch objective.

**Static prices (QUBO-v3), 80 hard batches, 95%** (`results/v3/campaign2/ablation*`)

| Variant | SQA gap | SQA optimal | SA gap | SA optimal |
|---|---|---|---|---|
| one-hot (base) | 0.829% | 0/80 | 1.619% | 0/80 |
| + reduction (R) | 0.185% | 8/80 | 0.137% | 17/80 |
| + domain wall (DW) | 0.554% | 0/80 | 0.325% | 1/80 |
| R + DW | 0.122% | 14/80 | 0.082% | 31/80 |
| R + DW + adaptive | **0.063%** | 24/80 | **0.031%** | 42/80 |
| greedy / regret / regret+LS | 0.789% / 0.016% / 0.012% | | | |

**QUBO-v3.1, 57 hard batches, 95%** (`results/v3/ablation_v31/`). All of these batches lie at the end of the horizon (batches ~243–249).

| Variant | Gap, SQA | Gap, SA | Time per batch, SQA / SA | Overloaded servers before repair |
|---|---|---|---|---|
| base (one-hot) | 5.55% | 7.01% | 1.50 s / 0.17 s | 2.2 |
| R | 3.45% | 4.20% | 1.10 / 0.12 | 2.0 |
| DW | 5.33% | 5.73% | 1.43 / 0.11 | 2.5 |
| R + DW | 3.42% | 3.45% | 1.04 / 0.08 | 2.1 |
| R + DW + adaptive | 2.92% | 3.10% | 2.46 / 0.19 | 1.2 |
| greedy / regret / regret+LS | 2.71% / 0.55% / 0.27% | | | |

**Reading.**

- The reduction is the main component (−2.1 pp, 10/10 seeds, p = 0.002).
- Pure annealing on the hardest batches is no better than greedy (p = 0.11). This is why the QUBO is used with local search.
- SQA ≈ SA (p ≥ 0.19), and SA is about 12× faster.
- In absolute terms the loss is about 6 objective units per seed out of about 6,100.

---

## 6. Information and demand margin

### 6.1 Margin tuning on validation seeds 100–109 (exact batches)

| Information | +0 | +5% | +10% | +15% | +20% | +30% |
|---|---|---|---|---|---|---|
| exact, 90% | 0.95 | **0.57** | 0.98 | 1.56 | 1.93 | 2.11 |
| exact, 95% | 1.74 | **0.92** | 1.44 | 1.68 | 1.86 | 1.87 |
| exact, 98% | 3.47 | **1.16** | 1.70 | 1.73 | 1.70 | 1.65 |
| statistics only, 90% | 2.07 | **1.41** | 1.45 | 1.88 | 2.22 | 2.45 |
| statistics only, 95% | 3.27 | **1.84** | 2.12 | 2.35 | 2.47 | 2.56 |
| statistics only, 98% | 4.97 | **2.44** | 2.66 | 2.73 | 2.74 | 2.89 |

+5% is best in all 6 rows and beats +0 on 10/10 seeds in every row. It was fixed before the test seeds were run.

### 6.2 Test seeds 42–51 (exact batches)

Holm correction over 8 comparisons. Effects are relative to exact information without a margin.

| Information for re-pricing | 75% | 90% | 95% | 98% |
|---|---|---|---|---|
| static prices | 4.33 | 8.55 | 11.85 | 16.48 |
| exact (oracle) | 0.45 | 1.03 | 1.64 | 3.35 |
| **exact + 5%** | **0.29** | **0.61** | **0.89** | **1.16** |
| statistics only (twin) | – | 2.56 | 3.40 | 4.93 |
| **statistics only + 5%** | 0.93 | 1.47 | 1.83 | 2.51 |
| noise ±30% per camera | – | 1.19 | 1.84 | 3.63 |
| demand underestimated by 10% | – | 5.22 | 8.21 | 12.94 |
| demand overestimated by 10% | – | 0.99 | 1.50 | 1.36 |

**Reading.**

- The margin improves exact information on 10/10 seeds at every utilisation.
- With statistics only and the margin, the policy is within 0.2–0.4 pp of exact information without a margin at 90–95%, and better than it at 98% (2.51 vs 3.35).
- Underestimating demand is much worse than overestimating it. This asymmetry is what motivates the margin.
- Uncorrelated noise of ±30% costs only 0.2–0.3 pp.

---

## 7. Online-allocation baselines

Exact batches, seeds 42–51. Times are per run on the Mac at 95%, except DMD and rolling horizon, which were timed on ASUS.

| Policy | 90% | 95% | 98% | Time |
|---|---|---|---|---|
| Static prices | 8.55 | 11.85 | 16.48 | 6 s |
| Rolling-horizon MILP, 2 batches | 26.55 | 35.53 | 46.16 [36 uncovered] | 22 s (ASUS) |
| DMD, η = 1, zero start | 13.31 | 17.13 | 22.20 | 51 s (ASUS) |
| DMD, η = 0.1, Lagrangian start | 6.42 | 10.16 | 15.38 | 50 s (ASUS) |
| Lagrangian re-pricing (v3.1) | 1.03 | 1.64 | 3.35 | 36 s |
| LP bid prices | 0.79 | 1.29 | 2.29 | 494–687 s |
| **Lagrangian + 5% (ours)** | 0.61 | **0.89** | 1.16 | 36–37 s |
| LP bid prices + 5% | **0.44** | 0.97 | **0.96** | 467–973 s |

**Reading.**

- DMD and rolling horizon lose to v3.1 on 10/10 seeds. Neither uses the demand that is still to come, and DMD barely beats static prices.
- Exact LP duals beat the Lagrangian prices (10/10 seeds) when neither has a margin.
- The margin helps both. LP+5% vs Lagrangian+5%: −10 at 90% (p = 0.002), +5 at 95% (n.s.), −12 at 98% (p = 0.004) objective units.
- So the Lagrangian+5% policy is within 0.2 pp of LP+5% at 1/8 (ASUS) to 1/17 (Mac) of the run time.

---

## 8. Clustered city layout

12 urban centres, exact batches, seeds 42–51. The LB is recomputed for the clustered instances.

| Policy | 90% | 95% | 98% |
|---|---|---|---|
| Static prices | 12.12 | 16.08 | 21.84 |
| Exact information | 1.13 | 1.83 | 4.25 |
| **Exact + 5%** | **0.94** | **1.42** | **1.84** |
| Statistics only + 5% | 1.61 | 2.16 | 2.96 |

Every conclusion from the uniform layout holds. The margin wins on 10, 9 and 10 of 10 seeds. Greedy and SQA-v3.1+LS stay within 0.1 pp of exact batches.

---

## 9. Sensitivity

Exact information, exact batches, seeds 42–51. Values are gap to LB (%); in brackets is the number of seeds better than the default ($B=80$, $T_p=5$). Times are on the Mac.

| Setting | 95% no margin | 95% +5% | 98% no margin | 98% +5% | Time 95% |
|---|---|---|---|---|---|
| B = 40 | 1.71 | 0.82 (10/10) | 3.63 | 0.98 (10/10) | 73 s |
| **B = 80, T_p = 5** | 1.64 | **0.89** | 3.35 | **1.16** | 36 s |
| B = 160 | 1.57 | 1.20 (0/10) | 2.76 | 2.00 [1 unc.] (0/10) | 20 s |
| B = 320 | 1.56 | 2.28 [17 unc.] (0/10) | 3.13 | 8.28 [197 unc.] (0/10) | 38 s |
| T_p = 1 | 1.66 | 0.79 (10/10) | 3.63 | 0.93 (10/10) | 178 s |
| T_p = 10 | 1.54 | 1.22 (0/10) | 2.81 | 1.96 (0/10) | 20 s |
| T_p = 25 | 1.65 | 1.71 (0/10) | 3.01 | 3.03 (0/10) | 9 s |

**Reading.**

- Without the margin, the policy is insensitive to B and $T_p$: within ±0.1 pp at 95%.
- With the margin, finer re-pricing is monotonically better. The default sits at the knee: 0.1–0.2 pp from $T_p=1$ or $B=40$ at 2–5× less time.
- The margin is tuned jointly with the re-pricing period. 5% is not enough when prices are refreshed rarely.

---

## 10. Scale and priority weight

**50,000 × 2,000, 95%, seeds 42–44.** Gap to LB in %; all cameras are covered under v3.1.

| Policy | Exact | Greedy | SQA-v3.1+LS |
|---|---|---|---|
| Static prices (v3) | 12.25 | 12.41 | 12.26 (SQA-v3) |
| Exact information | 1.36 | 1.42 | 1.41 |
| Exact + 5% | 1.05 | 1.09 | – |
| Statistics only + 5% | 1.52 | 1.55 | – |

The gap does not grow with size: at 20k the corresponding values are 1.64% (exact information) and 0.89% (+5%). The batch QUBO does not grow either: the median is 7 variables at 50k.

**Priority weight $w_i=p_i$, seeds 42–51, exact batches.** The LB is recomputed with this weight.

| Policy | 90% | 95% | 98% |
|---|---|---|---|
| Static prices | 2.65 | 3.79 | 5.96 |
| Exact information | 0.95 | 1.52 | 2.78 |
| Statistics only | 1.69 | 2.23 | 3.39 |

The ranking of policies does not depend on the weight. Static prices lose less here, because the expensive late cameras (low priority) now carry the smallest weight.

---

## 11. Run time

These numbers come from a single machine (ASUS, i7-11370H, `results/v3/timing_asus`): seeds 42–46, 95% and 98%, every method with the +5% margin. The times are wall-clock per 20k instance (250 batches).

| Method | Time per instance | Gap 95% | Gap 98% |
|---|---|---|---|
| Static prices (exact batches) | 9–12 s | 11.9 | 16.5 |
| Exact, greedy, regret, SA-v3.1 | 180–235 s | 0.95–0.98 | 1.30–1.36 |
| ILS | 390–415 s | 0.95–0.98 | 1.30–1.36 |
| SQA-v3.1 | 440–510 s | 0.95–0.98 | 1.30–1.36 |
| SQA-v3.1+LS | 570–650 s | 0.95–0.98 | 1.30–1.36 |
| LP bid prices + 5% | ~1,550 s | – | – |

Re-pricing and window construction take most of the time of the fast solvers. LP bid prices are about 8.5× slower than Lagrangian re-pricing on ASUS and 17× slower on the Mac.

---

## 12. Service and load metrics

**Setup.**

- Seeds 42–46 at 95% and 98%, exact batches.
- Four policies under the default weight: static prices, re-pricing, re-pricing + 5%, statistics only + 5%.
- Three policies under $w_i=p_i$: static prices, re-pricing, re-pricing + 5%.
- Every run uses `--trace`, so the full assignment is stored. The offline solution was recomputed with its assignment by `qsa/offline_global.py`.
- Job list: `jobs_service_traces.txt` (62 runs; 2 cloud vCPUs, about 1 min per run).
- Data: `results/v3/svctrace/`. Summary: `results/v3/service_summary/{service_summary.md, service_runs.csv, saturation_curves.json}`.

```
python qsa/offline_global.py results/v3/svctrace 42 95 42 98 ...            # offline solutions (w = 4 - p)
python qsa/offline_global.py results/v3/svctrace --weight prio 42 95 ...    # offline solutions (w = p)
python qsa/service_metrics.py --traces results/v3/figtrace results/v3/svctrace \
       --offline results/v3/svctrace --out results/v3/service_summary
```

With 5 seeds the results are descriptive (Section 8 of [`METRICS.md`](METRICS.md)).

### 12.1 Price of sequentiality and server load (default weight $w_i=4-p_i$)

| Policy | Gap to LB 95 / 98% | Gap to offline 95 / 98% | Servers ≥ 99% full 95 / 98% | CV of utilisation 95 / 98% | Half capacity saturated at batch (98%) |
|---|---|---|---|---|---|
| Static prices | 12.72 / 18.14 | 11.86 / 16.87 | 80 / 86% | 0.27 / 0.16 | 174 |
| Re-pricing | 1.79 / 4.02 | 1.01 / 2.90 | 79 / 88% | 0.28 / 0.16 | 235 |
| **Re-pricing + 5%** | **0.95 / 1.30** | **0.17 / 0.21** | 23 / 38% | 0.16 / 0.02 | never |
| Statistics only + 5% | 1.91 / 2.64 | 1.13 / 1.54 | 45 / 61% | 0.17 / 0.04 | 243 |
| Offline solution | 0.77 / 1.09 | 0 | 46 / 52% | 0.24 / 0.11 | never |

- **The price of sequentiality almost disappears with the margin.** Re-pricing + 5% is 0.17–0.21% above the offline solution, which sees all cameras and may revise everything. Static prices are 12–17% above it.
- **The mechanism is visible in the load.** With static prices, half of the capacity sits on saturated servers by batch 174 of 250, so the last 75 batches (all low priority) find their nearby servers full. With re-pricing and no margin, saturation comes later (batch 235) but still before the end. With the margin, saturation stays near 15% until the last few batches, and the load is spread almost evenly (CV 0.02 at 98%).

### 12.2 Who pays: service by priority class (default weight, 98%)

| Policy | Mean distance high / medium / low | Low priority on its cheapest server | Contribution to the gap to offline, pp: high / medium / low |
|---|---|---|---|
| Static prices | 48.7 / 30.9 / **61.8** | 37% | −0.57 / −0.59 / **+18.03** |
| Re-pricing | 56.1 / 28.8 / 38.2 | 81% | −0.22 / −0.22 / +3.34 |
| Re-pricing + 5% | **79.1** / 28.3 / 30.3 | 87% | +0.42 / −0.16 / −0.05 |
| Statistics only + 5% | 77.2 / 32.2 / 33.1 | 73% | +0.38 / +0.13 / +1.03 |
| Offline solution | 63.4 / 30.7 / 30.2 | 90% | – |

At 95% the pattern is the same and milder; the high-priority distances are 41.9 / 45.9 / 60.6 / 58.8 / 48.9 in the same row order.

- **Static prices:** the loss falls entirely on low-priority cameras. They come last in the order and travel twice as far as in the offline solution (61.8 vs 30.2).
- **Re-pricing with the margin** gives low-priority cameras the offline service level (30.3 vs 30.2; 87% on their cheapest server).
- **The margin moves high-priority cameras farther** (79 vs 63 offline at 98%). Under $w_i=4-p_i$ a high-priority camera has the *smallest* weight in the objective (1 vs 3), so moving it is the cheapest way to free capacity. The offline solution does the same, only less (63 vs 49 under static prices). This follows from the weight, not from the margin.

### 12.3 The same metrics with $w_i=p_i$ (high priority counts most)

| Policy | Gap to LB 95 / 98% | Gap to offline 95 / 98% | Distance high, 98% | Distance low, 98% |
|---|---|---|---|---|
| Static prices | 4.13 / 10.40 | 2.97 / 8.31 | 34.8 | 65.8 |
| Re-pricing | 1.67 / 3.35 | 0.53 / 1.40 | 35.7 | 47.7 |
| **Re-pricing + 5%** | **1.08 / 1.51** | **−0.05 / −0.41** | 39.8 | 32.9 |
| Offline solution | 1.13 / 1.92 | 0 | 42.0 | 32.2 |

- **The margin transfers to the other weight without re-tuning.** It was tuned under $w_i=4-p_i$ only. Under $w_i=p_i$ it improves re-pricing from 1.67 / 3.35% to 1.08 / 1.51% and is slightly *better* than the offline heuristic, which is weaker under this weight (1.1–1.9% above LB).
- **High-priority cameras are not displaced under this weight:** 39.8 vs 42.0 offline at 98%. Low-priority cameras again get the offline service level (32.9 vs 32.2).
- **Conclusion:** which class absorbs the scarcity is decided by the weights of the objective. The coordination layer reproduces the offline trade-off in both cases.

Figures: `fig3_mechanism` (default weight, 98%), `figS2_mechanism_u95` and `figS4_service_wprio` ($w_i=p_i$).

---

## 13. Anti-affinity extension

Pairwise costs for cameras with overlapping views, on the 5,000-camera map:

- The linearised MILP still solves each 80-camera batch optimally in 0.03–1.2 s.
- ILS (2 s per batch) is within 0.5% of the optimum, SA within 0.4–1.9%, and SQA within 0.6–4.6% (0.2–1.8% with LS).
- With 400-camera batches (about 4,000 pairwise terms), the MILP stops at a 0.6% gap after 60 s, but ILS and SA still beat SQA.

---

## 14. Figures

All figures are produced by one script from the raw results in this repository (vector PDF + 300 dpi PNG):
```
python qsa/make_figures.py --repo . --out figures                  # all figures
python qsa/make_figures.py --repo . --out figures gap_util margin  # selected figures
```

Style: Nature / Scientific Reports guidelines.

- Width 183 mm (double column) or 89 mm (single column).
- Arial-compatible sans-serif font, 6–7 pt, embedded.
- No titles inside the figures: panel letters only, and the description goes in the caption.
- Colour encodes the coordination policy and is identical in every figure. Utilisation is encoded by marker fill (95% hollow, 98% filled), never by colour.
- Logarithmic axes share one tick set.
- Error bars are 95% t-intervals over seeds.

| File | Content | Data |
|---|---|---|
| `fig1_method` | schematic: coordination layer (remaining demand with margin → residual Lagrangian prices → decision costs) and batch layer (windows → reduction → batch solver → commit) | – |
| `fig2_gap_util` | gap vs utilisation for all coordination policies and the offline solution; (a) uniform, (b) clustered | campaign 3, margin test, online baselines, clustered |
| `fig3_mechanism` | where the loss arises at 98%: (a) cumulative excess over offline by batch, seed 42; (b) capacity on full servers by batch; (c) distance by priority class | `figtrace`, `service_summary` |
| `fig4_margin` | (a) margin tuning on the validation seeds; (b) information levels on the test seeds | margin tuning, forecast, margin test |
| `fig5_policy_time` | (a) quality vs time of the coordination policies (95%); (b) time of the batch solvers inside the final policy, labelled with the gap | timing series (ASUS), online baselines |
| `fig6_ablation` | encoding ablation on hard batches, static vs re-pricing, SA vs SQA, on a shared axis | ablation folders |
| `figS1_sensitivity` | gap vs $T_p$ and vs $B$, with and without the margin | sensitivity, sensitivity_m05 |
| `figS2_mechanism_u95` | Fig. 3 at 95% | `figtrace`, `service_summary` |
| `figS3_weight` | policies under $w_i=p_i$ | weight_prio |
| `figS4_service_wprio` | saturation and distance by class under $w_i=p_i$ | `service_summary` |
