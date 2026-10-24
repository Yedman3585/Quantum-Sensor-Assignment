# Metrics and statistics

This document defines every number reported in the paper and in `results/`. For each metric it gives how the number is computed and which script computes it. Notation follows [`METHODOLOGY.md`](METHODOLOGY.md).

**Contents**

1. [Solution quality](#1-solution-quality)
2. [Trajectory metrics](#2-trajectory-metrics)
3. [Service metrics: who gets what](#3-service-metrics)
4. [Server load metrics](#4-server-load-metrics)
5. [Batch-level metrics (solver quality)](#5-batch-level-metrics)
6. [QUBO size metrics](#6-qubo-size-metrics)
7. [Time](#7-time)
8. [Statistics](#8-statistics)
9. [Where each metric is computed](#9-where-each-metric-is-computed)

---

## 1. Solution quality

| Metric | Definition | Notes |
|---|---|---|
| **Objective** $F$ | $\sum_i w_ic_{ij(i)}+P\cdot\#\text{uncovered}$ over the whole trajectory, with **true** costs (no prices) | the same function for every method; stored as `objective` in every run JSON |
| **Gap to LB** | $100\,(F-\mathrm{LB})/\mathrm{LB}$ | main quality measure. LB is the Lagrangian bound of the **same** instance (same seed, utilisation, layout and weight) |
| **Gap to offline** (price of sequentiality) | $100\,(F-F^{\rm off})/F^{\rm off}$ | $F^{\rm off}$ is the offline solution, which sees all cameras and may revise everything. This is the share of the loss that sequential decisions cause. It can be slightly negative, because the offline solution is a heuristic, not the optimum |
| **Uncovered** | cameras with no server, summed over seeds | 0 in all v3.1 runs with the default $B=80$, $T_p=5$ |
| **Coverage** | $100\cdot\#\text{covered}/N$ | |

**Which LB.** Each instance family has its own bound file, always computed with 400 subgradient iterations:

| Instance family | LB file |
|---|---|
| uniform 20k, seeds 42–51 | `results/v3/campaign2/global/*` |
| validation seeds 100–109 | `results/v3/margin_tuning_summary/lb_val_100_109.json` |
| clustered layout | `results/v3/clustered_summary/lb_clustered_20k.json` |
| $w_i=p_i$ | `results/v3/weight_prio_summary/lb_prio_20k.json` |
| 50k | `results/v3/campaign2/scale_global` |

The offline solution is 0.3–1.0% above LB, so LB is nearly tight and a gap to LB is an honest upper estimate of the distance to the optimum.

**Why two references.** Gap to LB answers "how far from the best possible". Gap to offline answers "how much do we lose *because* decisions are sequential and irrevocable". The difference between the two is the gap of the offline heuristic itself.

---

## 2. Trajectory metrics

| Metric | Definition | Use |
|---|---|---|
| **Cumulative excess** | $E_t=100\sum_{\tau\le t}\big(F_\tau-F^{\rm off}_\tau\big)/\mathrm{LB}$, where $F_\tau$ is the true cost of batch $\tau$ and $F^{\rm off}_\tau$ is the offline cost of the **same cameras** | where in the horizon the loss arises (`fig3_mechanism` a). Static prices: $E_t<0$ early (cheaper than offline), then a steep rise at the end. Re-pricing with a margin: a small early premium, then flat |
| **Final excess** | $E_T$, equal to (gap to LB) − (offline gap to LB) | |
| **Half-capacity saturation batch** | the first batch after which the servers holding 50% of the total capacity are at least 99% full | how early capacity becomes scarce (Section 4) |

Data: runs with `--trace`, which store per batch the objective, the coverage, the QUBO size and the full assignment (`cams`, `servers`).

---

## 3. Service metrics

*Who pays for sequential decisions?* These are computed per priority class $p\in\{3,2,1\}$ (high, medium, low) by `qsa/service_metrics.py`.

| Metric | Definition |
|---|---|
| Class coverage | share of the class that is assigned |
| **Mean distance** | mean $d_{ij(i)}$ over the assigned cameras of the class (plane units; the plane is $1000\times1000$) |
| Mean cost | mean unweighted $c_{ij(i)}$ |
| **On cheapest server** | share of the class assigned to its globally cheapest server $\arg\min_jc_{ij}$, ignoring capacity |
| Mean server rank | mean rank of the assigned server in the camera's own cost order (1 = cheapest) |
| **Contribution to the gap to offline** | $100\sum_{i\in p}\big(w_ic_{ij(i)}-w_ic_{ij^{\rm off}(i)}\big)/F^{\rm off}$ in pp; uncovered cameras count $P$. The three classes add up to the gap to offline, so this shows which class carries the loss of the sequential policy. A share of the total excess was rejected because it is unstable when the total is near zero |

These metrics turn an abstract objective gap into service terms: how much farther a camera's stream travels, and whether low-priority cameras, which come last in the order, absorb the whole loss.

---

## 4. Server load metrics

| Metric | Definition |
|---|---|
| **Saturated servers** | share of servers with final utilisation $\ge99\%$ |
| **CV of utilisation** | coefficient of variation (std/mean) of the final per-server utilisation $\sum_{i\to j}l_i/K_j$ |
| Half-capacity saturation batch | Section 2 |
| Saturation curve | share of the total capacity on servers ≥ 99% full after every batch (`saturation_curves.json`, `fig3_mechanism` b) |

At 95–98% utilisation most servers end up nearly full under every policy. The informative part is **when** they become full and **which cameras** were still unplaced at that time.

---

## 5. Batch-level metrics

These metrics isolate the batch solver from the coordination layer.

| Metric | Definition |
|---|---|
| **Batch gap** | $100\,(\hat F_t(x)-\hat F_t^\ast)/\hat F_t^\ast$ on the **priced** batch objective, against the exact batch MILP $\hat F_t^\ast$ on the same batch (same candidates, residuals and prices) |
| **Hard batch** | a batch on which greedy is at least 0.5% worse than the exact batch optimum. Static v3: 80 hard batches (8 per seed, 95%). QUBO-v3.1: 57 hard batches (4–7 per seed, all near the end of the horizon) |
| Ablation gap | mean over seeds of the per-seed mean batch gap on the hard batches |
| Optimal batches | number of hard batches solved to the batch optimum (gap < 1e−9) |
| Overloaded servers before repair | servers over capacity in the best raw read, before decoding repairs |
| Repairs | cameras moved by the repair step after decoding |

**Why hard batches.** In most batches, greedy already finds the batch optimum: under QUBO-v3.1 the exact reduction leaves a median of 2 of 80 cameras free. Averaged over all 250 batches, every solver would look optimal and the encodings could not be told apart. The 0.5% threshold (`--min-gap 0.005`) keeps the batches in which the batch solver can make a difference. Under QUBO-v3.1 they are the last batches of the horizon (about batches 243–249), where the residual capacity is nearly exhausted. The ablation therefore measures the encodings on the hardest subproblems that actually occur; it does not describe a typical batch. The effect on the full trajectory is reported separately (Section 4 of `EXPERIMENTS.md`).

Code: `qsa/qubo_v3_lab.py` (with `--v31` for the v3.1 batches).

---

## 6. QUBO size metrics

| Metric | Definition |
|---|---|
| Free cameras per batch | batch cameras left after the exact reduction (Proposition 2) |
| QUBO variables per batch | $\sum_{\rm free}(\#\text{options}-1)$ with domain wall |
| Fixed share | share of batch cameras fixed by the reduction |
| Batches without QUBO | share of batches in which every camera was fixed |

Stored per run as `qubo_batches`, `qubo_calls`, `vars_sum`, `fixed_sum` and `viol_sum`, and per batch in `--trace` runs.

---

## 7. Time

- **Wall-clock time per instance**, i.e. one full 250-batch trajectory, as `time_sec`. It includes pricing, window construction and batch solving.
- **Pricing time** is reported separately for LP bid prices, because it dominates their run time.
- **Never mix machines in one comparison.** The machines were an Intel i7-11370H laptop ("ASUS", 4 parallel jobs), an Apple-silicon MacBook Air (about 3× faster per run) and a 2-vCPU cloud machine. The definitive timing series is `results/v3/timing_asus` (85 runs, seeds 42–46, all methods with the +5% margin, on one machine).
- Objective values are identical across machines (differences ≤ 1e−12), so quality results can be pooled.

---

## 8. Statistics

- **Unit of analysis.** One instance, i.e. one seed at one utilisation. All comparisons are **paired**: two methods on the same instances.
- **Test.** Two-sided Wilcoxon signed-rank test on the per-seed objective differences.
- **Multiple comparisons.** Holm correction within each table or family of comparisons. For example, the forecast table uses 8 comparisons.
- **Effect size.** Matched-pairs rank-biserial correlation $r$ (Kerby 2014), together with wins/losses over the seeds.
- **Resolution.** With 10 paired seeds the smallest two-sided $p$ is 0.002. With 5 seeds it is 0.0625, so 5-seed comparisons (service metrics, timing) are **descriptive only**.
- **Validation vs test.** The demand margin and all other tuned parameters were chosen on seeds 100–109 and reported once on seeds 42–51. Nothing was re-tuned on the test seeds; the sensitivity results are reported as robustness.
- **Aggregation.** Tables show means over seeds. Ablation gaps are means of per-seed means, so that every seed has equal weight.

---

## 9. Where each metric is computed

| Output | Script | Data |
|---|---|---|
| objective, coverage, QUBO size, time per run | `qsa/v3_trajectory.py`, `qsa/online_baselines.py` | `results/v3/<campaign>/*.json` |
| LB and offline solution | `qsa/global_bound_check.py`, `qsa/lb_prio.py`, `qsa/offline_global.py` | the LB files of Section 1 |
| campaign tables with Wilcoxon/Holm | `qsa/c2_aggregate.py`, `qsa/c3_aggregate.py`, `qsa/fc20k_aggregate.py`, `qsa/weight_aggregate.py`, `qsa/r2_tables.py` | `*_summary.md` in the result folders |
| batch gaps, ablation | `qsa/qubo_v3_lab.py` | `results/v3/ablation_v31`, `results/v3/campaign2/ablation` |
| service and load metrics | `qsa/service_metrics.py` | `results/v3/svctrace`, `results/v3/figtrace` → `results/v3/service_summary/` |
| all figures | `qsa/make_figures.py` | reads the folders above |
