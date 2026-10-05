# Campaign 2 summary

Paired tests: SQA-v3 objective minus method (negative = SQA-v3 better); Holm correction within each table; rank-biserial r < 0 favours SQA-v3.

### QUBO-v3 architecture (per-camera window + prices): 20,000 cameras, utilisation 75%, seeds n=5

| Method | Objective mean ± sd | Gap to lower bound | Uncovered (total) | Time, s |
|---|---|---|---|---|
| greedy | 6,102.4 ± 73.7 | 4.36% | 0 | 1.8 |
| regret | 6,100.1 ± 73.2 | 4.32% | 0 | 11.6 |
| exact batch MILP | 6,100.0 ± 73.2 | 4.32% | 0 | 4.2 |
| ILS (equal time) | 6,100.4 ± 73.3 | 4.33% | 0 | 207.3 |
| SA-v3 | 6,101.8 ± 72.7 | 4.35% | 0 | 10.4 |
| SQA-v3 | 6,100.5 ± 72.6 | 4.33% | 0 | 617.1 |
| global solution (offline) | 5,864.9 ± 74.9 | 0.30% | 0 | 86.8 |

| SQA-v3 vs | mean diff | better / tie / worse | Wilcoxon p | Holm p | rank-biserial r |
|---|---|---|---|---|---|
| greedy | -1.93 | 5 / 0 / 0 | 0.0625 | 0.312 | -1.00 |
| regret | +0.44 | 1 / 0 / 4 | 0.438 | 1 | +0.47 |
| exact batch MILP | +0.45 | 1 / 0 / 4 | 0.438 | 1 | +0.47 |
| ILS (equal time) | +0.05 | 2 / 0 / 3 | 1 | 1 | +0.07 |
| SA-v3 | -1.26 | 5 / 0 / 0 | 0.0625 | 0.312 | -1.00 |

### QUBO-v3 architecture (per-camera window + prices): 20,000 cameras, utilisation 90%, seeds n=5

| Method | Objective mean ± sd | Gap to lower bound | Uncovered (total) | Time, s |
|---|---|---|---|---|
| greedy | 6,390.4 ± 92.5 | 8.37% | 0 | 2.1 |
| regret | 6,387.2 ± 90.0 | 8.32% | 0 | 15.4 |
| exact batch MILP | 6,387.0 ± 89.9 | 8.32% | 0 | 5.4 |
| ILS (equal time) | 6,388.7 ± 89.7 | 8.35% | 0 | 209.1 |
| SA-v3 | 6,392.0 ± 92.7 | 8.40% | 0 | 16.3 |
| SQA-v3 | 6,390.3 ± 90.8 | 8.37% | 0 | 732.9 |
| global solution (offline) | 5,928.7 ± 72.8 | 0.54% | 0 | 88.0 |

| SQA-v3 vs | mean diff | better / tie / worse | Wilcoxon p | Holm p | rank-biserial r |
|---|---|---|---|---|---|
| greedy | -0.09 | 3 / 0 / 2 | 1 | 1 | -0.07 |
| regret | +3.03 | 0 / 0 / 5 | 0.0625 | 0.312 | +1.00 |
| exact batch MILP | +3.23 | 0 / 0 / 5 | 0.0625 | 0.312 | +1.00 |
| ILS (equal time) | +1.57 | 2 / 0 / 3 | 0.625 | 1 | +0.33 |
| SA-v3 | -1.75 | 3 / 0 / 2 | 0.312 | 0.938 | -0.60 |

### QUBO-v3 architecture (per-camera window + prices): 20,000 cameras, utilisation 95%, seeds n=5

| Method | Objective mean ± sd | Gap to lower bound | Uncovered (total) | Time, s |
|---|---|---|---|---|
| greedy | 6,576.1 ± 115.1 | 11.06% | 1 | 2.1 |
| regret | 6,577.9 ± 120.6 | 11.09% | 1 | 15.3 |
| exact batch MILP | 6,574.4 ± 119.6 | 11.03% | 0 | 5.5 |
| ILS (equal time) | 6,575.7 ± 120.2 | 11.05% | 0 | 207.8 |
| SA-v3 | 6,581.9 ± 124.8 | 11.16% | 2 | 20.6 |
| SQA-v3 | 6,579.3 ± 122.8 | 11.11% | 2 | 672.2 |
| global solution (offline) | 5,962.9 ± 74.1 | 0.70% | 0 | 95.7 |

| SQA-v3 vs | mean diff | better / tie / worse | Wilcoxon p | Holm p | rank-biserial r |
|---|---|---|---|---|---|
| greedy | +3.23 | 3 / 0 / 2 | 0.812 | 1 | -0.20 |
| regret | +1.37 | 2 / 0 / 3 | 0.812 | 1 | +0.20 |
| exact batch MILP | +4.84 | 1 / 0 / 4 | 0.188 | 0.938 | +0.73 |
| ILS (equal time) | +3.60 | 1 / 0 / 4 | 0.438 | 1 | +0.47 |
| SA-v3 | -2.64 | 4 / 0 / 1 | 0.312 | 1 | -0.60 |

### QUBO-v3 architecture (per-camera window + prices): 20,000 cameras, utilisation 98%, seeds n=5

| Method | Objective mean ± sd | Gap to lower bound | Uncovered (total) | Time, s |
|---|---|---|---|---|
| greedy | 6,886.8 ± 217.3 | 15.94% | 14 | 2.0 |
| regret | 6,850.1 ± 197.2 | 15.32% | 4 | 14.7 |
| exact batch MILP | 6,838.6 ± 199.4 | 15.13% | 0 | 6.5 |
| ILS (equal time) | 6,855.2 ± 218.0 | 15.41% | 4 | 207.5 |
| SA-v3 | 6,856.1 ± 209.8 | 15.43% | 1 | 25.0 |
| SQA-v3 | 6,850.9 ± 203.1 | 15.34% | 2 | 733.5 |
| global solution (offline) | 5,997.9 ± 74.1 | 0.98% | 0 | 99.8 |

| SQA-v3 vs | mean diff | better / tie / worse | Wilcoxon p | Holm p | rank-biserial r |
|---|---|---|---|---|---|
| greedy | -35.86 | 4 / 0 / 1 | 0.125 | 0.5 | -0.87 |
| regret | +0.74 | 1 / 0 / 4 | 0.625 | 1 | +0.33 |
| exact batch MILP | +12.30 | 0 / 0 / 5 | 0.0625 | 0.312 | +1.00 |
| ILS (equal time) | -4.36 | 2 / 0 / 3 | 1 | 1 | -0.07 |
| SA-v3 | -5.25 | 3 / 0 / 2 | 0.438 | 1 | -0.47 |

## Shared 80×20 window, no prices (first-submission architecture)

| Utilisation | exact batch MILP | greedy | regret | gap of exact to lower bound |
|---|---|---|---|---|
| 75% | 11,846.3 | 11,849.0 | 11,837.9 | 102.6% |
| 90% | 11,458.0 | 11,542.6 | 11,520.6 | 94.3% |
| 95% | 11,294.8 | 11,416.2 | 11,252.1 | 90.8% |
| 98% | 11,141.5 | 11,339.2 | 11,135.6 | 87.6% |

## Ablation on hard batches (95% utilisation, per-camera window)

| Variant | batches | mean gap to batch optimum | median | optimal batches |
|---|---|---|---|---|
| exact | 40 | 0.000% | 0.000% | 40 |
| greedy | 40 | 0.735% | 0.613% | 0 |
| regret | 40 | 0.023% | 0.000% | 33 |
| regret_ls | 40 | 0.020% | 0.000% | 34 |
| base:SQA | 40 | 0.872% | 0.682% | 0 |
| base:SA | 40 | 1.642% | 1.547% | 0 |
| R:SQA | 40 | 0.208% | 0.120% | 3 |
| R:SA | 40 | 0.148% | 0.106% | 8 |
| DW:SQA | 40 | 0.585% | 0.479% | 0 |
| DW:SA | 40 | 0.368% | 0.284% | 0 |
| R+DW:SQA | 40 | 0.137% | 0.074% | 8 |
| R+DW:SA | 40 | 0.085% | 0.015% | 17 |
| R+DW:SQA [adaptive] | 40 | 0.059% | 0.032% | 13 |
| R+DW:SA [adaptive] | 40 | 0.023% | 0.000% | 22 |
| R+DW:SQA [beta20] | 40 | 0.833% | 0.001% | 20 |