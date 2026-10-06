# Campaign 2 summary

Paired tests: SQA-v3 objective minus method (negative = SQA-v3 better); Holm correction within each table; rank-biserial r < 0 favours SQA-v3.

### QUBO-v3 architecture (per-camera window + prices): 20,000 cameras, utilisation 75%, seeds n=10

| Method | Objective mean ± sd | Gap to lower bound | Uncovered (total) | Time, s |
|---|---|---|---|---|
| greedy | 6,150.9 ± 85.5 | 4.35% | 0 | 1.4 |
| regret | 6,149.8 ± 86.0 | 4.33% | 0 | 9.8 |
| exact batch MILP | 6,149.8 ± 86.0 | 4.33% | 0 | 3.4 |
| ILS (equal time) | 6,150.3 ± 86.0 | 4.34% | 0 | 205.4 |
| SA-v3 | 6,151.4 ± 86.1 | 4.36% | 0 | 8.1 |
| SQA-v3 | 6,149.9 ± 86.0 | 4.33% | 0 | 412.7 |
| global solution (offline) | 5,911.7 ± 84.8 | 0.29% | 0 | 74.3 |

| SQA-v3 vs | mean diff | better / tie / worse | Wilcoxon p | Holm p | rank-biserial r |
|---|---|---|---|---|---|
| greedy | -1.09 | 7 / 0 / 3 | 0.084 | 0.336 | -0.64 |
| regret | +0.03 | 4 / 0 / 6 | 1 | 1 | +0.02 |
| exact batch MILP | +0.04 | 3 / 0 / 7 | 0.922 | 1 | +0.05 |
| ILS (equal time) | -0.42 | 5 / 0 / 5 | 0.557 | 1 | -0.24 |
| SA-v3 | -1.58 | 10 / 0 / 0 | 0.00195 | 0.00977 | -1.00 |

### QUBO-v3 architecture (per-camera window + prices): 20,000 cameras, utilisation 90%, seeds n=10

| Method | Objective mean ± sd | Gap to lower bound | Uncovered (total) | Time, s |
|---|---|---|---|---|
| greedy | 6,455.9 ± 119.6 | 8.61% | 0 | 1.6 |
| regret | 6,452.4 ± 119.7 | 8.55% | 0 | 11.8 |
| exact batch MILP | 6,452.4 ± 120.0 | 8.55% | 0 | 4.0 |
| ILS (equal time) | 6,454.2 ± 119.7 | 8.58% | 0 | 206.4 |
| SA-v3 | 6,455.7 ± 119.0 | 8.60% | 0 | 12.4 |
| SQA-v3 | 6,454.4 ± 119.1 | 8.58% | 0 | 496.9 |
| global solution (offline) | 5,977.8 ± 87.0 | 0.57% | 0 | 76.3 |

| SQA-v3 vs | mean diff | better / tie / worse | Wilcoxon p | Holm p | rank-biserial r |
|---|---|---|---|---|---|
| greedy | -1.59 | 6 / 0 / 4 | 0.322 | 0.645 | -0.38 |
| regret | +1.99 | 2 / 0 / 8 | 0.0645 | 0.322 | +0.67 |
| exact batch MILP | +1.97 | 2 / 0 / 8 | 0.131 | 0.523 | +0.56 |
| ILS (equal time) | +0.12 | 4 / 0 / 6 | 0.432 | 0.645 | +0.31 |
| SA-v3 | -1.32 | 6 / 0 / 4 | 0.193 | 0.58 | -0.49 |

### QUBO-v3 architecture (per-camera window + prices): 20,000 cameras, utilisation 95%, seeds n=10

| Method | Objective mean ± sd | Gap to lower bound | Uncovered (total) | Time, s |
|---|---|---|---|---|
| greedy | 6,685.1 ± 165.9 | 11.97% | 5 | 1.6 |
| regret | 6,680.0 ± 160.7 | 11.88% | 1 | 11.9 |
| exact batch MILP | 6,679.7 ± 164.8 | 11.88% | 1 | 4.1 |
| ILS (equal time) | 6,681.3 ± 162.6 | 11.90% | 0 | 205.8 |
| SA-v3 | 6,682.3 ± 160.1 | 11.92% | 2 | 15.8 |
| SQA-v3 | 6,679.0 ± 157.6 | 11.87% | 2 | 517.5 |
| global solution (offline) | 6,014.1 ± 90.4 | 0.74% | 0 | 83.4 |

| SQA-v3 vs | mean diff | better / tie / worse | Wilcoxon p | Holm p | rank-biserial r |
|---|---|---|---|---|---|
| greedy | -6.06 | 8 / 0 / 2 | 0.084 | 0.42 | -0.64 |
| regret | -1.05 | 5 / 0 / 5 | 0.846 | 1 | -0.09 |
| exact batch MILP | -0.72 | 3 / 0 / 7 | 0.492 | 1 | +0.27 |
| ILS (equal time) | -2.32 | 6 / 0 / 4 | 0.375 | 1 | -0.35 |
| SA-v3 | -3.31 | 8 / 0 / 2 | 0.084 | 0.42 | -0.64 |

### QUBO-v3 architecture (per-camera window + prices): 20,000 cameras, utilisation 98%, seeds n=10

| Method | Objective mean ± sd | Gap to lower bound | Uncovered (total) | Time, s |
|---|---|---|---|---|
| greedy | 7,119.8 ± 373.0 | 18.80% | 96 | 1.5 |
| regret | 7,019.7 ± 293.0 | 17.14% | 29 | 11.4 |
| exact batch MILP | 6,989.1 ± 267.3 | 16.64% | 7 | 4.6 |
| ILS (equal time) | 7,013.3 ± 279.2 | 17.04% | 21 | 205.6 |
| SA-v3 | 7,012.5 ± 272.1 | 17.03% | 16 | 21.5 |
| SQA-v3 | 7,032.5 ± 312.0 | 17.35% | 36 | 605.7 |
| global solution (offline) | 6,052.7 ± 93.0 | 1.03% | 0 | 84.4 |

| SQA-v3 vs | mean diff | better / tie / worse | Wilcoxon p | Holm p | rank-biserial r |
|---|---|---|---|---|---|
| greedy | -87.27 | 9 / 0 / 1 | 0.00391 | 0.0195 | -0.96 |
| regret | +12.78 | 5 / 0 / 5 | 0.695 | 1 | -0.16 |
| exact batch MILP | +43.45 | 1 / 0 / 9 | 0.0137 | 0.0547 | +0.85 |
| ILS (equal time) | +19.18 | 5 / 0 / 5 | 0.922 | 1 | +0.05 |
| SA-v3 | +20.04 | 6 / 0 / 4 | 1 | 1 | +0.02 |

## Shared 80×20 window, no prices (first-submission architecture)

| Utilisation | exact batch MILP | greedy | regret | gap of exact to lower bound |
|---|---|---|---|---|
| 75% | 11,913.2 | 11,963.3 | 11,918.5 | 102.1% |
| 90% | 11,547.2 | 11,621.9 | 11,582.4 | 94.3% |
| 95% | 11,341.5 | 11,483.3 | 11,321.7 | 90.0% |
| 98% | 11,187.7 | 11,390.8 | 11,205.5 | 86.8% |

### Scaling: 5,000 cameras, utilisation 95%, seeds n=5

| Method | Objective mean ± sd | Gap to lower bound | Uncovered (total) | Time, s |
|---|---|---|---|---|
| greedy | 1,839.9 ± 139.0 | 13.83% | 12 | 0.2 |
| regret | 1,810.9 ± 123.3 | 12.08% | 3 | 2.0 |
| exact batch MILP | 1,802.5 ± 109.7 | 11.59% | 0 | 0.7 |
| ILS (equal time) | 1,804.4 ± 109.0 | 11.71% | 0 | 51.3 |
| SA-v3 | 1,804.7 ± 104.2 | 11.72% | 2 | 6.8 |
| SQA-v3 | 1,806.4 ± 104.5 | 11.83% | 3 | 222.2 |
| global solution (offline) | 1,627.3 ± 85.8 | 0.75% | 0 | 3.4 |

| SQA-v3 vs | mean diff | better / tie / worse | Wilcoxon p | Holm p | rank-biserial r |
|---|---|---|---|---|---|
| greedy | -33.51 | 4 / 0 / 1 | 0.125 | 0.625 | -0.87 |
| regret | -4.46 | 2 / 0 / 3 | 1 | 1 | -0.07 |
| exact batch MILP | +3.87 | 2 / 0 / 3 | 1 | 1 | +0.07 |
| ILS (equal time) | +2.01 | 2 / 0 / 3 | 1 | 1 | +0.07 |
| SA-v3 | +1.68 | 2 / 0 / 3 | 0.625 | 1 | +0.33 |

### Scaling: 50,000 cameras, utilisation 95%, seeds n=3

| Method | Objective mean ± sd | Gap to lower bound | Uncovered (total) | Time, s |
|---|---|---|---|---|
| greedy | 16,319.8 ± 404.3 | 12.41% | 3 | 3.8 |
| regret | 16,296.2 ± 372.5 | 12.25% | 0 | 23.8 |
| exact batch MILP | 16,296.0 ± 372.3 | 12.25% | 0 | 8.1 |
| ILS (equal time) | 16,297.9 ± 370.7 | 12.26% | 0 | 511.3 |
| SA-v3 | 16,298.6 ± 373.4 | 12.27% | 0 | 20.8 |
| SQA-v3 | 16,298.2 ± 372.8 | 12.26% | 0 | 508.8 |
| global solution (offline) | 14,628.7 ± 202.9 | 0.77% | 0 | 504.9 |

| SQA-v3 vs | mean diff | better / tie / worse | Wilcoxon p | Holm p | rank-biserial r |
|---|---|---|---|---|---|
| greedy | -21.62 | 3 / 0 / 0 | 1 | 1 | -1.00 |
| regret | +1.99 | 0 / 0 / 3 | 1 | 1 | +1.00 |
| exact batch MILP | +2.20 | 0 / 0 / 3 | 1 | 1 | +1.00 |
| ILS (equal time) | +0.29 | 2 / 0 / 1 | 1 | 1 | +0.00 |
| SA-v3 | -0.42 | 2 / 0 / 1 | 1 | 1 | -0.67 |

## Ablation on hard batches (95% utilisation, per-camera window)

| Variant | batches | mean gap to batch optimum | median | optimal batches |
|---|---|---|---|---|
| exact | 80 | 0.000% | 0.000% | 80 |
| greedy | 80 | 0.789% | 0.666% | 0 |
| regret | 80 | 0.016% | 0.000% | 68 |
| regret_ls | 80 | 0.012% | 0.000% | 71 |
| base:SQA | 80 | 0.829% | 0.731% | 0 |
| base:SA | 80 | 1.619% | 1.555% | 0 |
| R:SQA | 80 | 0.185% | 0.125% | 8 |
| R:SA | 80 | 0.137% | 0.102% | 17 |
| DW:SQA | 80 | 0.554% | 0.498% | 0 |
| DW:SA | 80 | 0.325% | 0.252% | 1 |
| R+DW:SQA | 80 | 0.122% | 0.079% | 14 |
| R+DW:SA | 80 | 0.082% | 0.025% | 31 |
| R+DW:SQA [adaptive] | 80 | 0.063% | 0.031% | 24 |
| R+DW:SA [adaptive] | 80 | 0.031% | 0.000% | 42 |
| R+DW:SQA [beta20] | 80 | 0.437% | 0.002% | 38 |