
### plain, capacity scale 1.0 (utilisation ~24.38%), seeds n=10

| Method | Objective mean ± sd | Gap to global LB, mean | Uncovered, total | Time, s mean |
|---|---|---|---|---|
| greedy | 5,853.7 ± 86.0 | 0.37% | 0 | 0.6 |
| regret | 5,853.6 ± 86.0 | 0.37% | 0 | 4.0 |
| exact | 5,853.6 ± 86.0 | 0.37% | 0 | 1.6 |
| sa_v3 | 5,853.6 ± 86.0 | 0.37% | 0 | 0.8 |
| sqa_v3 | 5,853.6 ± 86.0 | 0.37% | 0 | 4.4 |
| global_feasible | 5,831.9 ± 84.6 | 0.00% | 0 | 36.2 |

| SQA-v3 vs | mean diff (SQA − other) | SQA better / tie / worse | Wilcoxon p |
|---|---|---|---|
| greedy | -0.07 | 8 / 2 / 0 | 0.00781 |
| regret | +0.00 | 0 / 8 / 2 | 0.5 |
| exact | +0.00 | 0 / 8 / 2 | 0.5 |
| sa_v3 | +0.00 | 0 / 8 / 2 | 0.5 |

### plain, capacity scale 0.33 (utilisation ~73.89%), seeds n=10

| Method | Objective mean ± sd | Gap to global LB, mean | Uncovered, total | Time, s mean |
|---|---|---|---|---|
| greedy | 6,758.2 ± 105.8 | 14.71% | 0 | 0.5 |
| regret | 6,755.5 ± 105.5 | 14.66% | 0 | 3.3 |
| exact | 6,755.5 ± 105.5 | 14.66% | 0 | 1.4 |
| sa_v3 | 6,755.8 ± 105.3 | 14.67% | 0 | 3.7 |
| sqa_v3 | 6,755.8 ± 105.4 | 14.67% | 0 | 132.7 |
| global_feasible | 5,908.1 ± 85.2 | 0.27% | 0 | 30.2 |

| SQA-v3 vs | mean diff (SQA − other) | SQA better / tie / worse | Wilcoxon p |
|---|---|---|---|
| greedy | -2.35 | 9 / 0 / 1 | 0.00391 |
| regret | +0.32 | 3 / 0 / 7 | 0.193 |
| exact | +0.32 | 3 / 0 / 7 | 0.193 |
| sa_v3 | +0.04 | 6 / 0 / 4 | 0.922 |

### plain, capacity scale 0.25 (utilisation ~97.22%), seeds n=9; excluded (utilisation >= 100%, infeasible): [43]

| Method | Objective mean ± sd | Gap to global LB, mean | Uncovered, total | Time, s mean |
|---|---|---|---|---|
| greedy | 8,923.3 ± 972.4 | 48.85% | 206 | 0.6 |
| regret | 8,730.4 ± 791.8 | 45.63% | 94 | 4.2 |
| exact | 8,640.6 ± 702.8 | 44.15% | 35 | 2.5 |
| sa_v3 | 8,670.9 ± 737.5 | 44.65% | 54 | 18.6 |
| sqa_v3 | 8,714.8 ± 769.3 | 45.38% | 83 | 311.9 |
| global_feasible | 6,180.1 ± 423.9 | 3.15% | 82 | 32.2 |

| SQA-v3 vs | mean diff (SQA − other) | SQA better / tie / worse | Wilcoxon p |
|---|---|---|---|
| greedy | -208.53 | 9 / 0 / 0 | 0.00391 |
| regret | -15.66 | 4 / 0 / 5 | 0.91 |
| exact | +74.15 | 0 / 0 / 9 | 0.00391 |
| sa_v3 | +43.87 | 3 / 0 / 6 | 0.129 |

### priced, capacity scale 1.0 (utilisation ~24.38%), seeds n=10

| Method | Objective mean ± sd | Gap to global LB, mean | Uncovered, total | Time, s mean |
|---|---|---|---|---|
| greedy | 5,833.1 ± 84.5 | 0.02% | 0 | 0.5 |
| regret | 5,833.1 ± 84.5 | 0.02% | 0 | 3.5 |
| exact | 5,833.1 ± 84.5 | 0.02% | 0 | 1.5 |
| sa_v3 | 5,833.1 ± 84.5 | 0.02% | 0 | 0.7 |
| sqa_v3 | 5,833.1 ± 84.5 | 0.02% | 0 | 3.1 |
| global_feasible | 5,831.9 ± 84.6 | 0.00% | 0 | 36.2 |

| SQA-v3 vs | mean diff (SQA − other) | SQA better / tie / worse | Wilcoxon p |
|---|---|---|---|
| greedy | -0.05 | 7 / 3 / 0 | 0.0156 |
| regret | +0.01 | 0 / 8 / 2 | 0.5 |
| exact | +0.01 | 0 / 8 / 2 | 0.5 |
| sa_v3 | +0.00 | 0 / 10 / 0 | nan |

### priced, capacity scale 0.33 (utilisation ~73.89%), seeds n=10

| Method | Objective mean ± sd | Gap to global LB, mean | Uncovered, total | Time, s mean |
|---|---|---|---|---|
| greedy | 6,142.1 ± 92.1 | 4.25% | 0 | 0.7 |
| regret | 6,140.5 ± 91.9 | 4.22% | 0 | 4.3 |
| exact | 6,140.5 ± 91.9 | 4.22% | 0 | 1.8 |
| sa_v3 | 6,141.7 ± 91.2 | 4.24% | 0 | 3.8 |
| sqa_v3 | 6,141.4 ± 91.8 | 4.23% | 0 | 87.4 |
| global_feasible | 5,908.1 ± 85.2 | 0.27% | 0 | 30.2 |

| SQA-v3 vs | mean diff (SQA − other) | SQA better / tie / worse | Wilcoxon p |
|---|---|---|---|
| greedy | -0.69 | 7 / 0 / 3 | 0.193 |
| regret | +0.90 | 2 / 0 / 8 | 0.0645 |
| exact | +0.92 | 3 / 0 / 7 | 0.084 |
| sa_v3 | -0.26 | 7 / 0 / 3 | 0.492 |

### priced, capacity scale 0.25 (utilisation ~97.22%), seeds n=9; excluded (utilisation >= 100%, infeasible): [43]

| Method | Objective mean ± sd | Gap to global LB, mean | Uncovered, total | Time, s mean |
|---|---|---|---|---|
| greedy | 7,164.0 ± 607.6 | 19.54% | 134 | 0.6 |
| regret | 7,084.7 ± 515.5 | 18.22% | 85 | 4.4 |
| exact | 7,051.6 ± 504.4 | 17.67% | 64 | 2.3 |
| sa_v3 | 7,044.8 ± 472.5 | 17.56% | 59 | 12.8 |
| sqa_v3 | 7,058.0 ± 485.6 | 17.78% | 73 | 198.2 |
| global_feasible | 6,180.1 ± 423.9 | 3.15% | 82 | 32.2 |

| SQA-v3 vs | mean diff (SQA − other) | SQA better / tie / worse | Wilcoxon p |
|---|---|---|---|
| greedy | -106.03 | 7 / 0 / 2 | 0.0273 |
| regret | -26.65 | 6 / 0 / 3 | 0.0742 |
| exact | +6.43 | 4 / 0 / 5 | 1 |
| sa_v3 | +13.20 | 6 / 0 / 3 | 0.82 |