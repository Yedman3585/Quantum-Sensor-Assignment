# Priority weight w_i = p_i: full robustness study (20,000 x 800)

Gap to the Lagrangian lower bound (%) computed with w_i = p_i; [uncovered cameras, sum over seeds];
in brackets: mean objective difference to the reference row, wins/losses of the row, Wilcoxon p (Holm within column).

## 1. Main table (test seeds 42–51)

Reference for the margin rows: the same information without margin.

| Policy | Solver | 75% | 90% | 95% | 98% |
|---|---|---|---|---|---|
| static | exact | 1.52 | 2.65 | 3.79 | 5.96 |
| re-pricing | exact | 0.42 | 0.95 | 1.52 | 2.78 |
| re-pricing +5% | exact | 0.40 (-0.8; 9/1; p=0.012) | 0.81 (-6.1; 10/0; p=0.008) | 1.08 (-18.6; 10/0; p=0.008) | 1.37 (-60.1; 10/0; p=0.008) |
| statistics | exact | 0.94 | 1.69 | 2.23 | 3.39 |
| statistics +5% | exact | 0.88 (-2.3; 9/1; p=0.012) | 1.46 (-9.8; 9/1; p=0.012) | 1.86 (-15.6; 7/3; p=0.039) | 2.26 (-48.3; 10/0; p=0.008) |
| static | greedy | 1.53 | 2.68 | 3.85 | 6.09 |
| re-pricing | greedy | 0.43 | 0.97 | 1.56 | 2.86 |
| re-pricing +5% | greedy | 0.40 (-1.1; 10/0; p=0.008) | 0.81 (-6.7; 10/0; p=0.008) | 1.10 (-19.6; 10/0; p=0.008) | 1.38 (-63.0; 10/0; p=0.008) |
| statistics +5% | greedy | 0.88 (-2.4; 9/1; p=0.012) | 1.48 (-9.1; 8/2; p=0.020) | 1.88 (-17.3; 8/2; p=0.039) | 2.35 (-47.4; 10/0; p=0.008) |

## 2. Margin tuning (validation seeds 100–109, exact batches)

| Information, utilisation | +0% | +5% | +10% | +15% | +20% | +30% | best |
|---|---|---|---|---|---|---|---|
| exact demand, 90% | 0.86 | **0.80** | 1.23 | 1.71 | 2.05 | 2.37 | +5% |
| exact demand, 95% | 1.59 | **1.06** | 1.66 | 2.07 | 2.28 | 2.76 | +5% |
| exact demand, 98% | 2.97 | **1.51** | 1.64 | 1.96 | 2.20 | 3.03 | +5% |
| demand statistics, 90% | 1.52 | **1.43** | 1.69 | 2.07 | 2.37 | 2.67 | +5% |
| demand statistics, 95% | 2.33 | **1.85** | 2.20 | 2.52 | 2.73 | 3.17 | +5% |
| demand statistics, 98% | 3.60 | 2.48 | **2.38** | 2.60 | 2.84 | 3.55 | +10% |

## 3. Batch solvers inside the final policy

Reference: exact batch MILP with the same information and margin.

| Policy | Solver | 90% | 95% | 98% |
|---|---|---|---|---|
| re-pricing +5% | exact | 0.81 | 1.08 | 1.37 |
| re-pricing +5% | greedy | 0.81 (+0.3; 3/7; p=0.551) | 1.10 (+0.6; 1/9; p=0.041) | 1.38 (+0.4; 4/6; p=0.922) |
| re-pricing +5% | sa_v3 | 0.83 (+0.9; 0/10; p=0.008) | 1.12 (+1.8; 0/10; p=0.010) | 1.39 (+0.8; 4/6; p=0.422) |
| re-pricing +5% | sqa_v3 | 0.84 (+1.3; 0/10; p=0.008) | 1.12 (+1.8; 0/10; p=0.010) | 1.42 (+2.0; 0/10; p=0.010) |
| re-pricing +5% | sqa_v3+LS | 0.81 (+0.1; 4/6; p=0.557) | 1.09 (+0.2; 3/7; p=0.465) | 1.36 (-0.7; 6/4; p=0.551) |
| statistics +5% | exact | 1.46 | 1.86 | 2.26 |
| statistics +5% | sqa_v3+LS | – | 1.86 (-0.0; 5/5; p=0.922) | 2.28 (+0.6; 3/7; p=0.422) |

Mean time per run at 95% (s; exact/greedy on the Mac, SA/SQA on ASUS, not comparable across machines): exact 41, greedy 41, sa_v3 130, sqa_v3+LS 566, sqa_v3 474

## 4. Online-allocation baselines

Reference: re-pricing + 5% (Lagrangian). lp = exact LP bid prices; dmd = dual mirror descent (eta 0.1, Lagrangian start); lookahead = 2-batch rolling horizon.

| Policy | Solver | 90% | 95% | 98% |
|---|---|---|---|---|
| re-pricing +5% | exact | 0.81 | 1.08 | 1.37 |
| re-pricing | exact | 0.95 | 1.52 | 2.78 |
| lp | exact | 0.63 (-7.3; 10/0; p=0.008) | 0.93 (-6.1; 9/1; p=0.027) | 1.50 (+5.5; 3/7; p=0.049) |
| lp+5% | exact | 0.67 (-5.8; 8/2; p=0.084) | 1.42 (+14.6; 3/7; p=0.131) | 6.95 [182 unc.] (+239.0; 2/8; p=0.020) |
| dmd | exact | 7.21 (+265.6; 0/10; p=0.008) | 8.37 (+306.7; 0/10; p=0.008) | 9.68 (+353.8; 0/10; p=0.008) |
| lookahead | exact | 10.52 (+402.9; 0/10; p=0.008) | 13.86 (+538.0; 0/10; p=0.008) | 18.74 [33 unc.] (+740.3; 0/10; p=0.008) |
| static | exact | 2.65 | 3.79 | 5.96 |

Mean time per run at 95% (s): dmd 55, lookahead 17, lp 1564, lp+5% 1699, re-pricing 129, re-pricing +5% 41, static 6, statistics 128, statistics +5% 41

## 5. Clustered urban layout

Reference: clustered re-pricing without margin.

| Policy | Solver | 90% | 95% | 98% |
|---|---|---|---|---|
| static | exact | 3.62 (+100.7; 0/10; p=0.006) | 5.05 (+122.7; 0/10; p=0.006) | 8.26 (+170.8; 0/10; p=0.006) |
| re-pricing | exact | 1.26 | 2.23 | 4.40 |
| re-pricing +5% | exact | 1.24 (-0.6; 2/8; p=0.232) | 1.43 (-35.1; 9/1; p=0.012) | 2.52 (-83.3; 10/0; p=0.006) |
| statistics +5% | exact | 1.76 (+21.5; 1/9; p=0.055) | 2.10 (-6.0; 5/5; p=0.922) | 3.11 (-57.6; 9/1; p=0.014) |

## 6. Encoding ablation on hard batches (95%, re-pricing, priced batch objective)

59 hard batches (greedy >= 0.5% above the batch optimum). Mean over seeds of the per-seed mean gap to the batch optimum.

| Variant | SQA | SA |
|---|---|---|
| one-hot | 3.77% | 5.46% |
| + reduction | 2.23% | 2.90% |
| + domain wall | 3.46% | 3.92% |
| R + DW | 2.13% | 2.21% |
| R + DW + adaptive | 1.82% | 1.99% |
| greedy / regret / regret+LS | 2.03% / 0.68% / 0.34% | |
