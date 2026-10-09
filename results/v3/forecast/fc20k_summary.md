# Forecast experiment, 20,000 x 800, 10 seeds (42-51), exact batch solver

Mean gap to Lagrangian LB (%); [uncovered, sum]; Δ vs oracle in pp (wins/losses vs oracle; Holm p within column)

| information for re-pricing | 90% | 95% | 98% |
|---|---|---|---|
| static prices | 8.55 (+7.52; 0/10; p=0.012) | 11.85 (+10.21; 0/10; p=0.012) | 16.48 (+13.13; 0/10; p=0.012) |
| oracle | 1.03 | 1.64 | 3.35 |
| noise ±30% | 1.19 (+0.16; 1/9; p=0.020) | 1.84 (+0.20; 3/7; p=0.129) | 3.63 (+0.28; 3/7; p=0.168) |
| bias -10% | 5.22 (+4.19; 0/10; p=0.012) | 8.21 (+6.56; 0/10; p=0.012) | 12.94 (+9.59; 0/10; p=0.012) |
| bias +10% | 0.99 (-0.05; 7/3; p=0.084) | 1.50 (-0.14; 8/2; p=0.275) | 1.36 (-1.99; 10/0; p=0.012) |
| twin | 2.56 (+1.53; 0/10; p=0.012) | 3.40 (+1.76; 0/10; p=0.012) | 4.93 (+1.58; 0/10; p=0.012) |
| twin +10% | 1.46 (+0.43; 0/10; p=0.012) | 2.20 (+0.56; 2/8; p=0.029) | 2.68 (-0.67; 7/3; p=0.168) |

## SQA-v3.1+LS with the twin +10% forecast (95/98%)

| util | SQA+LS twin+10% | exact twin+10% | SQA+LS oracle (campaign 3) | SQA+LS twin+10% vs exact twin+10% (W/L, p) | time SQA+LS (s) |
|---|---|---|---|---|---|
| 95% | 2.18 | 2.20 | 1.63 | 9/1, p=0.064 | 338 |
| 98% | 2.65 | 2.68 | 3.31 | 8/2, p=0.049 | 389 |

Run time (s, mean, 4 parallel jobs): static prices 6, oracle 109, noise ±30% 143, bias -10% 142, bias +10% 142, twin 141, twin +10% 141
