# Priority-weight check: w_i = p_i (instead of 4 - p_i), 20,000 x 800, seeds 42-51

Duplicate runs (PC vs Mac) compared: 36, max |objective difference| = 9.09e-13

Mean gap to the Lagrangian LB computed with w_i = p_i (%); [uncovered]; n = seeds

## exact

| re-pricing | 90% | 95% | 98% |
|---|---|---|---|
| static prices | 2.65 | 3.79 | 5.96 |
| oracle | 0.95 | 1.52 | 2.78 |
| twin | 1.69 | 2.23 | 3.39 |

## greedy

| re-pricing | 90% | 95% | 98% |
|---|---|---|---|
| static prices | 2.68 | 3.85 | 6.09 |
| oracle | 0.97 | 1.56 | 2.86 |
| twin | 1.70 | 2.28 | 3.46 |

