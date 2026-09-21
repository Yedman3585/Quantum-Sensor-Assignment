# Article Revision Metric Summary

Generated from selected `summary_*.json` files. Raw QUBO energies are intentionally not compared across formulations.

## QUBO SQA Capacity-Stress Rows

| Util. | Method | Coverage | Objective | Assign. cost | Rejected | Time | QUBO terms | Conflict terms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 23.77% | AO-QUBO + SQA | 19.05% | 245,054.59 | 2,189.59 | 16,191 | 385.82s | 16,800 | 0 |
| 23.77% | PRC-QUBO + SQA | 99.60% | 12,852.25 | 11,652.25 | 0 | 406.16s | 16,800 | 0 |
| 23.77% | PRC-QUBO-C + SQA | 99.60% | 12,922.20 | 11,722.20 | 0 | 687.38s | 80,000 | 63,200 |
| 23.77% | PRC-QUBO-C-no-decoder + SQA | 99.60% | 12,890.60 | 11,690.60 | 0 | 697.13s | 80,000 | 63,200 |
| 23.77% | Static-QCP-QUBO + SQA | 19.30% | 244,324.36 | 2,239.36 | 16,139 | 398.32s | 80,000 | 0 |
| 47.53% | AO-QUBO + SQA | 8.91% | 274,223.43 | 938.43 | 18,219 | 389.65s | 16,800 | 0 |
| 47.53% | PRC-QUBO + SQA | 99.56% | 12,939.20 | 11,604.20 | 10 | 376.36s | 16,800 | 0 |
| 47.53% | PRC-QUBO-C + SQA | 99.60% | 13,052.14 | 11,852.14 | 0 | 378.54s | 80,000 | 63,200 |
| 47.53% | PRC-QUBO-C-no-decoder + SQA | 99.60% | 12,918.25 | 11,718.25 | 0 | 657.29s | 80,000 | 63,200 |
| 47.53% | Static-QCP-QUBO + SQA | 8.84% | 274,421.21 | 941.21 | 18,232 | 399.91s | 80,000 | 0 |
| 72.02% | AO-QUBO + SQA | 5.64% | 283,661.86 | 581.86 | 18,872 | 129.81s | 16,800 | 0 |
| 72.02% | PRC-QUBO + SQA | 98.69% | 15,659.36 | 11,714.36 | 633 | 403.70s | 16,800 | 0 |
| 72.02% | PRC-QUBO-C + SQA | 99.60% | 13,259.87 | 12,059.87 | 0 | 448.10s | 80,000 | 63,200 |
| 72.02% | PRC-QUBO-C-no-decoder + SQA | 99.59% | 13,172.53 | 11,942.53 | 23 | 930.91s | 80,000 | 63,200 |
| 72.02% | Static-QCP-QUBO + SQA | 5.42% | 284,312.32 | 557.32 | 18,917 | 391.66s | 80,000 | 0 |
| 95.07% | AO-QUBO + SQA | 4.18% | 287,890.04 | 430.04 | 19,164 | 123.37s | 16,800 | 0 |
| 95.07% | PRC-QUBO + SQA | 94.44% | 27,388.14 | 10,693.14 | 2,666 | 397.48s | 16,800 | 0 |
| 95.07% | PRC-QUBO-C + SQA | 99.60% | 12,863.76 | 11,663.76 | 0 | 506.19s | 80,000 | 63,200 |
| 95.07% | PRC-QUBO-C-no-decoder + SQA | 99.70% | 12,290.09 | 11,390.09 | 525 | 1,483.72s | 80,000 | 63,200 |
| 95.07% | Static-QCP-QUBO + SQA | 4.15% | 287,987.06 | 437.06 | 19,170 | 392.31s | 80,000 | 0 |

## Best Classical Baseline Per Utilization

| Util. | Method | Coverage | Objective | Assign. cost | Rejected | Time |
| --- | --- | --- | --- | --- | --- | --- |
| 23.77% | Regret-Best-Fit-20 + Regret Best-Fit Assignment | 99.60% | 13,193.74 | 11,993.74 | 0 | 24.83s |
| 47.53% | Regret-Best-Fit-20 + Regret Best-Fit Assignment | 99.60% | 13,111.88 | 11,911.88 | 0 | 25.53s |
| 72.02% | Regret-Best-Fit-20 + Regret Best-Fit Assignment | 99.60% | 13,444.36 | 12,244.36 | 0 | 45.83s |
| 95.07% | Regret-Best-Fit-20 + Regret Best-Fit Assignment | 99.60% | 13,062.92 | 11,862.92 | 0 | 75.08s |

## PRC-QUBO-C Ablation At 95.07% Utilization

| Util. | Method | Coverage | Objective | Assign. cost | Rejected | Time | QUBO terms | Conflict terms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 95.07% | PRC-QUBO + SQA | 94.44% | 27,388.14 | 10,693.14 | 2,666 | 397.48s | 16,800 | 0 |
| 95.07% | PRC-QUBO-C + SQA | 99.60% | 12,863.76 | 11,663.76 | 0 | 506.19s | 80,000 | 63,200 |
| 95.07% | PRC-QUBO-C-no-decoder + SQA | 99.70% | 12,290.09 | 11,390.09 | 525 | 1,483.72s | 80,000 | 63,200 |
| 95.07% | PRC-QUBO-D + SQA | 99.60% | 12,779.62 | 11,579.62 | 0 | 108.57s | 16,800 | 0 |

## Diagnostic Ablation At 95.07% Utilization

These rows are implementation diagnostics and must not be presented as OpenJij SQA results.

| Util. | Method | Coverage | Objective | Assign. cost | Rejected | Time | QUBO terms | Conflict terms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Notes

- `PRC-QUBO-C` is a stress-aware extension, not a silent replacement for `PRC-QUBO`.
- `PRC-QUBO-D` denotes decoder-only ablation: PRC linear model plus capacity-aware decoder, without conflict couplings.
- `PRC-QUBO-C-no-decoder` denotes conflict-coupled QUBO decoded with the original PRC decoder.
- Markdown tables select the latest preferred summary per method/utilization. The CSV file retains every raw row for auditability.
- The strongest currently logged classical baseline is usually `Regret-Best-Fit-20`; it is fast and feasible, so the paper should emphasize quality/runtime trade-off.
