# Service metrics (uniform layout, 20k x 800, exact batches; means over seeds)

Gap to offline = price of sequentiality. Class contribution: 100 * (class cost - offline class cost) / offline objective;
the three classes add up to the gap to offline. Distances in plane units (1000 x 1000 plane).


## Weight w = 4 - p (default), utilisation 95%

| Policy | seeds | Gap to LB, % | Gap to offline, % | Servers >= 99% full, % | CV of utilisation | Half capacity saturated at batch |
|---|---|---|---|---|---|---|
| Static prices | 5 | 12.72 | 11.86 | 80.0 | 0.270 | 180 |
| Re-pricing (exact info) | 5 | 1.79 | 1.01 | 79.1 | 0.275 | 239 |
| Re-pricing + 5% (exact info) | 5 | 0.95 | 0.17 | 23.2 | 0.158 | never |
| Re-pricing + 5% (statistics only) | 5 | 1.91 | 1.13 | 45.0 | 0.166 | 247 |
| Offline solution | 5 | 0.77 | 0.00 | 46.2 | 0.242 | never |

| Policy | Class | Coverage, % | Mean distance | Mean cost c_ij | On cheapest server, % | Mean server rank | Contribution to gap to offline, pp |
|---|---|---|---|---|---|---|---|
| Static prices | high | 100.00 | 41.9 | 0.1861 | 7.9 | 11.33 | -0.34 |
| Static prices | medium | 100.00 | 30.5 | 0.1128 | 57.9 | 1.86 | -0.44 |
| Static prices | low | 100.00 | 49.1 | 0.1415 | 39.8 | 10.86 | +12.64 |
| Re-pricing (exact info) | high | 100.00 | 45.9 | 0.1910 | 5.1 | 14.69 | -0.10 |
| Re-pricing (exact info) | medium | 100.00 | 28.6 | 0.1148 | 48.4 | 2.30 | -0.12 |
| Re-pricing (exact info) | low | 100.00 | 32.2 | 0.1223 | 83.6 | 2.63 | +1.22 |
| Re-pricing + 5% (exact info) | high | 100.00 | 60.6 | 0.2004 | 4.0 | 25.71 | +0.36 |
| Re-pricing + 5% (exact info) | medium | 100.00 | 28.2 | 0.1150 | 46.4 | 2.28 | -0.08 |
| Re-pricing + 5% (exact info) | low | 100.00 | 30.3 | 0.1201 | 90.6 | 1.11 | -0.10 |
| Re-pricing + 5% (statistics only) | high | 100.00 | 58.8 | 0.1995 | 3.7 | 24.00 | +0.32 |
| Re-pricing + 5% (statistics only) | medium | 100.00 | 30.4 | 0.1162 | 43.1 | 2.65 | +0.11 |
| Re-pricing + 5% (statistics only) | low | 100.00 | 32.2 | 0.1215 | 76.8 | 1.46 | +0.70 |
| Offline solution | high | 100.00 | 48.9 | 0.1930 | 4.9 | 16.92 | — |
| Offline solution | medium | 100.00 | 29.5 | 0.1155 | 50.8 | 2.81 | — |
| Offline solution | low | 100.00 | 30.2 | 0.1203 | 90.0 | 1.17 | — |

## Weight w = 4 - p (default), utilisation 98%

| Policy | seeds | Gap to LB, % | Gap to offline, % | Servers >= 99% full, % | CV of utilisation | Half capacity saturated at batch |
|---|---|---|---|---|---|---|
| Static prices | 5 | 18.14 | 16.87 | 86.4 | 0.164 | 174 |
| Re-pricing (exact info) | 5 | 4.02 | 2.90 | 88.3 | 0.164 | 235 |
| Re-pricing + 5% (exact info) | 5 | 1.30 | 0.21 | 38.1 | 0.024 | never |
| Re-pricing + 5% (statistics only) | 5 | 2.64 | 1.54 | 60.8 | 0.041 | 243 |
| Offline solution | 5 | 1.09 | 0.00 | 52.3 | 0.108 | never |

| Policy | Class | Coverage, % | Mean distance | Mean cost c_ij | On cheapest server, % | Mean server rank | Contribution to gap to offline, pp |
|---|---|---|---|---|---|---|---|
| Static prices | high | 100.00 | 48.7 | 0.1901 | 6.5 | 14.74 | -0.57 |
| Static prices | medium | 100.00 | 30.9 | 0.1133 | 55.9 | 1.95 | -0.59 |
| Static prices | low | 99.99 | 61.8 | 0.1504 | 37.1 | 21.24 | +18.03 |
| Re-pricing (exact info) | high | 100.00 | 56.1 | 0.1974 | 4.1 | 21.29 | -0.22 |
| Re-pricing (exact info) | medium | 100.00 | 28.8 | 0.1155 | 45.3 | 2.50 | -0.22 |
| Re-pricing (exact info) | low | 100.00 | 38.2 | 0.1260 | 81.1 | 8.04 | +3.34 |
| Re-pricing + 5% (exact info) | high | 100.00 | 79.1 | 0.2105 | 3.0 | 41.24 | +0.42 |
| Re-pricing + 5% (exact info) | medium | 100.00 | 28.3 | 0.1159 | 43.8 | 2.54 | -0.16 |
| Re-pricing + 5% (exact info) | low | 100.00 | 30.3 | 0.1202 | 87.4 | 1.15 | -0.05 |
| Re-pricing + 5% (statistics only) | high | 100.00 | 77.2 | 0.2096 | 3.0 | 39.28 | +0.38 |
| Re-pricing + 5% (statistics only) | medium | 100.00 | 32.2 | 0.1177 | 39.2 | 3.11 | +0.13 |
| Re-pricing + 5% (statistics only) | low | 100.00 | 33.1 | 0.1221 | 72.9 | 1.67 | +1.03 |
| Offline solution | high | 100.00 | 63.4 | 0.2018 | 4.0 | 27.64 | — |
| Offline solution | medium | 100.00 | 30.7 | 0.1169 | 48.2 | 4.00 | — |
| Offline solution | low | 100.00 | 30.2 | 0.1203 | 89.5 | 1.19 | — |

## Weight w = p, utilisation 95%

| Policy | seeds | Gap to LB, % | Gap to offline, % | Servers >= 99% full, % | CV of utilisation | Half capacity saturated at batch |
|---|---|---|---|---|---|---|
| Static prices | 5 | 4.13 | 2.97 | 81.0 | 0.278 | 194 |
| Re-pricing (exact info) | 5 | 1.67 | 0.53 | 84.4 | 0.283 | 236 |
| Re-pricing + 5% (exact info) | 5 | 1.08 | -0.05 | 50.0 | 0.224 | 247 |
| Offline solution | 5 | 1.13 | 0.00 | 62.5 | 0.264 | never |

| Policy | Class | Coverage, % | Mean distance | Mean cost c_ij | On cheapest server, % | Mean server rank | Contribution to gap to offline, pp |
|---|---|---|---|---|---|---|---|
| Static prices | high | 100.00 | 31.7 | 0.1760 | 25.7 | 7.13 | -0.53 |
| Static prices | medium | 100.00 | 28.6 | 0.1162 | 47.6 | 2.98 | -0.67 |
| Static prices | low | 100.00 | 46.2 | 0.1387 | 35.6 | 10.94 | +4.16 |
| Re-pricing (exact info) | high | 100.00 | 32.2 | 0.1773 | 24.0 | 7.81 | -0.26 |
| Re-pricing (exact info) | medium | 100.00 | 29.0 | 0.1179 | 41.7 | 3.46 | -0.27 |
| Re-pricing (exact info) | low | 100.00 | 35.2 | 0.1277 | 58.1 | 5.70 | +1.06 |
| Re-pricing + 5% (exact info) | high | 100.00 | 35.8 | 0.1794 | 23.8 | 10.08 | +0.18 |
| Re-pricing + 5% (exact info) | medium | 100.00 | 29.3 | 0.1188 | 40.3 | 3.76 | -0.07 |
| Re-pricing + 5% (exact info) | low | 100.00 | 29.1 | 0.1233 | 63.6 | 1.92 | -0.16 |
| Offline solution | high | 100.00 | 34.8 | 0.1786 | 24.5 | 9.19 | — |
| Offline solution | medium | 100.00 | 30.3 | 0.1191 | 43.6 | 4.69 | — |
| Offline solution | low | 100.00 | 29.8 | 0.1239 | 65.2 | 2.47 | — |

## Weight w = p, utilisation 98%

| Policy | seeds | Gap to LB, % | Gap to offline, % | Servers >= 99% full, % | CV of utilisation | Half capacity saturated at batch |
|---|---|---|---|---|---|---|
| Static prices | 5 | 10.40 | 8.31 | 87.1 | 0.169 | 190 |
| Re-pricing (exact info) | 5 | 3.35 | 1.40 | 89.5 | 0.169 | 233 |
| Re-pricing + 5% (exact info) | 5 | 1.51 | -0.41 | 71.2 | 0.121 | 243 |
| Offline solution | 5 | 1.92 | 0.00 | 73.4 | 0.156 | never |

| Policy | Class | Coverage, % | Mean distance | Mean cost c_ij | On cheapest server, % | Mean server rank | Contribution to gap to offline, pp |
|---|---|---|---|---|---|---|---|
| Static prices | high | 100.00 | 34.8 | 0.1788 | 23.3 | 9.01 | -0.95 |
| Static prices | medium | 100.00 | 29.4 | 0.1169 | 45.6 | 3.18 | -1.10 |
| Static prices | low | 99.91 | 65.8 | 0.1504 | 33.6 | 27.91 | +10.36 |
| Re-pricing (exact info) | high | 100.00 | 35.7 | 0.1803 | 21.7 | 10.07 | -0.65 |
| Re-pricing (exact info) | medium | 100.00 | 30.5 | 0.1191 | 39.1 | 3.87 | -0.59 |
| Re-pricing (exact info) | low | 100.00 | 47.7 | 0.1350 | 49.9 | 14.98 | +2.64 |
| Re-pricing + 5% (exact info) | high | 100.00 | 39.8 | 0.1823 | 22.0 | 12.62 | -0.24 |
| Re-pricing + 5% (exact info) | medium | 100.00 | 31.1 | 0.1203 | 37.1 | 4.36 | -0.30 |
| Re-pricing + 5% (exact info) | low | 100.00 | 32.9 | 0.1259 | 56.4 | 3.91 | +0.14 |
| Offline solution | high | 100.00 | 42.0 | 0.1834 | 22.0 | 14.35 | — |
| Offline solution | medium | 100.00 | 34.1 | 0.1217 | 41.5 | 7.63 | — |
| Offline solution | low | 100.00 | 32.2 | 0.1254 | 63.8 | 4.28 | — |
