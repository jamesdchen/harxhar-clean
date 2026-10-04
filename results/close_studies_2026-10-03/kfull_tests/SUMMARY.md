# Tests: LightGBM on the last k bars ending 16:00 against the linear 16:00 model (close study, 2026-10-04)

Written by `experiments/close_kfull_tests.py` from the CSVs in this folder (2026-10-04 07:14:35); every number below is in them.

## Forecasts and set
- Set: **existing**. Manifest `results/close_studies_2026-10-03/trees_kfull/manifest.csv`: 68 rows, complete: False.
- Family `nobar` (LightGBM, last k bars, no bar column): k = 1: seeds 42/43; k = 2: seeds 42/43; k = 3: seeds 42/43; k = 4: seeds 42/43; k = 5: seeds 42; k = 7: seeds 42; k = 13: seeds 42/43  (primary)
- This is the set that existed when the script ran: k = 6 and 8 .. 12, the bar-column arms for k = 2 .. 12 and seed 44 were not there yet. For each k the tested forecast is the average of the seeds listed (one seed for some k).
- Linear baseline: `ridge_bb0` (ridge, HAR + calendar unpenalized; QLIKE 0.0946); secondary `lasso_bb0`, `ridge_single`, `lasso_single`.
- Scorer: research convention (16:00-bar recalibration (f^2 + s) B from each forecast's own errors), 866 trade days 2020-01-03 .. 2024-04-30, one loss a day. For each k, the seed forecasts are averaged on the adjusted scale and the average is scored. Losses: QLIKE (primary), MSE on the variance level. d = L_linear - L_tree, positive = the tree has lower loss.
- Gate: trees_morebars/average.csv reproduced, 20 / 20 quantities (QLIKE, Sharpe mid, DM against the ridge) within 1e-12 (max |difference| 4.9e-15); e.g. QLIKE lgbm 4 bars 0.0974, ridge_bb0 0.0946, equal weights ridge + 4-bar pool 0.0914.
- Resampling: the repo's circular block bootstrap helper is switched off (commit b761b28: `atm_straddle_lib.circular_block_bootstrap_idx` returns the original order; `model_confidence_set._boot_col_means` likewise returns the sample in every replicate). SPA, Reality Check and the MCS here use this script's own stationary bootstrap (Politis-Romano), B = 10000, mean block 10 (also 5, 21 in the CSV); the MCS keeps the module's elimination rule with these replicate means.

## 1. Diebold-Mariano against the ridge (HLN-corrected), QLIKE
DM statistic (positive = tree lower loss) at Newey-West lag 0 / 5 / 6 (dm_test automatic) / 10 / 14 (newey_west_lag) / 21; one-sided normal p (H1: tree lower loss) and fixed-b one-sided p (Bartlett, b = (lag+1)/T, simulated) at the automatic lag.

| k | seeds | QLIKE tree | QLIKE ridge | mean d | DM lag 0 | DM lag 5 | DM lag 6 | DM lag 10 | DM lag 14 | DM lag 21 | p one (auto) | p two (auto) | fixed-b p one (auto) | fixed-b p two (auto) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | mean of 42/43 | 0.1000 | 0.0946 | -0.0054 | -1.86 | -1.68 | -1.62 | -1.58 | -1.64 | -1.81 | 0.948 | 0.105 | 0.947 | 0.107 |
| 2 | mean of 42/43 | 0.0982 | 0.0946 | -0.0036 | -1.17 | -1.18 | -1.15 | -1.12 | -1.15 | -1.28 | 0.876 | 0.248 | 0.875 | 0.252 |
| 3 | mean of 42/43 | 0.0976 | 0.0946 | -0.0030 | -0.81 | -0.83 | -0.82 | -0.79 | -0.80 | -0.86 | 0.793 | 0.414 | 0.794 | 0.417 |
| 4 | mean of 42/43 | 0.0976 | 0.0946 | -0.0030 | -0.86 | -0.86 | -0.85 | -0.82 | -0.84 | -0.91 | 0.802 | 0.396 | 0.802 | 0.398 |
| 5 | mean of 42 | 0.0984 | 0.0946 | -0.0038 | -1.08 | -1.08 | -1.07 | -1.04 | -1.06 | -1.14 | 0.858 | 0.283 | 0.859 | 0.286 |
| 7 | mean of 42 | 0.0996 | 0.0946 | -0.0050 | -1.27 | -1.28 | -1.28 | -1.25 | -1.28 | -1.37 | 0.899 | 0.202 | 0.898 | 0.205 |
| 13 | mean of 42/43 | 0.1016 | 0.0946 | -0.0070 | -1.63 | -1.67 | -1.67 | -1.66 | -1.68 | -1.79 | 0.952 | 0.096 | 0.951 | 0.097 |
| average of all pools | seed averages | 0.0968 | 0.0946 | -0.0022 | -0.67 | -0.67 | -0.66 | -0.65 | -0.66 | -0.73 | 0.747 | 0.506 | 0.747 | 0.509 |

Fixed-b critical values (simulated, 50000 draws, grid 2000): lag 0 (b = 0.0012): one-sided 5% 1.642, two-sided 5% 1.943; lag 5 (b = 0.0069): one-sided 5% 1.656, two-sided 5% 1.961; lag 6 (b = 0.0081): one-sided 5% 1.654, two-sided 5% 1.963; lag 10 (b = 0.0127): one-sided 5% 1.660, two-sided 5% 1.972; lag 14 (b = 0.0173): one-sided 5% 1.671, two-sided 5% 1.982; lag 21 (b = 0.0254): one-sided 5% 1.685, two-sided 5% 2.011.

MSE (variance level), automatic lag: k = 1: DM -1.23 (p one 0.891); k = 2: DM -0.98 (p one 0.835); k = 3: DM 0.85 (p one 0.197); k = 4: DM 0.81 (p one 0.210); k = 5: DM 1.04 (p one 0.150); k = 7: DM 1.12 (p one 0.132); k = 13: DM 0.53 (p one 0.298); k = average of all pools: DM 0.02 (p one 0.490).

Secondary baselines (QLIKE, automatic lag), DM for each k:

| baseline | QLIKE baseline | k = 1 | k = 2 | k = 3 | k = 4 | k = 5 | k = 7 | k = 13 | k = average of all pools |
|---|---|---|---|---|---|---|---|---|---|
| ridge_bb0 | 0.0946 | -1.62 | -1.15 | -0.82 | -0.85 | -1.07 | -1.28 | -1.67 | -0.66 |
| lasso_bb0 | 0.0957 | -1.50 | -0.97 | -0.57 | -0.61 | -0.87 | -1.14 | -1.60 | -0.40 |
| ridge_single | 0.1004 | 0.11 | 0.54 | 0.65 | 0.66 | 0.45 | 0.18 | -0.24 | 0.86 |
| lasso_single | 0.0998 | -0.07 | 0.63 | 0.81 | 0.85 | 0.55 | 0.06 | -0.66 | 1.24 |

Single seeds (QLIKE, automatic lag): k = 1 seed 42: DM -1.81; k = 1 seed 43: DM -1.62; k = 2 seed 42: DM -1.16; k = 2 seed 43: DM -1.30; k = 3 seed 42: DM -0.79; k = 3 seed 43: DM -0.96; k = 4 seed 42: DM -0.83; k = 4 seed 43: DM -0.98; k = 5 seed 42: DM -1.07; k = 7 seed 42: DM -1.28; k = 13 seed 42: DM -1.71; k = 13 seed 43: DM -1.65.
Outside the family: k = 13, column bar_end_minute, seed 42: QLIKE 0.1015, DM -1.61 (p one 0.946).

## 2. Giacomini-White against the ridge (HAC, newey_west_lag 14)

QLIKE: statistic (p), chi-square with q df.

| k | 1 (q = 1) | 1, d(t-1) (q = 2) | 1, d(t-1), implied variance(t-1) (q = 3) | 1, d(t-1), VIX(t-1) (days with a VIX value) (q = 3) |
|---|---|---|---|---|
| 1 | 2.69 (0.101) | 2.89 (0.236) | 5.37 (0.147) | 5.00 (0.172) |
| 2 | 1.32 (0.251) | 1.43 (0.489) | 4.35 (0.226) | 3.30 (0.348) |
| 3 | 0.64 (0.422) | 0.65 (0.723) | 2.91 (0.406) | 2.10 (0.552) |
| 4 | 0.70 (0.401) | 0.71 (0.703) | 3.47 (0.325) | 2.53 (0.470) |
| 5 | 1.13 (0.287) | 1.14 (0.565) | 4.28 (0.233) | 2.85 (0.415) |
| 7 | 1.64 (0.201) | 1.65 (0.438) | 5.23 (0.156) | 3.14 (0.371) |
| 13 | 2.81 (0.094) | 2.90 (0.235) | 8.43 (0.038) | 6.89 (0.076) |
| average of all pools | 0.44 (0.506) | 0.47 (0.789) | 2.92 (0.404) | 2.00 (0.572) |

MSE: statistic (p), chi-square with q df.

| k | 1 (q = 1) | 1, d(t-1) (q = 2) | 1, d(t-1), implied variance(t-1) (q = 3) | 1, d(t-1), VIX(t-1) (days with a VIX value) (q = 3) |
|---|---|---|---|---|
| 1 | 1.31 (0.253) | 1.32 (0.517) | 1.49 (0.686) | 1.97 (0.578) |
| 2 | 1.03 (0.310) | 1.03 (0.596) | 1.05 (0.790) | 1.65 (0.647) |
| 3 | 0.73 (0.392) | 0.76 (0.684) | 0.85 (0.836) | 1.39 (0.707) |
| 4 | 0.71 (0.399) | 1.00 (0.607) | 1.00 (0.802) | 1.29 (0.733) |
| 5 | 0.93 (0.335) | 0.99 (0.610) | 0.99 (0.804) | 1.50 (0.682) |
| 7 | 1.01 (0.314) | 1.01 (0.603) | 1.26 (0.739) | 1.83 (0.608) |
| 13 | 0.44 (0.508) | 1.09 (0.579) | 1.11 (0.776) | 1.17 (0.759) |
| average of all pools | 0.00 (0.972) | 0.90 (0.638) | 0.91 (0.823) | 1.17 (0.761) |

Conditional sets use trade days 2 .. 866 (d(t-1) is the previous trade day's d; the deck's trade days are not every session); implied variance(t-1) = the deck's iv_var on the previous trade day; VIX(t-1) = VIX at the bar ending 16:00 of the previous session, on the 812 days with a value (the VIX file ends 2024-02-12).

## 3. Multiple comparisons across k (against the ridge)
SPA and Reality Check: H0 max over k of E[d(k)] <= 0 (no pool has lower expected loss than the ridge); d(k) for the seed-averaged forecast of each k.
- QLIKE: SPA statistic 0.000 (largest studentized mean d at nobar:k3), p consistent / lower / upper = 0.723 / 0.241 / 0.723 (mean block 10); block 5: 0.711 / 0.244 / 0.711; block 21: 0.738 / 0.220 / 0.738. Reality Check p = 0.941 (block 10); block 5 0.934, block 21 0.956.
  Holm-Bonferroni, one-sided DM p (automatic lag), adjusted: k = 1 1.000; k = 2 1.000; k = 3 1.000; k = 4 1.000; k = 5 1.000; k = 7 1.000; k = 13 1.000; smallest adjusted 1.000.
- MSE: SPA statistic 1.034 (largest studentized mean d at nobar:k7), p consistent / lower / upper = 0.404 / 0.237 / 0.404 (mean block 10); block 5: 0.318 / 0.226 / 0.318; block 21: 0.465 / 0.230 / 0.465. Reality Check p = 0.538 (block 10); block 5 0.542, block 21 0.541.
  Holm-Bonferroni, one-sided DM p (automatic lag), adjusted: k = 1 1.000; k = 2 1.000; k = 3 0.985; k = 4 0.985; k = 5 0.925; k = 7 0.925; k = 13 0.985; smallest adjusted 0.925.

Model confidence set over {ridge_bb0, lasso_bb0, every tree pool} (T_max rule of `src/evaluation/model_confidence_set.py`, stationary bootstrap, mean block 10):

- QLIKE, alpha 0.10: 9 of 9 in the set: ridge_bb0 (p 0.226), lasso_bb0 (p 0.226), k = 4 (p 0.226), k = 3 (p 0.226), k = 2 (p 0.226), k = 5 (p 0.226), k = 7 (p 0.226), k = 1 (p 0.226), k = 13 (p 0.226).
- QLIKE, alpha 0.25: 8 of 9 in the set: ridge_bb0 (p 0.547), lasso_bb0 (p 0.547), k = 4 (p 0.547), k = 3 (p 0.547), k = 2 (p 0.547), k = 5 (p 0.547), k = 7 (p 0.547), k = 1 (p 0.547); eliminated in order: k = 13 (p 0.226).
- MSE, alpha 0.10: 9 of 9 in the set: k = 7 (p 0.343), lasso_bb0 (p 0.343), k = 5 (p 0.343), k = 3 (p 0.343), k = 4 (p 0.343), k = 13 (p 0.343), ridge_bb0 (p 0.343), k = 1 (p 0.343), k = 2 (p 0.343).
- MSE, alpha 0.25: 9 of 9 in the set: k = 7 (p 0.343), lasso_bb0 (p 0.343), k = 5 (p 0.343), k = 3 (p 0.343), k = 4 (p 0.343), k = 13 (p 0.343), ridge_bb0 (p 0.343), k = 1 (p 0.343), k = 2 (p 0.343).

## 4. Encompassing
k* = 1: lowest seed-mean QLIKE on the 189 16:00 rows 2019-04-04 .. 2020-01-02 (2019, before the trade days; the recalibration needs 63 earlier errors): k = 1 0.1616; k = 2 0.1691; k = 3 0.1757; k = 4 0.1700; k = 5 0.1726; k = 7 0.1766; k = 13 0.1731. The QLIKE of the seed-averaged forecast on the same rows is lowest at k = 1.

y = realized 16:00 variance, f = recalibrated forecasts (pred_clock), f_L = ridge_bb0, f_T = tree; HAC lag 14.

| tree forecast | joint MZ b_L (t of 0) | b_T (t of 0) | Wald (a, b_L, b_T) = (0, 1, 0) p | weighted MZ b_L (t) | b_T (t) | Wald p | lambda (se) | t lambda = 0 | t lambda = 1 | QLIKE moment t | reverse moment t |
|---|---|---|---|---|---|---|---|---|---|---|---|
| k* = 1 | -0.021 (-0.16) | 0.590 (5.95) | 2.6e-16 | 0.601 (6.40) | 0.309 (3.11) | 5.1e-07 | 0.050 (0.119) | 0.42 | -8.00 | -1.61 | -4.48 |
| average of all pools | -0.268 (-2.41) | 0.888 (8.49) | 9.5e-35 | 0.472 (4.24) | 0.444 (3.73) | 3.1e-08 | 0.505 (0.143) | 3.54 | -3.47 | -2.10 | -4.36 |
| k = 2 (not chosen in advance) | -0.194 (-1.29) | 0.701 (6.75) | 2.5e-16 | 0.496 (4.31) | 0.410 (3.44) | 2.8e-07 | -0.068 (0.150) | -0.45 | -7.12 | -1.95 | -5.03 |
| k = 3 (not chosen in advance) | -0.182 (-1.58) | 0.849 (8.30) | 1.3e-23 | 0.478 (4.81) | 0.436 (4.12) | 7.0e-09 | 0.705 (0.126) | 5.61 | -2.35 | -2.38 | -3.79 |
| k = 4 (not chosen in advance) | -0.265 (-2.68) | 0.913 (8.81) | 3.1e-49 | 0.486 (4.64) | 0.433 (3.84) | 2.2e-08 | 0.685 (0.125) | 5.50 | -2.53 | -2.22 | -4.66 |
| k = 5 (not chosen in advance) | -0.206 (-1.77) | 0.891 (8.10) | 1.0e-25 | 0.516 (4.70) | 0.405 (3.40) | 1.3e-07 | 0.835 (0.143) | 5.83 | -1.15 | -2.33 | -4.78 |
| k = 7 (not chosen in advance) | -0.150 (-1.50) | 0.858 (7.04) | 1.3e-52 | 0.516 (4.91) | 0.403 (3.55) | 1.0e-07 | 0.905 (0.147) | 6.14 | -0.64 | -2.14 | -4.80 |
| k = 13 (not chosen in advance) | -0.270 (-2.62) | 0.905 (7.68) | 7.5e-95 | 0.542 (4.20) | 0.375 (2.78) | 1.2e-05 | 0.624 (0.152) | 4.10 | -2.47 | -1.90 | -5.24 |

QLIKE moment t < 0: moving from f_L toward f_T lowers QLIKE at lambda = 0 (f_L does not encompass f_T); reverse moment t < 0: moving from f_T toward f_L lowers it.

Combinations against the ridge (DM, automatic lag; positive = combination lower loss):

| tree forecast | combination | sample | loss | mean ridge | mean combination | DM | p one | lambda first / median / last (min .. max) |
|---|---|---|---|---|---|---|---|---|
| k* = 1 | real-time lambda (QLIKE on past trade days, expanding) | trade days 251 .. 866 | qlike | 0.0842 | 0.0833 | 1.45 | 0.074 | 0.209 / 0.227 / 0.249 (0.199 .. 0.267) |
| k* = 1 | real-time lambda (QLIKE on past trade days, expanding) | trade days 251 .. 866 | mse | 1.477e-11 | 1.451e-11 | 0.99 | 0.160 | 0.209 / 0.227 / 0.249 (0.199 .. 0.267) |
| k* = 1 | equal weights, variance level | trade days 251 .. 866 | qlike | 0.0842 | 0.0833 | 0.69 | 0.244 | 0.5 |
| k* = 1 | equal weights, variance level | trade days 251 .. 866 | mse | 1.477e-11 | 1.455e-11 | 0.35 | 0.363 | 0.5 |
| k* = 1 | equal weights, variance level | all 866 trade days | qlike | 0.0946 | 0.0939 | 0.36 | 0.359 | 0.5 |
| k* = 1 | equal weights, variance level | all 866 trade days | mse | 7.915e-10 | 8.723e-10 | -0.85 | 0.803 | 0.5 |
| k* = 1 | equal weights, adjusted scale, rescored | trade days 251 .. 866 | qlike | 0.0842 | 0.0832 | 0.83 | 0.204 | 0.5 |
| k* = 1 | equal weights, adjusted scale, rescored | trade days 251 .. 866 | mse | 1.477e-11 | 1.450e-11 | 0.44 | 0.331 | 0.5 |
| k* = 1 | equal weights, adjusted scale, rescored | all 866 trade days | qlike | 0.0946 | 0.0933 | 0.71 | 0.239 | 0.5 |
| k* = 1 | equal weights, adjusted scale, rescored | all 866 trade days | mse | 7.915e-10 | 8.602e-10 | -0.77 | 0.778 | 0.5 |
| average of all pools | real-time lambda (QLIKE on past trade days, expanding) | trade days 251 .. 866 | qlike | 0.0842 | 0.0821 | 1.82 | 0.035 | 0.327 / 0.394 / 0.397 (0.321 .. 0.452) |
| average of all pools | real-time lambda (QLIKE on past trade days, expanding) | trade days 251 .. 866 | mse | 1.477e-11 | 1.493e-11 | -0.18 | 0.573 | 0.327 / 0.394 / 0.397 (0.321 .. 0.452) |
| average of all pools | equal weights, variance level | trade days 251 .. 866 | qlike | 0.0842 | 0.0819 | 1.70 | 0.045 | 0.5 |
| average of all pools | equal weights, variance level | trade days 251 .. 866 | mse | 1.477e-11 | 1.532e-11 | -0.40 | 0.654 | 0.5 |
| average of all pools | equal weights, variance level | all 866 trade days | qlike | 0.0946 | 0.0918 | 1.44 | 0.075 | 0.5 |
| average of all pools | equal weights, variance level | all 866 trade days | mse | 7.915e-10 | 7.402e-10 | 0.93 | 0.175 | 0.5 |
| average of all pools | equal weights, adjusted scale, rescored | trade days 251 .. 866 | qlike | 0.0842 | 0.0819 | 1.65 | 0.050 | 0.5 |
| average of all pools | equal weights, adjusted scale, rescored | trade days 251 .. 866 | mse | 1.477e-11 | 1.525e-11 | -0.35 | 0.638 | 0.5 |
| average of all pools | equal weights, adjusted scale, rescored | all 866 trade days | qlike | 0.0946 | 0.0916 | 1.62 | 0.052 | 0.5 |
| average of all pools | equal weights, adjusted scale, rescored | all 866 trade days | mse | 7.915e-10 | 7.379e-10 | 0.98 | 0.164 | 0.5 |

## 5. Stability (QLIKE, against the ridge)
Giacomini-Rossi fluctuation test, window 260 trade days (mu = 0.3), statistic = rolling sum of d / (full-sample HAC sd x sqrt(window)); simulated critical values (50000 draws): two-sided 5% 3.082 / 10% 2.813, one-sided 5% 2.821 / 10% 2.531.

| k | max (window end) | min (window end) | sup abs > two-sided 5% | max > one-sided 5% |
|---|---|---|---|---|
| 1 | -0.03 (2023-03-28) | -2.93 (2021-11-15) | False | False |
| 2 | 0.54 (2023-03-31) | -2.46 (2021-11-30) | False | False |
| 3 | 0.74 (2023-04-03) | -1.66 (2024-04-23) | False | False |
| 4 | 0.65 (2023-04-03) | -1.70 (2021-11-30) | False | False |
| 5 | 0.46 (2023-02-22) | -2.18 (2021-11-30) | False | False |
| 7 | 0.18 (2023-02-22) | -1.94 (2021-11-30) | False | False |
| 13 | 0.01 (2023-09-21) | -2.49 (2021-11-30) | False | False |
| average of all pools | 0.62 (2023-04-03) | -1.96 (2021-11-30) | False | False |

DM (automatic lag) in subsets; n days in the header:

| k | all 866 trade days (866) | 2020-2021 (316) | 2022-2024 (550) | previous-day VIX at or above its median (408) | previous-day VIX below its median (405) | previous trade day implied variance at or above its median (433) | previous trade day implied variance below its median (432) | FOMC days excluded (flags end 2023-11-01) (837) | early-close days excluded (866) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | -1.62 | -1.42 | -0.91 | -1.60 | -0.21 | -0.97 | -1.64 | -1.71 | -1.62 |
| 2 | -1.15 | -1.11 | -0.49 | -0.92 | -0.05 | -0.39 | -1.52 | -1.00 | -1.15 |
| 3 | -0.82 | -0.60 | -0.57 | -0.23 | 0.06 | 0.38 | -1.47 | -0.71 | -0.82 |
| 4 | -0.85 | -0.48 | -0.74 | -0.28 | -0.13 | 0.36 | -1.66 | -0.79 | -0.85 |
| 5 | -1.07 | -0.79 | -0.73 | -0.63 | 0.00 | 0.20 | -1.78 | -1.00 | -1.07 |
| 7 | -1.28 | -0.71 | -1.10 | -0.74 | -0.15 | -0.03 | -1.74 | -1.14 | -1.28 |
| 13 | -1.67 | -1.15 | -1.22 | -1.00 | -0.99 | -0.30 | -1.96 | -1.22 | -1.67 |
| average of all pools | -0.66 | -0.60 | -0.33 | -0.45 | 0.51 | 0.27 | -1.34 | -0.55 | -0.66 |

Mean QLIKE of the ridge in each subset: all 866 trade days 0.0946; 2020-2021 0.1118; 2022-2024 0.0847; previous-day VIX at or above its median 0.1010; previous-day VIX below its median 0.0841; previous trade day implied variance at or above its median 0.0925; previous trade day implied variance below its median 0.0969; FOMC days excluded (flags end 2023-11-01) 0.0951; early-close days excluded 0.0946.

Difference in mean d between groups (d on a constant and the group dummy, HAC t of the dummy): 2022-2024 vs 2020-2021: k = 1 0.98, k = 2 0.79, k = 3 0.18, k = 4 0.05, k = 5 0.32, k = 7 0.04, k = 13 0.40, k = average of all pools 0.38; VIX high vs low: k = 1 -1.28, k = 2 -0.78, k = 3 -0.22, k = 4 -0.17, k = 5 -0.54, k = 7 -0.57, k = 13 -0.41, k = average of all pools -0.61; implied variance high vs low: k = 1 -0.02, k = 2 0.50, k = 3 1.29, k = 4 1.30, k = 5 1.31, k = 7 1.20, k = 13 1.25, k = average of all pools 1.03; FOMC day vs other days: k = 1 1.21, k = 2 -1.01, k = 3 -0.71, k = 4 -0.38, k = 5 -0.41, k = 7 -0.94, k = 13 -2.25, k = average of all pools -0.76.

FOMC flags in `data/releases.parquet` stop at 2023-11-01, so FOMC days after that date stay in the 'FOMC days excluded' subset; 29 flagged FOMC days are trade days. Early-close days among the trade days: 0 (the 15:30 straddle deck has none), so that subset equals the full sample. VIX median split at 20.82 on the 813 trade days whose previous session has a VIX value (`data/vix_and_voldemand.parquet` ends 2024-02-12; last such trade day 2024-02-13); the implied-variance split uses the deck's iv_var of the previous trade day (median 5.678e-06) on trade days 2 .. 866.

## 6. Calibration: Mincer-Zarnowitz on the variance level, y = a + b f (HAC)

| forecast | a | b (se) | t of b = 1 | Wald (a, b) = (0, 1) p | R^2 | weighted b (t of b = 1) | weighted Wald p |
|---|---|---|---|---|---|---|---|
| ridge_bb0 | 2.63e-06 | 0.693 (0.048) | -6.41 | 5.1e-10 | 0.668 | 0.920 (-2.91) | 0.013 |
| lasso_bb0 | 1.99e-06 | 0.733 (0.057) | -4.67 | 1.1e-05 | 0.703 | 0.912 (-3.18) | 0.004 |
| ridge_single | 4.71e-06 | 0.496 (0.024) | -20.82 | 8.3e-102 | 0.663 | 0.941 (-2.05) | 0.123 |
| lasso_single | 3.47e-06 | 0.596 (0.031) | -13.14 | 2.6e-38 | 0.699 | 0.927 (-2.58) | 0.024 |
| k = 1 | 2.91e-06 | 0.575 (0.042) | -10.23 | 6.2e-37 | 0.728 | 0.908 (-2.97) | 0.003 |
| k = 2 | 3.32e-06 | 0.557 (0.030) | -14.83 | 1.1e-49 | 0.734 | 0.896 (-3.54) | 0.002 |
| k = 3 | 2.08e-06 | 0.684 (0.045) | -7.02 | 1.5e-15 | 0.739 | 0.907 (-3.06) | 0.007 |
| k = 4 | 2.20e-06 | 0.677 (0.041) | -7.95 | 5.7e-19 | 0.750 | 0.916 (-2.69) | 0.011 |
| k = 5 | 2.05e-06 | 0.699 (0.044) | -6.92 | 4.3e-14 | 0.736 | 0.920 (-2.58) | 0.014 |
| k = 7 | 1.78e-06 | 0.717 (0.047) | -6.01 | 1.5e-12 | 0.753 | 0.919 (-2.51) | 0.010 |
| k = 13 | 2.32e-06 | 0.668 (0.042) | -7.95 | 2.2e-15 | 0.755 | 0.915 (-2.79) | 0.002 |
| average of all pools | 2.36e-06 | 0.656 (0.040) | -8.59 | 7.1e-21 | 0.747 | 0.914 (-2.83) | 0.014 |

## 7. Seed spread and the curve of seed-averaged forecasts (QLIKE)

| k | seeds | QLIKE for each seed | seed mean (sd) | QLIKE of the seed average | step from the previous k | sd of the two ends | step exceeds sd | DM of the step |
|---|---|---|---|---|---|---|---|---|
| 1 | 42/43 | 0.1007, 0.1000 | 0.1003 (0.0005) | 0.1000 |  |  |  |  |
| 2 | 42/43 | 0.0982, 0.0987 | 0.0985 (0.0003) | 0.0982 | -0.0018 (k = 1) | 0.0004 | True | 0.91 |
| 3 | 42/43 | 0.0975, 0.0981 | 0.0978 (0.0004) | 0.0976 | -0.0006 (k = 2) | 0.0004 | True | 0.44 |
| 4 | 42/43 | 0.0974, 0.0981 | 0.0978 (0.0005) | 0.0976 | -0.0000 (k = 3) | 0.0004 | False | 0.00 |
| 5 | 42 | 0.0984 | 0.0984 (one seed) | 0.0984 | +0.0009 (k = 4) | 0.0005 | True | -1.07 |
| 7 | 42 | 0.0996 | 0.0996 (one seed) | 0.0996 | +0.0012 (k = 5) |  |  | -1.29 |
| 13 | 42/43 | 0.1019, 0.1014 | 0.1016 (0.0003) | 0.1016 | +0.0020 (k = 7) | 0.0003 | True | -1.52 |

sd = sample sd over seeds (ddof 1); sd of the two ends = root mean square of the two k's seed sd's (one end when the other has one seed); DM of the step: QLIKE of the previous k's seed average minus this k's (positive = this k lower).

## 8. Use of the bar column (bar_end_minute)
`results/close_studies_2026-10-03/trees_kfull/bar_column_by_k.csv` did not exist when the script ran (the other agent writes it from the importances of its new forecast files).

## 9. Within-day dependence of the adjusted target (lastbars13 design)

Mean pairwise correlation rho across the last k bars of the same session (columns = bars, rows = sessions with all 13 bars), design effect 1 + (k - 1) rho, effective n = 2000 k / design effect.

| k | added bar | corr of added bar with 16:00 | rho | design effect | rows | effective n | rho (all sessions) | effective n (all sessions) |
|---|---|---|---|---|---|---|---|---|
| 1 | 16:00 | 1.000 |  | 1.00 | 2000 | 2000 |  | 2000 |
| 2 | 15:30 | 0.794 | 0.794 | 1.79 | 4000 | 2230 | 0.775 | 2253 |
| 3 | 15:00 | 0.707 | 0.773 | 2.55 | 6000 | 2356 | 0.749 | 2401 |
| 4 | 14:30 | 0.622 | 0.740 | 3.22 | 8000 | 2485 | 0.725 | 2520 |
| 5 | 14:00 | 0.640 | 0.706 | 3.82 | 10000 | 2615 | 0.674 | 2704 |
| 6 | 13:30 | 0.633 | 0.690 | 4.45 | 12000 | 2697 | 0.653 | 2815 |
| 7 | 13:00 | 0.585 | 0.673 | 5.04 | 14000 | 2780 | 0.637 | 2902 |
| 8 | 12:30 | 0.599 | 0.665 | 5.65 | 16000 | 2831 | 0.628 | 2967 |
| 9 | 12:00 | 0.594 | 0.659 | 6.27 | 18000 | 2869 | 0.622 | 3013 |
| 10 | 11:30 | 0.556 | 0.651 | 6.86 | 20000 | 2915 | 0.617 | 3053 |
| 11 | 11:00 | 0.576 | 0.645 | 7.45 | 22000 | 2954 | 0.612 | 3089 |
| 12 | 10:30 | 0.549 | 0.638 | 8.01 | 24000 | 2995 | 0.603 | 3142 |
| 13 | 10:00 | 0.546 | 0.628 | 8.54 | 26000 | 3046 | 0.597 | 3185 |

Main columns: the 2000 full sessions before 2020-01-03 (2000 sessions, 2012-01-13 .. 2020-01-02); 'all sessions': 6352 sessions, 1998-12-17 .. 2024-04-30. The trees forecast only the 16:00 row, so no tree errors exist on the earlier bars; the target is used.

## Files
- `gate.csv`, `forecasts.csv` (every scored forecast, file, QLIKE, MSE), `kstar_selection.csv`, `dm.csv`, `fixedb_critical_values.csv`, `gw.csv`, `multiple_comparisons.csv`, `mcs.csv`, `encompassing.csv`, `realtime_combination.csv`, `realtime_lambda_path.csv`, `stability.csv`, `fluctuation.csv`, `fluctuation_path.csv`, `fluctuation_critical_values.csv`, `calibration.csv`, `seed_spread.csv`, `seed_curve.csv`, `bar_column.csv`, `within_day.csv`, `run_info.json`.
- CPU time of the run: 0.3 min (one process), wall 0.3 min.
