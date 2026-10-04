# Tests: LightGBM on the last k bars ending 16:00 against the linear 16:00 model (close study, 2026-10-04)

Written by `experiments/close_kfull_tests.py` from the CSVs in this folder (2026-10-04 10:24:02); every number below is in them.

## Forecasts and set
- Set: **existing**. Manifest `results/close_studies_2026-10-03/trees_kfull/manifest.csv`: 68 rows, complete: False.
- Family `nobar` (LightGBM, last k bars, no bar column): k = 1: seeds 42/43/44; k = 2: seeds 42/43; k = 3: seeds 42/43; k = 4: seeds 42/43; k = 5: seeds 42; k = 7: seeds 42; k = 13: seeds 42/43
- Family `bar` (LightGBM, last k bars, column bar_end_minute for k > 1): k = 1: seeds 42/43/44; k = 2: seeds 42/43/44; k = 3: seeds 42/43/44; k = 4: seeds 42/43/44; k = 5: seeds 42/43/44; k = 6: seeds 42/43/44; k = 7: seeds 42/43/44; k = 8: seeds 42/43/44; k = 9: seeds 42/43/44; k = 10: seeds 42/43/44; k = 11: seeds 42/43/44; k = 12: seeds 42/43/44; k = 13: seeds 42/43/44  (primary)
- This is the set that existed when the script ran; the main design (bar column, k = 1 .. 13, seeds 42 / 43 / 44) is complete, and the manifest's other planned arms (no bar column) were still running. For each k the tested forecast is the average of the seeds listed.
- Linear baseline: `ridge_bb0` (ridge, HAR + calendar unpenalized; QLIKE 0.0946); secondary `lasso_bb0`, `ridge_single`, `lasso_single`.
- Scorer: research convention (16:00-bar recalibration (f^2 + s) B from each forecast's own errors), 866 trade days 2020-01-03 .. 2024-04-30, one loss a day. For each k, the seed forecasts are averaged on the adjusted scale and the average is scored. Losses: QLIKE (primary), MSE on the variance level. d = L_linear - L_tree, positive = the tree has lower loss.
- Gate: trees_morebars/average.csv reproduced, 20 / 20 quantities (QLIKE, Sharpe mid, DM against the ridge) within 1e-12 (max |difference| 4.9e-15); e.g. QLIKE lgbm 4 bars 0.0974, ridge_bb0 0.0946, equal weights ridge + 4-bar pool 0.0914.
- Resampling: the repo's circular block bootstrap helper is switched off (commit b761b28: `atm_straddle_lib.circular_block_bootstrap_idx` returns the original order; `model_confidence_set._boot_col_means` likewise returns the sample in every replicate). SPA, Reality Check and the MCS here use this script's own stationary bootstrap (Politis-Romano), B = 10000, mean block 10 (also 5, 21 in the CSV); the MCS keeps the module's elimination rule with these replicate means.

## 1. Diebold-Mariano against the ridge (HLN-corrected), QLIKE
DM statistic (positive = tree lower loss) at Newey-West lag 0 / 5 / 6 (dm_test automatic) / 10 / 14 (newey_west_lag) / 21; one-sided normal p (H1: tree lower loss) and fixed-b one-sided p (Bartlett, b = (lag+1)/T, simulated) at the automatic lag.

| k | seeds | QLIKE tree | QLIKE ridge | mean d | DM lag 0 | DM lag 5 | DM lag 6 | DM lag 10 | DM lag 14 | DM lag 21 | p one (auto) | p two (auto) | fixed-b p one (auto) | fixed-b p two (auto) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | mean of 42/43/44 | 0.0996 | 0.0946 | -0.0051 | -1.75 | -1.57 | -1.52 | -1.48 | -1.53 | -1.71 | 0.936 | 0.128 | 0.934 | 0.131 |
| 2 | mean of 42/43/44 | 0.0982 | 0.0946 | -0.0036 | -1.21 | -1.17 | -1.14 | -1.10 | -1.13 | -1.25 | 0.874 | 0.253 | 0.873 | 0.256 |
| 3 | mean of 42/43/44 | 0.0969 | 0.0946 | -0.0023 | -0.66 | -0.66 | -0.65 | -0.63 | -0.64 | -0.70 | 0.743 | 0.515 | 0.743 | 0.518 |
| 4 | mean of 42/43/44 | 0.0980 | 0.0946 | -0.0034 | -0.96 | -0.93 | -0.92 | -0.89 | -0.90 | -0.97 | 0.821 | 0.357 | 0.822 | 0.359 |
| 5 | mean of 42/43/44 | 0.0975 | 0.0946 | -0.0029 | -0.80 | -0.80 | -0.79 | -0.77 | -0.78 | -0.83 | 0.786 | 0.428 | 0.786 | 0.431 |
| 6 | mean of 42/43/44 | 0.0982 | 0.0946 | -0.0036 | -1.04 | -1.02 | -1.01 | -0.99 | -1.01 | -1.08 | 0.844 | 0.312 | 0.844 | 0.315 |
| 7 | mean of 42/43/44 | 0.0992 | 0.0946 | -0.0046 | -1.16 | -1.17 | -1.17 | -1.14 | -1.17 | -1.25 | 0.878 | 0.244 | 0.878 | 0.247 |
| 8 | mean of 42/43/44 | 0.0994 | 0.0946 | -0.0048 | -1.16 | -1.20 | -1.19 | -1.17 | -1.19 | -1.27 | 0.884 | 0.232 | 0.883 | 0.235 |
| 9 | mean of 42/43/44 | 0.1011 | 0.0946 | -0.0065 | -1.48 | -1.56 | -1.56 | -1.54 | -1.57 | -1.65 | 0.940 | 0.119 | 0.939 | 0.122 |
| 10 | mean of 42/43/44 | 0.1004 | 0.0946 | -0.0058 | -1.35 | -1.39 | -1.38 | -1.37 | -1.40 | -1.48 | 0.917 | 0.166 | 0.916 | 0.169 |
| 11 | mean of 42/43/44 | 0.1005 | 0.0946 | -0.0059 | -1.34 | -1.38 | -1.38 | -1.37 | -1.40 | -1.50 | 0.916 | 0.168 | 0.915 | 0.170 |
| 12 | mean of 42/43/44 | 0.1009 | 0.0946 | -0.0064 | -1.37 | -1.39 | -1.39 | -1.37 | -1.39 | -1.48 | 0.918 | 0.165 | 0.917 | 0.167 |
| 13 | mean of 42/43/44 | 0.1015 | 0.0946 | -0.0069 | -1.57 | -1.61 | -1.61 | -1.60 | -1.62 | -1.73 | 0.946 | 0.108 | 0.945 | 0.110 |
| average of all pools | seed averages | 0.0975 | 0.0946 | -0.0030 | -0.81 | -0.82 | -0.81 | -0.80 | -0.81 | -0.88 | 0.792 | 0.416 | 0.792 | 0.419 |

Fixed-b critical values (simulated, 50000 draws, grid 2000): lag 0 (b = 0.0012): one-sided 5% 1.642, two-sided 5% 1.943; lag 5 (b = 0.0069): one-sided 5% 1.656, two-sided 5% 1.961; lag 6 (b = 0.0081): one-sided 5% 1.654, two-sided 5% 1.963; lag 10 (b = 0.0127): one-sided 5% 1.660, two-sided 5% 1.972; lag 14 (b = 0.0173): one-sided 5% 1.671, two-sided 5% 1.982; lag 21 (b = 0.0254): one-sided 5% 1.685, two-sided 5% 2.011.

MSE (variance level), automatic lag: k = 1: DM -1.18 (p one 0.881); k = 2: DM -0.94 (p one 0.826); k = 3: DM 0.92 (p one 0.178); k = 4: DM 0.96 (p one 0.169); k = 5: DM 1.17 (p one 0.120); k = 6: DM 1.16 (p one 0.122); k = 7: DM 1.16 (p one 0.122); k = 8: DM 1.11 (p one 0.133); k = 9: DM 0.76 (p one 0.225); k = 10: DM 1.18 (p one 0.119); k = 11: DM 1.12 (p one 0.131); k = 12: DM 1.16 (p one 0.122); k = 13: DM 0.55 (p one 0.292); k = average of all pools: DM 0.95 (p one 0.170).

Secondary baselines (QLIKE, automatic lag), DM for each k:

| baseline | QLIKE baseline | k = 1 | k = 2 | k = 3 | k = 4 | k = 5 | k = 6 | k = 7 | k = 8 | k = 9 | k = 10 | k = 11 | k = 12 | k = 13 | k = average of all pools |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ridge_bb0 | 0.0946 | -1.52 | -1.14 | -0.65 | -0.92 | -0.79 | -1.01 | -1.17 | -1.19 | -1.56 | -1.38 | -1.38 | -1.39 | -1.61 | -0.81 |
| lasso_bb0 | 0.0957 | -1.40 | -0.96 | -0.37 | -0.70 | -0.56 | -0.82 | -1.02 | -1.04 | -1.42 | -1.25 | -1.25 | -1.29 | -1.53 | -0.59 |
| ridge_single | 0.1004 | 0.19 | 0.54 | 0.83 | 0.55 | 0.68 | 0.50 | 0.25 | 0.21 | -0.15 | 0.00 | -0.03 | -0.11 | -0.22 | 0.65 |
| lasso_single | 0.0998 | 0.03 | 0.62 | 1.10 | 0.67 | 0.93 | 0.63 | 0.20 | 0.14 | -0.48 | -0.22 | -0.26 | -0.39 | -0.61 | 0.91 |

Single seeds (QLIKE, automatic lag): k = 1 seed 42: DM -1.81; k = 1 seed 43: DM -1.62; k = 1 seed 44: DM -1.50; k = 2 seed 42: DM -0.96; k = 2 seed 43: DM -1.14; k = 2 seed 44: DM -1.59; k = 3 seed 42: DM -0.75; k = 3 seed 43: DM -0.77; k = 3 seed 44: DM -0.68; k = 4 seed 42: DM -0.95; k = 4 seed 43: DM -1.11; k = 4 seed 44: DM -0.91; k = 5 seed 42: DM -0.78; k = 5 seed 43: DM -0.92; k = 5 seed 44: DM -0.86; k = 6 seed 42: DM -1.13; k = 6 seed 43: DM -0.93; k = 6 seed 44: DM -1.12; k = 7 seed 42: DM -1.34; k = 7 seed 43: DM -1.07; k = 7 seed 44: DM -1.22; k = 8 seed 42: DM -1.25; k = 8 seed 43: DM -1.28; k = 8 seed 44: DM -1.17; k = 9 seed 42: DM -1.51; k = 9 seed 43: DM -1.58; k = 9 seed 44: DM -1.68; k = 10 seed 42: DM -1.37; k = 10 seed 43: DM -1.28; k = 10 seed 44: DM -1.61; k = 11 seed 42: DM -1.37; k = 11 seed 43: DM -1.38; k = 11 seed 44: DM -1.47; k = 12 seed 42: DM -1.43; k = 12 seed 43: DM -1.42; k = 12 seed 44: DM -1.40; k = 13 seed 42: DM -1.61; k = 13 seed 43: DM -1.69; k = 13 seed 44: DM -1.59.

## 2. Giacomini-White against the ridge (HAC, newey_west_lag 14)

QLIKE: statistic (p), chi-square with q df.

| k | 1 (q = 1) | 1, d(t-1) (q = 2) | 1, d(t-1), implied variance(t-1) (q = 3) | 1, d(t-1), VIX(t-1) (days with a VIX value) (q = 3) |
|---|---|---|---|---|
| 1 | 2.35 (0.125) | 2.65 (0.266) | 5.19 (0.158) | 5.21 (0.157) |
| 2 | 1.27 (0.260) | 1.27 (0.530) | 4.15 (0.246) | 3.60 (0.308) |
| 3 | 0.41 (0.520) | 0.45 (0.797) | 2.52 (0.472) | 1.93 (0.586) |
| 4 | 0.82 (0.366) | 0.86 (0.652) | 3.68 (0.298) | 2.82 (0.420) |
| 5 | 0.61 (0.435) | 0.64 (0.726) | 3.47 (0.325) | 2.13 (0.545) |
| 6 | 1.02 (0.313) | 1.03 (0.596) | 4.65 (0.200) | 3.00 (0.392) |
| 7 | 1.36 (0.244) | 1.37 (0.505) | 4.99 (0.173) | 3.15 (0.369) |
| 8 | 1.43 (0.232) | 1.53 (0.466) | 4.99 (0.173) | 3.21 (0.361) |
| 9 | 2.46 (0.117) | 2.49 (0.288) | 6.49 (0.090) | 5.09 (0.165) |
| 10 | 1.95 (0.163) | 2.02 (0.365) | 5.90 (0.117) | 4.31 (0.230) |
| 11 | 1.95 (0.163) | 2.02 (0.364) | 5.69 (0.127) | 4.05 (0.256) |
| 12 | 1.93 (0.165) | 1.95 (0.377) | 6.43 (0.092) | 4.91 (0.178) |
| 13 | 2.64 (0.104) | 2.69 (0.260) | 8.07 (0.045) | 6.46 (0.091) |
| average of all pools | 0.66 (0.416) | 0.76 (0.684) | 3.57 (0.312) | 2.07 (0.558) |

MSE: statistic (p), chi-square with q df.

| k | 1 (q = 1) | 1, d(t-1) (q = 2) | 1, d(t-1), implied variance(t-1) (q = 3) | 1, d(t-1), VIX(t-1) (days with a VIX value) (q = 3) |
|---|---|---|---|---|
| 1 | 1.26 (0.261) | 1.27 (0.530) | 1.46 (0.690) | 2.03 (0.566) |
| 2 | 1.01 (0.315) | 1.02 (0.601) | 1.05 (0.789) | 1.70 (0.638) |
| 3 | 0.80 (0.372) | 0.83 (0.660) | 0.91 (0.822) | 1.47 (0.689) |
| 4 | 0.85 (0.355) | 0.99 (0.610) | 1.03 (0.794) | 1.42 (0.702) |
| 5 | 1.08 (0.298) | 1.09 (0.579) | 1.54 (0.673) | 1.68 (0.640) |
| 6 | 1.08 (0.298) | 1.10 (0.578) | 1.76 (0.623) | 1.70 (0.636) |
| 7 | 1.06 (0.304) | 1.06 (0.589) | 1.30 (0.729) | 1.71 (0.636) |
| 8 | 1.01 (0.316) | 1.03 (0.598) | 1.07 (0.784) | 1.51 (0.679) |
| 9 | 0.65 (0.419) | 0.91 (0.633) | 0.95 (0.813) | 1.10 (0.777) |
| 10 | 1.08 (0.299) | 1.10 (0.577) | 1.30 (0.730) | 1.62 (0.654) |
| 11 | 1.02 (0.313) | 1.06 (0.589) | 1.08 (0.781) | 1.55 (0.672) |
| 12 | 1.08 (0.299) | 1.11 (0.574) | 1.12 (0.772) | 1.31 (0.727) |
| 13 | 0.46 (0.499) | 1.10 (0.578) | 1.10 (0.776) | 1.19 (0.756) |
| average of all pools | 0.86 (0.354) | 0.96 (0.618) | 0.96 (0.810) | 1.25 (0.742) |

Conditional sets use trade days 2 .. 866 (d(t-1) is the previous trade day's d; the deck's trade days are not every session); implied variance(t-1) = the deck's iv_var on the previous trade day; VIX(t-1) = VIX at the bar ending 16:00 of the previous session, on the 812 days with a value (the VIX file ends 2024-02-12).

## 3. Multiple comparisons across k (against the ridge)
SPA and Reality Check: H0 max over k of E[d(k)] <= 0 (no pool has lower expected loss than the ridge); d(k) for the seed-averaged forecast of each k.
- QLIKE: SPA statistic 0.000 (largest studentized mean d at bar:k3), p consistent / lower / upper = 0.732 / 0.281 / 0.732 (mean block 10); block 5: 0.718 / 0.289 / 0.718; block 21: 0.744 / 0.264 / 0.744. Reality Check p = 0.914 (block 10); block 5 0.904, block 21 0.929.
  Holm-Bonferroni, one-sided DM p (automatic lag), adjusted: k = 1 1.000; k = 2 1.000; k = 3 1.000; k = 4 1.000; k = 5 1.000; k = 6 1.000; k = 7 1.000; k = 8 1.000; k = 9 1.000; k = 10 1.000; k = 11 1.000; k = 12 1.000; k = 13 1.000; smallest adjusted 1.000.
- MSE: SPA statistic 1.075 (largest studentized mean d at bar:k5), p consistent / lower / upper = 0.383 / 0.236 / 0.383 (mean block 10); block 5: 0.270 / 0.223 / 0.270; block 21: 0.456 / 0.229 / 0.456. Reality Check p = 0.509 (block 10); block 5 0.507, block 21 0.523.
  Holm-Bonferroni, one-sided DM p (automatic lag), adjusted: k = 1 1.000; k = 2 1.000; k = 3 1.000; k = 4 1.000; k = 5 1.000; k = 6 1.000; k = 7 1.000; k = 8 1.000; k = 9 1.000; k = 10 1.000; k = 11 1.000; k = 12 1.000; k = 13 1.000; smallest adjusted 1.000.

Model confidence set over {ridge_bb0, lasso_bb0, every tree pool} (T_max rule of `src/evaluation/model_confidence_set.py`, stationary bootstrap, mean block 10):

- QLIKE, alpha 0.10: 15 of 15 in the set: ridge_bb0 (p 0.201), lasso_bb0 (p 0.201), k = 3 (p 0.201), k = 5 (p 0.201), k = 4 (p 0.201), k = 6 (p 0.201), k = 2 (p 0.201), k = 7 (p 0.201), k = 8 (p 0.201), k = 1 (p 0.201), k = 10 (p 0.201), k = 11 (p 0.201), k = 12 (p 0.201), k = 9 (p 0.201), k = 13 (p 0.201).
- QLIKE, alpha 0.25: 13 of 15 in the set: ridge_bb0 (p 0.407), lasso_bb0 (p 0.407), k = 3 (p 0.407), k = 5 (p 0.407), k = 4 (p 0.407), k = 6 (p 0.407), k = 2 (p 0.407), k = 7 (p 0.407), k = 8 (p 0.407), k = 1 (p 0.407), k = 10 (p 0.407), k = 11 (p 0.407), k = 12 (p 0.407); eliminated in order: k = 13 (p 0.201), k = 9 (p 0.221).
- MSE, alpha 0.10: 15 of 15 in the set: k = 6 (p 0.602), k = 5 (p 0.602), k = 7 (p 0.602), k = 10 (p 0.602), k = 12 (p 0.602), k = 11 (p 0.602), k = 8 (p 0.602), lasso_bb0 (p 0.602), k = 4 (p 0.602), k = 3 (p 0.602), k = 9 (p 0.602), k = 13 (p 0.602), ridge_bb0 (p 0.602), k = 2 (p 0.602), k = 1 (p 0.602).
- MSE, alpha 0.25: 15 of 15 in the set: k = 6 (p 0.602), k = 5 (p 0.602), k = 7 (p 0.602), k = 10 (p 0.602), k = 12 (p 0.602), k = 11 (p 0.602), k = 8 (p 0.602), lasso_bb0 (p 0.602), k = 4 (p 0.602), k = 3 (p 0.602), k = 9 (p 0.602), k = 13 (p 0.602), ridge_bb0 (p 0.602), k = 2 (p 0.602), k = 1 (p 0.602).

## 4. Encompassing
k* = 1: lowest seed-mean QLIKE on the 189 16:00 rows 2019-04-04 .. 2020-01-02 (2019, before the trade days; the recalibration needs 63 earlier errors): k = 1 0.1626; k = 2 0.1695; k = 3 0.1767; k = 4 0.1722; k = 5 0.1726; k = 6 0.1715; k = 7 0.1743; k = 8 0.1735; k = 9 0.1768; k = 10 0.1735; k = 11 0.1761; k = 12 0.1756; k = 13 0.1732. The QLIKE of the seed-averaged forecast on the same rows is lowest at k = 1.

y = realized 16:00 variance, f = recalibrated forecasts (pred_clock), f_L = ridge_bb0, f_T = tree; HAC lag 14.

| tree forecast | joint MZ b_L (t of 0) | b_T (t of 0) | Wald (a, b_L, b_T) = (0, 1, 0) p | weighted MZ b_L (t) | b_T (t) | Wald p | lambda (se) | t lambda = 0 | t lambda = 1 | QLIKE moment t | reverse moment t |
|---|---|---|---|---|---|---|---|---|---|---|---|
| k* = 1 | -0.095 (-0.70) | 0.643 (6.04) | 3.8e-18 | 0.591 (6.04) | 0.319 (3.08) | 5.0e-07 | 0.041 (0.125) | 0.33 | -7.67 | -1.64 | -4.50 |
| average of all pools | -0.281 (-2.58) | 0.938 (8.45) | 4.3e-42 | 0.475 (4.12) | 0.443 (3.59) | 9.3e-08 | 0.754 (0.127) | 5.93 | -1.93 | -2.18 | -4.36 |
| k = 2 (not chosen in advance) | -0.163 (-1.11) | 0.701 (6.52) | 1.2e-17 | 0.515 (4.51) | 0.393 (3.28) | 1.9e-07 | -0.003 (0.158) | -0.02 | -6.34 | -2.01 | -4.76 |
| k = 3 (not chosen in advance) | -0.107 (-1.07) | 0.788 (8.89) | 7.5e-27 | 0.475 (4.81) | 0.440 (4.15) | 2.6e-09 | 0.720 (0.119) | 6.06 | -2.36 | -2.36 | -3.92 |
| k = 4 (not chosen in advance) | -0.267 (-2.25) | 0.928 (7.59) | 5.8e-36 | 0.501 (4.60) | 0.417 (3.54) | 4.9e-08 | 0.772 (0.135) | 5.73 | -1.69 | -2.23 | -4.54 |
| k = 5 (not chosen in advance) | -0.242 (-2.13) | 0.966 (7.52) | 3.5e-36 | 0.482 (4.41) | 0.438 (3.69) | 2.5e-08 | 1.092 (0.172) | 6.35 | 0.53 | -2.39 | -4.32 |
| k = 6 (not chosen in advance) | -0.222 (-1.81) | 0.975 (6.78) | 7.9e-36 | 0.498 (4.62) | 0.420 (3.62) | 5.9e-08 | 1.181 (0.198) | 5.96 | 0.92 | -2.42 | -5.25 |
| k = 7 (not chosen in advance) | -0.256 (-2.18) | 0.955 (7.02) | 2.1e-47 | 0.498 (4.34) | 0.422 (3.42) | 2.9e-07 | 0.976 (0.157) | 6.23 | -0.15 | -2.22 | -4.54 |
| k = 8 (not chosen in advance) | -0.232 (-2.34) | 0.913 (8.66) | 4.7e-48 | 0.498 (4.45) | 0.423 (3.53) | 2.4e-07 | 0.846 (0.120) | 7.06 | -1.28 | -2.23 | -4.40 |
| k = 9 (not chosen in advance) | -0.208 (-2.31) | 0.864 (10.36) | 9.2e-46 | 0.519 (4.69) | 0.398 (3.43) | 1.3e-06 | 0.669 (0.125) | 5.35 | -2.64 | -2.15 | -4.41 |
| k = 10 (not chosen in advance) | -0.221 (-2.39) | 0.920 (9.87) | 1.4e-43 | 0.511 (4.45) | 0.408 (3.35) | 7.0e-07 | 0.941 (0.120) | 7.87 | -0.49 | -2.20 | -4.56 |
| k = 11 (not chosen in advance) | -0.238 (-2.47) | 0.923 (9.20) | 1.0e-47 | 0.516 (4.36) | 0.403 (3.19) | 1.1e-06 | 0.875 (0.122) | 7.15 | -1.02 | -2.19 | -4.48 |
| k = 12 (not chosen in advance) | -0.328 (-2.96) | 0.996 (8.29) | 4.3e-68 | 0.518 (3.97) | 0.400 (2.90) | 4.0e-06 | 0.887 (0.119) | 7.43 | -0.94 | -2.05 | -4.43 |
| k = 13 (not chosen in advance) | -0.263 (-2.50) | 0.897 (7.49) | 8.1e-95 | 0.538 (4.24) | 0.378 (2.84) | 9.6e-06 | 0.625 (0.148) | 4.24 | -2.54 | -1.91 | -5.04 |

QLIKE moment t < 0: moving from f_L toward f_T lowers QLIKE at lambda = 0 (f_L does not encompass f_T); reverse moment t < 0: moving from f_T toward f_L lowers it.

Combinations against the ridge (DM, automatic lag; positive = combination lower loss):

| tree forecast | combination | sample | loss | mean ridge | mean combination | DM | p one | lambda first / median / last (min .. max) |
|---|---|---|---|---|---|---|---|---|
| k* = 1 | real-time lambda (QLIKE on past trade days, expanding) | trade days 251 .. 866 | qlike | 0.0842 | 0.0832 | 1.47 | 0.070 | 0.221 / 0.239 / 0.260 (0.209 .. 0.274) |
| k* = 1 | real-time lambda (QLIKE on past trade days, expanding) | trade days 251 .. 866 | mse | 1.477e-11 | 1.454e-11 | 0.86 | 0.195 | 0.221 / 0.239 / 0.260 (0.209 .. 0.274) |
| k* = 1 | equal weights, variance level | trade days 251 .. 866 | qlike | 0.0842 | 0.0832 | 0.73 | 0.232 | 0.5 |
| k* = 1 | equal weights, variance level | trade days 251 .. 866 | mse | 1.477e-11 | 1.463e-11 | 0.22 | 0.412 | 0.5 |
| k* = 1 | equal weights, variance level | all 866 trade days | qlike | 0.0946 | 0.0937 | 0.45 | 0.326 | 0.5 |
| k* = 1 | equal weights, variance level | all 866 trade days | mse | 7.915e-10 | 8.772e-10 | -0.82 | 0.794 | 0.5 |
| k* = 1 | equal weights, adjusted scale, rescored | trade days 251 .. 866 | qlike | 0.0842 | 0.0831 | 0.88 | 0.191 | 0.5 |
| k* = 1 | equal weights, adjusted scale, rescored | trade days 251 .. 866 | mse | 1.477e-11 | 1.457e-11 | 0.32 | 0.376 | 0.5 |
| k* = 1 | equal weights, adjusted scale, rescored | all 866 trade days | qlike | 0.0946 | 0.0931 | 0.80 | 0.213 | 0.5 |
| k* = 1 | equal weights, adjusted scale, rescored | all 866 trade days | mse | 7.915e-10 | 8.654e-10 | -0.75 | 0.772 | 0.5 |
| average of all pools | real-time lambda (QLIKE on past trade days, expanding) | trade days 251 .. 866 | qlike | 0.0842 | 0.0821 | 1.67 | 0.048 | 0.343 / 0.400 / 0.397 (0.338 .. 0.457) |
| average of all pools | real-time lambda (QLIKE on past trade days, expanding) | trade days 251 .. 866 | mse | 1.477e-11 | 1.507e-11 | -0.28 | 0.609 | 0.343 / 0.400 / 0.397 (0.338 .. 0.457) |
| average of all pools | equal weights, variance level | trade days 251 .. 866 | qlike | 0.0842 | 0.0819 | 1.55 | 0.060 | 0.5 |
| average of all pools | equal weights, variance level | trade days 251 .. 866 | mse | 1.477e-11 | 1.550e-11 | -0.46 | 0.676 | 0.5 |
| average of all pools | equal weights, variance level | all 866 trade days | qlike | 0.0946 | 0.0918 | 1.42 | 0.077 | 0.5 |
| average of all pools | equal weights, variance level | all 866 trade days | mse | 7.915e-10 | 7.005e-10 | 1.25 | 0.105 | 0.5 |
| average of all pools | equal weights, adjusted scale, rescored | trade days 251 .. 866 | qlike | 0.0842 | 0.0820 | 1.45 | 0.073 | 0.5 |
| average of all pools | equal weights, adjusted scale, rescored | trade days 251 .. 866 | mse | 1.477e-11 | 1.541e-11 | -0.41 | 0.660 | 0.5 |
| average of all pools | equal weights, adjusted scale, rescored | all 866 trade days | qlike | 0.0946 | 0.0916 | 1.56 | 0.060 | 0.5 |
| average of all pools | equal weights, adjusted scale, rescored | all 866 trade days | mse | 7.915e-10 | 6.988e-10 | 1.28 | 0.100 | 0.5 |

## 5. Stability (QLIKE, against the ridge)
Giacomini-Rossi fluctuation test, window 260 trade days (mu = 0.3), statistic = rolling sum of d / (full-sample HAC sd x sqrt(window)); simulated critical values (50000 draws): two-sided 5% 3.082 / 10% 2.813, one-sided 5% 2.821 / 10% 2.531.

| k | max (window end) | min (window end) | sup abs > two-sided 5% | max > one-sided 5% |
|---|---|---|---|---|
| 1 | 0.02 (2023-03-28) | -2.74 (2021-11-15) | False | False |
| 2 | 0.61 (2023-03-31) | -2.32 (2021-11-30) | False | False |
| 3 | 0.83 (2023-04-03) | -1.59 (2024-04-17) | False | False |
| 4 | 0.51 (2023-04-03) | -1.76 (2024-04-17) | False | False |
| 5 | 0.62 (2023-02-22) | -1.75 (2021-11-30) | False | False |
| 6 | 0.28 (2023-03-30) | -2.12 (2021-11-30) | False | False |
| 7 | 0.19 (2023-02-22) | -1.93 (2021-11-30) | False | False |
| 8 | 0.26 (2023-03-30) | -1.95 (2021-11-30) | False | False |
| 9 | 0.20 (2023-05-11) | -2.07 (2021-11-30) | False | False |
| 10 | 0.21 (2023-03-30) | -2.03 (2021-11-30) | False | False |
| 11 | 0.19 (2023-05-11) | -2.09 (2021-11-30) | False | False |
| 12 | 0.19 (2023-10-25) | -2.17 (2021-11-30) | False | False |
| 13 | 0.04 (2023-09-21) | -2.37 (2021-11-30) | False | False |
| average of all pools | 0.47 (2023-04-03) | -1.91 (2021-11-30) | False | False |

DM (automatic lag) in subsets; n days in the header:

| k | all 866 trade days (866) | 2020-2021 (316) | 2022-2024 (550) | previous-day VIX at or above its median (408) | previous-day VIX below its median (405) | previous trade day implied variance at or above its median (433) | previous trade day implied variance below its median (432) | FOMC days excluded (flags end 2023-11-01) (837) | early-close days excluded (866) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | -1.52 | -1.30 | -0.90 | -1.40 | -0.47 | -0.82 | -1.71 | -1.62 | -1.52 |
| 2 | -1.14 | -1.07 | -0.53 | -0.75 | -0.36 | -0.29 | -1.66 | -0.96 | -1.14 |
| 3 | -0.65 | -0.45 | -0.49 | -0.12 | 0.05 | 0.46 | -1.42 | -0.52 | -0.65 |
| 4 | -0.92 | -0.44 | -0.89 | -0.34 | -0.22 | 0.30 | -1.71 | -0.83 | -0.92 |
| 5 | -0.79 | -0.51 | -0.62 | -0.17 | -0.03 | 0.50 | -1.63 | -0.72 | -0.79 |
| 6 | -1.01 | -0.67 | -0.78 | -0.47 | -0.25 | 0.25 | -1.79 | -0.78 | -1.01 |
| 7 | -1.17 | -0.66 | -0.98 | -0.51 | -0.26 | 0.16 | -1.74 | -1.02 | -1.17 |
| 8 | -1.19 | -0.74 | -0.95 | -0.48 | -0.48 | 0.20 | -1.72 | -1.02 | -1.19 |
| 9 | -1.56 | -1.00 | -1.20 | -0.72 | -1.01 | 0.02 | -1.97 | -1.32 | -1.56 |
| 10 | -1.38 | -0.88 | -1.07 | -0.66 | -0.69 | 0.06 | -1.81 | -1.07 | -1.38 |
| 11 | -1.38 | -0.97 | -0.99 | -0.71 | -0.65 | 0.01 | -1.77 | -1.10 | -1.38 |
| 12 | -1.39 | -0.94 | -1.03 | -0.77 | -0.62 | 0.02 | -1.83 | -1.08 | -1.39 |
| 13 | -1.61 | -1.06 | -1.22 | -0.96 | -0.89 | -0.26 | -1.91 | -1.18 | -1.61 |
| average of all pools | -0.81 | -0.59 | -0.57 | -0.36 | 0.10 | 0.35 | -1.51 | -0.62 | -0.81 |

Mean QLIKE of the ridge in each subset: all 866 trade days 0.0946; 2020-2021 0.1118; 2022-2024 0.0847; previous-day VIX at or above its median 0.1010; previous-day VIX below its median 0.0841; previous trade day implied variance at or above its median 0.0925; previous trade day implied variance below its median 0.0969; FOMC days excluded (flags end 2023-11-01) 0.0951; early-close days excluded 0.0946.

Difference in mean d between groups (d on a constant and the group dummy, HAC t of the dummy): 2022-2024 vs 2020-2021: k = 1 0.88, k = 2 0.73, k = 3 0.12, k = 4 -0.04, k = 5 0.11, k = 6 0.21, k = 7 0.04, k = 8 0.12, k = 9 0.13, k = 10 0.14, k = 11 0.26, k = 12 0.24, k = 13 0.29, k = average of all pools 0.22; VIX high vs low: k = 1 -0.98, k = 2 -0.49, k = 3 -0.13, k = 4 -0.20, k = 5 -0.13, k = 6 -0.30, k = 7 -0.33, k = 8 -0.20, k = 9 -0.16, k = 10 -0.26, k = 11 -0.32, k = 12 -0.39, k = 13 -0.43, k = average of all pools -0.36; implied variance high vs low: k = 1 0.12, k = 2 0.64, k = 3 1.27, k = 4 1.29, k = 5 1.43, k = 6 1.33, k = 7 1.33, k = 8 1.39, k = 9 1.54, k = 10 1.42, k = 11 1.36, k = 12 1.39, k = 13 1.26, k = average of all pools 1.26; FOMC day vs other days: k = 1 1.28, k = 2 -1.33, k = 3 -0.81, k = 4 -0.58, k = 5 -0.40, k = 6 -1.29, k = 7 -1.08, k = 8 -1.34, k = 9 -1.41, k = 10 -1.65, k = 11 -1.57, k = 12 -1.65, k = 13 -2.17, k = average of all pools -1.19.

FOMC flags in `data/releases.parquet` stop at 2023-11-01, so FOMC days after that date stay in the 'FOMC days excluded' subset; 29 flagged FOMC days are trade days. Early-close days among the trade days: 0 (the 15:30 straddle deck has none), so that subset equals the full sample. VIX median split at 20.82 on the 813 trade days whose previous session has a VIX value (`data/vix_and_voldemand.parquet` ends 2024-02-12; last such trade day 2024-02-13); the implied-variance split uses the deck's iv_var of the previous trade day (median 5.678e-06) on trade days 2 .. 866.

## 6. Calibration: Mincer-Zarnowitz on the variance level, y = a + b f (HAC)

| forecast | a | b (se) | t of b = 1 | Wald (a, b) = (0, 1) p | R^2 | weighted b (t of b = 1) | weighted Wald p |
|---|---|---|---|---|---|---|---|
| ridge_bb0 | 2.63e-06 | 0.693 (0.048) | -6.41 | 5.1e-10 | 0.668 | 0.920 (-2.91) | 0.013 |
| lasso_bb0 | 1.99e-06 | 0.733 (0.057) | -4.67 | 1.1e-05 | 0.703 | 0.912 (-3.18) | 0.004 |
| ridge_single | 4.71e-06 | 0.496 (0.024) | -20.82 | 8.3e-102 | 0.663 | 0.941 (-2.05) | 0.123 |
| lasso_single | 3.47e-06 | 0.596 (0.031) | -13.14 | 2.6e-38 | 0.699 | 0.927 (-2.58) | 0.024 |
| k = 1 | 2.95e-06 | 0.571 (0.038) | -11.20 | 2.9e-41 | 0.736 | 0.907 (-3.03) | 0.002 |
| k = 2 | 3.14e-06 | 0.576 (0.030) | -14.15 | 2.9e-46 | 0.732 | 0.895 (-3.52) | 0.002 |
| k = 3 | 2.02e-06 | 0.691 (0.048) | -6.40 | 3.7e-13 | 0.736 | 0.911 (-2.93) | 0.011 |
| k = 4 | 2.07e-06 | 0.688 (0.042) | -7.45 | 9.6e-18 | 0.753 | 0.915 (-2.71) | 0.011 |
| k = 5 | 1.70e-06 | 0.732 (0.048) | -5.63 | 1.1e-10 | 0.748 | 0.918 (-2.67) | 0.011 |
| k = 6 | 1.44e-06 | 0.755 (0.047) | -5.17 | 1.3e-09 | 0.752 | 0.916 (-2.73) | 0.007 |
| k = 7 | 1.87e-06 | 0.715 (0.046) | -6.19 | 5.5e-12 | 0.753 | 0.919 (-2.56) | 0.008 |
| k = 8 | 2.06e-06 | 0.699 (0.042) | -7.12 | 1.1e-14 | 0.749 | 0.921 (-2.52) | 0.009 |
| k = 9 | 2.25e-06 | 0.678 (0.040) | -8.11 | 1.4e-17 | 0.742 | 0.912 (-2.80) | 0.003 |
| k = 10 | 1.90e-06 | 0.714 (0.041) | -7.00 | 3.2e-14 | 0.751 | 0.916 (-2.70) | 0.004 |
| k = 11 | 2.00e-06 | 0.703 (0.040) | -7.51 | 6.6e-16 | 0.750 | 0.918 (-2.64) | 0.004 |
| k = 12 | 2.01e-06 | 0.699 (0.039) | -7.80 | 5.3e-16 | 0.770 | 0.916 (-2.71) | 0.003 |
| k = 13 | 2.31e-06 | 0.668 (0.041) | -8.00 | 1.3e-15 | 0.757 | 0.914 (-2.80) | 0.002 |
| average of all pools | 2.12e-06 | 0.685 (0.041) | -7.71 | 3.4e-17 | 0.752 | 0.916 (-2.75) | 0.010 |

## 7. Seed spread and the curve of seed-averaged forecasts (QLIKE)

| k | seeds | QLIKE for each seed | seed mean (sd) | QLIKE of the seed average | step from the previous k | sd of the two ends | step exceeds sd | DM of the step |
|---|---|---|---|---|---|---|---|---|
| 1 | 42/43/44 | 0.1007, 0.1000, 0.0997 | 0.1001 (0.0005) | 0.0996 |  |  |  |  |
| 2 | 42/43/44 | 0.0975, 0.0982, 0.0999 | 0.0986 (0.0012) | 0.0982 | -0.0014 (k = 1) | 0.0009 | True | 0.75 |
| 3 | 42/43/44 | 0.0973, 0.0972, 0.0970 | 0.0971 (0.0002) | 0.0969 | -0.0014 (k = 2) | 0.0009 | True | 1.16 |
| 4 | 42/43/44 | 0.0979, 0.0988, 0.0979 | 0.0982 (0.0005) | 0.0980 | +0.0011 (k = 3) | 0.0004 | True | -1.23 |
| 5 | 42/43/44 | 0.0975, 0.0978, 0.0977 | 0.0977 (0.0002) | 0.0975 | -0.0005 (k = 4) | 0.0004 | True | 0.63 |
| 6 | 42/43/44 | 0.0989, 0.0978, 0.0985 | 0.0984 (0.0005) | 0.0982 | +0.0007 (k = 5) | 0.0004 | True | -1.17 |
| 7 | 42/43/44 | 0.1001, 0.0988, 0.0995 | 0.0994 (0.0006) | 0.0992 | +0.0010 (k = 6) | 0.0006 | True | -1.23 |
| 8 | 42/43/44 | 0.0996, 0.0996, 0.0995 | 0.0996 (0.0001) | 0.0994 | +0.0002 (k = 7) | 0.0005 | False | -0.29 |
| 9 | 42/43/44 | 0.1008, 0.1012, 0.1018 | 0.1013 (0.0005) | 0.1011 | +0.0017 (k = 8) | 0.0004 | True | -2.50 |
| 10 | 42/43/44 | 0.1004, 0.1001, 0.1012 | 0.1005 (0.0006) | 0.1004 | -0.0007 (k = 9) | 0.0005 | True | 1.29 |
| 11 | 42/43/44 | 0.1006, 0.1004, 0.1009 | 0.1007 (0.0002) | 0.1005 | +0.0001 (k = 10) | 0.0004 | False | -0.34 |
| 12 | 42/43/44 | 0.1011, 0.1012, 0.1009 | 0.1011 (0.0001) | 0.1009 | +0.0004 (k = 11) | 0.0002 | True | -0.73 |
| 13 | 42/43/44 | 0.1015, 0.1022, 0.1011 | 0.1016 (0.0005) | 0.1015 | +0.0005 (k = 12) | 0.0004 | True | -0.91 |

sd = sample sd over seeds (ddof 1); sd of the two ends = root mean square of the two k's seed sd's (one end when the other has one seed); DM of the step: QLIKE of the previous k's seed average minus this k's (positive = this k lower).

## 8. Use of the bar column (bar_end_minute)
From `results/close_studies_2026-10-03/trees_kfull/bar_column_by_k.csv` (the other agent's record, written by `experiments/close_trees_kfull.py`; copied to bar_column.csv): at every refit the booster's split count and gain of each column; share = the column's splits (gain) over all splits (gain) of that fit; rank 1 = most gain among the columns the fit kept; over the seeds and refits of each k.

| family | k | column | arms | refits | % refits with a split | mean splits | split share | gain share mean (min .. max) | gain rank median (best .. worst) | columns kept (median) |
|---|---|---|---|---|---|---|---|---|---|---|
| main | 2 | bar_end_minute | 3 | 402 | 46.3 | 0.7 | 0.0001 | 0.0000 (0.0000 .. 0.0002) | 333 (288 .. 347) | 375 |
| main | 3 | bar_end_minute | 3 | 402 | 64.9 | 1.3 | 0.0002 | 0.0000 (0.0000 .. 0.0004) | 332 (175 .. 349) | 376 |
| main | 4 | bar_end_minute | 3 | 402 | 95.0 | 4.3 | 0.0006 | 0.0002 (0.0000 .. 0.0009) | 302 (69 .. 351) | 377 |
| main | 5 | bar_end_minute | 3 | 402 | 100.0 | 9.3 | 0.0013 | 0.0003 (0.0000 .. 0.0008) | 204 (70 .. 338) | 378 |
| main | 6 | bar_end_minute | 3 | 402 | 99.5 | 7.4 | 0.0010 | 0.0002 (0.0000 .. 0.0007) | 228 (59 .. 351) | 379 |
| main | 7 | bar_end_minute | 3 | 402 | 99.5 | 6.6 | 0.0009 | 0.0002 (0.0000 .. 0.0007) | 234 (55 .. 352) | 379 |
| main | 8 | bar_end_minute | 3 | 402 | 99.0 | 5.8 | 0.0008 | 0.0001 (0.0000 .. 0.0006) | 252 (61 .. 350) | 379 |
| main | 9 | bar_end_minute | 3 | 402 | 99.5 | 5.5 | 0.0007 | 0.0001 (0.0000 .. 0.0006) | 258 (57 .. 352) | 380 |
| main | 10 | bar_end_minute | 3 | 402 | 100.0 | 6.3 | 0.0008 | 0.0001 (0.0000 .. 0.0004) | 238 (86 .. 350) | 381 |
| main | 11 | bar_end_minute | 3 | 402 | 99.8 | 6.8 | 0.0009 | 0.0001 (0.0000 .. 0.0004) | 224 (71 .. 356) | 381 |
| main | 12 | bar_end_minute | 3 | 402 | 99.0 | 7.6 | 0.0010 | 0.0001 (0.0000 .. 0.0004) | 202 (63 .. 357) | 381 |
| main | 13 | bar_end_minute | 3 | 402 | 100.0 | 36.9 | 0.0050 | 0.0008 (0.0005 .. 0.0013) | 40 (31 .. 59) | 382 |
| main | 1 | hour | 1 | 134 | 0.0 | 0.0 | 0.0000 | 0.0000 (0.0000 .. 0.0000) |  ( .. ) | 369 |
| main | 2 | hour | 3 | 402 | 98.5 | 21.5 | 0.0029 | 0.0009 (0.0000 .. 0.0028) | 82 (25 .. 337) | 375 |
| main | 3 | hour | 3 | 402 | 89.6 | 2.9 | 0.0004 | 0.0001 (0.0000 .. 0.0005) | 319 (138 .. 349) | 376 |
| main | 4 | hour | 3 | 402 | 100.0 | 26.6 | 0.0036 | 0.0012 (0.0001 .. 0.0033) | 60 (26 .. 313) | 377 |
| main | 5 | hour | 3 | 402 | 98.0 | 6.2 | 0.0008 | 0.0002 (0.0000 .. 0.0008) | 290 (64 .. 349) | 378 |
| main | 6 | hour | 3 | 402 | 90.8 | 4.6 | 0.0006 | 0.0001 (0.0000 .. 0.0006) | 306 (82 .. 346) | 379 |
| main | 7 | hour | 3 | 402 | 83.8 | 3.2 | 0.0004 | 0.0001 (0.0000 .. 0.0004) | 317 (98 .. 350) | 379 |
| main | 8 | hour | 3 | 402 | 86.3 | 3.0 | 0.0004 | 0.0001 (0.0000 .. 0.0004) | 319 (103 .. 356) | 379 |
| main | 9 | hour | 3 | 402 | 84.8 | 2.9 | 0.0004 | 0.0001 (0.0000 .. 0.0004) | 321 (95 .. 354) | 380 |
| main | 10 | hour | 3 | 402 | 92.3 | 3.3 | 0.0004 | 0.0001 (0.0000 .. 0.0003) | 318 (123 .. 354) | 381 |
| main | 11 | hour | 3 | 402 | 98.8 | 4.5 | 0.0006 | 0.0001 (0.0000 .. 0.0003) | 294 (94 .. 354) | 381 |
| main | 12 | hour | 3 | 402 | 99.0 | 5.0 | 0.0007 | 0.0001 (0.0000 .. 0.0004) | 278 (80 .. 346) | 381 |
| main | 13 | hour | 3 | 402 | 100.0 | 12.0 | 0.0016 | 0.0003 (0.0001 .. 0.0005) | 110 (52 .. 308) | 382 |

bar_end_minute: split on in 46.3 .. 100.0 % of the refits across k = 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13; no k without a split. 


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

## Other family
- `nobar` (LightGBM, last k bars, no bar column), QLIKE, DM against the ridge (automatic lag): k = 1 0.0996 / -1.52; k = 2 0.0982 / -1.15; k = 3 0.0976 / -0.82; k = 4 0.0976 / -0.85; k = 5 0.0984 / -1.07; k = 7 0.0996 / -1.28; k = 13 0.1016 / -1.67; k = average 0.0968 / -0.65. SPA p consistent / lower / upper 0.724 / 0.244 / 0.724; k* = 1.

## Files
- `gate.csv`, `forecasts.csv` (every scored forecast, file, QLIKE, MSE), `kstar_selection.csv`, `dm.csv`, `fixedb_critical_values.csv`, `gw.csv`, `multiple_comparisons.csv`, `mcs.csv`, `encompassing.csv`, `realtime_combination.csv`, `realtime_lambda_path.csv`, `stability.csv`, `fluctuation.csv`, `fluctuation_path.csv`, `fluctuation_critical_values.csv`, `calibration.csv`, `seed_spread.csv`, `seed_curve.csv`, `bar_column.csv`, `within_day.csv`, `run_info.json`.
- CPU time of the run: 0.3 min (one process), wall 0.4 min.
