# The 16:00 tree cadence ladder (checklist I2)

Written by `experiments/trees_cadence_ladder_1600.py` from its own CSVs in this folder (`ladder_levels.csv`, `ladder_pairs.csv`, `ladder_wide.csv`, `ladder_gates.csv`, `cluster_usage.csv`). Every number below is read from them.

## What is scored

Per model (LightGBM, XGBoost, random forest) and input set (`baseline` = HAR ladder `har_ma_*` + calendar; `live_feasible`; `all_features`), the forecast of the 15:30-16:00 bar's realized variance (row stamped 16:00, issued at 15:30), fitted on the per-bar design without the session-edge duplicates (commit 47f7f9c) with a 2000-session window:

| rung | configuration | refit | table |
|---|---|---|---|
| T10 | shipped configuration | every 10 sessions | `yhat_subtree_<model>_<bucket>` |
| T1 | shipped configuration | every session | `yhat_subtree_daily_<bucket>_<model>` |
| RS10 | random search, MSE rule | every 10 sessions | `yhat_subtree_tuned_<bucket>_<model>` |
| RS1 | random search, MSE rule | every session | `yhat_subtree_tuned_daily_<bucket>_<model>` |
| RS10q | random search, QLIKE rule | every 10 sessions | `yhat_subtree_tunedq_<bucket>_<model>` |
| RS1q | random search, QLIKE rule | every session | `yhat_subtree_tunedq_daily_<bucket>_<model>` |
| ridge | per-bar ridge, penalty re-chosen every 250 sessions | every session | `yhat_sub_ridge_<bucket>` |

Random search = 32 fixed candidates of each model's grid, re-chosen every 250 sessions on the 125-session validation tail of the current window (25-session embargo), by validation MSE (RS10, RS1) or validation QLIKE (RS10q, RS1q; supplementary). RS1 ran without TreeSHAP (the tuned workers record native importance at every refit); T1 records native importance and TreeSHAP at every 10th refit, T10 at every refit.

Per-bar ridge design used here: `baseline` dedup design, `live_feasible` dedup design, `all_features` dedup design.

## Scorer and days

The research scorer (the 16:00 bar recalibrated on its own; the master table's loader, target and trade imported): forecast = (f^2 + s) B with s = the forecast's own mean squared adjusted-scale error at 16:00 over the previous 250 sessions (at least 63), lagged one session; QLIKE against the per-bar spec's 16:00 target (lower is better). The last-30-min trade is the sign(s) rule on the straddle (nearest out-of-the-money call + nearest out-of-the-money put, same-day expiry, one position): buy when the recalibrated forecast exceeds the 15:30 implied variance, sell otherwise; annualized Sharpe at mid and at crossed fills. Every number is on the same 866 trade days (2020-01-03 .. 2024-04-30). Intervals: 95 % day-block bootstrap (21-day circular blocks, 2000 draws), paired (the same resampled days for both forecasts). Differences are a - b.

## Levels

**`baseline`**

| forecast | QLIKE | Sharpe mid | Sharpe crossed | % buy |
|---|---|---|---|---|
| LightGBM T10 | 0.1044 | 1.00 | 0.53 | 42.0 |
| LightGBM T1 | 0.1041 | 1.24 | 0.77 | 41.6 |
| LightGBM RS10 | 0.1117 | 0.91 | 0.44 | 42.3 |
| LightGBM RS1 | 0.1118 | 0.75 | 0.28 | 42.4 |
| LightGBM RS10q | 0.1077 | 0.97 | 0.50 | 40.8 |
| LightGBM RS1q | 0.1078 | 0.85 | 0.38 | 40.8 |
| XGBoost T10 | 0.1038 | 0.93 | 0.46 | 40.4 |
| XGBoost T1 | 0.1030 | 1.02 | 0.55 | 40.2 |
| XGBoost RS10 | 0.1098 | 1.14 | 0.67 | 41.1 |
| XGBoost RS1 | 0.1095 | 1.06 | 0.59 | 41.5 |
| XGBoost RS10q | 0.1118 | 1.08 | 0.61 | 41.0 |
| XGBoost RS1q | 0.1114 | 1.00 | 0.53 | 41.2 |
| random forest T10 | 0.1046 | 0.90 | 0.42 | 41.7 |
| random forest T1 | 0.1037 | 1.10 | 0.63 | 41.5 |
| random forest RS10 | 0.1094 | 0.90 | 0.43 | 42.0 |
| random forest RS1 | 0.1080 | 0.92 | 0.45 | 41.1 |
| random forest RS10q | 0.1020 | 0.99 | 0.52 | 41.1 |
| random forest RS1q | 0.1010 | 1.01 | 0.54 | 40.6 |
| per-bar ridge (dedup design) | 0.0982 | 1.22 | 0.74 | 37.9 |

**`live_feasible`**

| forecast | QLIKE | Sharpe mid | Sharpe crossed | % buy |
|---|---|---|---|---|
| LightGBM T10 | 0.0981 | 1.76 | 1.29 | 38.3 |
| LightGBM T1 | 0.0965 | 1.48 | 1.01 | 37.4 |
| LightGBM RS10 | 0.1001 | 1.59 | 1.12 | 39.0 |
| LightGBM RS1 | 0.0995 | 1.62 | 1.16 | 37.8 |
| LightGBM RS10q | 0.1025 | 1.41 | 0.94 | 39.1 |
| LightGBM RS1q | 0.1015 | 1.50 | 1.04 | 38.3 |
| XGBoost T10 | 0.0991 | 1.32 | 0.85 | 37.9 |
| XGBoost T1 | 0.0973 | 1.89 | 1.42 | 37.8 |
| XGBoost RS10 | 0.1082 | 1.30 | 0.84 | 37.9 |
| XGBoost RS1 | 0.1062 | 1.35 | 0.88 | 37.5 |
| XGBoost RS10q | 0.1076 | 1.41 | 0.94 | 39.0 |
| XGBoost RS1q | 0.1054 | 1.25 | 0.79 | 38.0 |
| random forest T10 | 0.1016 | 1.77 | 1.30 | 40.4 |
| random forest T1 | 0.0986 | 1.39 | 0.92 | 39.6 |
| random forest RS10 | 0.1055 | 1.05 | 0.58 | 39.8 |
| random forest RS1 | 0.1015 | 1.68 | 1.21 | 38.8 |
| random forest RS10q | 0.1062 | 1.35 | 0.87 | 40.9 |
| random forest RS1q | 0.1027 | 1.55 | 1.08 | 38.9 |
| per-bar ridge (dedup design) | 0.1005 | 1.90 | 1.44 | 40.2 |

**`all_features`**

| forecast | QLIKE | Sharpe mid | Sharpe crossed | % buy |
|---|---|---|---|---|
| LightGBM T10 | 0.1013 | 1.75 | 1.29 | 40.4 |
| LightGBM T1 | 0.0967 | 1.82 | 1.35 | 37.8 |
| LightGBM RS10 | 0.1007 | 1.61 | 1.15 | 39.0 |
| LightGBM RS1 | 0.1010 | 1.70 | 1.23 | 38.7 |
| LightGBM RS10q | 0.1015 | 1.05 | 0.58 | 39.1 |
| LightGBM RS1q | 0.0990 | 1.61 | 1.14 | 38.5 |
| XGBoost T10 | 0.1012 | 1.54 | 1.07 | 39.3 |
| XGBoost T1 | 0.0990 | 1.36 | 0.89 | 38.6 |
| XGBoost RS10 | 0.1049 | 1.62 | 1.15 | 38.0 |
| XGBoost RS1 | 0.1032 | 1.58 | 1.12 | 37.9 |
| XGBoost RS10q | 0.1053 | 1.18 | 0.71 | 39.0 |
| XGBoost RS1q | 0.1031 | 1.44 | 0.97 | 38.7 |
| random forest T10 | 0.1059 | 1.27 | 0.80 | 41.6 |
| random forest T1 | 0.1030 | 1.40 | 0.93 | 40.4 |
| random forest RS10 | 0.1093 | 1.23 | 0.77 | 40.5 |
| random forest RS1 | 0.1063 | 1.24 | 0.77 | 40.4 |
| random forest RS10q | 0.1067 | 1.20 | 0.74 | 40.3 |
| random forest RS1q | 0.1044 | 1.22 | 0.75 | 39.4 |
| per-bar ridge (dedup design) | 0.1004 | 1.66 | 1.20 | 36.1 |

## Paired differences (a - b) with 95 % intervals

**T1 - T10**

| model | input set | QLIKE diff (%) | QLIKE diff [interval] | Sharpe mid diff [interval] | Sharpe crossed diff [interval] | same side |
|---|---|---|---|---|---|---|
| LightGBM | `baseline` | -0.3 % | -0.0003 [-0.0019, +0.0011] | +0.25 [-0.02, +0.55] | +0.25 [-0.02, +0.55] | 97 % |
| XGBoost | `baseline` | -0.7 % | -0.0008 [-0.0018, +0.0001] | +0.09 [-0.15, +0.35] | +0.09 [-0.15, +0.36] | 99 % |
| random forest | `baseline` | -0.8 % | -0.0009 [-0.0031, +0.0014] | +0.20 [-0.51, +0.99] | +0.21 [-0.51, +0.99] | 94 % |
| LightGBM | `live_feasible` | -1.6 % | -0.0016 [-0.0051, +0.0009] | -0.28 [-1.03, +0.43] | -0.29 [-1.05, +0.44] | 94 % |
| XGBoost | `live_feasible` | -1.9 % | -0.0018 [-0.0039, -0.0002] | +0.57 [+0.12, +1.13] | +0.57 [+0.12, +1.13] | 97 % |
| random forest | `live_feasible` | -3.0 % | -0.0030 [-0.0065, -0.0001] | -0.37 [-0.97, +0.21] | -0.38 [-0.99, +0.21] | 94 % |
| LightGBM | `all_features` | -4.6 % | -0.0046 [-0.0084, -0.0017] | +0.07 [-0.54, +0.68] | +0.07 [-0.55, +0.68] | 92 % |
| XGBoost | `all_features` | -2.1 % | -0.0022 [-0.0060, +0.0003] | -0.18 [-0.67, +0.29] | -0.19 [-0.68, +0.29] | 96 % |
| random forest | `all_features` | -2.7 % | -0.0029 [-0.0071, +0.0006] | +0.13 [-0.33, +0.58] | +0.13 [-0.33, +0.58] | 93 % |

Intervals entirely below / above zero (9 rows): QLIKE 3 / 0; Sharpe mid 0 / 1; Sharpe crossed 0 / 1.

**RS10 - T10**

| model | input set | QLIKE diff (%) | QLIKE diff [interval] | Sharpe mid diff [interval] | Sharpe crossed diff [interval] | same side |
|---|---|---|---|---|---|---|
| LightGBM | `baseline` | +7.0 % | +0.0073 [+0.0019, +0.0143] | -0.08 [-0.61, +0.48] | -0.08 [-0.61, +0.48] | 92 % |
| XGBoost | `baseline` | +5.8 % | +0.0060 [+0.0003, +0.0148] | +0.21 [-0.32, +0.79] | +0.22 [-0.32, +0.79] | 95 % |
| random forest | `baseline` | +4.6 % | +0.0048 [-0.0020, +0.0144] | +0.01 [-1.05, +1.02] | +0.01 [-1.04, +1.03] | 91 % |
| LightGBM | `live_feasible` | +2.1 % | +0.0020 [-0.0012, +0.0058] | -0.17 [-0.89, +0.43] | -0.17 [-0.89, +0.44] | 92 % |
| XGBoost | `live_feasible` | +9.1 % | +0.0090 [+0.0017, +0.0187] | -0.02 [-0.63, +0.62] | -0.01 [-0.63, +0.63] | 92 % |
| random forest | `live_feasible` | +3.8 % | +0.0039 [+0.0001, +0.0082] | -0.71 [-1.60, +0.21] | -0.72 [-1.63, +0.21] | 94 % |
| LightGBM | `all_features` | -0.6 % | -0.0006 [-0.0044, +0.0035] | -0.14 [-0.66, +0.44] | -0.14 [-0.66, +0.44] | 91 % |
| XGBoost | `all_features` | +3.7 % | +0.0037 [-0.0004, +0.0089] | +0.08 [-0.38, +0.55] | +0.08 [-0.39, +0.55] | 92 % |
| random forest | `all_features` | +3.2 % | +0.0034 [-0.0011, +0.0085] | -0.04 [-0.68, +0.65] | -0.04 [-0.68, +0.65] | 90 % |

Intervals entirely below / above zero (9 rows): QLIKE 0 / 4; Sharpe mid 0 / 0; Sharpe crossed 0 / 0.

**RS1 - RS10**

| model | input set | QLIKE diff (%) | QLIKE diff [interval] | Sharpe mid diff [interval] | Sharpe crossed diff [interval] | same side |
|---|---|---|---|---|---|---|
| LightGBM | `baseline` | +0.1 % | +0.0001 [-0.0008, +0.0010] | -0.16 [-0.64, +0.22] | -0.16 [-0.64, +0.22] | 98 % |
| XGBoost | `baseline` | -0.4 % | -0.0004 [-0.0012, +0.0004] | -0.09 [-0.37, +0.22] | -0.09 [-0.37, +0.23] | 98 % |
| random forest | `baseline` | -1.2 % | -0.0014 [-0.0025, -0.0003] | +0.02 [-0.38, +0.41] | +0.02 [-0.38, +0.42] | 96 % |
| LightGBM | `live_feasible` | -0.7 % | -0.0007 [-0.0023, +0.0009] | +0.03 [-0.32, +0.41] | +0.03 [-0.32, +0.40] | 95 % |
| XGBoost | `live_feasible` | -1.9 % | -0.0020 [-0.0051, +0.0000] | +0.05 [-0.23, +0.34] | +0.05 [-0.23, +0.34] | 97 % |
| random forest | `live_feasible` | -3.8 % | -0.0040 [-0.0086, -0.0009] | +0.63 [-0.06, +1.37] | +0.63 [-0.06, +1.39] | 94 % |
| LightGBM | `all_features` | +0.3 % | +0.0003 [-0.0028, +0.0042] | +0.09 [-0.39, +0.56] | +0.09 [-0.40, +0.56] | 95 % |
| XGBoost | `all_features` | -1.6 % | -0.0017 [-0.0036, +0.0001] | -0.03 [-0.35, +0.29] | -0.03 [-0.35, +0.29] | 96 % |
| random forest | `all_features` | -2.7 % | -0.0029 [-0.0096, +0.0016] | +0.01 [-0.38, +0.41] | +0.01 [-0.39, +0.41] | 96 % |

Intervals entirely below / above zero (9 rows): QLIKE 2 / 0; Sharpe mid 0 / 0; Sharpe crossed 0 / 0.

**RS1 - T1**

| model | input set | QLIKE diff (%) | QLIKE diff [interval] | Sharpe mid diff [interval] | Sharpe crossed diff [interval] | same side |
|---|---|---|---|---|---|---|
| LightGBM | `baseline` | +7.5 % | +0.0078 [+0.0022, +0.0151] | -0.49 [-1.18, +0.18] | -0.49 [-1.19, +0.18] | 92 % |
| XGBoost | `baseline` | +6.2 % | +0.0064 [+0.0004, +0.0156] | +0.03 [-0.50, +0.52] | +0.04 [-0.50, +0.53] | 95 % |
| random forest | `baseline` | +4.2 % | +0.0043 [-0.0030, +0.0146] | -0.17 [-1.01, +0.63] | -0.17 [-1.01, +0.63] | 89 % |
| LightGBM | `live_feasible` | +3.1 % | +0.0030 [-0.0012, +0.0078] | +0.15 [-0.66, +0.96] | +0.15 [-0.66, +0.96] | 91 % |
| XGBoost | `live_feasible` | +9.1 % | +0.0088 [+0.0022, +0.0176] | -0.54 [-1.30, +0.18] | -0.54 [-1.29, +0.19] | 93 % |
| random forest | `live_feasible` | +3.0 % | +0.0029 [-0.0007, +0.0067] | +0.28 [-0.19, +0.76] | +0.29 [-0.19, +0.77] | 93 % |
| LightGBM | `all_features` | +4.5 % | +0.0043 [-0.0016, +0.0123] | -0.12 [-0.70, +0.45] | -0.12 [-0.70, +0.45] | 92 % |
| XGBoost | `all_features` | +4.3 % | +0.0042 [-0.0005, +0.0103] | +0.22 [-0.39, +0.84] | +0.23 [-0.39, +0.84] | 92 % |
| random forest | `all_features` | +3.2 % | +0.0033 [-0.0002, +0.0069] | -0.16 [-0.78, +0.47] | -0.16 [-0.79, +0.47] | 91 % |

Intervals entirely below / above zero (9 rows): QLIKE 0 / 3; Sharpe mid 0 / 0; Sharpe crossed 0 / 0.

**T10 - ridge**

| model | input set | QLIKE diff (%) | QLIKE diff [interval] | Sharpe mid diff [interval] | Sharpe crossed diff [interval] | same side |
|---|---|---|---|---|---|---|
| LightGBM | `baseline` | +6.3 % | +0.0062 [+0.0021, +0.0103] | -0.22 [-0.86, +0.39] | -0.22 [-0.87, +0.40] | 89 % |
| XGBoost | `baseline` | +5.7 % | +0.0056 [+0.0019, +0.0092] | -0.29 [-0.89, +0.32] | -0.29 [-0.89, +0.33] | 92 % |
| random forest | `baseline` | +6.4 % | +0.0063 [+0.0011, +0.0115] | -0.32 [-1.21, +0.49] | -0.32 [-1.21, +0.50] | 87 % |
| LightGBM | `live_feasible` | -2.4 % | -0.0025 [-0.0078, +0.0022] | -0.14 [-0.88, +0.70] | -0.15 [-0.89, +0.71] | 85 % |
| XGBoost | `live_feasible` | -1.4 % | -0.0014 [-0.0077, +0.0046] | -0.58 [-1.52, +0.28] | -0.59 [-1.52, +0.28] | 87 % |
| random forest | `live_feasible` | +1.1 % | +0.0011 [-0.0064, +0.0080] | -0.14 [-0.72, +0.50] | -0.14 [-0.73, +0.50] | 86 % |
| LightGBM | `all_features` | +0.9 % | +0.0009 [-0.0054, +0.0084] | +0.09 [-0.62, +0.74] | +0.09 [-0.62, +0.75] | 84 % |
| XGBoost | `all_features` | +0.8 % | +0.0008 [-0.0071, +0.0105] | -0.13 [-0.83, +0.59] | -0.13 [-0.83, +0.59] | 84 % |
| random forest | `all_features` | +5.5 % | +0.0055 [-0.0033, +0.0147] | -0.40 [-1.09, +0.31] | -0.40 [-1.10, +0.31] | 80 % |

Intervals entirely below / above zero (9 rows): QLIKE 0 / 3; Sharpe mid 0 / 0; Sharpe crossed 0 / 0.

**T1 - ridge**

| model | input set | QLIKE diff (%) | QLIKE diff [interval] | Sharpe mid diff [interval] | Sharpe crossed diff [interval] | same side |
|---|---|---|---|---|---|---|
| LightGBM | `baseline` | +6.0 % | +0.0058 [+0.0018, +0.0099] | +0.03 [-0.55, +0.60] | +0.03 [-0.55, +0.61] | 90 % |
| XGBoost | `baseline` | +4.9 % | +0.0048 [+0.0012, +0.0083] | -0.19 [-0.76, +0.42] | -0.19 [-0.77, +0.43] | 91 % |
| random forest | `baseline` | +5.6 % | +0.0055 [+0.0004, +0.0106] | -0.12 [-0.79, +0.58] | -0.12 [-0.80, +0.58] | 86 % |
| LightGBM | `live_feasible` | -4.0 % | -0.0040 [-0.0109, +0.0015] | -0.42 [-1.55, +0.67] | -0.43 [-1.56, +0.67] | 85 % |
| XGBoost | `live_feasible` | -3.2 % | -0.0032 [-0.0100, +0.0027] | -0.01 [-0.92, +0.86] | -0.02 [-0.95, +0.86] | 86 % |
| random forest | `live_feasible` | -1.9 % | -0.0019 [-0.0106, +0.0049] | -0.51 [-1.33, +0.36] | -0.52 [-1.35, +0.35] | 85 % |
| LightGBM | `all_features` | -3.7 % | -0.0037 [-0.0106, +0.0031] | +0.16 [-0.56, +0.87] | +0.16 [-0.56, +0.86] | 84 % |
| XGBoost | `all_features` | -1.4 % | -0.0014 [-0.0091, +0.0065] | -0.31 [-1.01, +0.48] | -0.31 [-1.01, +0.47] | 83 % |
| random forest | `all_features` | +2.6 % | +0.0027 [-0.0065, +0.0115] | -0.27 [-1.00, +0.42] | -0.27 [-1.01, +0.42] | 82 % |

Intervals entirely below / above zero (9 rows): QLIKE 0 / 3; Sharpe mid 0 / 0; Sharpe crossed 0 / 0.

**RS10 - ridge**

| model | input set | QLIKE diff (%) | QLIKE diff [interval] | Sharpe mid diff [interval] | Sharpe crossed diff [interval] | same side |
|---|---|---|---|---|---|---|
| LightGBM | `baseline` | +13.7 % | +0.0135 [+0.0071, +0.0214] | -0.30 [-0.83, +0.25] | -0.30 [-0.83, +0.26] | 90 % |
| XGBoost | `baseline` | +11.8 % | +0.0116 [+0.0048, +0.0212] | -0.07 [-0.68, +0.55] | -0.07 [-0.68, +0.56] | 89 % |
| random forest | `baseline` | +11.4 % | +0.0112 [+0.0048, +0.0200] | -0.32 [-1.07, +0.43] | -0.31 [-1.07, +0.43] | 89 % |
| LightGBM | `live_feasible` | -0.4 % | -0.0004 [-0.0057, +0.0051] | -0.31 [-0.92, +0.36] | -0.31 [-0.93, +0.35] | 86 % |
| XGBoost | `live_feasible` | +7.6 % | +0.0077 [+0.0001, +0.0178] | -0.60 [-1.37, +0.14] | -0.60 [-1.38, +0.14] | 85 % |
| random forest | `live_feasible` | +5.0 % | +0.0050 [-0.0015, +0.0116] | -0.85 [-1.84, +0.20] | -0.86 [-1.88, +0.20] | 86 % |
| LightGBM | `all_features` | +0.4 % | +0.0004 [-0.0068, +0.0093] | -0.05 [-0.63, +0.57] | -0.05 [-0.63, +0.57] | 85 % |
| XGBoost | `all_features` | +4.5 % | +0.0045 [-0.0027, +0.0142] | -0.05 [-0.83, +0.78] | -0.05 [-0.83, +0.78] | 83 % |
| random forest | `all_features` | +8.8 % | +0.0089 [+0.0003, +0.0191] | -0.43 [-1.32, +0.43] | -0.43 [-1.32, +0.43] | 81 % |

Intervals entirely below / above zero (9 rows): QLIKE 0 / 5; Sharpe mid 0 / 0; Sharpe crossed 0 / 0.

**RS1 - ridge**

| model | input set | QLIKE diff (%) | QLIKE diff [interval] | Sharpe mid diff [interval] | Sharpe crossed diff [interval] | same side |
|---|---|---|---|---|---|---|
| LightGBM | `baseline` | +13.9 % | +0.0136 [+0.0073, +0.0215] | -0.46 [-1.07, +0.19] | -0.46 [-1.06, +0.20] | 88 % |
| XGBoost | `baseline` | +11.4 % | +0.0112 [+0.0046, +0.0206] | -0.16 [-0.68, +0.38] | -0.16 [-0.67, +0.38] | 90 % |
| random forest | `baseline` | +10.0 % | +0.0098 [+0.0033, +0.0186] | -0.29 [-0.97, +0.38] | -0.29 [-0.97, +0.38] | 89 % |
| LightGBM | `live_feasible` | -1.1 % | -0.0011 [-0.0068, +0.0043] | -0.28 [-0.94, +0.47] | -0.28 [-0.95, +0.46] | 86 % |
| XGBoost | `live_feasible` | +5.6 % | +0.0056 [-0.0007, +0.0134] | -0.55 [-1.32, +0.20] | -0.56 [-1.33, +0.20] | 84 % |
| random forest | `live_feasible` | +1.0 % | +0.0010 [-0.0073, +0.0081] | -0.22 [-1.07, +0.63] | -0.23 [-1.09, +0.62] | 86 % |
| LightGBM | `all_features` | +0.7 % | +0.0007 [-0.0063, +0.0099] | +0.03 [-0.56, +0.65] | +0.04 [-0.56, +0.65] | 85 % |
| XGBoost | `all_features` | +2.8 % | +0.0028 [-0.0039, +0.0117] | -0.08 [-0.86, +0.74] | -0.08 [-0.86, +0.74] | 83 % |
| random forest | `all_features` | +5.9 % | +0.0060 [-0.0028, +0.0144] | -0.42 [-1.25, +0.47] | -0.43 [-1.26, +0.47] | 82 % |

Intervals entirely below / above zero (9 rows): QLIKE 0 / 3; Sharpe mid 0 / 0; Sharpe crossed 0 / 0.

## Cluster use (the campaign's own jobs, sacct)

| stage | allocations | CPUs per allocation | allocated CPU-hours | used CPU-hours | peak memory per allocation (GiB) | first start | last end |
|---|---|---|---|---|---|---|---|
| gates | 1 | 20-20 | 5.8 | 0.6 | 18.4 | 2026-09-29 19:39:31 | 2026-09-29 19:56:46 |
| canary | 1 | 20-20 | 7.9 | 2.0 | 24.6 | 2026-09-29 19:58:51 | 2026-09-29 20:22:29 |
| rs10_canary | 1 | 20-20 | 0.7 | 0.2 | 7.5 | 2026-09-29 19:58:51 | 2026-09-29 20:00:55 |
| rs1_canary | 1 | 20-20 | 1.7 | 1.3 | 7.6 | 2026-09-29 19:58:51 | 2026-09-29 20:04:05 |
| merge | 1 | 1-1 | 0.1 | 0.0 | 0.1 | 2026-09-29 20:54:09 | 2026-09-29 21:02:04 |
| untuned | 9 | 20-20 | 38.5 | 23.4 | 40.1 | 2026-09-29 20:22:44 | 2026-09-29 20:53:42 |
| rs10 | 54 | 20-20 | 21.3 | 4.4 | 9.4 | 2026-09-29 20:22:44 | 2026-09-29 20:33:18 |
| rs1 | 54 | 20-20 | 31.3 | 22.0 | 9.8 | 2026-09-29 20:32:12 | 2026-09-29 20:47:22 |

All stages: 107.3 allocated CPU-hours (54.0 used: the busy share of the allocated cores; a pack waits for its slowest process, a pool for its serial steps); peak concurrent allocated CPUs 360; wall-clock 2026-09-29 19:39:31 .. 2026-09-29 21:02:04.

## Gates

169 gates checked, 0 failed (`ladder_gates.csv`): every table's profile B equals the common target's; every stacked tree table carries its merged arm's pred_adj exactly; every forecast covers the same trade days; the per-day trade returns reproduce `score_linear_subsection.trade_1530`'s Sharpe.

Wording: sign(s) = the rule above; QLIKE differences are losses (negative = a lower loss); nothing here is a recommendation.
