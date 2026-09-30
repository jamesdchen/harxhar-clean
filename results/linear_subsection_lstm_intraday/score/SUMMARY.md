# The intraday-sequence LSTM at 16:00 (checklist I10)

Written by `experiments/score_lstm_intraday_1600.py` from its own CSVs in this folder (`lstmi_levels.csv`, `lstmi_pairs.csv`, `lstmi_hyperparameter_path.csv`, `lstmi_hyperparameter_share.csv`, `lstmi_kept.csv`, `lstmi_seed_spread.csv`, `lstmi_gates.csv`, `cluster_usage.csv`). Every number below is read from them.

## What is scored

The forecast of the 15:30-16:00 bar's realized variance (row stamped 16:00, issued at 15:30) by an LSTM over the last N half-hour bars of the near-24-hour panel ending with the 15:30 bar (`specs/causal_tune_lstm_intraday.py`): one step per bar, bar-level inputs (the bar's adjusted target `adj_RV`, the adjusted exogenous values `adj_*` of the input set, their availability / activity indicators, the calendar columns of the per-bar designs and a half-hour clock) instead of the HAR ladders `har_ma_*`; N tuned in {13, 48, 96} bars with the per-bar LSTM's grid (hidden 16 / 64, dropout 0 / 0.2, learning rate 1e-3 / 1e-2), re-chosen every 250 sessions on a 125-session validation tail after a 25-session embargo (MSE rule of record, QLIKE rule recorded), refitted every session on a 2000-session window, per-window column mask, the average of 5 seeds. Input sets: `baseline` (target series + calendar / clock), `live_feasible`, `all_features`.

Comparators: the per-bar LSTM (a sequence of the last L sessions' 16:00 rows of the per-bar design), the per-bar ridge, and the best tree rung on disk -- the tree table with the lowest 16:00 QLIKE on the same days, chosen ex post (so its comparison favours the tree).

## Scorer and days

The research scorer (the 16:00 bar recalibrated on its own; the master table's loader, target and trade imported): forecast = (f^2 + s) B with s = the forecast's own mean squared adjusted-scale error at 16:00 over the previous 250 sessions (at least 63), lagged one session; QLIKE against the per-bar spec's 16:00 target (lower is better). The last-30-min trade is the sign(s) rule on the straddle (nearest out-of-the-money call + nearest out-of-the-money put, same-day expiry, one position): buy when the recalibrated forecast exceeds the 15:30 implied variance, sell otherwise; annualized Sharpe at mid and at crossed fills. Intervals: 95 % day-block bootstrap (21-day circular blocks, 2000 draws), paired. Differences are a - b.

## Levels

| input set | forecast | QLIKE | Sharpe mid | Sharpe crossed | % buy | days | what |
|---|---|---|---|---|---|---|---|
| `live_feasible` | intraday | 0.1103 | 1.53 | 1.06 | 37.1 | 866 | intraday-sequence LSTM, MSE rule (of record) |
| `live_feasible` | intraday_qsel | 0.1080 | 1.71 | 1.23 | 35.2 | 866 | intraday-sequence LSTM, QLIKE rule (recorded) |
| `live_feasible` | perbar_lstm | 0.1220 | 1.13 | 0.66 | 39.8 | 866 | per-bar LSTM, de-dup design, no mask, refit every 10 sessions (agent A, I1) |
| `live_feasible` | ridge | 0.1005 | 1.90 | 1.44 | 40.2 | 866 | per-bar ridge |
| `live_feasible` | best_tree | 0.0955 | 1.42 | 0.95 | 37.3 | 866 | subtree_daily_live_feasible_lgbm (lowest QLIKE of 45 tree tables on these days, chosen ex post) |
| `all_features` | intraday | 0.1119 | 0.97 | 0.49 | 35.3 | 866 | intraday-sequence LSTM, MSE rule (of record) |
| `all_features` | intraday_qsel | 0.1125 | 1.29 | 0.82 | 36.3 | 866 | intraday-sequence LSTM, QLIKE rule (recorded) |
| `all_features` | perbar_lstm | 0.1753 | 0.97 | 0.50 | 41.3 | 866 | per-bar LSTM, de-dup design, no mask, refit every 10 sessions (agent A, I1) |
| `all_features` | ridge | 0.1004 | 1.66 | 1.20 | 36.1 | 866 | per-bar ridge |
| `all_features` | best_tree | 0.0982 | 1.66 | 1.19 | 37.9 | 866 | subtree_daily_all_features_lgbm (lowest QLIKE of 45 tree tables on these days, chosen ex post) |
| `baseline` | intraday | 0.1073 | 0.81 | 0.33 | 34.5 | 866 | intraday-sequence LSTM, MSE rule (of record) |
| `baseline` | intraday_qsel | 0.1071 | 0.87 | 0.39 | 34.2 | 866 | intraday-sequence LSTM, QLIKE rule (recorded) |
| `baseline` | perbar_lstm | 0.1068 | 1.20 | 0.73 | 39.1 | 866 | per-bar LSTM, de-dup design, no mask, refit every 10 sessions (agent A, I1) |
| `baseline` | ridge | 0.0982 | 1.22 | 0.74 | 37.9 | 866 | per-bar ridge |
| `baseline` | best_tree | 0.1022 | 0.88 | 0.41 | 40.1 | 866 | subtree_tunedq_baseline_rf (lowest QLIKE of 45 tree tables on these days, chosen ex post) |

## Paired differences (a - b) with 95 % intervals

| input set | a - b | QLIKE diff (%) | QLIKE diff [interval] | DM t | Sharpe mid diff [interval] | Sharpe crossed diff [interval] | same side |
|---|---|---|---|---|---|---|---|
| `live_feasible` | intraday - perbar_lstm | -9.6 % | -0.0117 [-0.0262, +0.0040] | -1.71 | +0.40 [-0.51, +1.29] | +0.40 [-0.51, +1.29] | 79 % |
| `live_feasible` | intraday - ridge | +9.7 % | +0.0098 [+0.0001, +0.0201] | +1.99 | -0.37 [-1.33, +0.60] | -0.38 [-1.34, +0.60] | 81 % |
| `live_feasible` | intraday - best_tree | +15.5 % | +0.0148 [+0.0048, +0.0268] | +3.00 | +0.11 [-0.73, +0.86] | +0.11 [-0.73, +0.86] | 82 % |
| `live_feasible` | intraday_qsel - intraday | -2.1 % | -0.0023 [-0.0051, -0.0000] | -1.95 | +0.18 [-0.20, +0.52] | +0.17 [-0.20, +0.52] | 96 % |
| `live_feasible` | perbar_lstm - ridge | +21.4 % | +0.0215 [+0.0075, +0.0366] | +3.41 | -0.77 [-1.91, +0.37] | -0.78 [-1.92, +0.37] | 77 % |
| `all_features` | intraday - perbar_lstm | -36.2 % | -0.0634 [-0.0964, -0.0359] | -4.76 | -0.01 [-1.20, +1.21] | -0.01 [-1.20, +1.21] | 73 % |
| `all_features` | intraday - ridge | +11.5 % | +0.0115 [+0.0022, +0.0226] | +2.32 | -0.70 [-1.74, +0.35] | -0.71 [-1.74, +0.34] | 80 % |
| `all_features` | intraday - best_tree | +14.0 % | +0.0137 [+0.0050, +0.0244] | +2.95 | -0.70 [-1.68, +0.19] | -0.70 [-1.68, +0.19] | 81 % |
| `all_features` | intraday_qsel - intraday | +0.5 % | +0.0006 [-0.0025, +0.0037] | +0.34 | +0.32 [-0.27, +0.99] | +0.32 [-0.27, +1.00] | 92 % |
| `all_features` | perbar_lstm - ridge | +74.6 % | +0.0749 [+0.0490, +0.1080] | +5.54 | -0.69 [-2.04, +0.63] | -0.70 [-2.05, +0.62] | 72 % |
| `baseline` | intraday - perbar_lstm | +0.4 % | +0.0005 [-0.0064, +0.0093] | +0.15 | -0.39 [-1.04, +0.26] | -0.40 [-1.04, +0.26] | 88 % |
| `baseline` | intraday - ridge | +9.2 % | +0.0090 [+0.0024, +0.0169] | +2.94 | -0.41 [-1.18, +0.41] | -0.41 [-1.18, +0.41] | 88 % |
| `baseline` | intraday - best_tree | +4.9 % | +0.0051 [-0.0015, +0.0129] | +1.60 | -0.07 [-0.79, +0.66] | -0.08 [-0.79, +0.65] | 85 % |
| `baseline` | intraday_qsel - intraday | -0.2 % | -0.0002 [-0.0040, +0.0039] | -0.10 | +0.06 [-0.45, +0.56] | +0.06 [-0.45, +0.56] | 94 % |
| `baseline` | perbar_lstm - ridge | +8.7 % | +0.0086 [+0.0036, +0.0134] | +3.79 | -0.02 [-0.65, +0.64] | -0.02 [-0.65, +0.64] | 89 % |

## Chosen configuration over time (MSE rule of record / QLIKE rule)

| input set | tuning row | first forecast | rule | N | hidden | dropout | lr | epochs (seeds) |
|---|---|---|---|---|---|---|---|---|
| `live_feasible` | 0 | 2018-06-25 | mse | 48 | 16 | 0.2 | 0.01 | 26;36;41;20;32 |
| `live_feasible` | 0 | 2018-06-25 | qlike | 48 | 64 | 0.2 | 0.001 | 20;77;30;20;33 |
| `live_feasible` | 250 | 2019-06-26 | mse | 96 | 64 | 0.0 | 0.01 | 3;15;3;1;1 |
| `live_feasible` | 250 | 2019-06-26 | qlike | 96 | 64 | 0.0 | 0.01 | 3;15;3;1;1 |
| `live_feasible` | 500 | 2020-06-23 | mse | 96 | 16 | 0.2 | 0.01 | 49;17;15;31;9 |
| `live_feasible` | 500 | 2020-06-23 | qlike | 48 | 64 | 0.2 | 0.01 | 13;3;2;4;4 |
| `live_feasible` | 750 | 2021-06-21 | mse | 48 | 64 | 0.2 | 0.01 | 1;1;3;5;1 |
| `live_feasible` | 750 | 2021-06-21 | qlike | 13 | 64 | 0.0 | 0.01 | 3;1;3;3;1 |
| `live_feasible` | 1000 | 2022-06-17 | mse | 13 | 64 | 0.2 | 0.01 | 1;2;5;12;12 |
| `live_feasible` | 1000 | 2022-06-17 | qlike | 13 | 64 | 0.2 | 0.01 | 1;2;5;12;12 |
| `live_feasible` | 1250 | 2023-06-16 | mse | 48 | 16 | 0.0 | 0.01 | 4;6;6;10;10 |
| `live_feasible` | 1250 | 2023-06-16 | qlike | 48 | 16 | 0.0 | 0.01 | 4;6;6;10;10 |
| `all_features` | 0 | 2018-06-25 | mse | 96 | 16 | 0.2 | 0.01 | 44;26;80;27;39 |
| `all_features` | 0 | 2018-06-25 | qlike | 13 | 64 | 0.2 | 0.001 | 15;20;21;37;13 |
| `all_features` | 250 | 2019-06-26 | mse | 96 | 64 | 0.0 | 0.01 | 1;2;1;7;6 |
| `all_features` | 250 | 2019-06-26 | qlike | 96 | 64 | 0.0 | 0.01 | 1;2;1;7;6 |
| `all_features` | 500 | 2020-06-23 | mse | 13 | 64 | 0.2 | 0.01 | 5;31;12;2;4 |
| `all_features` | 500 | 2020-06-23 | qlike | 13 | 64 | 0.0 | 0.01 | 5;6;13;8;7 |
| `all_features` | 750 | 2021-06-21 | mse | 96 | 16 | 0.0 | 0.01 | 2;5;4;3;4 |
| `all_features` | 750 | 2021-06-21 | qlike | 48 | 16 | 0.0 | 0.01 | 3;7;4;3;5 |
| `all_features` | 1000 | 2022-06-17 | mse | 48 | 16 | 0.2 | 0.01 | 7;13;3;24;32 |
| `all_features` | 1000 | 2022-06-17 | qlike | 13 | 64 | 0.0 | 0.01 | 5;3;2;2;5 |
| `all_features` | 1250 | 2023-06-16 | mse | 96 | 16 | 0.2 | 0.01 | 12;3;4;4;3 |
| `all_features` | 1250 | 2023-06-16 | qlike | 96 | 64 | 0.2 | 0.01 | 3;4;6;2;4 |
| `baseline` | 0 | 2018-06-25 | mse | 48 | 64 | 0.0 | 0.01 | 34;16;9;17;13 |
| `baseline` | 0 | 2018-06-25 | qlike | 48 | 64 | 0.0 | 0.01 | 34;16;9;17;13 |
| `baseline` | 250 | 2019-06-26 | mse | 48 | 16 | 0.2 | 0.01 | 4;39;2;2;3 |
| `baseline` | 250 | 2019-06-26 | qlike | 96 | 64 | 0.0 | 0.01 | 2;5;8;16;6 |
| `baseline` | 500 | 2020-06-23 | mse | 48 | 16 | 0.2 | 0.01 | 42;29;74;46;28 |
| `baseline` | 500 | 2020-06-23 | qlike | 96 | 64 | 0.2 | 0.001 | 90;80;59;107;54 |
| `baseline` | 750 | 2021-06-21 | mse | 48 | 16 | 0.0 | 0.001 | 73;32;68;38;26 |
| `baseline` | 750 | 2021-06-21 | qlike | 48 | 64 | 0.2 | 0.01 | 19;29;26;20;39 |
| `baseline` | 1000 | 2022-06-17 | mse | 13 | 64 | 0.2 | 0.01 | 27;12;16;4;6 |
| `baseline` | 1000 | 2022-06-17 | qlike | 13 | 64 | 0.2 | 0.01 | 27;12;16;4;6 |
| `baseline` | 1250 | 2023-06-16 | mse | 48 | 16 | 0.2 | 0.01 | 20;26;32;17;12 |
| `baseline` | 1250 | 2023-06-16 | qlike | 13 | 64 | 0.2 | 0.01 | 34;6;40;25;24 |

## Kept step columns (per-window mask)

| input set | where | step columns | fits | min kept | median | max |
|---|---|---|---|---|---|---|
| `live_feasible` | refit (mse rule) | 54 | 1469 | 37 | 40 | 41 |
| `live_feasible` | refit (qlike rule) | 54 | 1469 | 37 | 40 | 41 |
| `live_feasible` | tuning point, N = 13 | 54 | 6 | 37 | 37 | 37 |
| `live_feasible` | tuning point, N = 48 | 54 | 6 | 40 | 40 | 40 |
| `live_feasible` | tuning point, N = 96 | 54 | 6 | 40 | 40 | 40 |
| `all_features` | refit (mse rule) | 120 | 1469 | 74 | 80 | 82 |
| `all_features` | refit (qlike rule) | 120 | 1469 | 74 | 76 | 82 |
| `all_features` | tuning point, N = 13 | 120 | 6 | 74 | 74.5 | 76 |
| `all_features` | tuning point, N = 48 | 120 | 6 | 79 | 81 | 82 |
| `all_features` | tuning point, N = 96 | 120 | 6 | 79 | 81 | 82 |
| `baseline` | refit (mse rule) | 19 | 1469 | 16 | 19 | 19 |
| `baseline` | refit (qlike rule) | 19 | 1469 | 16 | 19 | 19 |
| `baseline` | tuning point, N = 13 | 19 | 6 | 16 | 16 | 16 |
| `baseline` | tuning point, N = 48 | 19 | 6 | 19 | 19 | 19 |
| `baseline` | tuning point, N = 96 | 19 | 6 | 19 | 19 | 19 |

## Seed spread (each network alone vs the 5-seed average, same recalibration and days)

| input set | seed | QLIKE | Sharpe mid | Sharpe crossed | % buy |
|---|---|---|---|---|---|
| `live_feasible` | 0 | 0.1245 | 1.20 | 0.73 | 36.5 |
| `live_feasible` | 1 | 0.1209 | 1.27 | 0.80 | 36.5 |
| `live_feasible` | 2 | 0.1158 | 1.01 | 0.54 | 38.1 |
| `live_feasible` | 3 | 0.1254 | 0.83 | 0.36 | 38.2 |
| `live_feasible` | 4 | 0.1312 | 1.28 | 0.81 | 40.1 |
| `live_feasible` | average | 0.1103 | 1.53 | 1.06 | 37.1 |
| `all_features` | 0 | 0.1338 | 1.18 | 0.71 | 40.6 |
| `all_features` | 1 | 0.1253 | 0.75 | 0.28 | 38.3 |
| `all_features` | 2 | 0.1318 | 0.74 | 0.27 | 37.6 |
| `all_features` | 3 | 0.1184 | 0.88 | 0.41 | 38.2 |
| `all_features` | 4 | 0.1241 | 1.08 | 0.61 | 38.8 |
| `all_features` | average | 0.1119 | 0.97 | 0.49 | 35.3 |
| `baseline` | 0 | 0.1113 | 0.86 | 0.38 | 35.5 |
| `baseline` | 1 | 0.1061 | 0.99 | 0.51 | 35.5 |
| `baseline` | 2 | 0.1152 | 0.98 | 0.50 | 36.3 |
| `baseline` | 3 | 0.1182 | 1.11 | 0.64 | 36.4 |
| `baseline` | 4 | 0.1125 | 0.46 | -0.02 | 34.6 |
| `baseline` | average | 0.1073 | 0.81 | 0.33 | 34.5 |

`live_feasible`: mean over rows of the sd of the 5 seeds' adjusted-scale forecasts 0.0766 (mean |forecast| 0.9634).

`all_features`: mean over rows of the sd of the 5 seeds' adjusted-scale forecasts 0.0859 (mean |forecast| 0.9640).

`baseline`: mean over rows of the sd of the 5 seeds' adjusted-scale forecasts 0.0456 (mean |forecast| 0.9545).

## Cluster use (sacct, this campaign's jobs)

| job | allocations | CPUs each | allocated CPU-h | used CPU-h | first start | last end |
|---|---|---|---|---|---|---|
| lstmi_gatesA | 1 | 20-20 | 0.5 | 0.0 | 2026-09-29 23:31:34 | 2026-09-29 23:33:03 |
| lstmi_gatesB | 2 | 20-20 | 2.1 | 0.7 | 2026-09-29 23:33:02 | 2026-09-30 02:29:20 |
| lstmi_canary | 1 | 20-20 | 1.1 | 0.8 | 2026-09-29 23:31:34 | 2026-09-29 23:34:59 |
| lstmi_tune | 36 | 20-20 | 27.2 | 15.0 | 2026-09-29 23:39:21 | 2026-09-30 02:49:44 |
| lstmi_merge | 3 | 1-1 | 0.0 | 0.0 | 2026-09-30 00:23:56 | 2026-09-30 03:46:30 |
| lstmi_refit | 76 | 20-20 | 166.3 | 142.7 | 2026-09-30 00:21:21 | 2026-09-30 03:44:20 |

All: 197.3 allocated CPU-hours; peak concurrent allocated CPUs 340; wall-clock 2026-09-29 23:31:34 .. 2026-09-30 03:46:30.

## Gates

438 scorer gates, 0 failed (`lstmi_gates.csv`). The spec's own gates (target identity, causality x2, determinism, chunk = unchunked, pool = serial, mask) are in `../gates/`:

| where | gate | input set | pass |
|---|---|---|---|
| cluster | determinism | live_feasible | yes |
| cluster | pool | live_feasible | yes |
| cluster | chunk | live_feasible | yes |
| cluster | causal_target | live_feasible | yes |
| cluster | causal_bar | live_feasible | yes |
| cluster | mask | synthetic | yes |
| cluster | e2e_chunk | live_feasible | yes |
| cluster | identity | live_feasible | yes |
| cluster | seq_end | live_feasible | yes |
| cluster | identity | all_features | yes |
| cluster | seq_end | all_features | yes |
| cluster | identity | baseline | yes |
| cluster | seq_end | baseline | yes |
| cluster | mask_flatten | live_feasible | yes |
| local | seq_end | live_feasible | yes |
| local | mask_flatten | live_feasible | yes |
| local | determinism | live_feasible | yes |
| local | pool | live_feasible | yes |
| local | chunk | live_feasible | yes |
| local | causal_target | live_feasible | yes |
| local | causal_bar | live_feasible | yes |
| local | e2e_chunk | live_feasible | yes |
| local | identity | baseline | yes |
| local | seq_end | baseline | yes |
| local | mask | synthetic | yes |

CPU class. The campaign ran pinned to one CPU class (epyc-7513). A first attempt on mixed nodes was stopped by the tuning cache's fingerprint: the executor's bar-level inputs differ in their last bits between Intel and AMD nodes. `../gates/cross_class_mixed_vs_pinned.csv` compares it with the pinned run: tuning points bit-identical 17 of 18 (the rest: `live_feasible` row 0 on Intel(R) Xeon(R) Silver 4116 CPU @ 2.10GHz, same picks True, max |validation MSE diff| 5.86e-03); refit chunks bit-identical 19 of 20 finished (not identical: `baseline` chunk 3 on Intel(R) Xeon(R) Silver 4116 CPU @ 2.10GHz, max |diff| 4.50e-02).

Wording: sign(s) = the rule above; QLIKE differences are losses (negative = a lower loss); nothing here is a recommendation.
