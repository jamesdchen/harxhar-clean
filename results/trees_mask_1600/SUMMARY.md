# The 16:00 ladder with the per-window mask (checklist I9)

Written by `experiments/trees_mask_ladder_1600.py` from its own CSVs in this folder (`mask_levels.csv`, `mask_pairs.csv`, `mask_wide.csv`, `mask_gates.csv`, `kept_counts.csv`, `cluster_usage.csv`). Every number below is read from them.

## What changed

Every tree fit and every LSTM fit now sees only the columns kept by `src.models.window_mask.window_keep` on the window it is trained on (the 2000 sessions before the refit, or the tuning window): a column constant on that window, or an exact byte-copy of an earlier kept column, is dropped (`WINDOW_MASK=1`). The per-bar ridge always had this rule (at each penalty tune). Everything else is the de-duplicated per-bar design of the unmasked runs (commit 47f7f9c), 2000-session window, 15:30-16:00 bar (row stamped 16:00, forecast issued at 15:30):

| rung | configuration | refit | table |
|---|---|---|---|
| T10 | shipped configuration | every 10 sessions | `yhat_subtree_<model>_<bucket>` |
| T1 | shipped configuration | every session | `yhat_subtree_daily_<bucket>_<model>` |
| RS10 | random search, MSE rule | every 10 sessions | `yhat_subtree_tuned_<bucket>_<model>` |
| RS1 | random search, MSE rule | every session | `yhat_subtree_tuned_daily_<bucket>_<model>` |
| RS10q | random search, QLIKE rule | every 10 sessions | `yhat_subtree_tunedq_<bucket>_<model>` |
| RS1q | random search, QLIKE rule | every session | `yhat_subtree_tunedq_daily_<bucket>_<model>` |
| LSTM | per-bar LSTM, 16 configurations x 5 seeds re-chosen every 250 sessions (MSE rule; LSTMq: QLIKE rule) | every session (daily refit; the canonical table) | `yhat_lstm[_qsel]_<bucket>` |
| LSTM10 | the same LSTM (LSTM10q: QLIKE rule) | every 10 sessions (the earlier cadence; the unmasked LSTM ran at it) | `results/trees_mask_1600/stack/lstm_re10/yhat_lstm[_qsel]_<bucket>` (masked); the snapshot (unmasked) |
| ridge | per-bar ridge, penalty re-chosen every 250 sessions | every session | `yhat_sub_ridge_<bucket>` |

Masked = the canonical tables in `results/spxw_pnl/`; unmasked = the same names in `results/spxw_pnl/dedup_nomask_2026-09-29/` (the de-duplicated runs before the mask). The LSTM rebuilds its network at every refit, so its mask is recomputed at every refit (as for the trees).

## Kept columns (median / min / max over refits)

The mask depends on the design window only, so every model of an input set keeps the same columns at the same refit row; the rungs differ only in which rows they refit at (every session for T1 / RS1 / LSTM, every 10th for T10 / RS10 / LSTM10).

| input set | p | rung | model | refits | kept median | kept min | kept max | tuning points kept median [min, max] |
|---|---|---|---|---|---|---|---|---|
| `all_features` | 628 | T10 | LightGBM | 147 | 369 | 360 | 378 |  |
| `all_features` | 628 | T10 | random forest | 147 | 369 | 360 | 378 |  |
| `all_features` | 628 | T10 | XGBoost | 147 | 369 | 360 | 378 |  |
| `all_features` | 628 | T1 | LightGBM | 1469 | 369 | 360 | 378 |  |
| `all_features` | 628 | T1 | random forest | 1469 | 369 | 360 | 378 |  |
| `all_features` | 628 | T1 | XGBoost | 1469 | 369 | 360 | 378 |  |
| `all_features` | 628 | RS10 | LightGBM | 147 | 369 | 360 | 378 | 373 [360, 378] |
| `all_features` | 628 | RS10 | random forest | 147 | 369 | 360 | 378 | 373 [360, 378] |
| `all_features` | 628 | RS10 | XGBoost | 147 | 369 | 360 | 378 | 373 [360, 378] |
| `all_features` | 628 | RS1 | LightGBM | 1469 | 369 | 360 | 378 | 373 [360, 378] |
| `all_features` | 628 | RS1 | random forest | 1469 | 369 | 360 | 378 | 373 [360, 378] |
| `all_features` | 628 | RS1 | XGBoost | 1469 | 369 | 360 | 378 | 373 [360, 378] |
| `all_features` | 628 | LSTM | LSTM | 1469 | 369 | 360 | 378 | 373 [360, 378] |
| `all_features` | 628 | LSTM10 | LSTM | 147 | 369 | 360 | 378 | 373 [360, 378] |
| `baseline` | 22 | T10 | LightGBM | 147 | 17 | 17 | 17 |  |
| `baseline` | 22 | T10 | random forest | 147 | 17 | 17 | 17 |  |
| `baseline` | 22 | T10 | XGBoost | 147 | 17 | 17 | 17 |  |
| `baseline` | 22 | T1 | LightGBM | 1469 | 17 | 17 | 17 |  |
| `baseline` | 22 | T1 | random forest | 1469 | 17 | 17 | 17 |  |
| `baseline` | 22 | T1 | XGBoost | 1469 | 17 | 17 | 17 |  |
| `baseline` | 22 | RS10 | LightGBM | 147 | 17 | 17 | 17 | 17 [17, 17] |
| `baseline` | 22 | RS10 | random forest | 147 | 17 | 17 | 17 | 17 [17, 17] |
| `baseline` | 22 | RS10 | XGBoost | 147 | 17 | 17 | 17 | 17 [17, 17] |
| `baseline` | 22 | RS1 | LightGBM | 1469 | 17 | 17 | 17 | 17 [17, 17] |
| `baseline` | 22 | RS1 | random forest | 1469 | 17 | 17 | 17 | 17 [17, 17] |
| `baseline` | 22 | RS1 | XGBoost | 1469 | 17 | 17 | 17 | 17 [17, 17] |
| `baseline` | 22 | LSTM | LSTM | 1469 | 17 | 17 | 17 | 17 [17, 17] |
| `baseline` | 22 | LSTM10 | LSTM | 147 | 17 | 17 | 17 | 17 [17, 17] |
| `live_feasible` | 232 | T10 | LightGBM | 147 | 138 | 136 | 145 |  |
| `live_feasible` | 232 | T10 | random forest | 147 | 138 | 136 | 145 |  |
| `live_feasible` | 232 | T10 | XGBoost | 147 | 138 | 136 | 145 |  |
| `live_feasible` | 232 | T1 | LightGBM | 1469 | 138 | 136 | 145 |  |
| `live_feasible` | 232 | T1 | random forest | 1469 | 138 | 136 | 145 |  |
| `live_feasible` | 232 | T1 | XGBoost | 1469 | 138 | 136 | 145 |  |
| `live_feasible` | 232 | RS10 | LightGBM | 147 | 138 | 136 | 145 | 138 [136, 138] |
| `live_feasible` | 232 | RS10 | random forest | 147 | 138 | 136 | 145 | 138 [136, 138] |
| `live_feasible` | 232 | RS10 | XGBoost | 147 | 138 | 136 | 145 | 138 [136, 138] |
| `live_feasible` | 232 | RS1 | LightGBM | 1469 | 138 | 136 | 145 | 138 [136, 138] |
| `live_feasible` | 232 | RS1 | random forest | 1469 | 138 | 136 | 145 | 138 [136, 138] |
| `live_feasible` | 232 | RS1 | XGBoost | 1469 | 138 | 136 | 145 | 138 [136, 138] |
| `live_feasible` | 232 | LSTM | LSTM | 1469 | 138 | 136 | 145 | 138 [136, 138] |
| `live_feasible` | 232 | LSTM10 | LSTM | 147 | 138 | 136 | 145 | 138 [136, 138] |

## CPU class of every chunk

Forecasts are bit-reproducible only within a CPU vector class (the design moves at ~1e-11 between AVX-512 xeon and AVX2 epyc nodes, and LightGBM's histogram bins with it), so every masked task was pinned to epyc-7513 and every finished chunk records its node (`class_census.csv`). The unmasked runs this page compares with ran unpinned (classes not recorded).

| results root | rung | CPU class | chunks |
|---|---|---|---|
| lstm | lstm | epyc-7513 | 18 |
| lstm10 | lstm10 | epyc-7513 | 18 |
| tuned | rs1 | epyc-7513 | 54 |
| tuned | rs10 | epyc-7513 | 54 |
| untuned | t1 | epyc-7513 | 161 |
| untuned | t10 | epyc-7513 | 22 |
| xeon | t1 | xeon-4116 | 5 |
| xeon | t10 | xeon-4116 | 1 |

Canary chunks that ran before the pinning, against the same chunk re-run on epyc-7513 (`class_chunks.csv`; pred_adj on the chunk's rows):

| chunk | canary class | fleet class | rows | bit-identical | max relative difference |
|---|---|---|---|---|---|
| `results/linear_subsection_trees_mask/t1/all_features/bar1600/lgbm/tw2000/chunks/c0` | epyc-7313 | epyc-7513 | 91 | True | 0.00e+00 |
| `results/linear_subsection_trees_mask/t1/all_features/bar1600/rf/tw2000/chunks/c0` | epyc-7313 | epyc-7513 | 16 | True | 0.00e+00 |
| `results/linear_subsection_trees_mask/t1/all_features/bar1600/xgb/tw2000/chunks/c0` | epyc-7313 | epyc-7513 | 144 | True | 0.00e+00 |
| `results/linear_subsection_trees_mask/t1/baseline/bar1600/lgbm/tw2000/chunks/c0` | epyc-7313 | epyc-7513 | 1469 | True | 0.00e+00 |
| `results/linear_subsection_trees_mask/t1/baseline/bar1600/rf/tw2000/chunks/c0` | epyc-7313 | epyc-7513 | 297 | True | 0.00e+00 |
| `results/linear_subsection_trees_mask/t1/baseline/bar1600/xgb/tw2000/chunks/c0` | epyc-7313 | epyc-7513 | 1469 | True | 0.00e+00 |
| `results/linear_subsection_trees_mask/t1/live_feasible/bar1600/lgbm/tw2000/chunks/c0` | epyc-7313 | epyc-7513 | 358 | True | 0.00e+00 |
| `results/linear_subsection_trees_mask/t1/live_feasible/bar1600/rf/tw2000/chunks/c0` | epyc-7313 | epyc-7513 | 60 | True | 0.00e+00 |
| `results/linear_subsection_trees_mask/t1/live_feasible/bar1600/xgb/tw2000/chunks/c0` | epyc-7313 | epyc-7513 | 441 | True | 0.00e+00 |
| `results/linear_subsection_trees_mask/t10/all_features/bar1600/lgbm/tw2000/chunks/c0` | epyc-7313 | epyc-7513 | 910 | True | 0.00e+00 |
| `results/linear_subsection_trees_mask/t10/all_features/bar1600/rf/tw2000/chunks/c0` | epyc-7313 | epyc-7513 | 160 | True | 0.00e+00 |
| `results/linear_subsection_trees_mask/t10/all_features/bar1600/xgb/tw2000/chunks/c0` | epyc-7313 | epyc-7513 | 1430 | True | 0.00e+00 |
| `results/linear_subsection_trees_mask/t10/baseline/bar1600/lgbm/tw2000/chunks/c0` | epyc-7313 | epyc-7513 | 1469 | True | 0.00e+00 |
| `results/linear_subsection_trees_mask/t10/baseline/bar1600/rf/tw2000/chunks/c0` | epyc-7313 | epyc-7513 | 1469 | True | 0.00e+00 |
| `results/linear_subsection_trees_mask/t10/baseline/bar1600/xgb/tw2000/chunks/c0` | epyc-7313 | epyc-7513 | 1469 | True | 0.00e+00 |
| `results/linear_subsection_trees_mask/t10/live_feasible/bar1600/lgbm/tw2000/chunks/c0` | epyc-7313 | epyc-7513 | 1469 | True | 0.00e+00 |
| `results/linear_subsection_trees_mask/t10/live_feasible/bar1600/rf/tw2000/chunks/c0` | epyc-7313 | epyc-7513 | 550 | True | 0.00e+00 |
| `results/linear_subsection_trees_mask/t10/live_feasible/bar1600/xgb/tw2000/chunks/c0` | epyc-7313 | epyc-7513 | 1469 | True | 0.00e+00 |
| `results/linear_subsection_trees_tuned_mask/rs1/live_feasible/bar1600/lgbm/tw2000/chunks/c0` | xeon-4116 | epyc-7513 | 250 | False | 5.07e-02 |
| `results/linear_subsection_trees_tuned_mask/rs1/live_feasible/bar1600/rf/tw2000/chunks/c0` | xeon-4116 | epyc-7513 | 250 | True | 0.00e+00 |
| `results/linear_subsection_trees_tuned_mask/rs1/live_feasible/bar1600/xgb/tw2000/chunks/c0` | xeon-4116 | epyc-7513 | 250 | False | 7.21e-07 |
| `results/linear_subsection_trees_tuned_mask/rs10/live_feasible/bar1600/lgbm/tw2000/chunks/c0` | xeon-4116 | epyc-7513 | 250 | False | 3.85e-02 |
| `results/linear_subsection_trees_tuned_mask/rs10/live_feasible/bar1600/rf/tw2000/chunks/c0` | xeon-4116 | epyc-7513 | 250 | True | 0.00e+00 |
| `results/linear_subsection_trees_tuned_mask/rs10/live_feasible/bar1600/xgb/tw2000/chunks/c0` | xeon-4116 | epyc-7513 | 250 | True | 0.00e+00 |
| `results/linear_subsection_lstm_mask_re10/live_feasible/bar1600/lstm/tw2000/chunks/c0` | xeon-4116 | epyc-7513 | 250 | False | 1.87e-01 |

Class gate (`class_arms.csv`): the live_feasible LightGBM arms run again on xeon-4116, against the fleet's epyc-7513 arms (pred_adj over every forecast row; QLIKE and Sharpe differences in the `CPU class` pairs below):

| rung | rows | bit-identical | max relative difference | median relative difference |
|---|---|---|---|---|
| T10 | 1469 | False | 4.51e-02 | 8.36e-03 |
| T1 | 1469 | False | 5.52e-02 | 8.36e-03 |

## Scorer and days

The research scorer (the 16:00 bar recalibrated on its own; the master table's loader, target and trade imported): forecast = (f^2 + s) B with s = the forecast's own mean squared adjusted-scale error at 16:00 over the previous 250 sessions (at least 63), lagged one session; QLIKE against the per-bar spec's 16:00 target (lower is better). The last-30-min trade is the sign(s) rule on the straddle (nearest out-of-the-money call + nearest out-of-the-money put, same-day expiry, one position): buy when the recalibrated forecast exceeds the 15:30 implied variance, sell otherwise; annualized Sharpe at mid and at crossed fills. Every number is on the same 866 trade days (2020-01-03 .. 2024-04-30). Intervals: 95 % day-block bootstrap (21-day circular blocks, 2000 draws), paired (the same resampled days for both forecasts). Differences are a - b.

## Levels (masked | unmasked)

**`baseline`**

| forecast | QLIKE masked | QLIKE unmasked | Sharpe mid masked | Sharpe mid unmasked | Sharpe crossed masked | Sharpe crossed unmasked | % buy masked |
|---|---|---|---|---|---|---|---|
| LightGBM T10 | 0.1049 | 0.1044 | 0.82 | 1.00 | 0.35 | 0.53 | 41.2 |
| LightGBM T1 | 0.1041 | 0.1041 | 0.99 | 1.24 | 0.52 | 0.77 | 41.3 |
| LightGBM RS10 | 0.1115 | 0.1117 | 1.00 | 0.91 | 0.53 | 0.44 | 41.7 |
| LightGBM RS1 | 0.1116 | 0.1118 | 0.71 | 0.75 | 0.24 | 0.28 | 42.0 |
| LightGBM RS10q | 0.1059 | 0.1077 | 1.45 | 0.97 | 0.98 | 0.50 | 40.9 |
| LightGBM RS1q | 0.1058 | 0.1078 | 1.22 | 0.85 | 0.75 | 0.38 | 40.4 |
| XGBoost T10 | 0.1042 | 0.1038 | 0.92 | 0.93 | 0.44 | 0.46 | 39.7 |
| XGBoost T1 | 0.1033 | 0.1030 | 0.95 | 1.02 | 0.48 | 0.55 | 40.6 |
| XGBoost RS10 | 0.1096 | 0.1098 | 1.12 | 1.14 | 0.65 | 0.67 | 41.6 |
| XGBoost RS1 | 0.1087 | 0.1095 | 1.22 | 1.06 | 0.75 | 0.59 | 41.3 |
| XGBoost RS10q | 0.1125 | 0.1118 | 1.01 | 1.08 | 0.54 | 0.61 | 41.6 |
| XGBoost RS1q | 0.1115 | 0.1114 | 1.16 | 1.00 | 0.69 | 0.53 | 41.1 |
| random forest T10 | 0.1050 | 0.1046 | 0.90 | 0.90 | 0.42 | 0.42 | 42.4 |
| random forest T1 | 0.1037 | 0.1037 | 1.16 | 1.10 | 0.69 | 0.63 | 42.4 |
| random forest RS10 | 0.1093 | 0.1094 | 1.06 | 0.90 | 0.59 | 0.43 | 41.8 |
| random forest RS1 | 0.1085 | 0.1080 | 1.24 | 0.92 | 0.77 | 0.45 | 42.1 |
| random forest RS10q | 0.1022 | 0.1020 | 0.88 | 0.99 | 0.41 | 0.52 | 40.1 |
| random forest RS1q | 0.1026 | 0.1010 | 0.89 | 1.01 | 0.42 | 0.54 | 40.9 |
| LSTM (LSTM, MSE rule, refit every session) | 0.1049 |  | 1.26 |  | 0.79 |  | 39.5 |
| LSTMq (LSTM, QLIKE rule, refit every session) | 0.1031 |  | 0.77 |  | 0.30 |  | 38.9 |
| LSTM10 (LSTM, MSE rule, refit every 10 sessions) | 0.1070 | 0.1068 | 1.17 | 1.20 | 0.70 | 0.73 | 40.2 |
| LSTM10q (LSTM, QLIKE rule, refit every 10 sessions) | 0.1039 | 0.1049 | 0.84 | 1.02 | 0.37 | 0.54 | 39.3 |
| per-bar ridge | 0.0982 | | 1.22 | | 0.74 | | 37.9 |

**`live_feasible`**

| forecast | QLIKE masked | QLIKE unmasked | Sharpe mid masked | Sharpe mid unmasked | Sharpe crossed masked | Sharpe crossed unmasked | % buy masked |
|---|---|---|---|---|---|---|---|
| LightGBM T10 | 0.0991 | 0.0981 | 1.59 | 1.76 | 1.12 | 1.29 | 39.8 |
| LightGBM T1 | 0.0955 | 0.0965 | 1.42 | 1.48 | 0.95 | 1.01 | 37.3 |
| LightGBM RS10 | 0.0992 | 0.1001 | 1.47 | 1.59 | 1.00 | 1.12 | 38.0 |
| LightGBM RS1 | 0.0994 | 0.0995 | 1.63 | 1.62 | 1.17 | 1.16 | 39.0 |
| LightGBM RS10q | 0.1015 | 0.1025 | 1.19 | 1.41 | 0.72 | 0.94 | 39.3 |
| LightGBM RS1q | 0.1008 | 0.1015 | 1.37 | 1.50 | 0.90 | 1.04 | 38.3 |
| XGBoost T10 | 0.0995 | 0.0991 | 1.27 | 1.32 | 0.80 | 0.85 | 38.5 |
| XGBoost T1 | 0.0986 | 0.0973 | 1.53 | 1.89 | 1.06 | 1.42 | 37.8 |
| XGBoost RS10 | 0.1070 | 0.1082 | 1.46 | 1.30 | 1.00 | 0.84 | 39.1 |
| XGBoost RS1 | 0.1050 | 0.1062 | 1.34 | 1.35 | 0.87 | 0.88 | 39.7 |
| XGBoost RS10q | 0.1067 | 0.1076 | 1.67 | 1.41 | 1.20 | 0.94 | 39.5 |
| XGBoost RS1q | 0.1052 | 0.1054 | 1.52 | 1.25 | 1.05 | 0.79 | 39.1 |
| random forest T10 | 0.1018 | 0.1016 | 1.69 | 1.77 | 1.22 | 1.30 | 40.9 |
| random forest T1 | 0.0988 | 0.0986 | 1.50 | 1.39 | 1.02 | 0.92 | 40.0 |
| random forest RS10 | 0.1190 | 0.1055 | 1.43 | 1.05 | 0.96 | 0.58 | 41.1 |
| random forest RS1 | 0.1165 | 0.1015 | 1.58 | 1.68 | 1.12 | 1.21 | 39.1 |
| random forest RS10q | 0.1040 | 0.1062 | 1.90 | 1.35 | 1.44 | 0.87 | 40.3 |
| random forest RS1q | 0.1023 | 0.1027 | 1.57 | 1.55 | 1.10 | 1.08 | 39.1 |
| LSTM (LSTM, MSE rule, refit every session) | 0.1193 |  | 0.91 |  | 0.43 |  | 37.8 |
| LSTMq (LSTM, QLIKE rule, refit every session) | 0.1182 |  | 1.03 |  | 0.56 |  | 38.8 |
| LSTM10 (LSTM, MSE rule, refit every 10 sessions) | 0.1224 | 0.1220 | 1.08 | 1.13 | 0.60 | 0.66 | 38.3 |
| LSTM10q (LSTM, QLIKE rule, refit every 10 sessions) | 0.1235 | 0.1215 | 1.18 | 0.94 | 0.71 | 0.46 | 39.7 |
| per-bar ridge | 0.1005 | | 1.90 | | 1.44 | | 40.2 |

**`all_features`**

| forecast | QLIKE masked | QLIKE unmasked | Sharpe mid masked | Sharpe mid unmasked | Sharpe crossed masked | Sharpe crossed unmasked | % buy masked |
|---|---|---|---|---|---|---|---|
| LightGBM T10 | 0.1007 | 0.1013 | 1.84 | 1.75 | 1.37 | 1.29 | 38.3 |
| LightGBM T1 | 0.0982 | 0.0967 | 1.66 | 1.82 | 1.19 | 1.35 | 37.9 |
| LightGBM RS10 | 0.1035 | 0.1007 | 1.67 | 1.61 | 1.20 | 1.15 | 39.1 |
| LightGBM RS1 | 0.1018 | 0.1010 | 1.40 | 1.70 | 0.93 | 1.23 | 39.4 |
| LightGBM RS10q | 0.1030 | 0.1015 | 1.58 | 1.05 | 1.11 | 0.58 | 39.6 |
| LightGBM RS1q | 0.1010 | 0.0990 | 1.39 | 1.61 | 0.92 | 1.14 | 39.0 |
| XGBoost T10 | 0.1021 | 0.1012 | 1.18 | 1.54 | 0.71 | 1.07 | 38.8 |
| XGBoost T1 | 0.0991 | 0.0990 | 1.69 | 1.36 | 1.22 | 0.89 | 38.7 |
| XGBoost RS10 | 0.1117 | 0.1049 | 1.18 | 1.62 | 0.71 | 1.15 | 38.9 |
| XGBoost RS1 | 0.1097 | 0.1032 | 1.22 | 1.58 | 0.76 | 1.12 | 38.8 |
| XGBoost RS10q | 0.1047 | 0.1053 | 1.26 | 1.18 | 0.79 | 0.71 | 39.0 |
| XGBoost RS1q | 0.1023 | 0.1031 | 1.49 | 1.44 | 1.01 | 0.97 | 38.6 |
| random forest T10 | 0.1060 | 0.1059 | 1.00 | 1.27 | 0.53 | 0.80 | 41.6 |
| random forest T1 | 0.1039 | 0.1030 | 1.23 | 1.40 | 0.76 | 0.93 | 41.0 |
| random forest RS10 | 0.1195 | 0.1093 | 1.54 | 1.23 | 1.08 | 0.77 | 42.4 |
| random forest RS1 | 0.1171 | 0.1063 | 1.40 | 1.24 | 0.94 | 0.77 | 40.0 |
| random forest RS10q | 0.1063 | 0.1067 | 1.53 | 1.20 | 1.07 | 0.74 | 41.3 |
| random forest RS1q | 0.1034 | 0.1044 | 1.35 | 1.22 | 0.88 | 0.75 | 39.1 |
| LSTM (LSTM, MSE rule, refit every session) | 0.1305 |  | 1.33 |  | 0.85 |  | 39.4 |
| LSTMq (LSTM, QLIKE rule, refit every session) | 0.1304 |  | 1.09 |  | 0.61 |  | 40.5 |
| LSTM10 (LSTM, MSE rule, refit every 10 sessions) | 0.1453 | 0.1753 | 0.96 | 0.97 | 0.48 | 0.50 | 42.6 |
| LSTM10q (LSTM, QLIKE rule, refit every 10 sessions) | 0.1449 | 0.1561 | 1.07 | 0.64 | 0.59 | 0.16 | 40.8 |
| per-bar ridge | 0.1004 | | 1.66 | | 1.20 | | 36.1 |

## Masked - unmasked (a - b) with 95 % intervals

**T10 masked - T10 unmasked**

| model | input set | QLIKE diff (%) | QLIKE diff [interval] | Sharpe mid diff [interval] | Sharpe crossed diff [interval] | same side | max forecast change |
|---|---|---|---|---|---|---|---|
| LightGBM | `baseline` | +0.5 % | +0.0005 [-0.0006, +0.0016] | -0.18 [-0.80, +0.30] | -0.18 [-0.81, +0.30] | 97 % | 18.9 % |
| XGBoost | `baseline` | +0.4 % | +0.0004 [-0.0003, +0.0011] | -0.01 [-0.21, +0.16] | -0.01 [-0.21, +0.16] | 99 % | 16.7 % |
| random forest | `baseline` | +0.4 % | +0.0004 [-0.0003, +0.0013] | -0.00 [-0.25, +0.25] | -0.00 [-0.26, +0.25] | 98 % | 11.1 % |
| LightGBM | `live_feasible` | +1.0 % | +0.0010 [-0.0004, +0.0023] | -0.17 [-0.80, +0.39] | -0.17 [-0.80, +0.39] | 95 % | 17.2 % |
| XGBoost | `live_feasible` | +0.3 % | +0.0003 [-0.0005, +0.0013] | -0.05 [-0.37, +0.33] | -0.05 [-0.37, +0.33] | 98 % | 16.0 % |
| random forest | `live_feasible` | +0.2 % | +0.0002 [-0.0008, +0.0011] | -0.08 [-0.36, +0.20] | -0.08 [-0.37, +0.20] | 97 % | 11.8 % |
| LightGBM | `all_features` | -0.7 % | -0.0007 [-0.0023, +0.0011] | +0.09 [-0.34, +0.53] | +0.08 [-0.34, +0.53] | 94 % | 19.6 % |
| XGBoost | `all_features` | +0.9 % | +0.0009 [-0.0002, +0.0021] | -0.36 [-0.87, +0.08] | -0.37 [-0.88, +0.08] | 97 % | 17.5 % |
| random forest | `all_features` | +0.1 % | +0.0001 [-0.0008, +0.0011] | -0.27 [-0.62, +0.06] | -0.27 [-0.62, +0.07] | 97 % | 16.5 % |

Intervals entirely below / above zero (9 rows): QLIKE 0 / 0; Sharpe mid 0 / 0; Sharpe crossed 0 / 0.

**T1 masked - T1 unmasked**

| model | input set | QLIKE diff (%) | QLIKE diff [interval] | Sharpe mid diff [interval] | Sharpe crossed diff [interval] | same side | max forecast change |
|---|---|---|---|---|---|---|---|
| LightGBM | `baseline` | -0.0 % | -0.0000 [-0.0009, +0.0009] | -0.25 [-0.74, +0.15] | -0.25 [-0.74, +0.15] | 97 % | 13.0 % |
| XGBoost | `baseline` | +0.3 % | +0.0003 [-0.0003, +0.0009] | -0.07 [-0.30, +0.11] | -0.07 [-0.30, +0.11] | 99 % | 9.3 % |
| random forest | `baseline` | -0.0 % | -0.0000 [-0.0007, +0.0007] | +0.06 [-0.16, +0.31] | +0.07 [-0.16, +0.31] | 98 % | 11.6 % |
| LightGBM | `live_feasible` | -1.0 % | -0.0010 [-0.0023, +0.0003] | -0.06 [-0.49, +0.37] | -0.06 [-0.49, +0.37] | 95 % | 21.8 % |
| XGBoost | `live_feasible` | +1.3 % | +0.0013 [+0.0005, +0.0023] | -0.36 [-0.73, -0.06] | -0.36 [-0.73, -0.06] | 98 % | 15.5 % |
| random forest | `live_feasible` | +0.1 % | +0.0001 [-0.0008, +0.0011] | +0.10 [-0.17, +0.44] | +0.10 [-0.17, +0.44] | 98 % | 14.1 % |
| LightGBM | `all_features` | +1.5 % | +0.0015 [-0.0003, +0.0032] | -0.16 [-0.73, +0.35] | -0.16 [-0.74, +0.36] | 95 % | 20.7 % |
| XGBoost | `all_features` | +0.1 % | +0.0001 [-0.0007, +0.0009] | +0.33 [-0.11, +0.80] | +0.34 [-0.11, +0.81] | 97 % | 16.6 % |
| random forest | `all_features` | +0.9 % | +0.0009 [-0.0000, +0.0018] | -0.17 [-0.70, +0.29] | -0.17 [-0.71, +0.29] | 97 % | 16.5 % |

Intervals entirely below / above zero (9 rows): QLIKE 0 / 1; Sharpe mid 1 / 0; Sharpe crossed 1 / 0.

**RS10 masked - RS10 unmasked**

| model | input set | QLIKE diff (%) | QLIKE diff [interval] | Sharpe mid diff [interval] | Sharpe crossed diff [interval] | same side | max forecast change |
|---|---|---|---|---|---|---|---|
| LightGBM | `baseline` | -0.2 % | -0.0003 [-0.0012, +0.0007] | +0.08 [-0.09, +0.27] | +0.09 [-0.09, +0.27] | 98 % | 18.5 % |
| XGBoost | `baseline` | -0.2 % | -0.0003 [-0.0020, +0.0015] | -0.03 [-0.48, +0.41] | -0.02 [-0.47, +0.41] | 98 % | 33.5 % |
| random forest | `baseline` | -0.1 % | -0.0001 [-0.0017, +0.0015] | +0.16 [-0.46, +0.79] | +0.16 [-0.46, +0.80] | 96 % | 39.4 % |
| LightGBM | `live_feasible` | -0.9 % | -0.0009 [-0.0027, +0.0008] | -0.12 [-0.67, +0.43] | -0.12 [-0.68, +0.43] | 94 % | 25.8 % |
| XGBoost | `live_feasible` | -1.1 % | -0.0012 [-0.0039, +0.0013] | +0.16 [-0.21, +0.60] | +0.16 [-0.20, +0.60] | 96 % | 31.5 % |
| random forest | `live_feasible` | +12.8 % | +0.0135 [-0.0008, +0.0346] | +0.37 [-0.48, +1.24] | +0.38 [-0.48, +1.25] | 91 % | 137.2 % |
| LightGBM | `all_features` | +2.7 % | +0.0027 [-0.0004, +0.0063] | +0.06 [-0.59, +0.63] | +0.06 [-0.60, +0.63] | 95 % | 50.3 % |
| XGBoost | `all_features` | +6.5 % | +0.0068 [+0.0019, +0.0122] | -0.44 [-0.92, +0.04] | -0.44 [-0.93, +0.04] | 93 % | 50.4 % |
| random forest | `all_features` | +9.4 % | +0.0103 [+0.0010, +0.0253] | +0.31 [-0.29, +0.94] | +0.31 [-0.28, +0.94] | 93 % | 93.6 % |

Intervals entirely below / above zero (9 rows): QLIKE 0 / 2; Sharpe mid 0 / 0; Sharpe crossed 0 / 0.

**RS1 masked - RS1 unmasked**

| model | input set | QLIKE diff (%) | QLIKE diff [interval] | Sharpe mid diff [interval] | Sharpe crossed diff [interval] | same side | max forecast change |
|---|---|---|---|---|---|---|---|
| LightGBM | `baseline` | -0.2 % | -0.0003 [-0.0013, +0.0007] | -0.04 [-0.13, +0.04] | -0.04 [-0.14, +0.04] | 99 % | 12.2 % |
| XGBoost | `baseline` | -0.7 % | -0.0007 [-0.0025, +0.0011] | +0.16 [-0.06, +0.40] | +0.16 [-0.06, +0.41] | 98 % | 33.2 % |
| random forest | `baseline` | +0.4 % | +0.0004 [-0.0012, +0.0021] | +0.32 [-0.26, +0.88] | +0.32 [-0.27, +0.88] | 95 % | 39.4 % |
| LightGBM | `live_feasible` | -0.1 % | -0.0001 [-0.0021, +0.0018] | +0.01 [-0.55, +0.58] | +0.01 [-0.55, +0.58] | 94 % | 25.1 % |
| XGBoost | `live_feasible` | -1.1 % | -0.0011 [-0.0043, +0.0014] | -0.01 [-0.29, +0.28] | -0.01 [-0.29, +0.28] | 97 % | 29.7 % |
| random forest | `live_feasible` | +14.7 % | +0.0149 [-0.0002, +0.0376] | -0.09 [-0.77, +0.71] | -0.09 [-0.77, +0.72] | 91 % | 175.0 % |
| LightGBM | `all_features` | +0.7 % | +0.0007 [-0.0035, +0.0042] | -0.30 [-0.95, +0.34] | -0.30 [-0.96, +0.34] | 93 % | 38.5 % |
| XGBoost | `all_features` | +6.2 % | +0.0064 [+0.0017, +0.0117] | -0.36 [-0.85, +0.09] | -0.36 [-0.86, +0.09] | 93 % | 52.3 % |
| random forest | `all_features` | +10.1 % | +0.0107 [+0.0006, +0.0281] | +0.16 [-0.29, +0.72] | +0.17 [-0.29, +0.72] | 94 % | 142.0 % |

Intervals entirely below / above zero (9 rows): QLIKE 0 / 2; Sharpe mid 0 / 0; Sharpe crossed 0 / 0.

**LSTM10 masked - LSTM10 unmasked**

| model | input set | QLIKE diff (%) | QLIKE diff [interval] | Sharpe mid diff [interval] | Sharpe crossed diff [interval] | same side | max forecast change |
|---|---|---|---|---|---|---|---|
| LSTM | `baseline` | +0.2 % | +0.0002 [-0.0031, +0.0044] | -0.03 [-0.55, +0.49] | -0.03 [-0.55, +0.50] | 93 % | 40.3 % |
| LSTM | `live_feasible` | +0.3 % | +0.0003 [-0.0036, +0.0041] | -0.05 [-0.69, +0.57] | -0.06 [-0.71, +0.57] | 92 % | 50.3 % |
| LSTM | `all_features` | -17.1 % | -0.0300 [-0.0582, -0.0073] | -0.01 [-0.84, +0.83] | -0.02 [-0.84, +0.83] | 82 % | 180.0 % |

Intervals entirely below / above zero (3 rows): QLIKE 1 / 0; Sharpe mid 0 / 0; Sharpe crossed 0 / 0.

All `masked - unmasked` rows: Intervals entirely below / above zero (39 rows): QLIKE 1 / 5; Sharpe mid 1 / 0; Sharpe crossed 1 / 0.

## Masked ladder (a - b) with 95 % intervals

**T1 masked - T10 masked**

| model | input set | QLIKE diff (%) | QLIKE diff [interval] | Sharpe mid diff [interval] | Sharpe crossed diff [interval] | same side | max forecast change |
|---|---|---|---|---|---|---|---|
| LightGBM | `baseline` | -0.8 % | -0.0008 [-0.0024, +0.0007] | +0.18 [-0.19, +0.52] | +0.18 [-0.19, +0.52] | 96 % | 22.4 % |
| XGBoost | `baseline` | -0.8 % | -0.0008 [-0.0017, -0.0000] | +0.03 [-0.16, +0.29] | +0.03 [-0.16, +0.29] | 99 % | 16.6 % |
| random forest | `baseline` | -1.3 % | -0.0013 [-0.0035, +0.0008] | +0.27 [-0.42, +1.07] | +0.27 [-0.42, +1.08] | 93 % | 32.3 % |
| LightGBM | `live_feasible` | -3.6 % | -0.0036 [-0.0073, -0.0008] | -0.17 [-1.11, +0.59] | -0.18 [-1.11, +0.59] | 93 % | 55.5 % |
| XGBoost | `live_feasible` | -0.9 % | -0.0009 [-0.0029, +0.0008] | +0.26 [-0.05, +0.61] | +0.26 [-0.05, +0.60] | 97 % | 38.6 % |
| random forest | `live_feasible` | -3.0 % | -0.0031 [-0.0067, +0.0002] | -0.19 [-0.88, +0.48] | -0.20 [-0.89, +0.48] | 93 % | 61.0 % |
| LightGBM | `all_features` | -2.5 % | -0.0025 [-0.0072, +0.0009] | -0.18 [-1.14, +0.77] | -0.18 [-1.15, +0.78] | 92 % | 53.3 % |
| XGBoost | `all_features` | -2.9 % | -0.0030 [-0.0058, -0.0010] | +0.52 [+0.10, +1.02] | +0.52 [+0.10, +1.02] | 96 % | 37.7 % |
| random forest | `all_features` | -2.0 % | -0.0021 [-0.0058, +0.0012] | +0.23 [-0.29, +0.73] | +0.23 [-0.29, +0.73] | 94 % | 43.8 % |

Intervals entirely below / above zero (9 rows): QLIKE 3 / 0; Sharpe mid 0 / 1; Sharpe crossed 0 / 1.

**RS1 masked - RS10 masked**

| model | input set | QLIKE diff (%) | QLIKE diff [interval] | Sharpe mid diff [interval] | Sharpe crossed diff [interval] | same side | max forecast change |
|---|---|---|---|---|---|---|---|
| LightGBM | `baseline` | +0.1 % | +0.0001 [-0.0009, +0.0011] | -0.29 [-0.76, +0.09] | -0.29 [-0.76, +0.09] | 98 % | 11.0 % |
| XGBoost | `baseline` | -0.8 % | -0.0009 [-0.0018, +0.0002] | +0.10 [-0.11, +0.38] | +0.10 [-0.11, +0.39] | 98 % | 11.2 % |
| random forest | `baseline` | -0.7 % | -0.0008 [-0.0019, +0.0003] | +0.19 [-0.20, +0.57] | +0.19 [-0.20, +0.57] | 96 % | 19.4 % |
| LightGBM | `live_feasible` | +0.2 % | +0.0002 [-0.0015, +0.0016] | +0.16 [-0.19, +0.58] | +0.17 [-0.19, +0.58] | 97 % | 26.0 % |
| XGBoost | `live_feasible` | -1.9 % | -0.0020 [-0.0056, +0.0001] | -0.13 [-0.52, +0.29] | -0.13 [-0.52, +0.30] | 96 % | 33.4 % |
| random forest | `live_feasible` | -2.2 % | -0.0026 [-0.0058, +0.0004] | +0.16 [-0.32, +0.62] | +0.16 [-0.33, +0.62] | 93 % | 32.1 % |
| LightGBM | `all_features` | -1.7 % | -0.0017 [-0.0053, +0.0007] | -0.27 [-0.61, +0.06] | -0.27 [-0.61, +0.06] | 96 % | 27.1 % |
| XGBoost | `all_features` | -1.8 % | -0.0020 [-0.0041, -0.0004] | +0.04 [-0.34, +0.45] | +0.04 [-0.34, +0.46] | 95 % | 23.5 % |
| random forest | `all_features` | -2.1 % | -0.0025 [-0.0068, +0.0004] | -0.14 [-0.61, +0.28] | -0.14 [-0.62, +0.28] | 95 % | 29.3 % |

Intervals entirely below / above zero (9 rows): QLIKE 1 / 0; Sharpe mid 0 / 0; Sharpe crossed 0 / 0.

**RS1 masked - T1 masked**

| model | input set | QLIKE diff (%) | QLIKE diff [interval] | Sharpe mid diff [interval] | Sharpe crossed diff [interval] | same side | max forecast change |
|---|---|---|---|---|---|---|---|
| LightGBM | `baseline` | +7.2 % | +0.0075 [+0.0022, +0.0146] | -0.28 [-0.89, +0.30] | -0.28 [-0.89, +0.30] | 92 % | 71.3 % |
| XGBoost | `baseline` | +5.2 % | +0.0054 [+0.0003, +0.0132] | +0.27 [-0.18, +0.76] | +0.27 [-0.18, +0.77] | 95 % | 78.8 % |
| random forest | `baseline` | +4.6 % | +0.0048 [-0.0023, +0.0146] | +0.08 [-0.55, +0.67] | +0.08 [-0.55, +0.67] | 91 % | 75.1 % |
| LightGBM | `live_feasible` | +4.1 % | +0.0039 [-0.0002, +0.0085] | +0.21 [-0.61, +1.07] | +0.22 [-0.60, +1.08] | 90 % | 64.0 % |
| XGBoost | `live_feasible` | +6.5 % | +0.0064 [+0.0012, +0.0129] | -0.19 [-0.77, +0.36] | -0.18 [-0.77, +0.37] | 93 % | 62.8 % |
| random forest | `live_feasible` | +17.9 % | +0.0177 [+0.0021, +0.0410] | +0.09 [-0.66, +0.91] | +0.09 [-0.67, +0.92] | 89 % | 187.3 % |
| LightGBM | `all_features` | +3.6 % | +0.0036 [-0.0004, +0.0083] | -0.26 [-1.05, +0.61] | -0.26 [-1.05, +0.63] | 91 % | 90.1 % |
| XGBoost | `all_features` | +10.7 % | +0.0106 [+0.0046, +0.0174] | -0.47 [-1.06, +0.11] | -0.47 [-1.06, +0.12] | 90 % | 67.0 % |
| random forest | `all_features` | +12.6 % | +0.0131 [+0.0014, +0.0311] | +0.17 [-0.59, +0.98] | +0.18 [-0.59, +1.00] | 89 % | 200.7 % |

Intervals entirely below / above zero (9 rows): QLIKE 0 / 6; Sharpe mid 0 / 0; Sharpe crossed 0 / 0.

**LSTM masked - LSTM10 masked**

| model | input set | QLIKE diff (%) | QLIKE diff [interval] | Sharpe mid diff [interval] | Sharpe crossed diff [interval] | same side | max forecast change |
|---|---|---|---|---|---|---|---|
| LSTM | `baseline` | -1.9 % | -0.0021 [-0.0037, -0.0003] | +0.09 [-0.17, +0.34] | +0.09 [-0.17, +0.34] | 96 % | 45.1 % |
| LSTM | `live_feasible` | -2.5 % | -0.0031 [-0.0096, +0.0039] | -0.17 [-0.80, +0.48] | -0.17 [-0.80, +0.48] | 89 % | 158.0 % |
| LSTM | `all_features` | -10.1 % | -0.0147 [-0.0261, -0.0030] | +0.37 [-0.39, +1.22] | +0.37 [-0.39, +1.21] | 85 % | 191.0 % |

Intervals entirely below / above zero (3 rows): QLIKE 2 / 0; Sharpe mid 0 / 0; Sharpe crossed 0 / 0.

All `masked ladder` rows: Intervals entirely below / above zero (30 rows): QLIKE 6 / 6; Sharpe mid 0 / 1; Sharpe crossed 0 / 1.

## Replaced table (a - b) with 95 % intervals

**LSTM masked (daily) - LSTM10 unmasked**

| model | input set | QLIKE diff (%) | QLIKE diff [interval] | Sharpe mid diff [interval] | Sharpe crossed diff [interval] | same side | max forecast change |
|---|---|---|---|---|---|---|---|
| LSTM | `baseline` | -1.7 % | -0.0019 [-0.0051, +0.0021] | +0.06 [-0.49, +0.61] | +0.06 [-0.49, +0.62] | 92 % | 40.4 % |
| LSTM | `live_feasible` | -2.3 % | -0.0028 [-0.0096, +0.0049] | -0.22 [-1.03, +0.60] | -0.23 [-1.03, +0.60] | 87 % | 152.1 % |
| LSTM | `all_features` | -25.5 % | -0.0448 [-0.0691, -0.0256] | +0.36 [-0.64, +1.40] | +0.35 [-0.64, +1.39] | 80 % | 258.7 % |

Intervals entirely below / above zero (3 rows): QLIKE 1 / 0; Sharpe mid 0 / 0; Sharpe crossed 0 / 0.

All `replaced table` rows: Intervals entirely below / above zero (3 rows): QLIKE 1 / 0; Sharpe mid 0 / 0; Sharpe crossed 0 / 0.

## Masked - ridge (a - b) with 95 % intervals

**T10 masked - ridge**

| model | input set | QLIKE diff (%) | QLIKE diff [interval] | Sharpe mid diff [interval] | Sharpe crossed diff [interval] | same side | max forecast change |
|---|---|---|---|---|---|---|---|
| LightGBM | `baseline` | +6.8 % | +0.0066 [+0.0025, +0.0109] | -0.40 [-1.05, +0.22] | -0.40 [-1.05, +0.22] | 89 % | 56.7 % |
| XGBoost | `baseline` | +6.1 % | +0.0059 [+0.0025, +0.0095] | -0.30 [-0.91, +0.30] | -0.30 [-0.90, +0.30] | 91 % | 62.3 % |
| random forest | `baseline` | +6.9 % | +0.0068 [+0.0015, +0.0120] | -0.32 [-1.16, +0.48] | -0.32 [-1.17, +0.48] | 87 % | 118.9 % |
| LightGBM | `live_feasible` | -1.4 % | -0.0014 [-0.0068, +0.0037] | -0.31 [-1.04, +0.41] | -0.32 [-1.04, +0.41] | 85 % | 381.7 % |
| XGBoost | `live_feasible` | -1.1 % | -0.0011 [-0.0070, +0.0048] | -0.63 [-1.51, +0.16] | -0.64 [-1.54, +0.16] | 86 % | 354.7 % |
| random forest | `live_feasible` | +1.3 % | +0.0013 [-0.0064, +0.0082] | -0.21 [-0.84, +0.42] | -0.22 [-0.85, +0.41] | 85 % | 485.2 % |
| LightGBM | `all_features` | +0.3 % | +0.0003 [-0.0060, +0.0079] | +0.17 [-0.66, +0.96] | +0.17 [-0.66, +0.97] | 84 % | 388.1 % |
| XGBoost | `all_features` | +1.7 % | +0.0017 [-0.0056, +0.0111] | -0.49 [-1.16, +0.24] | -0.49 [-1.16, +0.24] | 83 % | 315.1 % |
| random forest | `all_features` | +5.6 % | +0.0056 [-0.0031, +0.0149] | -0.66 [-1.35, +0.03] | -0.66 [-1.35, +0.03] | 81 % | 574.0 % |

Intervals entirely below / above zero (9 rows): QLIKE 0 / 3; Sharpe mid 0 / 0; Sharpe crossed 0 / 0.

**T1 masked - ridge**

| model | input set | QLIKE diff (%) | QLIKE diff [interval] | Sharpe mid diff [interval] | Sharpe crossed diff [interval] | same side | max forecast change |
|---|---|---|---|---|---|---|---|
| LightGBM | `baseline` | +5.9 % | +0.0058 [+0.0019, +0.0097] | -0.22 [-0.82, +0.37] | -0.22 [-0.82, +0.37] | 90 % | 56.3 % |
| XGBoost | `baseline` | +5.2 % | +0.0051 [+0.0017, +0.0085] | -0.27 [-0.92, +0.41] | -0.26 [-0.92, +0.41] | 91 % | 73.4 % |
| random forest | `baseline` | +5.5 % | +0.0054 [+0.0005, +0.0105] | -0.06 [-0.74, +0.61] | -0.05 [-0.74, +0.62] | 86 % | 128.9 % |
| LightGBM | `live_feasible` | -5.0 % | -0.0050 [-0.0116, +0.0004] | -0.48 [-1.58, +0.57] | -0.49 [-1.60, +0.57] | 85 % | 363.6 % |
| XGBoost | `live_feasible` | -1.9 % | -0.0020 [-0.0082, +0.0036] | -0.37 [-1.27, +0.50] | -0.38 [-1.29, +0.50] | 86 % | 332.5 % |
| random forest | `live_feasible` | -1.8 % | -0.0018 [-0.0104, +0.0054] | -0.41 [-1.21, +0.39] | -0.42 [-1.22, +0.39] | 85 % | 455.5 % |
| LightGBM | `all_features` | -2.2 % | -0.0022 [-0.0084, +0.0041] | -0.00 [-0.73, +0.78] | -0.01 [-0.75, +0.78] | 84 % | 365.7 % |
| XGBoost | `all_features` | -1.3 % | -0.0013 [-0.0089, +0.0068] | +0.03 [-0.64, +0.74] | +0.03 [-0.64, +0.74] | 84 % | 311.4 % |
| random forest | `all_features` | +3.5 % | +0.0036 [-0.0057, +0.0124] | -0.43 [-1.09, +0.27] | -0.44 [-1.10, +0.26] | 81 % | 531.1 % |

Intervals entirely below / above zero (9 rows): QLIKE 0 / 3; Sharpe mid 0 / 0; Sharpe crossed 0 / 0.

**RS10 masked - ridge**

| model | input set | QLIKE diff (%) | QLIKE diff [interval] | Sharpe mid diff [interval] | Sharpe crossed diff [interval] | same side | max forecast change |
|---|---|---|---|---|---|---|---|
| LightGBM | `baseline` | +13.5 % | +0.0132 [+0.0070, +0.0212] | -0.22 [-0.75, +0.34] | -0.21 [-0.74, +0.35] | 89 % | 66.8 % |
| XGBoost | `baseline` | +11.5 % | +0.0113 [+0.0053, +0.0198] | -0.10 [-0.56, +0.39] | -0.10 [-0.55, +0.39] | 91 % | 69.0 % |
| random forest | `baseline` | +11.3 % | +0.0111 [+0.0046, +0.0196] | -0.16 [-0.85, +0.50] | -0.16 [-0.84, +0.51] | 89 % | 67.6 % |
| LightGBM | `live_feasible` | -1.3 % | -0.0013 [-0.0066, +0.0041] | -0.43 [-1.22, +0.35] | -0.44 [-1.23, +0.35] | 85 % | 358.2 % |
| XGBoost | `live_feasible` | +6.5 % | +0.0065 [-0.0003, +0.0153] | -0.44 [-1.14, +0.25] | -0.44 [-1.14, +0.25] | 84 % | 551.8 % |
| random forest | `live_feasible` | +18.4 % | +0.0185 [+0.0049, +0.0393] | -0.48 [-1.17, +0.24] | -0.48 [-1.18, +0.23] | 85 % | 880.3 % |
| LightGBM | `all_features` | +3.1 % | +0.0031 [-0.0036, +0.0122] | +0.00 [-0.76, +0.73] | +0.01 [-0.77, +0.73] | 83 % | 423.4 % |
| XGBoost | `all_features` | +11.3 % | +0.0113 [+0.0017, +0.0228] | -0.48 [-1.42, +0.45] | -0.48 [-1.43, +0.45] | 81 % | 439.3 % |
| random forest | `all_features` | +19.1 % | +0.0192 [+0.0059, +0.0407] | -0.12 [-1.01, +0.78] | -0.12 [-1.01, +0.79] | 80 % | 755.5 % |

Intervals entirely below / above zero (9 rows): QLIKE 0 / 6; Sharpe mid 0 / 0; Sharpe crossed 0 / 0.

**RS1 masked - ridge**

| model | input set | QLIKE diff (%) | QLIKE diff [interval] | Sharpe mid diff [interval] | Sharpe crossed diff [interval] | same side | max forecast change |
|---|---|---|---|---|---|---|---|
| LightGBM | `baseline` | +13.6 % | +0.0133 [+0.0073, +0.0212] | -0.51 [-1.10, +0.14] | -0.50 [-1.10, +0.15] | 88 % | 66.7 % |
| XGBoost | `baseline` | +10.7 % | +0.0105 [+0.0044, +0.0186] | +0.00 [-0.47, +0.47] | +0.01 [-0.47, +0.48] | 90 % | 69.0 % |
| random forest | `baseline` | +10.4 % | +0.0102 [+0.0038, +0.0189] | +0.03 [-0.61, +0.59] | +0.03 [-0.61, +0.60] | 90 % | 67.6 % |
| LightGBM | `live_feasible` | -1.2 % | -0.0012 [-0.0066, +0.0040] | -0.27 [-1.07, +0.57] | -0.27 [-1.08, +0.57] | 86 % | 358.5 % |
| XGBoost | `live_feasible` | +4.5 % | +0.0045 [-0.0015, +0.0113] | -0.56 [-1.30, +0.18] | -0.57 [-1.30, +0.18] | 85 % | 552.1 % |
| random forest | `live_feasible` | +15.8 % | +0.0159 [+0.0035, +0.0350] | -0.32 [-0.91, +0.30] | -0.32 [-0.92, +0.30] | 85 % | 859.3 % |
| LightGBM | `all_features` | +1.4 % | +0.0014 [-0.0048, +0.0085] | -0.26 [-1.08, +0.48] | -0.26 [-1.09, +0.48] | 85 % | 422.5 % |
| XGBoost | `all_features` | +9.2 % | +0.0093 [+0.0003, +0.0195] | -0.44 [-1.33, +0.41] | -0.44 [-1.33, +0.41] | 81 % | 417.3 % |
| random forest | `all_features` | +16.6 % | +0.0167 [+0.0053, +0.0346] | -0.26 [-1.20, +0.62] | -0.26 [-1.20, +0.62] | 81 % | 678.2 % |

Intervals entirely below / above zero (9 rows): QLIKE 0 / 6; Sharpe mid 0 / 0; Sharpe crossed 0 / 0.

**LSTM masked - ridge**

| model | input set | QLIKE diff (%) | QLIKE diff [interval] | Sharpe mid diff [interval] | Sharpe crossed diff [interval] | same side | max forecast change |
|---|---|---|---|---|---|---|---|
| LSTM | `baseline` | +6.8 % | +0.0067 [+0.0019, +0.0119] | +0.04 [-0.50, +0.63] | +0.04 [-0.50, +0.63] | 89 % | 84.5 % |
| LSTM | `live_feasible` | +18.6 % | +0.0187 [+0.0058, +0.0338] | -1.00 [-2.14, +0.19] | -1.01 [-2.15, +0.18] | 77 % | 525.7 % |
| LSTM | `all_features` | +30.0 % | +0.0301 [+0.0167, +0.0454] | -0.33 [-1.69, +0.99] | -0.35 [-1.71, +0.98] | 75 % | 460.3 % |

Intervals entirely below / above zero (3 rows): QLIKE 0 / 3; Sharpe mid 0 / 0; Sharpe crossed 0 / 0.

**LSTM10 masked - ridge**

| model | input set | QLIKE diff (%) | QLIKE diff [interval] | Sharpe mid diff [interval] | Sharpe crossed diff [interval] | same side | max forecast change |
|---|---|---|---|---|---|---|---|
| LSTM | `baseline` | +8.9 % | +0.0088 [+0.0043, +0.0135] | -0.05 [-0.60, +0.55] | -0.05 [-0.60, +0.55] | 89 % | 84.4 % |
| LSTM | `live_feasible` | +21.7 % | +0.0218 [+0.0079, +0.0370] | -0.83 [-1.85, +0.26] | -0.84 [-1.86, +0.26] | 79 % | 362.5 % |
| LSTM | `all_features` | +44.7 % | +0.0449 [+0.0305, +0.0607] | -0.71 [-1.81, +0.44] | -0.72 [-1.84, +0.43] | 76 % | 516.0 % |

Intervals entirely below / above zero (3 rows): QLIKE 0 / 3; Sharpe mid 0 / 0; Sharpe crossed 0 / 0.

All `masked - ridge` rows: Intervals entirely below / above zero (42 rows): QLIKE 0 / 24; Sharpe mid 0 / 0; Sharpe crossed 0 / 0.

## CPU class (a - b) with 95 % intervals

**T10 masked, epyc-7513 (fleet) - T10 masked, xeon-4116 (class gate)**

| model | input set | QLIKE diff (%) | QLIKE diff [interval] | Sharpe mid diff [interval] | Sharpe crossed diff [interval] | same side | max forecast change |
|---|---|---|---|---|---|---|---|
| LightGBM | `live_feasible` | -0.1 % | -0.0001 [-0.0008, +0.0005] | -0.26 [-0.68, +0.11] | -0.26 [-0.69, +0.11] | 97 % | 7.9 % |

Intervals entirely below / above zero (1 rows): QLIKE 0 / 0; Sharpe mid 0 / 0; Sharpe crossed 0 / 0.

**T1 masked, epyc-7513 (fleet) - T1 masked, xeon-4116 (class gate)**

| model | input set | QLIKE diff (%) | QLIKE diff [interval] | Sharpe mid diff [interval] | Sharpe crossed diff [interval] | same side | max forecast change |
|---|---|---|---|---|---|---|---|
| LightGBM | `live_feasible` | -0.1 % | -0.0001 [-0.0008, +0.0008] | +0.07 [-0.30, +0.42] | +0.06 [-0.31, +0.42] | 98 % | 8.6 % |

Intervals entirely below / above zero (1 rows): QLIKE 0 / 0; Sharpe mid 0 / 0; Sharpe crossed 0 / 0.

All `CPU class` rows: Intervals entirely below / above zero (2 rows): QLIKE 0 / 0; Sharpe mid 0 / 0; Sharpe crossed 0 / 0.

## Cluster use (the masked re-run's own jobs, sacct)

| stage | allocations | CPUs per allocation | allocated CPU-hours | used CPU-hours | peak memory per allocation (GiB) | first start | last end |
|---|---|---|---|---|---|---|---|
| tm_gates_cadence | 1 | 20-20 | 3.2 | 0.6 | 18.8 | 2026-09-29 22:09:38 | 2026-09-29 22:19:23 |
| tm_gates_lstm | 2 | 20-20 | 13.8 | 1.3 | 4.4 | 2026-09-29 22:18:29 | 2026-09-29 23:59:31 |
| tm_gates_h | 1 | 20-20 | 2.2 | 1.4 | 20.0 | 2026-09-29 22:19:32 | 2026-09-29 22:26:04 |
| tm_gates_h_trees | 1 | 20-20 | 2.0 | 1.2 | 18.9 | 2026-09-29 23:04:38 | 2026-09-29 23:10:44 |
| tm_gates_h_lstm | 1 | 20-20 | 3.7 | 1.8 | 14.1 | 2026-09-29 23:08:28 | 2026-09-29 23:19:34 |
| tm_canary | 1 | 20-20 | 7.6 | 1.9 | 26.3 | 2026-09-29 23:15:37 | 2026-09-29 23:38:29 |
| tm_rs10_canary | 1 | 20-20 | 1.0 | 0.3 | 7.6 | 2026-09-29 23:19:34 | 2026-09-29 23:22:29 |
| tm_rs1_canary | 1 | 20-20 | 2.4 | 1.7 | 7.6 | 2026-09-29 23:22:29 | 2026-09-29 23:29:36 |
| lstm_mask_canary | 1 | 40-40 | 5.5 | 1.3 | 11.8 | 2026-09-29 23:36:15 | 2026-09-29 23:44:33 |
| lstm_mask10_canary | 1 | 20-20 | 1.1 | 0.8 | 10.7 | 2026-09-29 23:29:36 | 2026-09-29 23:33:02 |
| tm_merge | 1 | 1-1 | 0.0 | 0.0 | 0.1 | 2026-09-30 00:20:50 | 2026-09-30 00:21:38 |
| tm_untuned | 10 | 20-20 | 46.6 | 22.8 | 40.0 | 2026-09-29 23:54:23 | 2026-09-30 00:18:22 |
| tm_rs10 | 54 | 20-20 | 15.7 | 4.2 | 10.0 | 2026-09-29 23:53:52 | 2026-09-30 00:07:08 |
| tm_rs1 | 54 | 20-20 | 30.7 | 21.7 | 10.1 | 2026-09-30 00:06:50 | 2026-09-30 00:20:37 |
| tm_classgate | 1 | 8-8 | 3.6 | 1.4 | 7.7 | 2026-09-29 23:47:39 | 2026-09-30 00:14:41 |
| tm_classgate_merge | 1 | 1-1 | 0.0 | 0.0 | 0.0 | 2026-09-30 00:15:07 | 2026-09-30 00:15:25 |
| lm_merge | 1 | 1-1 | 0.2 | 0.0 | 0.1 | 2026-09-30 00:41:01 | 2026-09-30 00:50:17 |
| lstm_mask | 18 | 20-20 | 37.6 | 34.0 | 14.5 | 2026-09-30 00:15:07 | 2026-09-30 00:40:35 |
| lstm_mask10 | 18 | 20-20 | 9.4 | 6.1 | 13.5 | 2026-09-30 00:18:14 | 2026-09-30 00:24:21 |

All stages: 186.4 allocated CPU-hours (102.5 used); peak concurrent allocated CPUs 408; wall-clock 2026-09-29 22:09:38 .. 2026-09-30 00:50:17.

## Gates

333 gates checked, 0 failed (`mask_gates.csv`): every table's profile B equals the common target's; every masked tree and LSTM table carries its merged masked arm's pred_adj exactly; every forecast covers the same trade days; the per-day trade returns reproduce `score_linear_subsection.trade_1530`'s Sharpe; every arm's chunks ran with the mask; the kept sets agree across models at common refit rows.

Wording: sign(s) = the rule above; QLIKE differences are losses (negative = a lower loss); nothing here is a recommendation.
