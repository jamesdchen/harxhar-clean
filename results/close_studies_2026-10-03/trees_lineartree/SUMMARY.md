# Linear leaves for the 16:00 LightGBM (`linear_tree`)

Written by `experiments/close_trees_lineartree.py analyze` from the CSVs in this folder.

**Question (the user's):** do the 16:00-bar trees lose to the linear models because they approximate a mostly linear signal (the HAR ladder) with step functions? LightGBM's `linear_tree=True` fits, in every leaf, a linear model in the numerical features used on that leaf's branch instead of a constant (the first tree keeps constant leaves). One setting changed; everything else is the shipped configuration (`specs/causal_tune_trees.py` LGBM_PARAMS, `min_child_samples` 8 for the 2000-row window, seed 42, one thread), on the captured 16:00-bar design, window 2000 sessions, the per-window column mask on, a refit every 10 sessions (or every session where marked), forecasts from forecast row 130 (2019-01) on.

**Scorer:** the master table's research convention (`dense_vs_sparse_1530.research_frame` -> `deck_panel` -> `point`) on the 866 trade days 2020-01-03 .. 2024-04-30: QLIKE of the recalibrated 16:00 forecast, sign(s) straddle Sharpe at mid and crossed. Point estimates (the circular block bootstrap is off, commit b761b28). Differences against the local control of the same design and cadence: Diebold-Mariano on daily QLIKE (negative = linear leaves have lower loss) and the HAC t of the paired daily P&L difference (positive = linear leaves earn more).

## Gate

Stored tables re-scored here against `results/close_master_table/master_table.csv`: 48 of 48 checks within 1e-09 (largest difference 4.22e-15), on all 1469 forecast rows and on rows 130.. alike.

Local controls against the stored cluster runs (LightGBM 4.6.0 on the cluster, writeup/CAMPAIGN_16H_2026-09-29.md; same configuration otherwise):

| comparison | item | value |
|---|---|---|
| local control (LightGBM 4.7.0) vs stored subtree_lgbm_all_features (cluster run) | ctrl_all_r10: largest absolute forecast difference (adjusted scale) | 0.06283 |
| local control (LightGBM 4.7.0) vs stored subtree_lgbm_all_features (cluster run) | ctrl_all_r10: median absolute forecast difference (adjusted scale) | 0.00940 |
| local control (LightGBM 4.7.0) vs stored subtree_lgbm_all_features (cluster run) | ctrl_all_r10: QLIKE local - stored | 0.00001 |
| local control (LightGBM 4.7.0) vs stored subtree_lgbm_all_features (cluster run) | ctrl_all_r10: Sharpe mid local - stored | 0.14986 |
| local control (LightGBM 4.7.0) vs stored subtree_lgbm_all_features (cluster run) | ctrl_all_r10: Sharpe crossed local - stored | 0.15033 |
| local control (LightGBM 4.7.0) vs stored subtree_lgbm_all_features (cluster run) | ctrl_all_r10: DM (daily QLIKE) | 0.01455 |
| local control (LightGBM 4.7.0) vs stored subtree_lgbm_all_features (cluster run) | ctrl_all_r10: HAC t (daily P&L mid) | 0.98901 |
| local control (LightGBM 4.7.0) vs stored subtree_lgbm_baseline (cluster run) | ctrl_base_r10: largest absolute forecast difference (adjusted scale) | 0.00000 |
| local control (LightGBM 4.7.0) vs stored subtree_lgbm_baseline (cluster run) | ctrl_base_r10: median absolute forecast difference (adjusted scale) | 0.00000 |
| local control (LightGBM 4.7.0) vs stored subtree_lgbm_baseline (cluster run) | ctrl_base_r10: QLIKE local - stored | 0.00000 |
| local control (LightGBM 4.7.0) vs stored subtree_lgbm_baseline (cluster run) | ctrl_base_r10: Sharpe mid local - stored | 0.00000 |
| local control (LightGBM 4.7.0) vs stored subtree_lgbm_baseline (cluster run) | ctrl_base_r10: Sharpe crossed local - stored | 0.00000 |
| local control (LightGBM 4.7.0) vs stored subtree_lgbm_baseline (cluster run) | ctrl_base_r10: DM (daily QLIKE) | 0.65659 |
| local control (LightGBM 4.7.0) vs stored subtree_lgbm_baseline (cluster run) | ctrl_base_r10: HAC t (daily P&L mid) |  |
| local control (LightGBM 4.7.0) vs stored subtree_daily_baseline_lgbm (cluster run) | ctrl_base_r1: largest absolute forecast difference (adjusted scale) | 0.00000 |
| local control (LightGBM 4.7.0) vs stored subtree_daily_baseline_lgbm (cluster run) | ctrl_base_r1: median absolute forecast difference (adjusted scale) | 0.00000 |
| local control (LightGBM 4.7.0) vs stored subtree_daily_baseline_lgbm (cluster run) | ctrl_base_r1: QLIKE local - stored | 0.00000 |
| local control (LightGBM 4.7.0) vs stored subtree_daily_baseline_lgbm (cluster run) | ctrl_base_r1: Sharpe mid local - stored | 0.00000 |
| local control (LightGBM 4.7.0) vs stored subtree_daily_baseline_lgbm (cluster run) | ctrl_base_r1: Sharpe crossed local - stored | 0.00000 |
| local control (LightGBM 4.7.0) vs stored subtree_daily_baseline_lgbm (cluster run) | ctrl_base_r1: DM (daily QLIKE) | 1.80597 |
| local control (LightGBM 4.7.0) vs stored subtree_daily_baseline_lgbm (cluster run) | ctrl_base_r1: HAC t (daily P&L mid) |  |

## Control vs linear leaves

| arm | design | refit every | linear_lambda | QLIKE | Sharpe mid | Sharpe crossed | % buy | DM vs control | DM p | HAC t mid vs control | HAC t crossed vs control | fit s / refit | CPU s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ctrl_all_r10 | all_features | 10 |  | 0.1007 | 1.99 | 1.52 | 38.7 |  |  |  |  | 10.17 | 1353 |
| lt_all_r10 | all_features | 10 | 0.000 | 0.5269 | 0.42 | -0.05 | 70.3 | 5.55 | 0.000 | -2.87 | -2.86 | 10.79 | 1394 |
| lt_all_r10_lam1 | all_features | 10 | 1.000 | 0.0997 | 1.69 | 1.22 | 39.1 | -0.77 | 0.441 | -1.16 | -1.16 | 10.43 | 1414 |
| lt_all_r10_lam10 | all_features | 10 | 10.000 | 0.0994 | 1.43 | 0.97 | 40.1 | -1.19 | 0.233 | -2.21 | -2.21 | 9.97 | 1353 |
| ctrl_base_r1 | baseline | 1 |  | 0.1041 | 0.99 | 0.52 | 41.3 |  |  |  |  | 0.26 | 405 |
| lt_base_r1 | baseline | 1 | 0.000 | 0.1026 | 0.61 | 0.14 | 41.9 | -1.64 | 0.101 | -0.82 | -0.82 | 0.39 | 719 |
| ctrl_base_r10 | baseline | 10 |  | 0.1049 | 0.82 | 0.35 | 41.2 |  |  |  |  | 0.26 | 42 |
| lt_base_r10 | baseline | 10 | 0.000 | 0.1031 | 1.12 | 0.65 | 41.6 | -1.97 | 0.049 | 1.66 | 1.66 | 0.43 | 75 |

## Master-table rows on the same days

| stored forecast | design | QLIKE | Sharpe mid | Sharpe crossed |
|---|---|---|---|---|
| subtree_lgbm_all_features | all_features | 0.1007 | 1.84 | 1.37 |
| subtree_daily_all_features_lgbm | all_features | 0.0982 | 1.66 | 1.19 |
| sub_lasso_all_features | all_features | 0.0998 | 1.45 | 0.98 |
| sub_ridge_all_features | all_features | 0.1004 | 1.66 | 1.20 |
| subtree_lgbm_baseline | baseline | 0.1049 | 0.82 | 0.35 |
| subtree_daily_baseline_lgbm | baseline | 0.1041 | 0.99 | 0.52 |
| sub_lasso_baseline | baseline | 0.0972 | 1.51 | 1.04 |
| sub_ridge_baseline | baseline | 0.0982 | 1.22 | 0.74 |

## Against the linear rows (each arm minus the stored linear forecast)

| arm | linear row | QLIKE diff | DM | DM p | Sharpe mid diff | HAC t mid |
|---|---|---|---|---|---|---|
| lt_all_r10 | sub_lasso_all_features | 0.4271 | 5.56 | 0.000 | -1.03 | -2.11 |
| lt_all_r10 | sub_ridge_all_features | 0.4265 | 5.57 | 0.000 | -1.25 | -2.34 |
| ctrl_all_r10 | sub_lasso_all_features | 0.0009 | 0.28 | 0.779 | 0.54 | 1.46 |
| ctrl_all_r10 | sub_ridge_all_features | 0.0003 | 0.07 | 0.942 | 0.32 | 0.79 |
| lt_all_r10_lam1 | sub_lasso_all_features | -0.0000 | -0.02 | 0.986 | 0.24 | 0.65 |
| lt_all_r10_lam1 | sub_ridge_all_features | -0.0007 | -0.19 | 0.853 | 0.02 | 0.05 |
| lt_all_r10_lam10 | sub_lasso_all_features | -0.0004 | -0.12 | 0.902 | -0.01 | -0.03 |
| lt_all_r10_lam10 | sub_ridge_all_features | -0.0010 | -0.28 | 0.783 | -0.23 | -0.61 |
| ctrl_base_r10 | sub_lasso_baseline | 0.0077 | 3.95 | 0.000 | -0.69 | -1.91 |
| ctrl_base_r10 | sub_ridge_baseline | 0.0066 | 3.43 | 0.001 | -0.40 | -1.18 |
| lt_base_r10 | sub_lasso_baseline | 0.0059 | 2.73 | 0.006 | -0.39 | -1.06 |
| lt_base_r10 | sub_ridge_baseline | 0.0048 | 2.25 | 0.024 | -0.10 | -0.29 |
| ctrl_base_r1 | sub_lasso_baseline | 0.0069 | 3.75 | 0.000 | -0.51 | -1.55 |
| ctrl_base_r1 | sub_ridge_baseline | 0.0058 | 3.15 | 0.002 | -0.22 | -0.69 |
| lt_base_r1 | sub_lasso_baseline | 0.0054 | 2.52 | 0.012 | -0.90 | -1.65 |
| lt_base_r1 | sub_ridge_baseline | 0.0044 | 2.01 | 0.044 | -0.61 | -1.11 |

## Forecast extremes (adjusted scale: the model's target, before the scorer squares it)

| arm | min | max | largest abs forecast | its date | n < 0 | n > max target of own window | n < min target of own window | largest abs diff vs control | median abs diff vs control |
|---|---|---|---|---|---|---|---|---|---|
| lt_all_r10 | -176.2377 | 89.3903 | 176.2377 | 2024-04-16 | 22 | 14 | 27 | 177.6095 | 0.0305 |
| ctrl_all_r10 | 0.3230 | 3.6539 | 3.6539 | 2020-02-28 | 0 | 0 | 0 |  |  |
| lt_all_r10_lam1 | 0.2620 | 4.3064 | 4.3064 | 2020-02-28 | 0 | 0 | 0 | 0.6525 | 0.0163 |
| lt_all_r10_lam10 | 0.2906 | 4.1144 | 4.1144 | 2020-02-28 | 0 | 0 | 0 | 0.4605 | 0.0152 |
| ctrl_base_r10 | 0.3622 | 3.7174 | 3.7174 | 2020-02-28 | 0 | 0 | 0 |  |  |
| lt_base_r10 | 0.2962 | 3.7554 | 3.7554 | 2020-02-28 | 0 | 0 | 0 | 1.1194 | 0.0119 |
| ctrl_base_r1 | 0.3654 | 3.9254 | 3.9254 | 2020-03-03 | 0 | 0 | 0 |  |  |
| lt_base_r1 | 0.2970 | 3.7554 | 3.7554 | 2020-02-28 | 0 | 0 | 0 | 1.0586 | 0.0123 |

## Leaf models (linear-leaf arms)

| arm | refits | largest abs leaf coefficient (any refit) | median over refits of the largest abs leaf coefficient | share of leaves with a linear part | features in a linear leaf (mean) | in-window fit min | in-window fit max |
|---|---|---|---|---|---|---|---|
| lt_all_r10 | 134 | 24442.9350 | 49.7906 | 0.969 | 5.20 | -2042.0740 | 338.7671 |
| lt_all_r10_lam1 | 134 | 0.0227 | 0.0136 | 0.976 | 5.09 | 0.1732 | 6.7777 |
| lt_all_r10_lam10 | 134 | 0.0097 | 0.0080 | 0.972 | 5.17 | 0.2198 | 6.2747 |
| lt_base_r10 | 134 | 3.0473 | 0.8151 | 0.997 | 3.04 | 0.2584 | 6.8837 |
| lt_base_r1 | 1339 | 10.6545 | 0.8024 | 0.997 | 3.04 | -4.0957 | 7.9093 |

Leaf coefficients are as stored in the booster, i.e. after the learning-rate shrinkage (learning_rate 0.01296); the design columns are robust-scaled.

## Compute

CPU time of the local arms: 1.88 h in all (reused arms count the data-size run's CPU), one process, single-threaded (num_threads 1; OMP / OpenBLAS / MKL threads 1).

## Files

- `gate.csv`: stored tables re-scored against the master table; local controls against the stored runs.
- `arms.csv`: every arm and the stored reference rows: QLIKE, Sharpe mid / crossed, DM and HAC t against the local control, fit seconds, CPU seconds.
- `vs_linear.csv`: every arm against the master table's linear rows of its design.
- `extremes.csv`: forecast range, negative forecasts, the largest recalibrated forecast (`max_pred_clock`).
- `leaves.csv`: leaf-coefficient sizes and in-window fitted range of the linear-leaf arms.
- `_work/<arm>.npz`: forecasts and refit records (not committed).
