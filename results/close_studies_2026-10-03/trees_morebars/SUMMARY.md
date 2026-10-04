# Training the 16:00 trees on more bars than the last two (close study, 2026-10-04)

Question (the user's request): *run ablations on training on more bars than the last 2.* The data-size study trained LightGBM on both last-hour bars (2000 sessions = 4000 rows) and scored the 16:00 forecast only. This study trains on the N half-hour bars ending 16:00, N = 1, 2, 3, 4, 5, 7, 13, and again scores only the 16:00 forecast. Written by `experiments/close_trees_morebars.py analyze` from the CSVs in this folder; every number below is in them.

## Setup
- Design: `experiments/capture_design_close.py`, all features. N = 1 is the shipped one-bar design (`bar1600`), N = 2 is `last30`, N >= 3 is `lastbars<N>`. Each multi-bar design's rolling robust scaling uses its own window (2000 sessions x the median bars a session), so the 16:00 rows of two designs differ in the scaling of some columns; the target on the 16:00 rows is the same (asserted).
- Trees: the tree spec's shipped configs, refit every 10 sessions from forecast row 130 (2019-01), single-threaded, window mask on, one model forecasting the 16:00 row; the leaf minimum scaled to the training rows by the spec's rule (LightGBM `min_child_samples` = round(98 x rows / 24000), 98 at the pooled bank's 24000 rows). The model in force for the 16:00 row of day d was fitted on the design rows strictly before that row: the earlier bars of day d (targets realized by 15:30, when the forecast is issued) may enter it, nothing later does.
- Rungs N = 1 and N = 2 are the data-size study's `lgbm_bar1600_w2000` and `lgbm_pool_w4000` (and its XGBoost / ridge / lasso arms of the same rows), reused after the gates below.
- Scorer: the master table's research convention (16:00 recalibration over the previous 250 sessions, the 15:30 sign(s) straddle trade), 866 trade days 2020-01-03 .. 2024-04-30, on the one-bar design's forecast rows. Point estimates (the block bootstrap is off); DM = Diebold-Mariano on daily QLIKE (negative = the arm has lower loss), HAC t = Newey-West t on the paired daily mid-fill P&L difference (positive = the arm earns more).

## Main ladder: LightGBM, 2000 sessions, N bars

| bars | rows | sessions | leaf min | QLIKE | Sharpe mid | Sharpe crossed | DM vs 1 bar | HAC t vs 1 bar | DM vs 2 bars | HAC t vs 2 bars | fit s (mean) | CPU min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2000 | 2000 | 8 | 0.1007 | 1.99 | 1.52 |  |  | 1.16 | 0.13 | 9.9 | 21.3 |
| 2 | 4000 | 2000 | 16 | 0.0982 | 1.95 | 1.48 | -1.16 | -0.13 |  |  | 10.7 | 22.6 |
| pooled 48-bar LightGBM (master table `lgbm`, cited, not rerun) | 24000 bars | about 500 |  | 0.1006 | 1.48 | 0.99 |  |  |  |  |  |  |

![bars ladder](bars_ladder.png)

## Reading (numbers from the tables in this file)
- LightGBM, 2000 sessions, QLIKE for N = 1 / 2: 0.1007 / 0.0982; Sharpe mid 1.99 / 1.95.
- Lowest QLIKE on the ladder: N = 2 (4000 rows), 0.0982; against 1 bar DM -1.16 (p 0.247), against 2 bars DM  (p ).
- Far end, cited: the paper's pooled 48-bar LightGBM (24000 bars, about 500 sessions, a different design and its own recalibration of the 16:00 bar within the master table) QLIKE 0.1006, Sharpe mid 1.48; 1-bar control 0.1007 / 1.99, 2 bars 0.0982 / 1.95.

## Other arms

| arm | model | bars | rows | sessions | training rows | QLIKE | Sharpe mid | Sharpe crossed | DM vs 1 bar | HAC t vs 1 bar | DM vs 2 bars | HAC t vs 2 bars | fit s (mean) | CPU min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `xgb_bars1_r2000` | XGBoost | 1 | 2000 | 2000 | one-bar design (shipped) | 0.1021 | 1.30 | 0.83 | 0.86 | -1.89 | 1.97 | -1.68 | 6.7 | 14.8 |
| `xgb_bars2_r4000` | XGBoost | 2 | 4000 | 2000 | 2 bars ending 16:00 (last30) | 0.0993 | 1.45 | 0.98 | -0.63 | -1.83 | 0.87 | -1.92 | 8.1 | 17.9 |
| `ridge_bars1_r2000` | ridge | 1 | 2000 | 2000 | one-bar design (shipped) | 0.1004 | 1.66 | 1.20 | -0.07 | -0.79 | 0.52 | -0.68 |  | 0.1 |
| `ridge_bars2_r4000` | ridge | 2 | 4000 | 2000 | 2 bars ending 16:00 (last30) | 0.1048 | 1.29 | 0.82 | 0.88 | -1.46 | 1.50 | -1.55 |  | 0.2 |
| `lasso_bars1_r2000` | lasso | 1 | 2000 | 2000 | one-bar design (shipped) | 0.0998 | 1.45 | 0.98 | -0.28 | -1.46 | 0.61 | -1.48 |  | 1.3 |
| `lasso_bars2_r4000` | lasso | 2 | 4000 | 2000 | 2 bars ending 16:00 (last30) | 0.1049 | 1.21 | 0.74 | 0.96 | -2.09 | 1.66 | -1.93 |  | 3.1 |

## Pairwise differences (`vs_reference.csv`)

| arm | reference | dQLIKE | dQLIKE % | DM | p | dSharpe mid | HAC t |
|---|---|---|---|---|---|---|---|
| `lgbm_bars1_r2000` | `lgbm_bars2_r4000` | +0.0024 | +2.5 | 1.16 | 0.247 | +0.03 | 0.13 |
| `lgbm_bars2_r4000` | `lgbm_bars1_r2000` | -0.0024 | -2.4 | -1.16 | 0.247 | -0.03 | -0.13 |
| `xgb_bars1_r2000` | `lgbm_bars1_r2000` | +0.0014 | +1.4 | 0.86 | 0.389 | -0.69 | -1.89 |
| `xgb_bars1_r2000` | `lgbm_bars2_r4000` | +0.0039 | +3.9 | 1.97 | 0.048 | -0.65 | -1.68 |
| `xgb_bars1_r2000` | `xgb_bars2_r4000` | +0.0028 | +2.9 | 1.93 | 0.054 | -0.15 | -0.52 |
| `xgb_bars2_r4000` | `lgbm_bars1_r2000` | -0.0014 | -1.4 | -0.63 | 0.529 | -0.54 | -1.83 |
| `xgb_bars2_r4000` | `lgbm_bars2_r4000` | +0.0010 | +1.0 | 0.87 | 0.387 | -0.50 | -1.92 |
| `xgb_bars2_r4000` | `xgb_bars1_r2000` | -0.0028 | -2.8 | -1.93 | 0.054 | +0.15 | 0.52 |
| `ridge_bars1_r2000` | `lgbm_bars1_r2000` | -0.0003 | -0.3 | -0.07 | 0.942 | -0.32 | -0.79 |
| `ridge_bars1_r2000` | `lgbm_bars2_r4000` | +0.0021 | +2.2 | 0.52 | 0.601 | -0.29 | -0.68 |
| `ridge_bars1_r2000` | `ridge_bars2_r4000` | -0.0044 | -4.2 | -1.40 | 0.163 | +0.38 | 0.98 |
| `ridge_bars2_r4000` | `lgbm_bars1_r2000` | +0.0041 | +4.1 | 0.88 | 0.379 | -0.70 | -1.46 |
| `ridge_bars2_r4000` | `lgbm_bars2_r4000` | +0.0065 | +6.6 | 1.50 | 0.134 | -0.67 | -1.55 |
| `ridge_bars2_r4000` | `ridge_bars1_r2000` | +0.0044 | +4.4 | 1.40 | 0.163 | -0.38 | -0.98 |
| `lasso_bars1_r2000` | `lgbm_bars1_r2000` | -0.0009 | -0.9 | -0.28 | 0.780 | -0.54 | -1.46 |
| `lasso_bars1_r2000` | `lgbm_bars2_r4000` | +0.0015 | +1.5 | 0.61 | 0.539 | -0.51 | -1.48 |
| `lasso_bars1_r2000` | `lasso_bars2_r4000` | -0.0052 | -4.9 | -1.37 | 0.171 | +0.24 | 0.68 |
| `lasso_bars2_r4000` | `lgbm_bars1_r2000` | +0.0043 | +4.2 | 0.96 | 0.339 | -0.78 | -2.09 |
| `lasso_bars2_r4000` | `lgbm_bars2_r4000` | +0.0067 | +6.8 | 1.66 | 0.097 | -0.75 | -1.93 |
| `lasso_bars2_r4000` | `lasso_bars1_r2000` | +0.0052 | +5.2 | 1.37 | 0.171 | -0.24 | -0.68 |

## Gates
- lgbm_bars1_r2000 = data-size lgbm_bar1600_w2000: QLIKE re-scored here = its arms.csv: 0.100676 vs 0.100676 (|diff| 4.16e-17) PASS
- lgbm_bars1_r2000 = data-size lgbm_bar1600_w2000: Sharpe mid re-scored here = its arms.csv: 1.98862 vs 1.98862 (|diff| 0.00e+00) PASS
- lgbm_bars1_r2000 = data-size lgbm_bar1600_w2000: configuration, bucket (all_features): 1 vs 1 (|diff| 0.00e+00) PASS
- lgbm_bars1_r2000 = data-size lgbm_bar1600_w2000: configuration, model (lgbm): 1 vs 1 (|diff| 0.00e+00) PASS
- lgbm_bars1_r2000 = data-size lgbm_bar1600_w2000: configuration, window rows (2000): 1 vs 1 (|diff| 0.00e+00) PASS
- lgbm_bars1_r2000 = data-size lgbm_bar1600_w2000: configuration, rows (source) (bar1600): 1 vs 1 (|diff| 0.00e+00) PASS
- lgbm_bars1_r2000 = data-size lgbm_bar1600_w2000: configuration, whole arm (no chunk / slice) (chunk 'absent', slice absent): 1 vs 1 (|diff| 0.00e+00) PASS
- lgbm_bars1_r2000 = data-size lgbm_bar1600_w2000: configuration, params = spec rule at these rows (leaf minimum 8): 1 vs 1 (|diff| 0.00e+00) PASS
- lgbm_bars1_r2000 = data-size lgbm_bar1600_w2000: configuration, refit every 10, first forecast row 130 (10 130): 1 vs 1 (|diff| 0.00e+00) PASS
- lgbm_bars1_r2000 = data-size lgbm_bar1600_w2000: configuration, refit anchors (134): 1 vs 1 (|diff| 0.00e+00) PASS
- lgbm_bars1_r2000 = data-size lgbm_bar1600_w2000: configuration, training rows at every refit ([2000]): 1 vs 1 (|diff| 0.00e+00) PASS
- lgbm_bars2_r4000 = data-size lgbm_pool_w4000: QLIKE re-scored here = its arms.csv: 0.0982494 vs 0.0982494 (|diff| 5.55e-17) PASS
- lgbm_bars2_r4000 = data-size lgbm_pool_w4000: Sharpe mid re-scored here = its arms.csv: 1.95432 vs 1.95432 (|diff| 2.22e-16) PASS
- lgbm_bars2_r4000 = data-size lgbm_pool_w4000: configuration, bucket (all_features): 1 vs 1 (|diff| 0.00e+00) PASS
- lgbm_bars2_r4000 = data-size lgbm_pool_w4000: configuration, model (lgbm): 1 vs 1 (|diff| 0.00e+00) PASS
- lgbm_bars2_r4000 = data-size lgbm_pool_w4000: configuration, window rows (4000): 1 vs 1 (|diff| 0.00e+00) PASS
- lgbm_bars2_r4000 = data-size lgbm_pool_w4000: configuration, rows (source) (pool): 1 vs 1 (|diff| 0.00e+00) PASS
- lgbm_bars2_r4000 = data-size lgbm_pool_w4000: configuration, whole arm (no chunk / slice) (chunk 'absent', slice absent): 1 vs 1 (|diff| 0.00e+00) PASS
- lgbm_bars2_r4000 = data-size lgbm_pool_w4000: configuration, params = spec rule at these rows (leaf minimum 16): 1 vs 1 (|diff| 0.00e+00) PASS
- lgbm_bars2_r4000 = data-size lgbm_pool_w4000: configuration, refit every 10, first forecast row 130 (10 130): 1 vs 1 (|diff| 0.00e+00) PASS
- lgbm_bars2_r4000 = data-size lgbm_pool_w4000: configuration, refit anchors (134): 1 vs 1 (|diff| 0.00e+00) PASS
- lgbm_bars2_r4000 = data-size lgbm_pool_w4000: configuration, training rows at every refit ([4000]): 1 vs 1 (|diff| 0.00e+00) PASS
- xgb_bars1_r2000 = data-size xgb_bar1600_w2000: QLIKE re-scored here = its arms.csv: 0.102116 vs 0.102116 (|diff| 5.55e-17) PASS
- xgb_bars1_r2000 = data-size xgb_bar1600_w2000: Sharpe mid re-scored here = its arms.csv: 1.3026 vs 1.3026 (|diff| 0.00e+00) PASS
- xgb_bars1_r2000 = data-size xgb_bar1600_w2000: configuration, bucket (all_features): 1 vs 1 (|diff| 0.00e+00) PASS
- xgb_bars1_r2000 = data-size xgb_bar1600_w2000: configuration, model (xgb): 1 vs 1 (|diff| 0.00e+00) PASS
- xgb_bars1_r2000 = data-size xgb_bar1600_w2000: configuration, window rows (2000): 1 vs 1 (|diff| 0.00e+00) PASS
- xgb_bars1_r2000 = data-size xgb_bar1600_w2000: configuration, rows (source) (bar1600): 1 vs 1 (|diff| 0.00e+00) PASS
- xgb_bars1_r2000 = data-size xgb_bar1600_w2000: configuration, whole arm (no chunk / slice) (chunk '', slice 0): 1 vs 1 (|diff| 0.00e+00) PASS
- xgb_bars1_r2000 = data-size xgb_bar1600_w2000: configuration, params = spec rule at these rows (leaf minimum 2.072): 1 vs 1 (|diff| 0.00e+00) PASS
- xgb_bars1_r2000 = data-size xgb_bar1600_w2000: configuration, refit every 10, first forecast row 130 (10 130): 1 vs 1 (|diff| 0.00e+00) PASS
- xgb_bars1_r2000 = data-size xgb_bar1600_w2000: configuration, refit anchors (134): 1 vs 1 (|diff| 0.00e+00) PASS
- xgb_bars1_r2000 = data-size xgb_bar1600_w2000: configuration, training rows at every refit ([2000]): 1 vs 1 (|diff| 0.00e+00) PASS
- xgb_bars2_r4000 = data-size xgb_pool_w4000: QLIKE re-scored here = its arms.csv: 0.0992749 vs 0.0992749 (|diff| 2.78e-17) PASS
- xgb_bars2_r4000 = data-size xgb_pool_w4000: Sharpe mid re-scored here = its arms.csv: 1.45218 vs 1.45218 (|diff| 0.00e+00) PASS
- xgb_bars2_r4000 = data-size xgb_pool_w4000: configuration, bucket (all_features): 1 vs 1 (|diff| 0.00e+00) PASS
- xgb_bars2_r4000 = data-size xgb_pool_w4000: configuration, model (xgb): 1 vs 1 (|diff| 0.00e+00) PASS
- xgb_bars2_r4000 = data-size xgb_pool_w4000: configuration, window rows (4000): 1 vs 1 (|diff| 0.00e+00) PASS
- xgb_bars2_r4000 = data-size xgb_pool_w4000: configuration, rows (source) (pool): 1 vs 1 (|diff| 0.00e+00) PASS
- xgb_bars2_r4000 = data-size xgb_pool_w4000: configuration, whole arm (no chunk / slice) (chunk '', slice 0): 1 vs 1 (|diff| 0.00e+00) PASS
- xgb_bars2_r4000 = data-size xgb_pool_w4000: configuration, params = spec rule at these rows (leaf minimum 4.144): 1 vs 1 (|diff| 0.00e+00) PASS
- xgb_bars2_r4000 = data-size xgb_pool_w4000: configuration, refit every 10, first forecast row 130 (10 130): 1 vs 1 (|diff| 0.00e+00) PASS
- xgb_bars2_r4000 = data-size xgb_pool_w4000: configuration, refit anchors (134): 1 vs 1 (|diff| 0.00e+00) PASS
- xgb_bars2_r4000 = data-size xgb_pool_w4000: configuration, training rows at every refit ([4000]): 1 vs 1 (|diff| 0.00e+00) PASS
- ridge_bars1_r2000 = data-size ridge_bar1600_w2000: QLIKE re-scored here = its arms.csv: 0.100387 vs 0.100387 (|diff| 2.78e-17) PASS
- ridge_bars1_r2000 = data-size ridge_bar1600_w2000: Sharpe mid re-scored here = its arms.csv: 1.66493 vs 1.66493 (|diff| 0.00e+00) PASS
- ridge_bars1_r2000 = data-size ridge_bar1600_w2000: configuration, bucket (all_features): 1 vs 1 (|diff| 0.00e+00) PASS
- ridge_bars1_r2000 = data-size ridge_bar1600_w2000: configuration, model (ridge): 1 vs 1 (|diff| 0.00e+00) PASS
- ridge_bars1_r2000 = data-size ridge_bar1600_w2000: configuration, window rows (2000): 1 vs 1 (|diff| 0.00e+00) PASS
- ridge_bars1_r2000 = data-size ridge_bar1600_w2000: configuration, rows (source) (bar1600): 1 vs 1 (|diff| 0.00e+00) PASS
- ridge_bars1_r2000 = data-size ridge_bar1600_w2000: configuration, whole arm (no chunk / slice) (chunk 'absent', slice absent): 1 vs 1 (|diff| 0.00e+00) PASS
- ridge_bars1_r2000 = data-size ridge_bar1600_w2000: configuration, estimator (ridge): 1 vs 1 (|diff| 0.00e+00) PASS
- ridge_bars2_r4000 = data-size ridge_pool_w4000: QLIKE re-scored here = its arms.csv: 0.104768 vs 0.104768 (|diff| 1.39e-17) PASS
- ridge_bars2_r4000 = data-size ridge_pool_w4000: Sharpe mid re-scored here = its arms.csv: 1.28576 vs 1.28576 (|diff| 0.00e+00) PASS
- ridge_bars2_r4000 = data-size ridge_pool_w4000: configuration, bucket (all_features): 1 vs 1 (|diff| 0.00e+00) PASS
- ridge_bars2_r4000 = data-size ridge_pool_w4000: configuration, model (ridge): 1 vs 1 (|diff| 0.00e+00) PASS
- ridge_bars2_r4000 = data-size ridge_pool_w4000: configuration, window rows (4000): 1 vs 1 (|diff| 0.00e+00) PASS
- ridge_bars2_r4000 = data-size ridge_pool_w4000: configuration, rows (source) (pool): 1 vs 1 (|diff| 0.00e+00) PASS
- ridge_bars2_r4000 = data-size ridge_pool_w4000: configuration, whole arm (no chunk / slice) (chunk 'absent', slice absent): 1 vs 1 (|diff| 0.00e+00) PASS
- ridge_bars2_r4000 = data-size ridge_pool_w4000: configuration, estimator (ridge): 1 vs 1 (|diff| 0.00e+00) PASS
- lasso_bars1_r2000 = data-size lasso_bar1600_w2000: QLIKE re-scored here = its arms.csv: 0.0997564 vs 0.0997564 (|diff| 2.78e-17) PASS
- lasso_bars1_r2000 = data-size lasso_bar1600_w2000: Sharpe mid re-scored here = its arms.csv: 1.44642 vs 1.44642 (|diff| 0.00e+00) PASS
- lasso_bars1_r2000 = data-size lasso_bar1600_w2000: configuration, bucket (all_features): 1 vs 1 (|diff| 0.00e+00) PASS
- lasso_bars1_r2000 = data-size lasso_bar1600_w2000: configuration, model (lasso): 1 vs 1 (|diff| 0.00e+00) PASS
- lasso_bars1_r2000 = data-size lasso_bar1600_w2000: configuration, window rows (2000): 1 vs 1 (|diff| 0.00e+00) PASS
- lasso_bars1_r2000 = data-size lasso_bar1600_w2000: configuration, rows (source) (bar1600): 1 vs 1 (|diff| 0.00e+00) PASS
- lasso_bars1_r2000 = data-size lasso_bar1600_w2000: configuration, whole arm (no chunk / slice) (chunk 'absent', slice absent): 1 vs 1 (|diff| 0.00e+00) PASS
- lasso_bars1_r2000 = data-size lasso_bar1600_w2000: configuration, estimator (reclasso): 1 vs 1 (|diff| 0.00e+00) PASS
- lasso_bars2_r4000 = data-size lasso_pool_w4000: QLIKE re-scored here = its arms.csv: 0.104947 vs 0.104947 (|diff| 1.39e-17) PASS
- lasso_bars2_r4000 = data-size lasso_pool_w4000: Sharpe mid re-scored here = its arms.csv: 1.20599 vs 1.20599 (|diff| 0.00e+00) PASS
- lasso_bars2_r4000 = data-size lasso_pool_w4000: configuration, bucket (all_features): 1 vs 1 (|diff| 0.00e+00) PASS
- lasso_bars2_r4000 = data-size lasso_pool_w4000: configuration, model (lasso): 1 vs 1 (|diff| 0.00e+00) PASS
- lasso_bars2_r4000 = data-size lasso_pool_w4000: configuration, window rows (4000): 1 vs 1 (|diff| 0.00e+00) PASS
- lasso_bars2_r4000 = data-size lasso_pool_w4000: configuration, rows (source) (pool): 1 vs 1 (|diff| 0.00e+00) PASS
- lasso_bars2_r4000 = data-size lasso_pool_w4000: configuration, whole arm (no chunk / slice) (chunk 'absent', slice absent): 1 vs 1 (|diff| 0.00e+00) PASS
- lasso_bars2_r4000 = data-size lasso_pool_w4000: configuration, estimator (reclasso): 1 vs 1 (|diff| 0.00e+00) PASS
- lgbm_bars1_r2000: rows built here = data-size rows (X, y, forecast rows): 1 vs 1 (|diff| 0.00e+00) PASS
- lgbm_bars1_r2000: first 3 refits redone here vs stored lgbm_bar1600_w2000 (30 forecasts), max |difference|: 0 vs 0 (|diff| 0.00e+00) PASS
- lgbm_bars2_r4000: rows built here = data-size rows (X, y, forecast rows): 1 vs 1 (|diff| 0.00e+00) PASS
- lgbm_bars2_r4000: first 3 refits redone here vs stored lgbm_pool_w4000 (30 forecasts), max |difference|: 0 vs 0 (|diff| 0.00e+00) PASS

## Caveats
- The leaf minimum changes with the rows (the spec's rule), so each rung changes the rows, the bars and the leaf minimum together; the rows-held-at-4000 arms hold the rows and the leaf minimum (16) fixed.
- Each design has its own rolling scaling, so two rungs differ also in the scaling of some columns; in the data-size study the same LightGBM on two scalings of the same 16:00 rows gave QLIKE 0.1007 vs 0.1006 and Sharpe mid 1.99 vs 1.56.
- Linear arms solve at every row (the earlier bars included, not scored) and re-choose the penalty every 250 solves, i.e. every 250 / N sessions; tree arms refit every 10 sessions.
- Refit every 10 sessions, one seed, point estimates only; Sharpe ratios of forecasts with nearly equal QLIKE move by several tenths (data-size and tuning studies).

## Not run
- Arms defined in the script and not run: `lgbm_bars3_r6000`, `lgbm_bars4_r8000`, `lgbm_bars5_r10000`, `lgbm_bars7_r14000`, `lgbm_bars13_r26000`, `xgb_bars3_r6000`, `xgb_bars4_r8000`, `xgb_bars5_r10000`, `xgb_bars7_r14000`, `xgb_bars13_r26000`, `ridge_bars3_r6000`, `ridge_bars4_r8000`, `ridge_bars5_r10000`, `ridge_bars7_r14000`, `ridge_bars13_r26000`, `lasso_bars3_r6000`, `lasso_bars4_r8000`, `lasso_bars5_r10000`, `lasso_bars7_r14000`, `lasso_bars13_r26000`, `lgbm_bars4_r4000`, `lgbm_bars13_r4000`, `lgbm_bars1_r4000`, `lgbm_bars13_r26000_barmin`.

## Files
- `experiments/close_trees_morebars.py` (stages gate / run / analyze; fitting machinery from `experiments/close_trees_datasize.py`)
- `arms.csv`, `vs_reference.csv`, `gates.csv`, `cited.csv`, `bars_ladder.png`; forecasts in `_work/<arm>.npz` (not committed; rungs 1 and 2 read from `../trees_datasize/_work/`)
- CPU: 0 min over the 0 arms fitted here (every fit single-threaded, at most four processes at once); the reused arms took 81 min in the data-size study.
