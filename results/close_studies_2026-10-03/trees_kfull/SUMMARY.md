# LightGBM on the last k bars ending 16:00 with the bar-end time as a column, k = 1 .. 13, three seeds (close study, 2026-10-04)

The experiment: a linear model trained on the 16:00 bar against LightGBM trained on the pool of the last k half-hour bars ending 16:00 (k = 1 .. 13), every tree with k > 1 carrying the column `bar_end_minute` (bar-end time, minutes since midnight), all scored only on the 16:00 forecast. This folder holds the forecasts' manifest for the statistical tests, the score of each arm and the trees' use of the bar column. Written by `experiments/close_trees_kfull.py manifest` from the CSVs in this folder; every number below is in them.

## Setup
- Rows: the design `lastbars<k>` of `experiments/capture_design_close.py` (all features; k = 1 is `bar1600`, k = 2 is `last30`), the window 2000 sessions x k bars; the column `bar_end_minute` is added exactly as `close_trees_morebars.source` adds it for its k = 13 arm. k = 1 has no such column (every row ends 16:00).
- Trees: `close_trees_morebars.run` unchanged -- the tree spec's shipped LightGBM config, leaf minimum `min_child_samples` = round(98 x rows / 24000), refit every 10 sessions from forecast row 130 (2019-01, 134 refits), window mask on, single-threaded, LightGBM 4.7.0; random_state 42 (the spec's), 43, 44, nothing else changed between seeds. The model in force for the 16:00 row of day d was fitted on the design rows strictly before that row (the earlier bars of day d, realized by 15:30, may enter it).
- Bar-column use: at every refit the booster's split count and gain of each column it saw (`feature_importance`), recorded for `bar_end_minute` and the calendar column `hour` (which also separates the 16:00 row from the earlier bars, since the bars ending 15:00 and 15:30 share hour 15). Share = the column's splits (gain) over all splits (gain) of that fit; rank 1 = the column with the most gain (most splits) among the columns the fit saw.
- Scorer: the master table's research convention (16:00 recalibration over the previous 250 sessions, the 15:30 sign(s) straddle trade), 866 trade days 2020-01-03 .. 2024-04-30, forecast rows from row 130; point estimates (no bootstrap, no test here: the tests are built separately on these forecasts).
- Forecast files: `pred` holds 1469 values on the one-bar design's forecast rows (2018-06-25 .. 2024-04-30, the dates of `design_bar1600_all_features` from row W = 2000; `close_trees_datasize.sources()['dz']`), NaN before row 130 for the trees. New tree files also hold `imp_split`, `imp_gain`, `imp_kept` (refits x columns) and `anchors`.

## Gates
- stored lgbm_bars13_r26000_barmin.npz: configuration bars = this script's lgbm_bars13_r26000_barmin_seed42: 1 vs 1 (13 vs 13) PASS
- stored lgbm_bars13_r26000_barmin.npz: configuration rows = this script's lgbm_bars13_r26000_barmin_seed42: 1 vs 1 (26000 vs 26000) PASS
- stored lgbm_bars13_r26000_barmin.npz: configuration design = this script's lgbm_bars13_r26000_barmin_seed42: 1 vs 1 ('lastbars13' vs 'lastbars13') PASS
- stored lgbm_bars13_r26000_barmin.npz: configuration rows_from = this script's lgbm_bars13_r26000_barmin_seed42: 1 vs 1 ('all' vs 'all') PASS
- stored lgbm_bars13_r26000_barmin.npz: configuration cols = this script's lgbm_bars13_r26000_barmin_seed42: 1 vs 1 ('bar_end_minute' vs 'bar_end_minute') PASS
- stored lgbm_bars13_r26000_barmin.npz: params = the spec rule at 26000 rows: 1 vs 1 (leaf minimum 106) PASS
- stored lgbm_bars13_r26000_barmin.npz: random_state = the spec's 42 (no seed field = the spec's): 1 vs 1 (absent) PASS
- lgbm_bars13_r26000_barmin_seed42: first 3 refits redone here vs stored lgbm_bars13_r26000_barmin.npz (30 forecasts), max |difference|: 0 vs 0 (bitwise True, 37s) PASS
- lgbm_bars13_r26000_barmin_seed42: importances recorded at each of the 3 refits, bar column kept by the window mask: 1 vs 1 (bar_end_minute splits 39 / 33 / 33, gain share 0.0010 / 0.0010 / 0.0009) PASS
- lgbm_bars4_r8000 re-scored here: QLIKE = the stored 0.0974 (rounded) and = trees_morebars/arms.csv: 0.0974344 vs 0.0974344 (0.097434, 866 trade days) PASS
- lgbm_bars4_r8000 re-scored here: Sharpe mid = the stored 1.63 (rounded) and = trees_morebars/arms.csv: 1.6321 vs 1.6321 (1.6321) PASS
- Configuration of every LightGBM file present (design, rows, bar column, params at these rows, random_state, refit anchors, training rows, LightGBM version, forecast rows): 13 pass, none fail.

## QLIKE and Sharpe for each k: seed mean +- sd

Main spec = k = 1 without a column, k >= 2 with `bar_end_minute`. sd over the seeds present (ddof 1).

| k | rows | main: seeds | main: QLIKE mean +- sd | main: QLIKE 42 / 43 / 44 | main: Sharpe mid mean +- sd | no column: seeds | no column: QLIKE mean +- sd | no column: Sharpe mid mean +- sd |
|---|---|---|---|---|---|---|---|---|
| 1 | 2000 | 42 43 | 0.1003 +- 0.0005 | 0.1007 / 0.1000 / - | 2.02 +- 0.04 | 42 43 | 0.1003 +- 0.0005 | 2.02 +- 0.04 |
| 2 | 4000 |  |  |  |  | 42 43 | 0.0985 +- 0.0003 | 1.72 +- 0.33 |
| 3 | 6000 |  |  |  |  | 42 43 | 0.0978 +- 0.0004 | 1.14 +- 0.16 |
| 4 | 8000 |  |  |  |  | 42 43 | 0.0978 +- 0.0005 | 1.70 +- 0.09 |
| 5 | 10000 |  |  |  |  | 42 | 0.0984 | 1.70 |
| 6 | 12000 |  |  |  |  |  |  |  |
| 7 | 14000 |  |  |  |  | 42 | 0.0996 | 1.70 |
| 8 | 16000 |  |  |  |  |  |  |  |
| 9 | 18000 |  |  |  |  |  |  |  |
| 10 | 20000 |  |  |  |  |  |  |  |
| 11 | 22000 |  |  |  |  |  |  |  |
| 12 | 24000 |  |  |  |  |  |  |  |
| 13 | 26000 | 42 | 0.1015 | 0.1015 / - / - | 1.57 | 42 43 | 0.1016 +- 0.0003 | 1.63 +- 0.07 |

Linear models trained on the 16:00 bar (2000 sessions, same scorer and rows): `ridge_bars1_r2000` (ridge) QLIKE 0.1004, Sharpe mid 1.66; `lasso_bars1_r2000` (lasso) QLIKE 0.0998, Sharpe mid 1.45; `ridge_bb0` (ridge, HAR + calendar backbone unpenalized) QLIKE 0.0946, Sharpe mid 1.50.

## Each arm

| arm | family | k | bar column | seed | status | QLIKE | Sharpe mid | Sharpe crossed | buys % | CPU min |
|---|---|---|---|---|---|---|---|---|---|---|
| `lgbm_bars1_r2000` | main | 1 | no | 42 | stored (trees_datasize lgbm_bar1600_w2000) | 0.1007 | 1.99 | 1.52 | 38.7 | 21.3 |
| `lgbm_bars1_r2000_seed43` | main | 1 | no | 43 | stored (trees_morebars) | 0.1000 | 2.05 | 1.58 | 38.7 | 13.6 |
| `lgbm_bars13_r26000_barmin` | main | 13 | yes | 42 | stored (trees_morebars); refitted here as lgbm_bars13_r26000_barmin_seed42 | 0.1015 | 1.57 | 1.09 | 31.9 | 29.8 |
| `lgbm_bars2_r4000` | no_bar_column | 2 | no | 42 | stored (trees_datasize lgbm_pool_w4000) | 0.0982 | 1.95 | 1.48 | 37.0 | 22.6 |
| `lgbm_bars3_r6000` | no_bar_column | 3 | no | 42 | stored (trees_morebars) | 0.0975 | 1.03 | 0.55 | 33.9 | 16.1 |
| `lgbm_bars4_r8000` | no_bar_column | 4 | no | 42 | stored (trees_morebars) | 0.0974 | 1.63 | 1.16 | 33.6 | 18.1 |
| `lgbm_bars5_r10000` | no_bar_column | 5 | no | 42 | stored (trees_morebars) | 0.0984 | 1.70 | 1.23 | 32.1 | 19.2 |
| `lgbm_bars7_r14000` | no_bar_column | 7 | no | 42 | stored (trees_morebars) | 0.0996 | 1.70 | 1.23 | 31.3 | 22.0 |
| `lgbm_bars13_r26000` | no_bar_column | 13 | no | 42 | stored (trees_morebars) | 0.1019 | 1.68 | 1.20 | 32.6 | 30.8 |
| `lgbm_bars2_r4000_seed43` | no_bar_column | 2 | no | 43 | stored (trees_morebars) | 0.0987 | 1.49 | 1.02 | 36.6 | 14.5 |
| `lgbm_bars3_r6000_seed43` | no_bar_column | 3 | no | 43 | stored (trees_morebars) | 0.0981 | 1.26 | 0.78 | 34.2 | 16.0 |
| `lgbm_bars4_r8000_seed43` | no_bar_column | 4 | no | 43 | stored (trees_morebars) | 0.0981 | 1.76 | 1.29 | 33.1 | 17.6 |
| `lgbm_bars13_r26000_seed43` | no_bar_column | 13 | no | 43 | stored (trees_morebars) | 0.1014 | 1.58 | 1.10 | 32.1 | 30.1 |
| `ridge_bars1_r2000` | linear_1600 | 1 | no |  | stored (trees_datasize ridge_bar1600_w2000) | 0.1004 | 1.66 | 1.20 | 36.1 | 0.1 |
| `lasso_bars1_r2000` | linear_1600 | 1 | no |  | stored (trees_datasize lasso_bar1600_w2000) | 0.0998 | 1.45 | 0.98 | 39.6 | 1.3 |
| `ridge_bb0` | linear_1600 | 1 | no |  | stored (exog_penalty ridge_bb0) | 0.0946 | 1.50 | 1.03 | 35.8 | 0.1 |

## Not run
- Arms of the plan without a forecast file: `lgbm_bars1_r2000_seed44`, `lgbm_bars2_r4000_barmin_seed42`, `lgbm_bars2_r4000_barmin_seed43`, `lgbm_bars2_r4000_barmin_seed44`, `lgbm_bars3_r6000_barmin_seed42`, `lgbm_bars3_r6000_barmin_seed43`, `lgbm_bars3_r6000_barmin_seed44`, `lgbm_bars4_r8000_barmin_seed42`, `lgbm_bars4_r8000_barmin_seed43`, `lgbm_bars4_r8000_barmin_seed44`, `lgbm_bars5_r10000_barmin_seed42`, `lgbm_bars5_r10000_barmin_seed43`, `lgbm_bars5_r10000_barmin_seed44`, `lgbm_bars6_r12000_barmin_seed42`, `lgbm_bars6_r12000_barmin_seed43`, `lgbm_bars6_r12000_barmin_seed44`, `lgbm_bars7_r14000_barmin_seed42`, `lgbm_bars7_r14000_barmin_seed43`, `lgbm_bars7_r14000_barmin_seed44`, `lgbm_bars8_r16000_barmin_seed42`, `lgbm_bars8_r16000_barmin_seed43`, `lgbm_bars8_r16000_barmin_seed44`, `lgbm_bars9_r18000_barmin_seed42`, `lgbm_bars9_r18000_barmin_seed43`, `lgbm_bars9_r18000_barmin_seed44`, `lgbm_bars10_r20000_barmin_seed42`, `lgbm_bars10_r20000_barmin_seed43`, `lgbm_bars10_r20000_barmin_seed44`, `lgbm_bars11_r22000_barmin_seed42`, `lgbm_bars11_r22000_barmin_seed43`, `lgbm_bars11_r22000_barmin_seed44`, `lgbm_bars12_r24000_barmin_seed42`, `lgbm_bars12_r24000_barmin_seed43`, `lgbm_bars12_r24000_barmin_seed44`, `lgbm_bars13_r26000_barmin_seed42`, `lgbm_bars13_r26000_barmin_seed43`, `lgbm_bars13_r26000_barmin_seed44`, `lgbm_bars6_r12000_seed42`, `lgbm_bars6_r12000_seed43`, `lgbm_bars6_r12000_seed44`, `lgbm_bars8_r16000_seed42`, `lgbm_bars8_r16000_seed43`, `lgbm_bars8_r16000_seed44`, `lgbm_bars9_r18000_seed42`, `lgbm_bars9_r18000_seed43`, `lgbm_bars9_r18000_seed44`, `lgbm_bars10_r20000_seed42`, `lgbm_bars10_r20000_seed43`, `lgbm_bars10_r20000_seed44`, `lgbm_bars11_r22000_seed42`, `lgbm_bars11_r22000_seed43`, `lgbm_bars11_r22000_seed44`, `lgbm_bars12_r24000_seed42`, `lgbm_bars12_r24000_seed43`, `lgbm_bars12_r24000_seed44`.

## Caveats
- The leaf minimum, the rows and the bars change together along k (the spec's rule); each design has its own rolling scaling (close_trees_morebars SUMMARY).
- Three seeds give a rough sd only; the seed changes the bagging and feature-fraction draws and nothing else.
- `hour` and `bar_end_minute` carry overlapping timing information for k >= 2, so a low share for one of them does not by itself mean the trees ignore timing.
- CPU minutes depend on what else ran on the machine (four single-threaded fits at once here).

## Files
- `experiments/close_trees_kfull.py` (stages gate / run / manifest; machinery from `experiments/close_trees_morebars.py` and `experiments/close_trees_datasize.py`)
- `manifest.csv` (every forecast file: family, k, rows, bar column, seed, primary, path, configuration, check), `arms.csv` (manifest + scores), `seeds.csv`, `bar_column_refits.csv`, `bar_column_arms.csv`, `bar_column_by_k.csv`, `gate.csv`; new forecasts in `_work/<arm>.npz` (not committed)
- CPU: 0 min over the 0 arms fitted here (single-threaded, at most four at once).
