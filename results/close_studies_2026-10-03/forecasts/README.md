# Close studies 2026-10-03 / 10-04: forecast export

Written by `experiments/close_studies_export.py verify` on 2026-10-06 20:31. A compact, self-describing copy of every forecast series of the close studies (`results/close_studies_2026-10-03/*/`), the 16:00 targets and trade-day columns, a catalogue, and the model internals the npz files held. The raw `.npz` outputs in `*/_work/` are gitignored and never committed; this folder replaces them for analysis. Nothing here needs `_work/` or the design cache.

## How to load

```python
import sys; sys.path.insert(0, 'experiments')
import close_studies_load as L
cat = L.catalogue()                                    # one row for each series
F = L.forecasts(study='trees_kfull')                   # 1469 forecast rows x series (adjusted scale)
L.score(F)                                             # QLIKE, Sharpe mid / crossed on the 866 trade days
tab, P = L.score(['exog_penalty/ridge_bb0', 'trees_morebars/lgbm_bars4_r8000'], daily=True)  # P: daily QLIKE / P&L
L.score(L.rebuild('trees_morebars/average.csv', 'average of lgbm pools N = 1 .. 4'))
L.ridge_trees(L.rebuild('kfull_tests/forecasts.csv', 'bar:avg'), how='adjusted')  # equal weights, then score()
L.coefficients('ridge_bb0'); L.rechoices('ridge_bb0'); L.importance('lgbm_bars13_r26000_barmin_seed42', 'gain')
L.targets(); L.refits('trees_pretune/retune_any_r10'); L.pretune_trials()
```

Without the loader: `pd.read_parquet('forecasts.parquet').pivot(index='date', columns='series_id', values='pred_adj')`; the targets and trade-day columns are in `targets.parquet`, so QLIKE and P&L statistics need no other file. The scorer: pred_clock = (pred_adj^2 + s) x B (s from the forecast's own trailing 250-row squared errors, lagged one row), QLIKE = RV / pred_clock - log(RV / pred_clock) - 1, position = +1 (buy the straddle) when pred_clock > iv_var else -1, mid P&L = position x R, Sharpe = mean / sd x sqrt(252). `score()` starts the recalibration at row 130 by default (the first tree forecast); on the trade days (from row 382) any start at or before row 132 gives the same numbers. `pred_clock` in forecasts.parquet starts at each series' own first forecast row (row 0 for the linear arms), so it is defined earlier for them; on the trade days it equals what `score()` uses.

## Files

| file | size (MB) | content |
|---|---|---|
| `forecasts.parquet` | 2.50 | long: one row for each (series, forecast row); 148 series x 1469 rows |
| `linear_coefficients.parquet` | 11.01 | exog_penalty arms: coefficients at every forecast row (float32, wide) |
| `linear_rechoices.parquet` | 0.02 | every penalty re-choice of the linear arms: alpha, r, group penalties, validation grid, locked / masked sets |
| `pretune_trials.parquet` | 0.02 | trees_pretune pre-tune trials, every fold |
| `refits.parquet` | 0.21 | fit-level records: tree refits and linear solves |
| `targets.parquet` | 0.09 | one row for each 16:00 forecast row: target, B, trade-day straddle columns |
| `tree_importance.parquet` | 9.19 | trees_kfull arms: split count and gain of every kept column at every refit (long) |
| `VERIFY.csv` | 0.27 | the verify stage's checks |
| `catalogue.csv` | 0.29 | one row for each series (148 exported) and each npz file left out (12) |
| `linear_columns.csv` | 0.03 | names of the coefficient columns (628 design columns + intercept), block and group |
| `study_arms.csv` | 0.04 | each row of the study CSVs that verify checks -> the series or the recipe (seed / pool averages, ridge + trees) that rebuilds it |
| `tree_columns.csv` | 0.02 | names of the tree importance columns |

Total 23.7 MB (every file here, this README included); largest file `linear_coefficients.parquet` 11.0 MB (every file is below GitHub's 50 MB warning size, so no Git LFS). Storage: parquet, zstd level 12, floating columns byte-stream split (lossless); forecasts float64; internals float32. No table needed splitting or float16: the dense ridge coefficients compress to about 1.8 MB an arm with byte-stream split, and the tree importance is stored long, one row for each (refit, column) the window mask kept (a dropped column has neither splits nor gain, so nothing is lost).

## Series

| study | kind | series |
|---|---|---|
| exog_penalty | exact batch solution (every 25th row) | 12 |
| exog_penalty | forecast | 20 |
| master_table | stored master-table forecast | 14 |
| trees_datasize | forecast | 23 |
| trees_kfull | forecast | 43 |
| trees_lineartree | forecast | 8 |
| trees_morebars | forecast | 23 |
| trees_pretune | forecast | 5 |

148 series exported: every forecast npz of the seven studies (trees, linear arms, seed replicates, pools, linear-leaf arms, pretune walk arms, the data-size learning-curve arms, the exog_penalty arms and the HAR + calendar OLS), the exog_penalty `exact_pred` check series where present (lasso / elastic net arms; ridge arms have none), and the 14 stored master-table forecasts the studies compared against (`results/spxw_pnl/yhat_<key>.parquet`, 16:00 rows). A reused forecast is one series, stored once under the study that wrote the file (e.g. trees_morebars' 1- and 2-bar rungs are `trees_datasize/lgbm_bar1600_w2000` and `trees_datasize/lgbm_pool_w4000`); `study_arms.csv` maps every study's own arm names to the series, and `identical_to` in the catalogue lists series that are bit for bit equal (e.g. `trees_kfull/lgbm_bars13_r26000_barmin_seed42` refitted `trees_morebars/lgbm_bars13_r26000_barmin`).

Left out (listed in the catalogue with `exported = no`):

- `results/close_studies_2026-10-03/exog_penalty/_work/check_c_single_reclasso.npz`: check run (the C port's one-penalty arm, gate 2 of exog_penalty/SUMMARY.md): bit for bit equal to exog_penalty/lasso_single
- `results/close_studies_2026-10-03/exog_penalty/_work/check_c_single_reclasticnet.npz`: check run (the C port's one-penalty arm, gate 2 of exog_penalty/SUMMARY.md): bit for bit equal to exog_penalty/enet_single
- `results/close_studies_2026-10-03/exog_penalty/_work/check_c_single_ridge.npz`: check run (the C port's one-penalty arm, gate 2 of exog_penalty/SUMMARY.md): bit for bit equal to exog_penalty/ridge_single
- `results/close_studies_2026-10-03/exog_penalty/_work/gate_spec_reclasso.npz`: gate run (the spec's own Python class on the cached design, gate 1 of exog_penalty/SUMMARY.md): max relative difference 6.8e-13 to master_table/sub_lasso_all_features and 1.0e-12 to exog_penalty/lasso_single
- `results/close_studies_2026-10-03/exog_penalty/_work/gate_spec_reclasticnet.npz`: gate run (the spec's own Python class on the cached design, gate 1 of exog_penalty/SUMMARY.md): max relative difference 6.5e-03 to master_table/sub_enet_all_features and 2.7e-03 to exog_penalty/enet_single
- `results/close_studies_2026-10-03/exog_penalty/_work/gate_spec_ridge.npz`: gate run (the spec's own Python class on the cached design, gate 1 of exog_penalty/SUMMARY.md): max relative difference 1.1e-10 to master_table/sub_ridge_all_features and 8.6e-11 to exog_penalty/ridge_single
- `results/close_studies_2026-10-03/trees_lineartree/_work/smoke/ctrl_all_r10.npz`: smoke run: first 2 refits only (20 forecasts, rows 130..149); bit for bit equal to the whole arm on those rows
- `results/close_studies_2026-10-03/trees_lineartree/_work/smoke/ctrl_base_r10.npz`: smoke run: first 2 refits only (20 forecasts, rows 130..149); bit for bit equal to the whole arm on those rows
- `results/close_studies_2026-10-03/trees_lineartree/_work/smoke/lt_all_r10.npz`: smoke run: first 2 refits only (20 forecasts, rows 130..149); bit for bit equal to the whole arm on those rows
- `results/close_studies_2026-10-03/trees_lineartree/_work/smoke/lt_base_r10.npz`: smoke run: first 2 refits only (20 forecasts, rows 130..149); bit for bit equal to the whole arm on those rows
- `results/close_studies_2026-10-03/trees_pretune/_work/local/lgbm/pretune_s0.npz`: no forecasts: the pre-tune's trial records (one Optuna study); exported to pretune_trials.parquet
- `results/close_studies_2026-10-03/trees_pretune/_work/local/lgbm/pretune_s1.npz`: no forecasts: the pre-tune's trial records (one Optuna study); exported to pretune_trials.parquet

Derived forecasts are not stored: the averages and combinations of `trees_morebars/average.csv` and `kfull_tests/` (seed averages for each k, the average of all pools, ridge + trees with equal weights on the adjusted scale or the variance level, the real-time combination) are rebuilt by `L.rebuild(table, label)` and `L.ridge_trees(...)` from `study_arms.csv`, exactly as the studies built them (np.mean of the members in the same order).

## Verify

`python experiments/close_studies_export.py verify` reloads the export only, re-scores every exported series and every derived forecast, and compares with the study CSVs (tolerance 1e-10 absolute on QLIKE and Sharpe, 1e-10 relative on MSE): 854 comparisons, 0 failed, largest absolute difference 5.3e-15.

| table | rows | comparisons | failed | largest abs. difference |
|---|---|---|---|---|
| `trees_kfull/arms.csv` | 59 | 177 | 0 | 4.9e-15 |
| `trees_morebars/arms.csv` | 31 | 93 | 0 | 5.3e-15 |
| `trees_morebars/average.csv` | 15 | 59 | 0 | 4.9e-15 |
| `trees_datasize/arms.csv` | 23 | 69 | 0 | 4.9e-15 |
| `trees_lineartree/arms.csv` | 16 | 48 | 0 | 4.4e-15 |
| `trees_pretune/local_lgbm_arms.csv` | 12 | 36 | 0 | 5.1e-15 |
| `exog_penalty/headline.csv` | 26 | 78 | 0 | 4.2e-15 |
| `kfull_tests/forecasts.csv` | 90 | 270 | 0 | 9.7e-17 |
| `kfull_tests/realtime_combination.csv` | 12 | 24 | 0 | 6.9e-17 |

The catalogue's reported QLIKE / Sharpe mid of each series (its home study's CSV, or master_table.csv) against the export's score from the home study's first row: 272 comparisons, 0 failed, largest absolute difference 5.3e-15.
The stored pred_clock column: recomputed from the export's pred_adj bit for bit, and within 4.4e-16 relative of score()'s recalibration from row 130 on the trade days (272 checks, 0 failed).
pred_adj against the source npz / stored table, bit for bit: 148 series compared, 0 differ.
Internals against the npz: 63 checks, 0 failed (theta bit for bit; split / kept / anchors bit for bit; gain within float32 rounding).
Coefficient layout: for every exog_penalty arm, [X_row, 1] . theta (float32 theta, prescaled design_bar1600_all_features row, intercept last) reproduces pred within 1.0e-05 relative (the C kernel computes pred = sum_k X[t, k] theta[k] over the augmented row, experiments/close_exogpen_kernel.c).
Overall: PASS (1617 checks).

## Columns

### forecasts.parquet

| column | type | meaning |
|---|---|---|
| `series_id` | string | `<study>/<arm>`; `exog_penalty/<arm>:exact_pred` for the exact-solution check series; `master_table/<key>` for the stored master-table forecasts |
| `study` | string | folder of results/close_studies_2026-10-03/ the series comes from, or `master_table` |
| `row` | int16 | forecast row index 0 .. 1468 (0 = 2018-06-25); trees forecast from row 130 |
| `date` | timestamp | 16:00 bar-end stamp, naive ET, as in the designs |
| `pred_adj` | float64 | the model's own output on the adjusted scale: a forecast of sqrt(RV / B) (dimensionless); NaN where the series has no forecast |
| `pred_clock` | float64 | the research scorer's recalibrated variance forecast (pred_adj^2 + s) x B, s = mean squared adjusted-scale error over the previous 250 rows (at least 63), lagged one row, from the series' own first forecast row; units of true_raw; NaN where undefined |

### targets.parquet

| column | type | meaning |
|---|---|---|
| `row` | int16 | forecast row index |
| `date` | timestamp | 16:00 bar-end stamp, naive ET |
| `true_adj` | float64 | realized target on the adjusted scale, sqrt(RV / B) (the design's y) |
| `true_raw` | float64 | realized variance RV of the 15:30-16:00 bar (the target before the diurnal scaling) |
| `baseline` | float64 | diurnal scale B = true_raw / true_adj^2, as the scorer computes it |
| `baseline_design` | float64 | the design cache's stored B (equal to baseline within 1e-12 relative) |
| `is_trade_day` | bool | one of the 866 trade days 2020-01-03 .. 2024-04-30 of the straddle deck |
| `entry` | float64 | trade days: 15:30 mid of call + put (straddle price) |
| `iv_var` | float64 | trade days: implied variance of the last 30 minutes from the 15:30 book (iv_30^2); the scorer buys when pred_clock > iv_var, else sells |
| `R` | float64 | trade days: straddle return at mid, exit / entry - 1; mid P&L = position x R |
| `exit` | float64 | trade days: straddle payoff at the close |
| `ask_c, ask_p, bid_c, bid_p` | float32 | trade days: 15:30 call / put quotes (kept float32: the scorer adds them in float32); crossed P&L = position x (exit / ask - 1) when buying, x (exit / bid - 1) when selling |
| `signal` | float64 | trade days: the deck's own signal rv_hat - iv_var (not used by the scorer) |

### catalogue.csv

| column | type | meaning |
|---|---|---|
| `series_id, study, arm` |  | identity; arm = the name in the study's own CSVs (pretune walk arms without `walk_`) |
| `kind` |  | forecast / exact batch solution (every 25th row) / stored master-table forecast; for files left out: gate run / check run / smoke run / pre-tune trial records |
| `exported, exclusion_reason` |  | `no` for the npz files not stored as series, with the reason |
| `source_file, source_key` |  | file (relative to the repository) and array the forecast comes from |
| `model, estimator` |  | LightGBM / XGBoost / ridge / lasso / elastic net / OLS; spec estimator name |
| `family, design, bucket, k` |  | rows the model trained on (one-bar, last k bars ending 16:00, 16:00 rows of a longer design, ...), the design cache file, the feature bucket, bars |
| `rows, rows_min, rows_max, sessions_min, sessions_max, window` |  | training rows and distinct sessions in the window at the refits |
| `bar_column` |  | `yes` when the design has the added column bar_end_minute |
| `seed` |  | LightGBM / XGBoost random_state |
| `refit_every, first_forecast_row, n_forecasts, n_pred_clock, n_refits, n_columns` |  | cadence (sessions), first row with a forecast, counts |
| `penalty, alpha_grid, r_grid, l1_ratio` |  | linear arms: penalty structure, the alpha grid searched, the backbone : exogenous ratio grid, the l1 share |
| `linear_tree, linear_lambda, leaf_min, params_json, schedule_json` |  | tree settings (LightGBM min_child_samples or XGBoost min_child_weight; full parameter dict; trees_pretune: configuration schedule) |
| `tuning, group` |  | tuning protocol; the study's arm group |
| `versions_*` |  | library versions recorded in the npz meta (stored master-table trees: from the studies' SUMMARY.md) |
| `cpu_min, cpu_source, wall_min` |  | CPU minutes of the whole arm and where the figure comes from |
| `reported_qlike, reported_sharpe_mid, reported_sharpe_crossed, reported_in` |  | the home study's own numbers for the series (master table: master_table.csv) |
| `scored_from_row, scoreable` |  | first forecast row the home study passed to the scorer; `no` for the exact-solution check series |
| `is_exact_pred, identical_to` |  | exact-solution flag; other series with bit for bit the same pred_adj |
| `has_coefficients, has_importance, npz_keys` |  | which internals are exported; arrays in the npz |
| `meta.*, npz.*` |  | every field of the npz `meta` JSON, flattened with `.` (lists as JSON text); scalar npz arrays |
| `meta_json` |  | the raw `meta` JSON |

### study_arms.csv

| column | type | meaning |
|---|---|---|
| `study, table, label` |  | a row of a study CSV (`label` = its arm / key / forecast name) |
| `recipe` |  | single (an exported series), mean (np.mean of the members in order), weighted (sum of weight x member), variance / realtime (kfull_tests ridge + trees combinations on the variance level) |
| `series_id, members, weights` |  | the series, or the members (series ids, or labels of the same table / study) and weights as JSON |
| `score_start, metrics` |  | first forecast row the study scored from; the CSV columns verify compares |

### linear_coefficients.parquet

| column | type | meaning |
|---|---|---|
| `series_id, row, date` |  | exog_penalty arm and forecast row |
| `<628 design names>, intercept` | float32 | theta at every forecast row, the C kernel's layout: pred = [X_row, 1] . theta, X = the prescaled design_bar1600_all_features row; exactly 0 for columns out of the active set or masked; ols_baseline: the 22 backbone columns only (others NaN, intercept not stored) |

### linear_rechoices.parquet

| column | type | meaning |
|---|---|---|
| `series_id, block, row, date, n_rows` |  | re-choice block (every 250 sessions) and its first forecast row |
| `alpha, r, penalty_<group>` | float64 | chosen penalty, backbone : exogenous ratio, the multi-penalty ridge's 8 group penalties |
| `val_mse, val_mse_shape, val_mse_best, mpr_val_mse, alpha_grid, r_grid` |  | validation MSE over the (r, alpha) grid (flattened, r major), its minimum, the multi-penalty search's MSE, the grids |
| `n_reseed, n_locked, n_masked, locked_cols, masked_cols` |  | cold reseeds in the block; locked (unpenalized, intercept excluded from the count) and masked column sets as indices into linear_columns.csv |
| `n_train, source` |  | window rows; the data-size / bar-count linear arms (spec class) have one row for each 250-solve block with n_train and n_masked only |

### linear_columns.csv

| column | type | meaning |
|---|---|---|
| `col, name, block, group` |  | index into theta (0 .. 628), design name, backbone / exogenous / intercept, exogenous feature group |

### tree_importance.parquet

| column | type | meaning |
|---|---|---|
| `series_id, refit, row` |  | trees_kfull arm, refit index, refit anchor (forecast row) |
| `col` | int16 | index into tree_columns.csv |
| `split` | int16 | number of splits on the column in that refit's booster |
| `gain` | float32 | total gain of the column in that refit's booster |
| `(presence)` |  | a (refit, col) row exists exactly when the window mask kept the column (imp_kept); dropped columns have no split and no gain |

### tree_columns.csv

| column | type | meaning |
|---|---|---|
| `col, name` |  | 628 design columns, then bar_end_minute (index 628) for the arms with the bar column |

### refits.parquet

| column | type | meaning |
|---|---|---|
| `series_id, kind, refit, row, date, n_forecast_rows` |  | tree refit (anchor row, rows it forecasts) or linear solve (every forecast row) |
| `fit_sec, n_train, n_sessions, kept_n, leaf_min` |  | fit seconds, training rows, distinct sessions, columns kept by the window mask, leaf minimum |
| `coef_max, share_linear, mean_feat, fit_lo, fit_hi` | float64 | trees_lineartree: largest absolute leaf coefficient, share of linear leaves, mean features in a linear leaf, in-window fitted range |
| `cfg_*` | float64 | trees_pretune: configuration in force at the refit (num_leaves, min_child_samples, feature_fraction, bagging_fraction, lambda_l2, learning_rate, rounds) |
| `alpha, r, homotopy_events` |  | linear solves: penalty in force (data-size / bar-count arms: recorded at every row; exog_penalty: the block's re-choice), backbone ratio, homotopy events of the session's two updates |

### pretune_trials.parquet

| column | type | meaning |
|---|---|---|
| `study, tpe_seed, trial, fold` |  | trees_pretune pre-tune study (s0 / s1), TPE seed, trial, fold |
| `num_leaves .. learning_rate` | float64 | the trial's configuration |
| `val_mse, fold_val_mse, fold_val_qlike, fold_rounds, fold_rounds_max, fold_sec, fold_n_kept` | float64 | objective (mean validation MSE over folds) and the fold records |
| `fit_first, val_first, val_last` |  | fold dates |

### VERIFY.csv

| column | type | meaning |
|---|---|---|
| `check, table, label, quantity, value, reference, abs_diff, tolerance, passed, detail` |  | one row for each check of the verify stage |

## Not in the export

- The design matrices (`results/close_design/_work/`, about 1 GB, regenerable by `experiments/capture_design_close.py`): statistics that need the features (SHAP, partial dependence, refitting) need the design cache.
- The fitted boosters: no study saved them, so tree structure and leaf values are not available; trees_lineartree kept only summaries of the leaf models (refits.parquet).
- Coefficients of the trees_datasize / trees_morebars ridge and lasso arms: those runs recorded only the penalty at every row and the masked count at every tune (refits.parquet, linear_rechoices.parquet).
- `exact_pred` exists only every 25th session of the lasso / elastic net arms (59 rows), so it is not scoreable on the trade days.
- The Optuna journals of trees_pretune (`*.journal`; the trials are in pretune_trials.parquet) and the cluster CPU of the stored master-table runs.
