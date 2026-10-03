# The 16:00-bar linear models with the HAR + calendar backbone left (nearly) unpenalized

Written by `experiments/close_exogpen.py analyze` on 2026-10-03 20:12; every number below is read from the CSVs in this folder (and the gate CSVs in `_work/`).

## Question and design

The user: the other shrinkage models should also leave the base HAR features alone and mainly penalize the exogenous features, as the paper's 2-block ridge does (backbone penalty 1, exogenous block 100). The 16:00-bar linear models (`specs/causal_tune_linear.py`) put one penalty on every column. Here the same models are refit on the `all_features` 16:00-bar design (628 columns; cache `results/close_design/_work/design_bar1600_all_features.npz`) with a penalty factor on the backbone (the 22 columns of the `baseline` design: the target's HAR ladder `har_ma_1 .. har_ma_3125` and the calendar / expiry columns), glmnet's penalty.factor convention: column j carries alpha x pf_j x (l1 |b_j| + (1 - l1) / 2 b_j^2), pf = 1 on the 606 exogenous columns.

| arm | backbone | alpha | ratio r = backbone : exogenous penalty |
|---|---|---|---|
| one penalty (spec) | penalized like every column | spec grid, re-chosen every 250 sessions | 1 |
| backbone unpenalized | locked: in the active set always, no penalty (as the spec's intercept) | spec grid | 0 |
| backbone ratio r tuned | r chosen jointly with alpha on the same validation tail | spec grid | {0, 0.01, 0.1, 1} |
| backbone ratio 1/100 | penalized at r times the exogenous penalty | spec grid | 0.01 (the paper's 1 : 100 in ridge units) |
| ridge, one penalty for each group | locked | one ridge penalty for each of the 8 exogenous feature groups (multi-penalty ridge; van de Wiel, van Nee & Rauschenberger 2021), chosen on the validation tail by cyclic coordinate search over the ridge grid (2 passes, start = the backbone-unpenalized ridge's alpha) | 0 |

Why r is tuned rather than fixed: the paper's 1/100 was set for ridge on the pooled 48-bar design; for an L1 penalty the same ratio is a different amount of shrinkage, so the ratio is chosen causally with alpha (r = 1 nests the spec's model, r = 0 nests the unpenalized backbone), and the fixed 1/100 arm is recorded beside it.

Protocol (the spec's, unchanged): window = the 2000 sessions before the forecast, refit every session, intercept unpenalized, the identifiability mask at every re-choice, penalty re-chosen every 250 sessions on the last 125 sessions of the window after a 25-session embargo, the spec's grids (ridge 1e-2 .. 1e3; lasso / elastic net 1e-6 .. 1e-2; elastic net l1_ratio 0.5). When the backbone is locked, backbone columns constant in the window, byte-copies of an earlier backbone column, or beyond the numerical rank of the centered backbone are masked (the five weekday dummies sum to the intercept, so one of them is left out; fit and forecasts are unchanged by that choice).

Algorithm: the spec's own (`RollingTunedLinear` and `src/models/reclasso_har.py`), ported to C (`experiments/close_exogpen_kernel.c`, gcc -O3 -march=native, no -ffast-math, reference LAPACK / BLAS): ridge = Sherman-Morrison rank-one add / drop on the ridged inverse; lasso / elastic net = the Garrigues-El Ghaoui online homotopy (two `enet_online` updates a session), cold seed `_batch_theta` (FWL on the locked block + batch homotopy) at every re-choice, and for the lasso the between-re-choice mask additions (`_degenerate_live`). With pf != 1 the batch elastic net (re-choice candidates, cold seeds) is solved by column scaling and the batch homotopy, then certified by the KKT conditions of the unscaled problem; a solution that fails them (the homotopy's absolute tolerances can miss an event at the scaled magnitudes: found by the independent solver, gate 6) is repaired by support changes (solve the KKT system on the support, drop sign flips, add the worst violator), else by coordinate descent plus the exact KKT solve; counts in `cpu_seconds.csv`. With pf = 1 (the spec's models and the backbone-locked arms) the batch solution is the spec's, unchanged.

## Gates

1. The spec's own class (executed read-only on the cached design) against the stored 16:00 forecasts `results/spxw_pnl/yhat_sub_<est>_all_features.parquet`:

| est | n | max_rel_diff | cpu_sec | alphas |
|---|---|---|---|---|
| ridge | 1469 | 1.1e-10 | 6.8 | [100.0, 1000.0] |
| reclasso | 1469 | 6.8e-13 | 84.0 | [0.001, 0.01] |
| reclasticnet | 1469 | 6.5e-03 | 70.3 | [0.001, 0.01] |

2. The C port, one penalty, all 1469 sessions, against (1) and against the stored tables (relative gap of the 16:00 forecast; CPU seconds for one full arm, Python spec class vs C):

| est | same_alphas_as_spec | max_rel_gap_vs_spec | n_sessions_rel_gap_vs_spec_above_1e9 | max_rel_gap_vs_stored | n_sessions_rel_gap_vs_stored_above_1e9 | python_spec_cpu_sec_full_arm | c_cpu_sec_full_arm |
|---|---|---|---|---|---|---|---|
| ridge | True | 8.6e-11 | 0 | 2.7e-11 | 0 | 6.76 | 8.85 |
| reclasso | True | 1.0e-12 | 0 | 8.0e-13 | 0 | 84.05 | 41.54 |
| reclasticnet | True | 2.7e-03 | 147 | 6.5e-03 | 132 | 70.25 | 36.68 |

   The elastic net is the exception: the C port differs from the spec class on 147 sessions and from the stored table on 132 (largest 0.65%), and the spec class itself differs from the stored table by up to 0.65%. All three are the same warm homotopy (two rank-one updates a session from the last cold seed); on different floating-point paths it takes a different branch at some session and stays there until the next re-choice re-anchors it. Sessions with a relative gap above 1e-9, block by block (`one_penalty_warm_paths_by_block.csv`):

| block | first_forecast | sessions | c_vs_stored | spec_vs_stored | c_vs_spec |
|---|---|---|---|---|---|
| 0 | 2018-06-25 | 250 | 0 | 0 | 0 |
| 1 | 2019-06-26 | 250 | 0 | 0 | 0 |
| 2 | 2020-06-23 | 250 | 0 | 0 | 0 |
| 3 | 2021-06-21 | 250 | 0 | 147 | 147 |
| 4 | 2022-06-17 | 250 | 132 | 132 | 0 |
| 5 | 2023-06-16 | 219 | 0 | 0 | 0 |

3. On a slice (the sessions of the first block), the C backbone-locked arm against the spec's own class with the same locked set (a subclass whose mask step locks the backbone); CPU seconds for that slice, Python vs C:

| est | sessions | alpha_c | alpha_python | max_rel_gap | n_reseed_c_block0 | n_reseed_python_slice | python_cpu_sec_slice | c_cpu_sec_block0_250_sessions |
|---|---|---|---|---|---|---|---|---|
| ridge | 250 | 1000.0 | 1000.0 | 5.1e-13 | 0 | 0 | 1.07 | 1.37 |
| reclasso | 250 | 0.01 | 0.01 | 2.1e-14 | 1 | 1 | 11.7 | 5.17 |
| reclasticnet | 250 | 0.01 | 0.01 | 4.6e-14 | 0 | 0 | 9.7 | 5.59 |

4. The scorer (16:00-bar recalibration (f^2 + s) x B, QLIKE on the 866 trade days, the 15:30 sign(s) straddle; `experiments/dense_vs_sparse_1530.py` helpers) against the master table:

| forecast | qlike | qlike_master | sharpe_mid | sharpe_mid_master |
|---|---|---|---|---|
| ols_baseline | 0.0975 | 0.0975 | 1.33 | 1.33 |
| stored_sub_ridge_baseline | 0.0982 | 0.0982 | 1.22 | 1.22 |
| stored_sub_ridge_all_features | 0.1004 | 0.1004 | 1.66 | 1.66 |
| stored_sub_lasso_baseline | 0.0972 | 0.0972 | 1.51 | 1.51 |
| stored_sub_lasso_all_features | 0.0998 | 0.0998 | 1.45 | 1.45 |
| stored_sub_enet_baseline | 0.0969 | 0.0969 | 1.29 | 1.29 |
| stored_sub_enet_all_features | 0.0994 | 0.0994 | 1.33 | 1.33 |

5. Warm path vs the exact batch solution, every 25th session (lasso / elastic net arms):

| arm | n_checked | max_rel_gap | median_rel_gap | n_checked_rel_gap_above_1e6 |
|---|---|---|---|---|
| lasso_single | 59 | 2.0e-13 | 7.3e-15 | 0 |
| lasso_bb0 | 59 | 1.9e-13 | 7.5e-15 | 0 |
| lasso_bbr | 59 | 2.0e-13 | 6.4e-15 | 0 |
| lasso_bbfix | 59 | 2.0e-13 | 2.9e-15 | 0 |
| enet_single | 59 | 7.8e-13 | 9.1e-15 | 0 |
| enet_bb0 | 59 | 7.7e-06 | 8.3e-15 | 1 |
| enet_bbr | 59 | 2.9e-13 | 7.3e-15 | 0 |
| enet_bbfix | 59 | 3.2e-13 | 9.1e-15 | 0 |
| lasso_singlew | 59 | 2.0e-13 | 7.3e-15 | 0 |
| lasso_bb0w | 59 | 1.9e-13 | 7.5e-15 | 0 |
| enet_singlew | 59 | 2.9e-13 | 6.2e-15 | 0 |
| enet_bb0w | 59 | 7.7e-06 | 7.8e-15 | 1 |

6. An independent solver at a few dates (`experiments/close_exogpen_cdcheck.c`: centered Gram, Cholesky ridge, covariance-form coordinate descent + exact KKT solve on the support) against the C port's backbone-unpenalized arms, first sessions of block 0:

| est | alpha | sessions | max_rel_gap |
|---|---|---|---|
| ridge | 1000.0 | 10 | 4.6e-14 |
| reclasso | 0.01 | 10 | 1.6e-14 |
| reclasticnet | 0.01 | 10 | 2.8e-14 |

## QLIKE and the trade (point estimates; the block bootstrap is off, commit b761b28)

DM = Diebold-Mariano statistic on the daily QLIKE difference (negative: the row forecasts better); t = Newey-West HAC t of the daily mid-fill P&L difference (positive: the row trades better). Reference 'one penalty' = the C port's spec model of the same estimator; reference 'HAR + calendar' = the stored 16:00-bar model of the same estimator on the `baseline` design (the master table's `sub_<est>_baseline`); the multi-penalty ridge is set against the ridge rows.

| model | QLIKE | Sharpe mid / crossed | vs one penalty: dQLIKE %, DM (p) | vs one penalty: dSharpe, t | vs HAR + calendar: dQLIKE %, DM (p) | vs HAR + calendar: dSharpe, t |
|---|---|---|---|---|---|---|
| ridge: HAR + calendar | 0.0982 | 1.22 / 0.74 |  |  |  |  |
| ridge: one penalty (spec) | 0.1004 | 1.66 / 1.20 |  |  | +2.2, +0.41 (0.680) | +0.45, +0.89 |
| ridge: backbone unpenalized | 0.0946 | 1.50 / 1.03 | -5.8, -2.25 (0.025) | -0.17, -0.46 | -3.7, -1.00 (0.315) | +0.28, +0.80 |
| ridge: backbone ratio r tuned | 0.0968 | 1.37 / 0.90 | -3.6, -2.51 (0.012) | -0.29, -1.00 | -1.5, -0.31 (0.757) | +0.16, +0.38 |
| ridge: backbone ratio 1/100 | 0.0945 | 1.52 / 1.05 | -5.8, -2.33 (0.020) | -0.15, -0.41 | -3.8, -1.02 (0.307) | +0.30, +0.86 |
| ridge: backbone unpenalized, one penalty for each group | 0.0997 | 1.43 / 0.95 | -0.7, -0.26 (0.794) | -0.24, -0.76 | +1.5, +0.30 (0.765) | +0.21, +0.40 |
| ridge: one penalty, grid widened (supplement) | 0.1004 | 1.66 / 1.20 | +0.0, +0.00 (1.000) |  | +2.2, +0.41 (0.680) | +0.45, +0.89 |
| ridge: backbone unpenalized, grid widened (supplement) | 0.0940 | 1.52 / 1.05 | -6.4, -2.34 (0.019) | -0.15, -0.40 | -4.3, -1.23 (0.217) | +0.30, +0.90 |
| lasso: HAR + calendar | 0.0972 | 1.51 / 1.04 |  |  |  |  |
| lasso: one penalty (spec) | 0.0998 | 1.45 / 0.98 |  |  | +2.6, +0.72 (0.474) | -0.06, -0.15 |
| lasso: backbone unpenalized | 0.0957 | 1.52 / 1.05 | -4.1, -1.80 (0.072) | +0.07, +0.25 | -1.6, -0.52 (0.606) | +0.01, +0.04 |
| lasso: backbone ratio r tuned | 0.0959 | 1.36 / 0.90 | -3.8, -1.85 (0.064) | -0.08, -0.33 | -1.3, -0.44 (0.657) | -0.15, -0.42 |
| lasso: backbone ratio 1/100 | 0.0956 | 1.48 / 1.02 | -4.1, -1.84 (0.066) | +0.04, +0.13 | -1.6, -0.54 (0.589) | -0.02, -0.06 |
| lasso: one penalty, grid widened (supplement) | 0.0998 | 1.45 / 0.98 | +0.0, +0.00 (1.000) |  | +2.6, +0.72 (0.474) | -0.06, -0.15 |
| lasso: backbone unpenalized, grid widened (supplement) | 0.0957 | 1.52 / 1.05 | -4.1, -1.80 (0.072) | +0.07, +0.25 | -1.6, -0.52 (0.606) | +0.01, +0.04 |
| elastic net: HAR + calendar | 0.0969 | 1.29 / 0.82 |  |  |  |  |
| elastic net: one penalty (spec) | 0.0994 | 1.34 / 0.88 |  |  | +2.6, +0.64 (0.520) | +0.05, +0.12 |
| elastic net: backbone unpenalized | 0.0982 | 1.26 / 0.80 | -1.2, -0.69 (0.488) | -0.08, -0.25 | +1.3, +0.33 (0.739) | -0.03, -0.07 |
| elastic net: backbone ratio r tuned | 0.0987 | 1.21 / 0.74 | -0.8, -0.58 (0.563) | -0.13, -1.00 | +1.8, +0.48 (0.631) | -0.08, -0.18 |
| elastic net: backbone ratio 1/100 | 0.0981 | 1.26 / 0.80 | -1.3, -0.73 (0.468) | -0.08, -0.25 | +1.2, +0.32 (0.748) | -0.03, -0.07 |
| elastic net: one penalty, grid widened (supplement) | 0.1022 | 0.89 / 0.43 | +2.8, +1.80 (0.071) | -0.45, -2.44 | +5.5, +1.45 (0.148) | -0.40, -0.86 |
| elastic net: backbone unpenalized, grid widened (supplement) | 0.0984 | 1.28 / 0.81 | -1.0, -0.52 (0.600) | -0.07, -0.19 | +1.5, +0.40 (0.688) | -0.01, -0.04 |
| OLS: HAR + calendar | 0.0975 | 1.33 / 0.85 |  |  |  |  |

HAR + calendar OLS (master table `sub_ols_baseline`): QLIKE 0.0975, Sharpe 1.33 mid / 0.85 crossed. Each new arm against it: columns `*_vs_ols` of `headline.csv`.

## The penalty chosen at each re-choice

`*` = at an edge of the arm's grid (the spec's grid; the widened grid for the supplement rows).

| arm | 2018-06-25 | 2019-06-26 | 2020-06-23 | 2021-06-21 | 2022-06-17 | 2023-06-16 |
|---|---|---|---|---|---|---|
| ridge_single | 100 | 1000 * | 100 | 100 | 100 | 1000 * |
| ridge_bb0 | 1000 * | 1000 * | 1000 * | 1000 * | 1000 * | 1000 * |
| ridge_bbr | 1000 (r 0.1) * | 1000 (r 1) * | 1000 (r 0.01) * | 1000 (r 0) * | 1000 (r 0.1) * | 1000 (r 0.1) * |
| ridge_bbfix | 1000 * | 1000 * | 1000 * | 1000 * | 1000 * | 1000 * |
| lasso_single | 0.01 * | 0.01 * | 0.01 * | 0.001 | 0.001 | 0.01 * |
| lasso_bb0 | 0.01 * | 0.01 * | 0.01 * | 0.01 * | 0.001 | 0.01 * |
| lasso_bbr | 0.01 (r 1) * | 0.01 (r 1) * | 0.01 (r 0.1) * | 0.01 (r 0) * | 0.001 (r 1) | 0.01 (r 0.1) * |
| lasso_bbfix | 0.01 * | 0.01 * | 0.01 * | 0.01 * | 0.001 | 0.01 * |
| enet_single | 0.01 * | 0.01 * | 0.01 * | 0.001 | 0.001 | 0.01 * |
| enet_bb0 | 0.01 * | 0.01 * | 0.01 * | 0.01 * | 0.001 | 0.01 * |
| enet_bbr | 0.01 (r 1) * | 0.01 (r 1) * | 0.01 (r 0.1) * | 0.01 (r 0) * | 0.001 (r 1) | 0.01 (r 1) * |
| enet_bbfix | 0.01 * | 0.01 * | 0.01 * | 0.01 * | 0.001 | 0.01 * |
| ridge_mpr | groups | groups | groups | groups | groups | groups |
| ridge_singlew | 100 | 1000 | 100 | 100 | 100 | 1000 |
| ridge_bb0w | 1000 | 1000 | 1000 | 1000 | 1000 | 10000 |
| lasso_singlew | 0.01 | 0.01 | 0.01 | 0.001 | 0.001 | 0.01 |
| lasso_bb0w | 0.01 | 0.01 | 0.01 | 0.01 | 0.001 | 0.01 |
| enet_singlew | 0.01 | 0.01 | 0.01 | 0.1 | 0.001 | 0.01 |
| enet_bb0w | 0.01 | 0.01 | 0.1 | 0.01 | 0.001 | 0.01 |

58 of 108 alpha choices sit at an edge of their grid (spec grids: ridge top 1e3, lasso / elastic net top 1e-2; widened grids of the supplement: ridge top 1e6, lasso / elastic net top 1).

Multi-penalty ridge, the penalty of each exogenous group at each re-choice:

| group | 2018-06-25 | 2019-06-26 | 2020-06-23 | 2021-06-21 | 2022-06-17 | 2023-06-16 |
|---|---|---|---|---|---|---|
| moments | 1000 | 1000 | 1000 | 100 | 1000 | 1000 |
| liquidity | 0.1 | 0.1 | 10 | 1 | 0.1 | 1000 |
| market_ew | 1 | 1000 | 1000 | 100 | 1000 | 1000 |
| market_vw | 100 | 0.1 | 1000 | 100 | 1 | 10 |
| sentiment | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 |
| implied_vol | 1000 | 100 | 100 | 1000 | 1000 | 100 |
| vol_demand | 0.1 | 1000 | 1000 | 100 | 1000 | 10 |
| fomc | 0.01 | 1000 | 100 | 0.01 | 0.01 | 1000 |

## Backbone shrinkage and where the forecast variance sits

har-sum ratio = sum of the six HAR coefficients over the HAR + calendar OLS's (median over sessions; 1 = no shrinkage of the persistence); backbone variance ratio = in-window variance of the backbone part of the fit over that of the OLS fit; shares = the forecasts' variance over the 1469 sessions split into the backbone part, the exogenous part and twice their covariance (each part = coefficients x (row - window mean)).

| arm | har_sum_ratio_median | backbone_var_ratio_median | share_backbone_forecasts | share_exog_forecasts | share_cross_forecasts | corr_backbone_part_with_ols | n_exog_nonzero_median |
|---|---|---|---|---|---|---|---|
| ridge_single | 0.57 | 0.32 | 0.25 | 0.36 | 0.39 | 0.95 | 360 |
| ridge_bb0 | 0.85 | 0.70 | 0.68 | 0.07 | 0.25 | 1.00 | 360 |
| ridge_bbr | 0.78 | 0.56 | 0.44 | 0.25 | 0.32 | 0.95 | 360 |
| ridge_bbfix | 0.84 | 0.68 | 0.66 | 0.08 | 0.26 | 0.99 | 360 |
| lasso_single | 0.76 | 0.57 | 0.54 | 0.12 | 0.34 | 0.99 | 18 |
| lasso_bb0 | 0.88 | 0.80 | 0.75 | 0.05 | 0.20 | 1.00 | 14 |
| lasso_bbr | 0.81 | 0.65 | 0.62 | 0.10 | 0.28 | 0.99 | 16 |
| lasso_bbfix | 0.88 | 0.79 | 0.74 | 0.05 | 0.20 | 1.00 | 14 |
| enet_single | 0.74 | 0.54 | 0.50 | 0.14 | 0.36 | 0.99 | 38 |
| enet_bb0 | 0.85 | 0.73 | 0.69 | 0.07 | 0.23 | 0.99 | 33 |
| enet_bbr | 0.76 | 0.57 | 0.56 | 0.12 | 0.32 | 0.99 | 37 |
| enet_bbfix | 0.85 | 0.73 | 0.69 | 0.07 | 0.23 | 0.99 | 33 |
| ridge_mpr | 0.80 | 0.61 | 0.59 | 0.14 | 0.28 | 0.99 | 360 |
| ridge_singlew | 0.57 | 0.32 | 0.25 | 0.36 | 0.39 | 0.95 | 360 |
| ridge_bb0w | 0.85 | 0.71 | 0.70 | 0.06 | 0.24 | 1.00 | 360 |
| lasso_singlew | 0.76 | 0.57 | 0.54 | 0.12 | 0.34 | 0.99 | 18 |
| lasso_bb0w | 0.88 | 0.80 | 0.75 | 0.05 | 0.20 | 1.00 | 14 |
| enet_singlew | 0.74 | 0.54 | 0.54 | 0.13 | 0.34 | 0.99 | 37 |
| enet_bb0w | 0.87 | 0.74 | 0.73 | 0.07 | 0.20 | 0.99 | 31 |

Median HAR coefficients (prescaled design):

| arm | har_ma_1 | har_ma_5 | har_ma_25 | har_ma_125 | har_ma_625 | har_ma_3125 |
|---|---|---|---|---|---|---|
| ridge_single | +0.082 | +0.069 | +0.043 | +0.029 | +0.013 | -0.000 |
| ridge_bb0 | +0.155 | +0.109 | +0.046 | +0.037 | +0.005 | +0.004 |
| ridge_bbr | +0.125 | +0.092 | +0.046 | +0.035 | +0.009 | +0.002 |
| ridge_bbfix | +0.151 | +0.107 | +0.047 | +0.037 | +0.005 | +0.004 |
| lasso_single | +0.141 | +0.082 | +0.051 | +0.035 | +0.000 | +0.000 |
| lasso_bb0 | +0.169 | +0.110 | +0.054 | +0.039 | +0.000 | -0.000 |
| lasso_bbr | +0.156 | +0.099 | +0.058 | +0.038 | +0.000 | +0.000 |
| lasso_bbfix | +0.168 | +0.110 | +0.054 | +0.039 | +0.000 | +0.000 |
| enet_single | +0.130 | +0.087 | +0.048 | +0.039 | +0.000 | +0.000 |
| enet_bb0 | +0.148 | +0.109 | +0.055 | +0.041 | +0.002 | +0.002 |
| enet_bbr | +0.142 | +0.097 | +0.054 | +0.040 | +0.000 | +0.000 |
| enet_bbfix | +0.148 | +0.109 | +0.055 | +0.041 | +0.002 | +0.002 |
| ridge_mpr | +0.142 | +0.089 | +0.051 | +0.032 | +0.016 | +0.001 |
| ridge_singlew | +0.082 | +0.069 | +0.043 | +0.029 | +0.013 | -0.000 |
| ridge_bb0w | +0.158 | +0.108 | +0.046 | +0.041 | +0.004 | +0.004 |
| lasso_singlew | +0.141 | +0.082 | +0.051 | +0.035 | +0.000 | +0.000 |
| lasso_bb0w | +0.169 | +0.110 | +0.054 | +0.039 | +0.000 | -0.000 |
| enet_singlew | +0.136 | +0.080 | +0.054 | +0.039 | +0.000 | +0.000 |
| enet_bb0w | +0.149 | +0.109 | +0.050 | +0.039 | +0.001 | +0.002 |
| HAR + calendar OLS | +0.235 | +0.100 | +0.049 | +0.036 | -0.001 | +0.000 |

## Answer (recorded comparisons)

- ridge: HAR + calendar QLIKE 0.0982 / Sharpe 1.22; all features with one penalty 0.1004 / 1.66.
  - backbone unpenalized: 0.0946 / 1.50; vs one penalty DM -2.25 (p 0.025), HAC t -0.46; vs HAR + calendar DM -1.00 (p 0.315), HAC t +0.80.
  - backbone ratio r tuned: 0.0968 / 1.37; vs one penalty DM -2.51 (p 0.012), HAC t -1.00; vs HAR + calendar DM -0.31 (p 0.757), HAC t +0.38.
  - backbone ratio 1/100: 0.0945 / 1.52; vs one penalty DM -2.33 (p 0.020), HAC t -0.41; vs HAR + calendar DM -1.02 (p 0.307), HAC t +0.86.
  - backbone unpenalized, one penalty for each group: 0.0997 / 1.43; vs one penalty DM -0.26 (p 0.794), HAC t -0.76; vs HAR + calendar DM +0.30 (p 0.765), HAC t +0.40.
  - one penalty, grid widened (supplement): 0.1004 / 1.66; the same forecasts as the one-penalty arm (same choices); vs HAR + calendar DM +0.41 (p 0.680), HAC t +0.89.
  - backbone unpenalized, grid widened (supplement): 0.0940 / 1.52; vs one penalty DM -2.34 (p 0.019), HAC t -0.40; vs HAR + calendar DM -1.23 (p 0.217), HAC t +0.90.
- lasso: HAR + calendar QLIKE 0.0972 / Sharpe 1.51; all features with one penalty 0.0998 / 1.45.
  - backbone unpenalized: 0.0957 / 1.52; vs one penalty DM -1.80 (p 0.072), HAC t +0.25; vs HAR + calendar DM -0.52 (p 0.606), HAC t +0.04.
  - backbone ratio r tuned: 0.0959 / 1.36; vs one penalty DM -1.85 (p 0.064), HAC t -0.33; vs HAR + calendar DM -0.44 (p 0.657), HAC t -0.42.
  - backbone ratio 1/100: 0.0956 / 1.48; vs one penalty DM -1.84 (p 0.066), HAC t +0.13; vs HAR + calendar DM -0.54 (p 0.589), HAC t -0.06.
  - one penalty, grid widened (supplement): 0.0998 / 1.45; the same forecasts as the one-penalty arm (same choices); vs HAR + calendar DM +0.72 (p 0.474), HAC t -0.15.
  - backbone unpenalized, grid widened (supplement): 0.0957 / 1.52; vs one penalty DM -1.80 (p 0.072), HAC t +0.25; vs HAR + calendar DM -0.52 (p 0.606), HAC t +0.04.
- elastic net: HAR + calendar QLIKE 0.0969 / Sharpe 1.29; all features with one penalty 0.0994 / 1.34.
  - backbone unpenalized: 0.0982 / 1.26; vs one penalty DM -0.69 (p 0.488), HAC t -0.25; vs HAR + calendar DM +0.33 (p 0.739), HAC t -0.07.
  - backbone ratio r tuned: 0.0987 / 1.21; vs one penalty DM -0.58 (p 0.563), HAC t -1.00; vs HAR + calendar DM +0.48 (p 0.631), HAC t -0.18.
  - backbone ratio 1/100: 0.0981 / 1.26; vs one penalty DM -0.73 (p 0.468), HAC t -0.25; vs HAR + calendar DM +0.32 (p 0.748), HAC t -0.07.
  - one penalty, grid widened (supplement): 0.1022 / 0.89; vs one penalty DM +1.80 (p 0.071), HAC t -2.44; vs HAR + calendar DM +1.45 (p 0.148), HAC t -0.86.
  - backbone unpenalized, grid widened (supplement): 0.0984 / 1.28; vs one penalty DM -0.52 (p 0.600), HAC t -0.19; vs HAR + calendar DM +0.40 (p 0.688), HAC t -0.04.

## Caveats

- Point estimates with 866 trade days; the Sharpe ratios of these models differ by amounts of the order of their sampling error (the HAC t column).
- The ratio r and the group penalties are chosen on 125-session tails; with the spec's grids, several choices sit at a grid edge (table above), as in the spec's own one-penalty models.
- The lasso / elastic net warm path is the spec's algorithm; where it leaves the exact path (gate 5) the forecast differs from the exact window optimum until the next re-choice.
- The HAR + calendar references are the stored master-table forecasts (scorer gate 4).

## CPU

C walk-forward, all 19 arms: 1044 CPU seconds (`cpu_seconds.csv`; two processes, single-threaded BLAS). Batch solutions with pf != 1 repaired: 125 by support changes, 10 by descent, 8 left uncertified.

## Files

- `experiments/close_exogpen.py` (stages gate, check, run, analyze), `experiments/close_exogpen_kernel.c` (the C port), `experiments/close_exogpen_cdcheck.c` (an independent coordinate-descent / eigendecomposition solver kept as a cross-check, not the arms of record).
- `headline.csv`, `scorer_gate.csv`, `penalty_path.csv`, `group_penalties.csv`, `one_penalty_warm_paths_by_block.csv`, `warm_path_vs_exact.csv`, `backbone_shrinkage_and_variance_shares.csv`, `har_coefficients.csv`, `cpu_seconds.csv`; figures `fig_qlike_sharpe.png`, `fig_penalty_path.png`, `fig_variance_shares.png`; forecasts and coefficients `_work/runs/*.npz`; gates `_work/gate_*.csv`, `_work/check_*.csv`.
