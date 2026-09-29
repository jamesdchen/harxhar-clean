# Causally tuned per-bar trees (A2b), 2026-09-29

Sources: `a2b/*.csv` (`experiments/a2b_trees_tuned_report.py`), `rescore_local/*.csv` (`experiments/score_trees_tuned.py`), and the cluster scorer's tables in this folder. Every number below is read from those files.

## Run
- Spec `specs/causal_tune_trees_tuned.py` (md5 5c8ac99b0a28): LightGBM, XGBoost and RF on the per-bar linear design. 117 arms (3 input sets × 13 bars × 3 models), 2000-session window, refit every 10 sessions.
- Tuning every 250 sessions (6–7 points per arm, 2018-04 to 2023-11): fit block, 25-session embargo, 125-session validation tail. 32 fixed candidates drawn from grids of 700 / 2800 / 112 configurations; boosted rounds early-stopped, capped at 4× the shipped count.
- MSE picks the configuration of record; the QLIKE-selected path ("qsel") is saved beside it.
- Jobs 12464029–32: 766 of 766 finished with exit 0; 765 of 765 chunks and 117 of 117 arms merged.

## Scorer
- Research scorer: variance = (f² + s)·B, where s is the causal mean squared error of the arm's own previous 250 sessions at that bar, lagged one session. This is not the notebook's 13-bar recalibration.
- sign(s) rule: buy the straddle when the 16:00-bar forecast (issued at 15:30) exceeds the implied variance, otherwise sell it. Mid, and crossed (pay the ask / hit the bid).
- Rows: deck = 866 days (2020-01-03 to 2024-04-30); common = 1,406 sessions (2018-09-24 to 2024-04-30). The target is identical across forecasts to 1e-9.
- Intervals: circular day-block bootstrap (block 21, 2000 draws), paired; Sharpe differences use the same draws on both series.

## (i) 16:00 QLIKE, deck days (% vs reference [95 %]) — `a2b/qlike_1600.csv`, `a2b/qlike_1600_paired.csv`
| input set | forecast | QLIKE | vs ridge | vs lasso | vs untuned twin |
|---|---|---|---|---|---|
| all_features | ridge / lasso | 0.1004 / 0.0998 | lasso −0.6 [−7.0, +4.9] | | |
| | tuned LightGBM | 0.1011 | +0.7 [−5.4, +7.4] | +1.4 [−3.9, +7.7] | +0.9 [−2.7, +4.7] |
| | tuned XGBoost | 0.1049 | +4.5 [−2.7, +14.2] | +5.2 [−3.0, +15.5] | +4.5 [−0.4, +10.9] |
| | tuned RF | 0.1061 | +5.7 [−2.6, +14.1] | +6.4 [+1.2, +11.8] | +0.7 [−2.5, +3.6] |
| live_feasible | ridge / lasso | 0.1005 / 0.0964 | lasso −4.1 [−11.2, +1.8] | | |
| | tuned LightGBM | 0.1053 | +4.7 [−1.8, +11.4] | +9.2 [+3.0, +16.1] | +7.3 [+2.1, +12.6] |
| | tuned XGBoost | 0.1031 | +2.5 [−4.6, +10.4] | +6.9 [+0.2, +14.9] | +4.3 [+0.1, +9.0] |
| | tuned RF | 0.1025 | +1.9 [−6.7, +8.7] | +6.3 [+1.2, +11.5] | +1.6 [−2.4, +5.4] |
| baseline (HAR + calendar) | ridge / lasso | 0.0982 / 0.0972 | lasso −1.1 [−2.3, +0.1] | | |
| | tuned LightGBM | 0.1085 | +10.4 [+5.0, +17.6] | +11.6 [+6.1, +18.8] | +4.4 [−0.1, +10.3] |
| | tuned XGBoost | 0.1109 | +12.9 [+5.4, +23.1] | +14.1 [+6.6, +24.5] | +6.7 [+0.5, +16.2] |
| | tuned RF | 0.1120 | +14.1 [+6.6, +23.2] | +15.3 [+7.7, +24.4] | +7.3 [+0.9, +16.6] |

Untuned LightGBM / XGBoost / RF QLIKE: all_features 0.1002 / 0.1004 / 0.1054; live_feasible 0.0982 / 0.0988 / 0.1009; baseline 0.1039 / 0.1039 / 0.1044. The common sample says the same (e.g. all_features tuned LightGBM +2.6 % [−5.4, +10.5] vs ridge).

Read-off: no tuned-tree QLIKE difference vs the per-bar ridge or lasso has an interval below zero in any input set; tuned vs the shipped (untuned) configuration is above zero with an interval excluding zero in 4 of 9 model × input-set cells and never below.

## (ii) 15:30 sign(s) rule, 866 days, Sharpe mid / crossed (difference vs ridge, mid [95 %]) — `a2b/trade_1600.csv`, `a2b/trade_1600_paired.csv`
- **all_features:** ridge 1.66 / 1.20; lasso 1.45 / 0.98 (−0.22 [−0.81, +0.31]).
  - Tuned: LightGBM 1.81 / 1.35 (+0.15 [−0.60, +0.90]); XGBoost 1.45 / 0.99 (−0.21 [−0.91, +0.50]); RF 1.41 / 0.94 (−0.26 [−1.07, +0.54]). The tuned LightGBM agrees with the ridge's buy/sell call on 82 % of days.
  - Untuned: 1.35 / 1.26 / 1.33.
- **live_feasible:** ridge 1.90 / 1.44; lasso 1.83 / 1.36 (−0.08 [−0.89, +0.79]).
  - Tuned: LightGBM 1.14 / 0.68 (−0.76 [−1.26, −0.24], worse with an interval excluding zero); XGBoost 1.46 / 0.99 (−0.44 [−1.09, +0.26]); RF 1.56 / 1.09 (−0.34 [−0.92, +0.25]).
  - Untuned: 1.69 / 1.70 / 1.58.
- **baseline:** ridge 1.22 / 0.74; tuned 1.01 / 0.75 / 0.58 (all intervals include zero).
- **Tuned minus untuned (mid):** all_features LightGBM +0.47 [−0.26, +1.22], XGBoost +0.19, RF +0.08; live_feasible LightGBM −0.55 [−1.17, +0.04], XGBoost −0.24, RF −0.02; baseline XGBoost −0.37 [−0.80, −0.01], RF −0.59 [−1.12, −0.09].

## (iii) Grid edges (MSE rule; share of the 255 tuning picks per model at the edge / chance = candidate share at that value) — `a2b/grid_edges_{model_axis,bucket_axis,by_date,long}.csv`
- **Widened edges, now at or below chance:** LightGBM `feature_fraction` low 0.192 / 0.188; `lambda_l2` high 0.031 / 0.125. XGBoost `subsample` low 0.247 / 0.406; `colsample_bytree` low 0.125 / 0.250; `reg_lambda` high 0.035 / 0.219. RF `max_features` low 0.031 / 0.219.
- **Still above chance:** LightGBM `num_leaves` = 5: 0.235 / 0.125 (baseline 0.318); XGBoost `max_depth` = 2: 0.192 / 0.125 (baseline 0.247); LightGBM `min_child_samples` = 64: 0.145 / 0.125. The grid may still bind toward smaller trees, most in the HAR + calendar set.
- **Natural bounds:** XGBoost no-penalty 0.616, LightGBM 0.341; RF all columns 0.561, unbounded depth 0.306.
- **Round cap hit:** LightGBM 0.8 %, XGBoost 1.6 %. The shipped configuration was chosen at 2.9 / 4.2 / 3.1 % of tuning points.

## (iv) QLIKE rule vs MSE rule — `a2b/qsel_vs_mse.csv`
- Same configuration at 41–54 % of tuning points.
- Over 13 bars: −0.1 to −1.5 % QLIKE; better at 2–7 bars, worse at 0–3.
- 16:00 deck, qsel vs its MSE path: all_features LightGBM −1.9 % [−3.9, −0.3]; baseline RF −6.0 % [−14.4, −1.0]; live_feasible XGBoost +3.6 % [+0.8, +8.2].
- Sharpe difference (qsel minus MSE) ranges from −0.32 to +0.38; only baseline RF's interval excludes zero: +0.38 [+0.04, +0.81]. Neither rule produces a tree that beats the ridge.

## (v) Tuned vs untuned, 13 bars (mean %, bars better / worse, MSE rule) — `a2b/tuned_vs_untuned_bars.csv`, `a2b/vs_linear_bars.csv`
- LightGBM: all_features +2.0 (0 / 3), live_feasible +2.4 (0 / 6), baseline +3.0 (1 / 10).
- XGBoost: +0.7 (2 / 1), −0.1 (2 / 2), +0.9 (2 / 2).
- RF: +0.1 (1 / 2), +0.3 (2 / 3), −0.4 (5 / 2).
- Tuned vs ridge: +3.9 to +7.7 %, worse at 3–10 bars. Untuned vs ridge: +1.5 to +7.7 %.

## Extreme back-transform values — `a2b/qlike_1600.csv` (columns QLIKE_plain_f2B, QLIKE_as_scored, QLIKE_clock)
- Nothing is clipped or dropped. The stored all_features ridge has f = 0.0047 on 2023-11-24 (and 0.056 on 2020-11-27); its plain f²·B QLIKE on the common sample is 2.64 (one row at 3,517) against 0.119 clock. That day is not a trade day. On the 866 deck days that ridge's minimum f is 0.19 (plain 0.116 vs clock 0.100).
- The trees have f ≥ 0.23, no f ≤ 0, and plain QLIKE 0.004–0.007 below clock. The scored back-transform adds s ≥ 0, which bounds the forecast below by s·B.

## Gates and a reducer defect — `a2b/gates.csv`
- Local scorer: 435 of 435 step-1 gates pass. Tuned gates: 1,371 of 1,410 pass. The 39 failures are "edge labels vs grid file" on the RF arms: `experiments/reduce_trees_tuned_chunks.py`'s pandas round trip turns "None" (unbounded `max_depth`) into an empty cell. Forecasts are unaffected; the reducer is left as shipped.
- `a2b_trees_tuned_report.py` restores "None" from the chunk traces (38 arms, 164 cells, 0 mismatches). Report gates: 112 of 112 pass.

## Outputs
- `results/spxw_pnl/yhat_subtree_tuned_<bucket>_<model>.parquet` (MSE rule, committed) and `yhat_subtree_tunedq_*` (QLIKE rule, local only): 20,044 rows, 1,621 sessions, 2018-01-17 to 2024-04-30. Stamps and rv_raw identical to the untuned tables; the builder reproduces the untuned tables bit for bit.
- `importance/`: per-refit native importance for all 117 arms, TreeSHAP rows at 16:00 for 9 arms (additivity gap ≤ 6.9e-6), per-arm shares and a manifest (parquets local, not committed). TreeSHAP family shares at 16:00, all_features, LightGBM / XGBoost / RF (`a2b/importance_family_1600.csv`): `har_ma_<k>` 0.28 / 0.26 / 0.21; moments 0.17 / 0.17 / 0.17; liquidity 0.16 / 0.18 / 0.16; `har_ma_<k>_x_<session>` 0.13 / 0.09 / 0.24.

## Reproduce
1. `bash cluster/slurm/a2b_pull_treestuned_carc.sh`
2. `python experiments/score_trees_tuned.py --root results/linear_subsection_trees_tuned --untuned-root results/linear_subsection_trees --linear-root results/linear_subsection/arms_hoffman2 --linear-root results/linear_subsection --out results/linear_subsection_trees_tuned/rescore_local`
3. `python experiments/a2b_trees_tuned_report.py`
4. `python experiments/build_subsection_tree_tuned_yhat.py`
