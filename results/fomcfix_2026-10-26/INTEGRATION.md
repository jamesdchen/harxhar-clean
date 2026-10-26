# Integrating the FOMC-patched final design (cell B) into the paper

The author adopted cell B of `results/fomcfix_2026-10-26/RULE.txt`: the final 13-bar design (W = 3,000 days, 2014
start) with the two FOMC patches on and holiday sessions kept.  Everything below is on branch
`insurance-selling-2026-09-19-e34-fomcfix`; nothing in the main worktree was edited.

## 1. What to merge from the branch
| path | what |
|---|---|
| `src/data/loading.py` | `FOMC_SCHEDULE_2020_FIX = True` (P1), `NYSE_HOLIDAY_FUTURES_SESSIONS` (123 dates) |
| `src/features/transforms/target.py` | `FOMC_PROXIMITY_NO_WINSOR = True` (P2) |
| `src/backtest/executor.py` | `DROP_HOLIDAY_SESSIONS = False` (holiday arm; off = current design) |
| `experiments/pool_linear_pretune.py` | `--ex-flag` also reaches the loading / target flags |
| `experiments/pooled_trees_rerun_13.py` | walk `--config-file` / `--seed` (defaults unchanged) |
| `writeup/final_run.py` | the switch `REHAUL_FOMCFIX` (section 2) |
| `results/window_sweep_2014start_fomcfix/w3000/` | linear run, same layout as `results/window_sweep_2014start_2026-10-22/w3000` |
| `results/window_sweep_2014start_fomcfix/part2/` | `score14.py` tables for W = 3,000 (as `.../window_sweep_2014start_2026-10-22/part2`) |
| `results/pooled_trees_final_2014start_fomcfix_w3000/` | LightGBM + XGBoost, same layout as `results/pooled_trees_final_2014start_w3000` |
| `results/pooled_rf_skx_2014start_fomcfix_w3000/` | the forest on oneDAL, same layout as `results/pooled_rf_skx_2014start_w3000` -- NOT YET BUILT (section 7) |
| `results/fomcfix_2026-10-26/` | rule, gates, factorial scores, STEP 4 checks, run code |

With P1 and P2 default on, any design built after the merge is the patched design (with both flags off it is the
old one bit for bit, gate G2).

## 2. The switch
Run the generators with
```
export REHAUL_FINAL_RUN=1 REHAUL_FOMCFIX=1        # PowerShell: $env:REHAUL_FINAL_RUN = "1"; $env:REHAUL_FOMCFIX = "1"
```
`REHAUL_FOMCFIX=1` (needs `REHAUL_FINAL_RUN=1` and `REHAUL_FINAL_WINDOW` unset or 3000) sets, where unset,
`REHAUL_LINEAR_RUN=results/window_sweep_2014start_fomcfix/w3000`,
`REHAUL_TREES_RUN=results/pooled_trees_final_2014start_fomcfix_w3000`,
`REHAUL_RF_RUN=results/pooled_rf_skx_2014start_fomcfix_w3000`.  Setting those three variables by hand is equivalent.
To make it permanent instead, change the three defaults in `writeup/final_run.py` (`LINEAR_RUN`, `TREES_RUN`,
`RF_RUN`) to the `FOMCFIX_RUNS` paths for W = 3,000.

Readers that follow the switch (through `final_run` / `pooled_linear_13` / `pooled_trees_13`):
`make_rehaul_oos_rerun.py` (Table 1), `make_rehaul_economics.py`, `make_rehaul_results.py`,
`make_rehaul_oos_table.py`, `make_rehaul_data_methods.py`, `make_rehaul_disclosures.py`, `make_rehaul_pooled_extras.py`.
Checked: `pooled_linear_13` and `pooled_trees_13` load the new runs with all their gates (rows, alpha = frozen
penalty, plain = pred_adj^2 B, the 1,082-day 16:00 check against `scores.csv`, span / window / fold 3).

## 3. Readers that do NOT follow it (hardcoded paths; still the unpatched design)
| generator | reads | consequence |
|---|---|---|
| `make_rehaul_robust_window.py` | `window_sweep_2014start_2026-10-22/w3000, w3500`, `pooled_trees_final_2014start_w{3000,3500}`, `pooled_rf_skx_2014start_w{3000,3500}`, `part2/econ.csv` | the 3,000 vs 3,500 comparison stays on the old design (internally consistent, but its W = 3,000 numbers are not Table 1's any more).  Under the switch its path check is skipped (the variables are set). |
| `make_rehaul_fi_pooled.py` | SHAP / feature importance (`tree_shap_w3000_2014start_2026-10-23`), `window_sweep_.../w3000/frozen_penalties.json`, the old tree forecasts | Section 8 importance is on the unpatched design (FOMC series: P2 changes their values on ~87 % of bars) |
| `make_rehaul_fip_lc.py` | learning curves W = 1,000 .. 3,500 | unpatched |
| ablation run (branch `...-ablations`) | `window_sweep_.../w3000` templates | unpatched |
| MinGRU | | parked, unpatched |
Rerunning any of these is the coordinator's decision (not done here).

## 4. What moves (cell B vs the printed final run; Table 1 sample, 33,527 bars / 2,579 days at 16:00)
| class | QLIKE full, every bar | % vs HAR | QLIKE full, 16:00 | % vs HAR (DM p) | D6 full mid |
|---|---|---|---|---|---|
| ridge | 0.17912 -> 0.17783 | -5.36 -> -6.04 | 0.12789 -> 0.12913 | +0.80 (0.759) -> +1.78 (0.500) | 1.85 -> 1.75 |
| lasso | 0.17747 -> 0.17230 | -6.23 -> -8.97 | 0.13104 -> 0.13225 | +3.28 (0.300) -> +4.24 (0.178) | 1.72 -> 1.75 |
| elastic net | 0.17784 -> 0.17660 | -6.04 -> -6.69 | 0.12622 -> 0.12764 | -0.52 (0.832) -> +0.60 (0.807) | 1.94 -> 1.95 |
| LightGBM | 0.16128 -> 0.16106 | -13.99 -> -14.11 | 0.12013 -> 0.12021 | -2.73 (0.071) -> -2.67 (0.079) | 1.75 -> 1.67 |
| XGBoost | 0.16272 -> 0.16121 | -13.80 -> -14.60 | 0.11924 -> 0.11954 | -3.39 (0.030) -> -3.15 (0.044) | 1.67 -> 1.71 |
| random forest | PENDING: piece 85 of 87 (job 13465922_85) was still running at 15:11 PDT; see section 7 |
HAR-benchmark rows, the OLS, the penalties and the configurations do not change.  Every-bar full-vs-HAR tests stay at
p < 0.001.  Sign change: the elastic net's 16:00 comparison (insignificant both ways).  Generated macros that will
change: everything computed from the full-set pooled forecasts (Table 1, Table 2 / `\rhoos...`, `\rhr...`,
`\rhe...` economics, `\rhor...` counts, pooled extras); data / methods macros of the penalties do not.

## 5. Text the paper needs
* Section 4 (features): the FOMC proximity variables use the pre-announced schedule (the replaced 17-18 March 2020
  meeting is carried as scheduled until the 15 March announcement) and are not winsorized (deterministic ramps).
* The winsorizing sentence (fidelity-7): bounds need 240 non-missing values, so the gappy VIX family is not clipped
  (the text fix you are making), and the two proximity series are exempt.
* Holidays kept, with the planned note: rerun with holiday sessions dropped before official publication
  (`DROP_HOLIDAY_SESSIONS=1`; its effect in `results/fomcfix_2026-10-26/scores_linear`, cells C / D).
* Tree tuning disclosure: `results/fomcfix_2026-10-26/tune2013/` (pre-sample 2013 check: the transferred LightGBM and
  XGBoost configurations rank 5th of 8 / 4th of 8 and are beaten beyond noise by a larger boosting budget; the forest
  is within noise).

## 6. Regenerate and check
```
python writeup/final_run.py                     # every input "ok"
python writeup/tests/check_final_loaders.py
python writeup/make_rehaul_economics.py
python writeup/make_rehaul_results.py
python writeup/make_rehaul_oos_table.py
python writeup/make_rehaul_oos_rerun.py
python writeup/make_rehaul_data_methods.py
python writeup/tests/check_pending.py
python writeup/make_rehaul_disclosures.py
python writeup/make_rehaul_pooled_extras.py
```
(the list of `writeup/final_run.py` REGENERATE, all with `REHAUL_FINAL_RUN=1 REHAUL_FOMCFIX=1`).  The new run
directories' own scores (`scores.csv`, `scores/`, `part2/`) were produced by the printed runs' scorers
(`pooled_linear_rerun_2026_10_19_13.py score`, `score14.py`, `score_trees_final.py` with one compatibility line:
`results/fomcfix_2026-10-26/code/score_trees_fomcfix.py`).

## 7. Still to do: the forest (job 13465922_85)
Until this directory exists, the switch leaves the forest PENDING (final_run's pending rule), so do not regenerate
for the paper before it lands.
1. CARC login node, once `sacct -j 13465922` shows COMPLETED (one command):
   `cd /home1/jc_905/fomcfix_trees/code && PYTHONNOUSERSITE=1 PYTHONPATH=$PWD:$PWD/experiments OMP_NUM_THREADS=1
   OPENSSL_CONF=/home1/jc_905/envs/harxhar-r13/ssl/openssl.cnf /home1/jc_905/trees_final/venv_skx/bin/python
   experiments/pooled_trees_rerun_13.py merge --design /home1/jc_905/fomcfix_lin/runs/B --out
   /home1/jc_905/fomcfix_trees/runs/B --window-days 3000 --refit bar --first-day 2014-01-02 --last-day 2024-04-30
   --est rf_skx --set all_features --require-all --dest /home1/jc_905/fomcfix_trees/runs/B/forecasts_rf.parquet`
2. Fetch `forecasts_rf.parquet` and `forecasts_rf.json`.  Build `results/pooled_rf_skx_2014start_fomcfix_w3000/`
   as `results/pooled_trees_final_2014start_fomcfix_w3000/` was built: `forecasts.parquet` = the template's
   `rf_skx/baseline` rows followed by the new `rf_skx/all_features` rows (columns of the template), `forecasts.json`
   (refit bar, span 2014-01-02 .. 2024-04-30, window_days 3000, pieces), `run.json`, `design/design_check.json`
   (copy of the template's).
3. Scores: `python results/fomcfix_2026-10-26/code/score_trees_fomcfix.py <main worktree>
   3000=<abs path>/results/pooled_rf_skx_2014start_fomcfix_w3000=<abs path>/results/window_sweep_2014start_fomcfix/w3000`
   and `results/fomcfix_2026-10-26/code/fomcfix_score.py ... --trees A=templates --trees B=<lgbm+xgb+rf_skx cell-B file>`
   for the old-vs-new table.
