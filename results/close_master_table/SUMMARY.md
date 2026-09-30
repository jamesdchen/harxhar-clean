# Master table for the closing strategy (A4) — summary

Written by `experiments/master_table_close.py --arm-root linear_subsection_dedup` on 2026-09-30 08:30; every number below is read from
`master_table.csv` (and `before_after.csv`) of the same run. Full table: `writeup/master_table_close.pdf`; tex: `writeup/generated/table_master_close.tex`.

## Provenance: design, window mask, CPU class

- **Arm root:** `results/linear_subsection_dedup/` (de-duplicated design (16:00 campaign)); it supplies the arm-only rows, the per-bar OLS incumbent, the arm files the per-bar linear and VIX-only tables are gated against, and the common 16:00 target; its 16:00 target equals the default root's bit for bit (1469 rows, 0 differ; gate passed).
- **Forecast tables:** every `results/spxw_pnl/yhat_*.parquet` on disk at run time (glob). After the 16:00 campaign (`writeup/CAMPAIGN_16H_2026-09-29.md`) the per-bar tables are the de-duplicated design (the 12 HAR x open/close session-edge columns dropped from the per-bar design; the pooled 48-bar models keep them); every tree and LSTM fit uses the per-window mask (`src/models/window_mask.py`: drop columns constant on the fit's training window and exact copies of an earlier kept column) and ran pinned to one CPU class (CARC epyc-7513), because tree and LSTM numbers depend on the CPU class (agent D's finding).

| family | forecasts (A / B) | design | window mask | refit | tuning | CPU class | vs the pre-campaign table |
|---|---:|---|---|---|---|---|---|
| paper | 9 / 0 | pooled 48-bar models (session-edge interactions kept, campaign decision 1) | n/a | as the paper | as the paper | not re-run | nothing |
| per-bar linear | 10 / 0 | de-duplicated per-bar design (commit 47f7f9c: the 12 HAR x open/close session-edge columns dropped) | identifiability mask at every penalty tune (unchanged: it had already removed the 12 columns) | every session | penalty every 250 sessions | Hoffman2 (agent A) | design de-dup (the mask had already dropped the session-edge columns: float path only) |
| pooled twin | 9 / 0 | pooled 48-bar twins (session-edge interactions kept) | identifiability mask | every session | penalty every 250 sessions | not re-run | nothing |
| VIX-only family | 21 / 0 | de-duplicated per-bar design (commit 47f7f9c: the 12 HAR x open/close session-edge columns dropped) | identifiability mask at every penalty tune (unchanged: it had already removed the 12 columns) | every session | penalty every 250 sessions | Hoffman2 (agent A) | design de-dup (the mask had already dropped the session-edge columns: float path only) |
| per-bar tree (untuned) | 9 / 0 | de-duplicated per-bar design | per-window mask (src/models/window_mask.py, WINDOW_MASK=1) | every 10 sessions (T10) | none (shipped configuration) | CARC, pinned to one CPU class (epyc-7513) | design de-dup + per-window mask + CPU-class pinning (refit cadence unchanged, every 10) |
| per-bar tree (untuned, daily refit) | 9 / 0 | de-duplicated per-bar design | per-window mask (src/models/window_mask.py, WINDOW_MASK=1) | every session (T1) | none (shipped configuration) | CARC, pinned to one CPU class (epyc-7513) | new |
| per-bar tree (tuned) | 18 / 0 | de-duplicated per-bar design | per-window mask (src/models/window_mask.py, WINDOW_MASK=1) | every 10 sessions (RS10) | random search, 32 candidates, every 250 sessions (MSE rule; tunedq = QLIKE rule) | CARC, pinned to one CPU class (epyc-7513) | design de-dup + per-window mask + CPU-class pinning (refit / tuning cadence unchanged) |
| per-bar tree (random search, daily) | 18 / 0 | de-duplicated per-bar design | per-window mask (src/models/window_mask.py, WINDOW_MASK=1) | every session (RS1) | random search, 32 candidates, every 250 sessions (MSE rule; tunedq = QLIKE rule) | CARC, pinned to one CPU class (epyc-7513) | new |
| per-bar tree (Optuna, TUNE_PER=1, best-of-50) | 9 / 0 | de-duplicated per-bar design | per-window mask (src/models/window_mask.py, WINDOW_MASK=1) | every session | Optuna TPE, 50 trials, best of the first 50, every 1 session(s) | CARC, single-class re-run on epyc-7513 (agent C; canonical) | new |
| per-bar tree (Optuna, TUNE_PER=5, best-of-50) | 9 / 0 | de-duplicated per-bar design | per-window mask (src/models/window_mask.py, WINDOW_MASK=1) | every session | Optuna TPE, 50 trials, best of the first 50, every 5 session(s) | CARC, single-class re-run on epyc-7513 (agent C; canonical) | new |
| per-bar tree (Optuna, TUNE_PER=25, best-of-50) | 18 / 0 | de-duplicated per-bar design | per-window mask (src/models/window_mask.py, WINDOW_MASK=1) | every session | Optuna TPE, 50 trials, best of the first 50, every 25 session(s) (MSE rule; optunaq = QLIKE rule) | CARC, single-class re-run on epyc-7513 (agent C; canonical) | new |
| per-bar tree (Optuna, TUNE_PER=250, best-of-50) | 9 / 0 | de-duplicated per-bar design | per-window mask (src/models/window_mask.py, WINDOW_MASK=1) | every session | Optuna TPE, 50 trials, best of the first 50, every 250 session(s) | CARC, single-class re-run on epyc-7513 (agent C; canonical) | new |
| per-bar tree (Optuna, TUNE_PER=1, best-of-25) | 9 / 0 | de-duplicated per-bar design | per-window mask (src/models/window_mask.py, WINDOW_MASK=1) | every session | Optuna TPE, 50 trials, best of the first 25, every 1 session(s) | CARC, single-class re-run on epyc-7513 (agent C; canonical) | new |
| per-bar tree (Optuna, TUNE_PER=25, best-of-25) | 9 / 0 | de-duplicated per-bar design | per-window mask (src/models/window_mask.py, WINDOW_MASK=1) | every session | Optuna TPE, 50 trials, best of the first 25, every 25 session(s) | CARC, single-class re-run on epyc-7513 (agent C; canonical) | new |
| per-bar tree (Optuna, TUNE_PER=1, best-of-10) | 9 / 0 | de-duplicated per-bar design | per-window mask (src/models/window_mask.py, WINDOW_MASK=1) | every session | Optuna TPE, 50 trials, best of the first 10, every 1 session(s) | CARC, single-class re-run on epyc-7513 (agent C; canonical) | new |
| per-bar tree (Optuna, TUNE_PER=25, best-of-10) | 9 / 0 | de-duplicated per-bar design | per-window mask (src/models/window_mask.py, WINDOW_MASK=1) | every session | Optuna TPE, 50 trials, best of the first 10, every 25 session(s) | CARC, single-class re-run on epyc-7513 (agent C; canonical) | new |
| LSTM | 6 / 0 | de-duplicated per-bar design | per-window mask, recomputed at every refit | every session (was every 10) | 16 configurations x 5 seeds, every 250 sessions (MSE rule; qsel = QLIKE rule) | CARC, pinned to one CPU class (epyc-7513) | design de-dup + per-window mask + refit cadence 10 -> 1 + CPU-class pinning |
| LSTM (intraday sequence) | 6 / 0 | last N half-hour bars to 15:30 (N tuned in {13, 48, 96}), bar-level inputs | per-window mask on the step columns | every session | every 250 sessions (MSE rule; qsel = QLIKE rule) | CARC, pinned to epyc-7513 (agent I) | new |
| direct rest-of-day at 15:30 (check) | 9 / 0 | de-duplicated per-bar design, 15:30 clock of the rest-of-day model | identifiability mask | every session | penalty every 250 sessions | Hoffman2, each 15:30 arm pinned to its per-bar twin's CPU architecture (agent E) | design de-dup + CPU-architecture pinning to the per-bar twin |
| implied-vol representations | 15 / 0 | de-duplicated per-bar design (commit 47f7f9c: the 12 HAR x open/close session-edge columns dropped) | identifiability mask at every penalty tune (unchanged: it had already removed the 12 columns) | every session | penalty every 250 sessions | Hoffman2 (agent A) | design de-dup (the mask had already dropped the session-edge columns: float path only) |
| HAR-ladder variants | 9 / 0 | de-duplicated per-bar design (commit 47f7f9c: the 12 HAR x open/close session-edge columns dropped) | identifiability mask at every penalty tune (unchanged: it had already removed the 12 columns) | every session | penalty every 250 sessions | Hoffman2 (agent A) | design de-dup (the mask had already dropped the session-edge columns: float path only) |
| chain-period buckets | 0 / 21 | de-duplicated per-bar design (commit 47f7f9c: the 12 HAR x open/close session-edge columns dropped) | identifiability mask at every penalty tune (unchanged: it had already removed the 12 columns) | every session | penalty every 250 sessions | Hoffman2 (agent A) | design de-dup (the mask had already dropped the session-edge columns: float path only) |

## Scorer, days, reference

- **Scorer: the research convention** (`compare_mfiv_harlag.py` / `score_trees_subsection.py`, their functions imported): the 16:00 bar is recalibrated on its own, forecast = (f² + s)·B with s = the forecast's own mean squared adjusted-scale error at 16:00 over the previous 250 sessions (≥ 63), lagged one session; QLIKE against the per-bar spec's 16:00 target; the trade is the deck's 15:30 sign(s) on the straddle (`trade_1530`). The notebook's 13-bar Mincer–Zarnowitz map is **not** used: no number here may be set beside a notebook number.
- **Days:** table A = 220 forecasts in 20 families (+ 9 check rows) on the same 866 trade days (2020-01-03 .. 2024-04-30); intersection over table A = 866 of 866 trade days. Every family that covers the 866 trade days is in table A. Table B = 21 forecasts scored on their own days (paired with the reference on those days): 21 chain-period buckets forecasts from 2022-01-03 on 550 trade days. Forecasts that failed to load: none.
- **Which forecasts drop days:** no table-A forecast misses a trade day; 0 forecasts miss trade days inside their own window. The 21 table-B forecasts start at their comparison window (2022-01-03), so each omits the 316 earlier trade days by design (their implied-vol inputs are real chain data only from then; compare_mfiv_harlag.CHAIN_START). Coverage per forecast: `master_table_days.csv`.
- **Reference:** the block-diagonal ridge (`blk2`, `yhat_blk2_fomc1.parquet`) — the paper's headline forecast (HAR ladder, penalty 1, plus the exogenous block, penalty 100, on the panel of record), the forecast the paper's 15:30 deck trades. Under this scorer it has QLIKE 0.1058 and sign(s) Sharpe 1.00 mid / 0.53 crossed (the deck's own map gives it 1.34 mid; the difference is the scorer). Second reference: always short, Sharpe 0.20 mid / -0.27 crossed.

## (a) Recommended headline forecast

**per-bar ridge [live_feasible]** (`sub_ridge_live_feasible`), scored by the research convention above. On the 866 days: QLIKE 0.1005 (-5.0 % vs the reference, day-block interval on the daily difference [-0.0113, +0.0005], DM -1.42, p 0.156); sign(s) Sharpe 1.90 mid / 1.44 crossed, mean +0.134 per unit premium (crossed +0.101), hit rate 54.6 %, buys on 348 days (40.2 %). Sharpe difference vs the reference: mid +0.90 [+0.19, +1.66], crossed +0.91 [+0.19, +1.67]; vs always short: mid +1.70 [+0.09, +3.11], crossed +1.71 [+0.12, +3.12].

- **Does any forecast now beat the headline?** Of the 219 other table-A forecasts (check rows and exact duplicates left out), with the paired Sharpe-difference interval wholly above zero: mid 0 (none); crossed 0 (none). With the QLIKE-difference interval wholly below zero (better accuracy): 3 (per-bar elastic net [live_feasible], HAR ladder base3 -7.1 %; per-bar elastic net [live_feasible], HAR ladder base2 -6.1 %; per-bar lasso [live_feasible], HAR ladder base2 -5.8 %).

Why this one:
- It is a 15:30-specific model on the inputs a 15:30 forecaster can rebuild live (`live_feasible`), so the trade it scores can be run. It ranks 2 of 220 table-A forecasts on sign(s) Sharpe (mid) and 63 of 220 on the recalibrated QLIKE.
- The highest Sharpe in table A is per-bar lasso [live_vix_only] (1.91 mid / 1.44 crossed). Choosing the maximum of 220 Sharpe ratios would select on the trade's own noise; the headline is chosen on feasibility and on being the per-bar model of record of the research scorer (the per-bar ridge is the estimator the 15:30 study defined first), not on the maximum.
- Paired against the headline itself on the same days (`master_table_vs_headline.csv`; a negative QLIKE % / DM = the alternative forecasts better, a positive Sharpe difference = the alternative trades better; 95 % intervals):

  | alternative | QLIKE | % vs headline | DM | Sharpe mid / crossed | ΔSharpe mid vs headline | ΔSharpe crossed vs headline | same position |
  |---|---:|---:|---:|---|---|---|---:|
  | per-bar lasso [live_feasible] | 0.0964 | -4.1 | -1.20 | 1.83 / 1.36 | -0.08 [-0.89, +0.79] | -0.08 [-0.90, +0.79] | 89.3 % |
  | per-bar elastic net [live_feasible] | 0.0968 | -3.7 | -1.38 | 1.41 / 0.94 | -0.49 [-1.08, +0.14] | -0.50 [-1.09, +0.14] | 90.9 % |
  | per-bar ridge [all_features] | 0.1004 | -0.2 | -0.06 | 1.66 / 1.20 | -0.24 [-1.07, +0.54] | -0.24 [-1.08, +0.54] | 86.0 % |
  | per-bar ridge [baseline] | 0.0982 | -2.3 | -0.53 | 1.22 / 0.74 | -0.69 [-1.70, +0.23] | -0.70 [-1.72, +0.23] | 82.4 % |
  | per-bar lasso [free_vix_only] | 0.0973 | -3.2 | -0.93 | 1.88 / 1.42 | -0.02 [-0.77, +0.77] | -0.02 [-0.79, +0.77] | 88.0 % |
  | per-bar ridge [free_vix_only] | 0.0999 | -0.6 | -0.65 | 1.83 / 1.36 | -0.07 [-0.46, +0.29] | -0.08 [-0.46, +0.29] | 95.2 % |
  | per-bar LightGBM [live_feasible] | 0.0991 | -1.4 | -0.46 | 1.59 / 1.12 | -0.31 [-1.04, +0.41] | -0.32 [-1.04, +0.41] | 85.1 % |
  | per-bar XGBoost [live_feasible] | 0.0995 | -1.1 | -0.32 | 1.27 / 0.80 | -0.63 [-1.51, +0.16] | -0.64 [-1.54, +0.16] | 86.3 % |
  | per-bar LightGBM [all_features] | 0.1007 | +0.1 | +0.04 | 1.84 / 1.37 | -0.06 [-0.84, +0.71] | -0.07 [-0.85, +0.70] | 84.5 % |
  | per-bar XGBoost, daily refit [all_features] | 0.0991 | -1.5 | -0.43 | 1.69 / 1.22 | -0.21 [-0.88, +0.50] | -0.21 [-0.89, +0.49] | 84.6 % |
  | per-bar LightGBM, daily refit [live_feasible] | 0.0955 | -5.0 | -1.62 | 1.42 / 0.95 | -0.48 [-1.58, +0.57] | -0.49 [-1.60, +0.57] | 85.1 % |
  | tuned per-bar random forest [live_feasible], QLIKE-selected | 0.1040 | +3.4 | +0.86 | 1.90 / 1.44 | -0.00 [-0.56, +0.59] | -0.00 [-0.57, +0.59] | 85.8 % |
  | tuned per-bar LightGBM [live_feasible], MSE-selected | 0.0992 | -1.3 | -0.35 | 1.47 / 1.00 | -0.43 [-1.22, +0.35] | -0.44 [-1.23, +0.35] | 85.3 % |
  | tuned per-bar LightGBM, daily refit [live_feasible], MSE-selected | 0.0994 | -1.2 | -0.31 | 1.63 / 1.17 | -0.27 [-1.07, +0.57] | -0.27 [-1.08, +0.57] | 86.4 % |
  | Optuna per-bar LightGBM, TUNE_PER 1, best of 25 [all_features] | 0.1046 | +4.0 | +1.25 | 1.71 / 1.24 | -0.20 [-1.09, +0.67] | -0.20 [-1.10, +0.67] | 83.9 % |
  | Optuna per-bar random forest, TUNE_PER 25, best of 50, QLIKE-selected [live_feasible] | 0.0999 | -0.7 | -0.17 | 1.23 / 0.76 | -0.67 [-1.51, +0.18] | -0.68 [-1.53, +0.18] | 85.8 % |
  | per-bar LSTM [all_features] | 0.1305 | +29.8 | +4.29 | 1.33 / 0.85 | -0.57 [-1.85, +0.83] | -0.59 [-1.87, +0.82] | 73.6 % |
  | per-bar LSTM, QLIKE-selected [baseline] | 0.1031 | +2.5 | +0.63 | 0.77 / 0.30 | -1.13 [-2.08, -0.15] | -1.14 [-2.09, -0.15] | 81.4 % |
  | intraday-sequence LSTM, QLIKE-selected [live_feasible] | 0.1080 | +7.4 | +1.51 | 1.71 / 1.23 | -0.20 [-1.19, +0.89] | -0.21 [-1.22, +0.88] | 80.7 % |
  | intraday-sequence LSTM, QLIKE-selected [baseline] | 0.1071 | +6.5 | +1.38 | 0.87 / 0.39 | -1.04 [-1.99, -0.11] | -1.05 [-1.99, -0.12] | 77.8 % |

- Over all 219 other table-A forecasts, 0 trade better than the headline with an interval above zero (mid; 0 crossed) and 43 trade worse with an interval below zero (mid; 43 crossed); 3 forecast better on QLIKE with the day-block interval below zero, 37 worse.
- Against the reference, 10 of 219 table-A forecasts have a mid-fill Sharpe-difference interval wholly above zero; 10 at the crossed fill. Against always short, 47 (mid) and 58 (crossed) of 220.
- On forecast accuracy alone the best live-feasible forecast is per-bar elastic net [live_feasible], HAR ladder base3 (QLIKE 0.0934, Sharpe 1.57 / 1.11); a QLIKE-first choice would take it. The trade does not rank forecasts the way QLIKE does (section b), which is why the headline names the scorer AND the trade numbers.

## Before / after the 16:00 campaign

Before = the pre-campaign master table (commit 6177b74; snapshot `results/spxw_pnl/pre_dedup_2026-09-29/`), after = this run; every key present in both is paired on the same days with the same resampled days as every other interval (`before_after.csv`; difference = after minus before).
- **Headline** (`sub_ridge_live_feasible`; design de-dup (the mask had already dropped the session-edge columns: float path only); re-run on hoffman2 (was carc)): QLIKE 0.1005 -> 0.1005 (+0.00 %, interval on the daily difference [+0.00000, +0.00000]); Sharpe mid 1.90 -> 1.90 (+0.00 [+0.00, +0.00]), crossed 1.44 -> 1.44 (+0.00 [+0.00, +0.00]); positions changed on 0 of 866 days; largest relative change of the recalibrated forecast 1.01e-12.
- **Reference** (`blk2`; nothing): QLIKE 0.1058 -> 0.1058 (+0.00 %, interval on the daily difference [+0.00000, +0.00000]); Sharpe mid 1.00 -> 1.00 (+0.00 [+0.00, +0.00]), crossed 0.53 -> 0.53 (+0.00 [+0.00, +0.00]); positions changed on 0 of 866 days; largest relative change of the recalibrated forecast 0.00e+00.

| family | keys in both | what changed | QLIKE % change, median [min, max] | ΔSharpe mid, median [min, max] | ΔSharpe mid interval above / below 0 | ΔSharpe crossed interval above / below 0 | positions changed, median [max] |
|---|---:|---|---|---|---|---|---|
| paper | 9 | nothing | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | 0 / 0 | 0 / 0 | 0 [0] |
| per-bar linear | 10 | design de-dup (the mask had already dropped the session-edge columns: float path only) | -0.00 [-0.00, +0.05] | +0.00 [+0.00, +0.00] | 0 / 0 | 0 / 0 | 0 [0] |
| pooled twin | 9 | nothing | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | 0 / 0 | 0 / 0 | 0 [0] |
| VIX-only family | 21 | design de-dup (the mask had already dropped the session-edge columns: float path only) | +0.00 [-0.04, +0.09] | +0.00 [-0.05, +0.06] | 0 / 0 | 0 / 0 | 0 [2] |
| per-bar tree (untuned) | 9 | design de-dup + per-window mask + CPU-class pinning (refit cadence unchanged, every 10) | +0.63 [+0.26, +1.64] | -0.10 [-0.43, +0.49] | 0 / 2 | 0 / 2 | 26 [49] |
| per-bar tree (tuned) | 18 | design de-dup + per-window mask + CPU-class pinning (refit / tuning cadence unchanged) | +0.94 [-5.79, +16.14] | +0.07 [-0.27, +0.48] | 0 / 0 | 0 / 0 | 48 [80] |
| LSTM | 6 | design de-dup + per-window mask + refit cadence 10 -> 1 + CPU-class pinning | -4.40 [-17.46, -1.68] | +0.40 [-0.16, +0.80] | 1 / 0 | 1 / 0 | 128 [172] |
| direct rest-of-day at 15:30 (check) | 9 | design de-dup + CPU-architecture pinning to the per-bar twin | -0.00 [-0.00, +0.00] | +0.00 [-0.01, +0.00] | 0 / 0 | 0 / 0 | 0 [1] |
| implied-vol representations | 15 | design de-dup (the mask had already dropped the session-edge columns: float path only) | +0.00 [-0.00, +0.06] | +0.00 [+0.00, +0.00] | 0 / 0 | 0 / 0 | 0 [0] |
| HAR-ladder variants | 9 | design de-dup (the mask had already dropped the session-edge columns: float path only) | +0.00 [-0.01, +0.01] | +0.00 [-0.00, +0.00] | 0 / 0 | 0 / 0 | 0 [1] |
| chain-period buckets | 21 | design de-dup (the mask had already dropped the session-edge columns: float path only) | +0.00 [-0.00, +0.00] | +0.00 [+0.00, +0.00] | 0 / 0 | 0 / 0 | 0 [0] |

New in this run (114 keys, `status = new` in `before_after.csv`): per-bar tree (untuned, daily refit) 9; per-bar tree (random search, daily) 18; per-bar tree (Optuna, TUNE_PER=1, best-of-50) 9; per-bar tree (Optuna, TUNE_PER=5, best-of-50) 9; per-bar tree (Optuna, TUNE_PER=25, best-of-50) 18; per-bar tree (Optuna, TUNE_PER=250, best-of-50) 9; per-bar tree (Optuna, TUNE_PER=1, best-of-25) 9; per-bar tree (Optuna, TUNE_PER=25, best-of-25) 9; per-bar tree (Optuna, TUNE_PER=1, best-of-10) 9; per-bar tree (Optuna, TUNE_PER=25, best-of-10) 9; LSTM (intraday sequence) 6.

Keys whose own Sharpe (mid) moved with the paired interval excluding zero: per-bar XGBoost [live_feasible] 1.70 -> 1.27 (-0.43 [-0.94, -0.03]); per-bar XGBoost [baseline] 1.12 -> 0.92 (-0.21 [-0.42, -0.04]); per-bar LSTM [baseline] 0.61 -> 1.26 (+0.65 [+0.05, +1.27]).

## (b) QLIKE vs Sharpe across models

- **paper** (9): QLIKE 0.1006 .. 0.1172, Sharpe mid 1.00 .. 1.54; best QLIKE LightGBM (0.1006, Sharpe 1.48); best Sharpe lasso (fixed 1e-4) (1.54, QLIKE 0.1067).
- **per-bar linear** (10): QLIKE 0.0964 .. 0.1005, Sharpe mid 1.22 .. 1.90; best QLIKE per-bar lasso [live_feasible] (0.0964, Sharpe 1.83); best Sharpe per-bar ridge [live_feasible] (1.90, QLIKE 0.1005).
- **pooled twin** (9): QLIKE 0.1008 .. 0.1106, Sharpe mid 0.99 .. 1.32; best QLIKE pooled elastic net [all_features], same spec (0.1008, Sharpe 1.19); best Sharpe pooled elastic net [live_feasible], same spec (1.32, QLIKE 0.1020).
- **VIX-only family** (21): QLIKE 0.0954 .. 0.1016, Sharpe mid 1.26 .. 1.91; best QLIKE per-bar lasso [vix_rvol] (0.0954, Sharpe 1.63); best Sharpe per-bar lasso [live_vix_only] (1.91, QLIKE 0.0969).
- **per-bar tree (untuned)** (9): QLIKE 0.0991 .. 0.1060, Sharpe mid 0.82 .. 1.84; best QLIKE per-bar LightGBM [live_feasible] (0.0991, Sharpe 1.59); best Sharpe per-bar LightGBM [all_features] (1.84, QLIKE 0.1007).
- **per-bar tree (untuned, daily refit)** (9): QLIKE 0.0955 .. 0.1041, Sharpe mid 0.95 .. 1.69; best QLIKE per-bar LightGBM, daily refit [live_feasible] (0.0955, Sharpe 1.42); best Sharpe per-bar XGBoost, daily refit [all_features] (1.69, QLIKE 0.0991).
- **per-bar tree (tuned)** (18): QLIKE 0.0992 .. 0.1195, Sharpe mid 0.88 .. 1.90; best QLIKE tuned per-bar LightGBM [live_feasible], MSE-selected (0.0992, Sharpe 1.47); best Sharpe tuned per-bar random forest [live_feasible], QLIKE-selected (1.90, QLIKE 0.1040).
- **per-bar tree (random search, daily)** (18): QLIKE 0.0994 .. 0.1171, Sharpe mid 0.71 .. 1.63; best QLIKE tuned per-bar LightGBM, daily refit [live_feasible], MSE-selected (0.0994, Sharpe 1.63); best Sharpe tuned per-bar LightGBM, daily refit [live_feasible], MSE-selected (1.63, QLIKE 0.0994).
- **per-bar tree (Optuna, TUNE_PER=1, best-of-50)** (9): QLIKE 0.1002 .. 0.1086, Sharpe mid 0.60 .. 1.41; best QLIKE Optuna per-bar random forest, TUNE_PER 1, best of 50 [live_feasible] (0.1002, Sharpe 1.41); best Sharpe Optuna per-bar random forest, TUNE_PER 1, best of 50 [live_feasible] (1.41, QLIKE 0.1002).
- **per-bar tree (Optuna, TUNE_PER=5, best-of-50)** (9): QLIKE 0.1006 .. 0.1076, Sharpe mid 0.61 .. 1.54; best QLIKE Optuna per-bar random forest, TUNE_PER 5, best of 50 [live_feasible] (0.1006, Sharpe 1.32); best Sharpe Optuna per-bar XGBoost, TUNE_PER 5, best of 50 [all_features] (1.54, QLIKE 0.1030).
- **per-bar tree (Optuna, TUNE_PER=25, best-of-50)** (18): QLIKE 0.0999 .. 0.1097, Sharpe mid 0.60 .. 1.65; best QLIKE Optuna per-bar random forest, TUNE_PER 25, best of 50, QLIKE-selected [live_feasible] (0.0999, Sharpe 1.23); best Sharpe Optuna per-bar XGBoost, TUNE_PER 25, best of 50, QLIKE-selected [live_feasible] (1.65, QLIKE 0.1022).
- **per-bar tree (Optuna, TUNE_PER=250, best-of-50)** (9): QLIKE 0.1024 .. 0.1184, Sharpe mid 0.75 .. 1.64; best QLIKE Optuna per-bar XGBoost, TUNE_PER 250, best of 50 [live_feasible] (0.1024, Sharpe 1.64); best Sharpe Optuna per-bar XGBoost, TUNE_PER 250, best of 50 [live_feasible] (1.64, QLIKE 0.1024).
- **per-bar tree (Optuna, TUNE_PER=1, best-of-25)** (9): QLIKE 0.1007 .. 0.1086, Sharpe mid 0.69 .. 1.71; best QLIKE Optuna per-bar random forest, TUNE_PER 1, best of 25 [live_feasible] (0.1007, Sharpe 1.67); best Sharpe Optuna per-bar LightGBM, TUNE_PER 1, best of 25 [all_features] (1.71, QLIKE 0.1046).
- **per-bar tree (Optuna, TUNE_PER=25, best-of-25)** (9): QLIKE 0.1030 .. 0.1076, Sharpe mid 0.67 .. 1.49; best QLIKE Optuna per-bar XGBoost, TUNE_PER 25, best of 25 [live_feasible] (0.1030, Sharpe 1.12); best Sharpe Optuna per-bar random forest, TUNE_PER 25, best of 25 [all_features] (1.49, QLIKE 0.1049).
- **per-bar tree (Optuna, TUNE_PER=1, best-of-10)** (9): QLIKE 0.1001 .. 0.1089, Sharpe mid 0.73 .. 1.66; best QLIKE Optuna per-bar random forest, TUNE_PER 1, best of 10 [live_feasible] (0.1001, Sharpe 1.66); best Sharpe Optuna per-bar random forest, TUNE_PER 1, best of 10 [live_feasible] (1.66, QLIKE 0.1001).
- **per-bar tree (Optuna, TUNE_PER=25, best-of-10)** (9): QLIKE 0.1003 .. 0.1097, Sharpe mid 0.67 .. 1.67; best QLIKE Optuna per-bar random forest, TUNE_PER 25, best of 10 [live_feasible] (0.1003, Sharpe 1.41); best Sharpe Optuna per-bar XGBoost, TUNE_PER 25, best of 10 [all_features] (1.67, QLIKE 0.1030).
- **LSTM** (6): QLIKE 0.1031 .. 0.1305, Sharpe mid 0.77 .. 1.33; best QLIKE per-bar LSTM, QLIKE-selected [baseline] (0.1031, Sharpe 0.77); best Sharpe per-bar LSTM [all_features] (1.33, QLIKE 0.1305).
- **LSTM (intraday sequence)** (6): QLIKE 0.1071 .. 0.1125, Sharpe mid 0.81 .. 1.71; best QLIKE intraday-sequence LSTM, QLIKE-selected [baseline] (0.1071, Sharpe 0.87); best Sharpe intraday-sequence LSTM, QLIKE-selected [live_feasible] (1.71, QLIKE 0.1080).
- **implied-vol representations** (15): QLIKE 0.0950 .. 0.1001, Sharpe mid 0.97 .. 1.87; best QLIKE per-bar elastic net [ivrep_innovations] (0.0950, Sharpe 1.61); best Sharpe per-bar lasso [ivrep_target_scale] (1.87, QLIKE 0.0960).
- **HAR-ladder variants** (9): QLIKE 0.0934 .. 0.2184, Sharpe mid -0.02 .. 1.79; best QLIKE per-bar elastic net [live_feasible], HAR ladder base3 (0.0934, Sharpe 1.57); best Sharpe per-bar lasso [live_feasible], HAR ladder base2 (1.79, QLIKE 0.0948).
- Over all of table A the lowest QLIKE is per-bar elastic net [live_feasible], HAR ladder base3 (0.0934; Sharpe 1.57, rank 53 of 220 on Sharpe) and the highest Sharpe is per-bar lasso [live_vix_only] (1.91; QLIKE 0.0969, rank 26 of 220 on QLIKE).
- Against the reference's QLIKE, 51 forecasts have a day-block interval wholly below zero (better) and 10 wholly above (worse). The raw (plain back-transform) QLIKE ranks forecasts differently from the recalibrated one (Spearman across table A +0.74); the recalibration's term s lowers the loss most for per-bar ridge [free_feasible] 0.1407 raw vs 0.1016 recalibrated; per-bar ridge [live_feasible], HAR ladder base3 0.1237 raw vs 0.0998 recalibrated; per-bar ridge [free_feasible_vol] 0.1198 raw vs 0.1001 recalibrated.

Rank correlation across table-A forecasts (check rows and exact duplicates excluded), point and 95 % day-block bootstrap interval (the same resampled days as every other interval; a negative value = lower QLIKE goes with higher Sharpe):

| set | forecasts | QLIKE measure | Sharpe | Spearman | 95 % interval |
|---|---:|---|---|---:|---|
| all table-A forecasts | 220 | recal | mid | -0.56 | [-0.65, -0.07] |
| all table-A forecasts | 220 | recal | crossed | -0.56 | [-0.65, -0.07] |
| all table-A forecasts | 220 | raw | mid | -0.34 | [-0.59, +0.04] |
| all table-A forecasts | 220 | raw | crossed | -0.34 | [-0.59, +0.04] |
| per-bar forecasts (linear, VIX-only, trees, tuned, LSTM, bucket studies) | 202 | recal | mid | -0.57 | [-0.67, -0.06] |
| per-bar forecasts (linear, VIX-only, trees, tuned, LSTM, bucket studies) | 202 | recal | crossed | -0.57 | [-0.67, -0.06] |
| per-bar forecasts (linear, VIX-only, trees, tuned, LSTM, bucket studies) | 202 | raw | mid | -0.35 | [-0.63, +0.07] |
| per-bar forecasts (linear, VIX-only, trees, tuned, LSTM, bucket studies) | 202 | raw | crossed | -0.35 | [-0.63, +0.07] |
| per-bar linear + VIX-only family | 31 | recal | mid | +0.20 | [-0.70, +0.64] |
| per-bar linear + VIX-only family | 31 | recal | crossed | +0.21 | [-0.70, +0.64] |
| per-bar linear + VIX-only family | 31 | raw | mid | +0.28 | [-0.70, +0.72] |
| per-bar linear + VIX-only family | 31 | raw | crossed | +0.28 | [-0.70, +0.72] |
| per-bar nonlinear (every tree rung, LSTM) | 147 | recal | mid | -0.34 | [-0.59, +0.12] |
| per-bar nonlinear (every tree rung, LSTM) | 147 | recal | crossed | -0.33 | [-0.58, +0.12] |
| per-bar nonlinear (every tree rung, LSTM) | 147 | raw | mid | -0.33 | [-0.58, +0.15] |
| per-bar nonlinear (every tree rung, LSTM) | 147 | raw | crossed | -0.33 | [-0.58, +0.15] |
| 48-bar forecasts (paper + pooled twins) | 18 | recal | mid | -0.37 | [-0.69, +0.47] |
| 48-bar forecasts (paper + pooled twins) | 18 | recal | crossed | -0.34 | [-0.69, +0.47] |
| 48-bar forecasts (paper + pooled twins) | 18 | raw | mid | +0.02 | [-0.66, +0.63] |
| 48-bar forecasts (paper + pooled twins) | 18 | raw | crossed | +0.02 | [-0.66, +0.63] |

Reading: across all of table A a lower QLIKE goes with a higher Sharpe (-0.56 [-0.65, -0.07]). Part of it is the split between the 18 forecasts fitted on all 48 bars (median QLIKE 0.1039, median Sharpe 1.20) and the 202 per-bar forecasts (median QLIKE 0.1032, median Sharpe 1.33); among the per-bar forecasts alone it is -0.57 [-0.67, -0.06], carried by the 26 per-bar forecasts in the worst quarter on both counts (per-bar ridge [live_feasible], HAR ladder intra; per-bar lasso [live_feasible], HAR ladder intra; per-bar elastic net [live_feasible], HAR ladder intra; per-bar LSTM [live_feasible]; Optuna per-bar LightGBM, TUNE_PER 250, best of 50 [baseline]; tuned per-bar XGBoost [baseline], QLIKE-selected; ...); within the per-bar linear and VIX-only family (QLIKE 0.0954 .. 0.1016, Sharpe 1.22 .. 1.91) it is +0.20 [-0.70, +0.64]: among forecasts of similar accuracy, QLIKE does not order the trade.

## (c) Extreme back-transform values

- **Rule (named):** PLAIN_FLOOR(d) = the smallest 16:00 target of the previous 250 sessions (the recalibration window SMEAR_W), lagged one session. A plain back-transform f²·B below it forecasts a quieter 15:30–16:00 half hour than any of the past year; it is flagged, listed in `master_table_extremes.csv`, and the raw QLIKE is reported as is (`qlike_raw`) and with flagged forecasts raised to the floor (`qlike_raw_floored`). Nothing is clipped silently; the recalibrated forecast (≥ s·B) needs no treatment and is never below the floor on a trade day in this run (max count 0).
- **On the trade days:** 0 flagged forecasts over all forecasts, so `qlike_raw_floored` = `qlike_raw` for 250 of 250 rows.
- **On every 16:00 session of the arm span** (1406 sessions, the arm span 2018-06-25 .. 2024-04-30 after the floor's 63-session warm-up, early closes included): 380 flagged rows over 132 forecasts, 380 of them on 8 early-close sessions (2019-07-03, 2019-11-29, 2019-12-24, 2020-11-27, 2020-12-24, 2022-11-25, 2023-07-03, 2023-11-24): 13:00 closes, where the 15:30–16:00 bar lies after the cash close and the trade frame never enters. Negative adjusted-scale forecasts occur there too (the plain back-transform squares them).
  - per-bar ridge [all_features] on 2023-11-24 (early close): f = 0.0047 on the adjusted scale, plain forecast 5.51e-11 vs floor 2.14e-07 and target 1.94e-07; that one day's raw QLIKE is 3516.8. Over all 1406 sessions the raw QLIKE is 2.6419 as is, 0.1327 floored, 0.1293 without early closes; on the trade days 0.1164 (recalibrated 0.1004).
  - per-bar LSTM [live_feasible] on 2020-11-27 (early close): f = -0.0079 on the adjusted scale, plain forecast 7.41e-10 vs floor 1.22e-07 and target 4.9e-07; that one day's raw QLIKE is 654.8. Over all 1406 sessions the raw QLIKE is 0.6085 as is, 0.1427 floored, 0.1378 without early closes; on the trade days 0.1199 (recalibrated 0.1193).
  - per-bar ridge [live_feasible_plus_ivslice], 500 sessions on 2023-07-03 (early close): f = -0.0185 on the adjusted scale, plain forecast 1.05e-09 vs floor 2.14e-07 and target 4.56e-07; that one day's raw QLIKE is 427.2. Over all 1406 sessions the raw QLIKE is 0.4410 as is, 0.1330 floored, 0.1313 without early closes; on the trade days 0.0860 (recalibrated 0.0833).
- The handoff's '16:00 ridge QLIKE 2.65 unrecalibrated' is this same early-close day: `score_rest_of_day.csv` reports the plain QLIKE 2.6480 for the all_features ridge at the 15:30 clock on 1406 sessions against the unclipped variance; the recalibration (its `qlike_causal` 0.1248) and the trade frame's exclusion of early closes both remove it.

## Gates

2207 gates checked, 0 failed (`master_table_gates.csv`, column `asserted`). Among them: every arm file's 16:00 target equals the common target and the common target of `linear_subsection_dedup` equals `linear_subsection`'s bit for bit; every table's baseline equals its B, its rv_raw the production table's, and its 16:00 stamps the target's; every arm file's recalibrated forecast equals the research reader's on the trade days; every per-bar linear and VIX-only table equals its arm file; every row's trade aggregates equal `trade_1530`'s; no table falls into 'other table'; the paper's always-short row is reproduced; the stored research numbers of the design in use are reproduced (16:00 campaign: 1013; paper: 2; pre-campaign master table: 18).

Reported, not asserted (192): stored numbers written on another design, compared for the record — pre-campaign design: 166 of 192 agree to 1e-09. A difference there is the design change (see the before/after), not a failure.

## Top of table A by sign(s) Sharpe (mid)

| forecast | QLIKE raw | QLIKE recal | Δ% vs ref | DM | Sharpe mid | Sharpe crossed | ΔSharpe mid vs ref [95 %] | ΔSharpe mid vs short [95 %] | buy days |
|---|---:|---:|---:|---:|---:|---:|---|---|---:|
| per-bar lasso [live_vix_only] | 0.0937 | 0.0969 | -8.4 | -2.83 | 1.91 | 1.44 | +0.91 [-0.05, +1.90] | +1.71 [+0.30, +3.02] | 327 |
| per-bar ridge [live_feasible] | 0.1184 | 0.1005 | -5.0 | -1.42 | 1.90 | 1.44 | +0.90 [+0.19, +1.66] | +1.70 [+0.09, +3.11] | 348 |
| tuned per-bar random forest [live_feasible], QLIKE-selected | 0.0993 | 0.1040 | -1.8 | -0.55 | 1.90 | 1.44 | +0.90 [+0.14, +1.71] | +1.70 [+0.29, +2.95] | 349 |
| per-bar elastic net [live_vix_only] | 0.0952 | 0.0968 | -8.6 | -3.05 | 1.90 | 1.43 | +0.90 [+0.12, +1.77] | +1.69 [+0.30, +2.98] | 332 |
| per-bar lasso [free_vix_only] | 0.0940 | 0.0973 | -8.0 | -2.72 | 1.88 | 1.42 | +0.88 [-0.06, +1.86] | +1.68 [+0.26, +3.02] | 330 |
| per-bar lasso [ivrep_target_scale] | 0.0932 | 0.0960 | -9.3 | -2.81 | 1.87 | 1.41 | +0.87 [+0.06, +1.68] | +1.67 [+0.25, +2.97] | 343 |
| per-bar ridge [live_vix_only] | 0.1179 | 0.1005 | -5.0 | -1.41 | 1.86 | 1.39 | +0.86 [+0.06, +1.66] | +1.66 [+0.04, +3.13] | 351 |
| per-bar lasso [free_feasible_vol] | 0.0930 | 0.0963 | -9.0 | -3.07 | 1.85 | 1.39 | +0.85 [-0.17, +1.88] | +1.65 [+0.20, +2.93] | 336 |
| per-bar LightGBM [all_features] | 0.0962 | 0.1007 | -4.9 | -1.31 | 1.84 | 1.37 | +0.84 [+0.03, +1.66] | +1.63 [+0.31, +2.87] | 332 |
| per-bar elastic net [live_vix_rvol] | 0.0942 | 0.0958 | -9.4 | -3.42 | 1.83 | 1.36 | +0.83 [+0.15, +1.61] | +1.63 [+0.16, +2.97] | 329 |
| per-bar ridge [free_vix_only] | 0.1194 | 0.0999 | -5.5 | -1.62 | 1.83 | 1.36 | +0.83 [-0.00, +1.66] | +1.62 [-0.04, +3.11] | 348 |
| per-bar lasso [live_feasible] | 0.0931 | 0.0964 | -8.8 | -3.04 | 1.83 | 1.36 | +0.82 [-0.18, +1.86] | +1.62 [+0.20, +2.91] | 337 |

Wording: sign(s) = buy the straddle when the recalibrated 16:00 forecast exceeds the 15:30 implied variance, sell otherwise; straddle = nearest out-of-the-money call + nearest out-of-the-money put, same-day expiry, one position; the last-30-min trade enters at 15:30 and settles at the close. Buckets: `baseline` = HAR + calendar (`har_ma_*`, calendar dummies), `all_features` = the full design, `live_feasible` = the 16 series a 15:30 forecaster rebuilds live (ES return moments and liquidity, VIX/VVIX/VIX3M (`adj_vix_ma_*` …), FOMC calendar), `vix_only` = HAR + calendar + VIX level, `live_vix_only` = live_feasible minus VVIX and VIX3M, `free_vix_only` = the free feed (ES 1-min bars + VIX + FOMC calendar). Tree rungs: T10 / T1 = the shipped configuration refit every 10 sessions / every session; RS10 / RS1 = random search (32 candidates, every 250 sessions) refit every 10 / every session; Optuna = TPE, 50 trials per tuning point, tuning every TUNE_PER sessions, best of the first k trials, refit every session.
