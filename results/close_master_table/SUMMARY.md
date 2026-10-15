# Master table for the closing strategy (A4) — summary

Written by `experiments/master_table_close.py --arm-root linear_subsection_dedup` on 2026-10-14 00:00; every number below is read from
`master_table.csv` (and `before_after.csv`, `recal_vs_plain.csv`) of the same run. Full table: `writeup/master_table_close.pdf`; tex: `writeup/generated/table_master_close.tex`. Every forecast is the plain back-transform f²·B; every number is a point estimate.

## Provenance: design, window mask, CPU class

- **Arm root:** `results/linear_subsection_dedup/` (de-duplicated design (16:00 campaign)); it supplies the arm-only rows, the per-bar OLS incumbent, the arm files the per-bar linear and VIX-only tables are gated against, and the common 16:00 target; its 16:00 target equals the default root's bit for bit (1469 rows, 0 differ; gate passed).
- **Forecast tables:** every `results/spxw_pnl/yhat_*.parquet` on disk at run time (glob). After the 16:00 campaign (`writeup/CAMPAIGN_16H_2026-09-29.md`) the per-bar tables are the de-duplicated design (the 12 HAR x open/close session-edge columns dropped from the per-bar design; the pooled 48-bar models keep them); every tree and LSTM fit uses the per-window mask (`src/models/window_mask.py`: drop columns constant on the fit's training window and exact copies of an earlier kept column) and ran pinned to one CPU class (CARC epyc-7513), because tree and LSTM numbers depend on the CPU class (agent D's finding).

| family | forecasts (A / B) | design | window mask | refit | tuning | CPU class | vs the pre-campaign table |
|---|---:|---|---|---|---|---|---|
| paper | 9 / 0 | pooled 48-bar models (session-edge interactions kept, campaign decision 1) | n/a | as the paper | as the paper | not re-run | nothing |
| per-bar linear | 22 / 0 | de-duplicated per-bar design (commit 47f7f9c: the 12 HAR x open/close session-edge columns dropped) | identifiability mask at every penalty tune (unchanged: it had already removed the 12 columns) | every session | penalty every 250 sessions | Hoffman2 (agent A) | design de-dup (the mask had already dropped the session-edge columns: float path only) |
| pooled twin | 21 / 0 | pooled 48-bar twins (session-edge interactions kept) | identifiability mask | every session | penalty every 250 sessions | not re-run | nothing |
| VIX-only family | 21 / 0 | de-duplicated per-bar design (commit 47f7f9c: the 12 HAR x open/close session-edge columns dropped) | identifiability mask at every penalty tune (unchanged: it had already removed the 12 columns) | every session | penalty every 250 sessions | Hoffman2 (agent A) | design de-dup (the mask had already dropped the session-edge columns: float path only) |
| per-bar tree (untuned) | 15 / 0 | de-duplicated per-bar design | per-window mask (src/models/window_mask.py, WINDOW_MASK=1) | every 10 sessions (T10) | none (shipped configuration) | CARC, pinned to one CPU class (epyc-7513) | design de-dup + per-window mask + CPU-class pinning (refit cadence unchanged, every 10) |
| per-bar tree (untuned, daily refit) | 15 / 0 | de-duplicated per-bar design | per-window mask (src/models/window_mask.py, WINDOW_MASK=1) | every session (T1) | none (shipped configuration) | CARC, pinned to one CPU class (epyc-7513) | new |
| per-bar tree (tuned) | 24 / 0 | de-duplicated per-bar design | per-window mask (src/models/window_mask.py, WINDOW_MASK=1) | every 10 sessions (RS10) | random search, 32 candidates, every 250 sessions (MSE rule; tunedq = QLIKE rule) | CARC, pinned to one CPU class (epyc-7513) | design de-dup + per-window mask + CPU-class pinning (refit / tuning cadence unchanged) |
| per-bar tree (random search, daily) | 24 / 0 | de-duplicated per-bar design | per-window mask (src/models/window_mask.py, WINDOW_MASK=1) | every session (RS1) | random search, 32 candidates, every 250 sessions (MSE rule; tunedq = QLIKE rule) | CARC, pinned to one CPU class (epyc-7513) | new |
| per-bar tree (Optuna, TUNE_PER=1, best-of-50) | 12 / 0 | de-duplicated per-bar design | per-window mask (src/models/window_mask.py, WINDOW_MASK=1) | every session | Optuna TPE, 50 trials, best of the first 50, every 1 session(s) | CARC, single-class re-run on epyc-7513 (agent C; canonical) | new |
| per-bar tree (Optuna, TUNE_PER=5, best-of-50) | 12 / 0 | de-duplicated per-bar design | per-window mask (src/models/window_mask.py, WINDOW_MASK=1) | every session | Optuna TPE, 50 trials, best of the first 50, every 5 session(s) | CARC, single-class re-run on epyc-7513 (agent C; canonical) | new |
| per-bar tree (Optuna, TUNE_PER=25, best-of-50) | 24 / 0 | de-duplicated per-bar design | per-window mask (src/models/window_mask.py, WINDOW_MASK=1) | every session | Optuna TPE, 50 trials, best of the first 50, every 25 session(s) (MSE rule; optunaq = QLIKE rule) | CARC, single-class re-run on epyc-7513 (agent C; canonical) | new |
| per-bar tree (Optuna, TUNE_PER=250, best-of-50) | 12 / 0 | de-duplicated per-bar design | per-window mask (src/models/window_mask.py, WINDOW_MASK=1) | every session | Optuna TPE, 50 trials, best of the first 50, every 250 session(s) | CARC, single-class re-run on epyc-7513 (agent C; canonical) | new |
| per-bar tree (Optuna, TUNE_PER=1, best-of-25) | 12 / 0 | de-duplicated per-bar design | per-window mask (src/models/window_mask.py, WINDOW_MASK=1) | every session | Optuna TPE, 50 trials, best of the first 25, every 1 session(s) | CARC, single-class re-run on epyc-7513 (agent C; canonical) | new |
| per-bar tree (Optuna, TUNE_PER=25, best-of-25) | 12 / 0 | de-duplicated per-bar design | per-window mask (src/models/window_mask.py, WINDOW_MASK=1) | every session | Optuna TPE, 50 trials, best of the first 25, every 25 session(s) | CARC, single-class re-run on epyc-7513 (agent C; canonical) | new |
| per-bar tree (Optuna, TUNE_PER=1, best-of-10) | 12 / 0 | de-duplicated per-bar design | per-window mask (src/models/window_mask.py, WINDOW_MASK=1) | every session | Optuna TPE, 50 trials, best of the first 10, every 1 session(s) | CARC, single-class re-run on epyc-7513 (agent C; canonical) | new |
| per-bar tree (Optuna, TUNE_PER=25, best-of-10) | 12 / 0 | de-duplicated per-bar design | per-window mask (src/models/window_mask.py, WINDOW_MASK=1) | every session | Optuna TPE, 50 trials, best of the first 10, every 25 session(s) | CARC, single-class re-run on epyc-7513 (agent C; canonical) | new |
| LSTM | 6 / 0 | de-duplicated per-bar design | per-window mask, recomputed at every refit | every session (was every 10) | 16 configurations x 5 seeds, every 250 sessions (MSE rule; qsel = QLIKE rule) | CARC, pinned to one CPU class (epyc-7513) | design de-dup + per-window mask + refit cadence 10 -> 1 + CPU-class pinning |
| LSTM (intraday sequence) | 6 / 0 | last N half-hour bars to 15:30 (N tuned in {13, 48, 96}), bar-level inputs | per-window mask on the step columns | every session | every 250 sessions (MSE rule; qsel = QLIKE rule) | CARC, pinned to epyc-7513 (agent I) | new |
| direct rest-of-day at 15:30 (check) | 9 / 0 | de-duplicated per-bar design, 15:30 clock of the rest-of-day model | identifiability mask | every session | penalty every 250 sessions | Hoffman2, each 15:30 arm pinned to its per-bar twin's CPU architecture (agent E) | design de-dup + CPU-architecture pinning to the per-bar twin |
| implied-vol representations | 15 / 0 | de-duplicated per-bar design (commit 47f7f9c: the 12 HAR x open/close session-edge columns dropped) | identifiability mask at every penalty tune (unchanged: it had already removed the 12 columns) | every session | penalty every 250 sessions | Hoffman2 (agent A) | design de-dup (the mask had already dropped the session-edge columns: float path only) |
| HAR-ladder variants | 9 / 0 | de-duplicated per-bar design (commit 47f7f9c: the 12 HAR x open/close session-edge columns dropped) | identifiability mask at every penalty tune (unchanged: it had already removed the 12 columns) | every session | penalty every 250 sessions | Hoffman2 (agent A) | design de-dup (the mask had already dropped the session-edge columns: float path only) |
| chain-period buckets | 0 / 21 | de-duplicated per-bar design (commit 47f7f9c: the 12 HAR x open/close session-edge columns dropped) | identifiability mask at every penalty tune (unchanged: it had already removed the 12 columns) | every session | penalty every 250 sessions | Hoffman2 (agent A) | design de-dup (the mask had already dropped the session-edge columns: float path only) |

## Scorer, days, reference

- **Scorer: the plain back-transform.** Every forecast is used as f²·B (f = the model's forecast on the adjusted scale sqrt(RV/B), B = the diurnal baseline; the `yhat`² · `baseline` of `results/spxw_pnl/yhat_*.parquet`); QLIKE (`qlike_raw`) against the per-bar spec's 16:00 target; the trade is the deck's 15:30 sign(s) on the straddle (`trade_1530`): buy when f²·B exceeds the 15:30 implied variance, sell otherwise. The research scorer's 16:00-bar recalibration (f² + s)·B (s = the forecast's own trailing 250-session mean squared adjusted-scale error, lagged one) was removed on 2026-10-06 (user decision); it survives only as the diagnostic column `pred_clock` and in `recal_vs_plain.csv` (section below). The notebook's 13-bar Mincer–Zarnowitz map is **not** used either: no number here may be set beside a notebook number.
- **Point estimates only.** The circular block bootstrap is disabled (commit b761b28: every draw is the original sample), so no interval is reported. The inference shown is the Diebold–Mariano statistic on the daily QLIKE difference (`dm_test`, Newey–West variance with the Harvey–Leybourne–Newbold correction) and the Newey–West HAC t of the daily mid-return difference (`newey_west_t`); counts 'significant' use the two-sided 5 % level of those statistics.
- **Days:** table A = 295 forecasts in 20 families (+ 9 check rows) on the same 866 trade days (2020-01-03 .. 2024-04-30); intersection over table A = 866 of 866 trade days. Every family that covers the 866 trade days is in table A. Table B = 21 forecasts scored on their own days (paired with the reference on those days): 21 chain-period buckets forecasts from 2022-01-03 on 550 trade days. Forecasts that failed to load: none.
- **Which forecasts drop days:** no table-A forecast misses a trade day; 0 forecasts miss trade days inside their own window. The 21 table-B forecasts start at their comparison window (2022-01-03), so each omits the 316 earlier trade days by design (their implied-vol inputs are real chain data only from then; compare_mfiv_harlag.CHAIN_START). Coverage per forecast: `master_table_days.csv`.
- **Reference:** the block-diagonal ridge (`blk2`, `yhat_blk2_fomc1.parquet`) — the paper's headline forecast (HAR ladder, penalty 1, plus the exogenous block, penalty 100, on the panel of record), the forecast the paper's 15:30 deck trades. Under this scorer it has QLIKE 0.1068 and sign(s) Sharpe 1.28 mid / 0.80 crossed (the deck's own map gives it 1.34 mid; the difference is the scorer). Second reference: always short, Sharpe 0.20 mid / -0.27 crossed.

## (a) Recommended headline forecast

**per-bar ridge [live_feasible]** (`sub_ridge_live_feasible`), scored with the plain forecast. On the 866 days: QLIKE 0.1184 (+10.9 % vs the reference, DM +0.60, p 0.549); sign(s) Sharpe 1.79 mid / 1.31 crossed, mean +0.126 per unit premium (crossed +0.093), hit rate 57.0 %, buys on 269 days (31.1 %). Sharpe difference vs the reference: mid +0.51, crossed +0.51 (HAC t +1.12, p 0.262 on the daily mid-return difference); vs always short: mid +1.59, crossed +1.58 (HAC t +2.26, p 0.024).

- **Does any forecast now beat the headline?** Of the 294 other table-A forecasts (check rows and exact duplicates left out), 30 have a higher Sharpe point estimate at the mid fill (31 crossed); with the HAC t of the daily mid-return difference significant at 5 %: 2 (per-bar ridge [live_vix_only] (+0.73, HAC t +2.46, p 0.014); per-bar ridge [free_vix_only] (+0.70, HAC t +2.45, p 0.014)). 282 have a lower QLIKE point estimate; with DM p < 0.05: 0.
  Highest Sharpe differences over the headline (mid, point estimates): per-bar ridge [live_vix_only] +0.73 (t +2.46, p 0.014); per-bar ridge [free_vix_only] +0.70 (t +2.45, p 0.014); Optuna per-bar XGBoost, TUNE_PER 25, best of 50, QLIKE-selected [live_feasible] +0.40 (t +1.05, p 0.292); per-bar ridge [lean_vol_vix] +0.31 (t +0.83, p 0.407); per-bar ridge [free_feasible] +0.28 (t +1.14, p 0.253).
  RECORDED, not acted on: the headline constant stays `sub_ridge_live_feasible` (it is chosen on feasibility and as the per-bar model of record, not on the maximum of many Sharpe ratios); the forecasts above are the candidates to look at.

Why this one:
- It is a 15:30-specific model on the inputs a 15:30 forecaster can rebuild live (`live_feasible`), so the trade it scores can be run. It ranks 31 of 295 table-A forecasts on sign(s) Sharpe (mid) and 283 of 295 on QLIKE (plain forecast).
- The highest Sharpe in table A is per-bar ridge [live_vix_only] (2.53 mid / 2.05 crossed). Choosing the maximum of 295 Sharpe ratios would select on the trade's own noise; the headline is chosen on feasibility and on being the per-bar model of record (the per-bar ridge is the estimator the 15:30 study defined first), not on the maximum.
- Paired against the headline itself on the same days (`master_table_vs_headline.csv`; a negative QLIKE % / DM = the alternative forecasts better, a positive Sharpe difference = the alternative trades better; point estimates, HAC t of the daily mid-return difference):

  | alternative | QLIKE | % vs headline | DM | Sharpe mid / crossed | ΔSharpe mid vs headline | ΔSharpe crossed vs headline | HAC t (mid) | same position |
  |---|---:|---:|---:|---|---:|---:|---:|---:|
  | per-bar lasso [live_feasible] | 0.0931 | -21.4 | -1.28 | 1.87 / 1.39 | +0.08 | +0.08 | +0.19 | 90.2 % |
  | per-bar elastic net [live_feasible] | 0.0950 | -19.7 | -1.26 | 1.56 / 1.08 | -0.23 | -0.22 | -0.55 | 91.7 % |
  | per-bar ridge [all_features] | 0.1164 | -1.7 | -0.25 | 1.71 / 1.22 | -0.08 | -0.08 | -0.18 | 87.4 % |
  | per-bar ridge [baseline] | 0.0946 | -20.1 | -1.16 | 1.36 / 0.88 | -0.43 | -0.43 | -0.77 | 83.9 % |
  | per-bar lasso [free_vix_only] | 0.0940 | -20.6 | -1.24 | 1.91 / 1.43 | +0.12 | +0.12 | +0.29 | 89.8 % |
  | per-bar ridge [free_vix_only] | 0.1194 | +0.9 | +0.27 | 2.49 / 2.02 | +0.70 | +0.71 | +2.45 | 96.5 % |
  | per-bar LightGBM [live_feasible] | 0.0939 | -20.7 | -1.26 | 1.54 / 1.05 | -0.25 | -0.25 | -0.60 | 86.3 % |
  | per-bar XGBoost [live_feasible] | 0.0943 | -20.4 | -1.23 | 1.78 / 1.29 | -0.01 | -0.01 | -0.03 | 87.5 % |
  | per-bar XGBoost [lean_vol_vix] | 0.0923 | -22.1 | -1.36 | 1.78 / 1.30 | -0.01 | -0.01 | -0.02 | 87.5 % |
  | per-bar XGBoost, daily refit [lean_vol_vix] | 0.0911 | -23.0 | -1.39 | 1.76 / 1.28 | -0.03 | -0.03 | -0.07 | 87.2 % |
  | tuned per-bar XGBoost [live_feasible], MSE-selected | 0.1003 | -15.3 | -1.00 | 1.96 / 1.48 | +0.17 | +0.17 | +0.36 | 85.3 % |
  | tuned per-bar LightGBM [live_feasible], MSE-selected | 0.0932 | -21.3 | -1.28 | 1.76 / 1.28 | -0.03 | -0.03 | -0.06 | 86.8 % |
  | tuned per-bar XGBoost, daily refit [live_feasible], QLIKE-selected | 0.0986 | -16.7 | -1.04 | 1.99 / 1.51 | +0.20 | +0.21 | +0.43 | 85.8 % |
  | tuned per-bar random forest, daily refit [lean_vol_vix], QLIKE-selected | 0.0926 | -21.8 | -1.28 | 1.47 / 0.98 | -0.32 | -0.33 | -0.59 | 88.9 % |
  | Optuna per-bar XGBoost, TUNE_PER 25, best of 50, QLIKE-selected [live_feasible] | 0.0970 | -18.1 | -1.07 | 2.19 / 1.70 | +0.40 | +0.39 | +1.05 | 85.7 % |
  | Optuna per-bar random forest, TUNE_PER 250, best of 50 [lean_vol_vix] | 0.0930 | -21.5 | -1.26 | 1.29 / 0.80 | -0.50 | -0.51 | -1.01 | 89.0 % |
  | per-bar LSTM [baseline] | 0.1009 | -14.8 | -0.86 | 1.37 / 0.88 | -0.42 | -0.42 | -0.81 | 80.6 % |
  | per-bar LSTM, QLIKE-selected [baseline] | 0.0992 | -16.2 | -0.95 | 1.13 / 0.65 | -0.66 | -0.66 | -1.26 | 83.7 % |
  | intraday-sequence LSTM [all_features] | 0.1117 | -5.6 | -0.38 | 1.55 / 1.06 | -0.24 | -0.24 | -0.49 | 80.4 % |
  | intraday-sequence LSTM, QLIKE-selected [live_feasible] | 0.1070 | -9.6 | -0.63 | 1.46 / 0.97 | -0.33 | -0.34 | -0.77 | 82.1 % |

- Over all 294 other table-A forecasts (point estimates): 30 trade better than the headline at the mid fill (31 crossed), 2 of them with the HAC t of the daily mid-return difference significant at 5 %; 264 trade worse (30 significant); 282 have a lower QLIKE (0 with DM p < 0.05), 12 a higher one (3 with DM p < 0.05).
- Against the reference, 148 of 294 table-A forecasts have a higher mid-fill Sharpe (147 at the crossed fill); 4 of them with the HAC t of the daily mid-return difference significant at 5 %. Against always short, 293 (mid) and 293 (crossed) of 295 have a higher Sharpe; 109 with a significant HAC t (mid).
- On forecast accuracy alone the best live-feasible forecast is per-bar LightGBM, daily refit [live_feasible] (QLIKE 0.0911, Sharpe 1.61 / 1.12); a QLIKE-first choice would take it. The trade does not rank forecasts the way QLIKE does (section b), which is why the headline names the scorer AND the trade numbers.

## Scorer change: recalibrated -> plain (2026-10-06)

The recalibration was removed on 2026-10-06 (user decision): every number in this file uses the plain forecast f²·B. `recal_vs_plain.csv` records, per forecast on its own scored days, the parked recalibrated forecast (`pred_clock` = (f² + s)·B, diagnostic) against the plain one. Over the 286 table-A forecasts (check rows and exact duplicates left out, 866 days):
- QLIKE, plain minus recalibrated: median -4.8 (-8.0 .. +38.4) %; Sharpe mid median -0.04 (-0.99 .. +0.71), crossed median -0.06 (-1.01 .. +0.70); buy share median -9.6 (-12.5 .. -6.9) percentage points.
- Positions changed: median 83 (60 .. 108) of 866 days; buy -> sell 23527, sell -> buy 0 over all of them (s·B ≥ 0, so removing it can only lower a forecast: mean s·B / f²·B median 0.08 (0.08 .. 0.18)).
- **Headline** (`sub_ridge_live_feasible`): QLIKE 0.1005 -> 0.1184; Sharpe mid 1.90 -> 1.79, crossed 1.44 -> 1.31; buys 40.2 % -> 31.1 %; positions changed on 79 days.
- **Reference** (`blk2`): QLIKE 0.1058 -> 0.1068; Sharpe mid 1.00 -> 1.28, crossed 0.53 -> 0.80; buys 36.0 % -> 27.0 %; positions changed on 78 days.
- Largest Sharpe (mid) falls: Optuna per-bar random forest, TUNE_PER 1, best of 25 [all_features] 1.56 -> 0.57; per-bar lasso [vix_rvol] 1.63 -> 0.79; per-bar XGBoost, daily refit [all_features] 1.87 -> 1.12. Largest rises: per-bar elastic net [all_features] 1.33 -> 2.04; per-bar ridge [free_vix_only] 1.83 -> 2.49; per-bar ridge [live_vix_only] 1.86 -> 2.53.
- Rank agreement across these forecasts, recalibrated vs plain: Spearman +0.80 on QLIKE, +0.67 on Sharpe mid.

## Before / after the 16:00 campaign (both sides plain)

Before = the pre-campaign master table (commit 6177b74; snapshot `results/spxw_pnl/pre_dedup_2026-09-29/`), re-scored here with the plain forecast from its daily frame's `plain` column; after = this run; every key present in both is paired on the same days (`before_after.csv`; difference = after minus before; point estimates, HAC t of the daily mid-return difference).
- **Headline** (`sub_ridge_live_feasible`; design de-dup (the mask had already dropped the session-edge columns: float path only); re-run on hoffman2 (was carc)): QLIKE 0.1184 -> 0.1184 (+0.00 %); Sharpe mid 1.79 -> 1.79 (+0.00, HAC t n/a), crossed 1.31 -> 1.31 (+0.00); positions changed on 0 of 866 days; largest relative change of the plain forecast 1.13e-12.
- **Reference** (`blk2`; nothing): QLIKE 0.1068 -> 0.1068 (+0.00 %); Sharpe mid 1.28 -> 1.28 (+0.00, HAC t n/a), crossed 0.80 -> 0.80 (+0.00); positions changed on 0 of 866 days; largest relative change of the plain forecast 0.00e+00.

| family | keys in both | what changed | QLIKE % change, median (min .. max) | ΔSharpe mid, median (min .. max) | ΔSharpe crossed, median (min .. max) | HAC t significant, up / down | positions changed, median (max) |
|---|---:|---|---|---|---|---|---|
| paper | 9 | nothing; nothing (the campaign did not re-run it | +0.00 (+0.00 .. +0.00) | +0.00 (+0.00 .. +0.00) | +0.00 (+0.00 .. +0.00) | 0 / 0 | 0 (0) |
| per-bar linear | 10 | design de-dup (the mask had already dropped the session-edge columns: float path only) | +0.00 (-0.00 .. +0.05) | +0.00 (+0.00 .. +0.00) | +0.00 (+0.00 .. +0.00) | 0 / 0 | 0 (0) |
| pooled twin | 9 | nothing | +0.00 (+0.00 .. +0.00) | +0.00 (+0.00 .. +0.00) | +0.00 (+0.00 .. +0.00) | 0 / 0 | 0 (0) |
| VIX-only family | 21 | design de-dup (the mask had already dropped the session-edge columns: float path only) | +0.00 (-0.03 .. +0.06) | +0.00 (-0.01 .. +0.00) | +0.00 (-0.01 .. +0.00) | 0 / 0 | 0 (1) |
| per-bar tree (untuned) | 9 | design de-dup + per-window mask + CPU-class pinning (refit cadence unchanged, every 10) | +0.67 (-2.01 .. +1.94) | -0.10 (-0.62 .. +0.05) | -0.10 (-0.62 .. +0.05) | 0 / 2 | 37 (61) |
| per-bar tree (tuned) | 18 | design de-dup + per-window mask + CPU-class pinning (refit / tuning cadence unchanged) | +1.16 (-5.21 .. +13.66) | +0.01 (-0.69 .. +0.48) | +0.01 (-0.69 .. +0.48) | 0 / 2 | 60 (89) |
| LSTM | 6 | design de-dup + per-window mask + refit cadence 10 -> 1 + CPU-class pinning | -4.21 (-17.72 .. -1.80) | +0.10 (-0.31 .. +0.56) | +0.10 (-0.31 .. +0.56) | 0 / 0 | 110 (154) |
| direct rest-of-day at 15:30 (check) | 9 | design de-dup + CPU-architecture pinning to the per-bar twin | -0.00 (-0.00 .. +0.01) | +0.00 (+0.00 .. +0.00) | +0.00 (+0.00 .. +0.00) | 0 / 0 | 0 (0) |
| implied-vol representations | 15 | design de-dup (the mask had already dropped the session-edge columns: float path only) | +0.00 (-0.00 .. +0.06) | +0.00 (+0.00 .. +0.02) | +0.00 (+0.00 .. +0.02) | 0 / 0 | 0 (1) |
| HAR-ladder variants | 9 | design de-dup (the mask had already dropped the session-edge columns: float path only) | +0.00 (-0.01 .. +0.02) | +0.00 (-0.05 .. +0.00) | +0.00 (-0.05 .. +0.00) | 0 / 0 | 0 (2) |
| chain-period buckets | 21 | design de-dup (the mask had already dropped the session-edge columns: float path only) | -0.00 (-0.00 .. +0.00) | +0.00 (+0.00 .. +0.00) | +0.00 (+0.00 .. +0.00) | 0 / 0 | 0 (0) |

New in this run (189 keys, `status = new` in `before_after.csv`): per-bar linear 12; pooled twin 12; per-bar tree (untuned) 6; per-bar tree (untuned, daily refit) 15; per-bar tree (tuned) 6; per-bar tree (random search, daily) 24; per-bar tree (Optuna, TUNE_PER=1, best-of-50) 12; per-bar tree (Optuna, TUNE_PER=5, best-of-50) 12; per-bar tree (Optuna, TUNE_PER=25, best-of-50) 24; per-bar tree (Optuna, TUNE_PER=250, best-of-50) 12; per-bar tree (Optuna, TUNE_PER=1, best-of-25) 12; per-bar tree (Optuna, TUNE_PER=25, best-of-25) 12; per-bar tree (Optuna, TUNE_PER=1, best-of-10) 12; per-bar tree (Optuna, TUNE_PER=25, best-of-10) 12; LSTM (intraday sequence) 6.

Keys whose own mid-fill return moved with the HAC t significant at 5 % (Sharpe before -> after): tuned per-bar LightGBM [all_features], QLIKE-selected 2.23 -> 1.54 (t -2.45, p 0.014); per-bar random forest [all_features] 1.89 -> 1.27 (t -2.35, p 0.019); tuned per-bar LightGBM [baseline], MSE-selected 1.45 -> 0.95 (t -3.11, p 0.002); per-bar LightGBM [baseline] 1.19 -> 0.73 (t -2.28, p 0.023).

## (b) QLIKE vs Sharpe across models

- **paper** (9): QLIKE 0.1054 .. 0.1232, Sharpe mid 0.43 .. 1.50; best QLIKE lasso (fixed 1e-4), earlier panel [on disk, unused by the paper] (0.1054, Sharpe 1.36); best Sharpe lasso (causally tuned) (1.50, QLIKE 0.1090).
- **per-bar linear** (22): QLIKE 0.0914 .. 0.1184, Sharpe mid 1.23 .. 2.11; best QLIKE per-bar lasso [all_features], pre-tuned (0.0914, Sharpe 1.88); best Sharpe per-bar ridge [lean_vol_vix] (2.11, QLIKE 0.0966).
- **pooled twin** (21): QLIKE 0.0995 .. 0.1067, Sharpe mid 0.69 .. 1.50; best QLIKE pooled elastic net [all_features], same spec (0.0995, Sharpe 1.15); best Sharpe pooled elastic net [all_features], pre-tuned (1.50, QLIKE 0.1016).
- **VIX-only family** (21): QLIKE 0.0922 .. 0.1407, Sharpe mid 0.74 .. 2.53; best QLIKE per-bar lasso [vix_rvol] (0.0922, Sharpe 0.79); best Sharpe per-bar ridge [live_vix_only] (2.53, QLIKE 0.1179).
- **per-bar tree (untuned)** (15): QLIKE 0.0923 .. 0.0993, Sharpe mid 0.73 .. 1.78; best QLIKE per-bar XGBoost [lean_vol_vix] (0.0923, Sharpe 1.78); best Sharpe per-bar XGBoost [lean_vol_vix] (1.78, QLIKE 0.0923).
- **per-bar tree (untuned, daily refit)** (15): QLIKE 0.0911 .. 0.0986, Sharpe mid 0.66 .. 1.76; best QLIKE per-bar XGBoost, daily refit [lean_vol_vix] (0.0911, Sharpe 1.76); best Sharpe per-bar XGBoost, daily refit [lean_vol_vix] (1.76, QLIKE 0.0911).
- **per-bar tree (tuned)** (24): QLIKE 0.0932 .. 0.1118, Sharpe mid 0.65 .. 1.96; best QLIKE tuned per-bar LightGBM [live_feasible], MSE-selected (0.0932, Sharpe 1.76); best Sharpe tuned per-bar XGBoost [live_feasible], MSE-selected (1.96, QLIKE 0.1003).
- **per-bar tree (random search, daily)** (24): QLIKE 0.0926 .. 0.1093, Sharpe mid 0.46 .. 1.99; best QLIKE tuned per-bar random forest, daily refit [lean_vol_vix], QLIKE-selected (0.0926, Sharpe 1.47); best Sharpe tuned per-bar XGBoost, daily refit [live_feasible], QLIKE-selected (1.99, QLIKE 0.0986).
- **per-bar tree (Optuna, TUNE_PER=1, best-of-50)** (12): QLIKE 0.0941 .. 0.1028, Sharpe mid 0.59 .. 1.50; best QLIKE Optuna per-bar random forest, TUNE_PER 1, best of 50 [lean_vol_vix] (0.0941, Sharpe 1.50); best Sharpe Optuna per-bar random forest, TUNE_PER 1, best of 50 [lean_vol_vix] (1.50, QLIKE 0.0941).
- **per-bar tree (Optuna, TUNE_PER=5, best-of-50)** (12): QLIKE 0.0939 .. 0.1016, Sharpe mid 0.41 .. 1.48; best QLIKE Optuna per-bar random forest, TUNE_PER 5, best of 50 [lean_vol_vix] (0.0939, Sharpe 1.48); best Sharpe Optuna per-bar random forest, TUNE_PER 5, best of 50 [lean_vol_vix] (1.48, QLIKE 0.0939).
- **per-bar tree (Optuna, TUNE_PER=25, best-of-50)** (24): QLIKE 0.0934 .. 0.1042, Sharpe mid 0.45 .. 2.19; best QLIKE Optuna per-bar random forest, TUNE_PER 25, best of 50, QLIKE-selected [lean_vol_vix] (0.0934, Sharpe 1.24); best Sharpe Optuna per-bar XGBoost, TUNE_PER 25, best of 50, QLIKE-selected [live_feasible] (2.19, QLIKE 0.0970).
- **per-bar tree (Optuna, TUNE_PER=250, best-of-50)** (12): QLIKE 0.0930 .. 0.1068, Sharpe mid 0.64 .. 1.88; best QLIKE Optuna per-bar random forest, TUNE_PER 250, best of 50 [lean_vol_vix] (0.0930, Sharpe 1.29); best Sharpe Optuna per-bar LightGBM, TUNE_PER 250, best of 50 [live_feasible] (1.88, QLIKE 0.0990).
- **per-bar tree (Optuna, TUNE_PER=1, best-of-25)** (12): QLIKE 0.0944 .. 0.1029, Sharpe mid 0.46 .. 1.43; best QLIKE Optuna per-bar random forest, TUNE_PER 1, best of 25 [lean_vol_vix] (0.0944, Sharpe 1.06); best Sharpe Optuna per-bar random forest, TUNE_PER 1, best of 25 [live_feasible] (1.43, QLIKE 0.0963).
- **per-bar tree (Optuna, TUNE_PER=25, best-of-25)** (12): QLIKE 0.0948 .. 0.1016, Sharpe mid 0.33 .. 1.31; best QLIKE Optuna per-bar random forest, TUNE_PER 25, best of 25 [lean_vol_vix] (0.0948, Sharpe 1.31); best Sharpe Optuna per-bar random forest, TUNE_PER 25, best of 25 [lean_vol_vix] (1.31, QLIKE 0.0948).
- **per-bar tree (Optuna, TUNE_PER=1, best-of-10)** (12): QLIKE 0.0948 .. 0.1025, Sharpe mid 0.59 .. 1.57; best QLIKE Optuna per-bar random forest, TUNE_PER 1, best of 10 [lean_vol_vix] (0.0948, Sharpe 1.10); best Sharpe Optuna per-bar LightGBM, TUNE_PER 1, best of 10 [lean_vol_vix] (1.57, QLIKE 0.0974).
- **per-bar tree (Optuna, TUNE_PER=25, best-of-10)** (12): QLIKE 0.0937 .. 0.1053, Sharpe mid 0.29 .. 1.71; best QLIKE Optuna per-bar random forest, TUNE_PER 25, best of 10 [lean_vol_vix] (0.0937, Sharpe 1.41); best Sharpe Optuna per-bar XGBoost, TUNE_PER 25, best of 10 [lean_vol_vix] (1.71, QLIKE 0.1005).
- **LSTM** (6): QLIKE 0.0992 .. 0.1345, Sharpe mid 0.42 .. 1.37; best QLIKE per-bar LSTM, QLIKE-selected [baseline] (0.0992, Sharpe 1.13); best Sharpe per-bar LSTM [baseline] (1.37, QLIKE 0.1009).
- **LSTM (intraday sequence)** (6): QLIKE 0.1070 .. 0.1129, Sharpe mid 0.76 .. 1.55; best QLIKE intraday-sequence LSTM, QLIKE-selected [live_feasible] (0.1070, Sharpe 1.46); best Sharpe intraday-sequence LSTM [all_features] (1.55, QLIKE 0.1117).
- **implied-vol representations** (15): QLIKE 0.0919 .. 0.1145, Sharpe mid 1.25 .. 1.90; best QLIKE per-bar elastic net [ivrep_vrp] (0.0919, Sharpe 1.59); best Sharpe per-bar elastic net [ivrep_term] (1.90, QLIKE 0.0943).
- **HAR-ladder variants** (9): QLIKE 0.0930 .. 0.2288, Sharpe mid -0.12 .. 2.02; best QLIKE per-bar lasso [live_feasible], HAR ladder base2 (0.0930, Sharpe 2.02); best Sharpe per-bar lasso [live_feasible], HAR ladder base2 (2.02, QLIKE 0.0930).
- Over all of table A the lowest QLIKE is per-bar XGBoost, daily refit [lean_vol_vix] (0.0911; Sharpe 1.76, rank 42 of 295 on Sharpe) and the highest Sharpe is per-bar ridge [live_vix_only] (2.53; QLIKE 0.1179, rank 282 of 295 on QLIKE). Median over table A: QLIKE 0.0986, Sharpe mid 1.28, crossed 0.80.
- Against the reference's QLIKE, 259 forecasts have a lower point estimate (better), 137 of them with DM p < 0.05; 35 a higher one, 6 with DM p < 0.05.

Rank correlation across table-A forecasts (check rows and exact duplicates excluded), point estimate (a negative value = lower QLIKE goes with higher Sharpe):

| set | forecasts | QLIKE | Sharpe | Spearman |
|---|---:|---|---|---:|
| all table-A forecasts | 295 | plain | mid | -0.28 |
| all table-A forecasts | 295 | plain | crossed | -0.28 |
| per-bar forecasts (linear, VIX-only, trees, tuned, LSTM, bucket studies) | 265 | plain | mid | -0.27 |
| per-bar forecasts (linear, VIX-only, trees, tuned, LSTM, bucket studies) | 265 | plain | crossed | -0.27 |
| per-bar linear + VIX-only family | 43 | plain | mid | +0.49 |
| per-bar linear + VIX-only family | 43 | plain | crossed | +0.49 |
| per-bar nonlinear (every tree rung, LSTM) | 198 | plain | mid | -0.34 |
| per-bar nonlinear (every tree rung, LSTM) | 198 | plain | crossed | -0.34 |
| 48-bar forecasts (paper + pooled twins) | 30 | plain | mid | -0.39 |
| 48-bar forecasts (paper + pooled twins) | 30 | plain | crossed | -0.41 |

Reading: across all of table A the rank correlation of QLIKE with Sharpe is -0.28. The 30 forecasts fitted on all 48 bars have median QLIKE 0.1049 and median Sharpe 1.23; the 265 per-bar forecasts median QLIKE 0.0979 and median Sharpe 1.31; among the per-bar forecasts alone it is -0.27; 24 per-bar forecasts sit in the worst quarter on both counts (per-bar ridge [live_feasible], HAR ladder intra; per-bar lasso [live_feasible], HAR ladder intra; per-bar elastic net [live_feasible], HAR ladder intra; per-bar LSTM [all_features]; per-bar LSTM, QLIKE-selected [all_features]; intraday-sequence LSTM [baseline]; ...); within the per-bar linear and VIX-only family (QLIKE 0.0914 .. 0.1407, Sharpe 0.74 .. 2.53) it is +0.49.

## (c) Extreme back-transform values

- **Rule (named):** PLAIN_FLOOR(d) = the smallest 16:00 target of the previous 250 sessions (FLOOR_W, the parked recalibration's window), lagged one session. A plain forecast f²·B below it forecasts a quieter 15:30–16:00 half hour than any of the past year; it is flagged, listed in `master_table_extremes.csv`, and the QLIKE is reported as is (`qlike_raw`, the reported number) and, as a diagnostic, with flagged forecasts raised to the floor (`qlike_raw_floored`). Nothing is clipped.
- **On the trade days:** 0 flagged forecasts over all forecasts, so `qlike_raw_floored` = `qlike_raw` for 325 of 325 rows.
- **On every 16:00 session of the arm span** (1406 sessions, the arm span 2018-06-25 .. 2024-04-30 after the floor's 63-session warm-up, early closes included): 448 flagged rows over 177 forecasts, 448 of them on 8 early-close sessions (2019-07-03, 2019-11-29, 2019-12-24, 2020-11-27, 2020-12-24, 2022-11-25, 2023-07-03, 2023-11-24): 13:00 closes, where the 15:30–16:00 bar lies after the cash close and the trade frame never enters. Negative adjusted-scale forecasts occur there too (the plain back-transform squares them).
  - per-bar ridge [all_features] on 2023-11-24 (early close): f = 0.0047 on the adjusted scale, plain forecast 5.51e-11 vs floor 2.14e-07 and target 1.94e-07; that one day's QLIKE is 3516.8. Over all 1406 sessions the QLIKE is 2.6419 as is, 0.1327 floored, 0.1293 without early closes; on the trade days 0.1164.
  - per-bar LSTM [live_feasible] on 2020-11-27 (early close): f = -0.0079 on the adjusted scale, plain forecast 7.41e-10 vs floor 1.22e-07 and target 4.9e-07; that one day's QLIKE is 654.8. Over all 1406 sessions the QLIKE is 0.6085 as is, 0.1427 floored, 0.1378 without early closes; on the trade days 0.1199.
  - per-bar ridge [live_feasible_plus_ivslice], 500 sessions on 2023-07-03 (early close): f = -0.0185 on the adjusted scale, plain forecast 1.05e-09 vs floor 2.14e-07 and target 4.56e-07; that one day's QLIKE is 427.2. Over all 1406 sessions the QLIKE is 0.4410 as is, 0.1330 floored, 0.1313 without early closes; on the trade days 0.0860.
- The handoff's '16:00 ridge QLIKE 2.65 unrecalibrated' is this same early-close day: `score_rest_of_day.csv` reports the plain QLIKE 2.6480 for the all_features ridge at the 15:30 clock on 1406 sessions against the unclipped variance; the trade frame excludes early closes, so it does not enter any number of this table.

## Gates

1877 gates checked, 0 failed (`master_table_gates.csv`, column `asserted`). Among them: every arm file's 16:00 target equals the common target and the common target of `linear_subsection_dedup` equals `linear_subsection`'s bit for bit; every table's baseline equals its B, its rv_raw the production table's, and its 16:00 stamps the target's; every arm file's plain forecast equals the research reader's f²·B on the trade days; every per-bar linear and VIX-only table equals its arm file; every row's trade aggregates equal `trade_1530`'s on the plain forecast; every row of `master_table_daily.parquet` has q = sign(plain - iv_var) and the daily frame reproduces every row's Sharpe, QLIKE and buys; no table falls into 'other table'; the paper's always-short row is reproduced; the families the campaign did not touch equal the pre-campaign table under the plain scorer; stored plain numbers (16:00 campaign, plain forecast: 45; paper: 2; pre-campaign master table (plain): 17).

Reported, not asserted (2000): stored numbers written with the recalibrated forecast (compared with the same quantity recomputed here from the diagnostic `pred_clock`) or on another design — : 0 of 1 agree to 1e-09; 16:00 campaign, plain forecast: 0 of 36 agree to 1e-09; 16:00 campaign, recalibrated scorer: 530 of 770 agree to 1e-09; committed recalibrated master table (2026-09-30): 756 of 1000 agree to 1e-09; pre-campaign design: 166 of 192 agree to 1e-09; pre-campaign master table (plain): 1 of 1 agree to 1e-09. A difference there is a design change or an input that changed since the stored number was written, not a failure.

## Top of table A by sign(s) Sharpe (mid)

| forecast | QLIKE | Δ% vs ref | DM | Sharpe mid | Sharpe crossed | ΔSharpe mid vs ref | HAC t vs ref | ΔSharpe mid vs short | buy days |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| per-bar ridge [live_vix_only] | 0.1179 | +10.5 | +0.56 | 2.53 | 2.05 | +1.24 | +2.78 | +2.32 | 272 |
| per-bar ridge [free_vix_only] | 0.1194 | +11.9 | +0.58 | 2.49 | 2.02 | +1.21 | +2.65 | +2.29 | 271 |
| Optuna per-bar XGBoost, TUNE_PER 25, best of 50, QLIKE-selected [live_feasible] | 0.0970 | -9.1 | -2.48 | 2.19 | 1.70 | +0.91 | +1.97 | +1.99 | 231 |
| per-bar ridge [lean_vol_vix] | 0.0966 | -9.6 | -1.88 | 2.11 | 1.63 | +0.83 | +1.87 | +1.90 | 257 |
| per-bar ridge [free_feasible] | 0.1407 | +31.8 | +0.90 | 2.08 | 1.59 | +0.79 | +1.47 | +1.87 | 254 |
| per-bar elastic net [all_features] | 0.0986 | -7.6 | -2.03 | 2.04 | 1.56 | +0.76 | +1.63 | +1.84 | 256 |
| per-bar ridge [live_vix_rvol] | 0.1134 | +6.2 | +0.41 | 2.04 | 1.56 | +0.76 | +1.66 | +1.83 | 272 |
| per-bar ridge [free_feasible_vol] | 0.1198 | +12.2 | +0.61 | 2.03 | 1.55 | +0.75 | +1.51 | +1.83 | 268 |
| per-bar lasso [live_feasible], HAR ladder base2 | 0.0930 | -12.8 | -3.87 | 2.02 | 1.54 | +0.74 | +2.05 | +1.82 | 237 |
| per-bar lasso [all_features] | 0.0973 | -8.9 | -2.78 | 2.00 | 1.53 | +0.72 | +1.53 | +1.80 | 259 |
| tuned per-bar XGBoost, daily refit [live_feasible], QLIKE-selected | 0.0986 | -7.7 | -1.87 | 1.99 | 1.51 | +0.71 | +1.49 | +1.79 | 252 |
| tuned per-bar XGBoost [live_feasible], MSE-selected | 0.1003 | -6.1 | -1.41 | 1.96 | 1.48 | +0.68 | +1.48 | +1.75 | 260 |

Wording: sign(s) = buy the straddle when the plain 16:00 forecast f²·B exceeds the 15:30 implied variance, sell otherwise; straddle = nearest out-of-the-money call + nearest out-of-the-money put, same-day expiry, one position; the last-30-min trade enters at 15:30 and settles at the close. Buckets: `baseline` = HAR + calendar (`har_ma_*`, calendar dummies), `all_features` = the full design, `live_feasible` = the 16 series a 15:30 forecaster rebuilds live (ES return moments and liquidity, VIX/VVIX/VIX3M (`vix_ma_*` …), FOMC calendar), `vix_only` = HAR + calendar + VIX level, `live_vix_only` = live_feasible minus VVIX and VIX3M, `free_vix_only` = the free feed (ES 1-min bars + VIX + FOMC calendar). Tree rungs: T10 / T1 = the shipped configuration refit every 10 sessions / every session; RS10 / RS1 = random search (32 candidates, every 250 sessions) refit every 10 / every session; Optuna = TPE, 50 trials per tuning point, tuning every TUNE_PER sessions, best of the first k trials, refit every session.
