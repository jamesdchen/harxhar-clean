# Master table for the closing strategy (A4) — summary

Written by `experiments/master_table_close.py` on 2026-09-29 02:55; every number below is read from
`master_table.csv` of the same run. Full table: `writeup/master_table_close.pdf`; tex: `writeup/generated/table_master_close.tex`.

## Scorer, days, reference

- **Scorer: the research convention** (`compare_mfiv_harlag.py` / `score_trees_subsection.py`, their functions imported): the 16:00 bar is recalibrated on its own, forecast = (f² + s)·B with s = the forecast's own mean squared adjusted-scale error at 16:00 over the previous 250 sessions (≥ 63), lagged one session; QLIKE against the per-bar spec's 16:00 target; the trade is the deck's 15:30 sign(s) on the straddle (`trade_1530`). The notebook's 13-bar Mincer–Zarnowitz map is **not** used: no number here may be set beside a notebook number.
- **Days:** table A = 100 forecasts (+ 9 check rows) on the same 866 trade days (2020-01-03 .. 2024-04-30); intersection over table A = 866 of 866 trade days. Table B = 21 forecasts scored on their own days (paired with the reference on those days): 21 chain-period buckets forecasts from 2022-01-03 on 550 trade days. Forecasts that failed to load: none.
- **Which forecasts drop days:** no table-A forecast misses a trade day; 0 forecasts miss trade days inside their own window. The 21 table-B forecasts start at their comparison window (2022-01-03), so each omits the 316 earlier trade days by design (their implied-vol inputs are real chain data only from then; compare_mfiv_harlag.CHAIN_START). Tuned per-bar trees on disk: 18 tables; LSTM: 0 tables (both are picked up by glob on the next run). Coverage per forecast: `master_table_days.csv`.
- **Reference:** the block-diagonal ridge (`blk2`, `yhat_blk2_fomc1.parquet`) — the paper's headline forecast (HAR ladder, penalty 1, plus the exogenous block, penalty 100, on the panel of record), the forecast the paper's 15:30 deck trades. Under this scorer it has QLIKE 0.1058 and sign(s) Sharpe 1.00 mid / 0.53 crossed (the deck's own map gives it 1.34 mid; the difference is the scorer). Second reference: always short, Sharpe 0.20 mid / -0.27 crossed.

## (a) Recommended headline forecast

**per-bar ridge [live_feasible]** (`sub_ridge_live_feasible`), scored by the research convention above. On the 866 days: QLIKE 0.1005 (-5.0 % vs the reference, day-block interval on the daily difference [-0.0113, +0.0005], DM -1.42, p 0.156); sign(s) Sharpe 1.90 mid / 1.44 crossed, mean +0.134 per unit premium (crossed +0.101), hit rate 54.6 %, buys on 348 days (40.2 %). Sharpe difference vs the reference: mid +0.90 [+0.19, +1.66], crossed +0.91 [+0.19, +1.67]; vs always short: mid +1.70 [+0.09, +3.11], crossed +1.71 [+0.12, +3.12].

Why this one:
- It is a 15:30-specific model on the inputs a 15:30 forecaster can rebuild live (`live_feasible`), so the trade it scores can be run. It ranks 2 of 100 table-A forecasts on sign(s) Sharpe (mid) and 55 of 100 on the recalibrated QLIKE.
- The highest Sharpe in table A is per-bar lasso [free_vix_only] (1.91 mid / 1.45 crossed). Choosing the maximum of 100 Sharpe ratios would select on the trade's own noise; the headline is chosen on feasibility and on being the per-bar model of record of the research scorer (the per-bar ridge is the estimator the 15:30 study defined first), not on the maximum.
- Paired against the headline itself on the same days (`master_table_vs_headline.csv`; a negative QLIKE % / DM = the alternative forecasts better, a positive Sharpe difference = the alternative trades better; 95 % intervals):

  | alternative | QLIKE | % vs headline | DM | Sharpe mid / crossed | ΔSharpe mid vs headline | ΔSharpe crossed vs headline | same position |
  |---|---:|---:|---:|---|---|---|---:|
  | per-bar lasso [live_feasible] | 0.0964 | -4.1 | -1.20 | 1.83 / 1.36 | -0.08 [-0.89, +0.79] | -0.08 [-0.90, +0.79] | 89.3 % |
  | per-bar elastic net [live_feasible] | 0.0968 | -3.7 | -1.38 | 1.41 / 0.94 | -0.49 [-1.08, +0.14] | -0.50 [-1.09, +0.14] | 90.9 % |
  | per-bar ridge [all_features] | 0.1004 | -0.2 | -0.06 | 1.66 / 1.20 | -0.24 [-1.07, +0.54] | -0.24 [-1.08, +0.54] | 86.0 % |
  | per-bar ridge [baseline] | 0.0982 | -2.3 | -0.53 | 1.22 / 0.74 | -0.69 [-1.70, +0.23] | -0.70 [-1.72, +0.23] | 82.4 % |
  | per-bar lasso [free_vix_only] | 0.0972 | -3.3 | -0.95 | 1.91 / 1.45 | +0.01 [-0.75, +0.79] | +0.01 [-0.76, +0.79] | 87.9 % |
  | per-bar ridge [free_vix_only] | 0.0999 | -0.6 | -0.65 | 1.83 / 1.36 | -0.07 [-0.46, +0.29] | -0.08 [-0.46, +0.29] | 95.2 % |
  | per-bar LightGBM [live_feasible] | 0.0982 | -2.3 | -0.74 | 1.69 / 1.22 | -0.21 [-0.86, +0.51] | -0.21 [-0.87, +0.51] | 85.8 % |
  | per-bar XGBoost [live_feasible] | 0.0988 | -1.7 | -0.49 | 1.70 / 1.23 | -0.21 [-0.82, +0.43] | -0.21 [-0.83, +0.43] | 86.1 % |
  | tuned per-bar LightGBM [all_features], MSE-selected | 0.1011 | +0.6 | +0.16 | 1.81 / 1.35 | -0.09 [-0.85, +0.63] | -0.09 [-0.85, +0.63] | 84.9 % |
  | tuned per-bar LightGBM [all_features], QLIKE-selected | 0.0992 | -1.4 | -0.39 | 1.75 / 1.29 | -0.15 [-0.79, +0.50] | -0.15 [-0.79, +0.51] | 85.1 % |

- Over all 99 other table-A forecasts, 0 trade better than the headline with an interval above zero (mid; 0 crossed) and 14 trade worse with an interval below zero (mid; 14 crossed); 3 forecast better on QLIKE with the day-block interval below zero, 11 worse.
- Against the reference, 9 of 99 table-A forecasts have a mid-fill Sharpe-difference interval wholly above zero; 9 at the crossed fill. Against always short, 27 (mid) and 32 (crossed) of 100.
- On forecast accuracy alone the best live-feasible forecast is per-bar elastic net [live_feasible], HAR ladder base3 (QLIKE 0.0934, Sharpe 1.57 / 1.11); a QLIKE-first choice would take it. The trade does not rank forecasts the way QLIKE does (section b), which is why the headline names the scorer AND the trade numbers.

## (b) QLIKE vs Sharpe across models

- **paper** (9): QLIKE 0.1006 .. 0.1172, Sharpe mid 1.00 .. 1.54; best QLIKE LightGBM (0.1006, Sharpe 1.48); best Sharpe lasso (fixed 1e-4) (1.54, QLIKE 0.1067).
- **per-bar linear** (10): QLIKE 0.0964 .. 0.1005, Sharpe mid 1.22 .. 1.90; best QLIKE per-bar lasso [live_feasible] (0.0964, Sharpe 1.83); best Sharpe per-bar ridge [live_feasible] (1.90, QLIKE 0.1005).
- **pooled twin** (9): QLIKE 0.1008 .. 0.1106, Sharpe mid 0.99 .. 1.32; best QLIKE pooled elastic net [all_features], same spec (0.1008, Sharpe 1.19); best Sharpe pooled elastic net [live_feasible], same spec (1.32, QLIKE 0.1020).
- **VIX-only family** (21): QLIKE 0.0954 .. 0.1016, Sharpe mid 1.26 .. 1.91; best QLIKE per-bar lasso [vix_rvol] (0.0954, Sharpe 1.68); best Sharpe per-bar lasso [free_vix_only] (1.91, QLIKE 0.0972).
- **per-bar tree (untuned)** (9): QLIKE 0.0982 .. 0.1054, Sharpe mid 0.89 .. 1.70; best QLIKE per-bar LightGBM [live_feasible] (0.0982, Sharpe 1.69); best Sharpe per-bar XGBoost [live_feasible] (1.70, QLIKE 0.0988).
- **per-bar tree (tuned)** (18): QLIKE 0.0992 .. 0.1120, Sharpe mid 0.58 .. 1.81; best QLIKE tuned per-bar LightGBM [all_features], QLIKE-selected (0.0992, Sharpe 1.75); best Sharpe tuned per-bar LightGBM [all_features], MSE-selected (1.81, QLIKE 0.1011).
- **implied-vol representations** (15): QLIKE 0.0950 .. 0.1001, Sharpe mid 0.97 .. 1.87; best QLIKE per-bar elastic net [ivrep_innovations] (0.0950, Sharpe 1.61); best Sharpe per-bar lasso [ivrep_target_scale] (1.87, QLIKE 0.0960).
- **HAR-ladder variants** (9): QLIKE 0.0934 .. 0.2184, Sharpe mid -0.02 .. 1.79; best QLIKE per-bar elastic net [live_feasible], HAR ladder base3 (0.0934, Sharpe 1.57); best Sharpe per-bar lasso [live_feasible], HAR ladder base2 (1.79, QLIKE 0.0948).
- Over all of table A the lowest QLIKE is per-bar elastic net [live_feasible], HAR ladder base3 (0.0934; Sharpe 1.57, rank 38 of 100 on Sharpe) and the highest Sharpe is per-bar lasso [free_vix_only] (1.91; QLIKE 0.0972, rank 30 of 100 on QLIKE).
- Against the reference's QLIKE, 44 forecasts have a day-block interval wholly below zero (better) and 4 wholly above (worse). The raw (plain back-transform) QLIKE ranks forecasts differently from the recalibrated one (Spearman across table A +0.73); the recalibration's term s lowers the loss most for per-bar ridge [free_feasible] 0.1407 raw vs 0.1016 recalibrated; per-bar ridge [live_feasible], HAR ladder base3 0.1237 raw vs 0.0998 recalibrated; per-bar ridge [free_feasible_vol] 0.1198 raw vs 0.1001 recalibrated.

Rank correlation across table-A forecasts (check rows and exact duplicates excluded), point and 95 % day-block bootstrap interval (the same resampled days as every other interval; a negative value = lower QLIKE goes with higher Sharpe):

| set | forecasts | QLIKE measure | Sharpe | Spearman | 95 % interval |
|---|---:|---|---|---:|---|
| all table-A forecasts | 100 | recal | mid | -0.64 | [-0.73, -0.11] |
| all table-A forecasts | 100 | recal | crossed | -0.64 | [-0.73, -0.11] |
| all table-A forecasts | 100 | raw | mid | -0.35 | [-0.67, +0.04] |
| all table-A forecasts | 100 | raw | crossed | -0.35 | [-0.67, +0.05] |
| per-bar forecasts (linear, VIX-only, trees, tuned, LSTM, bucket studies) | 82 | recal | mid | -0.52 | [-0.74, -0.01] |
| per-bar forecasts (linear, VIX-only, trees, tuned, LSTM, bucket studies) | 82 | recal | crossed | -0.51 | [-0.74, -0.00] |
| per-bar forecasts (linear, VIX-only, trees, tuned, LSTM, bucket studies) | 82 | raw | mid | -0.27 | [-0.68, +0.14] |
| per-bar forecasts (linear, VIX-only, trees, tuned, LSTM, bucket studies) | 82 | raw | crossed | -0.27 | [-0.68, +0.14] |
| per-bar linear + VIX-only family | 31 | recal | mid | +0.18 | [-0.70, +0.64] |
| per-bar linear + VIX-only family | 31 | recal | crossed | +0.19 | [-0.70, +0.64] |
| per-bar linear + VIX-only family | 31 | raw | mid | +0.26 | [-0.70, +0.72] |
| per-bar linear + VIX-only family | 31 | raw | crossed | +0.26 | [-0.70, +0.72] |
| 48-bar forecasts (paper + pooled twins) | 18 | recal | mid | -0.37 | [-0.69, +0.47] |
| 48-bar forecasts (paper + pooled twins) | 18 | recal | crossed | -0.34 | [-0.69, +0.47] |
| 48-bar forecasts (paper + pooled twins) | 18 | raw | mid | +0.02 | [-0.66, +0.63] |
| 48-bar forecasts (paper + pooled twins) | 18 | raw | crossed | +0.02 | [-0.66, +0.63] |

Reading: across all of table A a lower QLIKE goes with a higher Sharpe (-0.64 [-0.73, -0.11]). Part of it is the split between the 18 forecasts fitted on all 48 bars (median QLIKE 0.1039, median Sharpe 1.20) and the 82 per-bar forecasts (median QLIKE 0.0993, median Sharpe 1.51); among the per-bar forecasts alone it is -0.52 [-0.74, -0.01], carried by the 14 per-bar forecasts in the worst quarter on both counts (per-bar ridge [live_feasible], HAR ladder intra; per-bar lasso [live_feasible], HAR ladder intra; per-bar elastic net [live_feasible], HAR ladder intra; tuned per-bar random forest [baseline], MSE-selected; tuned per-bar XGBoost [baseline], QLIKE-selected; tuned per-bar XGBoost [baseline], MSE-selected; ...); within the per-bar linear and VIX-only family (QLIKE 0.0954 .. 0.1016, Sharpe 1.22 .. 1.91) it is +0.18 [-0.70, +0.64]: among forecasts of similar accuracy, QLIKE does not order the trade.

## (c) Extreme back-transform values

- **Rule (named):** PLAIN_FLOOR(d) = the smallest 16:00 target of the previous 250 sessions (the recalibration window SMEAR_W), lagged one session. A plain back-transform f²·B below it forecasts a quieter 15:30–16:00 half hour than any of the past year; it is flagged, listed in `master_table_extremes.csv`, and the raw QLIKE is reported as is (`qlike_raw`) and with flagged forecasts raised to the floor (`qlike_raw_floored`). Nothing is clipped silently; the recalibrated forecast (≥ s·B) needs no treatment and is never below the floor on a trade day in this run (max count 0).
- **On the trade days:** 0 flagged forecasts over all forecasts, so `qlike_raw_floored` = `qlike_raw` for 130 of 130 rows.
- **On every 16:00 session of the arm span** (1406 sessions, the arm span 2018-06-25 .. 2024-04-30 after the floor's 63-session warm-up, early closes included): 296 flagged rows over 90 forecasts, 296 of them on 8 early-close sessions (2019-07-03, 2019-11-29, 2019-12-24, 2020-11-27, 2020-12-24, 2022-11-25, 2023-07-03, 2023-11-24): 13:00 closes, where the 15:30–16:00 bar lies after the cash close and the trade frame never enters. Negative adjusted-scale forecasts occur there too (the plain back-transform squares them).
  - per-bar ridge [all_features] on 2023-11-24 (early close): f = 0.0047 on the adjusted scale, plain forecast 5.51e-11 vs floor 2.14e-07 and target 1.94e-07; that one day's raw QLIKE is 3516.8. Over all 1406 sessions the raw QLIKE is 2.6419 as is, 0.1327 floored, 0.1293 without early closes; on the trade days 0.1164 (recalibrated 0.1004).
  - per-bar ridge [live_feasible_plus_ivslice], 500 sessions on 2023-07-03 (early close): f = -0.0185 on the adjusted scale, plain forecast 1.05e-09 vs floor 2.14e-07 and target 4.56e-07; that one day's raw QLIKE is 427.2. Over all 1406 sessions the raw QLIKE is 0.4410 as is, 0.1330 floored, 0.1313 without early closes; on the trade days 0.0860 (recalibrated 0.0833).
  - per-bar elastic net [vix_rvol] on 2020-11-27 (early close): f = -0.0120 on the adjusted scale, plain forecast 1.7e-09 vs floor 1.22e-07 and target 4.9e-07; that one day's raw QLIKE is 282.2. Over all 1406 sessions the raw QLIKE is 0.3157 as is, 0.1153 floored, 0.1126 without early closes; on the trade days 0.0925 (recalibrated 0.0956).
- The handoff's '16:00 ridge QLIKE 2.65 unrecalibrated' is this same early-close day: `score_rest_of_day.csv` reports the plain QLIKE 2.6480 for the all_features ridge at the 15:30 clock on 1406 sessions against the unclipped variance; the recalibration (its `qlike_causal` 0.1248) and the trade frame's exclusion of early closes both remove it.

## Gates

588 gates checked, 0 failed (`master_table_gates.csv`). Among them: every arm file's 16:00 target equals the common target; every table's baseline equals its B; every arm file's recalibrated forecast equals the research reader's on the trade days; every row's trade aggregates equal `trade_1530`'s; the stored research trade numbers (untuned trees and their linear comparators, the bucket-study compare CSVs) and the paper's always-short row are reproduced.

## Top of table A by sign(s) Sharpe (mid)

| forecast | QLIKE raw | QLIKE recal | Δ% vs ref | DM | Sharpe mid | Sharpe crossed | ΔSharpe mid vs ref [95 %] | ΔSharpe mid vs short [95 %] | buy days |
|---|---:|---:|---:|---:|---:|---:|---|---|---:|
| per-bar lasso [free_vix_only] | 0.0939 | 0.0972 | -8.1 | -2.75 | 1.91 | 1.45 | +0.91 [-0.04, +1.88] | +1.71 [+0.28, +3.04] | 329 |
| per-bar ridge [live_feasible] | 0.1184 | 0.1005 | -5.0 | -1.42 | 1.90 | 1.44 | +0.90 [+0.19, +1.66] | +1.70 [+0.09, +3.11] | 348 |
| per-bar elastic net [live_vix_only] | 0.0952 | 0.0967 | -8.6 | -3.06 | 1.90 | 1.43 | +0.90 [+0.12, +1.77] | +1.69 [+0.30, +2.98] | 332 |
| per-bar lasso [ivrep_target_scale] | 0.0932 | 0.0960 | -9.3 | -2.81 | 1.87 | 1.41 | +0.87 [+0.06, +1.68] | +1.67 [+0.25, +2.97] | 343 |
| per-bar elastic net [live_vix_rvol] | 0.0943 | 0.0959 | -9.4 | -3.42 | 1.87 | 1.40 | +0.86 [+0.19, +1.66] | +1.66 [+0.20, +2.99] | 330 |
| per-bar ridge [live_vix_only] | 0.1179 | 0.1005 | -5.0 | -1.41 | 1.86 | 1.39 | +0.86 [+0.06, +1.66] | +1.66 [+0.04, +3.13] | 351 |
| per-bar lasso [free_feasible_vol] | 0.0930 | 0.0963 | -9.0 | -3.07 | 1.85 | 1.39 | +0.85 [-0.17, +1.88] | +1.65 [+0.20, +2.93] | 336 |
| per-bar lasso [live_vix_only] | 0.0938 | 0.0970 | -8.4 | -2.82 | 1.85 | 1.38 | +0.85 [-0.10, +1.86] | +1.65 [+0.23, +2.97] | 329 |
| per-bar ridge [free_vix_only] | 0.1194 | 0.0999 | -5.5 | -1.62 | 1.83 | 1.36 | +0.83 [-0.00, +1.66] | +1.62 [-0.04, +3.11] | 348 |
| per-bar lasso [live_feasible] | 0.0931 | 0.0964 | -8.8 | -3.04 | 1.83 | 1.36 | +0.82 [-0.18, +1.86] | +1.62 [+0.20, +2.91] | 337 |
| per-bar elastic net [free_vix_only] | 0.0950 | 0.0966 | -8.7 | -3.13 | 1.81 | 1.35 | +0.81 [-0.01, +1.69] | +1.61 [+0.21, +2.88] | 333 |
| tuned per-bar LightGBM [all_features], MSE-selected | 0.0963 | 0.1011 | -4.4 | -1.20 | 1.81 | 1.35 | +0.81 [-0.07, +1.71] | +1.61 [+0.23, +2.74] | 341 |

Wording: sign(s) = buy the straddle when the recalibrated 16:00 forecast exceeds the 15:30 implied variance, sell otherwise; straddle = nearest out-of-the-money call + nearest out-of-the-money put, same-day expiry, one position; the last-30-min trade enters at 15:30 and settles at the close. Buckets: `baseline` = HAR + calendar (`har_ma_*`, calendar dummies), `all_features` = the full design, `live_feasible` = the 16 series a 15:30 forecaster rebuilds live (ES return moments and liquidity, VIX/VVIX/VIX3M (`adj_vix_ma_*` …), FOMC calendar), `vix_only` = HAR + calendar + VIX level, `live_vix_only` = live_feasible minus VVIX and VIX3M, `free_vix_only` = the free feed (ES 1-min bars + VIX + FOMC calendar).
