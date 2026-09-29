# Shift (dating) test of the headline forecast under the 16:00-bar recalibration

Written by `experiments/close_shift_test_headline.py` from `shift_test_headline.csv` and `shift_test_gates.csv` in this folder. Headline: the per-bar ridge on the live-feasible inputs (`results/spxw_pnl/yhat_sub_ridge_live_feasible.parquet`), scored exactly as the master table (`experiments/master_table_close.py`: pred_clock = (f² + s)·B with the causal 250-session smear of the 16:00 clock, the deck's 15:30 sign(s) straddle trade on `results/atm_straddle_0dte_1530/daily_blk2.parquet`, mid and crossed fills, paired circular day-block bootstrap, block 21, 2000 draws).

## What is shifted
- The forecast row read by the trade: k = 0 is the row labelled 16:00 (issued at 15:30); k < 0 reads the same session's earlier rows (k = −1 is the 15:30 row … k = −12 the 10:00 row); k > 0 reads the NEXT session's rows (k = +1 the 10:00 row … +6 the 12:30 row) — the headline is a per-bar model with session rows only, and the next session's first row is the first row whose lag ladder contains the traded bar.
- Every shifted row is placed on the traded bar's own scale, (ŷ_k² + s_16:00)·B_16:00, with the same smear s and profile B the trade uses; at k = 0 this is the trade's forecast itself (gated). The star is the traded bar's realized variance (the scorer's target) in place of the forecast.
- Days: the 865 deck days for which every shifted row exists (of 866; the last deck day has no next session, and the 10:00–11:00 rows are absent on two sessions). k = 0 on all 866 days reproduces the master table: Sharpe 1.9025 mid / 1.4396 crossed (gated to 1e-9).

## Results (annualized Sharpe; Δ vs k = 0 with the paired 95 % interval)

| row | session | clock | Sharpe mid | Sharpe crossed | buy % | agree with k=0 % | Δ mid [95 %] | Δ crossed [95 %] |
|---|---|---|---|---|---|---|---|---|
| bar-12 | same | 10:00 | 0.12 | -0.35 | 49 | 69 | -1.73 [-2.91, -0.57] | -1.74 [-2.92, -0.57] |
| bar-11 | same | 10:30 | -0.31 | -0.78 | 48 | 69 | -2.17 [-3.34, -1.03] | -2.17 [-3.35, -1.02] |
| bar-10 | same | 11:00 | -0.03 | -0.50 | 48 | 66 | -1.88 [-3.07, -0.67] | -1.89 [-3.08, -0.67] |
| bar-9 | same | 11:30 | 0.35 | -0.13 | 43 | 68 | -1.50 [-2.70, -0.28] | -1.51 [-2.71, -0.28] |
| bar-8 | same | 12:00 | 0.42 | -0.05 | 46 | 70 | -1.44 [-2.52, -0.36] | -1.44 [-2.53, -0.36] |
| bar-7 | same | 12:30 | 0.51 | 0.03 | 42 | 69 | -1.35 [-2.47, -0.22] | -1.36 [-2.49, -0.23] |
| bar-6 | same | 13:00 | 0.64 | 0.17 | 44 | 69 | -1.21 [-2.46, +0.07] | -1.22 [-2.48, +0.07] |
| bar-5 | same | 13:30 | 0.95 | 0.47 | 38 | 70 | -0.91 [-1.92, +0.11] | -0.91 [-1.93, +0.11] |
| bar-4 | same | 14:00 | 0.93 | 0.45 | 37 | 71 | -0.93 [-2.11, +0.21] | -0.94 [-2.11, +0.21] |
| bar-3 | same | 14:30 | 1.29 | 0.80 | 26 | 72 | -0.57 [-1.67, +0.46] | -0.59 [-1.69, +0.45] |
| bar-2 | same | 15:00 | 0.50 | 0.02 | 25 | 71 | -1.35 [-2.72, -0.01] | -1.37 [-2.74, -0.02] |
| bar-1 | same | 15:30 | -0.00 | -0.47 | 32 | 75 | -1.86 [-2.96, -0.78] | -1.86 [-2.96, -0.79] |
| bar+0 | same | 16:00 | 1.85 | 1.39 | 40 | 100 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| bar+1 | next | 10:00 | 0.88 | 0.41 | 44 | 68 | -0.97 [-2.22, +0.31] | -0.97 [-2.24, +0.32] |
| bar+2 | next | 10:30 | 0.93 | 0.46 | 44 | 68 | -0.92 [-1.91, +0.12] | -0.92 [-1.91, +0.13] |
| bar+3 | next | 11:00 | 1.00 | 0.53 | 41 | 66 | -0.85 [-2.12, +0.45] | -0.86 [-2.12, +0.46] |
| bar+4 | next | 11:30 | 0.83 | 0.36 | 42 | 66 | -1.02 [-2.30, +0.26] | -1.03 [-2.31, +0.26] |
| bar+5 | next | 12:00 | 1.22 | 0.75 | 41 | 65 | -0.63 [-1.93, +0.63] | -0.64 [-1.95, +0.63] |
| bar+6 | next | 12:30 | 1.51 | 1.04 | 42 | 63 | -0.34 [-1.66, +0.89] | -0.35 [-1.67, +0.90] |
| realized (target) | traded bar | 16:00 | 4.48 | 4.02 | 27 | 67 | +2.63 [+1.21, +4.17] | +2.63 [+1.20, +4.17] |
| realized (unclipped) | traded bar | 16:00 | 4.50 | 4.03 | 27 | 67 | +2.64 [+1.22, +4.19] | +2.65 [+1.21, +4.19] |
| always short |  |  | 0.27 | -0.22 | 0 | 60 | -1.58 [-2.98, -0.13] | -1.60 [-3.00, -0.16] |

## Read-offs (from the table above)
- One bar stale (k = −1, the 15:30 row): Sharpe -0.00 mid, Δ -1.86 [-2.96, -0.78]; its position agrees with the trade's on 75 % of days.
- Stale rows (k = −12 … −1): 0 of 12 sit above k = 0 in point estimate; 0 with an interval above zero, 8 with an interval below zero. Best stale row: bar-3 at 1.29 (Δ -0.57 [-1.67, +0.46]).
- First row after the close (k = +1, the next session's 10:00 row): Sharpe 0.88 mid / 0.41 crossed, Δ -0.97 [-2.22, +0.31]; agreement with the trade 68 %. Next-session rows k = +1 … +6: Sharpe mid 0.83 to 1.51; 0 of 6 intervals above zero.
- Realized variance in place of the forecast: 4.48 mid / 4.02 crossed (unclipped realized: 4.50 / 4.03). Always short on the same days: 0.27 / -0.22.

## Context, not comparable
- The notebook's own shift test (`results/atm_straddle_0dte_1530/forecast_shift_cliff.csv`, section 11) runs the paper's eight forecasts under the deck's session-bar recalibration and has after-hours rows for k > 0; its numbers are on the other scorer and are not set beside these.

## Gates
- baseline equals the target's B: 0 ≤ 1e-09 on n = 1469 — PASS
- recovered s16 equals the causal smear: 1.51e-14 ≤ 1e-09 on n = 1406 — PASS
- k = 0 days = the master table's: 0 ≤ 0 on n = 866 — PASS
- k = 0 Sharpe mid = the master table's: 0 ≤ 1e-09 on n = 866 — PASS
- k = 0 Sharpe crossed = the master table's: 0 ≤ 1e-09 on n = 866 — PASS
- bar+0 placed equals pred_clock: 0 ≤ 1e-09 on n = 866 — PASS
