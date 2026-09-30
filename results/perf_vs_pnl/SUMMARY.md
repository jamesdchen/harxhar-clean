# Forecast accuracy vs P&L of the 15:30 last-30-min trade (checklist C3)

Written by `experiments/perf_vs_pnl_1530.py` from its own outputs in this folder (`forecast_metrics.csv`, `rank_vs_sharpe.csv`, `rank_vs_sharpe_by_family.csv`, `rank_vs_sharpe_by_model_family.csv`, `rank_vs_sharpe_differences.csv`, `per_day_regressions.csv`, `numbers.json`); every qualitative claim is a check on those numbers (`claims.csv`): the asserted ones are verified before this file is written, the others choose the wording.

**Universe:** 220 forecasts of the bar ending 16:00: the table-A rows of the closing-strategy master table (`results/close_master_table/master_table.csv`, `master_table_daily.parquet`; check rows, exact duplicates and the always-short rule left out), in 20 model families. By display class: 48-bar (paper + pooled twins) 18, per-bar linear 46, per-bar linear, HAR-ladder variants 9, per-bar trees 135, LSTM (per-bar and intraday sequence) 12. Same 866 trade days for every forecast (2020-01-03 .. 2024-04-30). The master table is the one rebuilt after the 16:00 campaign (de-duplicated per-bar design; every tree and LSTM fit with the per-window mask; tree rungs T10, T1, RS10, RS1 and the Optuna rungs; the per-bar and the intraday-sequence LSTM).

**One scorer, the research scorer:** the 16:00 bar recalibrated alone (forecast = (ŷ² + s)·B, s = the forecast's own trailing-250-session mean squared error, lagged one session); every loss is against the per-bar spec's 16:00 target. The trade is the deck's 15:30 **sign(s)** rule: buy the **straddle** (nearest out-of-the-money call + nearest out-of-the-money put, same-day expiry, one position) when the forecast exceeds the 15:30 implied variance, sell it otherwise, hold to the close; Sharpe annualized with √252, mid and crossed fills. The per-day trade and QLIKE reproduce the master table's Sharpe and QLIKE to 5e-15.

**Metrics per forecast:** QLIKE; MSE (variance units); mean squared log error; calibration error |b − 1| with b the slope of log realized on log forecast; **sign accuracy** = the share of days on which sign(forecast − implied) equals sign(realized − implied); hit rate = the share of days the trade made money; the same on the **tail days** = the 20 / 50 trade days with the largest |straddle return| (|R| ≥ 3.02 / 1.91 premium, against a median of 0.79; the same days for every forecast). **Rank agreement** = Spearman correlation across the forecasts between a metric and the Sharpe, signed so that +1 means the metric orders the forecasts exactly as the Sharpe does; 95 % intervals from a circular block bootstrap over days (block 21 sessions, 2,000 draws; the whole panel of forecasts resampled together); a **paired difference** of two agreements is computed on the same draws.

## Answer in brief

1. **Across all 220 forecasts a lower QLIKE goes with a higher Sharpe (Spearman -0.56, interval [-0.65, -0.06]), but not among forecasts of similar accuracy.** Within the 46 per-bar linear forecasts (all buckets, the VIX-only and implied-vol families) the rank agreement of QLIKE with the Sharpe is -0.08 [-0.46, +0.62]; within the 135 per-bar trees +0.37 [-0.16, +0.65]. The best-QLIKE forecast (per-bar elastic net [live_feasible], HAR ladder base3, QLIKE 0.0934) ranks 53 of 220 on the Sharpe (1.57); the best-Sharpe forecast (per-bar lasso [live_vix_only], 1.91) ranks 26 of 220 on QLIKE.
2. **The other accuracy losses do no better.** MSE in variance units orders the forecasts the wrong way round (agreement -0.51 [-0.61, +0.45]); the squared log error +0.52 [+0.02, +0.64]; the calibration error -0.07 [-0.43, +0.36].
3. **Calling the side right tracks the trade: on the few days the straddle moves most and, across these 220 forecasts, on the other days as well.** Sign accuracy over all days hardly separates the forecasts (50–71 %, median 66 %) and ranks them (agreement +0.56 [+0.09, +0.67]). On the 20 largest-|return| days it does, and more strongly in the point estimate (+0.59 [+0.19, +0.79]; 50 days +0.46 [+0.09, +0.71]); the paired difference from the all-day agreement is +0.03 [-0.28, +0.51], an interval that covers zero; and on the other 846 days it does as well, less strongly in the point estimate (+0.50 [+0.01, +0.63]; paired difference of the 20-day agreement over it +0.09 [-0.26, +0.64]). Within the per-bar linear class, where QLIKE says nothing, the 20-day sign accuracy has agreement +0.72 [-0.01, +0.86]: suggestive, with an interval that reaches zero.
4. **Per day, being on the right side explains the P&L; the size of the forecast error does not.** Regressing the day's trade return on an indicator of a right sign gives R² 6.9 % (median; 6.2–7.6 % across the 220 forecasts; HAC t 7.5 to 8.7, every forecast significant). On the forecast's log error it gives R² 0.12 % (median; 2 of 220 with p < 0.05, the rate chance gives); on its absolute log error 1.17 % (median slope -0.41, 172 of 220 with p < 0.05: larger misses cost a little). Adding the log error to the sign indicator adds nothing (median R² 7.0 %). For the headline per-bar ridge (`sub_ridge_live_feasible`) the sign is right on 584 of 866 days (mean return +0.33 per unit premium) and wrong on 282 (-0.27); its 20 tail days carry 34 % of its total P&L (50 days 43 %).
5. **Within the model families** (`rank_vs_sharpe_by_model_family.csv`, figure `rank_vs_sharpe_by_model_family.png`; a family = the forecasts of one model class on one rung of the ladder, differing in input set, estimator or selection rule): QLIKE's agreement with the Sharpe has an interval wholly above zero in 1 of the 20 families (HAR-ladder variants +0.80 [+0.47, +0.95]) and wholly below zero in 0; its mean over the 20 families is +0.29 [-0.10, +0.52], over the 12 tree rungs +0.49 [-0.15, +0.68]; the 20-day sign accuracy's agreement has an interval wholly above zero in 0 of the 20 families and wholly below zero in 0; its mean over the 20 families is +0.45 [+0.13, +0.73], over the 12 tree rungs +0.37 [-0.02, +0.78]. Within a family, on average, QLIKE does not order the trade and the 20-day sign accuracy orders the trade; paired, the 20-day sign accuracy's agreement minus QLIKE's is +0.16 [-0.24, +0.71] over the families and -0.11 [-0.52, +0.76] over the tree rungs, intervals that cover zero: on these days the two are not told apart.

**Verdict.** QLIKE is the right loss for forecasting the 15:30–16:00 variance, and across model families a better QLIKE comes with a better trade: the 48-bar forecasts (18) forecast worse (median QLIKE 0.1039 against 0.1032) and trade worse (median Sharpe 1.20 against 1.33) than the per-bar ones (202), and among the per-bar trees the ordering points the same way (+0.37 [-0.16, +0.65]). Among the 46 per-bar linear forecasts, the candidates for the headline, it does not (-0.08 [-0.46, +0.62]): there a QLIKE gain does not imply a trade gain (e.g. the per-bar ridge on the VIX-only bucket: QLIKE 0.0959 against the headline's 0.1005, Sharpe 1.48 against 1.90). The trade is decided by the sign of forecast minus implied, with a large share of the P&L on a handful of large-move days (the 20 largest carry 17 % of total P&L for the median forecast), and QLIKE, an average over all 866 days of how far the forecast misses, gives them little weight (they make up 4 % of the median forecast's summed QLIKE). The trade's own measure (sign accuracy weighted by the size of the move, i.e. the P&L) is the relevant one for choosing among comparable forecasts, and on 866 days it is too noisy to rank them reliably.

## Rank agreement with the Sharpe, all forecasts

| metric | better | agreement with Sharpe (mid) | agreement with Sharpe (crossed) |
|---|---|---|---|
| QLIKE | lower | +0.56 [+0.06, +0.65] | +0.56 [+0.06, +0.65] |
| MSE (variance units) | lower | -0.51 [-0.61, +0.45] | -0.51 [-0.61, +0.44] |
| mean squared log error | lower | +0.52 [+0.02, +0.64] | +0.51 [+0.02, +0.64] |
| calibration error \|log MZ slope - 1\| | lower | -0.07 [-0.43, +0.36] | -0.07 [-0.43, +0.36] |
| sign accuracy, all days | higher | +0.56 [+0.09, +0.67] | +0.55 [+0.08, +0.67] |
| hit rate (trade made money), all days | higher | +0.65 [+0.46, +0.83] | +0.64 [+0.45, +0.83] |
| QLIKE, 20 largest-\|return\| days | lower | +0.40 [-0.05, +0.69] | +0.40 [-0.05, +0.69] |
| QLIKE, other 846 days | lower | +0.55 [+0.03, +0.63] | +0.54 [+0.03, +0.63] |
| QLIKE, 50 largest-\|return\| days | lower | +0.42 [-0.01, +0.66] | +0.43 [-0.00, +0.66] |
| QLIKE, other 816 days | lower | +0.53 [+0.03, +0.62] | +0.53 [+0.03, +0.62] |
| sign accuracy, 20 largest-\|return\| days | higher | +0.59 [+0.19, +0.79] | +0.59 [+0.20, +0.79] |
| sign accuracy, other 846 days | higher | +0.50 [+0.01, +0.63] | +0.50 [+0.01, +0.62] |
| sign accuracy, 50 largest-\|return\| days | higher | +0.46 [+0.09, +0.71] | +0.47 [+0.09, +0.72] |
| sign accuracy, other 816 days | higher | +0.49 [-0.01, +0.62] | +0.48 [-0.02, +0.61] |
| hit rate, 20 largest-\|return\| days | higher | +0.56 [+0.17, +0.78] | +0.56 [+0.18, +0.78] |
| hit rate, 50 largest-\|return\| days | higher | +0.36 [+0.13, +0.71] | +0.37 [+0.14, +0.72] |

Hit rate on the tail days is almost the P&L itself (a call on a day with |R| ≥ 3.02 wins or loses at least that many premiums), so its agreement is partly mechanical; the tail-day sign accuracy is the forecast-side version of the same quantity.

Paired differences of agreement with the Sharpe (mid), same draws (`rank_vs_sharpe_differences.csv`; positive = the first metric orders the forecasts more like the Sharpe):

| set | n | first metric | minus | difference [95 %] |
|---|---:|---|---|---|
| all forecasts | 220 | sign accuracy, 20 largest-\|return\| days | sign accuracy, all days | +0.03 [-0.28, +0.51] |
| all forecasts | 220 | sign accuracy, 20 largest-\|return\| days | sign accuracy, other 846 days | +0.09 [-0.26, +0.64] |
| all forecasts | 220 | sign accuracy, 20 largest-\|return\| days | QLIKE | +0.02 [-0.25, +0.54] |
| all forecasts | 220 | QLIKE | mean squared log error | +0.05 [-0.08, +0.11] |
| 48-bar (paper + pooled twins), all families | 18 | sign accuracy, 20 largest-\|return\| days | QLIKE | -0.28 [-0.68, +1.05] |
| per-bar linear, all families | 46 | sign accuracy, 20 largest-\|return\| days | QLIKE | +0.80 [-0.37, +1.22] |
| every Optuna rung | 81 | sign accuracy, 20 largest-\|return\| days | QLIKE | +0.00 [-0.47, +0.84] |
| per-bar trees, all families | 135 | sign accuracy, 20 largest-\|return\| days | QLIKE | +0.03 [-0.43, +0.84] |
| LSTM (per-bar and intraday sequence), all families | 12 | sign accuracy, 20 largest-\|return\| days | QLIKE | +1.11 [-0.30, +1.42] |
| mean over the 20 model families | 20 | sign accuracy, 20 largest-\|return\| days | QLIKE | +0.16 [-0.24, +0.71] |
| mean over the 12 tree rungs | 12 | sign accuracy, 20 largest-\|return\| days | QLIKE | -0.11 [-0.52, +0.76] |

(For the mean rows n is the number of families averaged.)

## Within classes of forecasts (figure `qlike_vs_sharpe.png` shows the classes)

| class | n | QLIKE | MSE (variance units) | sign accuracy, all days | QLIKE, 20 largest-\|return\| days | sign accuracy, other 846 days | sign accuracy, 20 largest-\|return\| days | sign accuracy, 50 largest-\|return\| days |
|---|---:|---|---|---|---|---|---|---|
| 48-bar (paper + pooled twins) | 18 | +0.37 [-0.46, +0.69] | -0.02 [-0.67, +0.59] | +0.21 [-0.50, +0.77] | +0.03 [-0.49, +0.79] | +0.15 [-0.56, +0.71] | +0.09 [-0.25, +0.82] | -0.17 [-0.52, +0.72] |
| per-bar linear | 46 | -0.08 [-0.46, +0.62] | -0.34 [-0.61, +0.48] | -0.03 [-0.29, +0.55] | +0.52 [-0.27, +0.69] | -0.12 [-0.35, +0.48] | +0.72 [-0.01, +0.86] | +0.62 [-0.13, +0.83] |
| per-bar linear, HAR-ladder variants | 9 | +0.80 [+0.47, +0.95] | -0.65 [-0.85, +0.88] | +0.61 [+0.33, +0.90] | +0.93 [+0.30, +0.95] | +0.62 [+0.28, +0.88] | +0.86 [-0.50, +0.95] | +0.89 [-0.54, +0.96] |
| per-bar trees | 135 | +0.37 [-0.16, +0.65] | -0.57 [-0.67, +0.39] | +0.57 [+0.00, +0.73] | -0.09 [-0.37, +0.65] | +0.51 [-0.12, +0.70] | +0.40 [+0.03, +0.79] | +0.28 [-0.09, +0.72] |
| LSTM (per-bar and intraday sequence) | 12 | -0.36 [-0.70, +0.72] | -0.59 [-0.80, +0.57] | +0.29 [-0.35, +0.80] | -0.13 [-0.61, +0.81] | +0.22 [-0.43, +0.76] | +0.76 [+0.04, +0.88] | +0.77 [-0.06, +0.90] |

## Within each model family (B4; figure `rank_vs_sharpe_by_model_family.png`)

Every master-table family with at least 5 forecasts, the pooled sets (a display class of several families; every Optuna rung together) and the mean of the within-family agreement over the families (its interval from the same draws). Rungs: T10 / T1 = the shipped tree configuration refit every 10 sessions / every session; RS10 / RS1 = random search (32 candidates, every 250 sessions) refit every 10 / every session; OP_tp<N>_k<k> = Optuna TPE tuned every N sessions, best of the first k of 50 trials, refit every session. Agreement with the Sharpe (mid), 95 % day-block interval.

| set | rung | n | QLIKE range | Sharpe (mid) range | QLIKE | sign accuracy, 20 largest-\|return\| days | sign accuracy, 50 largest-\|return\| days |
|---|---|---:|---|---|---|---|---|
| paper |  | 9 | 0.1006 – 0.1172 | 1.00 – 1.54 | +0.15 [-0.58, +0.75] | +0.54 [-0.24, +0.92] | +0.25 [-0.59, +0.85] |
| pooled twin |  | 9 | 0.1008 – 0.1106 | 0.99 – 1.32 | +0.43 [-0.73, +0.88] | -0.18 [-0.47, +0.94] | -0.44 [-0.67, +0.87] |
| 48-bar (paper + pooled twins), all families |  | 18 | 0.1006 – 0.1172 | 0.99 – 1.54 | +0.37 [-0.46, +0.69] | +0.09 [-0.25, +0.82] | -0.17 [-0.52, +0.72] |
| per-bar linear |  | 10 | 0.0964 – 0.1005 | 1.22 – 1.90 | -0.24 [-0.77, +0.83] | +0.76 [-0.15, +0.95] | +0.75 [-0.43, +0.94] |
| VIX-only family |  | 21 | 0.0954 – 0.1016 | 1.26 – 1.91 | -0.40 [-0.71, +0.74] | +0.84 [-0.11, +0.93] | +0.78 [-0.22, +0.91] |
| implied-vol representations |  | 15 | 0.0950 – 0.1001 | 0.97 – 1.87 | +0.27 [-0.35, +0.63] | +0.42 [-0.16, +0.76] | +0.51 [-0.14, +0.79] |
| per-bar linear, all families |  | 46 | 0.0950 – 0.1016 | 0.97 – 1.91 | -0.08 [-0.46, +0.62] | +0.72 [-0.01, +0.86] | +0.62 [-0.13, +0.83] |
| HAR-ladder variants |  | 9 | 0.0934 – 0.2184 | -0.02 – 1.79 | +0.80 [+0.47, +0.95] | +0.86 [-0.50, +0.95] | +0.89 [-0.54, +0.96] |
| per-bar tree (untuned) | T10 | 9 | 0.0991 – 0.1060 | 0.82 – 1.84 | +0.75 [-0.20, +0.90] | +0.28 [-0.31, +0.86] | +0.60 [-0.36, +0.87] |
| per-bar tree (untuned, daily refit) | T1 | 9 | 0.0955 – 0.1041 | 0.95 – 1.69 | +0.62 [-0.40, +0.93] | +0.30 [-0.35, +0.89] | +0.08 [-0.55, +0.85] |
| per-bar tree (tuned) | RS10 | 18 | 0.0992 – 0.1195 | 0.88 – 1.90 | +0.26 [-0.34, +0.62] | +0.53 [-0.04, +0.82] | +0.36 [-0.19, +0.76] |
| per-bar tree (random search, daily) | RS1 | 18 | 0.0994 – 0.1171 | 0.71 – 1.63 | +0.36 [-0.30, +0.69] | +0.01 [-0.32, +0.82] | +0.08 [-0.51, +0.75] |
| per-bar tree (Optuna, TUNE_PER=1, best-of-50) | OP_tp1_k50 | 9 | 0.1002 – 0.1086 | 0.60 – 1.41 | +0.48 [-0.52, +0.83] | +0.61 [-0.10, +0.95] | -0.03 [-0.38, +0.87] |
| per-bar tree (Optuna, TUNE_PER=5, best-of-50) | OP_tp5_k50 | 9 | 0.1006 – 0.1076 | 0.61 – 1.54 | +0.55 [-0.47, +0.87] | +0.28 [-0.42, +0.90] | +0.38 [-0.49, +0.89] |
| per-bar tree (Optuna, TUNE_PER=25, best-of-50) | OP_tp25_k50 | 18 | 0.0999 – 0.1097 | 0.60 – 1.65 | +0.35 [-0.37, +0.73] | +0.12 [-0.12, +0.83] | +0.30 [-0.17, +0.78] |
| per-bar tree (Optuna, TUNE_PER=250, best-of-50) | OP_tp250_k50 | 9 | 0.1024 – 0.1184 | 0.75 – 1.64 | +0.73 [-0.23, +0.85] | +0.45 [-0.52, +0.85] | +0.30 [-0.41, +0.75] |
| per-bar tree (Optuna, TUNE_PER=1, best-of-25) | OP_tp1_k25 | 9 | 0.1007 – 0.1086 | 0.69 – 1.71 | +0.60 [-0.38, +0.80] | +0.37 [-0.38, +0.93] | +0.08 [-0.52, +0.89] |
| per-bar tree (Optuna, TUNE_PER=25, best-of-25) | OP_tp25_k25 | 9 | 0.1030 – 0.1076 | 0.67 – 1.49 | +0.13 [-0.65, +0.77] | +0.47 [-0.11, +0.91] | +0.71 [-0.03, +0.91] |
| per-bar tree (Optuna, TUNE_PER=1, best-of-10) | OP_tp1_k10 | 9 | 0.1001 – 0.1089 | 0.73 – 1.66 | +0.43 [-0.43, +0.87] | +0.44 [-0.10, +0.91] | +0.21 [-0.32, +0.81] |
| per-bar tree (Optuna, TUNE_PER=25, best-of-10) | OP_tp25_k10 | 9 | 0.1003 – 0.1097 | 0.67 – 1.67 | +0.58 [-0.38, +0.82] | +0.62 [-0.13, +0.89] | +0.50 [-0.29, +0.83] |
| every Optuna rung | OP | 81 | 0.0999 – 0.1184 | 0.60 – 1.71 | +0.42 [-0.22, +0.66] | +0.42 [+0.01, +0.81] | +0.36 [-0.09, +0.74] |
| per-bar trees, all families |  | 135 | 0.0955 – 0.1195 | 0.60 – 1.90 | +0.37 [-0.16, +0.65] | +0.40 [+0.03, +0.79] | +0.28 [-0.09, +0.72] |
| LSTM |  | 6 | 0.1031 – 0.1305 | 0.77 – 1.33 | -0.60 [-0.94, +0.89] | +0.41 [-0.62, +0.96] | +0.44 [-0.68, +0.93] |
| LSTM (intraday sequence) |  | 6 | 0.1071 – 0.1125 | 0.81 – 1.71 | -0.43 [-0.77, +0.89] | +0.81 [-0.03, +0.94] | +0.81 [-0.03, +0.99] |
| LSTM (per-bar and intraday sequence), all families |  | 12 | 0.1031 – 0.1305 | 0.77 – 1.71 | -0.36 [-0.70, +0.72] | +0.76 [+0.04, +0.88] | +0.77 [-0.06, +0.90] |
| mean over the 20 model families |  | 20 families | 0.0934 – 0.2184 | -0.02 – 1.91 | +0.29 [-0.10, +0.52] | +0.45 [+0.13, +0.73] | +0.38 [+0.02, +0.66] |
| mean over the 12 tree rungs |  | 12 families | 0.0955 – 0.1195 | 0.60 – 1.90 | +0.49 [-0.15, +0.68] | +0.37 [-0.02, +0.78] | +0.30 [-0.13, +0.69] |

A family of 6–9 forecasts gives a rank correlation an interval of about ±0.7 on these days; the mean over the families pools them.

## Named forecasts

| forecast | QLIKE | sign accuracy | sign accuracy, 20 tail days | 50 tail days | hit rate | Sharpe mid / crossed | share of P&L on the 20 tail days |
|---|---:|---:|---:|---:|---:|---|---:|
| per-bar ridge [live_feasible] (`sub_ridge_live_feasible`) | 0.1005 | 67.4 % | 70 % | 68 % | 54.6 % | 1.90 / 1.44 | 34 % |
| per-bar lasso [live_feasible] (`sub_lasso_live_feasible`) | 0.0964 | 67.8 % | 70 % | 72 % | 55.0 % | 1.83 / 1.36 | 38 % |
| per-bar ridge [vix_only] (`sub_ridge_vix_only`) | 0.0959 | 68.9 % | 50 % | 58 % | 55.9 % | 1.48 / 1.00 | 11 % |
| per-bar LightGBM [live_feasible] (`subtree_lgbm_live_feasible`) | 0.0991 | 65.0 % | 50 % | 60 % | 54.5 % | 1.59 / 1.12 | 11 % |
| block-diagonal ridge (`blk2`) | 0.1058 | 67.2 % | 40 % | 58 % | 54.8 % | 1.00 / 0.53 | -9 % |
| per-bar elastic net [live_feasible], HAR ladder base3 (`sub_enet_live_feasible_har_base3`) | 0.0934 | 68.8 % | 60 % | 64 % | 54.4 % | 1.57 / 1.11 | 27 % |

## Per-day regressions (trade return, mid, on the forecast; HAC standard errors)

Newey–West lag 6 (the rule floor(4 (n/100)^(2/9)), n = 866); one regression per forecast (`per_day_regressions.csv`).

| regressor | R² median (range) | slope median | HAC t range | forecasts with p < 0.05 |
|---|---|---:|---|---:|
| sign right | 6.89 % (6.23–7.56 %) | +0.623 | +7.5 to +8.7 | 220 of 220 |
| log error | 0.12 % (0.00–1.02 %) | -0.068 | -2.4 to +1.9 | 2 of 220 |
| abs log error | 1.17 % (0.10–5.86 %) | -0.410 | -7.2 to -0.8 | 172 of 220 |
| log error + sign right | 6.96 % (6.24–8.74 %) | +0.024 | -1.2 to +2.6 | 4 of 220 |

(For the joint regression the slope and t are the log error's.)

## The claims re-checked against the 106-forecast run (commit 6177b74)

Every claim above is a check on the numbers (`claims.csv`). The same checks run on the committed outputs of the 106-forecast run (commit 6177b74, read with `git show`; that run had no model-family table and no paired differences, so those checks read 'not computed' there). 'asserted' = this script stops if the check fails; 'asserted at 6177b74' = the script of that commit did.

**Assertions of the 6177b74 script that fire on this table:** on the other 846 days the sign accuracy does not rank the forecasts (other-days sign-accuracy agreement interval covers 0: +0.50 [+0.01, +0.63]) — the prose now follows the check.

| claim | check | asserted (now / at 6177b74) | 106 forecasts | 220 forecasts (this run) |
|---|---|---|---|---|
| across all forecasts a lower QLIKE goes with a higher Sharpe | QLIKE agreement interval above 0 | yes / yes | holds: Spearman -0.69 [-0.75, -0.20] | holds: Spearman -0.56 [-0.65, -0.06] |
| but not among forecasts of similar accuracy (the per-bar linear class) | per-bar linear class: QLIKE agreement interval covers 0 | yes / yes | holds: 46 forecasts: -0.06 [-0.46, +0.62] | holds: 46 forecasts: -0.08 [-0.46, +0.62] |
| MSE in variance units orders the forecasts the wrong way round | MSE agreement < 0, interval covers 0 | yes / yes | holds: -0.49 [-0.61, +0.53] | holds: -0.51 [-0.61, +0.45] |
| the other accuracy losses do no better than QLIKE | agreement of squared log error and calibration error <= QLIKE's (point) | no / no | holds: QLIKE +0.69 [+0.20, +0.75]; squared log error +0.58 [+0.12, +0.72]; calibration +0.19 [-0.23, +0.53]; QLIKE minus squared log error, paired not computed | holds: QLIKE +0.56 [+0.06, +0.65]; squared log error +0.52 [+0.02, +0.64]; calibration -0.07 [-0.43, +0.36]; QLIKE minus squared log error, paired +0.05 [-0.08, +0.11] |
| the all-day sign accuracy ranks the forecasts | all-day sign-accuracy agreement interval excludes 0 | no / no | holds: +0.38 [+0.01, +0.64] | holds: +0.56 [+0.09, +0.67] |
| the 20-day tail sign accuracy ranks the forecasts | 20-day sign-accuracy agreement interval above 0 | yes / yes | holds: +0.69 [+0.17, +0.80] | holds: +0.59 [+0.19, +0.79] |
| the 20-day sign accuracy ranks them more strongly than the all-day one | 20-day agreement > all-day agreement (point) | yes / yes | holds: +0.69 vs +0.38 | holds: +0.59 vs +0.56 |
| ... and the paired difference (20-day minus all-day agreement) excludes 0 | paired-difference interval above 0 (same draws) | no / no | not computed | **does not hold**: +0.03 [-0.28, +0.51] |
| the 20-day sign accuracy ranks them more strongly than the other days' one | 20-day agreement > other-days agreement (point) | yes / no | holds: +0.69 vs +0.31; paired not computed | holds: +0.59 vs +0.50; paired +0.09 [-0.26, +0.64] |
| on the other 846 days the sign accuracy does not rank the forecasts | other-days sign-accuracy agreement interval covers 0 | no / yes | holds: +0.31 [-0.05, +0.58] | **does not hold**: +0.50 [+0.01, +0.63] |
| what tracks the trade is calling the side right on the few largest-move days (and not on the others) | rest_sign holds and tail_stronger_paired holds (or was not computed) | no / no | holds: other days +0.31 [-0.05, +0.58]; 20-day minus all-day not computed | **does not hold**: other days +0.50 [+0.01, +0.63]; 20-day minus all-day +0.03 [-0.28, +0.51] |
| within the per-bar linear class the 20-day sign accuracy is suggestive, its interval reaching zero | per-bar linear class: 20-day agreement > 0, interval covers 0 | no / no | holds: 46 forecasts: +0.70 [-0.02, +0.86] | holds: 46 forecasts: +0.72 [-0.01, +0.86] |
| per day, a right sign explains the P&L for every forecast; the log error only at the chance rate | sign indicator p < 0.05 for all 220; log error p < 0.05 for <= 10% of them | yes / yes | holds: sign 106 of 106; log error 2 of 106 | holds: sign 220 of 220; log error 2 of 220 |
| the log error explains less of the P&L than the sign for every forecast | max R² (log error) < min R² (sign indicator) | yes / yes | holds: 1.02 % < 6.23 % | holds: 1.02 % < 6.23 % |
| larger misses cost a little | median slope on the absolute log error < 0 | no / no | holds: median slope -0.33, 58 of 106 with p < 0.05 | holds: median slope -0.41, 172 of 220 with p < 0.05 |
| adding the log error to the sign indicator adds nothing | log error p < 0.05 in the joint regression for <= 10% of the forecasts | no / no | holds: 2 of 106; median R² 6.81 % vs 6.78 % | holds: 4 of 220; median R² 6.96 % vs 6.89 % |
| among the per-bar trees the QLIKE ordering holds | per-bar tree class: QLIKE agreement interval above 0 | no / no | holds: 27 forecasts: +0.67 [+0.01, +0.82] | **does not hold**: 135 forecasts: +0.37 [-0.16, +0.65] |
| the 48-bar forecasts forecast worse and trade worse than the per-bar ones | 48-bar median QLIKE > per-bar median and 48-bar median Sharpe < per-bar median | no / no | holds: median QLIKE 0.1039 vs 0.0998; median Sharpe 1.20 vs 1.46 | holds: median QLIKE 0.1039 vs 0.1032; median Sharpe 1.20 vs 1.33 |
| example: the VIX-only ridge forecasts better and trades worse than the headline | QLIKE below the headline's and Sharpe below the headline's | no / no | holds: QLIKE 0.0959 vs 0.1005; Sharpe 1.48 vs 1.90 | holds: QLIKE 0.0959 vs 0.1005; Sharpe 1.48 vs 1.90 |
| within the model families, on average, QLIKE orders the trade | mean over the families of the within-family agreement: interval above 0 | no / no | not computed | **does not hold**: +0.29 [-0.10, +0.52] |
| within the model families, on average, the 20-day sign accuracy orders the trade | mean over the families of the within-family agreement: interval above 0 | no / no | not computed | holds: +0.45 [+0.13, +0.73] |

**Statements that changed** (the check's outcome or the wording that follows from the numbers; old statement → new statement, each rendered by this script from its run's numbers; for the 106-forecast run these are the statements of its committed SUMMARY.md):

- *the all-day sign accuracy ranks the forecasts*: 106 forecasts: “ranks them only weakly (agreement +0.38 [+0.01, +0.64])” → 220 forecasts: “ranks them (agreement +0.56 [+0.09, +0.67])”.
- *the 20-day sign accuracy ranks them more strongly than the all-day one*: 106 forecasts: “and more strongly” → 220 forecasts: “and more strongly in the point estimate”.
- *on the other 846 days the sign accuracy does not rank the forecasts*: 106 forecasts: “on the other 846 days it does not (+0.31 [-0.05, +0.58])” → 220 forecasts: “on the other 846 days it does as well, less strongly in the point estimate (+0.50 [+0.01, +0.63]; paired difference of the 20-day agreement over it +0.09 [-0.26, +0.64])”.
- *what tracks the trade is calling the side right on the few largest-move days (and not on the others)*: 106 forecasts: “What does track the trade is calling the side right on the few days the straddle moves most” → 220 forecasts: “Calling the side right tracks the trade: on the few days the straddle moves most and, across these 220 forecasts, on the other days as well”.
- *among the per-bar trees the QLIKE ordering holds*: 106 forecasts: “holds as well” → 220 forecasts: “points the same way”.
- *display classes*: 106 forecasts: 48-bar (paper + pooled twins) 18, per-bar linear 46, per-bar linear, HAR-ladder variants 9, per-bar trees 27, other 6 → 220 forecasts: 48-bar (paper + pooled twins) 18, per-bar linear 46, per-bar linear, HAR-ladder variants 9, per-bar trees 135, LSTM (per-bar and intraday sequence) 12 (the LSTM rows, 'other' before, are a class of their own; every new tree rung joins the per-bar trees).

## Notes

* The headline forecast `sub_ridge_live_feasible`: QLIKE 0.1005, Sharpe 1.90 mid / 1.44 crossed, the master table's numbers.
* The intervals are percentile intervals of the bootstrap distribution of the rank correlation; resampling days adds noise to every forecast's metrics, which pulls the resampled correlations toward zero, so the point estimate can sit near one end of its interval.
* The classes are the master table's families grouped for the figure: 48-bar = the paper's forecasts and the pooled twins; per-bar linear = per-bar ridge / lasso / elastic net on every bucket, the VIX-only family and the implied-vol representations; HAR-ladder variants apart (3 of them build the ladder within the session and sit at QLIKE 0.21–0.22, drawn at the edge of the figure); per-bar trees = LightGBM / XGBoost / random forest on every rung of the 16:00 ladder (T10, T1, RS10, RS1, the Optuna rungs); LSTM = the per-bar LSTM and the intraday-sequence LSTM.
