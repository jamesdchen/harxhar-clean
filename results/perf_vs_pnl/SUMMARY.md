# Forecast accuracy vs P&L of the 15:30 last-30-min trade (checklist C3)

Written by `experiments/perf_vs_pnl_1530.py` from its own outputs in this folder (`forecast_metrics.csv`, `rank_vs_sharpe.csv`, `rank_vs_sharpe_by_family.csv`, `per_day_regressions.csv`, `numbers.json`); the claims the verdict rests on are asserted against those numbers before this file is written.

**Universe:** 106 forecasts of the bar ending 16:00: the table-A rows of the closing-strategy master table (`results/close_master_table/master_table.csv`, `master_table_daily.parquet`; check rows, exact duplicates and the always-short rule left out). By class: 48-bar (paper + pooled twins) 18, per-bar linear 46, per-bar linear, HAR-ladder variants 9, per-bar trees 27, other 6. Same 866 trade days for every forecast (2020-01-03 .. 2024-04-30).

**One scorer, the research scorer:** the 16:00 bar recalibrated alone (forecast = (ŷ² + s)·B, s = the forecast's own trailing-250-session mean squared error, lagged one session); every loss is against the per-bar spec's 16:00 target. The trade is the deck's 15:30 **sign(s)** rule: buy the **straddle** (nearest out-of-the-money call + nearest out-of-the-money put, same-day expiry, one position) when the forecast exceeds the 15:30 implied variance, sell it otherwise, hold to the close; Sharpe annualized with √252, mid and crossed fills. The per-day trade and QLIKE reproduce the master table's Sharpe and QLIKE to 5e-15.

**Metrics per forecast:** QLIKE; MSE (variance units); mean squared log error; calibration error |b − 1| with b the slope of log realized on log forecast; **sign accuracy** = the share of days on which sign(forecast − implied) equals sign(realized − implied); hit rate = the share of days the trade made money; the same on the **tail days** = the 20 / 50 trade days with the largest |straddle return| (|R| ≥ 3.02 / 1.91 premium, against a median of 0.79; the same days for every forecast). **Rank agreement** = Spearman correlation across the forecasts between a metric and the Sharpe, signed so that +1 means the metric orders the forecasts exactly as the Sharpe does; 95 % intervals from a circular block bootstrap over days (block 21 sessions, 2,000 draws; the whole panel of forecasts resampled together).

## Answer in brief

1. **Across all 106 forecasts a lower QLIKE goes with a higher Sharpe** (Spearman -0.69, interval [-0.75, -0.20]), **but not among forecasts of similar accuracy.** Within the 46 per-bar linear forecasts (all buckets, the VIX-only and implied-vol families) the rank agreement of QLIKE with the Sharpe is -0.06 [-0.46, +0.62]; within the 27 per-bar trees +0.67 [+0.01, +0.82]. The best-QLIKE forecast (per-bar elastic net [live_feasible], HAR ladder base3, QLIKE 0.0934) ranks 38 of 106 on the Sharpe (1.57); the best-Sharpe forecast (per-bar lasso [free_vix_only], 1.91) ranks 30 of 106 on QLIKE.
2. **The other accuracy losses do no better.** MSE in variance units orders the forecasts the wrong way round (agreement -0.49 [-0.61, +0.53]); the squared log error +0.58 [+0.12, +0.72]; the calibration error +0.19 [-0.23, +0.53].
3. **What does track the trade is calling the side right on the few days the straddle moves most.** Sign accuracy over all days hardly separates the forecasts (50–71 %, median 67 %) and ranks them only weakly (agreement +0.38 [+0.01, +0.64]). On the 20 largest-|return| days it does, and more strongly (+0.69 [+0.17, +0.80]; 50 days +0.58 [+0.02, +0.77]), and on the other 846 days it does not (+0.31 [-0.05, +0.58]). Within the per-bar linear class, where QLIKE says nothing, the 20-day sign accuracy has agreement +0.70 [-0.02, +0.86]: suggestive, with an interval that reaches zero.
4. **Per day, being on the right side explains the P&L; the size of the forecast error does not.** Regressing the day's trade return on an indicator of a right sign gives R² 6.8 % (median; 6.2–7.6 % across the 106 forecasts; HAC t 7.5 to 8.6, every forecast significant). On the forecast's log error it gives R² 0.13 % (median; 2 of 106 with p < 0.05, the rate chance gives); on its absolute log error 0.69 % (median slope -0.33, 58 of 106 with p < 0.05: larger misses cost a little). Adding the log error to the sign indicator adds nothing (median R² 6.8 %). For the headline per-bar ridge (`sub_ridge_live_feasible`) the sign is right on 584 of 866 days (mean return +0.33 per unit premium) and wrong on 282 (-0.27); its 20 tail days carry 34 % of its total P&L (50 days 43 %).

**Verdict.** QLIKE is the right loss for forecasting the 15:30–16:00 variance, and across model families a better QLIKE comes with a better trade: the 48-bar forecasts forecast worse and trade worse than the per-bar ones, and among the per-bar trees the ordering holds as well (+0.67 [+0.01, +0.82]). Among the 46 per-bar linear forecasts, the candidates for the headline, it does not (-0.06 [-0.46, +0.62]): there a QLIKE gain does not imply a trade gain (e.g. the per-bar ridge on the VIX-only bucket: QLIKE 0.0959 against the headline's 0.1005, Sharpe 1.48 against 1.90). The trade is decided by the sign of forecast minus implied on a handful of large-move days (the 20 largest carry 23 % of total P&L for the median forecast), and QLIKE, an average over all 866 days of how far the forecast misses, gives them little weight (they make up 4 % of the median forecast's summed QLIKE). The trade's own measure (sign accuracy weighted by the size of the move, i.e. the P&L) is the relevant one for choosing among comparable forecasts, and on 866 days it is too noisy to rank them reliably.

## Rank agreement with the Sharpe, all forecasts

| metric | better | agreement with Sharpe (mid) | agreement with Sharpe (crossed) |
|---|---|---|---|
| QLIKE | lower | +0.69 [+0.20, +0.75] | +0.69 [+0.20, +0.75] |
| MSE (variance units) | lower | -0.49 [-0.61, +0.53] | -0.50 [-0.62, +0.53] |
| mean squared log error | lower | +0.58 [+0.12, +0.72] | +0.57 [+0.12, +0.71] |
| calibration error \|log MZ slope - 1\| | lower | +0.19 [-0.23, +0.53] | +0.19 [-0.24, +0.53] |
| sign accuracy, all days | higher | +0.38 [+0.01, +0.64] | +0.37 [+0.00, +0.64] |
| hit rate (trade made money), all days | higher | +0.55 [+0.39, +0.81] | +0.54 [+0.38, +0.81] |
| QLIKE, 20 largest-\|return\| days | lower | +0.74 [+0.15, +0.78] | +0.74 [+0.15, +0.78] |
| QLIKE, other 846 days | lower | +0.67 [+0.17, +0.73] | +0.66 [+0.17, +0.73] |
| QLIKE, 50 largest-\|return\| days | lower | +0.72 [+0.15, +0.75] | +0.72 [+0.15, +0.76] |
| QLIKE, other 816 days | lower | +0.64 [+0.16, +0.73] | +0.64 [+0.16, +0.73] |
| sign accuracy, 20 largest-\|return\| days | higher | +0.69 [+0.17, +0.80] | +0.69 [+0.17, +0.80] |
| sign accuracy, other 846 days | higher | +0.31 [-0.05, +0.58] | +0.30 [-0.05, +0.58] |
| sign accuracy, 50 largest-\|return\| days | higher | +0.58 [+0.02, +0.77] | +0.58 [+0.03, +0.77] |
| sign accuracy, other 816 days | higher | +0.29 [-0.07, +0.56] | +0.28 [-0.08, +0.55] |
| hit rate, 20 largest-\|return\| days | higher | +0.62 [+0.12, +0.79] | +0.63 [+0.13, +0.79] |
| hit rate, 50 largest-\|return\| days | higher | +0.38 [+0.09, +0.72] | +0.39 [+0.10, +0.73] |

Hit rate on the tail days is almost the P&L itself (a call on a day with |R| ≥ 3.02 wins or loses at least that many premiums), so its agreement is partly mechanical; the tail-day sign accuracy is the forecast-side version of the same quantity.

## Within classes of forecasts (figure `qlike_vs_sharpe.png` shows the classes)

| class | n | QLIKE | MSE (variance units) | sign accuracy, all days | QLIKE, 20 largest-\|return\| days | sign accuracy, other 846 days | sign accuracy, 20 largest-\|return\| days | sign accuracy, 50 largest-\|return\| days |
|---|---:|---|---|---|---|---|---|---|
| 48-bar (paper + pooled twins) | 18 | +0.37 [-0.46, +0.69] | -0.02 [-0.67, +0.59] | +0.21 [-0.50, +0.77] | +0.03 [-0.49, +0.79] | +0.15 [-0.56, +0.71] | +0.09 [-0.25, +0.82] | -0.17 [-0.52, +0.72] |
| per-bar linear | 46 | -0.06 [-0.46, +0.62] | -0.32 [-0.61, +0.48] | -0.02 [-0.28, +0.54] | +0.52 [-0.27, +0.69] | -0.12 [-0.35, +0.48] | +0.70 [-0.02, +0.86] | +0.60 [-0.14, +0.83] |
| per-bar linear, HAR-ladder variants | 9 | +0.80 [+0.47, +0.95] | -0.65 [-0.85, +0.88] | +0.59 [+0.30, +0.90] | +0.93 [+0.30, +0.95] | +0.59 [+0.28, +0.88] | +0.86 [-0.50, +0.95] | +0.89 [-0.54, +0.97] |
| per-bar trees | 27 | +0.67 [+0.01, +0.82] | -0.66 [-0.78, +0.58] | +0.53 [-0.16, +0.79] | +0.31 [-0.20, +0.76] | +0.48 [-0.21, +0.77] | +0.27 [-0.25, +0.80] | +0.06 [-0.34, +0.77] |
| other | 6 | +0.60 [-0.77, +0.94] | +0.60 [-0.77, +0.94] | +0.67 [-0.66, +0.94] | +0.66 [-0.77, +0.94] | +0.62 [-0.72, +0.94] | +0.51 [-0.62, +0.96] | +0.24 [-0.64, +0.94] |

## Named forecasts

| forecast | QLIKE | sign accuracy | sign accuracy, 20 tail days | 50 tail days | hit rate | Sharpe mid / crossed | share of P&L on the 20 tail days |
|---|---:|---:|---:|---:|---:|---|---:|
| per-bar ridge [live_feasible] (`sub_ridge_live_feasible`) | 0.1005 | 67.4 % | 70 % | 68 % | 54.6 % | 1.90 / 1.44 | 34 % |
| per-bar lasso [live_feasible] (`sub_lasso_live_feasible`) | 0.0964 | 67.8 % | 70 % | 72 % | 55.0 % | 1.83 / 1.36 | 38 % |
| per-bar ridge [vix_only] (`sub_ridge_vix_only`) | 0.0959 | 68.9 % | 50 % | 58 % | 55.9 % | 1.48 / 1.00 | 11 % |
| per-bar LightGBM [live_feasible] (`subtree_lgbm_live_feasible`) | 0.0982 | 66.2 % | 55 % | 60 % | 54.5 % | 1.69 / 1.22 | 17 % |
| block-diagonal ridge (`blk2`) | 0.1058 | 67.2 % | 40 % | 58 % | 54.8 % | 1.00 / 0.53 | -9 % |
| per-bar elastic net [live_feasible], HAR ladder base3 (`sub_enet_live_feasible_har_base3`) | 0.0934 | 68.8 % | 60 % | 64 % | 54.4 % | 1.57 / 1.11 | 27 % |

## Per-day regressions (trade return, mid, on the forecast; HAC standard errors)

Newey–West lag 6 (the rule floor(4 (n/100)^(2/9)), n = 866); one regression per forecast (`per_day_regressions.csv`).

| regressor | R² median (range) | slope median | HAC t range | forecasts with p < 0.05 |
|---|---|---:|---|---:|
| sign right | 6.78 % (6.23–7.58 %) | +0.621 | +7.5 to +8.6 | 106 of 106 |
| log error | 0.13 % (0.00–1.02 %) | -0.081 | -2.4 to +1.5 | 2 of 106 |
| abs log error | 0.69 % (0.10–4.71 %) | -0.333 | -7.2 to -0.8 | 58 of 106 |
| log error + sign right | 6.81 % (6.24–8.83 %) | +0.007 | -1.2 to +2.6 | 2 of 106 |

(For the joint regression the slope and t are the log error's.)

## Notes

* The headline forecast `sub_ridge_live_feasible`: QLIKE 0.1005, Sharpe 1.90 mid / 1.44 crossed, the master table's numbers.
* The intervals are percentile intervals of the bootstrap distribution of the rank correlation; resampling days adds noise to every forecast's metrics, which pulls the resampled correlations toward zero, so the point estimate can sit near one end of its interval.
* The classes are the master table's families grouped for the figure: 48-bar = the paper's forecasts and the pooled twins; per-bar linear = per-bar ridge / lasso / elastic net on every bucket, the VIX-only family and the implied-vol representations; HAR-ladder variants apart (3 of them build the ladder within the session and sit at QLIKE 0.21–0.22, drawn at the edge of the figure); per-bar trees = untuned and tuned LightGBM / XGBoost / random forest.
