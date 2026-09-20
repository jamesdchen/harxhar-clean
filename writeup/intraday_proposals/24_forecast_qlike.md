# 24. Forecast QLIKE on the scored 2020–2024 bars

10,387 bars, 866 days. Patton QLIKE \(y/f-\log(y/f)-1\). Naive = lagged
expanding per-clock mean of RV (same history as \(w\)). Slice =
\(w\,\mathrm{IV}^2_{\mathrm{hr}} h\).

## Pooled

| forecast | QLIKE | corr(RV, f) | mean RV / mean f |
|---|---|---|---|
| implied slice | **0.181** | **0.86** | 0.80 |
| block-diagonal ridge | **0.182** | 0.79 | 0.91 |
| lasso (fixed \(10^{-4}\)) | 0.182 | 0.80 | 0.92 |
| LightGBM | 0.189 | 0.81 | 0.88 |
| naive clock mean | 0.792 | 0.10 | 1.02 |

Ridge **crushes** the naive (0.18 vs 0.79). The 2020–2024 forecasts are
not broken. They do **not** beat the implied slice as a predictor of
next-bar RV (slice corr 0.86 vs ridge 0.79). \(\mathrm{sign}(s)\) is
exactly forecast minus that slice.

## Body vs close

| | 10:00–15:00 | 15:30 |
|---|---|---|
| ridge QLIKE | 0.188 | **0.110** |
| naive | 0.782 | 0.901 |
| slice | 0.188 | 0.104 |

The close bar is the easy QLIKE clock. Daytime ridge is still far above
naive. 14:00 is the ugly clock (ridge 0.48 vs slice 0.20).

## Why daytime \(\mathrm{sign}(s)\) is not a 15:30 Sharpe

The forecast is doing its job vs a clock mean. The comparison object is
a constructed slice that already tracks next-bar RV. Residual
\(\widehat{RV}-\mathrm{slice}\) is then a small, noisy gap, and always-short
already harvests the rich remaining implied. That is 00/11, not a 2020–24
QLIKE failure.

Files: `24_forecast_qlike.{py,md}`, `results/atm_straddle_intraday/proposals/24/`.
