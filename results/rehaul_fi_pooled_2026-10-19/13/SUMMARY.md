# Variable importance, pooled intraday lasso and ridge, 13 regular-hours bars, monthly re-estimation (note E19)

Sample: the E15 variant on the 13 regular-hours bars (bar-end stamps 10:00 .. 16:00 ET), window W = 26,000 intervals
(2,000 trading days). Design and penalties: those of the 13-bar pooled linear rerun, branch
`insurance-selling-2026-09-19-pooled-linear-13` (commit 219ec7d): base-2 HAR ladder 1 .. 2048, no FOMC series, no
trailing averages of the activity indicators (E6), calendar = five weekday dummies and the hour (E9, overnight
indicator dropped on this sample), every data-fix flag on; p = 1,050 predictors in 45 groups (43 exogenous series in
7 categories, the HAR components, the calendar dummies). Frozen pre-sample penalties (`frozen_penalties_13.json`, a
copy of `results/pooled_linear_rerun_2026-10-19_13/frozen_penalties.json` on that branch): lasso lambda = 1e-3,
ridge lambda = 1e3 (scikit-learn parametrization), penalty weight omega = 0.01 on the HAR benchmark block (the 12 HAR
components and the 6 calendar dummies), 1 on the exogenous predictors.

Code: `experiments/rehaul_fi_pooled_compute.py` (stages `linear`, `gate`, `reduce`); the design cache was built
with the 13-bar branch's own runner (`experiments/pool_linear_pretune.py design --bucket all_features --sample
rth13 --har-base 2 --drop-fomc --ex-flag DROP_ACTIVITY_MA=1 --ex-flag CALENDAR_THROUGH_OVERNIGHT=1 --ex-flag
DROP_OVERNIGHT_INDICATOR=1 --ex-flag LEAN_GLOBAL_LAGS=1`, 84,747 x 1,050, not committed: 0.7 GB).

## Method

- Re-estimation k = 1, ..., K, K = 52: one at the first trading day of each calendar month from 3 January 2020 to
  1 April 2024 (span of Table tab:rh_oos, 3 January 2020 .. 30 April 2024). Re-estimation k is fit on the W = 26,000
  sample rows before the 10:00 bar of its first day t_k and scored on its out-of-sample block O_k, the bars of that
  month (14,262 bars in all).
- Estimator: the batch solution at the frozen penalty (FWL on the intercept, then the penalized fit; the lasso
  through the homotopy with penalty weights, certified by the KKT conditions: 70 of 70 lasso fits repaired to an
  exact KKT point, none failed), with the spec's identifiability mask (columns constant or byte-identical to an
  earlier column inside the window get a zero coefficient). This is the pooled walk's cold seed. Gate
  (`gates.csv`): at the runner's cold-seed row 70,250 this fit reproduces the runner's own walk forecast, lasso
  exactly and ridge to 3e-12.
- Measures: the closing-interval bullets (writeup/rehaul/08_feature_importance.tex:51-325), on the pooled model's
  coefficients. Random probes at every third re-estimation (18 fits), as in experiments/rehaul_fi_compute.py:
  continuous U(0,1), 20-level discrete, binary; robust-scaled on the window, omega = 1, not masked.

## Numbers

Lasso, over the sample

- Nonzero coefficients at a re-estimation: median 124.5 (range 111 .. 131), of which 107.5 on exogenous predictors.
- Predictors: 270 of 1,050 selected at least once, 107 in at least half of the re-estimations, 49 in every one.
- Exogenous series: 43 of 43 selected at least once; 41 in at least half of the re-estimations; 24 in every one
  (sumret, sumret3, sumret4, sumautocov, sumvolume, effspread_ewstock, spread_ewstock, effspread_vwstock,
  spread_vwstock, ofi_vwstock, sumret3_ewstock, sumabsret_ewstock, sumpret2_ewstock, sumret4_vwstock,
  sumabsret_vwstock, sumpret2_vwstock, stocktwits_attention, stocktwits_sentiment, vix, vvix, vix3m,
  voldemand_spx_open_and_close, voldemand_all_open_and_close, voldemand_all_open_only). Least often selected:
  sumbipow_vwstock (K_A = 9 of 52), voldemand_spx_open_only (24), sumret4_ewstock (28).
- HAR benchmark block: selected in every re-estimation; 37.9 % of sum_j |beta_{k,j}| (median re-estimation). The HAR
  components alone hold S_A = 31.9 % of the lasso's absolute coefficient mass over the sample, the calendar dummies
  6.2 %.
- Random probes: the continuous probe is selected in 5 of 18 fits (28 %), the 20-level probe in 5, the binary probe
  in 6. 154 predictors, in 41 exogenous series, are selected more often than the continuous probe.

Lasso, over time

- Mean rank by ||beta_{k,A}||_1 (`lasso_series.csv`): HAR components 1.0, sumret 2.4, vix 3.1, vix3m 4.1, calendar
  dummies 4.4, vvix 7.3, sumret4 7.4, sumabsret_ewstock 9.3, sumautocov 9.3, stocktwits_sentiment 10.1.
- Kendall's W of the 52 group rankings: 0.90. Magnitudes and on/off indicators at each re-estimation:
  `lasso_series_by_refit.csv`, `lasso_active_by_refit.csv`, shares `lasso_share_by_refit.csv`.

Ridge, over the sample

- Leading groups by S_A: HAR components 13.1 %, vix 6.0 %, vvix 4.2 %, sumret 3.8 %, vix3m 3.3 %, calendar dummies
  3.3 %; HAR benchmark block 16.4 % (median re-estimation).
- Scale check (only the averages of series values are robust-scaled): averages of series values, 516 predictors,
  window s.d. median 0.93 (5th .. 95th percentile 0.35 .. 2.06), interquartile range 0.98; averaged missing-data
  indicators, 516 predictors, s.d. 0.079 where not constant (418 of them are constant in the median window and
  masked); calendar dummies 0.40; HAR components 1.15.
- Standardized share S^sd_A: HAR components 15.0 %, sumret4 4.8 %, sumabsret 4.4 %, vix 4.2 %, sumpret2 3.9 %,
  sumret 3.5 %. Spearman rank correlation of S^sd_A with S_A across groups: 0.67 (lasso: 0.96).
- Ridge sets no coefficient to zero: all 45 groups have K_A = K (the 418 masked columns aside).
- Random probes: 424 predictors, in 43 exogenous series, have |beta_{k,j}| above the continuous probe's in most of
  the 18 probe fits.

Ridge, over time

- Mean rank by ||beta_{k,A}||_1: HAR components 1.0, vix 2.0, vvix 3.3, sumret 3.9, vix3m 5.9, calendar dummies 6.1,
  voldemand_spx_open_and_close 6.9, sumabsret 8.9, spread_vwstock 9.0, stocktwits_sentiment 10.8.
- Kendall's W: 0.92. S_{k,A}: `ridge_share_by_refit.csv`.
- Lasso vs ridge mean rank, Spearman rank correlation over the 45 groups: 0.81.

Forecast check on the scored blocks: median block MSE on the adjusted scale 0.065 (lasso), 0.066 (ridge).

## Compute

4 CPUs, 15 GB. Design cache: 4 minutes. Linear stage: 5.7 minutes (lasso 3.1 s, ridge 0.2 s a fit; probe fits 1.6 s),
reduce: 1 s.

## Files

`summary.json` (every number above), `lasso_series.csv` / `ridge_series.csv` (groups: mean rank, Borda count,
K_A, S_A, S^sd_A, ...), `*_columns.csv` (predictors: K_j, sum |beta|, sum |beta| sd), `*_series_by_refit.csv`
(||beta_{k,A}||_1), `*_share_by_refit.csv` (S_{k,A}), `*_active_by_refit.csv` (on/off), `*_probes.csv`,
`column_scales.csv`, `linear.npz` (coefficients, window s.d. and IQR, masks), `run.json`, `gates.csv`.

## Deviations from the closing-interval method

1. Cadence: monthly (decision 1) instead of every 10 trading days; K = 52.
2. Coefficients: the batch fit on each re-estimation's window, with the identifiability mask recomputed on that
   window. The closing-interval analysis read the walk's coefficient vector at the anchor (re-solved every day, mask
   refreshed every 250 solves). Both solve the same problem; they can differ only through the mask timing.
3. Gate: against the runner's walk at one cold-seed row (its walk forecasts are not committed); the closing-interval
   gate compared every forecast with the stored walk.
4. Not reproduced: the reconciliation with the earlier study's single-penalty lasso (no pooled counterpart).
5. Scale kinds: the "flag" kind holds the missing-data indicator averages only (E6 dropped the activity averages).
6. Probes: drawn with the same generator and seed, indexed by the monthly k, so the draws differ from the
   closing-interval ones.

## Open questions

- Masked columns: in the median window 418 of the 516 averaged missing-data indicators are constant (the series is
  live on every row of the window), so their coefficients are zero by construction and they count as "not selected"
  in K_j. The group counts K_A are unaffected (each group also has its value averages).
- The 13-bar design has no overnight indicator (DROP_OVERNIGHT_INDICATOR, as the rerun); the calendar group is the
  weekday dummies and the hour.
- The penalty picks were read from the 13-bar branch as committed at 219ec7d; if that rerun revises them, this run
  must be repeated (5.7 minutes).
