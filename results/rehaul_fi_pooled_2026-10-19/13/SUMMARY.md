# Variable importance, pooled intraday lasso and ridge, 13 regular-hours bars, monthly re-estimation (note E19)

Sample: the E15 variant on the 13 regular-hours bars (bar-end stamps 10:00 .. 16:00 ET), window W = 26,000 intervals
(2,000 trading days). Design and penalties: those of the 13-bar pooled linear rerun after the author's correction,
branch `insurance-selling-2026-09-19-pooled-linear-13` at b93aa67 (penalties committed in f30ad70): base-5 HAR ladder
(1, 5, 25, 125, 625, 3125 intervals), the four FOMC series in, no trailing averages of the activity indicators (E6),
calendar = five weekday dummies and the hour (E9; the overnight indicator dropped on this sample), every data-fix
flag on. p = 576 predictors in 49 groups: 47 exogenous series in 8 categories, the HAR components, the calendar
dummies. Frozen pre-sample penalties (`frozen_penalties_13.json`, a copy of
`results/pooled_linear_rerun_2026-10-19_13/frozen_penalties.json` on that branch): lasso lambda = 1e-3, ridge
lambda = 1e3 (scikit-learn parametrization); penalty weight omega = 0.01 on the HAR benchmark block (the 6 HAR
components and the 6 calendar dummies), 1 on the exogenous predictors. The file passes the author's landing rule
(HAR columns exactly the base-5 ladder, FOMC series present), checked in code (`check_landed`).

Code: `experiments/rehaul_fi_pooled_compute.py` (stages `linear`, `gate`, `reduce`). Design cache built with the
13-bar branch's own runner, `experiments/pool_linear_pretune.py design --bucket all_features` with the rerun's
configuration (`--sample rth13 --har-base 5 --ex-flag DROP_ACTIVITY_MA=1 --ex-flag CALENDAR_THROUGH_OVERNIGHT=1
--ex-flag DROP_OVERNIGHT_INDICATOR=1 --ex-flag LEAN_GLOBAL_LAGS=1`): 84,747 x 576, columns identical to the penalty
file's list; not committed (0.4 GB).

## Method

- Re-estimation k = 1, ..., K, K = 52: one at the first trading day of each calendar month from 3 January 2020 to
  1 April 2024 (the span of Table tab:rh_oos, 3 January 2020 .. 30 April 2024). Re-estimation k is fit on the
  W = 26,000 sample rows before the 10:00 bar of its first day t_k and scored on its out-of-sample block O_k, the bars
  of that month (14,262 bars in all).
- Estimator: the batch solution at the frozen penalty (FWL on the intercept, then the penalized fit; the lasso by the
  homotopy with penalty weights and a KKT certificate: 70 lasso fits, 5 exact as solved, 64 after support repair,
  1 after coordinate descent, none failed), with the spec's identifiability mask (columns constant or byte-identical to
  an earlier column inside the window get a zero coefficient). This is the pooled walk's cold seed.
- Gates. (1) `gates_runner.csv`: the 13-bar branch's runner, walked on the same design cache from its first cold-seed
  row (70,250), gives the same forecast as this fit: lasso to 1e-16, ridge to 1e-12. (2) `gates_forecasts.csv`: at
  all 58 cold-seed rows of the rerun's committed walk (`forecasts.parquet`), lasso to 1e-13, ridge to 2.4e-6
  (relative tolerance 1e-5). See Open questions for the ridge gap.
- Measures: the closing-interval bullets (writeup/rehaul/08_feature_importance.tex:51-325) on the pooled model's
  coefficients. Random probes at every third re-estimation (18 fits), as experiments/rehaul_fi_compute.py: continuous
  U(0,1), 20-level discrete, binary; robust-scaled on the window, omega = 1, not masked.

## Numbers

Lasso, over the sample

- Nonzero coefficients at a re-estimation: median 94.5 (range 80 .. 104), of which 83.5 on exogenous predictors.
- Predictors: 177 of 576 selected at least once, 85 in at least half of the re-estimations, 44 in every one.
- Exogenous series: 45 of 47 selected at least once (never: fomc_release, sumbipow_vwstock); 39 in at least half of
  the re-estimations; 27 in every one (sumret, sumabsret, sumret3, sumret4, sumautocov, sumvolume, numobs,
  effspread_ewstock, spread_ewstock, effspread_vwstock, spread_vwstock, ofi_vwstock, sumabsret_ewstock,
  sumpret2_ewstock, sumret4_vwstock, sumabsret_vwstock, sumpret2_vwstock, stocktwits_attention, stocktwits_sentiment,
  vix, vvix, vix3m, voldemand_spx_open_and_close, voldemand_all_open_and_close, voldemand_all_open_only, fomc_day,
  fomc_until_inv). Rarely selected: sumpret2 (K_A = 1 of 52), turnover_ewstock (5), ofi_ewstock (12), sumbipow (16).
- HAR benchmark block: selected in every re-estimation; 36.5 % of sum_j |beta_{k,j}| (median re-estimation). Over the
  sample the HAR components hold S_A = 31.5 % of the lasso's absolute coefficient mass, the calendar dummies 4.7 %.
- Random probes: the continuous probe is selected in 4 of 18 fits (22 %), the 20-level probe in 4, the binary probe in
  6. 130 predictors, in 42 exogenous series, are selected more often than the continuous probe.

Lasso, over time

- Mean rank by ||beta_{k,A}||_1 (`lasso_series.csv`): HAR components 1.0, sumret 2.4, vix 2.8, vix3m 4.3, calendar
  dummies 4.7, stocktwits_sentiment 7.3, sumret4 7.8, fomc_day 8.3, vvix 9.7, sumabsret_vwstock 10.6.
- Kendall's W of the 52 group rankings: 0.90. Magnitudes and on/off indicators at each re-estimation:
  `lasso_series_by_refit.csv`, `lasso_active_by_refit.csv`; shares `lasso_share_by_refit.csv`.

Ridge, over the sample

- Leading groups by S_A: HAR components 15.9 %, vix 4.9 %, sumret 4.7 %, vvix 4.2 %, fomc_until_inv 3.1 %; calendar
  dummies 2.8 %; HAR benchmark block 19.2 % (median re-estimation).
- Scale check (only the averages of series values are robust-scaled): averages of series values, 282 predictors,
  window s.d. median 0.90 (5th .. 95th percentile 0.17 .. 2.27), interquartile range 0.97; averaged missing-data
  indicators, 282 predictors, s.d. 0.059 where not constant (233 of them constant, hence masked, in the median
  window); calendar dummies 0.40; HAR components 1.11.
- Standardized share S^sd_A: HAR components 18.1 %, sumret 4.5 %, sumabsret 4.4 %, sumret4 4.4 %, vix 3.5 %, vvix
  3.1 %. Spearman rank correlation of S^sd_A with S_A across groups: 0.65 (lasso: 0.95).
- Ridge sets no unmasked coefficient to zero: all 49 groups have K_A = K.
- Random probes: 256 predictors, in all 47 exogenous series, have |beta_{k,j}| above the continuous probe's in most
  of the 18 probe fits.

Ridge, over time

- Mean rank by ||beta_{k,A}||_1: HAR components 1.0, vix 2.4, sumret 2.6, vvix 4.0, fomc_until_inv 6.7,
  voldemand_spx_open_and_close 6.8, vix3m 7.2, calendar dummies 8.4, sumabsret 8.8, stocktwits_sentiment 8.8.
- Kendall's W: 0.91. S_{k,A}: `ridge_share_by_refit.csv`.
- Lasso vs ridge mean rank, Spearman rank correlation over the 49 groups: 0.79.

FOMC series: fomc_day and fomc_until_inv are selected by the lasso in every re-estimation, fomc_since_inv in 48 of
52, fomc_release in none. In ridge, fomc_until_inv ranks fifth (S_A 3.1 %) and fomc_day twelfth (2.3 %); on the
standardized scale both are small (S^sd_A 2.8 % and 0.7 %), as the averages of an event indicator have a small
window s.d.

Forecast check on the scored blocks: median block MSE on the adjusted scale 0.066 (lasso), 0.067 (ridge).

## Compute

4 CPUs, 15 GB. Design cache: 3 minutes. Linear stage: 2.7 minutes (lasso 1.3 s, ridge 0.07 s a fit; probe fits
0.7 s); reduce: 5 s; gates: 1 minute.

## Files

`summary.json` (every number above), `lasso_series.csv` / `ridge_series.csv` (groups: mean rank, Borda count, K_A,
S_A, S^sd_A, ...), `*_columns.csv` (predictors: K_j, sum |beta|, sum |beta| sd), `*_series_by_refit.csv`
(||beta_{k,A}||_1), `*_share_by_refit.csv` (S_{k,A}), `*_active_by_refit.csv` (on/off), `*_probes.csv`,
`column_scales.csv`, `linear.npz` (coefficients, window s.d. and IQR, masks), `run.json`, `gates_runner.csv`,
`gates_forecasts.csv`, `frozen_penalties_13.json`.

## Deviations from the closing-interval method

1. Cadence: monthly (decision 1) instead of every 10 trading days; K = 52.
2. Coefficients: the batch fit on each re-estimation's window, the identifiability mask recomputed on that window.
   The closing-interval analysis read the walk's coefficient vector at the anchor (re-solved every day, mask refreshed
   every 250 solves). Both solve the same problem; they can differ only through the mask timing.
3. Not reproduced: the reconciliation with the earlier study's single-penalty lasso (no pooled counterpart).
4. Scale kinds: the "flag" kind holds the missing-data indicator averages only (E6 dropped the activity averages).
5. Probes: same generator and seed, indexed by the monthly k, so the draws differ from the closing-interval ones.

## Open questions

- Ridge vs the committed forecasts (gap up to 2.4e-6, lasso 1e-13): the 13-bar runner on this design cache matches
  this fit to 1e-12, so the gap is between this design rebuild and the one behind `forecasts.parquet`, in columns the
  lasso leaves at zero. Its size (relative 1e-6) cannot move a share or a rank here; the cause is not identified.
- Masked columns: in the median window 233 of the 282 averaged missing-data indicators are constant (the series is
  live on every row of the window), so their coefficients are zero by construction and they count as "not selected"
  in K_j. The group counts K_A are unaffected (each group also has its value averages).
- The penalty picks are those committed at f30ad70; if the 13-bar rerun revises them, this run must be repeated
  (about 3 minutes).
- The previous 13-bar results (base 2, no FOMC; commit 2dbb065) were withdrawn in 74a7805.
