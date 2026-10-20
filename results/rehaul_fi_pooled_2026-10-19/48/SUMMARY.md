# Variable importance, pooled intraday lasso and ridge, all 48 intervals, monthly re-estimation (note E19)

Sample: the E15 variant on all 48 half-hour intervals (overnight included), window W = 96,000 intervals (2,000
trading days). Design and penalties: those of the 48-interval pooled linear rerun, branch
`insurance-selling-2026-09-19-pooled-linear-48` at 8dace13 (design code 8f5d9ec, `pool_linear_pretune.py --rerun`):
base-5 HAR ladder (1, 5, 25, 125, 625, 3125 intervals), the four FOMC series in, no trailing averages of the activity
indicators (E6), calendar = five weekday dummies, the hour and the overnight indicator (E9), every data-fix flag on.
p = 577 predictors in 49 groups: 47 exogenous series in 8 categories, the HAR components, the calendar dummies.
Frozen pre-sample penalties (`frozen_penalties_48.json`, a copy of
`results/pooled_linear_rerun_2026-10-19_48/frozen_penalties.json` on that branch): lasso lambda = 1e-3, ridge
lambda = 1e3 (scikit-learn parametrization); penalty weight omega = 0.01 on the HAR benchmark block (the 6 HAR
components and the 7 calendar dummies), 1 on the exogenous predictors. The file passes the landing rule (base-5 HAR
columns, FOMC series present; `check_landed`).

Code: `experiments/rehaul_fi_pooled_compute.py` (stages `linear`, `gate`, `reduce`, `compare`). Design cache built
with the 48-interval branch's own runner (`experiments/pool_linear_pretune.py design --rerun --bucket all_features`):
300,317 x 577, columns identical to the penalty file's list; not committed (1.4 GB).

## Method

- Re-estimation k = 1, ..., K, K = 52: one at the first trading day of each calendar month from 3 January 2020 to
  1 April 2024 (the span of Table tab:rh_oos). A trading day is a date with regular-hours bars; its first interval is
  the first row labelled with that date (00:00), as the rerun's "first interval of 2020-01-03". Re-estimation k is fit
  on the 96,000 intervals before that row and scored on its out-of-sample block O_k, the intervals up to the next
  re-estimation (54,205 intervals in all; the Sunday-evening intervals before a month's first Monday stay in the
  previous month's block).
- Estimator: the batch solution at the frozen penalty with the spec's identifiability mask (the pooled walk's cold
  seed); 70 lasso fits, 6 exact as solved, 64 after support repair, none failed.
- Gate (`gates_runner.csv`): the 48-interval branch's runner (walk `--rerun`, its tracked one-number schedules
  `sched/{reclasso,ridge}_all_features.json`), walked on the same design cache from its first cold-seed row (246,000),
  gives the same forecast as this fit: lasso to 3e-15, ridge to 1e-13. The rerun committed no all-features forecasts,
  so there is no check against a stored walk.
- Measures and probes: as for the 13 bars (13/SUMMARY.md) and the closing-interval bullets
  (writeup/rehaul/08_feature_importance.tex:51-325).

## Numbers

Lasso, over the sample

- Nonzero coefficients at a re-estimation: median 88 (range 76 .. 94), of which 75 on exogenous predictors.
- Predictors: 159 of 577 selected at least once, 85 in at least half of the re-estimations, 38 in every one.
- Exogenous series: 44 of 47 selected at least once (never: sumret4_ewstock, ofi_ewstock, fomc_release); 35 in at
  least half of the re-estimations; 20 in every one (sumret, sumabsret, sumret3, sumret4, sumpret2, sumautocov,
  sumvolume, numobs, effspread_ewstock, spread_ewstock, sumret2_ewstock, sumret3_ewstock, sumabsret_vwstock,
  stocktwits_attention, stocktwits_sentiment, stocktwits_sentcount, vvix, voldemand_spx_open_and_close,
  voldemand_spx_open_only, voldemand_all_open_only). Rarely selected: fomc_day and turnover_ewstock (K_A = 4 of 52),
  sumbipow_vwstock (5).
- HAR benchmark block: selected in every re-estimation; 69.2 % of sum_j |beta_{k,j}| (median re-estimation). Over the
  sample the calendar dummies hold S_A = 47.4 % of the lasso's absolute coefficient mass, the HAR components 21.5 %
  (see Open questions on the calendar).
- Random probes: the continuous probe is selected in 2 of 18 fits (11 %), the 20-level and binary probes in none.
  128 predictors, in 40 exogenous series, are selected more often than the continuous probe.

Lasso, over time

- Mean rank by ||beta_{k,A}||_1: calendar dummies 1.0, HAR components 2.0, sumret 3.0, sumabsret 4.2, sumvolume 5.3,
  sumautocov 5.4, stocktwits_attention 7.0, stocktwits_sentiment 9.1, voldemand_spx_open_and_close 9.6, sumpret2
  10.2. Kendall's W: 0.87.

Ridge, over the sample

- Leading groups by S_A: calendar dummies 22.6 %, HAR components 9.3 %, sumabsret 5.2 %, vvix 3.5 %, sumvolume 3.1 %;
  HAR benchmark block 31.7 % (median re-estimation).
- Scale check: averages of series values, 282 predictors, window s.d. median 0.88 (5th .. 95th percentile
  0.17 .. 1.95), interquartile range 0.79; averaged missing-data indicators, 282 predictors, s.d. 0.070 where not
  constant (228 constant, hence masked, in the median window); calendar dummies 0.40; HAR components 1.25.
- Standardized share S^sd_A: HAR components 14.5 %, calendar dummies 11.0 %, sumabsret 9.9 %, sumpret2 5.2 %, sumret4
  3.3 %, sumvolume 3.1 %. Spearman rank correlation of S^sd_A with S_A: 0.64 (lasso: 0.98).
- All 49 groups have K_A = K. Random probes: 312 predictors, in all 47 exogenous series, have |beta_{k,j}| above the
  continuous probe's in most of the 18 probe fits.

Ridge, over time

- Mean rank: calendar dummies 1.0, HAR components 2.0, sumabsret 3.0, vvix 4.3, sumvolume 5.4, vix 5.9, sumret 8.0,
  sumautocov 8.6, vix3m 9.2, voldemand_spx_open_and_close and sumpret2 9.6. Kendall's W: 0.90.
- Lasso vs ridge mean rank, Spearman rank correlation over the 49 groups: 0.61.

FOMC series: the lasso selects fomc_since_inv in 51 of 52 re-estimations, fomc_until_inv in 44, fomc_day in 4,
fomc_release in none; in ridge fomc_until_inv ranks 13th (S_A 1.8 %), fomc_day 14th (1.7 %).

Forecast check on the scored blocks: median block MSE on the adjusted scale 0.081 (lasso and ridge).

## The two samples compared (`../compare.csv`, `../compare.json`)

Spearman rank correlation across the 49 groups, 48 intervals vs 13 bars (without the calendar dummies in brackets):

| measure | lasso | ridge |
|---|---|---|
| mean rank by ||beta_{k,A}||_1 | 0.52 (0.50) | 0.78 (0.77) |
| S_A | 0.57 (0.54) | 0.78 (0.77) |
| S^sd_A | 0.55 (0.54) | 0.79 (0.81) |
| K_A | 0.39 (0.38) | (K_A = K in both) |
| top-10 groups by mean rank in common | 4 (4) | 8 (7) |

- Ridge orders the groups much the same way on both samples; the lasso does not. On the 48 intervals the calendar
  dummies are first and the HAR components second in both models; on the 13 bars the HAR components are first, the
  calendar dummies 5th (lasso) and 8th (ridge). sumret is in the top ten of all four rankings (positions 3, 7, 2, 3
  for lasso 48, ridge 48, lasso 13, ridge 13), vvix in three (14, 4, 9, 4).
- The implied volatility series fall on the 48 intervals: in the lasso vix moves from 2.8 (13 bars) to 24.8, vix3m
  from 4.3 to 30.7; in ridge vix from 2.4 to 5.9, vix3m from 7.2 to 9.2. Volume and the realized moments of the index
  rise: lasso sumabsret 25.9 to 4.2, sumpret2 44.9 to 10.2, numobs 31.1 to 13.6; ridge sumvolume 19.7 to 5.4,
  sumpret2 28.3 to 9.6.
- fomc_day: selected in every 13-bar re-estimation (lasso rank 8.3), in 4 of 52 on the 48 intervals (42.9).
- Kendall's W is about 0.9 for both models on both samples, so each sample's ranking is stable over the span; the
  differences are between the samples, not noise across re-estimations.

## Compute

4 CPUs, 15 GB. Design cache: 26 minutes, peak memory about 6.5 GB. Linear stage: 16.7 minutes (lasso 8.1 s, ridge
0.4 s a fit; probe fits 6.3 s); reduce: 6 s; gate: 1 minute.

## Deviations from the closing-interval method

As for the 13 bars (13/SUMMARY.md): monthly cadence, batch fits with the mask recomputed on each window, no
reconciliation with the single-penalty lasso, the "flag" kind holds only missing-data indicator averages, probe
draws indexed by the monthly k. In addition: no gate against stored all-features forecasts (none committed).

## Open questions

- Calendar dummies on the 48 intervals: the weekday dummies are zero on the weekend-dated intervals (Sunday evening,
  4.5 % of rows), so with the free intercept they are contrasts against those intervals. All five coefficients sit at
  about -0.12 (median re-estimation, both models), one common level shift counted five times; that is what puts the
  calendar group first by S_A and by mean rank. Under another base category (for example Monday, with a Sunday
  dummy) the calendar coefficients, and so the calendar share, would be much smaller; because the calendar dummies
  are penalized (omega = 0.01), the re-based fit would also differ slightly. The comparison above therefore also
  reports every correlation without the calendar group. Whether the paper should re-base the weekday dummies is the
  author's call.
- The 48 and 13 samples rank the groups differently under the lasso (Spearman 0.52); a reader of a single ranking
  should be told which sample it comes from.
- Masked columns: 228 of the 282 averaged missing-data indicators are constant in the median window and get a zero
  coefficient by construction (as on 13 bars).
- The penalty picks are those committed at 8dace13; if the 48-interval rerun revises them, this run must be repeated
  (about 17 minutes, plus 26 for the design cache).
