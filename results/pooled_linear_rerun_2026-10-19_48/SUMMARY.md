# Pooled linear rerun on all 48 intervals (2026-10-19 / 20)

Branch `insurance-selling-2026-09-19-pooled-linear-48`. The pooled linear models, rerun on all 48 half-hour intraday
intervals (overnight included, note E15) under the author's decisions of 2026-10-19, with two decisions changed on
2026-10-20 (below). The sibling session runs the same models on the 13 regular-hours intervals; the author picks
between the two samples afterwards (E15). Nothing under `writeup/` was edited.

## What ran

| Step | Code | Output |
|---|---|---|
| Column flags (E6, E9) | `src/backtest/executor.py` `DROP_ACTIVITY_MA`, `CALENDAR_THROUGH_OVERNIGHT` (default off) | commit 56cd5d0 |
| Design caches | `experiments/pool_linear_pretune.py design --rerun` (all features: `--part 1/2`, `2/2`, then `merge`) | `_work/design_*` (gitignored) |
| Pre-tune | `pool_linear_pretune.py pretune --rerun` | `_work/pre/` |
| Frozen penalties | `experiments/pooled_linear_rerun_2026_10_19.py pick` | `frozen_penalties.json`, `prescreen.csv`, `sched/` |
| Walks | `pool_linear_pretune.py walk --rerun --sched sched/<est>_<set>.json` | `_work/walk/` |
| Scoring, export | `pooled_linear_rerun_2026_10_19.py score` | `scores.csv`, `dm.csv`, `forecasts/` |

Every step ran in this container (4 CPUs, 15 GB), so no cluster submission is needed. `cluster/pooled_linear_rerun_48_task.sh`
and `cluster/pooled_linear_rerun_48_tasks.txt` reproduce the walks on Hoffman2 if wanted.

Reproduce (repo root, Python 3.12):

```
python experiments/pool_linear_pretune.py design --rerun --bucket baseline --out results/pooled_linear_rerun_2026-10-19_48/_work/design_baseline
python experiments/pool_linear_pretune.py design --rerun --bucket all_features --part K/2 --out .../_work/design_af_pK      # K = 1, 2
python experiments/pool_linear_pretune.py merge  --rerun --parts .../_work/design_af_p1,.../_work/design_af_p2 --out .../_work/design_all_features
python experiments/pool_linear_pretune.py pretune --rerun --bucket <set> --est <ridge|reclasso|reclasticnet> --r <1.0|0.01> --start 246112 --halo 145200 --design .../_work/design_<set> --out .../_work/pre/<set>/<est>
python experiments/pooled_linear_rerun_2026_10_19.py pick
python experiments/pool_linear_pretune.py walk --rerun --bucket <set> --est <est> --r <1.0|0.01> --sched results/pooled_linear_rerun_2026-10-19_48/sched/<est>_<set>.json --start 246000 --end -1 --halo 96000 --design .../_work/design_<set> --out .../_work/walk/<set>/<est>
python experiments/pooled_linear_rerun_2026_10_19.py score
```

## Decisions applied

1. Sample: all 48 intervals (E15).
2. Window: 2,000 trading days = 96,000 intervals for every model, the OLS included (a0 used 24,000).
3. Penalty: one value for each model, chosen once on the pre-sample and frozen over the evaluation span; no margin
   re-tune. Pre-tune: 4 validation folds x 12,000 intervals tiling the 48,000 intervals before the first interval of
   2020-01-03 (design row 246,112, the same row as the 6 October run), each fit on the 96,000 intervals before a
   1,200-interval gap; criterion the fold mean of the validation MSE on the adjusted scale, first argmin (as
   `experiments/pool_linear_pretune_report.py`). Grids: ridge 1e-2 .. 1e3, lasso and elastic net 1e-6 .. 1e-2, one
   value a decade; elastic net l1_ratio 0.5. The walk seeds the frozen value at each 250th solve (the spec's cold
   re-anchor).
4. Data: every 2026-10-15 data-fix flag of `src/backtest/executor.py` at its default (on), including the forward-fill
   cap; `SPLICE_VIX_FREE_FEED` and `FOMC_CALENDAR_FIX` of `src/data/loading.py` on (defaults).
5. Columns: `DROP_ACTIVITY_MA` (E6) and `CALENDAR_THROUGH_OVERNIGHT` (E9) on. The calendar is DOW_0 .. DOW_4, hour,
   is_overnight; the overnight indicator stays on this sample.
6. **Changed 2026-10-20 (author, in this session: "Confirmed: base 5 and FOMC included"; first relayed by the main
   session):**
   - HAR ladder **base 5**, k = 1, 5, 25, 125, 625, 3125 intervals, for every model including the OLS (the brief said
     base 2, k = 1 .. 2048);
   - the four **FOMC series stay in all features** (the brief excluded them). Their activity averages are dropped
     with the others (E6).
   Both are asserted in `pool_linear_pretune.py` (`spec_section`). The base-2 / no-FOMC pass (1,051 columns) is
   parked under `parked_base2/`.
7. Models: ridge, lasso, elastic net on the HAR benchmark set and on all features, with the printed arms' penalty
   weight on the HAR benchmark block (1/100 on all features; 1 on the HAR benchmark set, where every column is in
   the block). The pooled OLS on the HAR benchmark set (a0's successor) is the spec's ridge path at penalty 0.
8. Scoring: as tab:rh_oos (`experiments/rolling_eval_2026_10_15.py`, its functions imported). Forecast
   sigma2 = pred_adj^2 x B on the 16:00 interval; truth the actual RV (16:00 sumret2 of `data/core_stats.parquet`);
   the 1,082 days 3 January 2020 .. 30 April 2024 (early closes and non-positive prints dropped; the count matches
   \rhoosDays). QLIKE, MSE, R2_OOS, Diebold-Mariano with the Harvey-Leybourne-Newbold correction.
9. Export: forecasts of every interval (N18, E26).
10. The earlier draft's pooled family (blk2, blk2_inc, lasso_t, lasso_f, enet, b2lasso) was not rerun.

## Column counts

| Set | Columns | HAR | Value averages | Availability averages | Activity averages | Calendar |
|---|---|---|---|---|---|---|
| HAR benchmark | 13 | 6 | 0 | 0 | 0 | 7 |
| All features | 577 | 6 | 282 (47 series x 6) | 282 | 0 | 7 |

All features hold 48 FOMC columns (4 series x 6 value + 4 x 6 availability; listed in `frozen_penalties.json`).
Design rows 300,317 (forecast rows 204,317, 2007-12-20 .. 2024-04-30). The walk's identifiability mask zeroes the
columns that are constant or duplicated inside the window: none on the HAR benchmark set, 216 to 234 of the 577 on
all features at each re-anchor of the evaluation span (mostly availability averages of series observed on every
interval of the window, which are constant at 1).

For comparison: with the flags off, the same all-features set would also carry *_active_ma_* and 9 more calendar
columns plus 12 HAR x open/close products (flag check on the sentiment set, commit 56cd5d0: 160 -> 91 columns at base 2).

## Frozen penalties (scikit-learn units)

| Model | HAR benchmark | All features |
|---|---|---|
| Ridge | 100 | 1000 |
| Lasso | 1e-4 | 1e-3 |
| Elastic net | 1e-3 | 1e-3 |
| OLS | 0 | -- |

Fold-mean validation MSE of every grid value: `prescreen.csv`.

## Results: the closing interval, 1,082 days (tab:rh_oos span)

QLIKE and MSE (x 1e10) against the actual RV. "Printed": the current tab:rh_oos row on the same span
(`results/rolling_eval_2026-10-15/metrics.csv`, span common; a0 for the OLS, poolpt_* for the others).

| Model | Set | QLIKE | MSE | R2_OOS % | Printed QLIKE | Printed MSE |
|---|---|---|---|---|---|---|
| OLS | HAR benchmark | 0.11159 | 4.295 | 77.46 | 0.12912 | 4.820 |
| Ridge | HAR benchmark | 0.11144 | 4.290 | 77.49 | 0.11431 | 4.739 |
| Lasso | HAR benchmark | 0.11137 | 4.285 | 77.52 | 0.11355 | 4.912 |
| Elastic net | HAR benchmark | 0.11078 | 4.244 | 77.73 | 0.11418 | 4.936 |
| Ridge | All features | 0.11366 | 4.999 | 73.77 | 0.10855 | 4.793 |
| Lasso | All features | 0.11333 | 4.816 | 74.73 | 0.11015 | 4.697 |
| Elastic net | All features | 0.11251 | 4.914 | 74.22 | 0.10784 | 4.716 |

Diebold-Mariano against the rerun OLS (loss differential model minus OLS; negative = lower loss than the OLS):

| Model | Set | DM QLIKE | p | DM squared error | p |
|---|---|---|---|---|---|
| Ridge | HAR benchmark | -3.84 | 0.0001 | -1.28 | 0.20 |
| Lasso | HAR benchmark | -3.49 | 0.0005 | -1.26 | 0.21 |
| Elastic net | HAR benchmark | -2.95 | 0.003 | -1.23 | 0.22 |
| Ridge | All features | 0.63 | 0.53 | 1.01 | 0.31 |
| Lasso | All features | 0.61 | 0.54 | 0.85 | 0.40 |
| Elastic net | All features | 0.31 | 0.76 | 0.95 | 0.34 |

## Results: every interval of the span (54,168 intervals with positive RV)

| Model | Set | QLIKE | MSE (x 1e10) | DM QLIKE vs OLS | p |
|---|---|---|---|---|---|
| OLS | HAR benchmark | 0.24572 | 1.310 | -- | -- |
| Ridge | HAR benchmark | 0.24586 | 1.310 | 2.69 | 0.007 |
| Lasso | HAR benchmark | 0.24596 | 1.311 | 2.59 | 0.010 |
| Elastic net | HAR benchmark | 0.24719 | 1.311 | 3.22 | 0.001 |
| Ridge | All features | 0.24363 | 1.640 | -1.14 | 0.26 |
| Lasso | All features | 0.24149 | 1.558 | -2.88 | 0.004 |
| Elastic net | All features | 0.24207 | 1.607 | -2.21 | 0.027 |

Reading: at the closing interval the penalized HAR benchmark models beat the OLS significantly (QLIKE), and the
all-features models do not beat it. Over every interval the ordering reverses for QLIKE: the all-features lasso and
elastic net beat the OLS, the penalized HAR benchmark models lose to it. MSE over every interval is higher for the
all-features models. Against the printed rows the HAR benchmark models and the OLS improve at the closing interval,
the all-features models get worse.

## Forecast files

`forecasts/<ols|ridge|lasso|enet>_<baseline|all_features>.parquet`, 54,317 rows each (every interval
2019-12-30 18:00 .. 2024-04-30 23:30, all 48 interval stamps), about 4 MB each. Columns: date (naive ET interval
end), true_adj, pred_adj (adjusted scale), true_raw (the winsorized target in raw units), pred_raw (the executor's
Duan-smeared forecast; depends on the chunk, not used in scoring), B (= true_raw / true_adj^2), plain
(= pred_adj^2 x B, the scored forecast), rv (actual RV, core_stats sumret2). B and plain are undefined on the 33
intervals with zero RV (all in the March 2020 overnight hours); they are left out of the scores.

## Environment and timings

Python 3.12.3; numpy 1.26.4 (OpenBLAS 0.3.23.dev), pandas 2.3.3, scipy 1.17.1, scikit-learn 1.9.1, numba 0.61.2,
pyarrow 18.1.0; Intel Xeon @ 2.10 GHz, 4 CPUs, one BLAS thread a process.

| Step | Wall time |
|---|---|
| HAR benchmark design | about 2 min |
| All-features design (2 parts in parallel + merge) | 27 min |
| Pre-tunes, HAR benchmark | seconds each |
| Pre-tunes, all features (ridge / lasso / elastic net) | 20 s / 100 s / 103 s |
| Walks, HAR benchmark (OLS, ridge / lasso, elastic net) | 13 s / 25-27 s |
| Walks, all features (ridge / lasso / elastic net, in parallel) | 14 / 35 / 31 min |

## Gates

- Flags off reproduce the old column set (sentiment set: 160 columns with both flags off).
- Column parts + merge equal the one-pass design byte for byte (sentiment set, base 2: X, y, baselines, column order,
  Fortran layout).
- The memory-mapped design cache gives byte-identical pred_adj (first 2,000 HAR benchmark intervals, ridge and lasso).
- The HAR benchmark design is byte-identical under Python 3.13 and 3.12.
- The scoring day set is the 1,082 days of tab:rh_oos; the forecast files' rv equals the tab:rh_oos truth on every day.
- Elastic net / lasso with penalty weights: every batch solve certified by the KKT conditions (repair counts in
  `_work/walk/*/*/run.json`: no failures).

## Deviations

- HAR ladder base 5 and FOMC included (decision 6 above), against the original brief.
- The pooled OLS is the spec's ridge path at penalty 0 (rank-one updates of the inverse X'X), not the
  `src/unification.py` a0_ols_har min-norm path; the two agree when the design has full rank (the mask removes
  constant and duplicated columns).
- The walk starts at design row 246,000 (the re-choice boundary just before 2020-01-03, as the printed run's K0 = 600),
  so forecasts exist from 2019-12-30 18:00; the scores use the tab:rh_oos span only.
- The all-features design is built in two column parts and merged (one pass needs more than 15 GB).
- First pass discarded: it ran under Python 3.13 with numpy 1.26.4 built from source without an optimized BLAS; its
  elastic-net pick on the HAR benchmark set differed. All numbers here are from Python 3.12.

## Open questions

1. **Ridge on all features picks 1000, the top of the grid, in all four folds** (the validation MSE still falls:
   0.0993 at 100, 0.0983 at 1000). The grid was kept as specified. A wider grid might pick a larger penalty.
2. **DM against the printed rows is not possible here:** their daily forecasts (`results/spxw_pnl/yhat_poolpt_*`,
   `yhat_a0.parquet`) are not in the repo. Only their printed QLIKE and MSE are compared.
3. **FOMC and the paper's text:** `writeup/rehaul/03_data.tex:128` says the FOMC series are "not in the pooled intraday
   model"; this run includes them (author, 2026-10-20). The text needs to change if this sample is kept.
4. **Pre-tune criterion:** the MSE on the adjusted scale over every interval of the validation folds (as the printed
   pre-tune); the evaluation is QLIKE on actual RV at the closing interval. The two disagree on the all-features sets
   (above).
5. **E28:** VIX, VIX3M and VVIX are spliced from the free hourly feed after February 2024; volatility demand ends in
   August 2023 and is then carried 26 intervals at most, after which its availability indicator is 0 and the value
   takes the causal median.
