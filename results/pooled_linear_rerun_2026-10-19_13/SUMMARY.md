# Pooled linear models, 13 regular-hours intervals, frozen penalties (2026-10-19)

Branch `insurance-selling-2026-09-19-pooled-linear-13`. The sibling run on all 48 intervals is
branch `insurance-selling-2026-09-19-pooled-linear-48`; the author picks between the two (note E15).
Nothing under `writeup/` was touched.

## Author corrections applied during the run

The author's main session relayed two corrections at 22:45 UTC. Neither had reached this session
before then. The two original correction messages were queued at 21:15 and 21:19 UTC but were only delivered to this session at 22:59 UTC, after the rerun was done.

1. **HAR ladder base 5** ("we need base 5 across the entire paper"): k in {1, 5, 25, 125, 625, 3125}
   intervals of the 48-interval grid, for every model including the OLS. The brief had base 2
   (1, 2, 4, ..., 2048).
2. **FOMC series in the full predictor set** (`FOMC_FEATURES`, `src/data/loading.py:136`), with
   `FOMC_CALENDAR_FIX` on (its default, `loading.py:402`). E6 still drops their activity averages.
   The brief had them out.

The first run, with base 2 and no FOMC, was complete when the correction arrived. It is parked, not
deleted:
- `frozen_penalties_base2_superseded.json`, which was pushed at 22:25 UTC and replaced at 22:52 UTC;
- `base2_superseded/`, holding its scores, DM tests, forecasts and pre-tune table.

Its headline, for reference: on the HAR benchmark the QLIKE was 0.1156–0.1159; on the full predictor
set (1,050 columns) it was 0.118–0.124. `experiments/pooled_linear_rerun_2026_10_19_13.py`
(`check_design`) now asserts two things before every stage: the HAR columns are exactly
`har_ma_{1,5,25,125,625,3125}`, and every FOMC series is present.

## What ran

**Sample.** The 13 intervals with end stamps 10:00–16:00 ET (segment `rth13`), as one pooled intraday
model. The target's HAR components and every trailing average are computed on the full 48-interval
grid, overnight intervals included (lag scope `global`); only then are the rows restricted. The HAR
burn-in (3,125 intervals) is dropped on the full grid before the restriction
(`BURNIN_IN_BAR_SPACE`). Horizon one: each row's predictors end at the previous interval of the full
grid. So the 16:00 row is the 15:30 forecast of the closing half hour, and the 10:00 row uses the
overnight intervals through 09:30.

**Window.** 2,000 trading days, which is 26,000 rows on the 13-interval grid. This holds for every
model, the OLS included. The robust standardization runs over the same 26,000-row window.

**Design ("pooled panel v2").** Every data-fix flag is at its default (on):
- `src/backtest/executor.py`: `APPLY_FFILL_LIMIT`, `ACTIVE_FROM_OBSERVED`, `CAUSAL_MODE_INFLATION`,
  `CAUSAL_NEUTRAL_FILL`, `CAUSAL_GUARD_MODE`, `B_REGULAR_BARS_ONLY`, `DROP_EARLY_CLOSE_SESSIONS`,
  `ERODE_IN_BARS`, `BURNIN_IN_BAR_SPACE`.
- `src/data/loading.py`: the invalid-print nulling (21,943 prints), the VIX / VIX3M / VVIX splice
  (1,476 / 702 / 648 intervals) and `FOMC_CALENDAR_FIX`.

Column flags set for this run (all default False; the defaults reproduce every existing path):

| flag | source | effect here |
|---|---|---|
| `DROP_ACTIVITY_MA` (E6) | cherry-picked from the 48-interval session's commit 56cd5d0 (here 3cab1ca); it appeared at 21:34 UTC, inside the two-hour limit | no `*_active_ma_*` columns |
| `CALENDAR_THROUGH_OVERNIGHT` (E9) | same commit | calendar = DOW_0..DOW_4, hour, is_overnight; no open / close / expiry / month-end / quarter-end columns, no HAR x open/close products |
| `DROP_OVERNIGHT_INDICATOR` | this session (a third flag, not a sample-dependent rule) | is_overnight out: on the 13 intervals it is 1 only on the 16:00 interval (`99_author_notes.tex:31`) |
| `LEAN_GLOBAL_LAGS` | this session | memory only: each trailing average is computed on the full grid with the same pandas operations, and only the 13-interval rows are kept |

`LEAN_GLOBAL_LAGS` gives the same design bit for bit. That was checked on the sentiment bucket in
this configuration: X, y, B, names, dates and memory layout. Without it, the base-2 full-set design
went past 12.5 GB and was killed on this 15 GB machine.

**Column counts.**
- HAR benchmark: **12** columns. These are har_ma_{1,5,25,125,625,3125}, DOW_0..DOW_4 and hour.
- Full predictor set: **576** columns. These are the 12 above plus 47 series × (value, availability)
  × 6 horizons. FOMC accounts for 48 of them.
- Both designs have 84,747 rows and share one row index.

The identifiability mask (columns constant or duplicated in the window) zeroes 224–230 columns of
the full set in the pre-tune fit windows. These are mostly availability averages of fully observed
series.

**Models.** Seven models:
- ridge, lasso and elastic net (l1_ratio 0.5), each on both predictor sets;
- the pooled OLS on the HAR benchmark, which replaces a0. It is min-norm least squares on the centred
  window: an eigh pseudo-inverse at numpy's rcond, `src/unification.py` `_walk_ols`. The Gram sums
  roll rank-one and are rebuilt exactly every 250 solves. Against a direct `lstsq` fit its forecasts
  agree to ≤ 3.5e-14.

The penalized models keep the printed arms' block structure: penalty weight 0.01 on the
HAR-benchmark block of the full predictor set, and 1 on the HAR benchmark set, where every column is
in that block. The intercept is unpenalized.

**Pre-tune.** This is the existing pretune stage:
- 4 folds × 250 sessions (3,250 rows) tile the 1,000 sessions before 2020-01-03. The validation
  spans run from 2016-01-19 13:00 to 2020-01-02 16:00.
- Each fold is fit on the 26,000 rows before a 25-session (325-row) embargo.
- Grids are one value a decade: ridge 1e-2..1e3; lasso and elastic net 1e-6..1e-2.
- The criterion is mean validation MSE on the adjusted scale, over all 13 intervals.

**Frozen penalties** (`frozen_penalties.json`):

| model | HAR benchmark | full predictor set |
|---|---|---|
| ridge | 1e-2 | 1e3 (top of grid) |
| lasso | 1e-6 | 1e-3 |
| elastic net | 1e-6 | 1e-2 (top of grid) |
| OLS | — | |

**Rolling-window exercise.** Each model uses one penalty over the whole span (`--sched const:<alpha>`),
with no re-choice. The identifiability mask and the cold seed still run every 250 solves. The run
starts at the 250-solve boundary at or before the first 2020-01-03 row (2019-12-04 16:00) and ends at
2024-04-30 16:00. Every model has 14,497 forecasts, one for every interval of the sample.
- Lasso and elastic net on the full set were each split into 2 chunks, each with a 26,000-row halo.
  A single walk across the lasso chunk boundary reproduces the stitched forecasts bit for bit
  (500 rows).
- Penalty-weight batch solves: none failed the KKT certification. Most were repaired by support
  changes, and one by coordinate descent.

**Scoring.** As `experiments/rolling_eval_2026_10_15.py`:
- the forecast is ŷ²·B at the 16:00 row, i.e. the 15:30 forecast of the closing half hour;
- the truth is the actual (unwinsorized) RV of the closing half hour from `data/core_stats.parquet`;
- the span is the common span of tab:rh_oos: 2020-01-03 to 2024-04-30, **1,082** days, early closes
  dropped;
- QLIKE, MSE and R²_OOS against the expanding mean, and `dm_test` (HLN, Newey-West lag 6).

## Results: the closing half hour, common span (1,082 days)

`scores.csv`. The "printed" columns are tab:rh_oos (`results/rolling_eval_2026-10-15/metrics.csv`,
span `common`): `poolpt_*` (arm f, 48 intervals, base 5, margin re-tune) and `a0`.

| model | set | λ | QLIKE | printed | MSE ×1e10 | printed | R²_OOS % | printed |
|---|---|---|---|---|---|---|---|---|
| OLS | HAR benchmark | — | 0.11524 | 0.12912 (a0) | 4.153 | 4.820 | 78.21 | 74.71 |
| ridge | HAR benchmark | 1e-2 | 0.11524 | 0.11431 | 4.153 | 4.739 | 78.21 | 75.14 |
| lasso | HAR benchmark | 1e-6 | 0.11524 | 0.11355 | 4.153 | 4.912 | 78.21 | 74.23 |
| elastic net | HAR benchmark | 1e-6 | 0.11524 | 0.11418 | 4.153 | 4.936 | 78.21 | 74.10 |
| ridge | full set | 1e3 | 0.12301 | 0.10855 | 6.926 | 4.793 | 63.66 | 74.85 |
| lasso | full set | 1e-3 | 0.12122 | 0.11015 | 7.821 | 4.697 | 58.96 | 75.35 |
| elastic net | full set | 1e-2 | 0.11476 | 0.10784 | 7.151 | 4.716 | 62.48 | 75.26 |

**Diebold–Mariano tests against the new pooled OLS** (`dm.csv`; dm < 0 means the model has the lower
loss):

| model | set | DM QLIKE | p | mean QLIKE diff | DM sq. error | p |
|---|---|---|---|---|---|---|
| ridge | HAR benchmark | −3.10 | 0.002 | −1.7e-8 | 0.76 | 0.44 |
| lasso | HAR benchmark | −4.80 | <0.001 | −1.2e-6 | −0.69 | 0.49 |
| elastic net | HAR benchmark | −4.94 | <0.001 | −6.0e-7 | −0.62 | 0.53 |
| ridge | full set | 1.49 | 0.14 | 7.8e-3 | 1.32 | 0.19 |
| lasso | full set | 1.38 | 0.17 | 6.0e-3 | 1.23 | 0.22 |
| elastic net | full set | −0.18 | 0.86 | −4.8e-4 | 1.22 | 0.22 |

**Reading.**
- On the HAR benchmark the three penalties are at or near zero, so ridge, lasso and elastic net are
  the OLS to 4 decimals. Their "significant" QLIKE gains are differences of 1e-8 to 1e-6 that are
  extremely steady. They carry no economic weight.
- The new OLS is far better than the printed a0: QLIKE 0.1152 vs 0.1291, MSE 4.15 vs 4.82.
- The 13-interval HAR-benchmark models have lower MSE and higher R²_OOS than every printed pooled
  linear row. Their QLIKE is 0.8–1.5 % higher than the printed penalized rows.
- On the full predictor set none of the three beats the OLS significantly. Only the elastic net has
  a lower QLIKE than the OLS (−0.4 %), and all three have a much larger MSE.
- The full-set rows are worse than the printed ones (QLIKE +6 % to +13 %, MSE +44 % to +66 %).
- No forecast blows up: ŷ²·B / RV stays in [0.10, 3.8] for every model. 2020-03-13 carries 35–40 %
  of the full-set squared error. Without 2020H1, full-set QLIKE is 0.1135 (elastic net), 0.1191
  (lasso) and 0.1189 (ridge), against 0.1146 for the HAR benchmark.

**Against the printed rows and the closing-interval OLS:** not computed. The printed forecast files
(`results/spxw_pnl/yhat_poolpt_*.parquet`, `yhat_a0.parquet`) and the closing-interval OLS arm file
(`results/linear_subsection_dedup/.../incumbent_ols/results_bar1600.csv`) are not in the repo.
`dm.csv` records this for every row. The score stage computes these tests automatically when the
files are present. Only the printed QLIKE / MSE / R²_OOS levels are compared above.

**Every interval of the sample (secondary)** (`scores_all_bars.csv`): 1,082 days × 13 = 14,066
intervals, scored on the actual RV of each interval.

| | QLIKE | MSE ×1e10 |
|---|---|---|
| HAR benchmark, all four models | 0.1926 | 2.21 |
| full set, ridge | 0.1860 | 4.38 |
| full set, lasso | 0.1866 | 4.28 |
| full set, elastic net | 0.1859 | 3.78 |

Over all 13 intervals the full set lowers QLIKE by 3.1–3.5 % but raises MSE (+71 % to +98 %).

## Files

- `frozen_penalties.json`: `{model: {predictor_set: alpha}}`, plus each design's column list and
  count, the penalty weight, the folds and the runner configuration.
- `pretune_mse.csv`: validation MSE of every grid penalty on each fold, with `picked` and
  `picked_at_grid_end`.
- `scores.csv`, `dm.csv`, `scores_all_bars.csv`.
- `forecasts.parquet` (4.4 MB): every interval of every model, 7 × 14,497 rows, 2019-12-04 16:00 to
  2024-04-30 16:00. Columns:
  - `pred_adj`, `true_adj`;
  - `B`, the diurnal baseline;
  - `plain` = pred_adj²·B;
  - `pred_raw_duan`, the executor's Duan-smeared forecast;
  - `true_raw`;
  - `rv_actual`, the core_stats RV of the interval.
- `base2_superseded/` and `frozen_penalties_base2_superseded.json`: the parked first run.
- `_work/` (gitignored, about 0.7 GB): design caches, pre-tune folds, walk tables and traces, logs.

## Timings (this container: 4 CPUs, 15 GB, Python 3.12.3)

| step | time |
|---|---|
| design, full set | 297 s (peak RSS 5.5 GB for the base-2 version) |
| design, HAR benchmark | 20 s |
| pre-tune, full set | ridge 10 s, lasso 40 s, elastic net 44 s |
| pre-tune, HAR benchmark | < 1 s |
| walks, full set | ridge 106 s; lasso 169 + 153 s; elastic net 103 + 93 s |
| walks, HAR benchmark and OLS | 1–3 s |
| end to end | about 10 min wall |

No cluster run is needed.

**Library versions.** Python 3.12.3, numpy 1.26.4 (OpenBLAS 0.3.23.dev), pandas 2.3.3, scipy 1.17.1,
numba 0.68.0, scikit-learn 1.9.1, pyarrow 19.0.1. The container's default Python 3.13 has no numpy
1.26 wheel, so pip built numpy from source without an optimized BLAS (0.2 GFLOP/s). Everything here
ran in a Python 3.12 venv instead, where the same matmul reaches ~95 GFLOP/s. The design arrays are
bit-identical under both builds.

## Reproduce

```
python experiments/pooled_linear_rerun_2026_10_19_13.py design  --jobs 2
python experiments/pooled_linear_rerun_2026_10_19_13.py pretune --jobs 4
python experiments/pooled_linear_rerun_2026_10_19_13.py select
python experiments/pooled_linear_rerun_2026_10_19_13.py walk --chunks all_features/lasso=2 --chunks all_features/enet=2 --jobs 4
python experiments/pooled_linear_rerun_2026_10_19_13.py score
```

The runner configuration (`CFG` in the script) is:

```
--sample rth13 --har-base 5 --ex-flag DROP_ACTIVITY_MA=1 --ex-flag CALENDAR_THROUGH_OVERNIGHT=1 --ex-flag DROP_OVERNIGHT_INDICATOR=1 --ex-flag LEAN_GLOBAL_LAGS=1
```

## Code changes

Every change sits behind a default that keeps the old behaviour. With the defaults, the 48-interval
sentiment design of `experiments/pool_linear_pretune.py` reproduces the base commit bit for bit.

- `src/backtest/segmentation.py`: `SAMPLE_SEGMENTS["rth13"]`. It is kept out of
  `SEGMENT_DEFINITIONS`, so segment `all` is unchanged. The existing `rth` segment also holds the
  09:30 stamp: 14 intervals.
- `src/backtest/executor.py`: the two shared flags (cherry-picked), `DROP_OVERNIGHT_INDICATOR` and
  `LEAN_GLOBAL_LAGS`.
- `experiments/pool_linear_pretune.py`: the options `--sample`, `--har-base`, `--drop-fomc` (unused
  after the correction), `--ex-flag`, `--sched const:A` and `--est ols`. `PERIODS_PER_DAY` follows
  the sample. The design cache records its configuration.
- `experiments/pooled_linear_rerun_2026_10_19_13.py`: the orchestration and scoring.

## Deviations and open questions

1. **Penalties at the grid edge.** On the full set, ridge (1e3) and elastic net (1e-2) pick the top
   of their grids, and the validation MSE is still falling there. The grid was kept as instructed.
   A wider grid might choose a stronger penalty.
2. **Flat HAR-benchmark criterion.** On the HAR benchmark the validation MSE is flat to 5 digits
   over 1e-2..1 (ridge) and 1e-6..1e-5 (lasso and elastic net). The first minimum, the smallest
   penalty, is kept. In effect these three models are the OLS.
3. **Early-close sessions.** `DROP_EARLY_CLOSE_SESSIONS` (data fix, default on) drops whole NYSE
   early-close sessions from the 13-interval sample, in both training and forecast rows. The pooled
   48-interval path keeps every interval. The evaluation drops early closes anyway. Read as the
   data-fix default, since the brief says every data-fix flag is at its default.
4. **Window units for the rolling standardization.** On the 13-interval sample the window is 26,000
   rows, i.e. 2,000 trading days. The erosion of the scale guards' availability mask converts a
   k-interval average to ceil(k·13/46) rows (`ERODE_IN_BARS`, `BARS_PER_SESSION` = 46), as the code
   already did for any time-of-day slice.
5. **Pre-tune validation over all 13 intervals.** This is the 13-interval analogue of the printed
   "any-clock" criterion, not the 16:00 rows only.
6. **The overnight indicator** is dropped by a flag, not by a rule read off the sample.
7. **Full-set rows worse than printed.** The rows differ from the printed ones in several ways at
   once: sample, ladder unchanged (base 5), no activity averages, a calendar ending at hour, FOMC
   calendar fix, causal data fixes, a frozen penalty against arm f's 15–35 switches, and a 26,000-row
   window against 96,000. This run does not separate those effects.
8. **Diebold–Mariano against the printed rows and the closing-interval OLS** is pending; the files
   are not in the repo (see above).
