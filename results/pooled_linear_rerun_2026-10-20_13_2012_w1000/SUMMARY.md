# Pooled linear models, 13 regular-hours intervals: evaluation from 2012-01-03, 1,000-day window (2026-10-20)

Branch `insurance-selling-2026-09-19-rerun-2012-w1000`, from `insurance-selling-2026-09-19` at eb109e3. Nothing under
`writeup/` was touched, and no Table 1 generator was edited.

## Configuration

Everything is as in the committed run (`results/pooled_linear_rerun_2026-10-19_13/SUMMARY.md`), except two options of
`experiments/pooled_linear_rerun_2026_10_19_13.py`:

| | committed run | this run |
|---|---|---|
| `--window-days` | 2000 (26,000 rows) | **1000 (13,000 rows)** |
| `--first-day` | 2020-01-03 | **2012-01-03** |

Unchanged: the 13-interval sample `rth13` (end stamps 10:00–16:00 ET) with HAR components and trailing averages on the
full 48-interval grid; HAR ladder base 5, `har_ma_{1,5,25,125,625,3125}`; the full predictor set with every FOMC
series and `FOMC_CALENDAR_FIX`; the flags `DROP_ACTIVITY_MA`, `CALENDAR_THROUGH_OVERNIGHT`, `DROP_OVERNIGHT_INDICATOR`,
`LEAN_GLOBAL_LAGS`; every data-fix flag at its default; the seven models (ridge, lasso, elastic net on both sets, and
the min-norm pooled OLS on the HAR benchmark); penalty weight 0.01 on the HAR-benchmark block of the full set; the
grids; the 4 × 250-session pre-tune folds with a 25-session embargo; frozen penalties (`--sched const:<alpha>`); the
identifiability mask and cold seed every 250 solves; the score stage.

The robust standardization and the window of every model, the OLS included, are 13,000 rows. The design cache records
`window: 13000`. Both designs have 84,747 rows (the same as the committed run) and 12 / 576 columns.
`check_design` passed at every stage: the HAR columns are exactly `har_ma_{1,5,25,125,625,3125}`, the calendar is
DOW_0..DOW_4 and hour, and every FOMC series is present (value and availability averages at all 6 horizons).

Commands (`OUT=results/pooled_linear_rerun_2026-10-20_13_2012_w1000`, `F="--window-days 1000 --first-day 2012-01-03"`):

```
python experiments/pooled_linear_rerun_2026_10_19_13.py design  --jobs 2 $F --out $OUT
python experiments/pooled_linear_rerun_2026_10_19_13.py pretune --jobs 4 $F --out $OUT
python experiments/pooled_linear_rerun_2026_10_19_13.py select  $F --out $OUT
python experiments/pooled_linear_rerun_2026_10_19_13.py walk --chunks all_features/ridge=2 --chunks all_features/lasso=4 --chunks all_features/enet=4 --jobs 4 $F --out $OUT
python experiments/pooled_linear_rerun_2026_10_19_13.py score $F --out $OUT
```

Environment: this container (4 CPUs, 15 GB), a Python 3.12.3 venv with numpy 1.26.4 (OpenBLAS 0.3.23.dev), pandas
2.3.3, scipy 1.17.1, scikit-learn 1.9.1, numba 0.68.0, pyarrow 19.0.1, and tqdm 4.70.1 (imported by
`src/backtest/multi_stage.py`; not in the committed run's version list). The container's default Python 3.13 was not
used.

## Pre-tune folds

The 4 folds tile the 1,000 sessions before 2012-01-03 (design rows 31,281–44,280). Each is fit on the 13,000 rows
before a 325-row embargo.

| fold | fit window | validation span |
|---|---|---|
| 0 | 2003-12-16 10:00 .. 2007-12-06 12:00 | 2008-01-15 12:30 .. 2009-01-13 13:00 |
| 1 | 2004-12-14 16:00 .. 2008-12-04 13:00 | 2009-01-13 13:30 .. 2010-01-11 14:00 |
| 2 | 2005-12-09 10:30 .. 2009-12-02 14:00 | 2010-01-11 14:30 .. 2011-01-05 15:00 |
| 3 | 2006-12-07 11:30 .. 2010-11-30 15:00 | 2011-01-05 15:30 .. 2011-12-30 16:00 |

The last validation span ends on 2011-12-30 16:00, the last session before 2012-01-03.
`frozen_penalties.json` records `window_days` 1000 and `window_rows` 13000.

## Frozen penalties

Mean validation MSE over the 4 folds, adjusted scale, all 13 intervals (`pretune_mse.csv`).

| model | set | this run | committed run | grid | at a grid edge |
|---|---|---|---|---|---|
| ridge | HAR benchmark | 1e-2 | 1e-2 | 1e-2 .. 1e3 | bottom |
| lasso | HAR benchmark | **1e-4** | 1e-6 | 1e-6 .. 1e-2 | no |
| elastic net | HAR benchmark | 1e-6 | 1e-6 | 1e-6 .. 1e-2 | bottom |
| ridge | full set | 1e3 | 1e3 | 1e-2 .. 1e3 | top |
| lasso | full set | 1e-3 | 1e-3 | 1e-6 .. 1e-2 | no |
| elastic net | full set | 1e-2 | 1e-2 | 1e-6 .. 1e-2 | top |

- Five of six picks equal the committed run's. The lasso on the HAR benchmark moves from 1e-6 to 1e-4.
- On the HAR benchmark the criterion is flat: 0.066485 (to 6 digits) for ridge 1e-2..1e-1, lasso 1e-6..1e-4 and
  elastic net 1e-6..1e-4.
- On the full set, ridge (1e3) and elastic net (1e-2) pick the top of their grids. The ridge criterion is still
  falling there (0.063981 at 1e2, 0.062188 at 1e3). The elastic net's is 0.062131 at 1e-3 and 0.061965 at 1e-2.
- At the smallest penalties the full set's fold-2 validation MSE is very large (ridge 1e-2: 24.9; lasso 1e-6: 37.8;
  elastic net 1e-6: 27.9; the other folds are 0.08–0.25).
- The `picked_at_grid_end` column of `pretune_mse.csv` is constant within a model and set: it says whether that pair's
  pick is at a grid edge. It does not describe the row's own penalty.

## Rolling-window exercise

The walks start at design row 44,250 (2011-12-28 14:00), the 250-solve boundary at or before the first 2012-01-03 row
(44,281), and end on 2024-04-30 16:00. Every model has **40,497** forecasts, one for every design row of that span,
all finite and positive (`forecasts.parquet`: 7 × 40,497 = 283,479 rows, 12 MB). The dates of each model equal the
design's dates row for row.

The span holds 3,077 days with 13 bars and 77 days with fewer. These are present in the design itself:

- 75 exchange holidays with a partial futures session (MLK, Presidents', Memorial, Independence, Labor Day,
  Thanksgiving): 14 days with 4 bars (10:00–11:30, 2012–2013), 61 with 7 bars (10:00–13:00, 2014 on) and 2018-09-03
  with 8;
- 2011-12-28, with 5 bars, because the walk starts at 14:00.

The committed run's span has 28 such holidays.

**Chunk boundary.** The full-set lasso, elastic net and ridge were split into 4, 4 and 2 chunks, each with a
13,000-row halo. A single lasso walk across the c0/c1 boundary (rows 54,250–54,749, 2015-01-23 15:30 to 2015-03-19
14:30, 500 rows) reproduces the stitched `pred_adj` and `true_adj` bit for bit, and so `plain` = pred_adj²·B as well.
`pred_raw_duan` differs by up to 1.2e-7: the executor's Duan factor is the mean squared residual over each walk's own
emitted rows (`src/evaluation/metrics.py:48`), so it depends on the chunk span. The score stage does not use it.

## Timings (this container: 4 CPUs, 15 GB)

| step | time |
|---|---|
| design, full set | 299 s |
| design, HAR benchmark | 35 s |
| pre-tune, full set | ridge 10 s, lasso 33 s, elastic net 35 s |
| pre-tune, HAR benchmark | 2 s each |
| walks, full set | ridge 85 + 80 s; lasso 182 + 191 + 172 + 173 s; elastic net 81 + 87 + 91 + 87 s |
| walks, HAR benchmark | ridge 5 s, lasso 15 s, elastic net 12 s; OLS 5 s |
| walks, wall (4 jobs) | 5.8 min |
| score | 15 s |
| end to end | about 12 min wall |

The full-set walk jobs sum to 1,229 s, against 724 s in the committed run, over a span 2.8 times as long (40,497 rows
against 14,497).

## Score stage: the closing half hour, common span of tab:rh_oos (2020-01-03 .. 2024-04-30, 1,082 days)

As in the committed run, the score stage scores only the 16:00 row on the common span of tab:rh_oos. Forecast
ŷ²·B, truth the actual RV of the closing half hour from `data/core_stats.parquet`.

`scores.csv` (the "committed" columns are the committed run's `scores.csv`; the printed tab:rh_oos rows are also in
`scores.csv`):

| model | set | λ | QLIKE | committed | MSE ×1e10 | committed | R²_OOS % | committed |
|---|---|---|---|---|---|---|---|---|
| OLS | HAR benchmark | — | 0.12045 | 0.11524 | 3.925 | 4.153 | 79.40 | 78.21 |
| ridge | HAR benchmark | 1e-2 | 0.12045 | 0.11524 | 3.925 | 4.153 | 79.40 | 78.21 |
| lasso | HAR benchmark | 1e-4 | 0.12030 | 0.11524 | 3.922 | 4.153 | 79.42 | 78.21 |
| elastic net | HAR benchmark | 1e-6 | 0.12045 | 0.11524 | 3.925 | 4.153 | 79.40 | 78.21 |
| ridge | full set | 1e3 | 0.12362 | 0.12301 | 6.253 | 6.926 | 67.19 | 63.66 |
| lasso | full set | 1e-3 | 0.12292 | 0.12122 | 6.800 | 7.821 | 64.32 | 58.96 |
| elastic net | full set | 1e-2 | 0.11940 | 0.11476 | 6.485 | 7.151 | 65.97 | 62.48 |

**Diebold–Mariano tests against this run's pooled OLS** (`dm.csv`, HLN, Newey–West lag 6, T = 1,082; dm < 0 means
the model has the lower loss):

| model | set | DM QLIKE | p | mean QLIKE diff | DM sq. error | p |
|---|---|---|---|---|---|---|
| ridge | HAR benchmark | −3.10 | 0.002 | −4.7e-8 | 0.50 | 0.61 |
| lasso | HAR benchmark | −5.65 | <0.001 | −1.6e-4 | −1.19 | 0.23 |
| elastic net | HAR benchmark | −5.71 | <0.001 | −8.0e-7 | −1.19 | 0.23 |
| ridge | full set | 0.61 | 0.54 | 3.2e-3 | 1.34 | 0.18 |
| lasso | full set | 0.54 | 0.59 | 2.5e-3 | 1.29 | 0.20 |
| elastic net | full set | −0.31 | 0.76 | −1.1e-3 | 1.28 | 0.20 |

The tests against the closing-interval OLS and against the printed rows are not computed: their forecast files are not
in the repo, and `dm.csv` records this in `note`, as in the committed run.

**Every interval of the sample, same 1,082 days (secondary)** (`scores_all_bars.csv`, 14,066 intervals):

| model | set | QLIKE | committed | MSE ×1e10 | committed |
|---|---|---|---|---|---|
| OLS, ridge, elastic net | HAR benchmark | 0.19511 | 0.1926 | 2.207 | 2.21 |
| lasso | HAR benchmark | 0.19515 | 0.1926 | 2.206 | 2.21 |
| ridge | full set | 0.18689 | 0.1860 | 3.895 | 4.38 |
| lasso | full set | 0.18742 | 0.1866 | 4.064 | 4.28 |
| elastic net | full set | 0.18738 | 0.1859 | 3.564 | 3.78 |

Ratio ŷ²·B / RV at the 16:00 row, 2012-01-03 .. 2024-04-30: HAR benchmark models 0.082–3.77; full set 0.069–4.28.

The main session scores the full 2012–2024 span (all 13 bars, Diebold–Mariano on daily averages, full span and
subperiods) from `forecasts.parquet`.

## Availability of the exogenous series

Share of rows on which a series has an observation in its last interval (`<series>_avail_ma_1` > 0 in the full-set
design). A share of 0 means the series is absent and neutral-filled on every row of that span. The series not listed
(SPY intraday moments, the equal- and value-weighted stock series, order flow, FOMC) have a share of 0.98–1.00 in
every span below.

| series | first observed | 1,000 days before start (2008-01-15 .. 2011-12-30) | first eval session (2012-01-03) | first 250 eval sessions |
|---|---|---|---|---|
| VVIX | 2012-03-26 11:00 | **0** | **0** | 0.70 |
| volatility demand (4 series) | 2011-01-03 10:30 | 0.23 | 0.92 | 0.91 |
| VIX3M | 2009-08-12 11:00 | 0.55 | 0.92 | 0.92 |
| StockTwits sentiment, sentiment count | 2010-06-02 11:30 | 0.34 | 0.77 | 0.95 |
| StockTwits attention | 2009-03-23 15:30 | 0.63 | 1.00 | 1.00 |
| VIX | 2006-05-22 11:00 | 0.91 | 0.92 | 0.92 |

At the start of the evaluation span, VVIX has no observed data: it is 0 on every row of the first fit window and of
the first evaluation session, and it begins on 2012-03-26. On the first row alone (2012-01-03 10:00, whose predictors end
at the 09:30 interval, before the open) VIX, VIX3M, the four volatility demand series and the two StockTwits sentiment
series also read 0. They are observed on the other 12 rows of that session, and the VIX reads 0 on every 10:00 row of
2012 (0.98 at 10:30, 1.00 at 12:00 and 16:00), so this is the pre-open interval and not a missing series.

The pre-tune folds (fit window / validation span):

| series | fold 0 | fold 1 | fold 2 | fold 3 |
|---|---|---|---|---|
| VVIX | **0 / 0** | **0 / 0** | **0 / 0** | **0 / 0** |
| volatility demand (4 series) | **0 / 0** | **0 / 0** | **0** / 0.010 | **0** / 0.92 |
| StockTwits sentiment, sentiment count | **0 / 0** | **0 / 0** | **0** / 0.41 | 0.078 / 0.95 |
| StockTwits attention | **0 / 0** | **0** / 0.52 | 0.10 / 1.00 | 0.35 / 1.00 |
| VIX3M | **0 / 0** | **0** / 0.38 | 0.072 / 0.92 | 0.30 / 0.91 |
| VIX | 0.36 / 0.92 | 0.58 / 0.92 | 0.81 / 0.92 | 0.92 / 0.91 |

So VVIX is absent from every pre-tune fit and validation span. The volatility demand series are absent from every fit
window. In fold 0 all eight StockTwits / VIX3M / VVIX / volatility demand series are absent from both fit and
validation. "First observed" is the first design row with an observation. The SPY, stock and FOMC series are observed
from the first dated design row (2002-06-05 12:30), so their true start is earlier.

## Files

- `frozen_penalties.json`: the penalties, each design's columns and counts, the penalty weight, the folds (fit and
  validation rows, validation first and last bar) and the runner configuration (with `--window-days 1000`).
- `pretune_mse.csv`: validation MSE of every grid penalty on each fold, with `picked` and `picked_at_grid_end`.
- `scores.csv`, `dm.csv`, `scores_all_bars.csv`: the score stage, as described above.
- `forecasts.parquet`: every interval of every model from 2011-12-28 14:00 to 2024-04-30 16:00. It has the committed
  run's columns: `series_id`, `model`, `predictor_set`, `alpha`, `date`, `pred_adj`, `true_adj`, `B`,
  `plain` = pred_adj²·B, `pred_raw_duan`, `true_raw`, `rv_actual`.
- `_work/` (gitignored): design caches, pre-tune folds, walk tables and traces, logs.
