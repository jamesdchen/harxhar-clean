# Pooled linear models, 13 regular-hours intervals, evaluation from 2012-01-03, 2,000-day window (2026-10-20)

Branch `insurance-selling-2026-09-19-rerun-2012-w2000`, from `insurance-selling-2026-09-19` at eb109e3. This is the
2,000-day companion of a 1,000-day 2012 run done in another session. Nothing under `writeup/` was touched.

## Configuration

Everything is as the committed run, `results/pooled_linear_rerun_2026-10-19_13/` (see its SUMMARY.md), except the
first evaluation day:

- first evaluation day 2012-01-03 (committed: 2020-01-03), through the end of the data, 2024-04-30 16:00;
- rolling window 2,000 trading days = 26,000 rows (unchanged), for every model, the OLS included;
- the 13 intervals with end stamps 10:00–16:00 ET, HAR ladder base 5 (`har_ma_{1,5,25,125,625,3125}`), FOMC series in
  the full predictor set, every data-fix flag at its default, and the runner configuration
  `--sample rth13 --har-base 5 --ex-flag DROP_ACTIVITY_MA=1 --ex-flag CALENDAR_THROUGH_OVERNIGHT=1
  --ex-flag DROP_OVERNIGHT_INDICATOR=1 --ex-flag LEAN_GLOBAL_LAGS=1` (`CFG` in the script, `runner_cfg` in
  `frozen_penalties.json`);
- column counts: HAR benchmark 12, full predictor set 576; both designs 84,747 rows on one row index.
  `check_design` passes for both sets (HAR columns exactly `har_ma_{1,5,25,125,625,3125}`, every FOMC series present).

Commands (`OUT=results/pooled_linear_rerun_2026-10-20_13_2012_w2000`, `F="--first-day 2012-01-03"`):

```
python experiments/pooled_linear_rerun_2026_10_19_13.py design  --jobs 2 $F --out $OUT
python experiments/pooled_linear_rerun_2026_10_19_13.py pretune --jobs 4 $F --out $OUT
python experiments/pooled_linear_rerun_2026_10_19_13.py select  $F --out $OUT
python experiments/pooled_linear_rerun_2026_10_19_13.py walk --chunks all_features/ridge=2 --chunks all_features/lasso=4 --chunks all_features/enet=4 --jobs 4 $F --out $OUT
python experiments/pooled_linear_rerun_2026_10_19_13.py score $F --out $OUT
```

No code was changed. The only repository change outside this directory is the `.gitignore` line for `$OUT/_work/`.

## Pre-tune and frozen penalties

4 folds × 250 sessions (3,250 rows) tile the 13,000 rows before the first 2012-01-03 row (design row 44,281). Each fold
is fit on the 26,000 rows before a 25-session (325-row) embargo. Grids and criterion as the committed run (ridge
1e-2..1e3, lasso and elastic net 1e-6..1e-2, one value a decade; mean validation MSE on the adjusted scale over all
13 intervals). `frozen_penalties.json` records `window_days` 2000 and `window_rows` 26000.

| fold | fit rows | validation rows | validation span |
|---|---|---|---|
| 0 | 4,956–30,956 | 31,281–34,531 | 2008-01-15 12:30 .. 2009-01-13 13:00 |
| 1 | 8,206–34,206 | 34,531–37,781 | 2009-01-13 13:30 .. 2010-01-11 14:00 |
| 2 | 11,456–37,456 | 37,781–41,031 | 2010-01-11 14:30 .. 2011-01-05 15:00 |
| 3 | 14,706–40,706 | 41,031–44,281 | 2011-01-05 15:30 .. 2011-12-30 16:00 |

The last validation row is 2011-12-30 16:00, the last session before 2012-01-03 (2012-01-02 was an NYSE holiday).

**Frozen penalties** (`frozen_penalties.json`; full table in `pretune_mse.csv`):

| model | HAR benchmark | committed run (2020) | full predictor set | committed run (2020) |
|---|---|---|---|---|
| ridge | 1e-2 (bottom of grid) | 1e-2 | 1e3 (top of grid) | 1e3 |
| lasso | 1e-4 | 1e-6 | 1e-3 | 1e-3 |
| elastic net | 1e-4 | 1e-6 | 1e-3 | 1e-2 (top of grid) |
| OLS | — | — | | |

Grid edges: ridge on the HAR benchmark picks the bottom of its grid and ridge on the full set the top, as in the
committed run. No lasso or elastic net pick is at a grid edge here; the committed run's elastic net on the full set
was at the top.

Mean validation MSE near the picks:
- HAR benchmark: flat to 5 digits, 0.065995 for ridge 1e-2..1, lasso 1e-6..1e-5, elastic net 1e-6..1e-5; the picks
  are 0.065994 (lasso 1e-4) and 0.065995 (elastic net 1e-4, lowest at more digits).
- Full set: ridge 0.063463 (1e2), 0.061749 (1e3); lasso 0.063533 (1e-4), 0.061323 (1e-3), 0.062753 (1e-2);
  elastic net 0.072613 (1e-4), 0.061175 (1e-3), 0.061855 (1e-2).

## Rolling-window exercise

Each model uses its frozen penalty over the whole span (`--sched const:<alpha>`); the identifiability mask and the
cold seed still run every 250 solves. The walk starts at the 250-solve boundary at or before the first 2012-01-03 row:
design row 44,250, 2011-12-28 14:00. It ends at 2024-04-30 16:00.

- `forecasts.parquet`: 7 models × 40,497 rows = 283,479 rows. For every model the dates equal every bar of the design
  from 2011-12-28 14:00 to 2024-04-30 16:00, and `pred_adj`, `plain` and `pred_raw_duan` are finite.
- Chunks: ridge on the full set in 2, lasso and elastic net on the full set in 4, each with a 26,000-row halo, split
  at 250-solve boundaries (rows 54,500, 64,750, 75,000 for lasso and elastic net; 64,500 for ridge).
- Chunk boundary check: a single lasso walk over rows 64,500–65,000 (2018-03-06 10:30 .. 2018-04-30 13:00, 500 rows),
  across the c1/c2 boundary at row 64,750, reproduces the stitched `pred_adj` bit for bit.
- KKT certification of the batch solves: no failures. Lasso, full set: 36 certified directly, 147 repaired by support
  changes, 0 by coordinate descent, 21 reseeds. Elastic net, full set: 20 certified directly, 142 repaired by support
  changes, 0 by coordinate descent.

## OLS equality check, 2020-01-03 .. 2024-04-30

The OLS (HAR benchmark, no penalty, same window) forecasts against the committed run's, on the committed run's
14,497 rows (2019-12-04 16:00 .. 2024-04-30 16:00) and on the 14,262 rows from 2020-01-03:

| quantity | result |
|---|---|
| dates, `true_adj`, `B`, `true_raw`, `rv_actual` | exactly equal |
| `pred_adj`, `plain` | **not bit for bit**: 3.5 % of rows exactly equal; max relative difference of `plain` 4.4e-14, max absolute difference of `pred_adj` 5.4e-14 |
| QLIKE / MSE / R²_OOS of the closing half hour | equal to the printed digits (0.115239, 4.153049e-10, 78.208682 %) |

What was checked to locate the difference:
- An OLS walk on this run's design from the committed run's start row (70,250, 2019-12-04 16:00) gives the same
  differences against the committed file (3.5 % exactly equal, max 5.4e-14). So the first evaluation day is not the
  cause: started at either row, this machine gives the same OLS forecasts to round-off, and differs from the
  committed file at that level.
- Forcing the OpenBLAS kernel (`OPENBLAS_CORETYPE` Haswell or SkylakeX) leaves the difference in place.
- Rebuilding the HAR benchmark design with numba compiled for a non-AVX-512 target (`NUMBA_CPU_NAME=haswell`) gives a
  design bit-identical to this run's.

This container is a different machine from the committed run's: Intel Xeon @ 2.80 GHz with AVX-512 (OpenBLAS
SkylakeX kernel), against Intel Xeon @ 2.10 GHz in RECONCILE.md. The library versions are the same. The remaining
source of the 1e-14 difference was not isolated further. For scale, the committed run reports its OLS agreeing
with a direct `lstsq` fit to ≤ 3.5e-14.

For reference, the penalized models with the same frozen penalty in both runs, on the same 14,497 rows: ridge on the
HAR benchmark (1e-2) has a max relative difference of `plain` of 3.4e-9, ridge on the full set (1e3) 4.1e-10.

## Scores: the closing half hour, common span of tab:rh_oos (1,082 days)

As the committed run, the score stage scores only the 16:00 row (ŷ²·B, the 15:30 forecast of the closing half hour)
on 2020-01-03 .. 2024-04-30, 1,082 days, early closes dropped, against the actual RV from `data/core_stats.parquet`.
Full-span and subperiod scores over 2012–2024 are done by the main session for Table 1.

`scores.csv` (the committed run's numbers in parentheses; "printed" is tab:rh_oos, `results/rolling_eval_2026-10-15/metrics.csv`):

| model | set | λ | QLIKE | MSE ×1e10 | R²_OOS % | printed QLIKE | printed MSE ×1e10 | printed R²_OOS % |
|---|---|---|---|---|---|---|---|---|
| OLS | HAR benchmark | — | 0.115239 (0.115239) | 4.1530 (4.1530) | 78.209 (78.209) | 0.129117 (a0) | 4.8202 | 74.708 |
| ridge | HAR benchmark | 1e-2 | 0.115239 (0.115239) | 4.1531 (4.1531) | 78.209 (78.209) | 0.114306 | 4.7388 | 75.135 |
| lasso | HAR benchmark | 1e-4 | 0.115126 (0.115238) | 4.1522 (4.1530) | 78.213 (78.209) | 0.113548 | 4.9116 | 74.228 |
| elastic net | HAR benchmark | 1e-4 | 0.115180 (0.115239) | 4.1526 (4.1530) | 78.211 (78.209) | 0.114180 | 4.9357 | 74.102 |
| ridge | full set | 1e3 | 0.123014 (0.123014) | 6.9262 (6.9262) | 63.658 (63.658) | 0.108552 | 4.7931 | 74.850 |
| lasso | full set | 1e-3 | 0.121274 (0.121221) | 7.7994 (7.8214) | 59.076 (58.961) | 0.110148 | 4.6974 | 75.352 |
| elastic net | full set | 1e-3 | 0.122708 (0.114755) | 7.3836 (7.1513) | 61.258 (62.477) | 0.107841 | 4.7157 | 75.256 |

**Diebold–Mariano tests against the pooled OLS of this run** (`dm.csv`, HLN, Newey–West lag 6, T = 1,082; dm < 0 means
the model has the lower loss):

| model | set | DM QLIKE | p | mean QLIKE diff | DM sq. error | p |
|---|---|---|---|---|---|---|
| ridge | HAR benchmark | −3.10 | 0.0019 | −1.7e-8 | 0.76 | 0.44 |
| lasso | HAR benchmark | −4.71 | 2.5e-6 | −1.1e-4 | −0.62 | 0.54 |
| elastic net | HAR benchmark | −4.90 | 9.4e-7 | −6.0e-5 | −0.59 | 0.55 |
| ridge | full set | 1.49 | 0.14 | 7.8e-3 | 1.32 | 0.19 |
| lasso | full set | 1.38 | 0.17 | 6.0e-3 | 1.23 | 0.22 |
| elastic net | full set | 1.53 | 0.13 | 7.5e-3 | 1.28 | 0.20 |

The tests against the closing-interval OLS and against the printed rows are not computed: their forecast files
(`results/linear_subsection_dedup/.../incumbent_ols/results_bar1600.csv`, `results/spxw_pnl/yhat_*.parquet`) are not
in the repo. `dm.csv` records this for every row.

**Every interval of the sample, same 1,082 days** (`scores_all_bars.csv`, 14,066 intervals, actual RV of each
interval; committed run in parentheses):

| model | set | QLIKE | MSE ×1e10 |
|---|---|---|---|
| OLS | HAR benchmark | 0.192628 (0.192628) | 2.2087 (2.2087) |
| ridge | HAR benchmark | 0.192628 (0.192628) | 2.2087 (2.2087) |
| lasso | HAR benchmark | 0.192664 (0.192628) | 2.2081 (2.2087) |
| elastic net | HAR benchmark | 0.192646 (0.192628) | 2.2084 (2.2087) |
| ridge | full set | 0.186006 (0.186006) | 4.3848 (4.3848) |
| lasso | full set | 0.186682 (0.186646) | 4.2661 (4.2816) |
| elastic net | full set | 0.186067 (0.185922) | 4.6803 (3.7765) |

## Timings (this container: 4 CPUs, 15 GB, Intel Xeon @ 2.80 GHz, Python 3.12.3)

| step | time |
|---|---|
| design, both sets (2 jobs) | 411 s |
| pre-tune (4 jobs) | 43 s: full set ridge 5 s, lasso 40 s, elastic net 40 s; HAR benchmark < 1 s each |
| select | 1 s |
| walks (4 jobs) | 481 s wall. Full set: ridge 2 × 105–109 s; lasso 4 × 191–205 s; elastic net 4 × 141–170 s. HAR benchmark and OLS 3–16 s |
| score | 3 s |
| end to end | about 16 min wall |

**Library versions.** Python 3.12.3 venv: numpy 1.26.4 (OpenBLAS 0.3.23.dev, SkylakeX kernel), pandas 2.3.3, scipy
1.17.1, scikit-learn 1.9.1, numba 0.68.0, pyarrow 19.0.1; also tqdm 4.70.1 and statsmodels 0.15.0, which the runner
imports (`src/backtest/multi_stage.py`) and which SUMMARY.md of the committed run does not list.

## Files

- `frozen_penalties.json`, `pretune_mse.csv` (`picked_at_grid_end` is set on every row of a model whose pick is at an
  end of its grid), `scores.csv`, `dm.csv`, `scores_all_bars.csv`;
- `forecasts.parquet` (12 MB): 7 × 40,497 rows, 2011-12-28 14:00 .. 2024-04-30 16:00, the same columns as the
  committed run (`pred_adj`, `true_adj`, `B`, `plain`, `pred_raw_duan`, `true_raw`, `rv_actual`, plus `series_id`,
  `model`, `predictor_set`, `alpha`, `date`);
- `_work/` (gitignored, about 0.45 GB): design caches, pre-tune folds, walk tables, logs.
