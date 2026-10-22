# Pooled linear models, 13 regular-hours intervals: evaluation from 2012-01-03, 2,500-day window (2026-10-22)

This is a window sensitivity run. The user asked on 2026-10-22: "if 2k is the best window, is 1.5k the elbow? What does
2.5k days look like?" It is reported alongside the 1,000-, 1,500- and 2,000-day runs
(`results/pooled_linear_rerun_2026-10-20_13_2012_w1000`, `..._2026-10-22_13_2012_w1500`, `..._2026-10-20_13_2012_w2000`).
It is not a specification choice. No paper file, generator or generated file was touched.

## Configuration

Everything is as in the w1500 run (its SUMMARY.md) except the window and the pre-sample folds:

| | w1500 run | this run |
|---|---|---|
| `--window-days` | 1500 (19,500 rows) | **2500 (32,500 rows)** |
| pre-tune folds | 0–3 | **1–3** (`--first-fold 1`, below) |

Unchanged: `--first-day 2012-01-03`; `--sample rth13 --har-base 5 --ex-flag DROP_ACTIVITY_MA=1 --ex-flag
CALENDAR_THROUGH_OVERNIGHT=1 --ex-flag DROP_OVERNIGHT_INDICATOR=1 --ex-flag LEAN_GLOBAL_LAGS=1`; the full predictor set
with every FOMC series, `fomc_release` included; the seven models and the 0.01 penalty factor on the full set's
HAR-benchmark block; the grids; 250-session folds with a 25-session (325-row) embargo; frozen penalties
(`--sched const:<alpha>`) with the identifiability mask and a cold seed every 250 solves; chunking; the score stage. The
robust standardization and every model's window are 32,500 rows (`design.json` records `window: 32500`).

The code is the w1000 run's commit **eb78186** in the detached worktree `C:\Users\james\AppData\Local\Temp\w1500_wt`
(the same worktree, uv venv and `OPENBLAS_CORETYPE=Haswell` as the w1500 run), plus the `--first-fold` patch below
(`first_fold.patch` in this directory). Commands, run from the worktree
(`OUT=results/pooled_linear_rerun_2026-10-22_13_2012_w2500`, `F="--window-days 2500 --first-day 2012-01-03 --first-fold 1"`):

```
python experiments/pooled_linear_rerun_2026_10_19_13.py design  --jobs 2 --window-days 2500 --first-day 2012-01-03 --out $OUT
python experiments/pooled_linear_rerun_2026_10_19_13.py pretune --jobs 3 $F --out $OUT
python experiments/pooled_linear_rerun_2026_10_19_13.py select  $F --out $OUT
python experiments/pooled_linear_rerun_2026_10_19_13.py walk --chunks all_features/ridge=2 --chunks all_features/lasso=4 --chunks all_features/enet=4 --jobs 4 $F --out $OUT
python experiments/pooled_linear_rerun_2026_10_19_13.py score $F --out $OUT
```

(select ran twice: the second time after the patch's rule-text edit; `pretune_mse.csv` was byte-identical both times.)

## Pre-sample folds: only folds 1–3 can be fit

The folds tile the 1,000 sessions before 2012-01-03 (validation rows 31,281–44,280). Fold f is fit on the 32,500 rows
before a 325-row embargo, so its first fit row is 44,281 − (4 − f)·3,250 − 325 − 32,500. For fold 0 this is −1,544: the
window would start 1,544 rows (≈ 119 sessions) before the design's first row (1998-06-04). The unpatched orchestrator
stops at `first_row()`'s check; without that check the runner's design-cache slice would start at index −1,544, which
wraps to the end of the array.

**Patch** (worktree only, defaults unchanged):
- `experiments/pool_linear_pretune.py`, pretune: `--first-fold F` (default 0) fits folds F..3 only. The halo becomes
  (4 − F)·3,250 + 325 + W, and fold f's fit and validation rows are the same global rows as with F = 0.
- `experiments/pooled_linear_rerun_2026_10_19_13.py`: `--first-fold F` (default 0) shortens `PRE_HALO` the same way and
  passes the option to the runner; select labels the columns `mse_fold1..3` and records `info.first_fold`, a note, the
  rule text ("over pre-sample folds 1 .. 3 of 4") and the folds' fit/val rows.

**Patch checks.**
- At the 1,500-day window, the patched runner with `--first-fold 1` on the w1500 design cache reproduced the unpatched
  w1500 `fold.npz` folds 1–3 **bit for bit** for all six model × set jobs (residuals of every grid penalty, the grid,
  the global fit/val rows, the mask counts, the column names and the backbone mask).
- The patched select with the default (no `--first-fold`), on copies of the w1500 `fold.npz` and `design.json`, wrote
  `pretune_mse.csv` and `frozen_penalties.json` byte-identical (md5) to the w1500 run's.

**Rule.** Each penalty is the lowest mean validation MSE over folds 1–3 (adjusted scale, all 13 intervals).

| fold | fit rows | fit window | validation span |
|---|---|---|---|
| 0 | −1,544–30,956 | not fitted | 2008-01-15 12:30 .. 2009-01-13 13:00 |
| 1 | 1,706–34,206 | 1998-12-09 15:30 .. 2008-12-04 13:00 | 2009-01-13 13:30 .. 2010-01-11 14:00 |
| 2 | 4,956–37,456 | 1999-12-08 13:30 .. 2009-12-02 14:00 | 2010-01-11 14:30 .. 2011-01-05 15:00 |
| 3 | 8,206–40,706 | 2000-12-05 14:00 .. 2010-11-30 15:00 | 2011-01-05 15:30 .. 2011-12-30 16:00 |

Columns masked in the pre-tune fit blocks (full set): 299, 276 and 259 in folds 1–3; none on the HAR benchmark.

## Gates

**(1) Design columns.** Both designs have 84,747 rows. The full set's 576 column names, in order, equal
`designs.all_features.columns` of the w1000 run's `frozen_penalties.json` (and of the w1500 and w2000 runs'), with the
12 `fomc_release` columns included; the HAR benchmark's 12 names also equal theirs.

**(2) `check_design`** passed at the design, pretune and walk stages.

**(3) Scoring.** The window scoring script (`score_windows.py`, unchanged) re-run on w1000/w1500/w2000 reproduced its
earlier output exactly (279 of 279 numeric cells), and its per-run checks reproduce `scores.csv` QLIKE to 1e-10, w2500
included.

## Frozen penalties

| model | set | w1000 | w1500 | w2000 | **w2500** | grid | grid edge (w2500) |
|---|---|---|---|---|---|---|---|
| ridge | HAR benchmark | 1e-2 | 1e-2 | 1e-2 | **1e-2** | 1e-2 .. 1e3 | bottom |
| lasso | HAR benchmark | 1e-4 | 1e-4 | 1e-4 | **1e-5** | 1e-6 .. 1e-2 | no |
| elastic net | HAR benchmark | 1e-6 | 1e-4 | 1e-4 | **1e-6** | 1e-6 .. 1e-2 | bottom |
| ridge | full set | 1e3 | 1e3 | 1e3 | **1e3** | 1e-2 .. 1e3 | top |
| lasso | full set | 1e-3 | 1e-3 | 1e-3 | **1e-3** | 1e-6 .. 1e-2 | no |
| elastic net | full set | 1e-2 | 1e-3 | 1e-3 | **1e-3** | 1e-6 .. 1e-2 | no |

- HAR benchmark: the criterion is flat, 0.0625324 to 6 significant digits.
- Full set: ridge 0.0600012 at 1e2, 0.0586088 at 1e3 (still falling at the top); lasso 0.0583698 at 1e-4, 0.0582868 at
  1e-3, 0.0597641 at 1e-2; elastic net 0.0596014 at 1e-4, 0.0580873 at 1e-3, 0.0589138 at 1e-2.
- At the smallest penalties the full set's fold-3 validation MSE (2011) is large: ridge 1e-2 1.117, lasso 1e-6 0.920,
  elastic net 1e-6 0.990 (folds 1–2: 0.08–0.14).

## Pre-sample criterion, four windows (folds 1–3)

The three validation spans (rows 34,531–44,280) and the target `y` are the same in all four runs.

| model | set | w1000 | w1500 | w2000 | w2500 |
|---|---|---|---|---|---|
| ridge | HAR | 0.0632788 | 0.0633798 | 0.0630804 | 0.0625324 |
| lasso | HAR | 0.0632798 | 0.0633812 | 0.0630797 | 0.0625324 |
| elastic net | HAR | 0.0632788 | 0.0633809 | 0.0630804 | 0.0625324 |
| ridge | full | 0.0591473 | 0.0591369 | 0.0589113 | 0.0586088 |
| lasso | full | 0.0589738 | 0.0587882 | 0.0587339 | 0.0582868 |
| elastic net | full | 0.0594098 | 0.0586423 | 0.0584241 | 0.0580873 |

(at each run's frozen penalty). The folds-1–3 rule would change four earlier picks: w1000 lasso HAR 1e-4 → 1e-6, w1000
elastic net full 1e-2 → 1e-3, w1500 lasso HAR and elastic net HAR 1e-4 → 1e-6. On the HAR benchmark the best and
second-best grid values differ by 7e-9 to 7e-7 in every window; on the full set by 8e-5 to 1.5e-3.

## Rolling-window exercise

The walks start at design row 44,250 (2011-12-28 14:00), as in the other three runs, and end on 2024-04-30 16:00; the
first window is rows 11,750–44,249 (2002-01-16 11:30 .. 2011-12-28 13:30). Every model has **40,497** forecasts, all
finite and positive; dates, `true_adj`, `B`, `true_raw` and `rv_actual` equal the w1500 run's. Chunks: full-set ridge
in 2 at row 64,500; lasso and elastic net in 4 at rows 54,500, 64,750 and 75,000; each with a 32,500-row halo. A single
full-set lasso walk across the c0/c1 boundary reproduces the stitched `pred_adj`, `true_adj` and `true_raw` bit for bit.
KKT certification had no failures.

## Score stage: closing half hour, 2020-01-03 .. 2024-04-30 (1,082 days)

| model | set | λ w2500 | QLIKE w1000 | w1500 | w2000 | **w2500** | MSE ×1e10 w1000 | w1500 | w2000 | **w2500** | R²_OOS % w1000 | w1500 | w2000 | **w2500** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| OLS | HAR | — | 0.12045 | 0.11652 | 0.11524 | 0.11437 | 3.925 | 4.108 | 4.153 | 4.236 | 79.40 | 78.44 | 78.21 | 77.77 |
| ridge | HAR | 1e-2 | 0.12045 | 0.11652 | 0.11524 | 0.11437 | 3.925 | 4.108 | 4.153 | 4.236 | 79.40 | 78.44 | 78.21 | 77.77 |
| lasso | HAR | 1e-5 | 0.12030 | 0.11640 | 0.11513 | 0.11436 | 3.922 | 4.107 | 4.152 | 4.236 | 79.42 | 78.45 | 78.21 | 77.77 |
| elastic net | HAR | 1e-6 | 0.12045 | 0.11646 | 0.11518 | 0.11437 | 3.925 | 4.108 | 4.153 | 4.236 | 79.40 | 78.45 | 78.21 | 77.77 |
| ridge | full | 1e3 | 0.12362 | 0.12174 | 0.12301 | 0.11935 | 6.253 | 6.584 | 6.926 | 7.483 | 67.19 | 65.46 | 63.66 | 60.74 |
| lasso | full | 1e-3 | 0.12292 | 0.12163 | 0.12127 | 0.11515 | 6.800 | 7.335 | 7.799 | 8.299 | 64.32 | 61.51 | 59.08 | 56.46 |
| elastic net | full | 1e-3 | 0.11940 | 0.12297 | 0.12271 | 0.11685 | 6.485 | 6.852 | 7.384 | 7.985 | 65.97 | 64.05 | 61.26 | 58.10 |

Every interval of the same 1,082 days (`scores_all_bars.csv`), QLIKE w1000 / w1500 / w2000 / w2500: OLS HAR 0.19511 /
0.19187 / 0.19263 / 0.19275; ridge full 0.18689 / 0.18461 / 0.18601 / 0.18535; lasso full 0.18742 / 0.18554 / 0.18668 /
0.18501; elastic net full 0.18737 / 0.18480 / 0.18607 / 0.18444.

## Window sensitivity: full set against HAR benchmark, four windows

(a) Closing-interval QLIKE, 16:00 rows, 1,082 days; Δ% = 100 (QLIKE_full / QLIKE_HAR − 1); DM on the daily QLIKE
difference (lag 6, HLN).

| model | window | QLIKE HAR | QLIKE full | Δ% | DM | p |
|---|---|---|---|---|---|---|
| elastic net | w1000 | 0.12045 | 0.11940 | −0.87 | −0.31 | 0.758 |
| elastic net | w1500 | 0.11646 | 0.12297 | +5.59 | 1.30 | 0.192 |
| elastic net | w2000 | 0.11518 | 0.12271 | +6.54 | 1.54 | 0.122 |
| elastic net | w2500 | 0.11437 | 0.11685 | +2.17 | 0.55 | 0.584 |
| lasso | w1000 | 0.12030 | 0.12292 | +2.18 | 0.58 | 0.564 |
| lasso | w1500 | 0.11640 | 0.12163 | +4.49 | 1.21 | 0.225 |
| lasso | w2000 | 0.11513 | 0.12127 | +5.34 | 1.41 | 0.159 |
| lasso | w2500 | 0.11436 | 0.11515 | +0.69 | 0.21 | 0.835 |
| ridge | w1000 | 0.12045 | 0.12362 | +2.63 | 0.61 | 0.542 |
| ridge | w1500 | 0.11652 | 0.12174 | +4.48 | 1.11 | 0.267 |
| ridge | w2000 | 0.11524 | 0.12301 | +6.75 | 1.49 | 0.137 |
| ridge | w2500 | 0.11437 | 0.11935 | +4.35 | 0.95 | 0.343 |

(b) Sign strategy on the 866 option days as in `make_rehaul_economics.py` (`add_pooled_linear` / `lin_16`); statistics
on the 614 post-warmup days. SR: per-year annualized (`sharpe_ann_by_year`); LW p: `lw_sharpe_test` (block 10, 4,999
resamples); LLSW: `llsw_test` of the mean return differential (S = 33, fixed-b p); SR₆₁₄: mean / sd · √252 over the
614 days.

| model | window | SR HAR | SR full | ΔSR mid | LW p | LLSW t (p) | ΔSR b/a | LW p b/a | ΔSR₆₁₄ mid | long HAR / full | log(plain/rv) HAR / full |
|---|---|---|---|---|---|---|---|---|---|---|---|
| elastic net | w1000 | 1.30 | 1.13 | −0.17 | 0.754 | 0.33 (0.754) | −0.17 | 0.753 | +0.11 | 0.187 / 0.178 | −0.045 / −0.056 |
| elastic net | w1500 | 1.22 | 1.63 | +0.41 | 0.156 | 1.47 (0.162) | +0.40 | 0.155 | +0.63 | 0.230 / 0.191 | −0.014 / −0.055 |
| elastic net | w2000 | 1.07 | 1.30 | +0.23 | 0.272 | 1.06 (0.312) | +0.22 | 0.274 | +0.49 | 0.238 / 0.178 | −0.002 / −0.062 |
| elastic net | w2500 | 0.96 | 1.58 | +0.62 | 0.071 | 1.55 (0.143) | +0.61 | 0.072 | +0.87 | 0.236 / 0.192 | +0.000 / −0.046 |
| lasso | w1000 | 1.28 | 1.02 | −0.26 | 0.996 | −0.01 (0.995) | −0.25 | 0.996 | −0.00 | 0.189 / 0.182 | −0.044 / −0.065 |
| lasso | w1500 | 1.22 | 1.55 | +0.33 | 0.191 | 1.24 (0.237) | +0.32 | 0.194 | +0.54 | 0.230 / 0.182 | −0.013 / −0.061 |
| lasso | w2000 | 1.07 | 1.23 | +0.16 | 0.335 | 0.92 (0.378) | +0.16 | 0.338 | +0.42 | 0.238 / 0.178 | −0.001 / −0.065 |
| lasso | w2500 | 0.96 | 1.71 | +0.76 | 0.029 | 2.04 (0.057) | +0.75 | 0.028 | +1.09 | 0.236 / 0.202 | +0.000 / −0.046 |
| ridge | w1000 | 1.30 | 1.09 | −0.22 | 0.963 | −0.05 (0.962) | −0.21 | 0.965 | −0.02 | 0.187 / 0.191 | −0.045 / −0.054 |
| ridge | w1500 | 1.22 | 1.61 | +0.39 | 0.191 | 1.20 (0.254) | +0.39 | 0.191 | +0.57 | 0.230 / 0.200 | −0.014 / −0.049 |
| ridge | w2000 | 1.07 | 1.49 | +0.42 | 0.145 | 1.39 (0.188) | +0.41 | 0.147 | +0.65 | 0.238 / 0.187 | −0.002 / −0.054 |
| ridge | w2500 | 0.96 | 1.62 | +0.67 | 0.060 | 1.62 (0.126) | +0.66 | 0.061 | +0.91 | 0.236 / 0.194 | +0.000 / −0.046 |

w2500 SR bid/ask, HAR / full: 0.62 / 1.23 (elastic net), 0.62 / 1.37 (lasso), 0.62 / 1.28 (ridge).

## Longer windows

At 3,000 days (39,000 rows) only fold 3 can be fit (folds 0–2 would start at rows −8,044, −4,794, −1,544); the walk
itself fits. Largest window for each fold set: all 4 folds 2,381 days; folds 1–3 2,631; folds 2–3 2,881; fold 3 only
3,131; the walk alone 3,406.

## Timings and environment

Intel Core i7-1355U (10 cores / 12 threads, 16 GB), another session running 4–6 heavy jobs throughout: design 1,398 s
(full) and 65 s (HAR); pre-tune 115 s; walks 2,149 s wall (4 jobs); score 3 s; about 63 min end to end. The w1500 run's
uv venv (Python 3.12.3, numpy 1.26.4 / OpenBLAS 0.3.23.dev, pandas 2.3.3, scipy 1.17.1, numba 0.68.0),
`OPENBLAS_CORETYPE=Haswell`. Window scoring on `miniconda3/envs/285J`.

## Files

`frozen_penalties.json` (with `info.first_fold`), `pretune_mse.csv` (columns `mse_fold1..3`), `scores.csv`, `dm.csv`,
`scores_all_bars.csv`, `forecasts.parquet` (every interval of every model, 2011-12-28 14:00 .. 2024-04-30 16:00),
`first_fold.patch` (the `--first-fold` patch on eb78186). Design caches and logs stayed in the worktree's `_work/`.
