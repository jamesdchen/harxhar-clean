# Pooled linear models, 13 regular-hours intervals: evaluation from 2012-01-03, 1,500-day window (2026-10-22)

This is a window sensitivity run. The user asked on 2026-10-22: "If the training window needs to be >1000 days, can we
test 1.5k days?" It is reported alongside the 1,000-day run (`results/pooled_linear_rerun_2026-10-20_13_2012_w1000`)
and the 2,000-day run (`results/pooled_linear_rerun_2026-10-20_13_2012_w2000`). It is not a specification choice. No
paper file, generator or generated file was touched.

## Configuration

Everything is as in the w1000 run (its SUMMARY.md and `frozen_penalties.json` `info.runner_cfg`) except one option of
`experiments/pooled_linear_rerun_2026_10_19_13.py`:

| | w1000 run | this run |
|---|---|---|
| `--window-days` | 1000 (13,000 rows) | **1500 (19,500 rows)** |

Unchanged:
- `--first-day 2012-01-03`;
- the runner options `--sample rth13 --har-base 5 --ex-flag DROP_ACTIVITY_MA=1 --ex-flag CALENDAR_THROUGH_OVERNIGHT=1
  --ex-flag DROP_OVERNIGHT_INDICATOR=1 --ex-flag LEAN_GLOBAL_LAGS=1` (the orchestrator appends `--window-days 1500`);
- the full predictor set with every FOMC series, `fomc_release` included;
- the seven models, and the penalty weight 0.01 on the HAR-benchmark block of the full set;
- the grids, and the 4 × 250-session pre-tune folds with a 25-session embargo;
- frozen penalties (`--sched const:<alpha>`), with the identifiability mask and a cold seed every 250 solves;
- the chunking and the score stage.

The robust standardization and the window of every model, the OLS included, are 19,500 rows (`design.json` records
`window: 19500`).

**The code is the w1000 run's commit.** HEAD of the main worktree has `EXCLUDE_FOMC_RELEASE = True` in
`src/data/loading.py` (0250f44), which drops `fomc_release` (576 → 564 columns); the w1000 and w2000 runs were fitted
with it. So this run was made in a detached worktree at **eb78186**, the w1000 run's commit (branch
`insurance-selling-2026-09-19-rerun-2012-w1000`), at `C:\Users\james\AppData\Local\Temp\w1500_wt`. Between eb78186 and
HEAD, the only change under `src/`, `data/`, `specs/causal_tune_linear.py`, `experiments/pool_linear_pretune.py` and
the orchestrator is that flag. `load_raw_data` merges every `.parquet` in `data/`; every one of them is tracked at
eb78186 and `git diff eb78186 HEAD -- data/` is empty (the main worktree's untracked `data/spxw_chain.parquet` and
`data/spxw_spot.parquet` were absent, as from the w1000 run's clone).

Commands, run from the worktree (`OUT=results/pooled_linear_rerun_2026-10-22_13_2012_w1500`,
`F="--window-days 1500 --first-day 2012-01-03"`, environment `OPENBLAS_CORETYPE=Haswell`):

```
python experiments/pooled_linear_rerun_2026_10_19_13.py design  --jobs 2 $F --out $OUT
python experiments/pooled_linear_rerun_2026_10_19_13.py pretune --jobs 4 $F --out $OUT
python experiments/pooled_linear_rerun_2026_10_19_13.py select  $F --out $OUT
python experiments/pooled_linear_rerun_2026_10_19_13.py walk --chunks all_features/ridge=2 --chunks all_features/lasso=4 --chunks all_features/enet=4 --jobs 4 $F --out $OUT
python experiments/pooled_linear_rerun_2026_10_19_13.py score $F --out $OUT
```

The result files were copied into this directory of the main worktree; `_work/` (design caches, pre-tune folds, walk
tables, logs) stayed in the worktree.

## Gates

**(1) Design columns.** Both designs have 84,747 rows, as in the w1000 and w2000 runs. The full set's 576 column names,
in order, equal `designs.all_features.columns` of both runs' `frozen_penalties.json` (the 12 `fomc_release` value and
availability columns included); the HAR benchmark's 12 names also equal both runs'. `check_design` passed at every
stage.

**(2) Reproduction of the w1000 run at `--window-days 1000`.** The same worktree and environment built a 1,000-day
HAR-benchmark design and walked ridge, lasso and elastic net at the w1000 frozen penalties, and the OLS, with the
orchestrator's own `walk_jobs` / `pool`. Against the w1000 `forecasts.parquet` (40,497 rows a model):

| quantity | ridge (1e-2) | lasso (1e-4) | elastic net (1e-6) | OLS |
|---|---|---|---|---|
| dates, `true_adj`, `B`, `true_raw` | bit for bit | bit for bit | bit for bit | bit for bit |
| `plain`, max relative difference | **1.2e-9** | 1.8e-14 | 1.9e-14 | 1.5e-14 |
| `pred_adj`, max (median) relative difference | 5.8e-10 (3.5e-11) | 8.8e-15 (3.3e-16) | 9.5e-15 (3.3e-16) | 7.1e-15 (4.4e-16) |
| mean QLIKE, 16:00 rows 2012–2024, w1000 / here | 0.132807137147 / 0.132807137149 | equal to 12 digits | equal to 12 digits | equal to 12 digits |

OLS, lasso and elastic net reproduce to ≤ 2e-14. **Ridge on the HAR benchmark does not meet the ~1e-12 criterion:**
its largest difference is 1.2e-9. The difference is already present on the first solve after each cold seed and does
not grow inside the 250-solve block (median 3.4e-11 in solves 0–49, 3.6e-11 in 200–249). The cause is conditioning: at
α = 1e-2 the penalized Gram (DOW_0..DOW_4 sum to the intercept) is badly conditioned, so the seed's `np.linalg.inv`
carries BLAS/LAPACK rounding at this scale. On this machine the same bit-identical design walked with OpenBLAS's
default kernel (Prescott) instead of Haswell moves ridge by up to 7.6e-10 and the other three arms by ≤ 4.1e-14; the
w2000 SUMMARY reports 3.4e-9 for this arm between two machines. 1e-12 is not reachable for this arm without the w1000
container's AVX-512 kernel. The run went ahead; this gate is reported as not met for ridge.

**Grid values (one ulp).** Here `np.logspace` gives 1e-5, 1e-4 and 0.1 exactly; on the w1000/w2000 container they were
one ulp low (9.999999999999999e-06, 9.999999999999999e-05, 0.09999999999999999). The walk asserts that
`--sched const:A` is a grid point, so this run's picks at 1e-4 are recorded as 0.0001. No grid, penalty or feature was
changed.

**(3) Scoring.** The window scoring script reproduces the w1000 Sharpe ratios at mid, HAR / full: elastic net
1.30 / 1.13, lasso 1.28 / 1.02, ridge 1.30 / 1.09, and every run's `scores.csv` QLIKE to 1e-10.

## Pre-tune folds

The 4 folds tile the 1,000 sessions before 2012-01-03 (design rows 31,281–44,280), as in the other two runs. Each is fit
on the 19,500 rows before a 325-row embargo; the first fit row is 11,456 ≥ 0 (the design starts on 1998-06-04).

| fold | fit rows | fit window | validation span |
|---|---|---|---|
| 0 | 11,456–30,956 | 2001-12-11 14:00 .. 2007-12-06 12:00 | 2008-01-15 12:30 .. 2009-01-13 13:00 |
| 1 | 14,706–34,206 | 2002-12-13 10:00 .. 2008-12-04 13:00 | 2009-01-13 13:30 .. 2010-01-11 14:00 |
| 2 | 17,956–37,456 | 2003-12-16 10:00 .. 2009-12-02 14:00 | 2010-01-11 14:30 .. 2011-01-05 15:00 |
| 3 | 21,206–40,706 | 2004-12-14 16:00 .. 2010-11-30 15:00 | 2011-01-05 15:30 .. 2011-12-30 16:00 |

## Frozen penalties

Mean validation MSE over the 4 folds, adjusted scale, all 13 intervals (`pretune_mse.csv`).

| model | set | w1500 | w1000 | w2000 | grid | grid edge (w1500) |
|---|---|---|---|---|---|---|
| ridge | HAR benchmark | 1e-2 | 1e-2 | 1e-2 | 1e-2 .. 1e3 | bottom |
| lasso | HAR benchmark | 1e-4 | 1e-4 | 1e-4 | 1e-6 .. 1e-2 | no |
| elastic net | HAR benchmark | 1e-4 | 1e-6 (bottom) | 1e-4 | 1e-6 .. 1e-2 | no |
| ridge | full set | 1e3 | 1e3 | 1e3 | 1e-2 .. 1e3 | top |
| lasso | full set | 1e-3 | 1e-3 | 1e-3 | 1e-6 .. 1e-2 | no |
| elastic net | full set | 1e-3 | 1e-2 (top) | 1e-3 | 1e-6 .. 1e-2 | no |

- The w1500 picks equal the w2000 picks in all six cases; they differ from w1000 for the elastic net on both sets.
- HAR benchmark: the criterion is flat, 0.066204 to 6 digits; the minima are lasso 0.0662024 and elastic net
  0.0662033, both at 1e-4.
- Full set: ridge 0.062616 at 1e2 and 0.061987 at 1e3 (still falling at the top); lasso 0.063086 at 1e-4, 0.061450 at
  1e-3, 0.062650 at 1e-2; elastic net 0.075493 at 1e-4, 0.061426 at 1e-3, 0.061837 at 1e-2.
- At the smallest penalties the full set's fold-3 validation MSE (2011) is very large (ridge 1e-2: 155.2; lasso 1e-6:
  144.3; elastic net 1e-6: 141.9; other folds 0.09–0.34). In the w1000 run the large fold was fold 2 (the late-start
  series scaling problem, writeup/LINEAR_SWITCH_AUDIT.md finding 2).
- Columns masked in the pre-tune fit blocks (full set): 304, 305, 283 and 266 in folds 0–3; none on the HAR benchmark.

## Rolling-window exercise

The walks start at design row 44,250 (2011-12-28 14:00), as in w1000 and w2000, and end on 2024-04-30 16:00. Every model
has **40,497** forecasts, all finite and positive (`forecasts.parquet`: 7 × 40,497 = 283,479 rows). Chunks: full-set
ridge split in 2 at row 64,500; lasso and elastic net in 4 at rows 54,500, 64,750 and 75,000; each chunk with a
19,500-row halo. A single full-set lasso walk across the c0/c1 boundary (rows 54,250–54,749) reproduces the stitched
`pred_adj` and `true_adj` bit for bit (`pred_raw_duan` differs by up to 1.5e-7; the score stage does not use it). KKT
certification of the full-set batch solves had no failures.

## Score stage: closing half hour, 2020-01-03 .. 2024-04-30 (1,082 days)

| model | set | λ | QLIKE w1500 | w1000 | w2000 | MSE ×1e10 w1500 | w1000 | w2000 | R²_OOS % w1500 | w1000 | w2000 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| OLS | HAR | — | 0.11652 | 0.12045 | 0.11524 | 4.108 | 3.925 | 4.153 | 78.44 | 79.40 | 78.21 |
| ridge | HAR | 1e-2 | 0.11652 | 0.12045 | 0.11524 | 4.108 | 3.925 | 4.153 | 78.44 | 79.40 | 78.21 |
| lasso | HAR | 1e-4 | 0.11640 | 0.12030 | 0.11513 | 4.107 | 3.922 | 4.152 | 78.45 | 79.42 | 78.21 |
| elastic net | HAR | 1e-4 | 0.11646 | 0.12045 | 0.11518 | 4.108 | 3.925 | 4.153 | 78.45 | 79.40 | 78.21 |
| ridge | full | 1e3 | 0.12174 | 0.12362 | 0.12301 | 6.584 | 6.253 | 6.926 | 65.46 | 67.19 | 63.66 |
| lasso | full | 1e-3 | 0.12163 | 0.12292 | 0.12127 | 7.335 | 6.800 | 7.799 | 61.51 | 64.32 | 59.08 |
| elastic net | full | 1e-3 | 0.12297 | 0.11940 | 0.12271 | 6.852 | 6.485 | 7.384 | 64.05 | 65.97 | 61.26 |

Every interval, same 1,082 days (`scores_all_bars.csv`, 14,066 intervals), QLIKE w1500 / w1000 / w2000: OLS HAR 0.19187 /
0.19511 / 0.19263; ridge full 0.18461 / 0.18689 / 0.18601; lasso full 0.18554 / 0.18742 / 0.18668; elastic net full
0.18480 / 0.18737 / 0.18607.

## Window sensitivity: full set against HAR benchmark, three windows

(a) Closing-interval QLIKE, 16:00 rows, 1,082 days; Δ% = 100 (QLIKE_full / QLIKE_HAR − 1); DM on the daily QLIKE
difference (lag 6, HLN).

| model | window | QLIKE HAR | QLIKE full | Δ% | DM | p |
|---|---|---|---|---|---|---|
| elastic net | w1000 | 0.12045 | 0.11940 | −0.87 | −0.31 | 0.758 |
| elastic net | w1500 | 0.11646 | 0.12297 | +5.59 | 1.30 | 0.192 |
| elastic net | w2000 | 0.11518 | 0.12271 | +6.54 | 1.54 | 0.122 |
| lasso | w1000 | 0.12030 | 0.12292 | +2.18 | 0.58 | 0.564 |
| lasso | w1500 | 0.11640 | 0.12163 | +4.49 | 1.21 | 0.225 |
| lasso | w2000 | 0.11513 | 0.12127 | +5.34 | 1.41 | 0.159 |
| ridge | w1000 | 0.12045 | 0.12362 | +2.63 | 0.61 | 0.542 |
| ridge | w1500 | 0.11652 | 0.12174 | +4.48 | 1.11 | 0.267 |
| ridge | w2000 | 0.11524 | 0.12301 | +6.75 | 1.49 | 0.137 |

(b) Sign strategy on the 866 option days as in `make_rehaul_economics.py` (`add_pooled_linear` / `lin_16`); statistics on
the 614 post-warmup days. SR: per-year annualized (`sharpe_ann_by_year`); LW p: `lw_sharpe_test` (block 10, 4,999
resamples); LLSW: `llsw_test` of the mean return differential (S = 33, fixed-b p); SR₆₁₄: mean / sd · √252 over the 614
days.

| model | window | SR HAR | SR full | ΔSR mid | LW p | LLSW t (p) | ΔSR b/a | LW p b/a | ΔSR₆₁₄ mid | long HAR / full | log(plain/rv) HAR / full |
|---|---|---|---|---|---|---|---|---|---|---|---|
| elastic net | w1000 | 1.30 | 1.13 | −0.17 | 0.754 | 0.33 (0.754) | −0.17 | 0.753 | +0.11 | 0.187 / 0.178 | −0.045 / −0.056 |
| elastic net | w1500 | 1.22 | 1.63 | +0.41 | 0.156 | 1.47 (0.162) | +0.40 | 0.155 | +0.63 | 0.230 / 0.191 | −0.014 / −0.055 |
| elastic net | w2000 | 1.07 | 1.30 | +0.23 | 0.272 | 1.06 (0.312) | +0.22 | 0.274 | +0.49 | 0.238 / 0.178 | −0.002 / −0.062 |
| lasso | w1000 | 1.28 | 1.02 | −0.26 | 0.996 | −0.01 (0.995) | −0.25 | 0.996 | −0.00 | 0.189 / 0.182 | −0.044 / −0.065 |
| lasso | w1500 | 1.22 | 1.55 | +0.33 | 0.191 | 1.24 (0.237) | +0.32 | 0.194 | +0.54 | 0.230 / 0.182 | −0.013 / −0.061 |
| lasso | w2000 | 1.07 | 1.23 | +0.16 | 0.335 | 0.92 (0.378) | +0.16 | 0.338 | +0.42 | 0.238 / 0.178 | −0.001 / −0.065 |
| ridge | w1000 | 1.30 | 1.09 | −0.22 | 0.963 | −0.05 (0.962) | −0.21 | 0.965 | −0.02 | 0.187 / 0.191 | −0.045 / −0.054 |
| ridge | w1500 | 1.22 | 1.61 | +0.39 | 0.191 | 1.20 (0.254) | +0.39 | 0.191 | +0.57 | 0.230 / 0.200 | −0.014 / −0.049 |
| ridge | w2000 | 1.07 | 1.49 | +0.42 | 0.145 | 1.39 (0.188) | +0.41 | 0.147 | +0.65 | 0.238 / 0.187 | −0.002 / −0.054 |

## Availability of the late-start exogenous series

Share of rows with an observation in the series' last interval (`<series>_avail_ma_1` > 0, full-set design):

| series | first observed | first fit window (2006-01-12 .. 2011-12-28) | first eval session | first 250 eval sessions |
|---|---|---|---|---|
| VVIX | 2012-03-26 11:00 | **0** | **0** | 0.70 |
| volatility demand (4 series) | 2011-01-03 10:30 | 0.15 | 0.92 | 0.91 |
| VIX3M | 2009-08-12 11:00 | 0.365 | 0.92 | 0.92 |
| StockTwits sentiment, sentiment count | 2010-06-02 11:30 | 0.22 | 0.77 | 0.95 |
| StockTwits attention | 2009-03-23 15:30 | 0.42 | 1.00 | 1.00 |
| VIX | 2006-05-22 11:00 | 0.86 | 0.92 | 0.92 |

## Timings and environment

Intel Core i7-1355U (10 cores / 12 threads, 16 GB; other sessions active): design 844 s (full) / 57 s (HAR); pre-tune
86 s wall; walks 27.0 min wall (4 jobs; full-set jobs sum to 5,570 s); score 5 s; about 44 min end to end. A uv venv
with the w1000 versions: Python 3.12.3, numpy 1.26.4 (OpenBLAS 0.3.23.dev), pandas 2.3.3, scipy 1.17.1, scikit-learn
1.9.1, numba 0.68.0, pyarrow 19.0.1, tqdm 4.70.1; `OPENBLAS_CORETYPE=Haswell`. Window scoring ran on
`miniconda3/envs/285J` (Python 3.12.0, numpy 2.2.6, pandas 2.3.0).

## Files

`frozen_penalties.json`, `pretune_mse.csv`, `scores.csv`, `dm.csv`, `scores_all_bars.csv`, `forecasts.parquet` (every
interval of every model, 2011-12-28 14:00 .. 2024-04-30 16:00, the w1000 columns). Scratch scripts (`gate_w1000.py`,
`gate_ridge_diag.py`, `chunk_check.py`, `score_windows.py`, `score_windows.csv`) are in the session scratchpad
`w1500\` folder.
