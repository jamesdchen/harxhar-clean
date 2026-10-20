# Pooled trees on the design of the pooled linear 13-bar rerun: runner, Slurm files, tests (2026-10-20)

Branch `insurance-selling-2026-09-19-trees-rerun13`, from `insurance-selling-2026-09-19` at 470e922. Nothing under
`writeup/` was touched. The full walks were not run here; they run on CARC Discovery
(`cluster/slurm/pooltree_rerun13.sbatch`). This file records the tests and timing probes run in this container.

## What was written

- `experiments/pooled_trees_rerun_13.py`: the pooled LightGBM, XGBoost and random forest, fitted on the design caches of
  `experiments/pooled_linear_rerun_2026_10_19_13.py design` (the same X, y, B and rows as the linear models; nothing is
  rebuilt).
  - The linear runner's `check_design` runs on each cache: the HAR columns are exactly `har_ma_{1,5,25,125,625,3125}`,
    the calendar is DOW_0..DOW_4 and hour, and every FOMC series is present. It also asserts 12 and 576 columns.
  - Window: W = window_days x 13 rows, read from the cache's `design.json` and asserted against `--window-days`. The
    forecast of design row r is fitted on rows [r - W, r): the rows the linear walk uses at r.
  - Models: the frozen configs `cluster/pooltree_lgbm_config.json` (lgbm_opt_00, sha abab811bd4b6),
    `cluster/xgbtop_xgb_config.json` (xgb_opt_00, 88bec9f385c7) and `cluster/pooltree_rf400_config.json`
    (rf400_rf_opt_00, 88eebb5fa9e8), constructed as `experiments/pooled_trees_base5.py` does: seed 42, LightGBM
    `num_threads=T`, XGBoost `n_jobs=T`, random forest `n_jobs=T` to fit and `n_jobs=1` to predict.
  - `--refit bar`: a refit at every row, the linear models' cadence. `--refit day`: a refit at the first row of each
    trading day, whose fit forecasts every row of that day. Rows are grouped by date: in dates_oos, 4,471 days have 13
    rows, and 110 have 1, 4, 7 or 8.
  - `--first-day`, `--last-day` (default: the last design row) and `--part i/n` (the i-th of n contiguous blocks of
    trading days). The tasks carry no state from one to another.
  - Modes:
    - `walk` writes pieces `<OUT>/pieces/<est>_<set>_<refit>_<first>_<last>_<i>of<n>.parquet` with columns model,
      predictor_set, date, pred_adj, true_adj, B, plain = pred_adj² x B, rv_actual (core_stats sumret2 of the bar) and
      fit_row (the first row of the fit window).
      - Each piece has a `.json` meta: config, sha, config file sha256, window, refit, threads, host, CPU model, library
        versions, wall seconds, refits and the Slurm ids.
      - A checkpoint is written every `--ckpt-sec` seconds (default 300) at a refit boundary. A rerun skips the rows
        already in the piece.
    - `merge` collects all pieces of a span and cadence into `<OUT>/forecasts.parquet`. It asserts that every row of
      the span is present exactly once and that all values are finite.
    - `time` reports seconds a refit and the projected CPU-hours and wall hours.
    - `tasks` writes the task file. It chooses n for each arm so that a task's projected wall time is near
      `--target-hours`, and puts the 2020-01-03..2024-04-30 lines first, then 2012-01-03..2019-12-31.
- `cluster/slurm/pooltree_rerun13.sbatch`: `MODE` = walk (array, the default), design, time or merge. The header holds
  the submission commands.
- `experiments/score_13bar_span.py` scores every (model, predictor_set) pair a forecasts.parquet holds. The linear pairs
  come first in Table 1's order, then the others. The "all features vs HAR benchmark" tests run for every model with both
  sets. The "vs OLS" tests run when the run holds the OLS. `--compare` covers the pairs both runs hold.
- `.gitignore`: `results/pooled_trees_rerun13_2026-10-20/_work/` (the design caches).

## Environment

This container has 4 CPUs (Intel Xeon @ 2.10GHz) and 15 GB. Everything ran in a Python 3.12.3 venv with numpy 1.26.4,
pandas 2.3.3, scipy 1.17.1, scikit-learn 1.9.1, numba 0.68.0, pyarrow 19.0.1, lightgbm 4.6.0 and xgboost 3.2.0. xgboost
3.2.0 imports and runs under numpy 1.26.4. tqdm and pyyaml were also installed, because the design stage imports them.

## 1. Design caches (2,000-day window, default options)

`python experiments/pooled_linear_rerun_2026_10_19_13.py design --jobs 2 --out results/pooled_trees_rerun13_2026-10-20`
(36 s and 352 s). Both caches have 84,747 rows (W = 26,000 + 58,747 OOS rows, 2006-05-31 16:00 .. 2024-04-30 16:00).
The baseline has 12 columns and all_features has 576. X is column-major, as the linear runner saves it.

| set | file | sha256 |
|---|---|---|
| baseline | X.npy | `5a1b8492837c8cc37642b891016988f4ef4dd595b9d31be3834905c38bc09bf5` |
| baseline | y.npy | `62285d583262c6fb11f99bbc7108397948f64faab6e4793eccf8ef741a10f588` |
| baseline | b_oos.npy | `8b04b2cc4296cde8b5d6ec7d461e3d1bbd127aad31c36eaeae665b6e4f6bd308` |
| baseline | dates_oos.csv | `7487d91edb411878c43100740689e4534e73edbadaf79313adf578d3424ccba6` |
| all_features | X.npy | `9cdf78c863fe379c5875954cc4732de559df1324bd673b06ed66517203acd834` |
| all_features | y.npy | `62285d583262c6fb11f99bbc7108397948f64faab6e4793eccf8ef741a10f588` |
| all_features | b_oos.npy | `8b04b2cc4296cde8b5d6ec7d461e3d1bbd127aad31c36eaeae665b6e4f6bd308` |
| all_features | dates_oos.csv | `7487d91edb411878c43100740689e4534e73edbadaf79313adf578d3424ccba6` |

On CARC: `cd <DESIGN>/_work/design_<set> && sha256sum X.npy y.npy b_oos.npy dates_oos.csv`. The sbatch design mode
prints these.

**Against the committed linear run.** The linear run's own caches are not in the repo (`_work/` is gitignored). Its
`forecasts.parquet` was compared on all 14,497 rows of each of its 7 models, as follows:
- B is bit-identical to `b_oos.npy` here.
- true_adj differs from `y.npy` here by at most 1 ulp (relative 3.5e-16, on 2,274 rows).
- Written with `to_csv` and read back with `read_csv`, as the linear walk tables are, y here equals the committed
  true_adj bit for bit.

So the cache built here is the design of the committed run. The trees' true_adj is the cache's y itself, without the
CSV round trip. Scoring uses rv_actual and plain, not true_adj.

CHECKS_PLACEHOLDER

## 4. Timing (this container, 4 threads, k = 3 refits at the first rows of 2020-01-03)

`python experiments/pooled_trees_rerun_13.py time --k 3 --threads 4` -> `timing_vm/sec_refit.csv`,
`timing_vm/projection_vm_4t.csv`. CPU-hours = seconds a refit x refits x 4 threads / 3600. The refit counts are as
follows:
- 2020-01-03..2024-04-30: 14,262 rows and 1,110 days;
- 2012-01-03..2024-04-30: 40,466 rows and 3,151 days.

| arm | columns | s a refit | 2020–2024 bar | 2020–2024 day | 2012–2024 bar | 2012–2024 day |
|---|---|---|---|---|---|---|
| lgbm, HAR benchmark | 12 | 0.52 | 8.2 | 0.6 | 23.3 | 1.8 |
| xgb, HAR benchmark | 12 | 0.46 | 7.3 | 0.6 | 20.6 | 1.6 |
| rf, HAR benchmark | 12 | 5.74 | 91.0 | 7.1 | 258.3 | 20.1 |
| lgbm, all features | 576 | 5.92 | 93.8 | 7.3 | 266.0 | 20.7 |
| xgb, all features | 576 | 8.25 | 130.7 | 10.2 | 370.9 | 28.9 |
| rf, all features | 576 | 306.50 | 4,857 | 378 | 13,781 | 1,073 |
| all six | | | 5,188 | 404 | 14,720 | 1,146 |

(CPU-hours. The wall hours of a single task are these divided by 4.)

- The random forest on all features takes 94 % of the total. With `max_features` 0.90 of 576 columns, `max_depth` 20,
  485 trees and `max_samples` 0.41 (10,700 rows a tree), one refit takes 5 minutes on 4 threads here.
- These numbers are from this container. The task file for CARC should be built from the CARC time probe (sbatch
  `MODE=time`, same `--cpus-per-task` as the array). The time mode writes `timing/sec_refit.csv`, which `tasks` reads.
  This container's table is kept apart in `timing_vm/`.
- With this container's table, `tasks --refits day --target-hours 3` gives 103 tasks: 37 for 2020–2024 (32 of them rf
  on all features) and 66 for 2012–2019 (58 rf on all features).

## Reproduce

```
python experiments/pooled_linear_rerun_2026_10_19_13.py design --jobs 2 --out results/pooled_trees_rerun13_2026-10-20
python results/pooled_trees_rerun13_2026-10-20/test_3day/run_checks.py --bar-parts-skip rf/all_features
python experiments/pooled_trees_rerun_13.py time --k 3 --threads 4
```

The commands for CARC are in the header of `cluster/slurm/pooltree_rerun13.sbatch`.
