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
  - Window: W = window_days x 13 rows, read from the cache's `design.json`. `--window-days` defaults to it and, when
    given, is asserted against it (2026-10-20, after the author's update; before, the default was 2000). The
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
      `--target-hours`.
      - The brief's order was the 2020-01-03..2024-04-30 lines first, then 2012-01-03..2019-12-31.
      - After the author's update, the default span is 2012-01-03..2024-04-30 and the arms are ordered cheapest
        first.
    - `compare` and `colsha` locate differences between two builds of a design matrix (section 1b).
- `cluster/slurm/pooltree_rerun13.sbatch`: `MODE` = walk (array, the default), design, time or merge. The header holds
  the submission commands.
- `experiments/score_13bar_span.py` scores every (model, predictor_set) pair a forecasts.parquet holds. The linear pairs
  come first in Table 1's order, then the others. The "all features vs HAR benchmark" tests run for every model with both
  sets. The "vs OLS" tests run when the run holds the OLS. `--compare` covers the pairs both runs hold. A test with
  fewer than 8 days now gets `hac_lag = -1` instead of stopping the script.
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

## 1b. The 1,000-day design caches and the CARC builds (author update, 2026-10-20)

The author's update was relayed by the main session: the trees run on CARC with the 1,000-day window over
2012-01-03..2024-04-30. Built here with `design --jobs 2 --window-days 1000 --out
results/pooled_trees_rerun13_2026-10-20/w1000` (22 s and 239 s). Both caches have 84,747 rows: W = 13,000 plus 71,747
OOS rows.

The sha256 values here and on CARC, as the main session reported them. CARC's machine is an AMD EPYC 7513 (AVX2, no
AVX-512) with numpy 1.26.4 from `cluster/requirements-rerun13.lock.txt` and Python 3.12.15.

| window | set | file | here | CARC | equal |
|---|---|---|---|---|---|
| 1,000 | baseline | X.npy | `7b6ebcf84f1418238ba8b5edfa079d100db62d922620f5927a9c24987962f0f8` | `7b6ebcf8…` | yes |
| 1,000 | both | y.npy | `62285d583262c6fb11f99bbc7108397948f64faab6e4793eccf8ef741a10f588` | `62285d58…` | yes |
| 1,000 | both | b_oos.npy | `554ba4929f62b1b56773db35b89125705a467f7965b371e3c5364b0989aa9c48` | `554ba492…` | yes |
| 1,000 | both | dates_oos.csv | `ae99b11349ca86537d95a6db5d080c07db9e7b11996c86b271c75f989a3d72e5` | `ae99b113…` | yes |
| 1,000 | all_features | X.npy | `3f235146a4cb370310702e54b693f193ef80d5f96db5643633feabbde09fe6e2` | `9485e67e…` | **no** |
| 2,000 | baseline | X.npy | `5a1b8492…` (section 1) | `5a1b8492…` | yes |
| 2,000 | both | b_oos.npy, dates_oos.csv | `8b04b2cc…`, `7487d91e…` | `8b04b2cc…`, `7487d91e…` | yes |
| 2,000 | all_features | X.npy | `9cdf78c8…` | `f79455fb…` | **no** |

(design.json differs on every build because it records the host.)

**The full-set X depends on the CPU.**
- A second 1,000-day build here gives `3f235146…` again, so each build is reproducible on its own machine.
- This container's CPU has AVX-512, and numpy 1.26 picks its SIMD loops by CPU. A build with numpy's AVX-512 loops
  turned off (`NPY_DISABLE_CPU_FEATURES`, `OPENBLAS_CORETYPE=Haswell` or `Zen`) gives a third hash, `efe2d6bd…`.
  Adding `NUMBA_CPU_NAME=znver3` gives the same `efe2d6bd…`.
- So CARC's file could not be reproduced exactly here.

This build against the AVX-512-off build:
- 86.6 % of the entries are exactly equal, and 126 of 576 columns differ.
- In 125 of those columns the largest difference is below 2e-12 (the last bits of the trailing averages).
- One column, `adj_voldemand_spx_open_only_ma_5`, differs in 17,329 rows by up to 4.4e-3 (its standard deviation is
  0.51).

The baseline set, y, B and the dates are identical everywhere.

**Tools for the comparison** (the X.npy files are not in git):
- `colsha` writes the sha256 of each column. `colsha/` holds this container's: `colsha_all_features_w1000d.csv`,
  `colsha_all_features_w1000d_numpy_avx512_off.csv` and `colsha_all_features_w2000d.csv`.
- `compare A.npy B.npy --names design.json` gives the max |diff|, the share of exactly equal entries and the differing
  columns.
- `x_colstats_vm_w1000.csv` holds the sum, sum of squares, min and max of each column of the 1,000-day full-set X here.
  CARC writes the same file as `x_colstats_carc.csv`.

The author has not yet decided between keeping CARC's build for the trees and fitting them on a cloud-built X.

## 4b. Timing on the 1,000-day design (this container, 4 threads, k = 3)

`timing_vm_w1000/`. The table gives CPU-hours on 2012-01-03..2024-04-30 (40,466 rows, 3,151 days).

| arm | s a refit | bar | day |
|---|---|---|---|
| lgbm, HAR benchmark | 0.35 | 15.6 | 1.2 |
| xgb, HAR benchmark | 0.34 | 15.3 | 1.2 |
| rf, HAR benchmark | 2.73 | 122.6 | 9.5 |
| lgbm, all features | 3.75 | 168.7 | 13.1 |
| xgb, all features | 4.60 | 207.0 | 16.1 |
| rf, all features | 111.28 | 5,003 | 390 |
| all six | | 5,533 | 431 |

**Task file.** With this table, `tasks --refits day` (the defaults after the update: span 2012-01-03..2024-04-30, arms
cheapest first) gives 40 tasks, 33 of them rf on all features.

**CPU contention.** With two 4-thread LightGBM processes on these 4 cores, a 3-refit baseline walk took 220 s instead
of 1.5 s. Each array task needs its own `--cpus-per-task`.

## 2. Walks of 3 trading days (2020-01-03 .. 2020-01-07, 2,000-day cache, 4 threads)

`test_3day/run_checks.py --bar-parts-skip rf/all_features` -> `test_3day/pieces/`, `test_3day/checks.json`. Each arm
was walked with `--refit day` and `--refit bar` as `--part 1/1`, and again as `--part 1/2` + `2/2`. The 3 days have 39
rows; part 1/2 is 2020-01-03 and 2020-01-06, part 2/2 is 2020-01-07.

| arm | day first rows = bar rows | parts 1/2 + 2/2 = 1/1, day | parts, bar | max \|day − bar\| on the other 36 rows |
|---|---|---|---|---|
| lgbm, HAR benchmark | yes | yes | yes | 0.045 |
| xgb, HAR benchmark | yes | yes | yes | 0.032 |
| rf, HAR benchmark | yes | yes | yes | 0.064 |
| lgbm, all features | yes | yes | yes | 0.062 |
| xgb, all features | yes | yes | yes | 0.049 |
| rf, all features | yes | yes | not run | 0.056 |

**How "equal" is checked.** Equal means every column of the piece is bit for bit the same (`DataFrame.equals`): model,
predictor_set, date, pred_adj, true_adj, B, plain, rv_actual and fit_row. The last column is the max \|day − bar\| of
pred_adj on the rows that are not the first of their day; it shows that the two cadences differ there, as they should.

**What the structure checks show.**
- Under `--refit day`, every row of a day carries the fit_row of the day's first row.
- Under `--refit bar`, fit_row runs r − W, one apart.

**Coverage gaps.**
- The bar walk of rf on all features was run as `--part 1/1` only. It takes 39 refits at about 300 s here (3.3 h); its
  part split is checked under `--refit day`.
- The `wall_sec` in the test pieces' meta is not a timing. Several pieces ran while design builds or the time probe
  shared the 4 cores, and the full-set LightGBM bar walks ran up to 15 times slower than alone. Section 4 holds the timings.

**Other checks.**
- **Threads.** xgb on the HAR benchmark, `--refit day`, gives the same piece with 2 threads (through the sbatch file,
  with a mock Slurm environment) as with 4 threads.
- **Resume.** The lgbm HAR-benchmark bar piece was cut back to its first 10 rows, as after a time limit, and walked
  again. The walk ran the 29 missing refits only, and the piece equals the uninterrupted one bit for bit.
- **Against the linear run's rows.** On the 39 rows, B and rv_actual equal those of the committed linear
  forecasts.parquet. true_adj differs by at most 1 ulp (relative 2.2e-16), the CSV round trip of section 1.

## 3. Merge and schema

`merge --refit day|bar --n 1` -> `test_3day/forecasts_day.parquet`, `forecasts_bar.parquet`. Each holds 234 rows (6
arms × 39). The checks are as follows:
- every span row of every arm is present exactly once, and all values are finite;
- plain = pred_adj² × B;
- the column names and Arrow types are model, predictor_set (string), date (timestamp[ns]), and pred_adj, true_adj, B,
  plain, rv_actual (double). They equal those columns of `results/pooled_linear_rerun_2026-10-19_13/forecasts.parquet`,
  in the same order.

`--n 1` picks the 1/1 split, because the 1/2 + 2/2 pieces of the same span are also complete. Without `--n`, merge
refuses when two splits are complete.

**The extended scorer.**
- Command: `experiments/score_13bar_span.py --run linear=results/pooled_linear_rerun_2026-10-19_13 --run
  trees=<forecasts_bar> --span 2020-01-03:2020-01-07`, output in `test_3day/score_3day_bar/`.
- It scores the 7 linear and 6 tree pairs on one common set of 39 bars.
- With 3 days the Diebold–Mariano tests are empty: `dm_test` needs 8 days. The scorer now writes `hac_lag = -1` for
  them, where it used to stop with a KeyError on any span shorter than 8 days. On the linear forecasts its output is
  still byte-identical to the earlier version (metrics.csv, dm.csv and stdout, on two calls).
- A 3-day span tests only the plumbing. Its numbers say nothing about the models.

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
python experiments/pooled_trees_rerun_13.py time --k 3 --threads 4 --window-days 2000
python experiments/pooled_linear_rerun_2026_10_19_13.py design --jobs 2 --window-days 1000 --out results/pooled_trees_rerun13_2026-10-20/w1000
python experiments/pooled_trees_rerun_13.py time --design results/pooled_trees_rerun13_2026-10-20/w1000 --out results/pooled_trees_rerun13_2026-10-20/w1000 --window-days 1000 --k 3 --threads 4 --span 2012-01-03:2024-04-30 --span 2020-01-03:2024-04-30
```

(`time` writes `<OUT>/timing/`; this container's tables were moved to `timing_vm/` and `timing_vm_w1000/`, so that
the CARC probe writes a clean `timing/sec_refit.csv`.)

The commands for CARC are in the header of `cluster/slurm/pooltree_rerun13.sbatch`.
