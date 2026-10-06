# Reproducing the close studies of 2026-10-03 / 04

**Short answer: yes.** On an AVX-512 machine with the locked packages and two settings
(`export OPENBLAS_CORETYPE=SkylakeX`, and the exog_penalty C kernel built with
`python experiments/close_studies_repro_check.py build-kernel --march haswell`), every sampled rerun from a clean
checkout of the committed code matched the committed `_work` files bit for bit (`repro_check.md`, 21 pass).
Without the two settings, the design cache, every tree run and the tests still match bit for bit. The C-kernel
arms and the Python linear models then differ in the last bits: up to 5e-12, the 5 failing rows of
`repro_check.md`. Their scored QLIKE / Sharpe agree to 1e-16. On an AVX2-only machine the design cache differs
(up to 2.8e-11), so the tree forecasts can move (see the table at the end).

**Why the settings:** the runs of 10-03 / 04 ran on a different host from today's, although both report
"Intel(R) Xeon(R) Processor @ 2.10GHz" under KVM. Today's CPU (family 6, model 207) is unknown to numpy's OpenBLAS
0.3.23, which then falls back to its Prescott kernels, and gcc's `-march=native` gives sapphirerapids tuning. The
stored results match OpenBLAS's SkylakeX / Cooperlake kernels and gcc builds for haswell / skylake-avx512 /
icelake-server. gcc's generic and sapphirerapids tuning do not match them, and plain x86-64 (no FMA) changes
the lasso path at 2.5e-4.

**Inputs in git:** `data/*.parquet`; the trade days `results/atm_straddle_0dte_1530/daily_blk2.parquet`; the stored
forecasts the gates compare with, `results/spxw_pnl/yhat_*.parquet`; `results/close_master_table/master_table.csv`.
**Not in git (rebuilt below):** the design cache `results/close_design/_work/` (1.1 GB) and every `*/_work/` folder.
`design_hashes.json` holds the sha256 of every cached array, so a rebuilt cache can be verified without the old one;
`forecasts/` holds an export of every `_work` forecast (`experiments/close_studies_export.py`).

## Setup

- Python 3.11 (3.11.15 used): `python3.11 -m pip install -r results/close_studies_2026-10-03/requirements-lock.txt`.
- System packages: `gcc` (13.3 used); `liblapack3` + `libblas3` (reference LAPACK / BLAS 3.12: the C kernel links
  `liblapack.so.3` / `libblas.so.3`, which must resolve to these, not to OpenBLAS); `libgomp1` (LightGBM).
- `export OPENBLAS_CORETYPE=SkylakeX` (AVX-512 CPUs only) for every step.
- Run from the repository root. Each study script sets OMP / OPENBLAS / MKL threads to 1 itself and fits
  single-threaded. `environment.json` records today's machine (CPU flags, BLAS kernels, gcc, OS, threads) and the
  settings above.

## Steps, in order

CPU = single-core time from each study's own CPU columns (`cpu_seconds.csv`; `cpu_min` / `cpu_sec` of `arms.csv` /
`manifest.csv`; SUMMARY.md), reused arms counted once. Commands joined by `&` can run as two processes.

| # | study | commands | CPU | seeds |
|---|---|---|---|---|
| 1 | design cache | `for s in bar1600 last30; do python experiments/capture_design_close.py $s live_feasible all_features baseline > results/close_design/capture_$s.log 2>&1; done`<br>`for n in 3 4 5 6 7 8 9 10 11 12 13; do python experiments/capture_design_close.py lastbars$n all_features > results/close_design/capture_lastbars$n.log 2>&1; done`<br>verify: `python experiments/close_studies_repro_check.py check --only a0` | 23 min | none (deterministic) |
| 2 | exog_penalty | `python experiments/close_studies_repro_check.py build-kernel --march haswell`<br>`python experiments/close_exogpen.py gate`; `... check`<br>`python experiments/close_exogpen.py run ridge reclasso` & `... run reclasticnet`<br>`python experiments/close_exogpen.py run_supplement ridge reclasso` & `... run_supplement reclasticnet`<br>`python experiments/close_exogpen.py crosscheck`; `... analyze` | 24 min | none (deterministic) |
| 3 | trees_datasize | `python experiments/close_trees_datasize.py run` (23 arms; two processes may run it at once, claim files)<br>`python experiments/close_trees_datasize.py analyze` | 193 min | random_state 42 |
| 4 | trees_lineartree | `python experiments/close_trees_lineartree.py run ctrl_base_r10 lt_base_r10 lt_all_r10 ctrl_all_r10 lt_all_r10_lam1 lt_all_r10_lam10 ctrl_base_r1 lt_base_r1`<br>`python experiments/close_trees_lineartree.py analyze` | 90 min | random_state 42 |
| 5 | trees_pretune | `python experiments/close_trees_pretune.py gate`<br>`CTP_SEED=0 CTP_TRIALS=45 python experiments/close_trees_pretune.py pretune` & `CTP_SEED=1 CTP_TRIALS=26 ... pretune` (with `CTP_WORKERS=1` for two at once)<br>`CTP_SEEDS=0,1 CTP_CHECKPOINTS=10 CTP_RETUNE_TRIALS=10 python experiments/close_trees_pretune.py walk`; the same env with `report` | 231 min | TPE 20261003 + CTP_SEED; retune TPE 20261003 + CTP_SEED + row; LightGBM 42 |
| 6 | trees_morebars | `python experiments/close_trees_morebars.py gate`; `... run` (13 arms of ORDER)<br>`python experiments/close_trees_morebars.py run xgb_bars13_r26000 xgb_bars7_r14000 xgb_bars5_r10000 xgb_bars4_r8000 xgb_bars3_r6000 lgbm_bars1_r2000_seed43 lgbm_bars2_r4000_seed43 lgbm_bars3_r6000_seed43 lgbm_bars4_r8000_seed43 lgbm_bars13_r26000_seed43`<br>`python experiments/close_trees_morebars.py analyze`; `python experiments/close_trees_morebars_average.py` | 414 min | 42; seed replicate 43 (5 arms) |
| 7 | trees_kfull | `python experiments/close_trees_kfull.py gate`<br>`python experiments/close_studies_repro_check.py kfull-run` & the same again (43 arms)<br>`python experiments/close_trees_kfull.py manifest` | 877 min | 42, 43, 44 |
| 8 | kfull_tests | `python experiments/close_kfull_tests.py run existing` | 0.3 min | stationary bootstrap 20261004 (B 10000); simulations 1, 2 |

**Total about 31 CPU-hours** (about 16 h wall with two processes). Steps 2-8 need the cache of step 1. Step 4 reuses
`trees_datasize/_work/lgbm_bar1600_w2000.npz` for `ctrl_all_r10` after re-fitting its first 2 refits (23 min more
without it). Step 6 reuses the 1- and 2-bar arms of step 3, and its average needs `ridge_bb0` of step 2. Step 7
reads steps 3 and 6; step 8 reads steps 2, 3, 6 and 7.

## Notes for each step

1. **Design cache.** exog_penalty and trees_lineartree read `bar1600` all_features + baseline; trees_datasize and
   trees_pretune `bar1600` + `last30`; trees_morebars `lastbars3/4/5/7/13`; trees_kfull `last30` + `lastbars3..13`;
   kfull_tests `lastbars13`. No study reads the `live_feasible` designs; they are kept so the cache is rebuilt as
   it was. **Keep the log redirection:** `close_trees_kfull.py run` waits until
   `results/close_design/capture_<segment>.log` contains `captured <segment> all_features`. The thread count and
   the OpenBLAS kernel do not change the design: rebuilds with one thread and with OpenBLAS's Prescott, Haswell or
   SkylakeX kernels all equal the cache.
2. **exog_penalty.** No randomness. Without `build-kernel`, `kernel()` compiles for `-march=native`. On today's
   machine that build changes the lasso's ill-conditioned small-penalty candidates in the fourth 250-session block:
   41 LU fallbacks against the stored 37. The validation MSE at alpha 1e-6 is 1.8e+51 stored and 1.9e+26 rerun
   (a diverging homotopy), and 0.1231 against 0.1236 at alpha 1e-5. The chosen penalties are the same, and the
   forecasts differ on 8 of 1469 sessions by at most 8e-16 relative. The prebuilt `-march=haswell` kernel runs on any AVX2 CPU and reproduces the stored arms; `kernel()`
   uses it as long as the `.so` is newer than the `.c`, so rerun `build-kernel` after any change to the `.c` files.
   The committed one-penalty, backbone-unpenalized and
   supplement arms were written before the kernel's last change (KKT check with repair for penalty ratios
   r != 1; only `*_bbr` / `*_bbfix` were rerun with it). The final kernel reproduces `ridge_bb0` / `lasso_bb0`,
   so that change does not touch them. Elastic-net arms are the most sensitive: SUMMARY gate 2 shows the C port
   and the spec's Python class differ on 147 of 1469 sessions (up to 0.65 %).
3. **trees_datasize.** Refit every 10 sessions from forecast row 130. The ridge / lasso arms are the spec's Python
   `RollingTunedLinear` (BLAS through numpy), so they need `OPENBLAS_CORETYPE` (check e). The Hoffman2 arms
   (`cluster/submit_close_trees_datasize_h2.sh`) were not run and are not in the committed results.
4. **trees_lineartree.** `ORDER` (the default of `run`) lists `ctrl_all_r1` / `lt_all_r1`, which were never run;
   the arms of record are the eight named above.
5. **trees_pretune.** The pre-tune was stopped by hand after 45 (s0) and 26 (s1) trials of a 200-trial target;
   `CTP_TRIALS` reproduces those counts. TPE draws each trial from the earlier trials' validation MSEs, so the same
   sequence needs bit-identical fold fits (check c: trial 0's four folds are bit-identical). The walk is 1.42 of
   the 3.85 CPU-hours.
6. **trees_morebars.** The 1- and 2-bar rungs are the trees_datasize arms; `gate` refits their first 3 refits with
   this script's rows. The 13-bar linear controls (ridge 8 min, lasso 41 min) are the slowest non-tree arms.
7. **trees_kfull.** Use `close_studies_repro_check.py kfull-run`, not `close_trees_kfull.py run`. The latter's queue
   also holds 12 no-column arms that were never run (seeds 43 / 44 at k = 6, 8 .. 12), and kfull_tests would
   average them into the no-column family. `kfull-run` fits exactly the 43 arms the committed `manifest.csv` lists
   as fitted, with the same claim files, so you can run two. `check()` refuses any LightGBM other than 4.7.0.
8. **kfull_tests.** Deterministic given the forecasts (fixed seeds). `run existing` uses the forecasts on disk, so
   it reproduces the committed tables only when steps 2-7 produced the committed set.

## What is bit-exact, and what depends on the library version or the CPU class

| part | bit-exact with the lock, on an AVX-512 CPU | depends on |
|---|---|---|
| design cache | yes, with any OpenBLAS kernel and thread count (check a) | numpy's SIMD code paths: on an AVX2 CPU 120 of 628 columns differ, max 2.8e-11, same target (check a2; the repo's earlier finding: AVX-512 nodes agree, AVX2 nodes differ at about 1e-11); numpy / pandas versions |
| LightGBM forecasts (XGBoost not sampled) | yes, with any OpenBLAS kernel (check c) | the library version (stored cluster run, LightGBM 4.6.0: same QLIKE 0.1007, Sharpe mid 1.84 against 1.99 here); every bit of the design (split thresholds), hence the CPU class |
| pre-tune trial sequence | yes (check c: the fold fits) | anything that changes a fold MSE changes TPE's later draws |
| exog_penalty C kernel | yes with the `haswell` build (check b-fix) | gcc version and tuning (`-march`); the system LAPACK / BLAS; elastic net most sensitive |
| Python linear models (the linear spec's ridge / lasso class, the OLS) | yes with `OPENBLAS_CORETYPE=SkylakeX` (checks b-fix, e-fix) | numpy's OpenBLAS kernel: with the Prescott kernels the trees_datasize ridge differs by 5e-12, its lasso by 1e-13, the OLS by 2.4e-15 (same penalties chosen); an AVX2 CPU cannot run the SkylakeX kernels |
| kfull_tests | yes (check d) | nothing beyond its input forecasts |

On a machine where the trees move, expect a tree arm's QLIKE to move by about the seed-to-seed spread (0.0001 ..
0.0012 for each k, kfull_tests) and its Sharpe ratio by several tenths. The linear arms move far less.

## Checking a rerun against the committed numbers

- **Sample check, no full rerun (about 10 min):** `python experiments/close_studies_repro_check.py check --out <folder>`.
  Without `--out` it rewrites the committed `repro_check.csv` / `repro_check.md`. It reruns a cheap sample of each
  kind of run from a clean `git archive HEAD` into the gitignored `_repro_scratch/`, both as the scripts run and with
  the two settings, and compares bitwise. Its `env` row says how the machine differs from `environment.json`.
- **Design cache:** `python experiments/close_studies_repro_check.py check --only a0` (every array against `design_hashes.json`).
- **After a full rerun:** the study scripts rewrite their CSVs in place. `python experiments/close_studies_repro_check.py diff`
  compares every study CSV with the committed one (time columns ignored) and every `_work` forecast with the
  committed export in `forecasts/`, and prints the largest difference of each. On the unchanged tree all 194 are
  identical. `git diff --stat results/close_studies_2026-10-03/` lists what changed.
