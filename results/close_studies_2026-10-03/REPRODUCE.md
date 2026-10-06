# Reproducing the close studies of 2026-10-03 / 04

**Short answer.** Yes. Every input is in git, every stage is a command below, and a sample rerun from a clean
checkout of the committed code matched the committed results bit for bit (`repro_check.md`). Bit-exact
reproduction needs the locked packages *and* a CPU of the same vector class (AVX-512); on another library
version or CPU class the tree forecasts move (see the notes).

**Inputs in git:** `data/*.parquet`; the trade days `results/atm_straddle_0dte_1530/daily_blk2.parquet`; the stored
forecasts the gates compare with, `results/spxw_pnl/yhat_*.parquet`; `results/close_master_table/master_table.csv`.
**Not in git (rebuilt below):** the design cache `results/close_design/_work/` (1.1 GB) and every `*/_work/` folder.
`design_hashes.json` holds the sha256 of every cached array, so a rebuilt cache can be verified without the old one.
The forecasts of every `_work/` file are also committed in `forecasts/` (written by `experiments/close_studies_export.py`).

## Setup

- Python 3.11 (3.11.15 used): `python3.11 -m pip install -r results/close_studies_2026-10-03/requirements-lock.txt`.
- System packages: `gcc` (13.3 used) for the C kernel `experiments/close_exogpen_kernel.c`; `liblapack3` + `libblas3`
  (reference LAPACK / BLAS 3.12; the kernel links `liblapack.so.3` / `libblas.so.3`, which must resolve to these, not to
  OpenBLAS); `libgomp1` (LightGBM).
- Run every command from the repository root. Each study script sets OMP / OPENBLAS / MKL threads to 1 itself and fits
  single-threaded; `environment.json` records the machine of record (CPU, BLAS kernels, gcc, OS, thread settings).

## Steps, in order

CPU = single-core time from each study's own CPU columns (`cpu_seconds.csv`, `arms.csv` / `manifest.csv` `cpu_min` or
`cpu_sec`, SUMMARY.md); reused arms counted once. Commands on one line separated by `&` can run as two processes.

| # | study | commands | CPU | seeds |
|---|---|---|---|---|
| 1 | design cache | `for s in bar1600 last30; do python experiments/capture_design_close.py $s live_feasible all_features baseline > results/close_design/capture_$s.log 2>&1; done`<br>`for n in 3 4 5 6 7 8 9 10 11 12 13; do python experiments/capture_design_close.py lastbars$n all_features > results/close_design/capture_lastbars$n.log 2>&1; done`<br>check: `python experiments/close_studies_repro_check.py check --only a0` | 23 min | none (deterministic) |
| 2 | exog_penalty | `python experiments/close_exogpen.py gate`; `... check`<br>`python experiments/close_exogpen.py run ridge reclasso` & `... run reclasticnet`<br>`python experiments/close_exogpen.py run_supplement ridge reclasso` & `... run_supplement reclasticnet`<br>`python experiments/close_exogpen.py crosscheck`; `... analyze` | 24 min | none (deterministic) |
| 3 | trees_datasize | `python experiments/close_trees_datasize.py run` (23 arms; two processes may run it at once, claim files)<br>`python experiments/close_trees_datasize.py analyze` | 193 min | random_state 42 (LightGBM, XGBoost, random forest) |
| 4 | trees_lineartree | `python experiments/close_trees_lineartree.py run ctrl_base_r10 lt_base_r10 lt_all_r10 ctrl_all_r10 lt_all_r10_lam1 lt_all_r10_lam10 ctrl_base_r1 lt_base_r1`<br>`python experiments/close_trees_lineartree.py analyze` | 90 min | random_state 42 |
| 5 | trees_pretune | `python experiments/close_trees_pretune.py gate`<br>`CTP_SEED=0 CTP_TRIALS=45 python experiments/close_trees_pretune.py pretune` & `CTP_SEED=1 CTP_TRIALS=26 ...pretune` (set `CTP_WORKERS=1` to run both at once)<br>`CTP_SEEDS=0,1 CTP_CHECKPOINTS=10 CTP_RETUNE_TRIALS=10 python experiments/close_trees_pretune.py walk`; the same env with `report` | 231 min | TPE 20261003 + CTP_SEED; retune TPE 20261003 + CTP_SEED + row; LightGBM 42 |
| 6 | trees_morebars | `python experiments/close_trees_morebars.py gate`; `... run` (13 arms of ORDER)<br>`python experiments/close_trees_morebars.py run xgb_bars13_r26000 xgb_bars7_r14000 xgb_bars5_r10000 xgb_bars4_r8000 xgb_bars3_r6000 lgbm_bars1_r2000_seed43 lgbm_bars2_r4000_seed43 lgbm_bars3_r6000_seed43 lgbm_bars4_r8000_seed43 lgbm_bars13_r26000_seed43`<br>`python experiments/close_trees_morebars.py analyze`; `python experiments/close_trees_morebars_average.py` | 414 min | 42; seed replicate 43 (5 arms) |
| 7 | trees_kfull | `python experiments/close_trees_kfull.py gate`<br>`python experiments/close_studies_repro_check.py kfull-run` & the same again (43 arms)<br>`python experiments/close_trees_kfull.py manifest` | 877 min | 42, 43, 44 |
| 8 | kfull_tests | `python experiments/close_kfull_tests.py run existing` | 0.3 min | stationary bootstrap 20261004 (B 10000); simulations 1, 2 |

**Total about 31 CPU-hours** (about 16 h wall with two processes). Dependencies: 2-8 need the cache of step 1; 4 reuses
`trees_datasize/_work/lgbm_bar1600_w2000.npz` for `ctrl_all_r10` (after re-fitting its first 2 refits; 23 min more
without it); 6 reuses the 1- and 2-bar arms of 3, and its average needs `ridge_bb0` of 2; 7 reads 3 and 6; 8 reads 2, 3,
6 and 7.

## Notes for each step

1. **Design cache.** Which designs each study reads: exog_penalty and trees_lineartree `bar1600` all_features +
   baseline; trees_datasize and trees_pretune `bar1600` + `last30`; trees_morebars `lastbars3/4/5/7/13`; trees_kfull
   `last30` + `lastbars3..13`; kfull_tests `lastbars13`. The `live_feasible` designs are used by none (kept to rebuild
   the cache as it was). **Keep the log redirection:** `close_trees_kfull.py run` waits until
   `results/close_design/capture_<segment>.log` contains `captured <segment> all_features`. The executor's scaling pool
   (`UNIF_SCALE_PROCS`, default min(cores, 8)) does not change the design. Bit-exact for the same numpy / pandas on an
   AVX-512 CPU; see "CPU class" below.
2. **exog_penalty.** No randomness; the C kernel is compiled on first use (`gcc -O3 -march=native`, no -ffast-math,
   reference LAPACK / BLAS). The committed one-penalty, backbone-unpenalized and supplement arms were written by the
   kernel before its last change (the KKT check with repair for penalty ratios r != 1; only `*_bbr` / `*_bbfix` were rerun
   with it); the final kernel reproduces `ridge_bb0` and `lasso_bb0` bit for bit (check b). The elastic net's warm
   homotopy is sensitive to any floating-point change (SUMMARY gate 2: the C port and the spec's Python class differ on
   147 of 1469 sessions, up to 0.65 %), so `enet_*` arms are the first to move on another machine.
3. **trees_datasize.** LightGBM / XGBoost refit every 10 sessions from forecast row 130; the linear arms are the spec's
   Python `RollingTunedLinear` (BLAS through numpy). The Hoffman2 arms (`cluster/submit_close_trees_datasize_h2.sh`) were
   not run and are not part of the committed results.
4. **trees_lineartree.** `ORDER` (the default of `run`) lists `ctrl_all_r1` / `lt_all_r1`, which were not run; the arms
   of record are the eight named above.
5. **trees_pretune.** The pre-tune was stopped by hand after 45 (s0) and 26 (s1) trials of a 200-trial target;
   `CTP_TRIALS` reproduces those counts. TPE draws each trial from the earlier trials' validation MSEs, so the same trial
   sequence needs bit-identical fold fits (check c: trial 0's four folds are). The walk's 550 fits are 1.42 CPU-hours of
   the 231 min.
6. **trees_morebars.** The 1- and 2-bar rungs are the trees_datasize arms (`gate` refits their first 3 refits with this
   script's rows). The two linear controls at 13 bars (ridge 8 min, lasso 41 min) are the slowest non-tree arms.
7. **trees_kfull.** Use `close_studies_repro_check.py kfull-run`, not `close_trees_kfull.py run`: the latter's queue also
   holds 12 no-column arms (seeds 43 / 44 at k = 6, 8 .. 12) that were never run, and kfull_tests would then average them
   into the no-column family. `kfull-run` fits exactly the arms the committed `manifest.csv` lists as fitted (same claim
   files, so run two). `check()` refuses any LightGBM other than 4.7.0.
8. **kfull_tests.** Deterministic given the forecasts (fixed bootstrap and simulation seeds); `run existing` uses the
   forecasts on disk, so it reproduces the committed tables only when steps 2-7 produced the committed set.

## What is bit-exact, and what depends on the library version or the CPU class

| part | bit-exact with the lock on an AVX-512 CPU | depends on |
|---|---|---|
| design cache | yes (check a) | numpy / pandas versions; the CPU's vector class (check a2; the repo's earlier finding: AVX-512 nodes agree bit for bit, AVX2 nodes give a different X, max abs about 1e-11, same target) |
| exog_penalty C kernel | yes (check b) | gcc version and -march (check b2), the system LAPACK / BLAS; elastic net most sensitive |
| Python linear arms (ridge / lasso, OLS) | yes (check b, d) | numpy's OpenBLAS kernel: here numpy's OpenBLAS 0.3.23 runs its Prescott kernels (it does not recognize this CPU) and scipy's OpenBLAS 0.3.30 its SkylakeX kernels; another CPU gets other kernels |
| LightGBM / XGBoost forecasts | yes (check c) | the library version (stored cluster run, LightGBM 4.6.0: same QLIKE 0.1007, Sharpe mid 1.84 vs 1.99 here); every bit of the design (a split threshold moves with X), hence the CPU class |
| pre-tune trial sequence | yes if every fold fit is (check c) | anything that changes a fold MSE changes TPE's later draws |
| kfull_tests | yes (check d) | nothing beyond its input forecasts |

On a different machine, expect the QLIKE of a tree arm to move by about the seed-to-seed spread (0.0001 .. 0.0012 for
each k, kfull_tests) and its Sharpe ratio by several tenths; the linear arms move far less.

## Checking a rerun against the committed numbers

- **Sample check, no full rerun (about 15 min):** `python experiments/close_studies_repro_check.py check --out <folder>`
  (without `--out` it rewrites the committed `repro_check.csv` / `repro_check.md`). It reruns a cheap sample of each kind
  of run from a clean `git archive HEAD` and compares bitwise; the `env` row says how this machine differs from
  `environment.json`.
- **Design cache:** `python experiments/close_studies_repro_check.py check --only a0` (every array against `design_hashes.json`).
- **After a full rerun:** the study scripts rewrite their CSVs in place. `python experiments/close_studies_repro_check.py diff`
  compares every study CSV with the committed one (columns that record time ignored) and prints the largest
  difference; `git diff --stat results/close_studies_2026-10-03/` lists what changed. The forecasts can also be compared
  series by series with the committed export in `forecasts/` (`experiments/close_studies_load.py`).
