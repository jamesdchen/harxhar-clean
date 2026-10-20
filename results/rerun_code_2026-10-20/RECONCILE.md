# Rerun code reconciliation (2026-10-20)

Branch `insurance-selling-2026-09-19-rerun-code`, from `insurance-selling-2026-09-19` at 3fa819e. No result file
and nothing under `writeup/` was changed.

## What was merged

`origin/insurance-selling-2026-09-19-pooled-linear-13` (head b93aa67), the 13-bar rerun's code, merged with a merge
commit (febd750, no rebase). Its results were already on the branch, and the merge leaves them byte-identical. New or
changed by the merge:

- `experiments/pooled_linear_rerun_2026_10_19_13.py`, the 13-bar orchestrator (new);
- `experiments/pool_linear_pretune.py`, 7 conflict blocks with the 48-interval rerun's `--rerun` extensions;
- `src/backtest/executor.py`, 2 conflict blocks: `DROP_OVERNIGHT_INDICATOR`, `LEAN_GLOBAL_LAGS` and the
  `har_builder` hook of `_build_har_and_calendar`;
- `src/backtest/segmentation.py`, the `rth13` sample segment (merged cleanly);
- `.gitignore`, 1 conflict block.

`experiments/pool_linear_pretune.py` now runs both reruns. Each rerun's own commands are unchanged.

## How each conflict was resolved

| File / block | 48-interval rerun (HEAD) | 13-bar rerun (merged branch) | Resolution |
|---|---|---|---|
| `.gitignore` | Overleaf token script, base-5 cache, `_48/_work/` | `_13/_work/` | Both kept. `writeup/overleaf_sync/overleaf_git.sh` stays ignored and is not committed. |
| `executor.py` flags | none | `DROP_OVERNIGHT_INDICATOR`, `LEAN_GLOBAL_LAGS` | Taken, both default `False`. |
| `executor.py` `_build_har_and_calendar` | `generate_har_features(...)` | `(generate_har_features if har_builder is None else har_builder)(...)` | Taken (`har_builder=None` is the old call). |
| `pool_linear_pretune.py` docstring | `--rerun`, `--part`, `merge` | `--sample`, `--har-base`, `--drop-fomc`, `--ex-flag`, `--sched const:A`, min-norm `--est ols` | Both kept, plus a 2026-10-20 paragraph on how they combine. |
| backbone check | exact `BACKBONE_RERUN` under `--rerun`, else `BACKBONE` | exact `BACKBONE` on the default configuration, else only `har_ma_*` and calendar columns | All three kept: `--rerun` with none of the 13-bar options checks for an exact `BACKBONE_RERUN` (48-interval rerun). No options checks for an exact `BACKBONE` (printed run). Any 13-bar option gets the loose check (13-bar rerun). The old line is parked. |
| `merge` vs `check_meta` | `merge` mode added | `check_meta` added | Both kept. `check_meta` also asserts the design cache's `rerun` flag. |
| design-cache checks in `cached_backtest`, `pretune` | `bucket` + `rerun` asserts | `check_meta` | `check_meta`, which does both. The old asserts are parked. |
| schedule in `walk` | a JSON file holding one number gives a constant schedule (`ConstSched`) | `parse_sched`: `causal`, `const:A`, JSON `{block: alpha}` (`_ConstSched`) | `parse_sched` reads all four forms into one `ConstSched`. `_ConstSched` and the old reading are parked. |
| `main` options | `--rerun`, `--part`, `--parts`, mode `merge` | `--sample`, `--har-base`, `--drop-fomc`, `--ex-flag`, `configure(a)` | All kept, plus `--ols`. `--rerun` refuses `--drop-fomc` and any `--har-base` but 5. `--ex-flag` applies after `--rerun`'s own flags. |

**Both added `--est ols`, as two different estimators.** Both stay selectable through the new `--ols`:

- `ridge0` is the spec's ridge path at penalty 0, with rank-one updates. The 48-interval rerun used it.
- `minnorm` is min-norm least squares on the centred window, using an eigh pseudo-inverse at rcond 1e-15. The 13-bar
  rerun used it.
- `auto` is the default. It gives `ridge0` under `--rerun` and `minnorm` otherwise, which is what each rerun ran.

Every new option defaults to the behaviour before it:

- `--sample 48`, `--har-base ""`, no `--drop-fomc`, no `--ex-flag`;
- `--sched causal`, no `--rerun`, no `--part`;
- the new executor flags are `False`.

`--ols` only acts on `--est ols`, which did not exist before either rerun.

Replaced text is parked in `# 2026-10-20 (user): parked -- ... Was:` comments.

One addition outside the conflicts: `experiments/pooled_linear_rerun_2026_10_19_13.py` takes `--out DIR`. The default
is the committed `results/pooled_linear_rerun_2026-10-19_13/`. The option let the reproduction below write to a
scratch directory.

## Reproduction from the merged code

Environment: this container (4 CPUs, 15 GB, Intel Xeon @ 2.10 GHz). Both reruns ran in Python 3.12.3 venvs with the
library versions their SUMMARY.md files record:

- 13-bar: numpy 1.26.4, pandas 2.3.3, scipy 1.17.1, scikit-learn 1.9.1, numba 0.68.0, pyarrow 19.0.1;
- 48-interval: the same but numba 0.61.2 and pyarrow 18.1.0.

The container's default Python 3.13 has none of these installed.

### 13 bars: the whole rerun, end to end, exact

Wall time 9.5 min:

| Stage | Time |
|---|---|
| design | 5.3 min |
| pre-tune | 0.7 min |
| walks | 3.3 min |

```
OUT=<scratch>/r13   # python = the 3.12 venv
python experiments/pooled_linear_rerun_2026_10_19_13.py design  --jobs 2 --out $OUT
python experiments/pooled_linear_rerun_2026_10_19_13.py pretune --jobs 4 --out $OUT
python experiments/pooled_linear_rerun_2026_10_19_13.py select  --out $OUT
python experiments/pooled_linear_rerun_2026_10_19_13.py walk --chunks all_features/lasso=2 --chunks all_features/enet=2 --jobs 4 --out $OUT
python experiments/pooled_linear_rerun_2026_10_19_13.py score   --out $OUT
```

Compared with the committed `results/pooled_linear_rerun_2026-10-19_13/`:

- `forecasts.parquet`: the frame equals the committed one, 101,479 rows. For each of the 7 models (OLS, plus ridge,
  lasso and elastic net on both sets), `plain`, `pred_adj` and `alpha` are exactly equal.
- `frozen_penalties.json`, `pretune_mse.csv`, `scores.csv`, `dm.csv` and `scores_all_bars.csv` are byte-identical.

### 48 intervals: HAR benchmark set, the whole evaluation span, exact

The design took 68 s. The four walks ran in parallel and took 24 s.

```
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
W=<scratch>/r48; SC=results/pooled_linear_rerun_2026-10-19_48/sched
python experiments/pool_linear_pretune.py design --rerun --bucket baseline --out $W/design_baseline
python experiments/pool_linear_pretune.py walk --rerun --bucket baseline --est <ols|ridge|reclasso|reclasticnet> --r 1.0 \
    --sched $SC/<est>_baseline.json --start 246000 --end -1 --halo 96000 --design $W/design_baseline --out $W/walk/baseline/<est>
```

Compared with the committed `results/pooled_linear_rerun_2026-10-19_48/forecasts/<model>_baseline.parquet`, with
`plain = pred_adj^2 x true_raw / true_adj^2` as in the runner's `score`:

| Model | Rows | Span | `pred_adj`, `plain`, `true_adj`, `pred_raw` |
|---|---|---|---|
| OLS (`--ols auto` gives ridge0) | 54,317 | 2019-12-30 18:00 .. 2024-04-30 23:30 | exact |
| ridge | 54,317 | same | exact |
| lasso | 54,317 | same | exact |
| elastic net | 54,317 | same | exact |

The committed `sched/*.json` were used as they are. The runner's `pick` and `score` stages were not rerun because
they write into the committed directory.

### 48 intervals: all features

Running at the time of this commit (design in 2 column parts + merge, then the ridge, lasso and elastic net walks); the result is added in a later commit.

## Checks

- `ruff format --check` passes on the four changed Python files.
- `ruff check` reports no new findings: the merged files have 29 findings, the same count as the 13-bar branch's
  versions of the same files, all of them pre-existing kinds (C408, RUF100 and others).
- The repo has no test suite for these files. Its only GitHub workflow is `close_signal.yml`, which is unrelated.
