# Forecast tables — provenance manifest

Nine `yhat_*.parquet` files sit in this directory. Eight are in use (one per
tag in `atm_straddle_lib.yhat_paths`); the ninth, `yhat_b2lasso.parquet`, is
the fixed lasso on the earlier panel and is kept on disk but unused. The
parquet files themselves are untracked (about 9 MB each, regenerable from the
chunk trees in the main repository); this manifest is tracked.

Written 2026-09-05. Every sha256, size and row count below was computed from
the file as it sits in this directory; the chunk-tree attribution was computed
by re-reading each arm's 100 chunk `.npz` files and comparing `yhat` bit for
bit, chunk by chunk.

## Common to all nine

| property | value |
| --- | --- |
| rows | 276,317 |
| columns | `t`, `yhat`, `baseline`, `rv_raw` (all float64 after the dump) |
| stamp span | 2001-07-13 16:00 UTC .. 2024-05-01 03:30 UTC (naive ET bar-end labels localized to ET, then converted to UTC by the dump script) |
| `t`, `rv_raw`, `baseline` | byte-identical across all nine files — any difference between two models is `yhat` alone |
| chunks | 100 per arm, chunk 0 starting at panel row 24,000 |
| estimation window | 24,000 bars ending strictly before the target row (about 480 sessions, not 500) |
| dump script | `experiments/dump_unif_yhat.py` in the main repository (`--arm`, `--root`, `--out`) |

## The eight tables in use

| tag | file | sha256 | bytes | arm | panel | chunk tree |
| --- | --- | --- | --- | --- | --- | --- |
| `a0` | `yhat_a0.parquet` | `e3e4d9caaddfe7421fa4f5da6acbedeaa458bb221d8ef26c1a5d7995ffc7be9d` | 9,543,510 | `a0_ols_har` | incumbent (FOMC-panel run pending) | `results/unification_carc/a0_ols_har`, 100/100 chunks bit-exact |
| `blk2` | `yhat_blk2_fomc1.parquet` | `f0edd6db636c8ae82e0d08f397f523b5fac46695259a51bb3955e4cb829bb6a1` | 9,543,510 | `blk2_user` | FOMC (panel of record) | `results/unification_fomc1/blk2_user`, 100/100 chunks bit-exact |
| `blk2_inc` | `yhat_blk2.parquet` | `70e363048bb81945ade1544eb17412c07f13cc47e889f48b84d970fe164589f9` | 9,543,510 | `blk2_user` | incumbent (no FOMC columns; diagnostic row) | `results/unification_carc/blk2_user`, 100/100 chunks bit-exact |
| `lgbm` | `yhat_tree00.parquet` | `e52bb99852c0fa7e2fdf05385e28be8f372da52a00a8487c64d274c137de37e4` | 9,543,510 | `tree_expert_00` | incumbent (FOMC-panel run pending) | `results/unification_carc/tree_expert_00`, 100/100 chunks bit-exact |
| `xgb` | `yhat_tree16.parquet` | `cc9fe56b858ccd64825754d8d51a82c37b13f061efdee5697b91d13201b4258b` | 8,958,047 | `tree_expert_16` | incumbent (FOMC-panel run pending) | `results/unification_carc/tree_expert_16`, 100/100 chunks bit-exact |
| `lasso_t` | `yhat_b2lasso_tuned.parquet` | `f3965d42324f004e1acae61417db3a880369ffb66f57c603e71d2689f9f6c9c5` | 9,543,512 | `b2_lasso_tuned` | incumbent (FOMC-panel run pending) | `results/unification_carc/b2_lasso_tuned`, 100/100 chunks bit-exact |
| `lasso_f` | `yhat_b2lasso_fomc1.parquet` | `79e82822c027355762762b334337795c1305257544da091024f2881961c90301` | 9,543,510 | `b2_lasso` | FOMC (panel of record) | `results/unification_fomc1/b2_lasso`, 100/100 chunks bit-exact |
| `enet` | `yhat_b3enet_tuned.parquet` | `26356bd813cbda4f0b876ebb4f000827886841f17f5d32e73b752be3eee25e0e` | 9,543,510 | `b3_enet_tuned` | incumbent (FOMC-panel run pending) | `results/unification_carc/b3_enet_tuned`, 100/100 chunks bit-exact |

## The unused ninth table

| file | sha256 | bytes | arm | panel | chunk tree |
| --- | --- | --- | --- | --- | --- |
| `yhat_b2lasso.parquet` | `69b2734bca6c2406daf49b0b1c1d48bb814c4896bc389d4016ff18244f7acc1b` | 9,543,512 | `b2_lasso` | incumbent | `results/unification_carc/b2_lasso`, 100/100 chunks bit-exact |

The fixed lasso is reported on the panel of record, so no tag points at this
file. It is kept because it is the comparator behind the measured panel effect
(`lasso_f` on the two panels).

## Panels

* **FOMC panel** (`results/unification_fomc1`, 1,264 design columns): the
  panel of record. Carries the FOMC calendar channels. Cannot be rebuilt
  locally — the panel build is about 32 GB and lives cluster-side.
* **incumbent panel** (`results/unification_carc`, 1,144 design columns): the
  earlier panel, without those columns. The panel swap alone moves the ridge's
  16:00 forecast by a median 1.07% (p95 4.05%, max 15%), more than 1% on 53%
  of trading days; `a0` is panel-invariant to 2e-11.

`yhat_a0.parquet` (incumbent) is bit-exact to the FOMC panel's `a0_ols_har`
chunks on 83 of 100 chunks, consistent with that panel-invariance to rounding.

## Run records and code fingerprints

**FOMC-panel tables** (`blk2`, `lasso_f`) — journaled run:

| field | value |
| --- | --- |
| run id | `main-a95517c1` |
| cluster | `carc-d1` |
| submitted | 2026-08-27T14:07:19Z, 600 tasks |
| hpc-agent | 0.11.4 |
| `cmd_sha` | `a95517c1b70e71bc51e68c720e9159fd2774a915acaa6c40d80e9759fb7ffd35` |
| `env_hash` | `f16c865d31c97196fbf382f7f83719c87a1415eb7f96ee6838d0a92073259f29` |
| `tasks_py_sha` | empty (not recorded) |
| remote path | `/scratch1/jc_905/harxhar-clean` |
| code fingerprints | `src/unification.py` sha256 `7128cd62...`, `src/data/loading.py` sha256 `532fe9bc...` |

Caveat on that code: the fingerprinted content is not present locally. The
local working tree was edited 2026-08-27 at 17:17Z — after the 14:54-15:11Z
harvest — adding 247 unidentified bytes to `src/unification.py` and 14 bytes (a
`noqa` comment) to `src/data/loading.py`; those edits were committed as
`5db8240` and postdate the run, so they do not bear on these tables. The
committed code reproduces the incumbent-panel `blk2_user` chunks to 2e-13 and
`a0` to 3e-14, so the estimator specification is confirmed; the FOMC-panel
build under the committed code is not re-derived. The cheap check is
`sha256sum /scratch1/jc_905/harxhar-clean/src/unification.py` on the cluster.

**Incumbent-panel tables** (`a0`, `blk2_inc`, `lgbm`, `xgb`, `lasso_t`,
`enet`, and the unused `yhat_b2lasso.parquet`): **no campaign record.** None of
the 103 runs in the main repository's `.hpc` journal corresponds to them. The
chunk trees exist and the tables are bit-exact to them, but the submitting run,
the code sha and the dump command were not recorded. The chunk `.meta.json`
files carry only `arm`, `chunk_index`, `chunk_start`, `chunk_end`,
`n_rows_panel` and `wall_sec` — no code sha, no library versions, no thread
count, no refit cadence.

**Dump commands.** Recorded for `lasso_f` only:

```
python experiments/dump_unif_yhat.py --arm b2_lasso \
    --root results/unification_fomc1 \
    --out results/spxw_pnl/yhat_b2lasso_fomc1.parquet
```

run from the main repository on 2026-09-05 against the 100 chunks in
`results/unification_fomc1/b2_lasso`. The other eight dumps were not recorded;
their file mtimes (UTC) are `yhat_a0` and `yhat_blk2` 2026-08-18T04:32,
`yhat_b2lasso` and `yhat_b2lasso_tuned` 2026-08-26T15:10, `yhat_tree16`
2026-08-26T15:11, `yhat_tree00` 2026-08-26T15:17, `yhat_blk2_fomc1`
2026-08-27T15:43, `yhat_b3enet_tuned` 2026-08-27T17:00.

`yhat_blk2.parquet` in this directory is a hardlink to the main repository's
copy, not a second dump.

## Refit contract

| tag | contract |
| --- | --- |
| `a0`, `blk2`, `blk2_inc` | refit every bar on the trailing 24,000-bar window |
| `lasso_f` | refit every bar; penalty fixed, never retuned |
| `lasso_t`, `enet` | refit every bar; reseeded and retuned every 250 solves |
| `lgbm`, `xgb` | **unrecorded.** CARC ran a per-bar refit at 8 threads; Hoffman2 ran `TREE_REFIT_EVERY=10` at 2 threads (`jobs/sge/unification_tree.sge:17`). Both clusters refilled tree arms during the harvest window and neither the chunk meta nor the parquet records which produced a given chunk, so the cadence behind these two tables is 1 or 10 and cannot be told apart from the artifacts. Both shipped tables are bit-exact chunk for chunk to `results/unification_carc/<arm>`, which is as far as attribution goes: the directory name is not a record of the cluster that computed its contents. Cadence greater than 1 is stale-but-causal — instrumented runs found 0 leaks at cadence 1 and 10. |

## Hyperparameter provenance

| tag | provenance |
| --- | --- |
| `a0` | OLS on the HAR ladder plus calendar dummies; no hyperparameters |
| `blk2`, `blk2_inc` | block-diagonal ridge, penalties 1 on the backbone block and 100 on the exogenous block, HAR ladder base 2; recorded in the chunk meta's arm spec |
| `lasso_f` | fixed penalty 1e-4, confirmed in code |
| `lasso_t`, `enet` | causally tuned: enumerated penalty grid, validation tail = the 125 bars ending at the 15:30 bar after a 25-bar embargo, selection criterion is fit-scale SSE (deliberately not the reported loss); retuned every 250 solves |
| `lgbm`, `xgb` | **unrecoverable.** Arms 00 and 16 of a frozen 20-arm bank defined by `experiments/tree_menu.json`, which is absent from disk and from every commit in the main repository's history; the Optuna journals and per-chunk trial records are gone as well. The two arms were hand-picked from the bank rather than taken from the scorer's expert selection, and the basis for the pick is unrecorded. Arm 16 was the 7th-ranked XGBoost development trial under the nominal freeze. Library versions were not pinned and are not recorded. Arm 16's predictions were float32 in the chunk npz, which is why `yhat_tree16.parquet` is smaller on disk. |

## Identity checks re-run 2026-09-05

* All nine files: 276,317 rows.
* `t` identical across all nine (as int64 nanoseconds).
* `rv_raw` and `baseline` byte-identical across all nine.
* `yhat_b2lasso_fomc1.parquet`: 100/100 `results/unification_fomc1/b2_lasso`
  chunks bit-exact; 0/100 against every other candidate chunk tree.

## Known gaps

1. Four of the eight tables in use (`lgbm`, `xgb`, `lasso_t`, `enet`) are on
   the earlier panel while the ridge and the fixed lasso are on the panel of
   record; the comparison across them mixes panel with estimator. A cluster
   campaign to regenerate them on the FOMC panel is pending.
2. The tree menu cannot be recovered, so the two tree arms cannot be
   reproduced. Re-freezing and committing a menu is part of that campaign.
3. Nothing in the chunk meta records the code sha, the cadence, the thread
   count or the library versions. The campaign should write all four into
   every chunk meta.
