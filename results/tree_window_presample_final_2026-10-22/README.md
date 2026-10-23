# tree_window_presample_final_2026-10-22

The pooled trees' pre-2012 window check (`results/tree_window_presample_2026-10-22/`) rerun on the final design:
the same method, frozen configurations, folds and criteria, on design caches that carry every merged design flag
(LATE_START_WARMUP, UNOBSERVED_NEUTRAL, EXCLUDE_FOMC_RELEASE). STANDARDIZE_PENALTY, still pending, does not reach the
trees because they are scale-invariant. No walk was run and nothing was scored out of sample. This is a sensitivity
check, not a specification choice.

- **Design:** the window sweep's caches, built on CARC on 9d3fc05 (e34 e0f3a7a + `--first-fold`) with the linear
  runner's `design` stage. Sources: `/scratch1/jc_905/wsweep/code/results/<dir>/_work/design_{baseline,all_features}`,
  with `<dir>` = `gate_w1000_nofomcrel_carc` (W = 1,000) and
  `pooled_linear_rerun_2026-10-22_13_2012_w<W>_latestart_nofomcrel` (W = 1,500 … 3,000). They were read only and
  copied to `/scratch1/jc_905/tree_wcheck_final/designs/w<W>`; the md5 of each copy equals the md5 of its source, and
  each source still matched its md5 after the copy. Shapes are 84,747 × 564 (full) and 84,747 × 12 (HAR), the same
  column names and cache config as the earlier caches. UNOBSERVED_NEUTRAL is a module default and is therefore not
  in the config; the build logs show `[scale] unobserved-neutral: 134 columns, 2492143 values set to 0`. md5 sums
  are in `designs_md5.txt`.
- **What changed against the earlier caches (e34 45183a0)** (`design_diff/`): the HAR set is bit-identical at every
  W. In the full set only values set to 0 by UNOBSERVED_NEUTRAL differ: 107 to 114 of 564 columns, 0.22 to 0.32 % of
  entries. Before 2012 (every row this check fits or validates on), 17,412 to 18,847 values in 74 to 77 columns
  differ, 0.07 to 0.08 % of entries. y, B and the dates are equal.
- **Code:** a clone of 9d3fc05 at `/scratch1/jc_905/tree_wcheck_final/repo`.
  `experiments/pooled_trees_rerun_13.py`, the three tree configs and `data/core_stats.parquet` are unchanged from
  45183a0. The driver is the earlier `fold_fit.py` with two provenance fields added (code_commit, design_source). The
  environment is the same: `/scratch1/jc_905/envs/harxhar-r13` (lightgbm 4.6.0, xgboost 3.2.0, scikit-learn 1.9.1,
  numpy 2.3.3), 8 threads, EPYC 7513 nodes.
- **Gate** (`gate_check.csv`, job 13315152): the earlier check's W = 1,000 fold-3 fits were refitted on the earlier
  caches with this run's code and driver. LightGBM full set reproduces validation MSE .06743222153279194 bit for
  bit. All six arms (three trees × two sets) reproduce validation MSE, QLIKE and every validation prediction bit for
  bit, with the same fit and validation rows, config sha, X md5 and library versions.
- **Fits** (job 13315192): 96 fits, one per window × fold × tree × set, the same task list as before
  (`code/make_tasks.py`, byte-identical to the earlier `tasks.txt`). Folds: all 4 for W ≤ 2,000, folds 1–3 for
  W = 2,500, fold 3 for W = 3,000. Each fit uses the fold's rows before the 325-row embargo and is scored on its
  3,250 validation rows; the validation dates are asserted equal to the linear pre-tune's.
- **Criteria:** validation MSE on the adjusted scale (mean of fold means, the linear criterion) and QLIKE (pooled
  rows).
- **Compute:** jobs 13315151 (staging and diff), 13315152 (gate) and 13315192 (fits). 4.97 CPU-h allocated, 3.72
  used (`cpu.txt`, `sacct_*`). Full-set random forest takes 42 / 72 / 102 / 138 / 182 s per fit at W = 1,000 … 3,000.
- **Files:** `trees_final_per_fold.csv` (every fit), `trees_final_tables.csv` (the window tables, as before),
  `final_vs_earlier_tables.csv` / `_per_fold.csv` (against the earlier check), `aggregate.txt` / `compare.txt`
  (printouts), `gate_check.csv`, `designs_md5.txt`, `design_diff/` (per-column differences and staging logs),
  `sacct_*`, `code/` (Slurm scripts and drivers; the paths inside are CARC-absolute). CARC copy, with the per-fit json
  and parquet files: `/scratch1/jc_905/tree_wcheck_final/`.

## Validation MSE (adjusted scale), folds 1–3, W = 1,000 / 1,500 / 2,000 / 2,500

| arm | 1,000 | 1,500 | 2,000 | 2,500 | earlier design, % change |
|---|---|---|---|---|---|
| LightGBM, full | .058883 | .059070 | .058258 | **.057694** | −0.09 / +0.04 / −0.21 / −0.07 |
| LightGBM, HAR | .065368 | .065276 | .064655 | **.064138** | 0 (bit-equal) |
| random forest, full | .059671 | .059441 | .059302 | **.058957** | −0.13 / −0.07 / +0.04 / +0.05 |
| random forest, HAR | .064245 | .064126 | .063716 | **.063179** | 0 (bit-equal) |
| XGBoost, full | .057924 | .057520 | .057472 | **.057235** | −0.04 / +0.02 / +0.00 / +0.05 |
| XGBoost, HAR | .064856 | .064551 | .064190 | **.063469** | 0 (bit-equal) |

## Validation MSE, fold 3, W = 1,000 / 1,500 / 2,000 / 2,500 / 3,000

| arm | 1,000 | 1,500 | 2,000 | 2,500 | 3,000 | earlier design, % change |
|---|---|---|---|---|---|---|
| LightGBM, full | .067475 | .068249 | .067521 | .066897 | **.066365** | +0.06 / +0.09 / −0.31 / −0.05 / −0.13 |
| LightGBM, HAR | .076857 | .077704 | .077126 | .076009 | **.075798** | 0 (bit-equal) |
| random forest, full | .068623 | .069349 | .069047 | .068993 | **.067730** | +0.06 / −0.03 / +0.03 / +0.22 / −0.14 |
| random forest, HAR | .075140 | .075955 | .075244 | .074812 | **.074213** | 0 (bit-equal) |
| XGBoost, full | .066284 | .065958 | .065859 | .065688 | **.065462** | −0.14 / −0.04 / −0.08 / +0.11 / −0.00 |
| XGBoost, HAR | .076889 | .076530 | .076746 | **.075128** | .075320 | 0 (bit-equal) |

## Validation MSE, all 4 folds, W = 1,000 / 1,500 / 2,000

| arm | 1,000 | 1,500 | 2,000 | earlier design, % change |
|---|---|---|---|---|
| LightGBM, full | .062875 | .062879 | **.061887** | −0.08 / +0.07 / −0.21 |
| LightGBM, HAR | .068670 | .068396 | **.067754** | 0 (bit-equal) |
| random forest, full | .063738 | .063309 | **.063122** | −0.10 / −0.03 / −0.08 |
| random forest, HAR | .067508 | .067211 | **.066858** | 0 (bit-equal) |
| XGBoost, full | .061534 | **.060790** | .061293 | +0.00 / +0.02 / +0.08 |
| XGBoost, HAR | .067920 | .067463 | **.067311** | 0 (bit-equal) |

Bold marks each row's lowest value. QLIKE for the same three fold sets is in `trees_final_tables.csv` and
`aggregate.txt`.

## Final design against the earlier design

- **HAR arms:** the caches are bit-identical, and so is every one of the 48 fits (MSE and QLIKE).
- **Full-set arms:** the pooled validation MSE moves by at most 0.31 % in any cell (LightGBM, fold 3, W = 2,000) and
  by at most 0.21 % over folds 1–3. Single fits move by at most 0.38 %; 26 of 48 go down. Mean change per fit:
  LightGBM −0.07 %, random forest −0.06 %, XGBoost +0.03 %. Pooled QLIKE moves by at most 0.28 % in any cell.
- **Ranking of windows:** unchanged in every cell (6 arms × 2 criteria × 3 fold sets). With folds 1–3, W = 2,500 is
  lowest for every arm. With fold 3 alone, W = 3,000 is lowest except XGBoost HAR (2,500). With all 4 folds,
  W = 2,000 is lowest except XGBoost full (1,500). The window ranking from the earlier check therefore holds on the
  final design.
