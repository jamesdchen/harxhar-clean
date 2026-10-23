# fi_by_window_2014start_2026-10-22

Does the pooled linear models' exogenous signal look "dense and weak" across training windows? The test uses the
2014-start window learning-curve sweep on the final specification (`results/window_sweep_2014start_2026-10-22`,
W = 1,000 to 3,500 days). The coefficients are replayed from each W's own walk at monthly origins. No new walks were
run. The predictions, measures and decision rule are in `RULE_fiw.txt`, committed (a5d373b, 21:36 PDT) and pushed
before anything was computed. Owner's approval: James, "maybe the training window ablation will help sell the dense
but weak story".

## Verdict

**The window ablation does not support the dense-but-weak story, and the evidence points the other way.** Hold the
penalty fixed and lengthen the window: the lasso and elastic net keep fewer exogenous variables and fewer columns, and
all three models put their standardized weight on fewer variables. Under the final specification, counts rise with W
in only one way: the pretune picks a penalty 10 times smaller at some windows. The window length itself does not
cause the rise.

| prediction | model | primary (each W at its frozen penalty) | S1 (common penalty) |
|---|---|---|---|
| P1 more variables / columns selected | lasso | PARTLY (5 of 6 rise). The counts follow the penalty (1e-4 at W 2,000 / 3,000 / 3,500, 1e-3 at the others; 2 of 5 steps up) | NOT SUPPORTED at 1e-4 and at 1e-3 (5 or 6 of 6 fall) |
| P1 | elastic net | SUPPORTED (6 of 6), all of it from one step, W 1,000 to 1,500, where the penalty drops from 1e-2 to 1e-3. From 1,500 to 3,500 at the same penalty, every count falls | NOT SUPPORTED at 1e-3 and at 1e-2 |
| P2 weight spreads (N_eff, n50, n80) | ridge | NOT SUPPORTED (0 of 9 rise: N_eff flat, n50 and n80 fall) | NOT SUPPORTED at 1e3, at 1e4 and at 1e4 x W/1,000 (9 of 9 fall) |
| P2 | lasso | NOT SUPPORTED (9 of 9 fall) | NOT SUPPORTED (9 of 9 fall) |
| P2 | elastic net | SUPPORTED (9 of 9), again only from the W 1,000 to 1,500 penalty step | NOT SUPPORTED |
| P3 full-set advantage widens | all | not established (see the P3 section) | n/a |

The rule's labels are mechanical. Read the elastic net's "SUPPORTED" and the lasso's "PARTLY" together with S1: at a
fixed penalty, no measure of any model rises with W.

## Gate

Every replayed forecast row equals the W's `forecasts.parquet` `pred_adj` (model, all_features). That is
3 models x 6 W x 15,483 rows, compared from the tune boundary g through each origin t. The largest absolute
difference is **8.9e-16** (rule: <= 1e-12); every W x model passes. The checker ran on CARC
(`fiw_replay.py --merge`) and again locally against the committed parquet files (`fiw_reduce.py`; the
`run_info.json` gate field). The replay used the current runner code, `experiments/pool_linear_pretune.py`
(STANDARDIZE_PENALTY in scaled coordinates, the binomial availability rule, WALK_KKT_RESEED, all at their defaults).
The SHA256 of each code file is the same on this branch and in CARC's copy.

## P1: exogenous variables / columns with gamma != 0 (lasso; mean over the 126 refits)

| | W 1,000 | 1,500 | 2,000 | 2,500 | 3,000 | 3,500 | slope /1000 d [95%] |
|---|---|---|---|---|---|---|---|
| walk, alpha | 1e-3 | 1e-3 | 1e-4 | 1e-3 | 1e-4 | 1e-4 | |
| walk V (variables) | 38.8 | 37.4 | 44.0 | 35.2 | 42.5 | 41.9 | +1.26 [+0.80, +1.57] |
| walk V_half | 43 | 41 | 46 | 36 | 42 | 42 | -0.69 [-0.97, +0.34] flat |
| walk C (columns) | 93.4 | 89.0 | 202.5 | 77.7 | 184.1 | 177.5 | +33.2 [+30.0, +35.3] |
| S1 1e-3 V | 38.7 | 37.5 | 36.2 | 35.2 | 33.4 | 31.9 | -2.70 [-3.77, -1.86] |
| S1 1e-3 C | 93.4 | 89.0 | 82.7 | 77.5 | 73.9 | 70.0 | -9.6 [-13.0, -7.1] |
| S1 1e-4 V | 45.6 | 44.8 | 44.0 | 43.3 | 42.5 | 41.9 | -1.48 [-1.82, -1.26] |
| S1 1e-4 C | 213.3 | 209.8 | 202.6 | 193.9 | 184.2 | 177.5 | -15.1 [-19.2, -12.6] |
| live exogenous variables (ridge) | 45.6 | 44.9 | 44.1 | 43.3 | 42.6 | 42.0 | |

At 1e-4 the lasso keeps every live variable, so V only tracks the live universe there. Elastic net, walk:
V = 19.1 / 42.1 / 41.0 / 39.0 / 37.9 / 37.1 and C = 27.6 / 127.7 / 119.8 / 112.6 / 105.1 / 102.2 (alpha 1e-2,
then 1e-3). At S1 1e-3: V = 43.2 / 42.2 / 41.0 / 39.0 / 37.9 / 37.1.

## P2: spread of the standardized weight over the exogenous variables (mean over refits)

| ridge | W 1,000 | 1,500 | 2,000 | 2,500 | 3,000 | 3,500 | slope [95%] |
|---|---|---|---|---|---|---|---|
| walk, alpha | 1e4 | 1e4 | 1e3 | 1e3 | 1e3 | 1e3 | |
| walk N_eff (1/HHI) | 29.2 | 27.8 | 31.0 | 30.1 | 29.2 | 28.2 | -0.10 [-0.25, +0.12] flat |
| walk N_eff, pooled S^sd_A | 30.9 | 29.2 | 32.9 | 32.3 | 31.0 | 29.3 | flat |
| walk n50 | 12.5 | 11.9 | 12.2 | 11.7 | 11.1 | 10.5 | -0.75 [-0.86, -0.66] |
| walk n80 | 27.4 | 26.6 | 27.2 | 26.8 | 25.9 | 25.3 | -0.75 [-0.97, -0.57] |
| S1 1e3 N_eff | 33.9 | 33.0 | 31.0 | 30.2 | 29.2 | 28.2 | -2.33 [-2.56, -2.11] |
| S1 1e4 N_eff | 29.2 | 27.8 | 26.4 | 25.3 | 23.9 | 23.5 | -2.37 [-2.68, -2.03] |
| S1 1e4 x W/1,000 N_eff | 29.2 | 27.4 | 26.0 | 24.4 | 23.0 | 22.4 | -2.79 [-3.08, -2.43] |

Lasso, walk: N_eff = 16.7 / 14.9 / 13.9 / 12.6 / 11.7 / 10.9. Every table and every measure, with its bootstrap
interval, is in `fiw_tables.txt` and `fiw_trend.csv`.

## Do the same variables carry the ridge weight? (`fiw_ridge_top10.csv`)

Only partly.

- **Top 5 by pooled share, walk:**
  - W 1,000: sumret .104, sumautocov, vix, stocktwits_sentiment, sumvolume.
  - W 3,500: sumabsret .067, fomc_until_inv .064, sumret .063, fomc_day .056, sumret2_ewstock.
- **Overlap with W 1,000's top 10:** 10 / 10 / 6 / 6 / 5 / 5 of 10.
- **Spearman of the 46 pooled shares, W 1,000 vs W 3,500:** 0.29.
- **Weight shifts as W grows:**
  - The FOMC calendar gains: fomc_until_inv goes from .019 to .064.
  - sumret falls from .104 to .063.
  - stocktwits_sentiment falls from .037 to .007.
- **At a fixed penalty the reshuffle is milder:**
  - Spearman from 0.42 to 0.54 and overlap from 6 to 8 of 10 (`fiw_ridge_share_stability.csv`).
  - W 1,500 vs 2,000 (where the walk's penalty drops 1e4 to 1e3): Spearman 0.79 for the walk against 0.94 to 0.96 at fixed penalties.

## P3 (already measured; `part1/linear_gap.csv`, `part2/gap.csv`; full minus HAR, %)

| model | fold-3 MSE gap, W 1,000 / 1,500 / 2,000 / 2,500 / 3,000 / 3,500 | fold-3 extra gain vs W 1,000, p at 2,500+ | OOS every-bar QLIKE gap |
|---|---|---|---|
| lasso | -6.1 / -7.5 / -7.4 / -6.5 / -6.4 / -6.1 | 0.52 / 0.75 / 0.98 | -4.9 / -5.9 / -6.2 / -5.3 / -6.2 / -7.6 |
| elastic net | -6.0 / -7.9 / -7.6 / -6.7 / -6.5 / -6.3 | 0.20 / 0.38 / 0.67 | -5.9 / -5.6 / -5.7 / -5.2 / -6.0 / -6.7 |
| ridge | -5.5 / -6.6 / -6.9 / -6.0 / -6.0 / -5.8 | 0.38 / 0.41 / 0.67 | -5.5 / -5.4 / -4.9 / -4.7 / -5.4 / -6.3 |

- **2013 validation year:** the gap widens up to W 1,500 to 2,000, then narrows back to its W 1,000 size.
- **OOS 2014-01 to 2024-04, every bar:** the gap is largest at W 3,500 for all three models but is not monotone
  (smallest at W 2,500 for ridge and elastic net). No test of the change in the gap across W exists.
- **16:00 QLIKE gaps:** mostly not significant and not monotone (`fiw_tables.txt`).
- **P3 is not established.**

## How to read this

- **Penalty confound (pre-stated).**
  - The frozen penalties change with W, which is why S1 exists.
  - Lasso and elastic-net penalties are per observation, so a 10x smaller alpha mechanically selects more.
  - Every primary rise above coincides with such a step.
- **What a falling count means (pre-stated reading, partly).** The rule stated that at a fixed per-observation alpha,
  sparse-and-strong predicts a flat or falling selection count, and dense-and-weak a rising one. The counts fall, which
  is the sparse side of that reading. A falling count is also what a time-varying signal would produce: a longer
  window averages over more regimes and attenuates each one. That alternative is post hoc and this study does not
  test it. The result is "not dense-and-weak in the P1 / P2 sense". It does not show "sparse and strong".
- **The live universe shrinks with W (found after the rule; post hoc check).**
  - LATE_START_WARMUP keeps a late-start series' value averages at 0 until the scaler's window holds W live rows.
  - So stocktwits_sentcount and three voldemand_* series are masked at more origins as W grows, and at every origin
    at W 3,500: 46 down to 42 live variables.
  - `code/fiw_posthoc.py` redoes everything on the 42 variables that are always live (`fiw_posthoc_42.csv`). The
    verdicts are the same:
    - lasso S1 1e-3: V 35.4 to 31.9; columns fall at 1e-4.
    - every model at every common penalty: N_eff, n50 and n80 fall.
    - ridge walk: N_eff on the 42 rises slightly (26.7 to 28.2), only through the 1e4 to 1e3 penalty step. At a
      fixed penalty it falls.
- **Uncertainty.** The intervals come from a paired moving-block bootstrap over refits (block 12, B = 2,000). They
  measure the variation over the evaluation span's origins. Windows overlap and are nested across W, so these
  intervals understate total uncertainty. The S1 declines are monotone in W for almost every measure (0 of 5 steps up).
- **What S1 is.**
  - S1 is the runner's cold seed at t: a fresh PFTuned on [t - W, t), with the identifiability mask and the
    standardization recomputed at t. It is not the walk.
  - This is the walk's own chunk-start path. Each replay's first row is exactly that seed at g, and all of those rows
    pass the gate.
  - At the frozen penalty, S1's per-W summaries are nearly identical to the walk's (lasso V 38.7 vs 38.8 at W 1,000;
    41.9 vs 41.9 at W 3,500). Its selected-variable count differs from the walk's by at most 5 at any origin.
  - Its forecasts differ from the walk's by at most 2.5 to 14% relative (lasso / elastic net, per W) and at 2 to 5 origins for ridge (mask timing). The walk holds the lasso / elastic-net scaling
    from the last seed, so this is a different problem by construction, not a gate failure.
- **Descriptive (not a prediction).** The exogenous share of sum|gamma| (backbone included in the denominator) falls
  with W for the lasso at 1e-3 (.687 to .537). For ridge it rises slightly at a fixed alpha (.781 to .802 at 1e3) and falls at fixed relative shrinkage (.563 to .497).

## Files

| file | contents |
|---|---|
| `RULE_fiw.txt` | predictions, measures, decision rule, secondary S1 (written before computing) |
| `fiw_tables.txt` | every table: gate, P1 / P2 by model x source with intervals and verdicts, ridge top 10, Spearman, share stability, P3 |
| `fiw_trend.csv`, `fiw_verdicts.csv` | per-measure W values, slope, 3,500 - 1,000 difference, intervals, steps, verdict; per-prediction verdicts |
| `fiw_summary.csv` | per W x model x source statistics; `fiw_refits.csv`: per-refit V, C, N_eff, n50, n80, exogenous share |
| `fiw_variables.csv` | per W x model x source x variable: selection frequency, pooled share, median per-refit share |
| `fiw_ridge_top10.csv`, `fiw_ridge_share_spearman.csv`, `fiw_ridge_share_stability.csv` | ridge heaviest variables and their stability |
| `fiw_p3.csv` | P3, from the sweep's part1 / part2 |
| `fiw_posthoc_42.csv` | POST HOC: the measures on the 42 always-live variables |
| `fiw_gamma.npz` | gamma (coefficient on the standardized column) of the walk and of every S1 fit, per W x model x origin, plus the walk's masks |
| `fiw_origins.csv` | the 126 origins (2014-01-02 to 2024-04-05, every 21st session), the OOS row and tune boundary per W |
| `run_info.json` | penalties, origins, the gate in full, CARC hosts and environment, bootstrap, code SHA256s |
| `code/` | `fiw_replay.py` (replay + S1 on CARC; `--shard k/n`, `--merge`), `fiw_shard.sh` (180-task array), `fiw_merge.sh`, `fiw_job.sh` (the probe), `fiw_reduce.py`, `fiw_posthoc.py` |

**Reproduce:**

1. On CARC, in `/scratch1/jc_905/fi_window` (copies of the sweep's design caches, its frozen penalties and
   forecasts, and the code tree): `sbatch --array=0-179 scripts/fiw_shard.sh`, then `sbatch scripts/fiw_merge.sh`.
2. Locally: `python code/fiw_reduce.py --in <out dir>`, then `python code/fiw_posthoc.py`.

The shards ran an earlier `fiw_replay.py` (SHA256 3c12ca28...). The committed version differs only in the merge
block, which tolerates the duplicated W 1,000 ridge job (1e4 x W/1,000 = 1e4), and in formatting.
