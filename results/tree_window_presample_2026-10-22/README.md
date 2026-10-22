# tree_window_presample_2026-10-22

Pre-2012 window check for the pooled trees (LightGBM, XGBoost, random forest; HAR benchmark and full set), the trees'
counterpart of the pooled linear models' pre-sample criterion. No walk was run; this is a sensitivity check, not a
specification choice.

- **Design:** the fixed 13-bar design (e34 at exactly 45183a0: LATE_START_WARMUP on, EXCLUDE_FOMC_RELEASE on; 564 / 12
  columns, 84,747 rows) built on CARC for W = 1,000 / 1,500 / 2,000 / 2,500 / 3,000 days with the linear runner's
  `design` stage (rth13, base 5, the four --ex-flags, --first-day 2012-01-03). Gate: the CARC 1,000-day build equals the
  local one (`gate_w1000_all_features.csv`: baseline identical; full set 92.89% of entries equal, max difference 3.1e-13).
- **Fits:** one per window × fold × tree × set (96), the frozen configurations through
  `experiments/pooled_trees_rerun_13.py` `model_factory` / `fit_predict` (unchanged), fit on the fold's rows before the
  325-row embargo and scored on its 3,250 validation rows (the linear pre-tune's folds; validation dates asserted equal).
  Folds that each window can fit: all 4 (W ≤ 2,000), folds 1–3 (W ≤ 2,500), fold 3 (all W).
- **Criteria:** validation MSE on the adjusted scale (mean of fold means, the linear criterion) and QLIKE (pooled rows).
- **Compute:** CARC jobs 13309314 (designs) and 13309739 (fits), 5.70 CPU-h allocated, 3.93 used; full-set random forest
  41 / 72 / 100 / 138 / 166 s per fit at W = 1,000 … 3,000.
- **Files:** `wsweep_trees_per_fold.csv` (every fit), `wsweep_trees_tables.csv` (the window tables), the gate, `sacct_*`,
  `code/` (Slurm scripts and drivers; paths inside are CARC/laptop absolute). CARC copy: `/home1/jc_905/wsweep_trees/`.

## Validation MSE (adjusted scale), folds 1–3, W = 1,000 / 1,500 / 2,000 / 2,500

| arm | 1,000 | 1,500 | 2,000 | 2,500 |
|---|---|---|---|---|
| LightGBM, full | .058934 | .059048 | .058377 | .057733 |
| LightGBM, HAR | .065368 | .065276 | .064655 | .064138 |
| random forest, full | .059746 | .059480 | .059279 | .058927 |
| random forest, HAR | .064245 | .064126 | .063716 | .063179 |
| XGBoost, full | .057948 | .057510 | .057471 | .057209 |
| XGBoost, HAR | .064856 | .064551 | .064190 | .063469 |

Fold 3 alone (adds W = 3,000) and all 4 folds (W ≤ 2,000), and QLIKE, are in `wsweep_trees_tables.csv`.
