# Pooled trees, final design (2014 start), w2000, bar refits, 2014-01-02..2024-04-30 (CARC Discovery)

Runner commit c989ee0 (code at 3cbb85f when submitted), python /home1/jc_905/trees_final/venv_skx/bin/python (python 3.12.15, numpy 1.26.4, pandas 2.3.3, scipy 1.17.1, sklearn 1.9.1, lightgbm 4.6.0, xgboost 3.2.0, pyarrow 19.0.1, sklearnex 2026.2.0, daal 2026.2.0). WINDOW_DAYS=2000, REFIT=bar. First row 2014-01-02 10:00:00, 33,970 bars an arm on 2,643 days.

## Jobs
- 13324535 tfs_w2000 (1 tasks): {'CANCELLED': 1}, 0.0 allocated CPU-h, 0.0 used.
- 13325350 tfs_w2000 (1 tasks): {'CANCELLED': 1}, 0.0 allocated CPU-h, 0.0 used.
- 13326247 tfs_w2000 (15 tasks): {'COMPLETED': 14, 'CANCELLED': 1}, 357.3 allocated CPU-h, 339.5 used.
- 13329101 tfs_w2000 (39 tasks): {'COMPLETED': 39}, 1186.6 allocated CPU-h, 1165.6 used.
- 13348942 merge.

## Merge
forecasts.parquet: 53 pieces, arms ['rf_skx/all_features', 'rf_skx/baseline'].

## Pieces by arm (pieces / tasks, rows, CPU-hours = wall_sec x threads)
- rf_skx all_features: 51 / 51 pieces, 33970 rows, 1511.2 CPU-h, threads [8, 20], longest piece 5.12 h
- rf_skx baseline: 2 / 2 pieces, 33970 rows, 31.3 CPU-h, threads [20], longest piece 0.79 h
- total: 1542.5 CPU-h

## CPU model by piece
- AMD EPYC 7513 32-Core Processor: 45
- AMD EPYC 7542 32-Core Processor: 8

## Files
forecasts.parquet + forecasts.json, pieces/*.json (piece parquets stay on CARC: /home1/jc_905/trees_final/runs_skx/w2000/pieces), tasks*.txt, timing/, jobs.csv (sacct), progress.csv, run.json, design/ (md5, design_check.json), scores/, code/.

## Notes
- The forest of the final rerun at W = 2,000 days (window robustness): rf_skx, the same rule (RULE_rf_skx.txt), gate
  (gate/), environment and /home1 layout as results/pooled_rf_skx_2014start_w3000 (see its NOTES.md).  design/: the
  MD5SUMS of the /home1 design caches (= the 2014-start window sweep's, code/stage_home.sbatch there), the design check
  of w1000 / w2000 (design_check.json, stage job 13322587) and its log.
- Arrays: 13324535 (20 threads, 02:04) and 13325350 (8 threads, 03:18), each held and cancelled before any task
  started (the slots went to W3000 / W3500 first); 13326247 (20 threads, 4 GB, code_3cbb85f, 04:27): lines 1-14 ran,
  its pending lines 15-53 cancelled at 08:51 (20-core holes scarce); 13329101 (8 threads, 4 GB, 6:00:00, code_c989ee0):
  lines 15-53.  The forecasts do not depend on the thread count (gate (d)); 45 pieces on EPYC 7513, 8 on EPYC 7542
  (one CPU family by gate (b)).
- Runner: lines 1-14 code_3cbb85f, lines 15-53 and the merge (job 13348942) code_c989ee0 (3cbb85f with the one-CPU
  checks read by CPU family; nothing else differs).  No piece was resumed or resubmitted.
- Scores (scores/): code/score_trees_final.py of results/pooled_trees_final_2014start_w3000 on this run with the linear
  w2000 run of results/window_sweep_2014start_2026-10-22, on the 654c476 worktree of the w3500 scores (see
  results/pooled_rf_skx_2014start_w3500/NOTES.md); one window, so no cross_window_dm.csv.
