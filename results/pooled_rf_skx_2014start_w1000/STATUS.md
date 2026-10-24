# Pooled trees, final design (2014 start), w1000, bar refits, 2014-01-02..2024-04-30 (CARC Discovery)

Runner commit c989ee0 (code at c989ee0 when submitted), python /home1/jc_905/trees_final/venv_skx/bin/python (python 3.12.15, numpy 1.26.4, pandas 2.3.3, scipy 1.17.1, sklearn 1.9.1, lightgbm 4.6.0, xgboost 3.2.0, pyarrow 19.0.1, sklearnex 2026.2.0, daal 2026.2.0). WINDOW_DAYS=1000, REFIT=bar. First row 2014-01-02 10:00:00, 33,970 bars an arm on 2,643 days.

## Jobs
- 13324536 tfs_w1000 (1 tasks): {'CANCELLED': 1}, 0.0 allocated CPU-h, 0.0 used.
- 13325351 tfs_w1000 (1 tasks): {'CANCELLED': 1}, 0.0 allocated CPU-h, 0.0 used.
- 13326248 tfs_w1000 (1 tasks): {'CANCELLED': 1}, 0.0 allocated CPU-h, 0.0 used.
- 13329102 tfs_w1000 (26 tasks): {'COMPLETED': 26}, 668.9 allocated CPU-h, 656.5 used.
- 13348559 merge.

## Merge
forecasts.parquet: 26 pieces, arms ['rf_skx/all_features', 'rf_skx/baseline'].

## Pieces by arm (pieces / tasks, rows, CPU-hours = wall_sec x threads)
- rf_skx all_features: 25 / 25 pieces, 33970 rows, 655.0 CPU-h, threads [8], longest piece 3.99 h
- rf_skx baseline: 1 / 1 pieces, 33970 rows, 13.4 CPU-h, threads [8], longest piece 1.67 h
- total: 668.4 CPU-h

## CPU model by piece
- AMD EPYC 7513 32-Core Processor: 22
- AMD EPYC 7542 32-Core Processor: 4

## Files
forecasts.parquet + forecasts.json, pieces/*.json (piece parquets stay on CARC: /home1/jc_905/trees_final/runs_skx/w1000/pieces), tasks*.txt, timing/, jobs.csv (sacct), progress.csv, run.json, design/ (md5, design_check.json), scores/, code/.

## Notes
- The forest of the final rerun at W = 1,000 days (window robustness): rf_skx, the same rule (RULE_rf_skx.txt), gate
  (gate/), environment and /home1 layout as results/pooled_rf_skx_2014start_w3000 (see its NOTES.md).  design/: the
  MD5SUMS of the /home1 design caches (= the 2014-start window sweep's, code/stage_home.sbatch there), the design check
  of w1000 / w2000 (design_check.json, stage job 13322587) and its log.
- Arrays: 13324536 (20 threads, 02:04), 13325351 (8 threads, 03:18) and 13326248 (20 threads, 04:27), each held and
  cancelled before any task started (the slots went to W3000 / W3500 first); 13329102 (8 threads, 4 GB, 6:00:00,
  submitted 08:51 with code_c989ee0, released at once by the round-robin throttle): all 26 lines.  The forecasts do not
  depend on the thread count (gate (d)); 22 pieces on EPYC 7513, 4 on EPYC 7542 (one CPU family by gate (b)).
- Runner: code_c989ee0 (3cbb85f with the one-CPU checks read by CPU family; nothing else differs), also for the merge
  (job 13348559).  No piece was resumed or resubmitted.
- Scores (scores/): code/score_trees_final.py of results/pooled_trees_final_2014start_w3000 on this run with the linear
  w1000 run of results/window_sweep_2014start_2026-10-22, on the 654c476 worktree of the w3500 scores (see
  results/pooled_rf_skx_2014start_w3500/NOTES.md); one window, so no cross_window_dm.csv.
