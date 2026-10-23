# Pooled trees, final design (2014 start), w3500, bar refits, 2014-01-02..2024-04-30 (CARC Discovery)

Runner commit 3cbb85f (code at 3cbb85f when submitted), python /home1/jc_905/trees_final/venv_skx/bin/python (python 3.12.15, numpy 1.26.4, pandas 2.3.3, scipy 1.17.1, sklearn 1.9.1, lightgbm 4.6.0, xgboost 3.2.0, pyarrow 19.0.1, sklearnex 2026.2.0, daal 2026.2.0). WINDOW_DAYS=3500, REFIT=bar. First row 2014-01-02 10:00:00, 33,970 bars an arm on 2,643 days.

## Jobs
- 13324534 tfs_w3500 (1 tasks): {'CANCELLED': 1}, 0.0 allocated CPU-h, 0.0 used.
- 13325349 tfs_w3500 (21 tasks): {'COMPLETED': 20, 'CANCELLED': 1}, 544.2 allocated CPU-h, 533.8 used.
- 13326246 tfs_w3500 (98 tasks): {'COMPLETED': 98}, 2896.7 allocated CPU-h, 2756.7 used.
- 13335328 merge.

## Merge
forecasts.parquet: 118 pieces, arms ['rf_skx/all_features', 'rf_skx/baseline'].

## Pieces by arm (pieces / tasks, rows, CPU-hours = wall_sec x threads)
- rf_skx all_features: 115 / 115 pieces, 33970 rows, 3381.1 CPU-h, threads [8, 20], longest piece 4.74 h
- rf_skx baseline: 3 / 3 pieces, 33970 rows, 55.1 CPU-h, threads [8], longest piece 2.32 h
- total: 3436.2 CPU-h

## CPU model by piece
- AMD EPYC 7513 32-Core Processor: 88
- AMD EPYC 7542 32-Core Processor: 30

## Files
forecasts.parquet + forecasts.json, pieces/*.json (piece parquets stay on CARC: /home1/jc_905/trees_final/runs_skx/w3500/pieces), tasks*.txt, timing/, jobs.csv (sacct), progress.csv, run.json, design/ (md5, design_check.json), scores/, code/.

## Notes
- The forest of the final rerun at W = 3,500 days (robustness window): rf_skx, the same rule (RULE_rf_skx.txt), gate
  (gate/), environment, code and /home1 layout as results/pooled_rf_skx_2014start_w3000 (see its NOTES.md); design/
  holds the design's MD5SUMS and design_check.json (the same design as results/pooled_trees_final_2014start_w3500).
- Arrays: 13324534 (20 threads, submitted 02:03, held, cancelled at 03:18 before any task started); 13325349 (8
  threads, 4 GB: lines 1-20 ran, its pending lines 21-118 cancelled at 04:27); 13326246 (20 threads, 4 GB: lines
  21-118).  The thread count follows the cluster's free cores; the forecasts do not depend on it (gate (d)).  Each
  piece's meta records its threads, host and CPU model; 88 pieces on EPYC 7513, 30 on EPYC 7542 (one CPU family by
  gate (b)).
- Merge (job 13335328) with code_c989ee0 (the one-CPU checks read by CPU family).  No piece was resumed or
  resubmitted.
- Scores (scores/; cross_window_dm.csv, 3,500 vs 3,000 days, also in results/pooled_rf_skx_2014start_w3000/scores/):
  code/score_trees_final.py of results/pooled_trees_final_2014start_w3000 on the w3000 and w3500 forest runs with the
  linear w3000 / w3500 runs of results/window_sweep_2014start_2026-10-22, run on a separate worktree at 654c476 (from
  10:00 the main worktree's writeup modules differ from 654c476's); its untracked inputs (data/spxw_chain.parquet,
  results/close_master_table, results/spxw_pnl, the MinGRU forecasts the economics module loads, ...; all last
  modified by 2026-10-18) copied from the main worktree.  The run reproduces the w3000 scores of e9b01fa byte for byte.
