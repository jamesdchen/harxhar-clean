# Pooled trees, final design (2014 start), w3000, bar refits, 2014-01-02..2024-04-30 (CARC Discovery)

Runner commit 3cbb85f (code at 3cbb85f when submitted), python /home1/jc_905/trees_final/venv_skx/bin/python (python 3.12.15, numpy 1.26.4, pandas 2.3.3, scipy 1.17.1, sklearn 1.9.1, lightgbm 4.6.0, xgboost 3.2.0, pyarrow 19.0.1, sklearnex 2026.2.0, daal 2026.2.0). WINDOW_DAYS=3000, REFIT=bar. First row 2014-01-02 10:00:00, 33,970 bars an arm on 2,643 days.

## Jobs
- 13324443 tfs_w3000 (31 tasks): {'COMPLETED': 30, 'CANCELLED': 1}, 929.6 allocated CPU-h, 889.2 used.
- 13325348 tfs_w3000 (60 tasks): {'COMPLETED': 60}, 1699.4 allocated CPU-h, 1669.4 used.
- 13334408 merge.

## Merge
forecasts.parquet: 90 pieces, arms ['rf_skx/all_features', 'rf_skx/baseline'].

## Pieces by arm (pieces / tasks, rows, CPU-hours = wall_sec x threads)
- rf_skx all_features: 87 / 87 pieces, 33970 rows, 2575.5 CPU-h, threads [8, 20], longest piece 6.16 h
- rf_skx baseline: 3 / 3 pieces, 33970 rows, 50.3 CPU-h, threads [20], longest piece 0.87 h
- total: 2625.8 CPU-h

## CPU model by piece
- AMD EPYC 7513 32-Core Processor: 65
- AMD EPYC 7542 32-Core Processor: 25

## Files
forecasts.parquet + forecasts.json, pieces/*.json (piece parquets stay on CARC: /home1/jc_905/trees_final/runs_skx/w3000/pieces), tasks*.txt, timing/, jobs.csv (sacct), progress.csv, run.json, design/ (md5, design_check.json), scores/, code/.

## Notes
- The forest of the final rerun at W = 3,000 days: rf_skx (the scikit-learn forest on oneDAL, scikit-learn-intelex
  2026.2.0), its rule and dated addenda in RULE_rf_skx.txt, its gate in gate/ (gate/fold/gate_skx.csv: fold-3 fits equal
  to the benchmark's engine B on EPYC 7513 and 7542; gate/P*/: walk pieces equal across runs and CPU models;
  gate/t20/gate_skx.csv: 20 threads = 8 threads; gate/logs/: the gate jobs' logs).
- Everything on /home1 (some compute nodes have a failing /scratch1 mount): the env copy, venv_skx, code archive
  code_3cbb85f (walks) and the md5-verified copies of the 2014-start w3000 design caches (code/setup_home.sh,
  code/stage_home.sbatch); design/ holds the design's MD5SUMS and design_check.json (the same design as
  results/pooled_trees_final_2014start_w3000).
- Threads: lines 1-30 ran at 20 threads (array 13324443; its pending lines 31-90 cancelled at 03:18), lines 31-90 at
  8 threads with 4 GB (array 13325348): the cluster's free cores were fragmented; the forecasts do not depend on the
  thread count (gate (d)).  Each piece's meta records its threads, host and CPU model; 65 pieces on EPYC 7513, 25 on
  EPYC 7542 (one CPU family by gate (b)).
- Merge (job 13334408) with code_c989ee0: the runner of 3cbb85f with the one-CPU checks read by CPU family
  (CPU_FAMILY: EPYC 7513 / 7542); the 3cbb85f merge would have refused pieces from both models.  No piece was resumed
  or resubmitted.
- Scores (scores/): code/score_trees_final.py of results/pooled_trees_final_2014start_w3000 on this directory with the
  linear w3000 run of results/window_sweep_2014start_2026-10-22.
