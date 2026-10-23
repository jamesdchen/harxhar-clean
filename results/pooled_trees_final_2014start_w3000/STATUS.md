# Pooled trees, final design (2014 start), w3000, bar refits, 2014-01-02..2024-04-30 (CARC Discovery)

Runner commit 654c476 (code at 7e25336 when submitted), python /scratch1/jc_905/envs/harxhar-r13/bin/python (python 3.12.15, numpy 1.26.4, pandas 2.3.3, scipy 1.17.1, sklearn 1.9.1, lightgbm 4.6.0, xgboost 3.2.0, pyarrow 19.0.1). WINDOW_DAYS=3000, REFIT=bar. First row 2014-01-02 10:00:00, 33,970 bars an arm on 2,643 days.

## Jobs
- 13320585 tf_gb_w3000 (50 tasks): {'COMPLETED': 50}, 699.9 allocated CPU-h, 658.6 used.
- 13324595 merge.

## Merge
forecasts.parquet: 50 pieces, arms ['lgbm/all_features', 'lgbm/baseline', 'xgb/all_features', 'xgb/baseline'].

## Pieces by arm (pieces / tasks, rows, CPU-hours = wall_sec x threads)
- lgbm all_features: 14 / 14 pieces, 33970 rows, 249.5 CPU-h, threads [8], longest piece 3.29 h
- lgbm baseline: 3 / 3 pieces, 33970 rows, 39.3 CPU-h, threads [8], longest piece 2.0 h
- xgb all_features: 29 / 29 pieces, 33970 rows, 383.8 CPU-h, threads [8], longest piece 1.99 h
- xgb baseline: 4 / 4 pieces, 33970 rows, 26.5 CPU-h, threads [8], longest piece 1.16 h
- total: 699.0 CPU-h

## CPU model by piece
- AMD EPYC 7513 32-Core Processor: 35
- AMD EPYC 7542 32-Core Processor: 15

## Files
forecasts.parquet + forecasts.json, pieces/*.json (piece parquets stay on CARC: /scratch1/jc_905/trees_final/runs/w3000/pieces), tasks*.txt, timing/, jobs.csv (sacct), progress.csv, run.json, design/ (md5, design_check.json), scores/, code/.

## Notes
- Gate (gate/gate_check.csv): 24 of 24 pieces of the printed 2012-start w1000 run reproduced bit for bit -- the
  printed run's runner db59af24 (A) and this branch's runner (B), EPYC 7513 and 7542, the forest at 16 threads; 4
  whole pieces (xgb / lgbm HAR) and the first and last 1-3 days of 20 others through the resume path.
- Design: design/ (md5 of the copied 2014-start caches = the window sweep's; X, y, B and dates of w3000 equal the
  2012-start w3000 caches of results/tree_window_presample_final_2026-10-22); trees from 2014-01-02 10:00 (design
  row 50,777), the linear walk from 2013-12-27 16:00 (row 50,750).
- The stock scikit-learn forest (tasks_rf.txt, arrays 13321040 / 13321041) was cancelled on the owner's decision; its
  14 started pieces (rf HAR, 32.4 allocated CPU-h, jobs_rf_sklearn_cancelled.csv) are discarded.  The forest of the
  final rerun is rf_skx (results/pooled_rf_skx_2014start_w<W>/, RULE_rf_skx.txt).
- daycheck/: one month (2014-01) of day refits against these bar forecasts (information only; every tree refits
  every bar).
