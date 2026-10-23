# Pooled trees, final design (2014 start), w3500, bar refits, 2014-01-02..2024-04-30 (CARC Discovery)

Runner commit 654c476 (code at 7e25336 when submitted), python /scratch1/jc_905/envs/harxhar-r13/bin/python (python 3.12.15, numpy 1.26.4, pandas 2.3.3, scipy 1.17.1, sklearn 1.9.1, lightgbm 4.6.0, xgboost 3.2.0, pyarrow 19.0.1). WINDOW_DAYS=3500, REFIT=bar. First row 2014-01-02 10:00:00, 33,970 bars an arm on 2,643 days.

## Jobs
- 13320635 tf_gb_w3500 (32 tasks): {'COMPLETED': 32}, 752.4 allocated CPU-h, 706.1 used.
- 13325807 merge.

## Merge
forecasts.parquet: 32 pieces, arms ['lgbm/all_features', 'lgbm/baseline', 'xgb/all_features', 'xgb/baseline'].

## Pieces by arm (pieces / tasks, rows, CPU-hours = wall_sec x threads)
- lgbm all_features: 11 / 11 pieces, 33970 rows, 264.3 CPU-h, threads [8], longest piece 3.33 h
- lgbm baseline: 2 / 2 pieces, 33970 rows, 42.7 CPU-h, threads [8], longest piece 3.16 h
- xgb all_features: 17 / 17 pieces, 33970 rows, 417.8 CPU-h, threads [8], longest piece 4.0 h
- xgb baseline: 2 / 2 pieces, 33970 rows, 26.8 CPU-h, threads [8], longest piece 1.7 h
- total: 751.7 CPU-h

## CPU model by piece
- AMD EPYC 7513 32-Core Processor: 28
- AMD EPYC 7542 32-Core Processor: 4

## Files
forecasts.parquet + forecasts.json, pieces/*.json (piece parquets stay on CARC: /scratch1/jc_905/trees_final/runs/w3500/pieces), tasks*.txt, timing/, jobs.csv (sacct), progress.csv, run.json, design/ (md5, design_check.json), scores/, code/.

## Notes
- The 3,500-day robustness run of the final design: the same runner, environment, cadence, piece rule and gate as
  results/pooled_trees_final_2014start_w3000 (gate/ there: 24 of 24 printed pieces reproduced bit for bit); code/ holds
  copies of the scripts used (the originals in results/pooled_trees_final_2014start_w3000/code).
- Design: design/ (md5 of the copied 2014-start w3500 caches = the window sweep's); trees from 2014-01-02 10:00
  (design row 50,777); the first fit window starts at design row 5,277.
- The w3500 time probe (timing/) came out below the w3000 one (node contention), so the pieces ran longer than the
  3-hour target (progress.csv); the forecasts do not depend on it.
- The stock scikit-learn forest (tasks_rf.txt, array 13321041) was cancelled before any task started; the forest of
  the final rerun is rf_skx (results/pooled_rf_skx_2014start_w3500/).
