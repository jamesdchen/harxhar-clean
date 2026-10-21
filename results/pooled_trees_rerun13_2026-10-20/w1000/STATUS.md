# Pooled trees rerun13, w1000, bar refits, 2012-01-03..2024-04-30 (CARC Discovery)

Runner commit db59af2 (CARC checkout db59af24). WINDOW_DAYS=1000, REFIT=bar.

## Jobs
- 13262219 ptr13 (main array, 218 tasks): 105 COMPLETED, 81 FAILED on Intel d-nodes (rerun in 13265651), 32 CANCELLED (moved to 13265608).
- 13265608 ptr13p (32 tasks, epyc-7513): 32 COMPLETED.
- 13265651 ptr13r (81 reruns, epyc-7513, resumed from checkpoints): 81 COMPLETED.
- 13278970 ptr13_merge: COMPLETED in 29 s on a01-11.

All 218 task indices COMPLETED; 218 finished piece .json files.

## Merge
Merged 242,796 rows (6 arms x 40,466 bars) into forecasts.parquet.

## Pieces by arm (pieces, rows, CPU-hours = wall_sec x threads)
- lgbm all_features: 7 pieces, 40466 rows, 159.3 CPU-h
- lgbm baseline: 3 pieces, 40466 rows, 16.5 CPU-h
- rf all_features: 190 pieces, 40466 rows, 4994.7 CPU-h
- rf baseline: 5 pieces, 40466 rows, 104.6 CPU-h
- xgb all_features: 11 pieces, 40466 rows, 278.0 CPU-h
- xgb baseline: 2 pieces, 40466 rows, 11.7 CPU-h
- total: 5564.8 CPU-h

## CPU model by piece (progress.csv `cpu` column)
- AMD EPYC 7513 32-Core Processor: 179
- AMD EPYC 7542 32-Core Processor: 21
- Intel(R) Xeon(R) Silver 4116 CPU @ 2.10GHz: 18

## Files
forecasts.parquet + forecasts.json, pieces/*.json (piece parquets not committed), tasks.txt, timing/, x_colstats_carc.csv, jobs.csv (sacct), progress.csv (one row per piece, with CPU model).
