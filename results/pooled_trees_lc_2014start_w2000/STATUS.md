# Pooled trees, learning curve, final design (2014 start), w2000, bar refits, 2014-01-02..2024-04-30 (Hoffman2)

LightGBM and XGBoost, HAR benchmark and full set.  Runner 654c476 (code tree 475319a, job scripts f88c6b4), python 3.12.3, numpy 1.26.4, lightgbm 4.6.0, xgboost 3.2.0, sklearn 1.9.1, one thread a task.  WINDOW_DAYS=2000, REFIT=bar.  First row 2014-01-02 10:00:00, 33,970 bars an arm on 2,643 days.

## UGE jobs
- 15250856 tlc_gb_w2000 (122 records): exit status {0: 120, 137: 2}, 364.4 slot-hours, 362.5 CPU-hours, 28 hosts.
- 15253205 tlc_gb_w2000r (2 records): exit status {0: 2}, 2.7 slot-hours, 2.6 CPU-hours, 1 hosts.
- 15254375 tlc_merge_w2000 (1 records): exit status {0: 1}, 0.0 slot-hours, 0.0 CPU-hours, 1 hosts.

## Merge
forecasts.parquet: 122 pieces, arms ['lgbm/all_features', 'lgbm/baseline', 'xgb/all_features', 'xgb/baseline'].

## Pieces by arm (pieces / tasks, rows, CPU-hours = wall_sec x threads)
- lgbm all_features: 50 / 50 pieces, 33970 rows, 172.7 CPU-h, threads [1], longest piece 7.76 h
- lgbm baseline: 3 / 3 pieces, 33970 rows, 9.2 CPU-h, threads [1], longest piece 5.88 h
- xgb all_features: 66 / 66 pieces, 33970 rows, 177.6 CPU-h, threads [1], longest piece 7.33 h
- xgb baseline: 3 / 3 pieces, 33970 rows, 6.9 CPU-h, threads [1], longest piece 2.47 h
- total: 366.4 CPU-h

## CPU model by piece (each model passed the 1-thread gate: ../pooled_trees_lc_2014start_w1000/gate/)
- Intel(R) Xeon(R) 6736P: 50
- Intel(R) Xeon(R) Gold 6140 CPU @ 2.30GHz: 45
- Intel(R) Xeon(R) Gold 6240 CPU @ 2.60GHz: 13
- Intel(R) Xeon(R) Gold 6342 CPU @ 2.80GHz: 10
- Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz: 4

## Files
forecasts.parquet + forecasts.json, pieces/*.json (the piece parquets stay on Hoffman2: /u/scratch/j/jamesdc1/tlc/runs/w2000/pieces), tasks*.txt, timing/, jobs.csv (qacct), progress.csv, run.json, design/ (md5, design_check.json), environment/ (pip freeze), scores/, code/.

## Notes
- Cluster: every piece of every arm on Hoffman2 (UGE arrays 15250856 and 15253205), one thread a task; the merge on
  Hoffman2 (15254375).
- Tasks 18 (lgbm all_features 12/50) and 116 (xgb all_features 60/66) of 15250856 crawled on an overloaded node (n1154,
  Xeon 6736P, 45-60 s a refit) and would have overrun h_rt; they were deleted (qdel 15250856.18 15250856.116, exit 137
  in jobs.csv, approved) and resubmitted as 15253205 (tasks 18 and 116, hostname != n1154), which resumed each piece from
  its checkpoint (544 and 340 rows on disk) on n7368 (E5-2650 v4).  Those two pieces were computed partly on the Xeon
  6736P and partly on the E5-2650 v4 (the meta records the last CPU); both models passed the 1-thread gate, and a
  resumed piece equals an uninterrupted one bit for bit (2026-10-20 runner test).  wall_sec / CPU-hours of those two
  pieces include the slow first run.
- Gate and CPU check: ../pooled_trees_lc_2014start_w1000/gate/ (every CPU model of this run's pieces gated).
- Scores: scored together with W = 1,000 (code/score_trees_final.py, writeup modules at 654c476); cross_window_dm.csv =
  2,000 vs 1,000 days.  The 1k / 2k / 3k / 3.5k comparison: ../pooled_trees_lc_2014start_cross_window/.
- Code: ../pooled_trees_lc_2014start_w1000/code/ (shared by both windows).
- The random forest was not run here (owner decision 2026-10-23); tasks_rf_not_run.txt was never submitted.
