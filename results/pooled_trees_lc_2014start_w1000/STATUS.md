# Pooled trees, learning curve, final design (2014 start), w1000, bar refits, 2014-01-02..2024-04-30 (Hoffman2)

LightGBM and XGBoost, HAR benchmark and full set.  Runner 654c476 (code tree 475319a, job scripts f88c6b4), python 3.12.3, numpy 1.26.4, lightgbm 4.6.0, xgboost 3.2.0, sklearn 1.9.1, one thread a task.  WINDOW_DAYS=1000, REFIT=bar.  First row 2014-01-02 10:00:00, 33,970 bars an arm on 2,643 days.

## UGE jobs
- 15250855 tlc_gb_w1000 (79 records): exit status {0: 79}, 246.1 slot-hours, 244.9 CPU-hours, 25 hosts.
- 15252387 tlc_merge_w1000 (1 records): exit status {0: 1}, 0.0 slot-hours, 0.0 CPU-hours, 1 hosts.

## Merge
forecasts.parquet: 79 pieces, arms ['lgbm/all_features', 'lgbm/baseline', 'xgb/all_features', 'xgb/baseline'].

## Pieces by arm (pieces / tasks, rows, CPU-hours = wall_sec x threads)
- lgbm all_features: 34 / 34 pieces, 33970 rows, 119.9 CPU-h, threads [1], longest piece 4.81 h
- lgbm baseline: 2 / 2 pieces, 33970 rows, 4.4 CPU-h, threads [1], longest piece 2.21 h
- xgb all_features: 41 / 41 pieces, 33970 rows, 116.8 CPU-h, threads [1], longest piece 4.48 h
- xgb baseline: 2 / 2 pieces, 33970 rows, 4.7 CPU-h, threads [1], longest piece 2.72 h
- total: 245.7 CPU-h

## CPU model by piece (each model passed the 1-thread gate: ../pooled_trees_lc_2014start_w1000/gate/)
- Intel(R) Xeon(R) Gold 6140 CPU @ 2.30GHz: 42
- Intel(R) Xeon(R) Gold 6240 CPU @ 2.60GHz: 16
- Intel(R) Xeon(R) 6736P: 16
- Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz: 4
- Intel(R) Xeon(R) Gold 6342 CPU @ 2.80GHz: 1

## Files
forecasts.parquet + forecasts.json, pieces/*.json (the piece parquets stay on Hoffman2: /u/scratch/j/jamesdc1/tlc/runs/w1000/pieces), tasks*.txt, timing/, jobs.csv (qacct), progress.csv, run.json, design/ (md5, design_check.json), environment/ (pip freeze), scores/, code/.

## Notes
- Cluster: every piece of every arm on Hoffman2 (UGE array 15250855), one thread a task; the merge on Hoffman2 (15252387).
- Gate (gate/gate_check.csv, gate/cpu_gate.csv): pieces of the printed 2012-start w1000 run (CARC) re-walked on Hoffman2,
  54 of 54 bit-equal (DataFrame.equals, refitted rows, Arrow schema): at 8 threads on Gold 6140 / 6240 / 6342 and E5-2650 v4, at 1 thread on
  Gold 6140 / 6240 / 6342, Xeon 6736P and E5-2650 v4; every CPU model of this run's pieces is gated (Hoffman2's arch
  request does not pin the CPU, so the gate is judged by the CPU each piece records).
- The HAR-benchmark arms equal the printed 2012-start w1000 run bit for bit on its 2014-01-02 .. 2024-04-30 rows
  (har_vs_printed.txt): the HAR design is the same file in both.
- Scores: code/score_trees_final.py with the writeup modules at 654c476 (scores/; cross_window_dm.csv = 2,000 vs 1,000 days).
- The random forest was not run here (owner decision 2026-10-23: the forest runs on CARC); tasks_rf_not_run.txt is the
  rule's task file of the forest arms, never submitted (arrays 15250857 / 15250858 held, then deleted unstarted).
