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
