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
