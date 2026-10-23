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
