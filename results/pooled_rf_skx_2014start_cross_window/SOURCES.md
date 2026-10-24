# Sources of the forest's cross-window comparison (2026-10-23)

- W = 1,000 / 2,000 / 3,000 / 3,500: this branch (insurance-selling-2026-09-19-e34-trees-final),
  results/pooled_rf_skx_2014start_w<W>/ forecasts.parquet, forecasts.json and scores/econ.csv (rf_skx on CARC; commits
  e9b01fa, 6e9a11f + ae3c6ad, de40ef0, 405c01e).
- Linear rows (the scored set): results/window_sweep_2014start_2026-10-22/w<W>/forecasts.parquet of each window, read
  from the 654c476 worktree.
- Scoring modules: writeup/, notebooks/, experiments/ and src/ of the worktree at 654c476
  (C:\Users\james\AppData\Local\Temp\score654, read only, python -I -B), the version that scored every forest run here.
- Code: code/cross_window.py = results/pooled_trees_lc_2014start_w1000/code/cross_window.py of
  insurance-selling-2026-09-19-e34-tree-lc (55cdaae) with the arms rf_skx / {HAR, full} in place of lgbm, xgb and the
  header lines changed; nothing else.
- Command: python -I -B results/pooled_rf_skx_2014start_cross_window/code/cross_window.py <score654> \
    results/pooled_rf_skx_2014start_cross_window 1000=<w1000 run>=<linear w1000> 2000=... 3000=... 3500=...
  (stdout.txt).  The per-window scores/ were not touched (md5 before and after equal).
- Checks: the 3,000 -> 3,500 rows of dm_pairs.csv equal the scorer's cross_window_dm.csv (3,500 vs 3,000) exactly; QLIKE
  by W equals each run's scores/metrics.csv.
