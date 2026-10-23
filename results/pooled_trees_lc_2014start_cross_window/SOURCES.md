# Sources of the cross-window comparison (2026-10-23)

- W = 1,000 and 2,000: this branch (insurance-selling-2026-09-19-e34-tree-lc), results/pooled_trees_lc_2014start_w1000/ and
  _w2000/forecasts.parquet (Hoffman2, one thread a task, every CPU model gated against the printed CARC pieces).
- W = 3,000 and 3,500: branch insurance-selling-2026-09-19-e34-trees-final at c989ee0,
  results/pooled_trees_final_2014start_w3000/ and _w3500/ forecasts.parquet, forecasts.json and scores/econ.csv (CARC,
  8 threads), read with `git show` (byte-equal copies, checked with cmp).
- Linear rows (the scored set): results/window_sweep_2014start_2026-10-22/w<W>/forecasts.parquet of each window.
- Scoring modules: writeup/ and notebooks/ of the worktree at 654c476 (C:\Users\james\AppData\Local\Temp\score654,
  read only, python -B), the version that scored every run here; the W = 1,000 re-score with them reproduced the
  committed scores byte for byte.
- Command: python -I -B results/pooled_trees_lc_2014start_w1000/code/cross_window.py <score654> \
    results/pooled_trees_lc_2014start_cross_window 1000=<w1000 run>=<linear w1000> 2000=... 3000=... 3500=...
