## Notes
- The forest of the final rerun at W = 2,000 days (window robustness): rf_skx, the same rule (RULE_rf_skx.txt), gate
  (gate/), environment and /home1 layout as results/pooled_rf_skx_2014start_w3000 (see its NOTES.md).  design/: the
  MD5SUMS of the /home1 design caches (= the 2014-start window sweep's, code/stage_home.sbatch there), the design check
  of w1000 / w2000 (design_check.json, stage job 13322587) and its log.
- Arrays: 13324535 (20 threads, 02:04) and 13325350 (8 threads, 03:18), each held and cancelled before any task
  started (the slots went to W3000 / W3500 first); 13326247 (20 threads, 4 GB, code_3cbb85f, 04:27): lines 1-14 ran,
  its pending lines 15-53 cancelled at 08:51 (20-core holes scarce); 13329101 (8 threads, 4 GB, 6:00:00, code_c989ee0):
  lines 15-53.  The forecasts do not depend on the thread count (gate (d)); 45 pieces on EPYC 7513, 8 on EPYC 7542
  (one CPU family by gate (b)).
- Runner: lines 1-14 code_3cbb85f, lines 15-53 and the merge (job 13348942) code_c989ee0 (3cbb85f with the one-CPU
  checks read by CPU family; nothing else differs).  No piece was resumed or resubmitted.
- Scores (scores/): code/score_trees_final.py of results/pooled_trees_final_2014start_w3000 on this run with the linear
  w2000 run of results/window_sweep_2014start_2026-10-22, on the 654c476 worktree of the w3500 scores (see
  results/pooled_rf_skx_2014start_w3500/NOTES.md); one window, so no cross_window_dm.csv.
