## Notes
- The forest of the final rerun at W = 1,000 days (window robustness): rf_skx, the same rule (RULE_rf_skx.txt), gate
  (gate/), environment and /home1 layout as results/pooled_rf_skx_2014start_w3000 (see its NOTES.md).  design/: the
  MD5SUMS of the /home1 design caches (= the 2014-start window sweep's, code/stage_home.sbatch there), the design check
  of w1000 / w2000 (design_check.json, stage job 13322587) and its log.
- Arrays: 13324536 (20 threads, 02:04), 13325351 (8 threads, 03:18) and 13326248 (20 threads, 04:27), each held and
  cancelled before any task started (the slots went to W3000 / W3500 first); 13329102 (8 threads, 4 GB, 6:00:00,
  submitted 08:51 with code_c989ee0, released at once by the round-robin throttle): all 26 lines.  The forecasts do not
  depend on the thread count (gate (d)); 22 pieces on EPYC 7513, 4 on EPYC 7542 (one CPU family by gate (b)).
- Runner: code_c989ee0 (3cbb85f with the one-CPU checks read by CPU family; nothing else differs), also for the merge
  (job 13348559).  No piece was resumed or resubmitted.
- Scores (scores/): code/score_trees_final.py of results/pooled_trees_final_2014start_w3000 on this run with the linear
  w1000 run of results/window_sweep_2014start_2026-10-22, on the 654c476 worktree of the w3500 scores (see
  results/pooled_rf_skx_2014start_w3500/NOTES.md); one window, so no cross_window_dm.csv.
