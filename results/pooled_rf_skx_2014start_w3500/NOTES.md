## Notes
- The forest of the final rerun at W = 3,500 days (robustness window): rf_skx, the same rule (RULE_rf_skx.txt), gate
  (gate/), environment, code and /home1 layout as results/pooled_rf_skx_2014start_w3000 (see its NOTES.md); design/
  holds the design's MD5SUMS and design_check.json (the same design as results/pooled_trees_final_2014start_w3500).
- Arrays: 13324534 (20 threads, submitted 02:03, held, cancelled at 03:18 before any task started); 13325349 (8
  threads, 4 GB: lines 1-20 ran, its pending lines 21-118 cancelled at 04:27); 13326246 (20 threads, 4 GB: lines
  21-118).  The thread count follows the cluster's free cores; the forecasts do not depend on it (gate (d)).  Each
  piece's meta records its threads, host and CPU model; 88 pieces on EPYC 7513, 30 on EPYC 7542 (one CPU family by
  gate (b)).
- Merge (job 13335328) with code_c989ee0 (the one-CPU checks read by CPU family).  No piece was resumed or
  resubmitted.
- Scores: not yet (the scorer's pinned modules, writeup/ and experiments/ at 654c476, differ from the main worktree's
  since 10:00; see the commit that adds scores/).
