## Notes
- The forest of the final rerun at W = 3,000 days: rf_skx (the scikit-learn forest on oneDAL, scikit-learn-intelex
  2026.2.0), its rule and dated addenda in RULE_rf_skx.txt, its gate in gate/ (gate/fold/gate_skx.csv: fold-3 fits equal
  to the benchmark's engine B on EPYC 7513 and 7542; gate/P*/: walk pieces equal across runs and CPU models;
  gate/t20/gate_skx.csv: 20 threads = 8 threads; gate/logs/: the gate jobs' logs).
- Everything on /home1 (some compute nodes have a failing /scratch1 mount): the env copy, venv_skx, code archive
  code_3cbb85f (walks) and the md5-verified copies of the 2014-start w3000 design caches (code/setup_home.sh,
  code/stage_home.sbatch); design/ holds the design's MD5SUMS and design_check.json (the same design as
  results/pooled_trees_final_2014start_w3000).
- Threads: lines 1-30 ran at 20 threads (array 13324443; its pending lines 31-90 cancelled at 03:18), lines 31-90 at
  8 threads with 4 GB (array 13325348): the cluster's free cores were fragmented; the forecasts do not depend on the
  thread count (gate (d)).  Each piece's meta records its threads, host and CPU model; 65 pieces on EPYC 7513, 25 on
  EPYC 7542 (one CPU family by gate (b)).
- Merge (job 13334408) with code_c989ee0: the runner of 3cbb85f with the one-CPU checks read by CPU family
  (CPU_FAMILY: EPYC 7513 / 7542); the 3cbb85f merge would have refused pieces from both models.  No piece was resumed
  or resubmitted.
- Scores (scores/): code/score_trees_final.py of results/pooled_trees_final_2014start_w3000 on this directory with the
  linear w3000 run of results/window_sweep_2014start_2026-10-22.
