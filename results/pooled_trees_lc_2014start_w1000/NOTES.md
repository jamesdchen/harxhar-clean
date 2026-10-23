## Notes
- Cluster: every piece of every arm on Hoffman2 (UGE array 15250855), one thread a task; the merge on Hoffman2 (15252387).
- Gate (gate/gate_check.csv, gate/cpu_gate.csv): pieces of the printed 2012-start w1000 run (CARC) re-walked on Hoffman2,
  bit-equal (DataFrame.equals, refitted rows, Arrow schema) at 8 threads on Gold 6140 / 6240 / 6342 and at 1 thread on
  Gold 6140 / 6240 / 6342, Xeon 6736P and E5-2650 v4; every CPU model of this run's pieces is gated (Hoffman2's arch
  request does not pin the CPU, so the gate is judged by the CPU each piece records).
- The HAR-benchmark arms equal the printed 2012-start w1000 run bit for bit on its 2014-01-02 .. 2024-04-30 rows
  (har_vs_printed.txt): the HAR design is the same file in both.
- Scores: code/score_trees_final.py with the writeup modules at 654c476 (scores/; cross_window_dm.csv = 2,000 vs 1,000 days).
- The random forest was not run here (owner decision 2026-10-23: the forest runs on CARC); tasks_rf_not_run.txt is the
  rule's task file of the forest arms, never submitted (arrays 15250857 / 15250858 held, then deleted unstarted).
