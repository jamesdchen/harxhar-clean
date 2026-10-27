# Factor test (CARC): status

| when (PDT) | job | what |
|---|---|---|
| 10-27 12:34 | 13498235 | maps (8 CPUs, EPYC 7513/7542) + gate GM on 8 days |
| 10-27 12:34 | 13498236 | gate (GB, T2 inputs, timing) on EPYC 7513, after the maps |
| 10-27 12:34 | 13498237 | gate on EPYC 7542 |
| 10-27 12:34 | 13498238 | gate on Xeon 4116 (main, oneweek) |
| 10-27 12:34 | 13498239 | gate on Xeon E5-2640 v4 (oneweek) |

Staging: CARC /home1/jc_905/beyond_trees (code = git archive 75f547c; runs/factor maps, runs/gate/<model>).
| 10-27 13:35 | 13499376 | walks, partition main (EPYC 7513 / 7542, Xeon 4116), task lines 1-35 (XGBoost + 4 LightGBM pieces), 8 CPUs, 10 h |
| 10-27 13:35 | 13499377 | walks, partition oneweek (Xeon E5-2640 v4, Xeon 4116), task lines 36-51 (LightGBM), 8 CPUs, 10 h |

## Gates (all PASS; factor/gates/)
- maps (job 13498235, EPYC 7513, 6.9 min): 2,643 days; GM on 8 spread days: same active sets, max |difference| of the
  normalized scores 1.8e-13 (all 13 components).
- GB: the base arm reproduces lgbm|add_rm_fomc and xgb|add_rm_fomc text-identically on all 6 origins on EPYC 7513,
  EPYC 7542, Xeon 4116 and Xeon E5-2640 v4 (gate1.json per model).
- T2: every (class, arm) of the 16 gate pieces bit-identical across the four models (gate2.json).
- Projection (project.json): 2,150 CPU-hours (slowest model's seconds a refit x 33,970 x 8 x 1.15) <= 2,500: submitted.
- Submission note: a single 144-line array (2 h pieces) was refused by Slurm (QOSMaxSubmitJobPerUserLimit: the oneweek
  QOS allows 50 submitted jobs / 208 CPUs a user); the task file was rewritten with 6 h pieces (51 lines, the same
  rule, tasks_factor.txt) and split across the two partitions.
