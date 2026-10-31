# ALL48 study: status (results/all48_2026-10-30/RULE.txt)

| When (2026-10-30) | What |
|---|---|
| 13:00 | Part 1 (train3way ALL48 linear forecasts, pre-FOMC-fix design) scored: part1/ (71544f3). |
| 14:52 | Part 2 linear chain resubmitted from the design gate after the gate fix a7b9741 (CARC jobids a48_3). |
| 14:57-16:50 | Part 2 linear walks: 19 tasks (array 13618318), all started at once; the full-set elastic net's 5 sequential chunks (~26,000 refits each) were the critical path (1 h 21 min - 1 h 47 min). Export 13618319 and score 13618320 done 16:50:57. |
| 16:52 | Tree gate epyc7513 (13618321): PASS for lgbm, xgb, rf_skx; epyc7542 (13618322) running. |
| 16:52-16:57 | Hoffman2 staged (code md5 vs git; designs D13 and D48 and trees48.txt md5 vs CARC; reference gate pieces of epyc7513) and gated per CPU model, each model's gate split by class (lgbm+xgb, rf_skx): Gold 6240 594450/594451, Gold 6442Y 594540/594544, Xeon 6736P 594546/594547. |
| 16:57 | Tree lines split by reservation (cluster/ablate/claim_reserve.sh): CARC 1-515 and 956-7000; Hoffman2 516-955 and 7001-10191 (markers reserved-h2 on CARC, reserved-carc on Hoffman2). CARC pools: main 13621550 (100 x 16 CPUs, two 8-thread workers, 4 h, epyc-7513, d05-07/d05-08 excluded), oneweek 13621551 (13 x 16 CPUs). Hoffman2 pools after their gates. |
| 17:00 | Part 2 linear scores fetched (md5 OK): part2/scores_linear/ (checks.json, dm.csv, metrics.csv, per_interval.csv, tables.txt). Part 1 checks.json refetched (it was committed empty; md5 65f6967d). |
| 17:07-17:16 | Hoffman2 XGBoost gates: the 16:52 "lgbm,xgb" gates had run LightGBM only (sbatch --export split GEST=lgbm,xgb at the comma; LightGBM PASS on Gold 6240, Gold 6442Y, Xeon 6736P); XGBoost-only gates PASS on all three (17:15-17:16); XGBoost-phase pools afterok on them. |
| 17:54-19:15 | Forest gates PASS: Gold 6442Y 17:54, Gold 6240 18:28, Xeon 6736P 18:56 (Hoffman2); EPYC 7542 19:15 (CARC 13618322, all three classes). Hoffman2 +5 Gold 6240 forest pools at 18:36 (16-node campus24 cap reached; the 6442Y / 6736P pools wait on it). CARC pools on epyc-7542 13625140 (60 x 16 CPUs, NW 2, 4 h) at 19:36. |
| 19:35 | Lines done: CARC 168, Hoffman2 67; none failed. |
