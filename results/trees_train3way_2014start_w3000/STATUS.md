# Three-way training-sample comparison, pooled trees (status)

Rule: `RULE.txt` (committed before any job).  Code: `experiments/trees_train3way.py` (walk / merge / time / tasks /
gates / design check), `experiments/trees_train3way_score.py` (CARC scoring), `experiments/trees_train3way_econ.py`
(local close economics), `code/` (CARC job scripts: `env.sh`, `stage.sbatch`, `line.sbatch`, `walk.sbatch`,
`submit1.sh`).  Staging: `/scratch1/jc_905/trees3way` on CARC Discovery.

| Time (PDT) | Event |
|---|---|
| 2026-10-24 00:06 | RULE.txt and code committed (d6e90d5) and pushed; no job before it |
| 00:08 | phase 1 submitted (`code/submit1.sh d6e90d5`; code archive `/scratch1/jc_905/trees3way/code_d6e90d5`; import of the module checked on the login node with both class environments) |

Phase 1 jobs (CARC, account pollok_1603, partition main, EPYC 7513 / 7542):

| Job | Name | What | Depends on |
|---|---|---|---|
| 13369745 | t3t_stage | design copies (md5) + design check (gate 0) | - |
| 13369746 | t3t_g1a | gate 1, 8 threads (6 tasks) | stage |
| 13369747 | t3t_g1b | gate 1, the forest at 20 threads (2) | stage |
| 13369748 | t3t_g2a | gate 2, LightGBM / XGBoost and the ALL48_N3k forest (20) | stage |
| 13369749 | t3t_g2b | gate 2, the ALL48 forest at 20 threads (4) | stage |
| 13369750 | t3t_tm8 | time probe, 8 threads (10) | stage |
| 13369751 | t3t_tm20 | time probe, the forest at 20 threads (3) | stage |
| 13369752 | t3t_gate | gate 1 + gate 2 comparisons (2) | g1a, g1b, g2a, g2b |
| 13369753 | t3t_tasks | task files and projections of every arm (6) | tm8, tm20 |
| 13369754 | t3t_cap | projection against 25,000 CPU-hours | tasks |
| 13369755 | t3t_w1g | walks CLOSE / RTH13_N3k / ALL48_N3k, LightGBM + XGBoost (<= 50, %30) | gate, tasks |
| 13369756 | t3t_w1r | the same arms, the forest (<= 50, %30) | gate, tasks |
| 13369757 | t3t_w2g | walks ALL48_N39k, LightGBM + XGBoost (<= 40, %15) | gate, cap |
| 13369758 | t3t_w2r | walks ALL48_N39k, the forest at 20 threads (<= 150, %25) | gate, cap |
| 13369759 | t3t_m1 | merge CLOSE, RTH13_N3k, ALL48_N3k (3) | w1g, w1r (afterany) |
| 13369760 | t3t_s1 | interim scores -> scores_w1 | m1 |
| 13369761 | t3t_m2 | merge ALL48_N39k | w2g, w2r (afterany) |
| 13369762 | t3t_s2 | interim scores -> scores_w2 (last job of phase 1) | m2, s1 |

Phase 2 (ALL48, 3,000 days): `code/submit2.sh` after the gates and the projection are read.

## Gate 0, gate 1, the probe and the cap (2026-10-24 00:40 PDT)

- Gate 0 (`design_check.json`, stage 13369745): PASS -- the 13-bar caches md5-identical to the templates' design; configs;
  ALL48 names = RTH13's + is_overnight; y and B equal at the 33,970 origins (CLOSE: 2,579 16:00 origins); the linear
  CLOSE cache = the 16:00 rows byte for byte (the trees' CLOSE windows inside it); the linear N3k / N39k caches = the
  sibling's rows from W_sibling - N on.
- Gate 1 (`gate/g1`, comparison run on the login node 00:41; the gate job will rerun it): PASS -- 8 of 8 pieces (3
  classes x 2 sets at 8 threads, the forest also at 20), 20 origins each, pred_adj text-identical to the templates'
  forecasts (max relative difference 0), true_adj and B equal; EPYC 7513 and 7542.
- Time probe (13369750 / 13369751) -> task files and projections (tasks 13369753); cap 13369754 FAILED with exit 3 =
  the designed stop: projection 33,649 CPU-hours > 25,000 (`runs/tasks/projection.json`).

| Arm | LightGBM | XGBoost | Forest (rf_skx) | Total |
|---|---|---|---|---|
| CLOSE (8 threads) | 4.6 | 11.0 | 7.9 | 23.5 |
| RTH13_N3k (8) | 59.6 | 168.3 | 102.2 | 330.1 |
| ALL48_N3k (8) | 56.5 | 134.7 | 127.6 | 318.8 |
| ALL48_N39k (8; forest 20) | 322.8 | 652.2 | 2,596.8 | 3,571.8 |
| ALL48 (8; forest 20) | 721.9 | 928.3 | 27,754.5 | 29,404.7 |
| total | 1,165.4 | 1,894.5 | 30,589.0 | 33,648.9 |

  ALL48 forest, full set: 145.9 s a refit at 20 threads (27,526.9 CPU-h; HAR 227.6) = 9.8x RTH13's probe at 20 threads
  (14.8 s); calibration: RTH13's forest probe at 8 threads (34.0 s -> 2,569 CPU-h) against the template's actual
  2,575.5, at 20 threads 2,797.
- 00:41: the cap-blocked ALL48_N39k walks 13369757 / 13369758 (DependencyNeverSatisfied) and their merge / score
  13369761 / 13369762 cancelled (never started).  Not resubmitted: ALL48_N39k and ALL48 wait for the owner.

## Gate 2 and addendum A (00:45-01:00 PDT)

- Gate 2, pairs finished by 00:42: ALL48_N3k 6 of 6 identical (91 every-row refits, 17 scored origins each); ALL48 (38
  every-row refits, 3 scored origins) LightGBM HAR, XGBoost HAR and full, the forest HAR identical; LightGBM full NOT
  identical at 2014-01-03 10:00 (0.8102594551794627 scored-only, 0.8102594551794626 every row; 1 ulp); the forest
  full still running (13369749_3).
- Diagnostic srun 13371033 (a04-17, EPYC 7513), the 10:00 origin fitted alone in 8 fresh processes: template config
  626 / 627 / 626 / 626 (LightGBM's auto choice col / col / col / row-wise), force_col_wise 626 / 626, force_row_wise
  627 / 627 -> LightGBM run-to-run non-determinism at 1 ulp on the 144,000-row window, not state across refits
  (a duplicate batch copy 13371140 was submitted while the srun ran; same script).
- RULE.txt addendum A: the cheap arms run after gate 1 + gate 2 on ALL48_N3k (new job t3t_gate3k); ALL48_N39k and ALL48
  wait for the owner.
- 00:47: code archive `code_028cfbb` (walk code unchanged; gate2 --arm added); t3t_gate3k 13371164 (gate1 + gate2
  --arm ALL48_N3k) COMPLETED: GATE 1 PASS, GATE 2 (ALL48_N3k) PASS; walk arrays 13369755 (w1g, 25 lines) / 13369756
  (w1r, 15 lines) re-pointed to afterok:13371164 (now pending on resources); merge 13369759 and interim score 13369760
  (scores_w1) as submitted = the last job of the chain.  Still running: 13369749_3 (gate 2, the ALL48 forest full
  set), 13369752 (the full gate comparison; no dependents now), 13371140 (the diagnostic's batch copy).

## Addendum B: the owner's scope and LightGBM col-wise (01:25 PDT)

RULE.txt addendum B committed in 8b85295 before its jobs; code archive `code_8b85295` (ARMS lgbm_col_wise for ALL48 and
ALL48_N39k; ALL48_N3k's LightGBM walks had started with the template setting and keep it).  Submitted 01:26 PDT
(`code/submit2b.sh code_8b85295 13369760 13369749`):

| Job | Name | What | Depends on |
|---|---|---|---|
| 13371780 | t3t_gc | gate C walks (24 lines: ALL48 LightGBM every / scored / scored2 / one1-4, ALL48_N39k scored / scored2 / one1-2, RTH13 col-wise record; both sets) | - |
| 13371781 | t3t_gcc | gate C (a): gate2 --arm ALL48 --est lgbm, gaterep | gc |
| 13371782 | t3t_gcb | gate C (b) record: colwise_vs_template.json | gc (afterany) |
| 13371783 | t3t_g2x | gate 2 on ALL48's XGBoost | - |
| 13371784 | t3t_g2r | gate 2 on ALL48's forest | 13369749 (its full set still running) |
| 13371785 | t3t_w2l | ALL48_N39k LightGBM, col-wise (12, %12) | gcc |
| 13371786 | t3t_w2x | ALL48_N39k XGBoost (23, %15) | g2x |
| 13371787 | t3t_w3l | ALL48 LightGBM, col-wise (31, %15) | gcc |
| 13371788 | t3t_w3x | ALL48 XGBoost (40, %15) | g2x |
| 13371789 | t3t_w2r | ALL48_N39k forest, 20 threads (44, %20) | g2r |
| 13371790 | t3t_w3h | ALL48 forest HAR set, 20 threads (4, %4) | g2r |
| 13371791 / 13371792 | t3t_m2 / t3t_s2 | merge ALL48_N39k / scores_w2 | w2l, w2x, w2r / m2, s1 13369760 |
| 13371793 / 13371794 | t3t_m3 / t3t_s3 | merge ALL48 (--skip rf_skx/all_features) / final scores | w3l, w3x, w3h / m3, s2 |

Not submitted: ALL48's forest full set (runs/tasks/w3rf_full.txt, 368 lines; `code/submit3_rf_full.sh`).

## Gate C result (01:35 PDT; read 01:55): FAIL, a genuine LightGBM mismatch (not infrastructure); LightGBM walks held

- Every gate C job COMPLETED (24 walk lines, the record 13371782); no I/O / Lustre error in any log; t3t_gcc 13371781
  task 1 (gate2 --arm ALL48 --est lgbm) PASS, task 2 (gaterep) exit 3 = the gate's own failure.
- (a) gate 2 with force_col_wise: ALL48 LightGBM both sets, every row vs scored-only, bit-identical (3 of 3 origins).
- (a) repeats: ALL48_N39k both sets identical (scored2 at 3 origins, one1-2 at 10:00); ALL48 HAR set identical (scored2,
  one1-4); ALL48 full set: scored, scored2 and every give 2014-01-03 10:00 = 0.8102594551794626, the 10:00 origin
  alone in fresh processes gave ...627 (one1, b10-19), ...627 (one2, b10-19), ...626 (one3, b10-19), ...627 (one4,
  b22-26) -- the same node gives both values, so force_col_wise does not remove the 1-ulp run-to-run
  non-determinism (2.2e-16 relative; 15:30 and 16:00 identical in every run).  The diagnostic's batch copy (13371140,
  a03-05) agrees: template setting 627 / 626 / 626 / 626, force_col_wise 626 / 627, force_row_wise 627 / 627.
- (b) record: force_col_wise against the template setting on RTH13 (gate-1 origins, both sets): text-identical, max
  relative difference 0.
- Gate 2 on ALL48's forest (t3t_g2r 13371784): PASS (full set bit-identical, 38 every-row refits at 151.5 s, 3 scored
  origins; HAR set too); XGBoost (13371783): PASS.  The forest walks 13371789 / 13371790 and the XGBoost walks
  13371786 / 13371788 run.
- Held: LightGBM walks 13371785 (ALL48_N39k) and 13371787 (ALL48), DependencyNeverSatisfied on 13371781; with them the
  merges / scores 13371791-13371794 (afterany on the LightGBM arrays) wait.  Nothing loosened; the owner decides.
