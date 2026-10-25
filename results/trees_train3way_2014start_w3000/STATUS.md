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

## Addenda C and D (02:00 PDT): the ALL48 forest full set (CARC + Hoffman2) and LightGBM deterministic

RULE.txt addenda C and D committed in d9d01fd before their jobs; code archive `code_d9d01fd`.  Submitted 02:01 PDT
(`code/submit_cd.sh`); cancelled first (never started): the addendum-B LightGBM walks 13371785 / 13371787 and their
merges / scores 13371791-13371794.

| Job | Name | What | Depends on |
|---|---|---|---|
| 13372422 | t3t_gda | gate D walks on EPYC 7513 (18 lines: ALL48 every / scored / one1-4, ALL48_N39k scored / one1-2, RTH13 record; both sets) | - |
| 13372423 | t3t_gdb | gate D walks on EPYC 7542 (14 lines: ALL48 scored2 / one5-8, ALL48_N39k scored2 / one3-4; both sets) | - |
| 13372424 | t3t_gdc | gate D (a): gate2 --arm ALL48 --est lgbm, gaterep --min-hosts 2 | gda, gdb |
| 13372425 | t3t_gdr | gate D (b) record -> gate/gdb/colwise_vs_template.json | gda (afterany) |
| 13372426 | t3t_w2l | ALL48_N39k LightGBM, deterministic + col-wise (12, %12) | gdc |
| 13372427 | t3t_w3l | ALL48 LightGBM, deterministic + col-wise (31, %15) | gdc |
| 13372428 / 13372429 | t3t_m2 / t3t_s2 | merge ALL48_N39k / scores_w2 | w2l, 13371786, 13371789 / m2, s1 13369760 |
| 13372430 / 13372431 | t3t_m3 / t3t_s3 | merge ALL48 without the forest full set / scores_w3 | w3l, 13371788, 13371790 / m3, s2 |
| 13372432 | t3t_w3f | ALL48 forest full set, lines 1-239 of w3rf_full.txt, 20 threads, 16 GB (%25) | - |
| 13372433 | t3t_m4 | ALL48 merge with every class: intake runs/pieces_h2 + --accept-cpus (HELD until the Hoffman2 pieces are in) | w3f, w3l, 13371788, 13371790 |
| 13372434 | t3t_s4 | final scores -> scores (last job of the whole run) | m4, s3 |

Hoffman2: `runs/tasks/h2_assignment.txt` (ASSIGNED w3rf_full.txt 240-368 ALL48 rf_skx all_features),
`runs/tasks/h2_code_commit.txt` (d9d01fd1...), incoming `runs/pieces_h2/`, gate runs `gate/h2/<tag>/{g1,g2}/`.

## Forest full-set split: throughput-balanced (RULE.txt note to addendum C, 02:15 PDT)

| Time (PDT) | Lines of w3rf_full.txt | From -> to | Why |
|---|---|---|---|
| 02:01 | 1-239 | CARC (array 13372432, %25) | addendum C |
| 02:01 | 240-368 | Hoffman2 (reserved) | addendum C, static 35 % |
| 02:15 | 240-348 | Hoffman2 -> CARC (second array, below) | owner: Hoffman2 slower; equal-finish-time split, Hoffman2 rounded down; none started on Hoffman2 |
| 02:15 | 349-368 | Hoffman2, first wave (20 lines) | sized to a first round on Hoffman2 after its gates; re-estimated from its measured lines a line |

Throughputs: CARC 3.7 h a line at 20 threads (probe; 25 slots -> 6.8 lines/h at %25); Hoffman2 not yet measured (its
agent's probe per CPU model).  CARC runs upward from line 1, Hoffman2 downward from 368.

| Time (PDT) | Event |
|---|---|
| 02:13 | live runs/tasks/h2_assignment.txt replaced by 2d7d2ab's (RETURNED 240-348, ASSIGNED 349-368; previous copy kept as h2_assignment.txt.bak_0215); runs/pieces_h2 and gate/h2 empty (nothing started on Hoffman2) |
| 02:13 | t3t_w3f2 13372801: lines 240-348 of w3rf_full.txt on CARC, 20 threads, 16 GB, %10 |
| 02:14 | snapshot: our running jobs 67 / 516 CPUs; LightGBM ALL48_N39k walks 13372426 running (6), so gate D (a) 13372424 exited 0 (afterok); forest slots running: 1 (13371789); 13372432, 13372801, 13371790 and 43 tasks of 13371789 pending |
| 04:10 | Hoffman2 placement at 8 threads (RULE.txt note to addendum C): Hoffman2 starts 20-slot jobs too slowly (3 of 16 in 1 h 45 min; 8-slot in 1-3 min); evidence of thread invariance: rf_skx gate (d) 20 = 8, the benchmark 1 = 8, this run's gate 1 (forest at 8 and 20 threads text-identical to the templates); Hoffman2 gates run at 8 threads (tags gold6140, gold6240, gold6342, e5v4); Hoffman2 env repair 15261300 (setuptools 84.1.0 / packaging 26.3), gates and its check 15261029 held behind it; no Hoffman2 gate piece yet; its estimate 9-15 h a line at 8 threads (CARC 3.7 h at 20); first wave 349-368 unchanged, later waves from runs/tasks/h2_throughput.txt |

## Interim results: CLOSE, RTH13_N3k, ALL48_N3k against the reused RTH13 (05:05 PDT)

- Walks 13369755 / 13369756 (40 pieces, all EPYC 7513 / 7542), merge 13369759, score 13369760 COMPLETED; the scorer's
  checks passed (RTH13 reproduces the templates' metrics.csv and full-vs-HAR DM; R2 equal too).  CPU-hours from the
  piece metas: CLOSE 23.3, RTH13_N3k 300.9, ALL48_N3k 301.7 (total 625.9; projected 672).
- Local economics (`experiments/trees_train3way_econ.py` with the main checkout b3ca310): check passed (RTH13
  reproduces the templates' econ.csv: D6 mid / bid-ask and the full-vs-HAR bootstrap p) -> `econ_w1/`.
- Gate D (13372424): PASS -- ALL48 LightGBM both sets every-row = scored-only bit for bit; repeats identical: ALL48
  full set 2014-01-03 10:00 = 0.8102594551794627 in the scored, scored2 and 8 fresh single fits (6 nodes, EPYC 7513
  and 7542); HAR set and ALL48_N39k likewise.  Record (b) (13372425): deterministic + col-wise against the template
  setting on RTH13, 20 origins a set: text-identical, max relative difference 0 (both sets).
- Files: scores_w1/, econ_w1/, gate/ (g1, g2, gc, gcb, gd, gdb JSONs), design_check.json, timing/, tasks/ (task files,
  projections, h2_assignment.txt), pieces/ (metas of the 40 pieces), CLOSE / RTH13_N3k / ALL48_N3k forecasts.json, logs/.

## Addendum E (07:06 PDT): Xeon 4116 / E5-2640 v4 gates on CARC, a claim pool for the forest full set

RULE.txt addendum E committed in 65f789a before any forest task on a new model; code archive `code_65f789a`.
`code/submit_e.sh` (07:07) submitted the gates and the EPYC pool (held), then stopped before any cancel: the Xeon 4116
pool (-p main,oneweek, 200 tasks) was refused, QOSMaxSubmitJobPerUserLimit (oneweek's MaxSubmitPU is 50 and a
multi-partition job counts against it).  `code/submit_e_fix.sh` (9448039, 07:09) completed it: the 4116 pool on main
only, the E5-2640 v4 pool as 20 tasks, then the pending tasks of 13372432 / 13372801 cancelled (none left pending;
13372432 lines 1-4 still running at 20 threads), then the pools released.

| Job | Name | What | Depends on |
|---|---|---|---|
| 13377713 / 13377714 | t3t_ge41a / b | Xeon 4116 gates (9 lines at 8 threads / 2 at 20 threads; main,oneweek) -> gate/carc_x4116 | - |
| 13377715 | t3t_ge41c | accept --require-tag carc_x4116 | ge41a, ge41b |
| 13377716 / 13377717 | t3t_ge26a / b | Xeon E5-2640 v4 gates (oneweek) -> gate/carc_x2640v4 | - |
| 13377718 | t3t_ge26c | accept --require-tag carc_x2640v4 | ge26a, ge26b |
| 13377719 | t3t_pE | pool on EPYC 7513 / 7542, 8 threads, 16 GB, 150 tasks %20 | - |
| 13377761 | t3t_p41 | pool on Xeon 4116 (main), 8 threads, 16 GB, 150 tasks %40 | ge41c |
| 13377762 | t3t_p26 | pool on Xeon E5-2640 v4 (oneweek), 20 threads, 24 GB, 20 tasks %10 | ge26c |

Pool: lines 5-348 of runs/tasks/w3rf_full.txt, claims in runs/claims_w3rf_full/<line>/ (claim, done, failed, term).
320 pool tasks for 344 lines: more tasks are added as the pools run down.  Held final merge 13372433 now waits on
13372432, the three pools, 13372427, 13371788, 13371790.

| Time (PDT) | Lines of w3rf_full.txt | From -> to | Why |
|---|---|---|---|
| 07:09 | 5-239 (13372432, pending) and 240-348 (13372801, pending) | EPYC 20-thread arrays -> claim pool (EPYC 8 threads; Xeon 4116 / E5-2640 v4 after their gates) | addendum E: 4 forest tasks running, 3 EPYC nodes with >= 20 idle CPUs |

## Xeon gate failures: infrastructure (07:22-07:25 PDT)

- Failed gate tasks (no comparison ran in any): x4116 13377713_1 (RTH13 rf HAR gate 1; d05-06) SIGBUS (exit 135);
  x2640v4 13377716_3 / _4 (d18-17), _9 (d18-26) SIGBUS, 13377717_1 (d18-28) BrokenPipeError errno 108 (ESHUTDOWN)
  reading the design csv, 13377717_2 (d18-29) SIGBUS -- the mmap / reads of the /scratch1 design on nodes with failing
  Lustre mounts.  Completed gate tasks ran on d05-13, d05-15, d11-11, d11-17, d18-13, d18-18, d18-24.
- RULE.txt note to addendum E (6e1eb98); `code/submit_e2.sh` (07:24): designs copied to /home1 (13378109, md5-checked);
  incomplete pieces of the failed lines removed; reruns with the /home1 designs, nodes d05-06, d18-17, d18-26,
  d18-28, d18-29 excluded: 13378110 (x4116, 1 line, 8 threads), 13378111 (x2640v4, 3 lines, 8 threads), 13378112
  (x2640v4, 2 lines, 20 threads); new checks 13378113 (x4116: afterany 13377713 / 13377714, afterok 13378110) and
  13378114 (x2640v4: afterany 13377716 / 13377717, afterok 13378111 / 13378112); never-started 13377715, 13377718,
  13377761, 13377762 cancelled; Xeon pools resubmitted with DESIGNS=/home1, the nodes excluded: 13378115 (x4116, main,
  150 %40, afterok 13378113), 13378116 (x2640v4, oneweek, 20 %10, afterok 13378114); held merge 13372433 re-pointed.
- EPYC pool 13377719: 0 of 150 running at 07:24, every task pending with reason None (not yet evaluated); EPYC 7513 /
  7542 nodes with >= 8 idle CPUs: 26 (780 idle CPUs); our running 47 jobs / 616 CPUs; scheduler: backfill tests at
  most 3,000 jobs a cycle (bf_max_job_test), window 48 h, resolution 10 min.
- 07:37 (coordinator 07:35): pool time limits lowered on the pending tasks with scontrol update (queue age kept), about
  2x the hours a line: EPYC 13377719 24 h -> 16 h (est. 8.5 h a line at 8 threads), Xeon 4116 13378115 -> 20 h (est.
  ~11 h), E5-2640 v4 13378116 -> 16 h (est. ~7.5 h at 20 threads); to be revisited once the first lines are timed.  A
  line that times out keeps its claim with a 'term' marker and its checkpointed piece; it is resumed by releasing the
  claim to a pool of the same CPU family (by hand, after a look).

## Xeon 4116 pool failures (09:25-09:35 PDT): /scratch1 Lustre on more d-nodes

- 13378115 failed tasks (5 by 09:30), all /scratch1 errors on their nodes, none a computation: _7 (d11-24, line 20)
  OSError errno 5 (I/O error) reading a file; _11 (d11-23, line 23) errno 108 stat()ing the piece's .part; _21 (d06-25,
  line 32) errno 108 writing its own claim marker; _16 (d11-24, line 27); _10 (d11-24, exit 0:53) failed before
  claiming any line (its log on /scratch1 empty).  No piece of lines 20, 23, 27, 32 had been written.
- `code/fix_pool_failures.sh` (37e5bb6) diagnosed them; `code/release_failed.sh` (2a6dce2) released lines 20, 23, 27,
  32 (claims removed, logged in runs/claims_released.log; any pool may run them from scratch) and set the pending tasks'
  ExcNodeList to d05-06, d06-25, d11-23, d11-24, d18-17, d18-26, d18-28, d18-29 (121 pending tasks, verified).
- Pools at 09:30: EPYC 13377719 1 running / 149 pending; Xeon 4116 13378115 24 running / 121 pending / 5 failed; E5-2640
  v4 13378116 9 running / 11 pending / 0 failed; 37 lines claimed, 0 done.

## Throughput-balanced split with Hoffman2; Xeon 4116 pool moved to /home1 (10:15-10:26 PDT)

- Xeon 4116 pool 13378115 failures (16 by 10:15): all /scratch1 Lustre errors (errno 5 input/output error, errno 108
  transport endpoint shutdown) on d11-23, d11-24, d11-29, d11-42, d06-19, d06-25 (and the earlier d05-06, d18-*).
  `code/move_home1.sh` (a5c4171) released lines 7, 8, 9 and then stopped (set -e on an empty grep) before changing
  anything else; `code/rebalance_h2.sh` (961c9ec, 10:25) released lines 20, 23, 26, 34, 35, 37, 39, 46, 47 (no piece
  of any had been written; runs/claims_released.log) and cancelled the pool's 105 pending tasks (29 running left).
- Hoffman2 gates passed for Gold 6140, 6240, 6342 and E5-2650 v4 (Hoffman2 executor, 10:25; gate 1 max rel 1.2e-15,
  gate 2 bit for bit); measured hours a line at 8 threads: Gold 6342 5.4, Gold 6140 11.2, E5-2650 v4 11.2, Gold
  6240 11.7.  Equal-finish split of the 294 unclaimed lines (Hoffman2 ~14 lines/h at <= ~120 tasks; CARC ~4.5 lines/h
  for new lines; Hoffman2's 20-line wave 1 counted; its share rounded down to a multiple of 10):
| Lines of w3rf_full.txt | Where | How |
|---|---|---|
| 1-4 | CARC 13372432 (20 threads) | running |
| ~48 claimed lines in 5-79 | CARC: old Xeon 4116 pool 13378115 (29 running), EPYC pool 13377719, E5-2640 v4 pool 13378116 | started |
| 25 unclaimed lines below 80 | CARC /scratch1 pools (EPYC 13377719, E5-2640 v4 13378116) | claim pool |
| 80-138 (59) | CARC Xeon 4116 pool on /home1, 13381306 (64 tasks %40, 8 threads, 20 h; code, claims, pieces, logs on /home1; nodes above excluded) | reserved_home1 |
| 139-348 (210) | Hoffman2 | pre-claimed owner_h2 |
| 349-368 (20) | Hoffman2 wave 1 (15262038, Gold 6342) | assigned 02:15 |
- runs/tasks/h2_assignment.txt: ASSIGNED 349-368 and 139-348.  Held final merge 13372433 also waits on 13381306; the
  /home1 pieces are copied (md5) into runs/pieces before it is released.

## A bug in the line releases (found 11:30 PDT): partial Xeon 4116 pieces

- The releases of 09:30-10:25 (release_failed.sh, move_home1.sh, rebalance_h2.sh) tested "no piece written" with
  `ls a.parquet a.json a.ckpt.json`, which fails when any one of the three is absent: lines whose piece had a parquet
  and a checkpoint but no final json (a partial piece, begun on Xeon 4116) were taken as unwritten and released.  No
  piece file was deleted (only claim directories and stray .part files).  Fixed in the three scripts (each file tested
  on its own).
- Effect: EPYC pool tasks 13377719_10, _11, _13-_21 (b10-20, a03-11) claimed released lines 7, 8, 20, 23, 26, 34, 35,
  37, 39, 46, 47 and stopped at once by design ("rf_skx: piece ... was begun on ['Intel(R) Xeon(R) Silver 4116 ...'],
  this node is AMD EPYC 7513"): the one-CPU-family guard; nothing was computed on EPYC for them.  Those 11 lines and
  line 52 (13378115_42, d11-42, errno 5) now hold a claim with a 'failed' marker and a partial Xeon 4116 piece
  (parquet + checkpoint) on /scratch1; no pool takes them.  The old pool's other failures (/scratch1 errno 5 / 108 on
  d06-19, d06-25, d11-23, d11-24, d11-29, d11-42) left no claim.
- Pools at 11:30: EPYC 13377719 12 running / 127 pending; old Xeon 4116 13378115 28 running; E5-2640 v4 13378116 9 / 11;
  /home1 Xeon 4116 13381306 0 / 64 (not starting); 330 claim directories, 0 done; unclaimed below 80: 66-79.
- Owner decision A (relayed ~11:40 PDT): the 12 parked lines restarted from scratch.  One CARC session (CARC clock
  10:39), every command verbatim in the heredoc: for each of lines 7, 8, 20, 23, 26, 34, 35, 37, 39, 46, 47, 52 it was
  checked that no final metadata (.json) and no 'done' marker existed, that the claim had a 'failed' marker, that the
  claimant task (13377719_10, _11, _13-_21; 13378115_42) was no longer in the queue, and that neither piece file had
  changed in the last 10 minutes; all 12 passed.  Deleted per line: the partial piece's parquet (4,096-6,648 bytes)
  and checkpoint json (100-101 bytes), both written on Xeon Silver 4116 between 09:37 and 10:23 (no .part existed),
  and the claim directory (claim + failed); each entry is in runs/claims_released.log.  The lines are unclaimed again
  in the /scratch1 claim namespace, below line 80: the EPYC pool 13377719 (13 running / 126 pending) and the E5-2640
  v4 pool 13378116 (9 / 11) take them next (lowest unclaimed first), on any gated model.
- Standing rule (coordinator ~12:05 PDT, option A extended): a pool task that fails on a /scratch1 I/O error and leaves
  only a partial piece -> the line restarts from scratch after the four checks (no final json / done; claim 'failed'
  and its task gone; piece files unchanged >= 10 min; the failure an I/O error), the deletes written out in the SSH
  session, logged in runs/claims_released.log.  Line 40 (13378115_28, d11-24, errno 5 "Failed stat()ing path" /
  input/output error, 1 h 16 min): deleted ALL48_rf_skx_all_features_40of368.ckpt.json (102 bytes),
  .parquet (0 bytes) and .part (0 bytes), and its claim; back in the /scratch1 pools (CARC clock 10:57).
- Standing rule, batch 2: line 58 (13378115_45, d06-19, errno 5 input/output error): deleted
  ALL48_rf_skx_all_features_58of368.ckpt.json (101 bytes) and .parquet (6,364 bytes) and its claim; back in the
  /scratch1 pools.  Pools then: EPYC 13377719 17 running / 122 pending; old Xeon 4116 13378115 26 running (21 failed);
  E5-2640 v4 13378116 9 / 11; /home1 Xeon 4116 13381306 3 running / 61 pending (its first tasks started).
- Standing rule, batch 3: restarted line 19 (13378116_9, E5-2640 v4, d18-35, errno 108; deleted .ckpt.json 101 bytes,
  .parquet 10,869 bytes), line 50 (13378115_40, d06-19, errno 108; 101 / 7,110 bytes), line 9 (13378115_46, d11-29,
  errno 5; 147 / 7,442 bytes), and their claims.  Kept, outside the rule: line 63 (13378115_47, d11-29, failed 11:44:36
  CARC clock, the same second as line 9's task on that node, exit 1:0) -- its log shows normal progress (12 refits,
  ~368 s a refit at 8 threads) and no error line, so the cause is not shown as an I/O error; claim 'failed' and partial
  piece (.ckpt.json 101 bytes, .parquet 6,818 bytes) left in place; asked.
- Pools then: EPYC 13377719 20 running / 119 pending; old Xeon 4116 13378115 23 running (24 failed); E5-2640 v4
  13378116 9 / 10 (1 failed); /home1 Xeon 4116 13381306 4 running / 60 pending; no forest line finished yet.
- Standing rule extended (coordinator ~12:45 PDT): it also covers a pool task whose own log shows no error but which
  failed on the same node within 5 minutes of a task with a logged /scratch1 I/O error (errno 5 / 108 / transport
  endpoint), provided its log has no traceback, exception, assertion or OOM message (any of those -> ask).  Applied to
  line 63: its task 13378115_47 ended on d11-29 at 11:44:36 (CARC clock), the same second as 13378115_46 there (errno 5
  input/output error); no error line in its log; the four checks passed; deleted
  ALL48_rf_skx_all_features_63of368.ckpt.json (101 bytes), .parquet (6,818 bytes) and its claim; back in the
  /scratch1 pools.
- Standing rule, batch 4 (steward alert: 13378116 2 failed): line 37 (13378116_10, E5-2640 v4, d18-35, errno 108 on the
  piece's .part, ended 12:36 CARC clock) kept for now: its piece files changed within the last 10 minutes (check 3);
  rechecked at the next alert.  First completed lines: 13 and 17 (E5-2640 v4, 20 threads, 3.65 / 3.66 h for 98 refits
  each).  Pools: EPYC 13377719 20 running / 119 pending (11 failed); old Xeon 4116 13378115 23 running (24 failed);
  E5-2640 v4 13378116 9 / 7 (2 failed, 2 done); /home1 Xeon 4116 13381306 4 / 60; unclaimed below 80: 40, 46, 47, 50,
  52, 58, 63, 67-79.

## Maximize throughput (owner ~13:35 PDT; one CARC session, state changes verbatim)

- Standing rule: line 37 (13378116_10, errno 108) restarted (deleted .ckpt.json 101 bytes, .parquet 6,931 bytes and its
  claim).
- EPYC /scratch1 pool 13377719: throttle 20 -> 40; pending tasks' time limit 16 h -> 13 h.  /home1 Xeon 4116 pool
  13381306: pending 20 h -> 14 h.  E5-2640 v4 pool 13378116: pending 16 h -> 6 h (measured 3.65 h a line at 20
  threads) and nodes d05-06, d06-19, d06-25, d11-23, d11-24, d11-29, d11-42, d18-17, d18-26, d18-28, d18-29, d18-35
  excluded.
- New pools on the /home1 claim namespace (lines 80-138, code_961c9ec, pieces / logs / claims on /home1): EPYC 7513 /
  7542 13382812 (55 tasks %40, 8 threads, 13 h) and oneweek Xeon 4116 13382813 (12 tasks %3, 8 threads, 14 h; the
  oneweek QOS had 180 of 208 CPUs in use by the E5-2640 v4 pool).
- Held final merge 13372433: dependency afterany 13377719, 13378115, 13378116, 13381306, 13382812, 13382813, 13372427
  (13372432, 13371788, 13371790 had left the queue).
- State: EPYC 13377719 20 running / 119 pending; old Xeon 4116 13378115 23 running; E5-2640 v4 13378116 9 / 7 (2 done);
  /home1 Xeon 4116 13381306 4 / 60; EPYC /home1 13382812 0 / 55; oneweek Xeon 4116 13382813 0 / 12.  Unclaimed: 21
  lines below 80 (/scratch1 namespace), 55 of 80-138 (/home1 namespace).  Hoffman2's 230 lines (139-368): its array
  15262074 had not started (its executor diagnosing); no line taken back.

## Lines 317-368 returned by Hoffman2 (coordinator ~15:05 PDT; one CARC session, state changes verbatim)

- None of 317-368 had started on Hoffman2 (its array 15263238 tasks 1-180 deleted; h2/STATUS.md 9a89328); no piece of
  them existed on CARC.  runs/tasks/h2_assignment.txt rows now: ASSIGNED 139-316, RETURNED 317-368 (backup kept).
- Lines 317-348: their owner_h2 claim markers replaced by reserved_home1 (the /scratch1 pools keep skipping them);
  349-368 had no /scratch1 claim.  178 owner_h2 claims remain (139-316).
- New pools on the /home1 claim namespace for lines 317-368 (code_961c9ec; pieces / logs / claims on /home1): EPYC
  7513 / 7542 13383080 (56 tasks %40, 8 threads, 13 h), Xeon 4116 13383081 (main, 30 %20, 8 threads, 14 h, bad nodes
  excluded), E5-2640 v4 13383082 (oneweek, 8 %8, 20 threads, 6 h).  Held final merge 13372433 afterany 13377719,
  13378115, 13378116, 13381306, 13382812, 13382813, 13383080, 13383081, 13383082, 13372427.
- State: EPYC /scratch1 13377719 28 running / 111 pending; old Xeon 4116 13378115 23 running; E5-2640 v4 13378116 9 / 1;
  /home1 Xeon 4116 13381306 4 / 60; EPYC /home1 13382812 0 / 55; oneweek Xeon 4116 13382813 3 / 9; new 13383080 0 / 56,
  13383081 0 / 30, 13383082 0 / 8.  Lines done 8 (/scratch1 namespace); unclaimed: 7 below 80, 52 of 80-138, 52 of
  317-368.  Our normal-QOS running jobs 59; oneweek 204 of 208 CPUs.
- Steward alert (~15:30 PDT): merge 13372430 (ALL48 without the forest full set) COMPLETED: lgbm / xgb both sets and
  rf_skx HAR, 33,970 rows each (169,850 rows) -> runs/ALL48/forecasts.parquet; its score 13372431 waits on 13372429,
  which waits on the ALL48_N39k merge 13372428 (the N39k forest 13371789: 43 of 44 tasks done, 1 running).  Line 39
  (13378116_13, E5-2640 v4, d18-35, ended 13:22 CARC clock, exit 1:0; log shows normal progress, ~129 s a refit at
  20 threads, no error line) kept: no I/O error of its own; the same-node check of the extension is run at the next
  alert.  Pools: EPYC 13377719 32 / 107; old Xeon 4116 13378115 23; E5-2640 v4 13378116 8 / 1 (3 failed); /home1
  13381306 4 / 60, 13382812 0 / 55, 13382813 3 / 9, 13383080 0 / 56, 13383081 0 / 30, 13383082 0 / 8; 8 lines done.
- Standing rule, second extension (coordinator ~15:45 PDT): it also covers a task whose own log is clean (no
  traceback, exception, assertion or OOM) when it ran on a node already on our excluded bad-node list because of a
  logged /scratch1 I/O error on that node earlier the same day (e.g. d18-35: line 37's errno 108 at 12:36).  Limit:
  a line that fails a second time and qualifies only under an extension (not a logged I/O error of its own) is parked
  and asked about.  Lines restarted under an extension so far: 63 (same node / same second as an errno 5); 39 is
  approved for restart (d18-35) and is restarted at the next alert's batch, with the same four checks.
- Standing rule batch (steward alert ~15:50 PDT): line 39 restarted (coordinator-approved; d18-35; deleted .ckpt.json
  100 bytes, .parquet 6,930 bytes, claim; its earlier restarts: the 10:25 release and decision A); line 51 restarted
  (13378115_41, d11-42, own errno 108; .ckpt.json 102 bytes, .parquet 7,650 bytes, claim).  Merges / scores: m3 done;
  m2, s2, s3, m4, s4 pending.  Pools: EPYC 13377719 32 running / 107 pending; old Xeon 4116 13378115 22 (25 failed);
  E5-2640 v4 13378116 9 / 0; /home1 13381306 4 / 60, 13382812 0 / 55, 13382813 3 / 9, 13383080 0 / 56, 13383081
  0 / 30, 13383082 0 / 8.  8 lines done.  Unclaimed: 4 below 80 (/scratch1), 52 of 80-138 and 52 of 317-368 (/home1).
- Lines moved to the /scratch1 claim namespace (coordinator ~16:00 PDT, one session, changes written out): the 84
  unclaimed lines 87-138 and 317-348, each first taken in the /home1 namespace (atomic mkdir + moved_scratch1 marker)
  and then freed in /scratch1 (reserved_home1 removed); lines 80-86 (claimed in /home1) and 349-368 (/home1) untouched.
  13377719 had 107 pending tasks (>= 84 + 10): no new array.  Unclaimed now: 88 lines in the /scratch1 namespace
  (5-348; EPYC 13377719 32 running / 107 pending, E5-2640 v4 13378116 9 / 0), 20 in the /home1 namespace (349-368;
  pools 13383080-82); the /home1 pools of 80-138 (13381306, 13382812, 13382813) have no unclaimed line left (their
  pending tasks exit when they start).

## Throughput batch (coordinator ~17:30 PDT; CARC sessions with changes written out)

- Standing rule: no claim marked failed.
- Measured (completed forest full-set lines, wall a line): EPYC 7513 8 threads n=3 max 6.24 h (mean 6.09); EPYC 7513
  20 threads n=3 max 5.04 h; EPYC 7542 20 threads n=1 5.66 h; E5-2640 v4 20 threads n=16 max 3.91 h (mean 3.81); Xeon
  4116 none completed yet (~360 s a refit).  MaxRSS of completed pool tasks at most 2.2 GB (requests 16 / 24 GB).
- Pending tasks right-sized (scontrol update; the first attempt failed on the memory unit and changed nothing): EPYC
  pools 13377719 (107 pending) and 13383080 (56) -> 7:50:00 (1.25 x 6.24 h) and 5,120 MB; Xeon 4116 13383081 (30)
  -> 5,120 MB, 14 h kept (no measured line).  E5-2640 v4 pools had no pending task.
- Partitions open to us with gated models: main and oneweek (ours); largemem (EPYC 7513, 0 idle); debug (1 h limit);
  gpu (EPYC 7513 / E5-2640 v4 GPU nodes: not used for CPU-only work).  No new pool.
- Pending tasks of the empty /home1 pools 13381306 (60), 13382812 (55), 13382813 (9) cancelled; running tasks kept.
  Held merge 13372433 afterany 13377719, 13378115, 13378116, 13381306, 13382813, 13383080, 13383081, 13383082.
- Watch 349-368: 8 started (349-356, E5-2640 v4 /home1 pool 13383082), 12 unclaimed; /scratch1 namespace 88
  unclaimed.  Lines done: 19.  Our main-partition running jobs 55 (440 CPUs); oneweek 204 of 208 CPUs.  Merges m2 / m4
  and scores s2 / s3 / s4 still pending.

## Interim results 2: ALL48_N39k and ALL48 (without the forest's full set) (pulled ~17:20 PDT)

- Merges 13372428 (ALL48_N39k, every class and set) and 13372430 (ALL48 without rf_skx/all_features) and scores
  13372429 (scores_w2) / 13372431 (scores_w3, every arm merged so far) COMPLETED; both scorer checks passed (RTH13
  reproduces the templates' metrics.csv and full-vs-HAR DM).  Local economics on scores_w3 (econ_w3/): check passed.
- Files: scores_w2/, scores_w3/, econ_w3/, ALL48 / ALL48_N39k forecasts.json, piece metas, logs.

## CARC tail decision (coordinator ~21:00 PDT; CARC clock 20:44)

- Unclaimed U = 54 lines, all in the /scratch1 namespace (117-138, 317-348; 349-368 all claimed); claims in the last
  3 h: 42 -> CARC start rate r = 14.0 lines/h; 6.76 h to 03:30 -> U3 = max(0, 54 - 14.0 x 6.76) = 0 -> k = 0: no line
  reserved for Hoffman2's Slurm round 3, nothing moved, h2_assignment.txt unchanged.
- At that rate the last unclaimed line starts around 00:40; EPYC lines take ~6.2 h (done by ~07:00), a Xeon 4116 line
  ~10 h.  Pools: EPYC 13377719 40 running / 73 pending; old Xeon 4116 13378115 10 running; EPYC /home1 13383080
  10 / 28; Xeon 4116 /home1 13383081 2 / 26; E5-2640 v4 pools empty.  Lines done: 55 (/scratch1) + 10 (/home1); no
  failed claim.
- Lines 164-167 returned by Hoffman2 (Slurm 494144_32 cancelled while pending, h2/STATUS.md f287259): no piece or
  other claim on CARC; their owner_h2 claims removed in the /scratch1 namespace; only the EPYC pool 13377719 (73
  pending) can take them (the /scratch1 Xeon 4116 / E5-2640 v4 pools have no pending task).  h2_assignment.txt:
  ASSIGNED 139-163, 168-316; RETURNED 164-167, 317-368.
- Lines 139-153 returned by Hoffman2 (~23:25 PDT; UGE 15263239 deleted while held, no task started; h2/STATUS.md
  a8ec987): no piece or other claim on CARC; owner_h2 claims removed; h2_assignment.txt ASSIGNED 154-163, 168-316 /
  RETURNED 139-153, 164-167, 317-368.  Fresh /scratch1 EPYC 7513 / 7542 pool 13422082 (15 tasks, 8 threads, 7:50:00,
  5 GB, code_65f789a as 13377719): 5 running, 10 pending at submission.  Held merge 13372433 afterany 13377719,
  13378115, 13383080, 13383081, 13422082.  Note: 13377719 still had 69 pending tasks but 40 running = its throttle (40);
  only 4 lines were claimed between 20:44 and 23:25 (unclaimed 54 -> 54 before the 15 returned lines).
- 13377719 throttle 40 -> 64 (coordinator-approved): main-partition running jobs 61 (13377719 40, the others 21) plus
  13422082's 10 pending, so 64 keeps our total at <= 95 under the normal QOS's 100.  From 02:30 each alert report gives
  the unclaimed lines and the claim rate of the last 2 h (Slurm frees ~60 Gold 6240 slots at ~03:30, finish ~11:00).

## Time-limit fix (2026-10-25 01:25 CARC clock; coordinator, urgent)

- 13377719_44 (line 39, a03-18, EPYC 7513) TIMED OUT at 07:50:29 with 102 of its 104 refits in its checkpointed
  piece (274.8 s a refit): the 7:50 limit (1.25 x the 6.24 h max of 3 lines) was too short for slower lines / nodes.
  Of 77 running EPYC tasks, 26 were projected past 7:50 (300-518 s a refit; 1-82 of 85-104 refits done).
- Pending tasks: raising the limit is allowed -> 13377719 (33 pending) and 13422082 (10) set to 17:00:00 (1.3 x 518 s x
  91 refits).  Running tasks: `scontrol update JobId=13377719_68 TimeLimit=17:00:00` -> "Access/permission denied"
  (Slurm; not retried).
- A timeout loses at most 5 minutes (the walk checkpoints every 300 s), so timed-out lines are RESUMED on the same CPU
  family from their pieces, not restarted (no deletion): line 39 by job 13422638 (3 h), and for each of the 26 at-risk
  running tasks a resume job that starts when the task ends (afterany; it finds the piece complete and exits if the task
  finishes in time): 13422639-13422665 (EPYC 7513 / 7542, 8 threads, 5 GB, 3-17 h; separate claim namespaces
  runs/claims_w3rf_full_resume and /home1/.../claims_w3rf_full_resume; the original claims stay); runs/resumes.log.
- Held merge 13372433 also waits on the 28 resume jobs (33 jobs in all).  Unclaimed lines at 01:25: 28 (/scratch1);
  claims in the last 2 h: 36 (18 / h).  Lines done: 82 (/scratch1) + 15 (/home1).
- Round-3 decision (coordinator 02:47; CARC clock 02:48): U = 13 unclaimed (336-348, /scratch1 namespace); 22 claims in
  the last 2 h (11.0 lines/h); median running-EPYC s/refit 264.6 s (75 tasks) -> 6.69 h a 91-refit line; CARC's last
  claim in ~1.2 h -> tail ~10:40 (CARC clock); CARC can claim ~16 lines in the window to 11:00 -> k = 0: nothing
  reserved for Hoffman2's Slurm round 3, h2_assignment.txt unchanged.
- 09:21 (CARC clock): 13377719_116 (line 319, a02-14; started 01:18 with the old 7:50 limit, before the 01:27 scan
  covered it) TIMED OUT with 86 of 91 refits checkpointed -> resume job 13425458 (3 h).  Every other timed-out task
  had its resume job (24 in resumes.log; their pieces complete except lines 144, 167, 318, still resuming).  No running
  task with a 7:50 limit is left uncovered.  Complete forest full-set pieces: 159 on /scratch1, 27 on /home1;
  running: 13377719 17, 13422082 2, resume jobs 3 (+1 pending).
- 11:16 (CARC clock): 357 of the 368 ALL48 forest full-set pieces complete (CARC /scratch1 176, /home1 30 incl. lines
  1-4 and the resumed ones; Hoffman2 159: Gold 6240 131, Gold 6140 28; pieces_h2 MD5SUMS check clean); no line in two
  places.  Missing 11, all running on EPYC (17 h limits): 323, 324, 336, 339, 340, 341, 342, 343, 345, 346, 348; the
  slowest (345: 47 of ~91 refits at 462 s; 340 / 341: 55 at ~400 s; 342: 59 at 368 s) finish around 15:00-17:00.
  Accept and the final merge wait for them (13372433 held).
