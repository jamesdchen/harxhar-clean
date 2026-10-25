# Trees' three-way comparison on Hoffman2 (status)

Assignment (CARC `runs/tasks/h2_assignment.txt`, RULE.txt addendum C): `ASSIGNED w3rf_full.txt 240-368 ALL48 rf_skx
all_features 20` (129 lines, the ALL48 forest's full set).  Owner, 02:40 PDT (relayed): lines go to Hoffman2 in waves
sized to the slots it actually gets; only started lines are Hoffman2's; the assignment is re-read before each wave.
Scripts: `code/` (staging `/u/scratch/j/jamesdc1/t3t` on Hoffman2, user jamesdc1).

| Time (PDT) | Event |
|---|---|
| 02:04-02:07 | transfer 1, CARC -> laptop pipe -> Hoffman2 dtn2 (one SSH session each side, one tar stream): 97 files, 1,812,282,101 bytes in 192 s (9.4 MB/s): the RTH13 and ALL48 design caches (both sets), runs/tasks, timing, CARC's forest gate pieces (gate/g1, gate/g2) and their jsons; manifest `h2/xfer1/MANIFEST.md5` (md5 7371ffd310d137a3e3c9df2f5aca4bc5 on both ends), `md5sum -c` OK on Hoffman2, each design dir's own MD5SUMS OK |
| 02:14 | code = `git archive` of d9d01fd (the commit in CARC's `runs/tasks/h2_code_commit.txt`; LF, no line-ending conversion) at `$B/code`: md5-identical to CARC's `code_d9d01fd` for trees_train3way.py, pooled_trees_rerun_13.py, data/core_stats.parquet, both templates' forecasts.parquet and the three configs |
| 02:14 | `code/submit_h2.sh`: env 15261016; gates per CPU tag (forest only, the assigned class): gold6140 15261017 / 15261018 / 15261019, gold6240 15261020-22, gold6342 15261023-25, e5v4 (E5-2650 v4) 15261026-28 (`t3t_g20L_<tag>` ALL48 full set every row 20 threads, `t3t_g20_<tag>` RTH13 --ends 10 at 20 threads + ALL48 scored-only + HAR every row, `t3t_g8_<tag>` RTH13 --ends 10 at 8 threads + ALL48_N3k every row / scored); check `t3t_gsum` 15261029 (accept + `gate_summary.py`) |

Environment (CARC, `env_carc.txt` of transfer 1): harxhar-r13 = python 3.12.15, numpy 1.26.4 (PyPI wheel, openblas64
0.3.23.dev), pandas 2.3.3, scipy 1.17.1, scikit-learn 1.9.1, lightgbm 4.6.0, xgboost 3.2.0, pyarrow 19.0.1, numba
0.68.0; venv_skx adds daal 2026.2.0, scikit-learn-intelex 2026.2.0, tbb 2023.2.0, tcmlib 1.6.0.  Hoffman2: the same
pins (`req_gb.txt`, `req_skx.txt` derived from CARC's venv_skx listing) in uv venvs (CPython 3.12.3) inside the Debian
bookworm container of the tree learning curve; threads = slots; numpy AVX-512 dispatch off; OpenBLAS Haswell.

## 04:04-04:08 PDT: why the gates did not run, and the fix (coordinator's check-in 04:01)

- Scheduler: `schedd_job_info false` (no per-job reason); the all-user RQS `single` (pes single/shared, pod/idre
  queues, 64 slots) is disabled and `qquota -u jamesdc1` shows no limit of ours.  Free slots at 04:04 (NCPU minus
  used/reserved, usable queue instances): hosts with >= 20 free: Gold 6140 45, Gold 6240 9, Gold 6342 3, E5-2650 v4 8;
  with >= 8 free: 64, 21, 8, 34.  The 8-slot tasks resubmitted at 04:04 started within 1-3 min; 20-slot tasks waited
  1 h 45 min (3 of 16 started).
- Second finding: the env build (15261016) had failed its sklearnex smoke test -- daal4py imports packaging, else
  distutils, neither in the uv venvs (CARC's harxhar-r13 has setuptools 84.1.0 and packaging 26.3) -- so no ENV_OK, and
  every gate task that started exited "no env" in seconds: no gate piece exists yet.
- fix1 (`code/fix1_8threads.sh`): every pending 20-slot gate task deleted (own tasks only) and its line resubmitted
  unchanged at 8 threads (the forest's thread-invariance, CARC's rf_skx gate (d)).  fix2-4 (`code/fix2_env.sh`,
  `fix3_env.sh`, `fix4_env.sh`): setuptools 84.1.0 and packaging 26.3 pinned (`req_gb.txt`, `req_skx.txt`),
  `t3t_envfix` 15261300 (running from 04:07:54) re-checks the env and writes ENV_OK; every line whose task exited
  "no env" resubmitted at 8 threads; every queued gate job held on 15261300; `t3t_gsum` 15261029 held on 15261300 and
  every gate job.  (fix3 parsed two ids for the env job, so its qsubs failed and only its holds took; fix4 submitted
  its four lines.)
- Gate jobs now (all 8 threads, 12 lines a tag): gold6140 15261301, 15261302, 15261306, 15261293 (task 2);
  gold6240 15261294, 15261295, 15261303; gold6342 15261296, 15261297, 15261304, 15261025 (tasks 3-6); e5v4 15261307,
  15261299 (tasks 2-5), 15261308, 15261305.  The walks will run at 8 threads (the gated count; recorded in each meta).
- Assignment relayed (CARC, read 04:05): RETURNED w3rf_full.txt 240-348, ASSIGNED 349-368 (20 lines).

## 10:17-10:20 PDT: gates passed; delivered; wave 1

- Gates (t3t_gsum 15261029 at 09:38; `accept` on CARC's code): **all four models pass** -- gate 1 max relative
  difference to the templates <= 1.2e-15, gate 2 bit for bit; Gold 6140 = Gold 6240 bit for bit; the other models
  and CARC's EPYC differ from each other by <= 1.2e-15 relative.  Seconds a refit (ALL48 full set, 8 threads): Gold
  6342 212.1, Gold 6140 443.2, E5-2650 v4 444.1, Gold 6240 463.3 (CARC EPYC 151.5 at 20 threads); `h2_throughput.txt`.
  Gold 6342 gate piece: 8,174 s wall, maxvmem 5.5 GB.
- Delivered (Hoffman2 deliver.sh gates -> laptop -> CARC carc_recv.sh, one session each side): 114 files, manifest
  md5 e096304bd053ceabb7982907fa651011, `md5sum -c` OK on CARC, every pieces dir's MD5SUMS OK; /scratch1 write check
  OK, so /scratch1/jc_905/trees3way/gate/h2/<tag>/{g1,g2}/pieces, h2/delivery_20261024_101930/ and
  runs/tasks/h2_throughput.txt.  (A first attempt at 10:17 opened the CARC session without its command -- a local
  command file had not been written -- so CARC's login shell read the tar stream as commands: all 64 lines it read
  failed as "command not found" / "No such file" and the shell exited at a syntax error; no file in ~ changed, checked
  at 10:19.)
- Live assignment re-read 10:19 (mtime 02:13:47): ASSIGNED 349-368, RETURNED 240-348.  Wave 1: job 15262038
  (t3t_w1), 20 lines 368..349, 8 threads, Gold 6342 only (fastest; 37 startable 8-slot tasks at 10:17), -tc 20,
  h_rt 12 h, h_data 2G a slot.

## 10:35-10:40 PDT: waves for lines 139-348 (owner release)

- CARC live assignment (mtime 10:25:18): ASSIGNED 349-368 and 139-348 (8 threads; 139-348 pre-claimed owner_h2); no
  CARC piece or checkpoint of lines 139-368; w3rf_full.txt md5 5222fbdd00a97093c8d3f13994a6ee85 on both clusters.
- Every qsub / qalter verbatim in the session heredoc (standing rule of 10:45).  First attempt (10:36) aborted on a
  wrong overlap check of mine before any change; the task files it wrote (new files) were kept.
- 10:37: wave 1 15262038 h_rt 12 h -> 8 h (qalter; 5.4 h a line).  Submitted, 8 threads, h_data 2G a slot:
  Gold 6342 15262055 (lines 289-348, -tc 15, 8 h) and 15262056 (239-288, -tc 20, 8 h, held on wave 1);
  Gold 6140 15262057 (174-238, -tc 50, 16 h); E5-2650 v4 15262058 (154-173, -tc 20, 16 h); Gold 6240 15262059
  (139-153, -tc 15, 17 h).
- After 150 s: Gold 6140 25 running (pod_smp.q, 20 nodes), 40 queued; no task of Gold 6342 (wave 1 and 15262055),
  E5-2650 v4 or Gold 6240 had started (wave 1 queued since 10:19).  The gates of those three models had each run on
  one or two nodes only (6342 n1077, 6240 n1849, E5-2650 v4 n7368 / n7005); much of their capacity is in owner pods
  closed to us (e.g. bern_pod_6240.q / 6342.q).  RQS h_rt limits seen: andrewst 12 h, econdjs 8 h, sch 4 h.

## 10:42-10:45 PDT (Hoffman2 clock): unstarted tasks moved to Gold 6140 (coordinator's go, ~11:50 PDT)

- All six wave arrays put on user hold (qhold), then running tasks listed (`qstat -s r -g d`): only Gold 6140
  15262057 tasks 1-25 (lines 174-198) had started; nothing finished.  First attempt (10:42) aborted before any qdel:
  `qstat -g d` did not list the held pending tasks, so its coverage check failed and the holds were released.  Second
  attempt (10:43): pending = every task of each array's task file that is not running (no task had finished), coverage
  checked = exactly lines 139-368, each once; no piece file for any moved line.
- qdel of the 205 pending tasks (15262038: 20, 15262055: 60, 15262056: 50, 15262057: 40, 15262058: 20, 15262059: 15),
  every qdel verbatim in the heredoc and each returned success; 15262057's 25 running tasks untouched (25 running after).
- Gold 6140, 8 threads, h_rt 16 h, h_data 2G a slot, fresh pieces only:
  15262074 (t3t_w3a_6140) 190 lines = 199-368 and 154-173, -tc 95;
  15262075 (t3t_w3b_6140) 15 lines = 139-153, -tc 25, held on 15262057 (starts as its 25 finish).
  Total concurrent <= 120 (25 + 95, then 95 + 25).
- Projected (11.2 h a line on Gold 6140): 15262057 ~21:50 today; 15262074 round 1 ~22:00, round 2 ~09:15 Oct 25;
  15262075 ~09:05 Oct 25 -> all 230 Hoffman2 lines ~09:00-10:00 PDT Oct 25, if the tasks start promptly.

## 12:39 PDT (Hoffman2 clock): why 15262074 does not start -- a per-user slot cap

- `qquota -u jamesdc1`: `smp/1 slots=200/200 users jamesdc1 pes !dc* queues campus*.q,pod*.q,msa*.q` (and h_data
  400G/2400G): every shared-PE job of ours on the campus / pod / msa queues counts against 200 slots, and 15262057's
  25 running 8-slot tasks use all 200.  No hold, no dependency, request identical to 15262057's (arch=intel-gold-6140,
  h_data=2G, h_rt=16h, pe shared 8, project campus_prj, -tc 95); Gold 6140 had 73 hosts with >= 8 free slots
  (216 eight-slot tasks) at 12:39; scheduler info is off; maxujobs 0, max_u_jobs 500.  Not a request problem of ours:
  nothing changed.
- Realistic Hoffman2 concurrency: 200 cores = 25 tasks at 8 threads (11.2 h a line on Gold 6140) = about 2.2 lines/h.
- The five deleted arrays are gone (`qstat -j`: do not exist).
- Task -> line: 15262074 task k = line 369 - k for k <= 170 (368..199), line 344 - k for k 171-190 (173..154);
  15262075 task k = line 154 - k (153..139).  Queued tasks start by task number as slots free (from ~21:50).

## 12:40 PDT: the new Hoffman2 Slurm cluster (hoffman2.oarc.ucla.edu; read only, two sessions)

- Same /u/scratch (the t3t staging, env and designs are visible there); apptainer/1.3.4 module.  Association
  pi_mcucurin with QOS campus24 / pi_shared24 / highp.  PreemptMode OFF; sched/backfill.
- campus partition (QOS campus24: per user cpu=1024, node=16, 100 jobs, 500 submitted, 1 day): idle 8-CPU slots now
  Gold 6240 173, Xeon 6736P 97, Gold 6442Y 84; 63 jobs (1,577 CPUs) pending in it.
- pi_shared partition (default; QOS pi_shared24: no per-user CPU limit, 1 day; PI nodes, no preemption; nothing
  pending): idle 8-CPU slots now Gold 6342 277 (50 nodes), Gold 6240 118, Gold 6140 105, E5-2650 v4 74, Xeon 6736P
  160 (ungated), E5-2697A v4 14.
- Gold 6342, 6240, 6140 and E5-2650 v4 are gate-passed models (if the Slurm nodes report the same CPU model strings,
  which walk.sh checks at start); Xeon 6736P and Gold 6442Y are not gated.
- UGE (the old scheduler): the per-user cap smp/1 = 200 slots binds (25 eight-thread tasks), so raising -tc or
  shortening h_rt there does not help; nothing changed on UGE.

## 12:49-12:52 PDT (Hoffman2 clock): option 1 (Slurm pi_shared Gold 6342) refused by Slurm; lines back on UGE

- Prepared on /u/scratch (each file written verbatim in a session heredoc): jobs/walk_slurm.sh (md5 5c791306...),
  jobs/line_slurm.sh, jobs/cmp_slurm.sh / .py (the bit-identity check: 5 UGE Gold 6342 gate lines rerun on Slurm and
  compared bit for bit), lines/chk_gold6342.txt, h2/passed_cpus_slurm.txt (Gold 6342 only),
  runs/tasks/h2_slurm1_gold6342.txt (205 lines).  qdel 15262074 15262075 (205 pending, none running).
- sbatch (pi_shared, --qos=pi_shared24) refused all three jobs: "pi_shared is for groups whose nodes are in SLURM;
  account pi_mcucurin has no pi_* partition yet" -- nothing ran on Slurm (runs_slurm/pieces empty).  Only the campus
  partition (QOS campus24: 1,024 CPUs, 16 nodes, 100 jobs a user, 1 day) is open to us; it has no Gold 6342 nodes
  (Gold 6240, Xeon 6736P, Gold 6442Y, E5-2670).
- 12:52: the 205 lines requeued on UGE Gold 6140 exactly as before: 15263238 (190 lines, -tc 95) and 15263239
  (15 lines, -tc 25, held on 15262057); runs/tasks/h2_placement.txt updated.
- Slurm GPUs open to us (campus): g15 A100 x4 (drained), g33 RTX PRO 6000 Blackwell x2 (1 in use), g35 RTX PRO 6000
  Blackwell x7 (7 in use); campus24 caps 2 GPUs a user, 1 day; highp only for pi_* accounts.

## 12:54-12:58 PDT (Hoffman2 clock): rebalance (coordinator 14:45 PDT) -- Slurm campus Gold 6240, UGE 25, 52 lines back to CARC

- UGE: qhold 15263238, no task running, `qdel 15263238 -t 1-180` (lines 368..199 and 173..164, none started),
  qrls.  UGE keeps one round of 25 after 15262057: 15263238 tasks 181-190 (lines 163-154) and 15263239 (lines
  153-139, held on 15262057).  15262057's 25 running tasks (lines 174-198) untouched.
- RETURNED to CARC (never started on Hoffman2): lines 317-368 (52).  Hoffman2 copies: runs/tasks/h2_assignment.txt
  now RETURNED 317-368 / ASSIGNED 139-316; runs/tasks/h2_placement.txt.
- Slurm campus (QOS campus24), Gold 6240 only (h2/passed_cpus_slurm.txt; condition (b) amended to 6240):
  check 494142 (the 5 UGE Gold 6240 gate lines rerun at 8 threads -> gate_slurm/gold6240), compare 494143
  (afterany; jobs/cmp_slurm.py bit for bit against gate/h2/gold6240 + gate2 on ALL48_N3k), production 494144
  (runs/tasks/h2_slurm1_gold6240.txt = 128 lines 316-199 and 173-164; 32 jobs x 4 lines, 32 CPUs, 48G, 15.5 h, %16;
  the campus QOS counts every job as a node, 16 a user, so each job runs 4 eight-thread lines; 173 idle 8-CPU Gold 6240
  slots at submission, 64 used at most) -> runs_slurm/pieces, not delivered before the check passes.
- 12:58: 14 production jobs (56 lines) and 3 check jobs running; every log shows "Intel(R) Xeon(R) Gold 6240 CPU @
  2.60GHz"; no refusal or error.

## 14:24 PDT (Hoffman2 clock): Slurm Gold 6240 bit-identity check PASSED

- 494142 (5 lines) and 494143 COMPLETED: all 5 Slurm pieces identical bit for bit to the UGE Gold 6240 gate pieces
  (dates, pred_adj, true_adj, B) and their parquet files byte-identical -- RTH13 full set --ends 10 (20 rows), ALL48
  full set scored (3), ALL48_N3k full set every (91) and scored (17), ALL48 HAR scored (3); one CPU model (Intel(R)
  Xeon(R) Gold 6240 CPU @ 2.60GHz, n1803 / n1804 / n1807); gate 2 on the Slurm ALL48_N3k every / scored pair PASS.
  gate_slurm/gold6240/check.json.  Slurm Gold 6240 pieces may be delivered to CARC.
- Production 494144: 16 jobs (64 lines) running, 16 jobs (64 lines) pending (array limit); no error in any log;
  64 checkpoints, 292.9 s a refit on average (271-522) with 4 walks a node = about 7.4 h a 91-refit line (UGE Gold
  6240 gate piece: 463 s).  No piece finished yet.  Projected: round 1 ~20:30, round 2 ~04:30 Sunday (Hoffman2 clock).

## 20:41-20:42 PDT: first delivery; Slurm round 2 running

- Slurm 494144: 15 jobs COMPLETED (round 1), 16 running (round 2 from ~20:10-20:40), 1 pending (job 32, 4 lines;
  starts when a slot frees ~03:30); 61 pieces finished, 282.1 s a refit (272-302) = about 7.1 h a line; no error.
  163 idle 8-CPU Gold 6240 slots on campus at 20:41.
- UGE: 15262057 23 running (2 finished), 15263238 2 running + 1 queued, 15263239 held.
- Delivered 20:42 (Hoffman2 inline tar -> laptop -> CARC inline receive; every mv and MD5SUMS append written out in
  the command): 63 pieces = lines 175, 188 (UGE Gold 6140) and 253-312, 314 (Slurm Gold 6240); manifest md5
  44d9a40c60d67c934d1750966196de14, 127 files md5-verified in CARC staging, /scratch1 write check ok, moved into
  /scratch1/jc_905/trees3way/runs/pieces_h2 (MD5SUMS: 63 pieces, md5sum -c ok); the Slurm check record ->
  h2/slurm_gold6240_check.json.

## 20:46 PDT: lines 164-167 returned to CARC

- Slurm 494144_32 (task-file lines 125-128 = w3rf_full.txt lines 167, 166, 165, 164) was PENDING, never started (no
  log, no piece or checkpoint on either cluster): `scancel 494144_32`; the 16 running jobs untouched (16 running
  after); UGE untouched.  Hoffman2 copies: runs/tasks/h2_assignment.txt RETURNED 317-368 and 164-167, ASSIGNED
  168-316 and 139-163; runs/tasks/h2_placement.txt.  Slurm Gold 6240 now 124 lines (316-199, 173-168).

## 21:22 PDT: second delivery

- UGE: 15262057 15 running (10 finished), 15263238 10 running (lines 163-154), 15263239 held.  Slurm 494144: 16
  running, 15 completed, 1 cancelled (job 32); 62 pieces finished; no error.
- Delivered: 9 pieces = lines 178, 180, 182, 183, 192, 194, 196, 198 (UGE) and 315 (Slurm); manifest md5
  ff0f9539f87f2e088d50c745c3c60789, 18 files md5-verified on CARC, /scratch1 ok; runs/pieces_h2 MD5SUMS ok, 72 pieces.

## 21:56 PDT: third delivery

- UGE: 15262057 6 running (19 finished), 15263238 10 running, 15263239 held; Slurm (shared FS): 62 done, 62
  checkpointed; no error in any UGE or Slurm log.
- Delivered: 9 pieces = lines 177, 179, 181, 184, 185, 189, 190, 191, 195 (UGE); manifest md5
  f5d52332f9858964b7a6ded9582ddc5d, 18 files md5-verified on CARC, /scratch1 ok; runs/pieces_h2 MD5SUMS ok, 81 pieces.

## 22:36 PDT: fourth delivery

- UGE: 15262057 3 running (22 finished), 15263238 10 running, 15263239 held; Slurm (shared FS): 64 done (round 1
  complete: lines 253-316), 60 checkpointed (round 2); no error in any log.
- Delivered: 5 pieces = lines 176, 193, 197 (UGE) and 313, 316 (Slurm); manifest md5
  0044599ab064742aa875cc0eab666c3b, 10 files md5-verified on CARC, /scratch1 ok; runs/pieces_h2 MD5SUMS ok, 86 pieces.
