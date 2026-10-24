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
