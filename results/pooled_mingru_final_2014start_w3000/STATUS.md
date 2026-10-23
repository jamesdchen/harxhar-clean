# Final pooled ridge-init MinGRU (W 3,000, first day 2014-01-02, fold 3): status log

Rule: `RULE_gru.txt` (written 2026-10-22 21:45 PDT, before any tuning job). Runner
`experiments/pooled_mingru_final_2014_w3000.py`; gate `experiments/pooled_mingru_final_gate.py`; GPU job check
`experiments/pooled_mingru_final_post_check.py`; Slurm `cluster/pooled_mingru_final_gpu_carc.sbatch`,
`cluster/pooled_mingru_final_cpu_carc.sbatch`, `cluster/pooled_mingru_final_submit_carc.sh`; scorer
`experiments/pooled_mingru_final_score.py`. CARC root `/home1/jc_905/pmf` (NFS, as the previous run after its Lustre
errors); binary copied from `/home1/jc_905/pm13/bin` (md5 44ab6fff).

## Log
- 2026-10-22 21:38 PDT: gate jobs submitted: 13319779 (tune, the previous frozen configurations on fold 3, 3 seeds a
  set) and 13319780 (refit, the first walk origin row 44,250, 3 seeds a set), both on the previous design's data
  (`/home1/jc_905/pm13/data`), P100, CONC 6, POST = the gate check.
- 2026-10-22 21:43 PDT: CPU prep 13319882 (b04-01): export of the W = 3,000 2014-start design caches (84,747 x 12 and
  x 564, HAR block 12 a set; dates of rows >= 39,000 equal the previous design's, y equal to 8.9e-16); plan: fold 3
  fit [8,202, 47,202), validation [47,527, 50,777) = 2013-01-02 15:30 .. 2013-12-31 16:00; 2,643 daily origins
  2014-01-02 .. 2024-04-30 (13 bars on 2,579 sessions, 7 on 61, 8 on 1, 4 on 2); fold-3 ridge inits for every ladder
  L x wd_lin (keep 293 of 564 columns on the full set, 12 of 12 on HAR); round 0: 1,296 configurations x 5 seeds =
  6,480 tune jobs a set in 19 files a set; canary jobs.
- 2026-10-22 21:49 PDT: canary on the new design, 13319943 (tune: one round-0 configuration a set, 1 seed) and
  13319944 (refit at the first and the last daily origin, 2 epochs), POST = check-run. Round-0 tune jobs
  13319945-13319982 (38 files) submitted held afterok on the two gate jobs and the two canaries: no tune job can start
  before the gate and the canary pass. All P100s of the gpu partition busy at submission (1,143 pending GPU jobs).
- 2026-10-22 21:57 PDT: owner (relayed): "Test mingru 3.5k as well". RULE_gru.txt addendum (same rule at W = 3,000 and
  W = 3,500, tuned separately); the runner takes the window from PMF_WINDOW_DAYS (fold 3 fit = [47,202 - W, 47,202));
  W = 3,500 in /home1/jc_905/pmf35 and results/pooled_mingru_final_2014start_w3500/ (its own STATUS.md). The W = 3,000
  jobs keep priority (W = 3,500 GPU jobs submitted with a Slurm nice value).
- 2026-10-22 22:45 PDT: owner (relayed): "All models need to refit every period. Don't cut corners." RULE_gru.txt
  ADDENDUM 2 (both windows, before any walk job): a refit at every bar from 2014-01-02 10:00 (33,970 bars an arm),
  5 seeds, from scratch (as every tuning job and every previous refit: weights_init + the window's ridge init, no
  init_from). Runner: bar-origin blocks of 1,000 (34 GPU job files an arm), per-block ridge-init subdirectories
  (INITSUB staging); timing probe cluster/pooled_mingru_final_probe_carc.sbatch (+ experiments/
  pooled_mingru_final_probe.py) at concurrency C, 2C, 4C with a bit-identity check. Projection sent to the
  coordinator (from the previous run's P100 walk logs scaled by window rows): about 175 P100-hours for the four bar
  walks at 10 epochs (W 3,000 full ~46, HAR ~34; W 3,500 full ~54, HAR ~39), more if the tuned H or L is large.
- 2026-10-23 00:11-00:18 PDT: gate 13319779 (tune, d23-14) and 13319780 (refit, e21-09): the trainer finished (pool
  phase 6.1 s for the 6 tune jobs) and the outputs were copied back and md5-checked, but the POST python (the
  /scratch1 env) died with a bus error on both GPU nodes, so both jobs show FAILED. The same gate script was rerun on
  CPU nodes (13322463 a03-13, 13322495 a01-12): GATE PASS for both. Tune: the previous frozen configurations on
  fold 3 (3 seeds a set, 9,750 validation forecasts a set) are text-identical row by row to the stored runs
  (13303258, 13303117), every best epoch equal (HAR 11/10/10, full 12/10/11), seed-mean validation QLIKE 0.12224961394006083
  (HAR) and 0.10838531784708406 (full) = the stored pretune_qlike_folds.csv values (relative 2e-16, 7e-16: the
  CSV text). Refit: the first walk origin (row 44,250, 3 seeds a set, 1,536 forecast rows a set) text-identical to
  the stored walk runs (13303665, 13303659).
- Canaries: 13319943 (tune, e21-07) and 13320230 (W 3,500 tune) COMPLETED; 13319944 / 13320231 (refits) FAILED only
  on a check-run bug (it tested positivity on the fc_lin rows, the linear shortcut alone, which can be negative);
  fixed (fc / val rows only) and rerun on CPU (13322495): all four canaries pass (status ok, trainer kept columns =
  the ridge init's: 293 at the first and 345 at the last origin on the full set at W 3,000; rows exactly [o, next)).
- 2026-10-23 00:23 PDT: the 76 held round-0 tune jobs (W 3,000 13319945-82, W 3,500 13320232-69) released
  (dependency cleared after the CPU checks); every pending job also excludes d23-14 (a Lustre bus error there today
  and on 2026-10-22).
- 2026-10-23 00:40-01:45 PDT, the bus error (owner: "just fix the python error"). Reproduced on e21-09 with the same
  check, CPU only (13322650): `df /scratch1` times out on that node; md5sum of the env's .so files fails with
  "Input/output error" / "Cannot send after transport endpoint shutdown" (617 failing lines of 428 files' checks); the
  check dies with exit 135 (SIGBUS) after 325 s. faulthandler: inside `import pandas` (gate script line 21). gdb:
  SIGBUS in the dynamic loader (_dl_lookup_symbol_x -> _dl_relocate_object, resolving OpenBLAS's srot_k_CORE2 while
  dlopen-ing numpy's OpenBLAS) = a page of a memory-mapped .so on /scratch1 that the node's broken Lustre client
  could not read. Cause: the /scratch1 client of those GPU nodes (the same "transport endpoint shutdown" as the
  2026-10-22 failures), not the env (no file of the env modified since 2026-10-20; on healthy nodes every md5 matches).
  Fix: the env copied to /home1/jc_905/envs/harxhar-r13 (rsync from a healthy node; all 428 binaries md5-equal to
  the originals, same file list), the GPU, probe and CPU Slurm scripts run Python from it (PY default), the CPU
  stages import window_mask from /home1/jc_905/pmf/repo (md5 e597d28a, the copy of harxhar-trees13's), and OpenSSL
  is pointed at the copy (OPENSSL_CONF: the env's libcrypto has the /scratch1 prefix compiled in, which cost a 49-s
  stall in `import _hashlib` on the broken node). No GPU job reads /scratch1 any more. Verified ON e21-09 with its
  /scratch1 still broken (13323199): md5 check 0 mismatches, both gate checks GATE PASS, gdb 0 SIGBUS. The round-0
  tune jobs were never exposed (they run no Python on the GPU node and stage only from /home1). The d23-14 exclusion
  was dropped from all 76 pending jobs. Diagnosis script: cluster/pooled_mingru_final_diag_buserror.sh (outputs in
  /home1/jc_905/pmf/diag/).
- 2026-10-23 01:50-02:50 PDT (coordinator: speed up). GPU types for pollok_1603 (QOS normal, 36 GPUs a user): P100 74
  (3 free), V100 52 (2), A40 50 (2), A100 36 (4), L40S 19 (6) in the gpu partition; debug partition: e23-02 (2 P100,
  idle), b11-09 (2 A40). Nothing of ours starts in the gpu partition: fairshare factor 159 of 4,000 (priority ~3,200
  vs ~10,000 for the jobs ahead), and backfill reserves for the top 100 pending jobs a partition
  (bf_job_part_count_reserve=100) while ~520 higher-priority any-GPU jobs wait. Gate jobs on V100 / A40 / A100 /
  L40S (cluster/pooled_mingru_final_gate_gputype_carc.sbatch: the P100 gate's jobs + a self-consistency rerun)
  submitted 13324299-302, pending. Slurm scripts take GPUTYPE + --constraint (ADDENDUM 3 if a window moves).
- Debug P100s: tune_baseline_r0.6 (13324590, 12 min: pool phase 675 s, 720 jobs, 23.6 epochs a job) and
  tune_all_features_r0.6 (13324591, 24 min: 1,372 s, 26.9 epochs) COMPLETED, check-run ok; 0.040 / 0.071 P100-s a
  job-epoch at W 3,000. Feeder cluster/pooled_mingru_final_debug_feed.sh (every 12 min from the laptop) moves the
  remaining pending shards to debug (1 h limit), cancelling each pending gpu-partition twin (a shard runs once);
  the six heavy full-set shards (H 256 x L 20 / 48 by lr) split in halves of whole configurations
  (tune_all_features_r0.<k>a / b). W 3,000 first, then W 3,500.
- CPU (coordinator): no CPU path implements this trainer (CUDA-only native_g9, cuRAND initial weights; the PyTorch
  port experiments/puffer_forecast/torch_train.py lacks QLIKE / early stopping, the ridge-initialised shortcut,
  softplus, wd_enc / wd_lin, dec_zero, the window mask). Timing proxy (experiments/pooled_mingru_final_cpu_probe.py,
  13324616, EPYC 7513, torch 2.6 CPU, L 20 H 64, 39,000 rows): 5-6 s an epoch at 8-16 threads, 11-13 s at 32
  (~48 core-s an epoch vs 0.071 P100-s): not pursued.
- 2026-10-23 06:00-07:36 PDT: the gpu partition started our jobs (16 W 3,000 shards there, 2 W 3,500), the debug
  P100s ran the rest; W 3,000 round 0 complete at 07:36 (44 run directories = 32 shards + 12 halves; every
  check-run ok). Allocated tune P100-hours so far 18.37.
- 2026-10-23 07:38 PDT: select round 0 (13328049, a01-09): full set FROZEN L20_H64_lr0.003_we3_wl3_ll1e-05 (val
  QLIKE 0.113381; interior on every axis; = the previous run's full pick; epochs 33 / 19 / 25 / 57 / 41). HAR set:
  L48_H256_lr0.003_we3_wl30_ll0.0001 (0.122699; epochs 7 / 17 / 6 / 3 / 12) at the top of L, H, wd_lin and lr_lin
  -> round 1 adds L 115, H 1,024, wd_lin 300, lr_lin 1e-3 (EXTEND): 2,544 new configurations, 12,720 jobs, 405
  files (cost-aware sharding, ~45 P100-min a file), estimated 212 P100-hours (H 1,024 groups 186, L 115 at H <= 256
  21, the new wd_lin / lr_lin values on the round-0 H x L 5). HAR surface flat (best a H x L cell 0.12270 ..
  0.12314; 5 configurations within 0.1 % of the best). The 14 files with H <= 256 and L <= 48 submitted
  (13328108-21, also in the debug feeder's queue); the L 115 and H 1,024 files held pending the owner's decision
  (cost and bar-walk feasibility, message to the coordinator 07:50).
- 2026-10-23 07:45 PDT: select re-run (13328140) to write out/frozen_partial.json (the full set, so its walk can
  start while the HAR set extends; select now skips a set open in another round); full-set bar-walk ridge inits
  13328147 (8-task array) -> walk jobs 13328148.
- 2026-10-23 08:05-08:20 PDT: owner: "I want max tuning" -> RULE_gru.txt ADDENDUM 3 (08:09; the per-window compute
  cap removed, ladder caps unchanged; written before any L 115 / H 1,024 job ran). The held W 3,000 HAR round-1
  files released: 52 submitted at once before the gpu partition's per-user submit limit (QOS gpu: 100 submitted,
  36 running; it rejected tune_baseline_r1.65) stopped the loop; the other 340 go through
  cluster/pooled_mingru_final_gpu_feed.sh (keeps <= 96 of ours in the gpu partition), the debug-eligible ones
  (H <= 256 or L <= 20; the H 1,024 x L >= 48 files wait for a measured cost) also through the debug feeder.
- GPU-type gates: A40 (13324300, a03-15) FAILED in 8 s: the trainer binary segfaults on the A40 (sm_86) at once.
  V100 / A100 / L40S gates still pending.
- 2026-10-23 08:09 PDT: walk jobs for the W 3,000 full set written (13328148: 169,850 refits, 34 blocks; inits for
  all 33,970 bar origins, 13328147); timing probe 13328546 (debug P100: 24 origins x 5 seeds at CONC 6 / 12 / 24).
- 2026-10-23 08:20-08:25 PDT: paused by the owner (local feeder loop stopped; nothing cancelled); resumed 08:27.
- Probe 13328546 (debug P100, W 3,000 full set, 24 bar origins x 5 seeds, 35 epochs): CONC 6 292.8 s (2.44 P100-s a
  refit), CONC 12 244.4 s (2.04), CONC 24 244.7 s (2.04); forecasts text-identical across 6 / 12 / 24 -> the walk runs
  at CONC 24 (the highest identical). Projected full-set bar walk 169,850 x 2.04 s = 96 P100-hours. The 34 blocks are
  cut into 68 halves (walk_all_features.b<k>a / b, 500 bar origins each, INITSUB bar_b<k>), at the head of the gpu
  submit queue (gpu_queue2.txt), time limit 1:56.
- Throughput (coordinator): experiments/pooled_mingru_final_tlim.py sets every file's time limit to its expected
  runtime (jobs x the median measured P100-s a job of its (set, H, L); unmeasured cells: the cost model x 1.6) x 1.3
  + 5 min; 86 pending jobs lowered by scontrol (none cancelled); the submit feeder reads the same map. V100 gate
  (13324299, d13-08) FAILED like the A40: the binary segfaults at start. A100 / L40S gates pending (limits 20 min).
  Tune concurrency test 13329097 (debug): round-0 shards rerun at higher CONC, compared with the stored runs.
- 2026-10-23 09:00-11:50 PDT: tune packing tests (cluster/pooled_mingru_final_conc_test_carc.sbatch, debug P100,
  round-0 shard r0.0 H 16 L 8, forecasts compared with the stored run): full CONC 12 450 s / 24 340 s vs 707 s at 6;
  HAR CONC 24 237 s / 32 241 s vs 351 s at 12; all text-identical; HAR CONC 48 out of GPU memory. Approved (coord.):
  CONC 24 for unsubmitted files of shapes that pass; heavy shapes tested next (13335886). V100 / A40 segfault
  narrowed (read-only, binary unchanged): a single refit job runs cleanly on both (gdb, compute-sanitizer 0 errors;
  13329189, 13329190); the gate's 6 tune jobs at CONC 6 with CUDA_DEVICE_MAX_CONNECTIONS 32 / 8 under gdb pending
  (13331862, 13331863).
- 2026-10-23 11:06-11:30 PDT: H 1,024 round-1 files ran out of time: r1.43 / r1.44 (H 1,024 L 8, 80 jobs) TIMEOUT
  at the 1:17 limit set from the cost model (H 1,024 = 10 x H 256). Node-local progress (ssh to the job's node,
  read-only): GPU memory admits only 7 H 1,024 models a P100, ~95-145 P100-s a job at L 8, ~200 at L 20 (model 33 /
  58). Revised W 3,000 HAR round-1 cost of the H 1,024 groups ~650 P100-hours (model 186). The 336 unsubmitted
  H 1,024 files are held (gpu_queue_h1024_held.txt) pending the owner; the 24 running H 1,024 jobs are salvaged:
  their completed jobs copied from the nodes to /home1/jc_905/pmf/salvage/ (a remote loop in each job's last 4
  minutes). Runner: collect_jobs / tune_table now credit jobs by id across runs and salvaged copies (the last line
  of a salvaged copy dropped); checked on CARC (13345844): reproduces the round-0 tables (|dQLIKE| <= 1.2e-16, equal
  epochs, same picks); new stage `remainder` writes the not-yet-complete lines of timed-out files. GPU sbatch:
  `--signal=B:USR1@240` + a handler copying the partial outputs to salvage/ before a time limit.
- 2026-10-23 12:50-13:10 PDT: read-only tuning-noise analysis (coordinator; experiments/
  pooled_mingru_final_tune_analysis.py, 13348038, CPU; outputs analysis_w3000/): HAR (2,973 configurations complete
  incl. round-1 runs and salvaged H 1,024 jobs) leader L8_H1024_lr0.003_we15_wl0.03_ll0.001 (0.122417); ranks 2-21
  within 0.00008-0.00037 of it, block-bootstrap SE of the differences 0.00018-0.00055 (29 configurations within
  1 SE, 284 within 2 SE); seed SE of the mean 0.00023; the ridge shortcut alone 0.1302 (net -6 %). Full set: pick
  L20_H64 0.113381, 8 within 1 SE, 44 within 2 SE. The HAR edge drift is selection among statistically equivalent
  configurations.
- GPU-type gates resubmitted (13346565-68) after finding the first V100 / A40 segfaults were the gate script's nested
  --out path (diagnosis v2 13331862: the gate's own jobs run cleanly on a V100). Heavy-shape packing (13335886): no
  gain at H 256 (memory-bound, ~860 deferrals), OOM at CONC 24 for full H 256 L 48: CONC 24 only for H <= 64, L <= 20.
- 2026-10-23 13:30-14:00 PDT: GPU-type gates (fixed script): L40S (13346568) and A40 (13346566) NOT identical to the
  P100 (validation forecasts max rel. diff 3.1 % / 1.6 % HAR, 10.3 % / 9.1 % full; best epochs identical; seed-mean
  QLIKE within 0.05 %) but self-consistent (tune twice -> identical); L40S ~2x the P100's speed. V100 / A100 pending.
- Read-only replay with the owner's fixed extension trigger (experiments/pooled_mingru_final_replay.py, 13351216,
  replay_w3000/replay.json): extend an edge axis only if the best configuration at the edge beats the best at the
  adjacent value by more than k bootstrap SE. HAR (k = 1 and 2): every edge test L48 / H256 / wd_lin 30 / lr_lin 1e-4
  vs the adjacent best L20_H64_lr0.01_we3_wl3_ll1e-05: d = 0.000035, SE 0.000507 -> no extension; frozen at round 0
  L48_H256_lr0.003_we3_wl30_ll0.0001 (0.122699; epochs 7/17/6/3/12; ~5.5 P100-s a refit, walk ~258 P100-h by tune
  epoch cost). Full set: round-0 pick interior, unchanged. W 3,500 round 0 at 20 / 44 (not yet replayable).
