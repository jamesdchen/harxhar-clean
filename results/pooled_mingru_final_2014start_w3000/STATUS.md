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
