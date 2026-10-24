# Pooled ridge-init MinGRU v2 ("memorize, then regularize"), W 3,000: status log

Rule: `RULE_gru_v2.txt` (written 2026-10-23 18:40 PDT, before any v2 job). Branch insurance-selling-2026-09-19-e34-gru-v2
(from 2cc624c). CARC root `/home1/jc_905/pmf2` (W 3,500: `/home1/jc_905/pmf2_35`, its own STATUS.md). Inputs read from
the old run (read-only): `/home1/jc_905/pmf/data` (design copies, fold-3 ridge inits), `/home1/jc_905/pmf/bin`
(forecast_train md5 44ab6fff), env `/home1/jc_905/envs/harxhar-r13`.

## Log
- 2026-10-23 18:31 PDT, cluster state before any v2 action (ours, jc_905; 61 jobs): 19 old-rule W 3,500 round-0 tune
  files PENDING in the gpu partition (nice 1000), 40 old-rule W 3,000 full-set every-bar walk halves (8 RUNNING
  b14a-b17b, 32 PENDING b18a-b33b; all 68 halves b0a-b33b submitted), 2 old GPU-type gate jobs PENDING (pmfgt
  13346565, 13346567). No old-rule W 3,000 HAR round-1 tune file is in Slurm: the 336 held H 1,024 files exist only as
  lines of /home1/jc_905/pmf/gpu_queue.txt / gpu_queue2.txt / gpu_queue_h1024_held.txt (never submitted). Feeders:
  gpu_feed.log and debug_feed.log last written 2026-10-23 16:14:35; no crontab; no feeder process of ours on
  discovery1 / discovery2 or on the laptop (Windows process list: only another session's paper build).
- Cancellation plan (owner: old-rule tuning not needed; running jobs finish; the old W 3,000 walk and its ridge-init
  jobs stay): scancel the 19 PENDING W 3,500 round-0 tune jobs (pmft, /home1/jc_905/pmf35/inputs/submitted.txt):
  13320235-13320250 (tune_baseline_r0.3 .. r0.18, 16 files), 13320257 (tune_all_features_r0.6), 13320262
  (tune_all_features_r0.11), 13320263 (tune_all_features_r0.12). Not touched: the 40 walk halves, the 2 GPU-type
  gates (not tuning jobs), anything RUNNING, other users' jobs. The 336 H 1,024 lines stay unsubmitted (no feeder runs;
  the old root is not modified).
