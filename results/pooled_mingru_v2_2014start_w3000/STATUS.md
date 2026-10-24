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
- 2026-10-23 18:47 PDT: cancelled the 19 (each checked: ours, pmft, PENDING, in pmf35/inputs/submitted.txt): 13320235-50,
  13320257, 13320262, 13320263 -- "cancelled 19". Left: 2 pmfgt PENDING (old GPU-type gates), the old walk (4 RUNNING,
  32 PENDING; 0 of its 68 halves unsubmitted). No feeder restarted.
- Tooling: experiments/pooled_mingru_v2.py (prep / post / bar-inits on CARC; jobs / s1-table / s2-table / s2-report
  locally), cluster/pooled_mingru_v2_gpu_carc.sbatch (node-local staging, md5-checked binary 44ab6fff, P100 check,
  USR1 salvage, POST = check + epoch curves + compact forecasts), cluster/pooled_mingru_v2_cpu_carc.sbatch,
  cluster/pooled_mingru_v2_submit_carc.sh. CARC roots /home1/jc_905/pmf2 (bin = copy of pmf/bin, code) and
  /home1/jc_905/pmf2_35 (bin, code: links to pmf2's). Only trainer keys of forecast_train.cu's known-key list are used.
- 2026-10-23 18:47 PDT G0 PASS (W 3,000, CPU 13353637, a03-05; g0.json, init_index_v2.csv): fold rows and validation
  dates = folds14.json (fold 1 val 2011-01-05 13:30 .. 2011-12-30 14:00; fold 2 2011-12-30 14:30 .. 2013-01-02 15:00;
  fold 3 2013-01-02 15:30 .. 2013-12-31 16:00); fit ends 2010-11-30 13:00 / 2011-11-22 16:00 / 2012-11-22 10:30.
  Fit starts (rows 1,702 / 4,952 / 8,202 are < 39,000, undated in the design; the regular-session bar calendar of
  core_stats aligned at row 39,000 gives about 1999-01-12 / 2000-01-07 / 2001-01-04, +- days: the design calendar
  departs from core_stats' in places). Stage-1 inits: the old fold-3 tune inits recomputed byte for byte (L 8 / 20 /
  48 / 115, wl 0.003), copied to pmf2/data/pmf_lin_init; kept columns HAR 12, full 293 (P = keep + 1 for all 32
  inits). TQ_0 (training QLIKE of the frozen shortcut alone, trainer formula): HAR 0.121085 / 0.121038 / 0.120994 /
  0.121074 (L 8 / 20 / 48 / 115), full 0.112318 / 0.112273 / 0.112209 / 0.112258. Shortcut-alone fold-3 validation
  QLIKE reproduces the old analysis (HAR L115 0.130134 vs 0.13013; full L48 0.122677 vs 0.12268). Stage-2 per-fold
  inits: kept columns full 283 / 288 / 293 (HAR 12); partly zeroed late-start columns (full) 6 / 5 / 5 of the kept
  (2.1 / 1.7 / 1.7 %; mean zeroed share of the window 16 %, max 96 / 88 / 80 %). Shortcut-alone validation QLIKE (L20)
  by fold: HAR 0.12647 / 0.13470 / 0.13015, full 0.11548 / 0.12704 / 0.12270.
- 2026-10-23 18:52 PDT: G1 submitted (13353680, debug P100: g1_baseline.txt = the Stage-1 tune line, its refit twin
  with fc_train, and the shortcut alone through the trainer: refit, lr 0, 1 epoch, fc_train -> checks TQ_0's offline
  formula). G2 submitted held afterok on G1's job (its POST checks only status / kept columns / rows; the G1 criteria
  are checked before any G2 number is used, G2 cancelled if G1 fails): E 300 files 13353696 (HAR) / 13353697 (full) on
  debug, E 1,000 files 13353698 / 13353699 on gpu.
