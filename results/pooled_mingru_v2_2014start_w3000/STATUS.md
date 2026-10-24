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
- 2026-10-23 19:10 PDT: G1 still PENDING on debug (e23-02's two P100s held by another user's 55-min jobs, 4 jobs ahead;
  our priority 3,163, fairshare factor 151). The same G1 file also submitted to the gpu partition (13354135, 12 min;
  P100s: 72 of 74 allocated): whichever starts first is kept, the other cancelled (G2's afterok then moved to it).
- 2026-10-23 19:58 PDT: G1 still PENDING (debug: expected start 20:42; gpu: priority). The two E 1,000 G2 files (63 / 78
  min limits, poor backfill odds) cancelled while PENDING (13353698 / 13353699, ours) and replaced by 6 files of one
  (set, lr) each (v2g2m_*, 2 jobs, 28 / 33 min: 13355108-13). Every G2 file now waits afterok on either G1 copy.
- 2026-10-23 20:41 PDT, hand-back (the agent's run ended before G1 ran): queued, nothing of v2 has run on a GPU yet.
  G1 13353680 (debug, expected start 20:42) and its gpu copy 13354135 (cancel whichever has not started once one has);
  G2 E 300 files 13353696 / 13353697 (debug), E 1,000 files 13355108-13 (gpu), all afterok on either G1 copy.
  Next: pull runs/g1_baseline_<id> (summary/, tune/steps.csv.gz, tune/epochs.csv) and run
  `pooled_mingru_v2.py g1-check`; then pull runs/v2g2*_*/summary and run `g2-decide` (E1 per set, Stage-1 lr set);
  then Stage 1 at both windows (`jobs --kind s1`, W 3,500 with nice), `s1-table`, etc. per RULE_gru_v2.txt.
  The old walk's pending halves (15 PENDING, 12 RUNNING) have older submit times and take the gpu partition's P100s
  ahead of v2 jobs; not touched.
- 2026-10-23 20:43 PDT (continuation agent): G1 13353680 started on debug at 20:42:33 (e23-02). Its gpu copy 13354135
  (ours, PENDING) cancelled; the 8 G2 files' dependency moved from "afterok:13353680?afterok:13354135" to
  afterok:13353680 (13353696 / 13353697 debug, 13355108-13 gpu). Polls from here: one background wait at a time,
  900 s between checks (low laptop RAM).
- 2026-10-23 20:59 PDT G1 PASS (13353680, debug e23-02 P100, 20:42:33-20:44:45, wall 127 s; g1_check.json): 300 epochs
  run by the tune job and its refit twin; steps.csv text-identical (91,500 step lines, 0 differ); exact in-sample QLIKE
  of the twin's fc_train rows [8,221, 47,202) 0.0946888 vs last-epoch running mean TQ_300 0.0946890 (rel 2.4e-6 <= 1%);
  twin's validation QLIKE (trainer formula, rows [47,527, 50,777)) 0.14297255 = epochs.csv epoch 300 0.14297256 (rel
  5.0e-8 <= 1e-5); no non-finite; the shortcut alone through the trainer (lr 0, 1 epoch) 0.121038240 = offline TQ_0
  0.121038241 (rel 3.2e-9); mu_y / sd_y of the job = the init's. Diagnostic: rho_300 of (H 64, L 20, lr 0.003) = 0.782
  (does not memorize); validation QLIKE best at epoch 28 (0.12465), 0.14297 at epoch 300. G2's 8 files now released
  (dependency met), PENDING (debug: Resources / Priority; gpu: behind the old walk's halves).
- 2026-10-23 21:48 PDT: G2 not started yet. Debug: 13353696 expected 22:00, 13353697 22:40 (debug's one P100 node
  e23-02 held by another user's 55-min jobs). gpu: the 6 E 1,000 files are the top P100 pending jobs (priority 3,194);
  the P100s are full (the old walk's 15 running halves end 22:40-23:21, none of its 68 halves pending). The local
  background wait was killed by the harness for low laptop memory at ~21:47 (not the wait's own use); re-armed.
- 2026-10-23 22:09-22:42 PDT G2 ran: E 300 files 13353696 (HAR, debug, 520 s) and 13353697 (full, debug, 733 s) and the
  E 1,000 file 13355108 (HAR lr 0.001, gpu d23-16, 666 s) COMPLETED (checks n_bad 0). 13355109 / 13355110 (HAR E 1,000
  lr 0.003 / 0.01) FAILED exit 1 at 22:40 / 22:42 with an empty log; 13355111-13 (full E 1,000, all lr) FAILED in 2 s
  (signal 53, no log file). Diagnosis: the /home1/jc_905 quota (100 GiB) is exhausted ("Disk quota exceeded" on a
  touch at 22:45; myquota's cached 89.2 GiB is stale): the old W 3,000 walk writes ~385 MB of steps.csv.gz a half to
  /home1/jc_905/pmf/runs (pmf 33 GB). Old walk halves 13346982 / 13347424 failed the same way at 22:38 / 22:42 (not
  touched; reported to the coordinator). The A100 gate 13346567 (writes to /home1/jc_905/pmf) will fail too if it
  starts before space is freed.
- 2026-10-23 22:35 PDT owner decision (relayed): every CARC GPU type may be used, one type per comparison group, each
  validated first. RULE ADDENDUM 1 written (both windows) at 22:55 PDT before any v2 job on another type: validated A40
  (13346566) and L40S (13346568) pass; V100 (13346565, read) FAILS the full-set conditions (seed-mean QLIKE 0.152 % >
  0.1 %, best epochs 12/10/13 vs 12/10/11) and is not used; A100 pending. Free GPUs at 22:42 (gpu partition, nodes
  up): L40S 2 / 19, A100 2 / 34, A40 0 / 50, P100 0 / 74, V100 0 / 52 (the coordinator's ~48 free V100 no longer held).
- 2026-10-23 23:00 PDT: v2 roots moved to /scratch1/jc_905/pmf2 and /scratch1/jc_905/pmf2_35 (bin, code, inputs,
  data/pmf_lin_init copied, md5 of every file equal: 59 / 44 files; bin and code of pmf2_35 link to pmf2's). Scripts
  updated (GPUTYPE + --constraint; the sbatch path read from ROOT/code; a failed trainer leaves its log in salvage/).
  Completed G1 / G2 runs stay in /home1/jc_905/pmf2/runs (read only).
- 2026-10-23 22:52 PDT: the 5 failed G2 files rerun unchanged on P100 from /scratch1/jc_905/pmf2 (13366974 HAR lr
  0.003 / 13366975 HAR lr 0.01 / 13366976-78 full lr 0.001 / 0.003 / 0.01; run dirs v2g2m_*_<id> there).
  G2 decision read so far (g2-decide on the 3 completed files; the E 300 files are complete): Stage-1 lr set
  {0.003, 0.01, 0.03} (argmin of the 2-seed mean TQ_300: full (64,20) 0.003, (256,20) 0.003; HAR (64,20) 0.01 ->
  + 0.03, (256,20) 0.003; 0.001 the argmin nowhere). Full set: (256,20) at its best lr 0.003 has rho_300 ~ 0 (TQ_300
  -2e-10: float32 rounding of a perfect training fit) <= 0.5 -> E1 = 300 for the full set at both windows (its E 1,000
  runs do not enter). HAR: (256,20) at 0.003 has rho_300 0.582 > 0.5 -> E1 waits for the E 1,000 run at lr 0.003
  (13366974). Diagnostics: full set memorizes fast (H 64 L 20 lr 0.003 rho_300 0.123; validation QLIKE 0.45-0.88 at
  epoch 300 vs best 0.118-0.121 at epochs 3-7); HAR does not at E 300 (best rho_300 0.576 / 0.588).
- 2026-10-23 22:55 PDT G3 (W 3,000) and G4 (W 3,500), full set, submitted: core grid 9 cells x lr {0.003, 0.01, 0.03}
  x 3 seeds, E 300 (81 jobs a window). GPU type of the Stage-1 group at each window: L40S (ADDENDUM 1 choice: free in
  the gpu partition at 22:55:53 L40S 2, A100 2 (not validated), A40 / P100 / V100 0). W 3,000 13367188-92 (5 files),
  W 3,500 13367193-98 (6 files, nice 100). Job files now hold one concurrency each (pack_files: a file runs at its
  minimum CONC; mixing CONC 24 and 6 cells would have slowed the small cells). The HAR half of each window's grid
  follows on L40S once G2 fixes HAR's E1.
- 2026-10-23 23:35-23:45 PDT (coordinator 23:36: all CARC resources, debug included; per-type QOS caps A40 12, A100 12,
  L40S 6 a user). G3 full set (W 3,000) COMPLETED on L40S b17-16: 13367188-92, 23:02-23:33, walls 277 / 209 / 390 /
  528 / 381 s (est. 914-1,428 P100-s a file: L40S ~3x the P100 table's rate). G4 full set: 13367193 RUNNING on L40S
  since 23:33, 13367194-98 pending (stay L40S). G2: 13366974 (HAR lr 0.003 E 1,000, the decision run) RUNNING on P100
  e22-03 since 23:34; 13366977 / 13366978 cancelled while pending and resubmitted to the debug P100s as 13367856 /
  13367857; 13366975 / 13366976 stay pending in gpu. RULE ADDENDUM 1a (both windows): a Stage-1 group is one set's grid
  at a window; groups not yet started take A40 (12 + 2 debug); debug used for < 1 h files (<= 5 jobs). The old walk's
  reruns 13367745-7 (pending) are not ours to touch.
