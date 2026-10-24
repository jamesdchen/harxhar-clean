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
- 2026-10-23 23:37 PDT G3 full set (W 3,000) final (stage1/s1_w3000_full_*.csv / _pick.json; L40S): pick H 64, L 8,
  lr 0.003, "memorizing": rho_300 0.1211 / 0.1214 / 0.1245 (3 seeds agree; rho <= 0.1 not reached). The cheaper
  (16, 8) / (16, 20) / (16, 48) cells do not memorize (rho 0.676-0.678); every H 64 cell does (0.122-0.126), every
  H 256 cell to rho ~ 0 (<= 0.1 too). lr 0.03 diverges in every cell (TQ_300 16-933, no non-finite); lr 0.003 is the
  argmin in all 9 cells. Validation QLIKE at epoch 300 0.16-0.93 vs per-run minima 0.1175-0.1203 at epochs 2-8
  (diagnostic). No conditional edge applies (a core cell memorizes; the pick is at L 8). Measured on the L40S: 0.025
  GPU-s a model-epoch for H <= 64, L <= 20 at CONC 24 (P100 table 0.037-0.071); 0.046-0.25 for the CONC 6 files.
- 2026-10-23 23:38 PDT G5 + G6 (full set, W 3,000), GPU type A40 (ADDENDUM 1a): state s2_state.json (H 64, L 8, lr
  0.003; tau {0, 0.3, 1, 3} x wd_enc {0, 3, 15} x E {10, ..., 300}); canary 13367874 (debug A40: fold 1, tau 1, wd_enc 3,
  E 20, seed 20261012, tag v2g5); screen (72 configurations x 3 folds x 5 seeds = 1,080 jobs, 15 files, est. 5.95
  P100-h with the 15 s a job overhead) afterok on the canary: 13367875-87 (gpu A40), 13367888-89 (debug A40).
- 2026-10-23 23:56 PDT G2 decision (g2_decide.json; P100): HAR E1 = 1,000: (256, 20) at its best lr 0.003 (2-seed mean
  TQ_300) rho_300 0.582 > 0.5, rho_1,000 0.375, relative gain 35.5 % >= 20 %. Full set E1 = 300 (rho_300 ~ 0). Stage-1
  lr set {0.003, 0.01, 0.03} (argmins: full (64,20) 0.003, (256,20) 0.003; HAR (64,20) 0.01, (256,20) 0.003). Both
  windows. Diagnostics: HAR E 1,000 at lr 0.001 rho 0.495, at lr 0.01 rho 1.0000000 (both seeds: the recurrent branch
  collapsed to zero output, the forecast = the shortcut); full E 1,000 at 0.003 rho ~ 0, at 0.01 0.0009. G2 reruns
  13366974 / 13366975 (gpu P100), 13367856 / 13367857 (debug P100) completed (n_bad 0); 13366976 (full lr 0.001
  E 1,000, wall-clock only) running. G3 / G4 full set: W 3,500 13367193-95 completed, 13367196 running, 13367197-98
  pending (L40S). Canary 13367874 still pending in debug (A40).
- 2026-10-24 00:05 PDT: RULE ADDENDUM 1b: the HAR Stage-1 groups (both windows) on P100 (the type with the most cap
  headroom at submission: A40's 12 are taken by our full-set screen).
- 2026-10-24 00:08 PDT G3 HAR (W 3,000) submitted on P100: 12 files (81 jobs: 9 cells x lr {0.003, 0.01, 0.03} x 3
  seeds, E 1,000; est. 4.20 P100-h): 13369109-10 debug P100, 13369111-20 gpu P100. G4 HAR (W 3,500): 14 files
  13369121-42 gpu P100, nice 100 (est. 4.86 P100-h). The full-set canary also submitted to gpu A40 (13369150; the debug
  copy 13367874 waits behind other users' debug jobs); the 15 screen files now wait on "afterok:13367874 ?
  afterok:13369150" (either copy). The commit "G2 decided ..." swept in another session's untracked
  hoffman2_gpu_test/ files by a directory add; untracked again in the next commit (files left on disk).
- 2026-10-24 00:15 PDT G5 PASS (full set, W 3,000; A40): canary 13367874 (debug b11-09) and its gpu copy 13369150
  (b11-13): status ok, forecast rows exactly [40,702, 44,277), finite and positive, kept columns = the init's (POST
  n_bad 0). Screen running on A40: 13367875-78, 13367888-89 completed (n_bad 0), 13367879-80 running, 7 pending.
  G3 HAR: 13369109 (debug P100) completed, 13369110-12 running, 9 pending (P100). G2's last file 13366976 completed.
- 2026-10-24 01:31 PDT: G6 screen (full set, W 3,000, A40) complete: 15 files 13367875-89, 1,080 jobs, all n_bad 0
  (walls 26-324 s a file on A40). Screen top 5 by C (stage2/s2_screen_all_features.csv): tau1_we3_E40 0.115805,
  tau3_we3_E80 0.116024, tau0_we3_E40 0.116063, tau0.3_we15_E160 0.116111, tau3_we15_E300 0.116127 (the shortcut
  alone, L 20: 0.12174). Fresh-seed confirmation (75 jobs, seeds 20261017-21, tag v2s2c) submitted on the debug A40s:
  13371985-86.
- 2026-10-24 01:03-01:29 PDT: 4 HAR Stage-1 files failed on /scratch1 (Lustre client "Cannot send after transport
  endpoint shutdown" on e21-01, e22-10, e22-14: the copy back / log write failed; W 3,000 13369114 (.5), 13369116 (.7);
  W 3,500 13369129 (.6), 13369131 (.7)). /home1 is writable again (80 of 100 GiB, someone freed space) but too tight
  for v2's outputs (3 GB so far). New v2 root: /project2/pollok_1603/jc_905/pmf2(_35) (VAST NFS, 5 TiB group space;
  the jc_905 dir mode 700; bin, code, inputs, inits copied from /scratch1, md5 of 102 / 84 files equal). Reruns from
  there: W 3,000 .5 / .7 on the debug P100s (13371959-60), W 3,500 .6 / .7 on gpu P100 (13371961-62, nice 100).
  Pending scratch jobs moved (cancelled while pending, resubmitted unchanged from /project2): W 3,500 screen .9-.13
  (13370090-94 -> 13371964-68, L40S) and HAR .13 (13369142 -> 13371969, P100). Completed runs stay in /scratch1.
- 2026-10-24 02:17 PDT (coordinator ~02:15, owner: "be careful about ssh connections"): every CARC call of this session
  now goes through one wrapper with a lock (at most one SSH session at a time, the background wait included; batched
  commands a step; no SSH in loops; the single wait polls >= 15 min apart and retries a failed connection only after
  another 15 min). The wait re-armed on 13371959-60 / 13371985-86.
- 2026-10-24 02:12 PDT G3 HAR (W 3,000) final (P100; 12 files incl. the 2 reruns from /project2, all n_bad 0;
  stage1/s1_w3000_*.csv / _pick.json now hold both sets): pick H 256, L 8, lr 0.003, "memorizing": rho_1,000 0.3760 /
  0.3846 / 0.3792 (3 seeds agree; rho <= 0.1 not reached). Only the H 256 cells memorize (0.377-0.380); H 64 0.729-0.733,
  H 16 0.875 (lr 0.01 there). lr 0.03 leaves the recurrent branch at zero (rho 1.000) in most cells; at H 256 lr 0.01
  also collapses (0.97-0.98). No conditional edge (a core cell memorizes; the pick is at L 8). Validation QLIKE at
  epoch 1,000 0.29-0.31 at the H 256 cells vs minima 0.1240-0.1244 at epochs 10-12 (diagnostic).
- 2026-10-24 02:12 PDT G6 full set (W 3,000) round 0 (A40): fresh-seed C of the shortlist (stage2/s2_fresh_*_round0):
  tau3_we3_E80 0.115961 (pick), tau1_we3_E40 0.116016, tau3_we15_E300 0.116185, tau0.3_we15_E160 0.116303, tau0_we3_E40
  0.116317. The pick is at the tau high edge (3) -> extension 1: tau 9 added (18 configurations x 3 folds x 5 screen
  seeds = 270 jobs, tag v2s2x1, A40): 13372774-75 gpu, 13372776-77 debug. State: s2_state.json (ext tau high 1).
- 2026-10-24 02:13 PDT G5 + G6 HAR (W 3,000), GPU type P100 (ADDENDUM 1b criterion at submission: P100 no type cap,
  36 - our 7 gpu jobs + 2 debug; A40 12 + 2; L40S 6 - 5 pending): state baseline H 256, L 8, lr 0.003; grid tau
  {0, 0.3, 1, 3} x E {10, ..., 300, 600, 1,000} (32 configurations, 480 jobs, est. 7.39 P100-h); canary (fold 1, tau 1,
  E 20) 13372778 (debug) / 13372779 (gpu); screen afterok on either: 13372780-96 gpu, 13372797-98 debug.
- 2026-10-24 02:30 PDT: G5 PASS (HAR, W 3,000; P100): canaries 13372778 (debug) / 13372779 (gpu) completed, POST n_bad
  0, finite, positive. HAR W 3,000 screen running (P100: 13372797-98 done, 8 running, 9 pending). Full set W 3,000
  extension 1 (tau 9, A40: 13372774-77) done; round-1 shortlist (screen seeds, all 90 configurations): tau1_we3_E40,
  tau9_we3_E160, tau3_we3_E80, tau0_we3_E40, tau0.3_we15_E160; fresh seeds for the new member tau9_we3_E160 (15 jobs,
  tag v2s2c1, debug A40): 13373058.
- 2026-10-24 02:31 PDT: G7 preparation: every Stage-1 pick is at L 8, so the 24 bar-origin ridge inits (L 8, mask and
  Gram on [o - W, o), o = 50,777 .. 50,800) are computed now for both sets at both windows (CPU 13373087 W 3,000,
  13373088 W 3,500, /project2 roots). The G7 job writer now cuts a probe into whole-origin files under the 1 h debug
  limit (HAR H 256 at E 1,000 would be ~4.9 P100-h in one file).
- 2026-10-24 02:46 PDT G6 full set (W 3,000) round 1 (A40): fresh-seed C of the round-1 shortlist (stage2/
  s2_fresh_all_features_round1.csv): tau9_we3_E160 0.115927 (pick), tau3_we3_E80 0.115961, tau1_we3_E40 0.116016,
  tau0.3_we15_E160 0.116303, tau0_we3_E40 0.116317 (spread of the top 3: 0.00009). The pick is again at the tau high
  edge (9) -> extension 2 (the last on that side): tau 27, 18 configurations, 270 jobs (tag v2s2x2, A40):
  13373353-54 gpu, 13373355-56 debug. G7 bar inits done (CPU 13373087 / 13373088: 24 files a set, both windows).
- 2026-10-24 03:03 PDT G6 HAR (W 3,000) screen complete (P100, 20 files, 480 jobs, all n_bad 0; walls 389-1,277 s).
  Screen top 5 by C (stage2/s2_screen_baseline_round0.csv): tau1_we0_E10 0.126221, tau0.3_we0_E10 0.126234,
  tau1_we0_E20 0.126251, tau3_we0_E40 0.126288, tau3_we0_E20 0.126296 (the shortcut alone, L 20: 0.13044); long
  training hurts (E 1,000 configurations 0.13-0.15). Fresh seeds for the shortlist (75 jobs, tag v2s2hc, debug P100):
  13373698. Full-set extension 2: debug files 13373355-56 done, gpu 13373353-54 pending (A40).
- 2026-10-24 03:19 PDT G6 HAR (W 3,000) round 0 (P100): fresh-seed C (stage2/s2_fresh_baseline_round0.csv):
  tau3_we0_E40 0.126140 (pick), tau3_we0_E20 0.126283, tau1_we0_E20 0.126398, tau0.3_we0_E10 0.126432, tau1_we0_E10
  0.126451. The pick is at the tau high edge (3) -> extension 1: tau 9 x E {10, ..., 1,000} (8 configurations, 120
  jobs, tag v2s2hx1, P100): 13373988-89 gpu, 13373990-92 debug. (E 40 is interior.)
- 2026-10-24 04:39 PDT: the background wait was killed by the harness (low laptop memory, ~1.8 GB free); re-armed.
  03:20-05:40 the gpu partition started none of our pending jobs (P100, A40, L40S all queued behind higher-priority
  work); the debug partition ran the HAR tau-9 files.
- 2026-10-24 05:50 PDT G6 HAR (W 3,000) extension 1 done (tau 9, 120 jobs, P100: 13373988-92, n_bad 0). Round-1
  shortlist (screen seeds, stage2/s2_screen_baseline_round1.csv): tau9_we0_E80 0.126084, tau1_we0_E10, tau0.3_we0_E10,
  tau1_we0_E20, tau3_we0_E40; fresh seeds for the new member tau9_we0_E80 (15 jobs, tag v2s2hc1, debug P100).
- 2026-10-24 06:06 PDT G6 full set (W 3,000) FINAL (A40): extension 2 (tau 27) done (13373353 gpu, 13373355-56 and
  13376916 debug; 13373354 cancelled while pending and rerun on the debug A40s); no tau-27 configuration enters the
  screen top 5, so the shortlist and the fresh-seed pick stand: tau 9, wd_enc 3, E 160, fresh C 0.115927 (interior
  on every axis: tau 9 < 27, wd_enc 3 in {0, 3, 15}, E 160 in {10, ..., 300}). Reported, not deciding
  (stage2/s2_report_all_features.json): bootstrap SE of C - C(pick) 0.00017-0.00040; seed-jackknife SE of C
  0.00003-0.00018; 90 % MCS = the whole shortlist (p 0.80). Pick vs runner-up tau3_we3_E80: 0.000034 (0.2 SE).
- 2026-10-24 06:06 PDT G6 HAR (W 3,000) round 1: fresh-seed pick tau9_we0_E80 0.126130 (tau3_we0_E40 0.126140,
  tau1_we0_E20 0.126398, ...): at the tau high edge (9) -> extension 2 (the last): tau 27 (120 jobs, tag v2s2hx2, P100):
  13377080-81 gpu, 13377082-84 debug.
- 2026-10-24 06:06 PDT G7 full set (W 3,000), P100 (the probe's output is P100-hours; one file): 24 origins x 5 seeds at
  H 64, L 8, lr 0.003, tau 9 (wd 0.12218715 at S 305), wd_enc 3, E 160; first submission 13377085 cancelled while
  pending: the bar inits of the full set were missing (the CPU jobs got SETS=baseline only: a comma inside --export
  splits the variable); full-set bar inits recomputed (CPU 13377088 W 3,000, 13377089 W 3,500); G7 resubmitted
  13377090 (debug P100) afterok on 13377088.
- 2026-10-24 06:39 PDT: HAR tau-27 extension: gpu files 13377080-81 done; the debug files 13377082-84 and the G7 probe
  13377090 waited 30 min behind other users in debug (its 2 P100s), cancelled while pending and resubmitted unchanged
  to gpu P100: 13377401-03 and G7 13377404.
- 2026-10-24 06:55 PDT G7 full set (W 3,000) done (13377404, P100 e21-01, 120 refits, all ok, finite, positive; wall
  546 s at CONC 24; t_train a refit 101 s mean while 24 share the GPU; g7/): 4.55 P100-s a bar refit (E 160) ->
  every-bar walk 33,970 bars x 5 seeds = 214.7 P100-h for the full-set arm at W 3,000 (projection; no walk submitted).
- 2026-10-24 07:13 PDT G6 HAR (W 3,000) extension 2 (tau 27) done (13377080-81, 13377401-03; n_bad 0). Round-2 screen
  shortlist (stage2/s2_screen_baseline_round2.csv): tau27_we0_E300 0.126069, tau9_we0_E80 0.126084, tau1_we0_E10,
  tau0.3_we0_E10, tau1_we0_E20; fresh seeds for tau27_we0_E300 (15 jobs, P100): 13377955. tau has had its 2 high
  extensions: a pick at tau 27 stands (the ladder is exhausted).
- 2026-10-24 07:29 PDT G6 HAR (W 3,000) FINAL (P100): fresh-seed pick tau27_we0_E300 0.126108 (tau9_we0_E80 0.126130,
  tau1_we0_E20 0.126398, tau0.3_we0_E10 0.126432, tau1_we0_E10 0.126451). tau 27 is the ladder's last value (2 high
  extensions done): the rule stops there ("extended per the rule"); E 300 interior in {10, ..., 1,000}. Reported
  (stage2/s2_report.json, both sets): bootstrap SE of C - C(pick) 0.00014-0.00023, seed-jackknife SE 0.00005-0.00015,
  90 % MCS = the whole shortlist (p 0.42). G7 HAR (W 3,000), P100: H 256, L 8, lr 0.003, tau 27 (wd 0.196067752),
  E 300, 4 files of 6 origins x 5 seeds: 13378169-72.
- 2026-10-24 08:00 PDT G7 HAR done, both windows (P100; all refits ok, finite, positive). W 3,000: 13378169-72, 4 files
  of 30 refits, walls 1,267-1,270 s -> 42.3 P100-s a bar refit (E 300) -> every-bar walk 1,995 P100-h. W 3,500:
  13378173, 120 refits, wall 788 s -> 6.57 P100-s a refit (E 40) -> 310 P100-h. The old A100 gate 13346567 ran
  (06:47, b05-13): self-consistent; its check numbers not read (A100 not used by v2).
- 2026-10-24 08:05 PDT SUMMARY (all v2 gates passed; STOP before any every-bar walk, none submitted):
    picks (H, L, lr | tau, wd_enc, E | fresh-seed C):
      W 3,000 full: 64, 8, 0.003 | 9, 3, 160 | 0.115927   (Stage 1 L40S, Stage 2 A40)
      W 3,000 HAR : 256, 8, 0.003 | 27, 0, 300 | 0.126108  (P100 / P100; tau at the ladder's last value)
      W 3,500 full: 64, 8, 0.003 | 1, 3, 40 | 0.115610     (L40S / L40S)
      W 3,500 HAR : 256, 8, 0.003 | 3, 0, 40 | 0.125707    (P100 / P100)
    GPU time used by v2 (allocated elapsed, incl. staging, POST and failed attempts; v2_gpu_jobs_acct.txt):
      P100 30.25 h (114 jobs), L40S 2.33 h (29), A40 1.07 h (28).
    Projected every-bar walk (33,970 bars x 5 seeds an arm; G7 P100-s a refit): full W 3,000 214.7, full W 3,500 73.6,
      HAR W 3,000 1,995, HAR W 3,500 310 P100-h; 2,593 P100-h in all. Wall-clock at 17 P100s (the most v2 held at
      once, 06:05) 153 h (6.4 days; HAR W 3,000 alone 117 h); at the QOS's 36 GPUs 72 h.
- 2026-10-24 09:00-09:20 PDT (owner, relayed: "A, walk all four as pre-registered"; 09:10 coordinator: HOLD, no GPU
  walk job until the go). RULE ADDENDUM 2 (both windows) written before any walk job: the walk, SCORE (as the final
  run's scorer), placement (HAR W 3,000 P100 27 lanes x 2; HAR W 3,500 A40 4 x 2; full W 3,000 A40 3 x 2; full W 3,500
  L40S 2 x 2: 72 files, 36 lanes = the gpu QOS's 36 running jobs), steps log to /dev/null after a slice check, HOLD.
  Read for it: QOS gpu = 36 running / 100 submitted, A40 12, L40S 6, A100 12; gpu partition 48 h, debug 1 h / 5 jobs;
  the A100 gate 13346567 fails ADDENDUM 1 (full-set best epochs 12/10/10 vs P100 12/10/11; QLIKE 0.013 % / 0.027 %,
  self-consistent); free at 09:06: P100 28 / 74, A40 1 / 50, L40S 2 / 19. The trainer has no switch for its step log
  (forecast_train.cu opens steps.csv unconditionally): walk files point it at /dev/null.
  Tooling: walk-plan / walk-inits / walk-merge / walk-combine (pooled_mingru_v2.py; merge tested on synthetic
  forecasts: remainder file and complete paths), GPU sbatch NOSTEPS / PARTIAL_S, walk submit script (lanes,
  afterok on the bar-init task and the slice check), slice check sbatch, CPU sbatch walkinits / merge modes.
- 2026-10-24 09:18-09:25 PDT walk prepared (HOLD: no GPU job submitted). /project2/pollok_1603/jc_905/pmf2(_35)/inputs:
  walk files v2w_<set>.<i>.txt (W 3,000: HAR 54, full 6; W 3,500: HAR 8, full 4; every origin 50,777 .. 84,746 x 5
  seeds; plan files files_v2w_<set>.csv with lane / seq / time limit; est. 1,995.7 / 214.7 / 310.0 / 73.6 P100-h;
  time limits 2,610-2,614 / 2,529 / 2,734-2,735 / 1,716-1,717 min), the slice file v2wslice.txt (W 3,000: 2 full-set
  bars x 5 seeds + 1 HAR bar x 5 seeds), code/go_walk.sh (the slice check, then the 72 walk files; NOT run).
  Bar inits submitted (CPU, main; oneweek refused: its QOS allows 50 submitted / 208 CPUs a user and our other
  session holds 20 / 180): W 3,000 HAR 13379947 (54 tasks), full 13379948 (6); W 3,500 HAR 13379951 (8), full
  13379952 (4, nice 100); ids in inputs/walkinit_<set>.jobid. core_stats.parquet is not under /home1/jc_905/pmf/repo
  on CARC: the merge runs locally on pulled outputs (tested) or on CARC with RV=<path>.
