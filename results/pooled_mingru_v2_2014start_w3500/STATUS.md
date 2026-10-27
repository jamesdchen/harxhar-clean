# Pooled ridge-init MinGRU v2 ("memorize, then regularize"), W 3,500: status log

Rule: `RULE_gru_v2.txt` (written 2026-10-23 18:40 PDT, before any v2 job). Branch insurance-selling-2026-09-19-e34-gru-v2
(from 2cc624c). CARC root `/home1/jc_905/pmf2_35` (W 3,000: `/home1/jc_905/pmf2`, the shared cluster log). Inputs read from
the old run (read-only): `/home1/jc_905/pmf35/data` (design copies, fold-3 ridge inits), `/home1/jc_905/pmf/bin`
(forecast_train md5 44ab6fff), env `/home1/jc_905/envs/harxhar-r13`.

## Log
- Cluster housekeeping (cancellations, feeders): logged in ../pooled_mingru_v2_2014start_w3000/STATUS.md.
- 2026-10-23 18:49 PDT G0 PASS (W 3,500, CPU 13353666; first attempt 13353638 failed on its own assertion: rows < 45,500
  are undated in the W 3,500 export, fixed by dating rows [39,000, 45,500) from the W 3,000 export, equal on every row
  dated in both). Folds: fit [0, 40,702) (89.5 % of W), [0, 43,952) (96.6 %), [1,702, 47,202); validation dates =
  folds14.json. Stage-1 inits recomputed byte for byte (full keep 295, HAR 12); TQ_0 HAR 0.116384 / 0.116390 /
  0.116387 / 0.116427 (L 8 / 20 / 48 / 115), full 0.107735 / 0.107740 / 0.107750 / 0.107789. Stage-2 kept columns full
  284 / 290 / 295; partly zeroed late-start columns 3 / 3 / 6. Shortcut-alone validation QLIKE (L20) by fold: HAR
  0.12640 / 0.13490 / 0.12991, full 0.11586 / 0.12709 / 0.12300.
- 2026-10-23 22:55 PDT: v2 root moved to /scratch1/jc_905/pmf2_35 (the /home1 quota; see the W 3,000 STATUS.md and
  RULE ADDENDUM 1). G4 (Stage 1, full set) submitted on L40S, nice 100: 13367193-98 (6 files, 81 jobs: core grid x lr
  {0.003, 0.01, 0.03} x 3 seeds, E 300; w0 1,702). GPU type of this window's Stage-1 group: L40S.
- 2026-10-24 00:08 PDT: G2 decided (W 3,000 STATUS.md, g2_decide.json): HAR E1 = 1,000, full set 300; lr {0.003, 0.01,
  0.03}. G4 HAR submitted on P100 (ADDENDUM 1b), nice 100: 13369121-42 (14 files, 81 jobs, E 1,000, w0 1,702). G4 full
  set: 13367193-95 completed, 13367196 running, 13367197-98 pending (L40S).
- 2026-10-24 00:15 PDT G4 full set final (stage1/s1_w3500_full_*; L40S b17-16, 13367193-98 all n_bad 0): pick H 64,
  L 8, lr 0.003, "memorizing": rho_300 0.1557 / 0.1584 / 0.1585 (3 seeds agree). (16, *) cells rho 0.697-0.699 (do not
  memorize); H 64 cells 0.158-0.163; H 256 ~ 0. lr 0.03 diverges everywhere; 0.003 the argmin in every cell. Same pick
  as W 3,000. Measured L40S 0.028 GPU-s a model-epoch at CONC 24 (H <= 64, L <= 20).
- 2026-10-24 00:16 PDT G5 + G6 full set (W 3,500), GPU type L40S (ADDENDUM 1a criterion: L40S cap 6 with none of our
  jobs on it; A40's 12 with 7 of our screen files queued; P100 with 24 of our files queued): s2_state.json (H 64, L 8,
  lr 0.003; 72 configurations); canary 13370080 (fold 1 = the truncated fit [0, 40,702), tau 1, wd_enc 3, E 20, seed
  20261012); screen 13370081-95 (15 files, 1,080 jobs) afterok on it; nice 100.
- 2026-10-24 02:30 PDT G4 HAR final (P100; 14 files incl. the 2 reruns from /project2 and .13 moved there; all n_bad 0;
  stage1/s1_w3500_*.csv / _pick.json hold both sets): pick H 256, L 8, lr 0.003, "memorizing": rho_1,000 0.4215 /
  0.4362 / 0.4281 (3 seeds agree). H 256 cells 0.426-0.429; H 64 0.750-0.753; H 16 0.876-0.879. Same pick as W 3,000.
- 2026-10-24 02:31 PDT G5 + G6 HAR (W 3,500), GPU type P100 (ADDENDUM 1b criterion: P100 36 - 23 of our gpu jobs + 2
  debug = 15, A40 12 + 2 = 14, L40S 6 - 5 = 1): s2_state.json baseline H 256, L 8, lr 0.003, tau {0, 0.3, 1, 3} x E
  {10, ..., 1,000} (480 jobs, est. 8.00 P100-h); canary 13373061 (debug) / 13373062 (gpu); screen 13373063-80 (gpu) and
  13373081-82 (debug) afterok on either; nice 100. Full-set W 3,500 screen: 5 files still pending on L40S (13371964-68,
  queued since 01:44; the group stays L40S).
- 2026-10-24 02:46 PDT: G5 PASS (HAR, W 3,500; P100): canaries 13373061 (debug) / 13373062 (gpu) n_bad 0 (fold 1 = the
  truncated fit [0, 40,702)); HAR screen running (13373081-82 done, the rest running / pending). Full-set screen: 10 of 15
  files done; 13371964-68 still pending on L40S (queued 01:44; the L40S partition is contended).
- 2026-10-24 06:22 PDT G6 screens (W 3,500) complete. Full set (L40S; the last 5 files waited 01:44-06:06 for the L40S,
  ran on b17-15): top 5 tau0.3_we3_E40 0.115296, tau1_we3_E40 0.115507, tau3_we3_E80 0.115525, tau0_we3_E40 0.115782,
  tau1_we15_E300 0.115797. HAR (P100, 20 files, all n_bad 0): top 5 tau3_we0_E40 0.125666, tau1_we0_E20 0.125732,
  tau3_we0_E20 0.125787, tau0.3_we0_E10 0.125823, tau1_we0_E10 0.125863 (stage2/s2_screen_*_round0.csv; the
  shortcut alone, L 20: full 0.12198, HAR 0.13040). Fresh seeds (75 jobs a set, nice 100): full 13377257-58 (L40S),
  HAR 13377259 (P100).
- 2026-10-24 06:39 PDT G6 HAR (W 3,500) round 0 (P100): fresh-seed pick tau3_we0_E40 0.125707 (tau0.3_we0_E10
  0.125837, tau1_we0_E20 0.125891, ...) at the tau high edge -> extension 1: tau 9 (120 jobs, tag v2s2hx1, gpu P100,
  nice 100): 13377396-400. Full-set fresh seeds 13377257-58 pending on L40S.
- 2026-10-24 06:56 PDT G6 full set (W 3,500) FINAL (L40S): fresh-seed C (stage2/s2_fresh_all_features_final.csv):
  tau1_we3_E40 0.115610 (pick), tau0_we3_E40 0.115630, tau3_we3_E80 0.115707, tau0.3_we3_E40 0.115745, tau1_we15_E300
  0.115904; interior on every axis (no extension). Reported (stage2/s2_report_all_features.json): bootstrap SE of
  C - C(pick) 0.00014-0.00032, seed-jackknife SE 0.00007-0.00016, 90 % MCS = the whole shortlist.
- 2026-10-24 06:57 PDT G7 full set (W 3,500), P100: 24 origins x 5 seeds at H 64, L 8, lr 0.003, tau 1 (wd 0.045674614 at
  S 356), wd_enc 3, E 40; 13377590 (gpu, nice 100).
- 2026-10-24 07:13 PDT G7 full set (W 3,500) done (13377590, P100 e21-11, 120 refits ok, finite, positive; wall 187 s
  at CONC 24; g7/): 1.56 P100-s a bar refit (E 40) -> every-bar walk 33,970 x 5 = 73.6 P100-h (projection).
  G6 HAR (W 3,500) extension 1 (tau 9) done (13377396-400, n_bad 0). Round-1 screen shortlist (stage2/
  s2_screen_baseline_round1.csv): tau9_we0_E80 0.125603, tau3_we0_E40, tau1_we0_E20, tau3_we0_E20, tau0.3_we0_E10;
  fresh seeds for tau9_we0_E80 (15 jobs, P100, nice 100): 13377956.
- 2026-10-24 07:29 PDT G6 HAR (W 3,500) FINAL (P100): the round-1 fresh-seed pick stays tau3_we0_E40 0.125707
  (tau9_we0_E80 0.125766, tau0.3_we0_E10 0.125837, tau1_we0_E20 0.125891, tau3_we0_E20 0.125927); interior (tau 3 <
  9). Reported (stage2/s2_report.json): bootstrap SE 0.00009-0.00026, jackknife SE 0.00004-0.00011, 90 % MCS = the
  whole shortlist. G7 HAR (W 3,500), P100: H 256, L 8, lr 0.003, tau 3 (wd 0.137023842), E 40: 13378173 (nice 100).
- 2026-10-24 09:20 PDT: RULE ADDENDUM 2 (walk of all four arms, placement, HOLD) -- see the W 3,000 STATUS.md. W 3,500
  walk files prepared (HAR 8 files on A40, full 4 on L40S), bar inits submitted (CPU 13379951, 13379952); no GPU job.
- 2026-10-24 10:15 PDT RELEASE (owner): HOLD lifted, the walk submitted by go_walk.sh (see the W 3,000 STATUS.md).
- 2026-10-24 ~10:20 PDT WALK SUBMITTED: HAR W 3,500 A40 13380643-50 (8; lane ends 13380647-50), full W 3,500 L40S
  13380651-54 (4; lane ends 13380653-54), nice 100; ids in walk/walk_ids_<set>.txt; slice check 13380579.
- 2026-10-24 12:45 PDT ADDENDUM 3: HAR W 3,500 re-pinned to P100, its 8 files cancelled while pending and replaced by
  45 pieces (20 submitted: 13382912-31, 5 of them on the debug P100s); full W 3,500 (L40S) 4 files replaced by 12
  pieces (4 submitted: 13382932-35). Unsubmitted pieces in walk/pieces_unsubmitted_<set>.csv.
- 2026-10-24 14:06 PDT: debug pieces 13382912 / 13382913 (HAR W 3,500 d00 / d01, P100 e23-02) ended 13:57 with exit
  143 after 38 min: PREEMPTED, not timed out -- the debug partition has PreemptMode=CANCEL (GraceTime 10 s) and
  another user's jobs started on e23-02 at the same seconds; SIGTERM is not trapped, so no salvage, but the 15-min
  partial copies hold 264 of 350 refits each. The 86 missing refits of each were written to new remainder files
  (pieces/v2w_baseline.000.d00r.txt, .d01r.txt) and submitted on gpu P100, 30 min: 13383630, 13383631 (walk_ids).
  No further debug pieces (preemption); the 3 pending (13382914-16) stay, their remainders follow the same way.
  gpu partition: 97 of our jobs.
- 2026-10-24 15:30-15:33 PDT CORRECTION: the four "preempted" debug pieces (13382912-15) were NOT preempted. Every
  one ended at its trainer's end (elapsed 2,265-2,271 s, the same each time) with exit 143 inside the job script: the
  15-min partial-copy loop (PARTIAL_S, added 2026-10-24) is stopped with kill + wait, and under set -e the wait's
  status 143 ended the script before the outputs were copied back (reproduced locally). Other users' debug jobs
  started at those seconds because ours had just ended. Consequences: every walk job submitted before 15:32 ends as
  FAILED 143 without POST, keeping only its last 15-min partial copy (it loses at most its last 15 min of refits).
  Fix (cluster/pooled_mingru_v2_gpu_carc.sbatch, md5 802e37ae): "|| true" on that kill and wait, applied on CARC
  with an explicit sed (15:31). All 63 pending walk jobs (submitted with the old script) cancelled while pending
  and resubmitted unchanged with the fixed script (walk_cancelled_r3_fix.log; walk_ids "resubmitted" lines),
  except HAR W 3,000 piece 023.p04 (back to the unsubmitted list: QOS 100). New remainder files for 13382914/15
  (d02r, d03r) submitted; d04 (13382916) moved from debug to gpu P100: no walk work is left on debug. gpu partition:
  100 of ours (36 running with the old script, 51 P100 + 13 A40 pending with the new one). The 36 running
  (HAR W 3,000 originals 000-012 and 23 pieces) will end FAILED 143: their missing tail refits are rerun from
  walk-merge's remainder files as each finishes.
- 2026-10-25 12:16 PDT ADDENDUM 3f: full W 3,500 HELD for the Hoffman2 L40S validation (owner: "validate on hoffman").
  Its 4 pending gpu A40 pieces 13385646-49 cancelled (walk/walk_cancelled_all_features.txt); its 18 debug A40 pieces
  (file 003) withdrawn, never submitted (removed from the W 3,000 dir's walk/dbg_pieces_all_features.csv); all 12 pieces
  listed in walk/pieces_unsubmitted_all_features.csv as HELD. 0 refits done; nothing new submitted for it.
- 2026-10-25 13:24 PDT ADDENDUM 3g (hold released) / 3d choice: CARC L40S in the gpu partition: 22 GPUs, 14 in use, 0
  usable at cpu 4 / 24G (the 8 free ones sit on nodes without 4 idle CPUs / 24G); ours running 34 of 36, L40S 0 of 6 ->
  L40S concurrency 0 -> P100 (the rule: L40S if >= 1). G5 canary of full W 3,500 on a debug P100: 13446389 (120 refits,
  rows 50,777-50,800 x 5 seeds, pieces/canary/v2w_all_features.000.c24.txt, cpu 2 / 16G, CONC 24).
- 2026-10-25 14:25 PDT ADDENDUM 3d / 3g: G5 canary of full W 3,500 on P100 13446389 PASS (debug e23-02, COMPLETED 3:57;
  120 jobs status ok, finite, positive, n_bad 0; trainer wall 182 s = 1.52 s a refit, the G7 P100 rate 1.56). RE-PIN:
  the whole arm runs on P100 (one type): its 12 pieces (14,150-14,155 refits each, est at 1.56 s = 6.1 h, time limit
  est x 1.3 + 30 = 509 min, cpu 2 / 16G, CONC 24, nice 100 behind both W 3,000 arms) are listed in
  walk/pieces_unsubmitted_all_features.csv and go into the gpu partition as our 100-job cap frees after the W 3,000
  pieces. Its walk-merge runs with --gputype P100.
- 2026-10-26 16:43 PDT: W 3,500 top-up behind the W 3,000 queue (nice 100): HAR 13471944-45 (003.p00-p01, cpu 2 / 8G).
  Full W 3,500 P100 pieces not yet submitted: the guard skipped 000.p00-p01 because walk_ids_all_features.txt still
  holds their cancelled L40S / A40 lines; the next session matches P100 lines only.
- 2026-10-26 17:14 PDT: first full W 3,500 P100 pieces 13473343 / 13473345 (000.p00 / 000.p01, nice 100, 509 min, cpu 2
  / 16G); HAR W 3,500 13473344 (003.p02). gpu 99 (32 running P100, 67 pending). No FAILED since 16:40.
- 2026-10-26 17:45 PDT: W 3,500 top-up (nice 100): HAR 13474336 / 13474338 (003.p03 / p04), full 13474337 (000.p02).
  gpu 99 (34 running P100, 65 pending). No FAILED since 17:10.
- 2026-10-26 18:26 PDT: full W 3,500 13475516 (001.p00, nice 100). gpu 99 (36 running P100, 63 pending). No FAILED.
- 2026-10-26 20:57 PDT: W 3,500 top-up (nice 100): HAR 13478019 / 21 / 23 (004.p00-p02), full 13478020 / 22 / 24 (001.p01, 001.p02, 002.p00). gpu 99 (36 running P100, 63 pending). No FAILED since 20:10.
- 2026-10-26 21:38 PDT: W 3,500 top-up (nice 100): HAR 13478603 / 05 / 07 / 09 (004.p03-005.p01), full 13478604 / 06 / 08 (002.p01, 002.p02, 003.p00). gpu 99 (35 running P100, 64 pending). No FAILED since 20:50.
