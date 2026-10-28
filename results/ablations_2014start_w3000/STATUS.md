# Augmented-HAR benchmarks and predictor-group ablations (status)

Rule: `RULE.txt` (committed before any job).  Code: `experiments/ablate_arms.py` (arms), `experiments/ablate_design.py`
(gate 0, the new-series design, the linear arms' caches, late-start windows), `experiments/ablate_linear.py` (linear
walks through `pool_linear_pretune.py`, gate L, merge), `experiments/ablate_trees.py` (tree walks, gates T1 / T2,
projection, merge), `cluster/ablate/` (Hoffman2 job scripts: `py.sh`, `prep0.sh`, `prep1.sh`, `lingate.sh`,
`tgate.sh`, `project.sh`, `pool.sh`, `merge.sh`).  Task files: `tasks/trees.txt` (470 lines), `tasks/linear.txt`
(135 lines; 1-14 = gate L).  Staging: `/u/scratch/j/jamesdc1/abl` on Hoffman2.

| Time (PDT) | Event |
|---|---|
| 2026-10-25 13:40 | RULE.txt written; code smoke-tested locally on a synthetic full-size design (tree walk / gate / projection code paths, linear walks on column-subset and classical caches); the abl_new design built locally as a test (84 columns; y and b_oos md5-equal to the templates') |
| 13:45 | RULE.txt and code committed (5f0cbaf) and pushed; no job before it |
| 13:48 | code = git archive of 5f0cbaf -> /u/scratch/j/jamesdc1/abl/code (md5 cc2b317e...); design caches copied from the three-way comparison's Hoffman2 staging (t3t/carc/designs/RTH13), md5 = the templates' (X 374bea97..., 05da4b5b...; y, b_oos, dates) |
| 13:51 | first submission: every sbatch refused ("DOS line breaks"): the laptop's git archive applied core.autocrlf (CRLF scripts); the two UGE jobs it had queued (15272771 tA_u6140, 15272772 pool, never started) deleted; the CRLF tree moved aside (code_crlf_unused) and the commit re-archived with core.eol=lf (md5 ee801277..., data md5-identical) |
| 13:53 | submitted (Slurm campus, QOS campus24; UGE): see the job table |
| 14:09 | poll: gate 0a (499293) FAIL only on design.json md5 (the copies record the three-way comparison's rebuild; X / y / b_oos / dates md5-equal, HAR subset = baseline X, E-mini availability constant 1); gate 0b (499294) FAIL only on sumret4's six value averages (<= 2.8e-14 absolute; numpy power through the container's libm); HAR, calendar, sumret, sumpret2 bit-identical; abl_new built (84 columns, its statistics equal the laptop build's) |
| 14:10 | scancel 499295-499317 (dependants of the failed gates; never started); RULE.txt addendum A |
| 14:12 | addendum A committed (8923fd9) and pushed; its four changed files extracted into the Hoffman2 code tree from the LF archive of 8923fd9 (md5 4c8254e9...); tar -d: no content difference between the tree and the archive |
| 14:13 | resubmitted unchanged on 8923fd9 (second job table); UGE 15272780 / 15272781 kept |

First submission (13:53; 499293 / 499294 failed gate 0, 499295-499317 cancelled unstarted):

| Job | Name | What | Depends on |
|---|---|---|---|
| 499293 | abl_p0 | gate 0a (md5, HAR subset = baseline X, E-mini availability constant) | - |
| 499294 | abl_p1 | abl_new design + gate 0b, the linear arms' caches, late-start windows | - |
| 499295 | abl_lg | gate L: the final run's 14 linear walks by this code path, bit for bit (Gold 6240) | p1 |
| 499296 | abl_lp | linear pool, 16 one-thread workers (Gold 6240) | lg |
| 499297 / 499298 | abl_tA_s6240 / abl_tB_s6240 | tree gates T1 + T2 pieces, LightGBM / XGBoost (A), forest (B), Slurm Gold 6240 | p0 |
| 499299 / 499300 | abl_tA_s6736p / abl_tB_s6736p | the same, Slurm Xeon 6736P | p0 |
| 499301 / 499302 | abl_tA_s6442y / abl_tB_s6442y | the same, Slurm Gold 6442Y | p0 |
| 499303 | abl_proj | T2 across models, projection (cap 15,000 CPU-hours) -> projection.done / rf_skip.txt | all tA / tB (afterany) |
| 499304-499306 | abl_pool_s6736p | tree pools, 9 workers x 8 threads each | tA_s6736p (afterany) |
| 499307-499308 | abl_pool_s6442y | tree pools, 6 workers each | tA_s6442y (afterany) |
| 499309-499317 | abl_pool_s6240 | tree pools, 4 workers each | tA_s6240 (afterany) |
| 15272780 | abl_tA_u6140 | tree gate A on UGE Gold 6140 | - |
| 15272781 | abl_pool_u6140 | UGE tree pool, 22 tasks x 1 worker (8 slots; LightGBM / XGBoost only) | tA_u6140 |

Second submission (14:13, code 8923fd9), the same jobs and dependencies: p0 499345, p1 499346, lg 499347, lp 499348,
tA_s6240 499349, tB_s6240 499350, tA_s6736p 499351, tB_s6736p 499352, tA_s6442y 499353, tB_s6442y 499354, proj 499355,
pool_s6736p 499356-499358, pool_s6442y 499359-499360, pool_s6240 499361-499369; UGE tA_u6140 15272780, pool 15272781.

| Time (PDT) | Event |
|---|---|
| 14:13-14:27 | p0 499345 PASS (gate 0a); p1 499346: abl_new rebuilt, gate 0b PASS, 21 linear caches, latestart_w3000.csv, then exit 3 on the late-start cross-check (vix3m k = 1, 5, 25: UNOBSERVED_NEUTRAL zeros after the rule's prefix; addendum B (a)) |
| 14:16-14:27 | tree gates T1: PASS for LightGBM, XGBoost and the forest on all three Slurm models (Gold 6240, Gold 6442Y, Xeon 6736P): LightGBM / XGBoost 20 of 20 origins text-identical to the templates for the full set and the HAR benchmark, the forest within 1e-14; projection 499355: 12,255 CPU-hours (Gold 6240 speed), no forest arm dropped -> projection.done |
| 14:16 | tree pools started (9 x Gold 6240 x 4, 2 x 6442Y x 6, 3 x 6736P x 9 = 75 workers, 600 CPUs); first round: XGBoost / LightGBM lines (the forest became eligible at 14:27) |
| 14:29 | lg / lp resubmitted without the prep1 dependency: 499465 / 499466 (499347 / 499348 cancelled unstarted) |
| 14:30 | UGE gate replaced by an 8-slot job 15273084 (+ pool 15273085); T2 job 499467 over the three Slurm gate dirs; RULE.txt addendum B |
| 14:43 | gate L 499465 PASS on Gold 6240: the seven final linear walks (ridge, lasso, elastic net on the full cache; the four HAR-benchmark walks on the 12-column subset of the full cache) reproduce w3000/forecasts.parquet's pred_adj and true_adj bit for bit on all 33,997 rows -> gate/passed_lin; linear pool 499466 started 14:43 (16 workers) |
| 14:30 | T2 499467 (runs/gate2_all.json): LightGBM and XGBoost lo_rm first-10 forecasts bit-identical on Gold 6240, Gold 6442Y, Xeon 6736P; the forest differs across them by at most 1.8e-15 relative (recorded; a forest piece stays on one CPU model) |
| 14:45 | UGE gate 15273084 PASS (Gold 6140, LightGBM and XGBoost text-identical to the templates); UGE pool 15273085 running (22 x 8 slots) |
| 15:20 | linear pool 499466 done: all 121 ablation walks (16 group arms x ridge / lasso / elastic net in 1-4 chunks, 4 classical arms x 4 models, add_rm OLS) exit 0, no failed claim |
| 15:37 | tree pools: 85 lines claimed (78 XGBoost, 6 LightGBM, 1 forest), none finished yet (XGBoost lo arms 4.6-4.8 s a refit on Gold 6240); linear merge 499726 |
| 15:53 | linear merge 499726 failed on its own check: the walk tables' true_adj (read back by pandas' default float parser, as the templates') differs from y.npy in the last bits; check changed to: equal to the template's true_adj exactly, to y.npy within 1e-14 relative (ablate_linear.py); merge resubmitted |
| 15:54 | linear merge 499791 (code 8923fd9 + ablate_linear.py of 9a4ede6): 72 walks (65 ablation + the 7 gate reproductions), 33,997 rows each, no walk missing |
| 16:09 | fetched to the laptop: runs/lin/forecasts_linear.parquet (the 7 reproductions = the templates bit for bit, checked again), gate records (gate/, runs/), late-start table (latestart/latestart_w3000.csv); the 65 ablation walks stored compact as forecasts/linear_wide.parquet (2014-01-02 on, one pred_adj column per "<model>\|<arm>") |
| 16:15 | experiments/ablate_score.py on the linear arms and the templates: checks PASS (scored set = Table 1's 33,527 bars; Table 1's QLIKE / MSE / 16:00 QLIKE and full-vs-HAR DM reproduced) -> scores/ (interim, linear only) |
| 17:12 | tree pools: 53 of 470 lines done (XGBoost 48, LightGBM 4, forest 1), 35 forest lines running; seconds a refit (8 threads): XGBoost lo arms 3.4-4.7, LightGBM lo arms 2.1-3.0, the forest add_vd 1.5 (6442Y); no failed line |
| 17:25 | final tree merge prepared: experiments/ablate_widen.py (T2 over the four gate dirs -> gate2_final.json, every complete arm merged, compact trees_wide.parquet) and cluster/ablate/merge.sh calling it (neither read by a running job) |
| 17:40 | 71 of 470 tree lines done (XGBoost 64, LightGBM 6, forest 1), 61 running, 0 failed; the laptop's local poll timer was stopped by Claude Code (low laptop memory), reported to the coordinator; the pools run unattended |
| 22:55 | 177 of 470 tree lines done (forest 72, XGBoost 99, LightGBM 6), 56 running, 0 failed; workers exiting early on a pool.sh eligibility race (addendum C); pool2.sh written |
| 22:57 | pool2.sh installed (new file, c3df1f7; md5 of its archive 97191d64...); submitted: Slurm 500642-500648 (Xeon 6736P, 9 workers; 5 running at once with the old jobs, the rest queued on the 16-node cap) and 500649-500651 (Gold 6442Y, 6 workers, queued); UGE 15279049 (22 x 8 slots, queued behind the 200-slot cap); the old pools keep their running lines |
| 23:10 | scorer: live-window subsamples of the late groups (sent from 2022-04-19, 508 days; iv = VIX from 2018-04-19, 1,506 days; ivb = VIX and VIX3M from 2021-07-06, 706 days; vd 2022-11-16 .. 2023-08-31, 196 days), Panel E of the table; experiments/ablate_summary.py (markdown tables for SUMMARY.md) |
| 2026-10-26 13:02 | (no session 03:00-13:00: a usage limit stopped every agent) all 470 tree lines done, 0 failed, 0 take-overs; no pool left in either queue |
| 13:02 | final merge 504404 (merge.sh -> ablate_widen.py): T2 over the four gate dirs (runs/gate2_final.json: LightGBM / XGBoost bit-identical on Gold 6140, 6240, 6442Y, Xeon 6736P; forest <= 1.8e-15 relative), 41 tree arms merged (true_adj / B = the templates', CPU models per arm in runs/tree_merge/*.json), none missing -> trees_wide.parquet (md5 e2d7196a...) |
| 13:10 | scored on the laptop: 119 (class, arm) forecasts, checks PASS (Table 1's 33,527 bars, its QLIKE / MSE / 16:00 QLIKE and full-vs-HAR DM reproduced); live-window subsamples; tab:rh_ablation and \rhab macros regenerated (compile-tested); draft results paragraphs written.  All numbers on the pre-patch FOMC design (FOMC rows preliminary) |
| 14:50 | coordinator: the author adopted the FOMC-patched design (fomcfix cell B); rerun the linear arms and the FOMC tree arms on it.  fomcfix merged into this branch (7abfaea); RULE.txt addendum D (8d5ddea) before any job |
| 14:57 | code 6401ad5 (addendum D: checkfix, --cell, --ref-full, --template-gb, pool3.sh, prepfix / tgatefix / merge_fix) archived (md5 a948dc68...) and installed as /u/scratch/j/jamesdc1/abl/code (the previous tree kept as code_8923fd9; no job of this run was queued or running) |
| 15:00 | cell-B full cache piped CARC (/home1/jc_905/fomcfix_lin/runs/B/_work/design_all_features, read-only session) -> laptop -> Hoffman2 designs_fix/full: X md5 725eb201... (= fomcfix G3), y / b_oos / dates md5 = the final cache's, design.json c7ab33ce... (equal at both ends) |
| 15:00 | Slurm: prepfix 506296, tgatefix 506297, linfix 506298 / 506299 (2 x 16 one-thread workers, all 135 lines), treefix 506300 (Xeon 6736P, 9 workers) / 506301 / 506302 (Gold 6240, 4 workers) on tasks/trees_fomcfix.txt (15 lines) |
| 15:04 | gate F0 PASS (exactly the 12 FOMC proximity value averages differ; 21 linear arm caches rebuilt); gate FT PASS (Gold 6240: LightGBM / XGBoost full set 20 of 20 origins text-identical to fomcfix's forecasts_gb, HAR to the templates) |
| 15:37 | linear fix pools done: 135 of 135 walks, 0 failed; merge 507331: gate LF PASS (the 10 patched full-set walks = fomcfix lin_B bit for bit, the 4 HAR walks = w3000), 72 walks merged |
| 15:55 | fetched; forecasts/linear_wide_fix.parquet; the 41 arms without a FOMC column reproduce their pre-patch forecasts bit for bit, the 24 with one changed (gate/linear_fix_invariance.json); scores_fomcfix/ (design fix: patched full rows, checks PASS incl. fomcfix cell-B scores to 1e-12) -- interim, tree add_fomc and the forest's patched full set pending |
| 17:15 | coordinator (author's request): add the forest's lo_fomc arm; budget about lo_rm's, report first if the projection exceeds 2,500 CPU-hours |
| 17:22 | RULE.txt addendum E (6550c17, pushed before any job): 101 pieces, projection 2,190 CPU-hours (lo_rm's measured cost x 528 / 480; 2,474 all on Gold 6240); RF_ARMS + lo_fomc; installed on Hoffman2 (code/COMMIT) |
| 17:24 | Slurm 509070-509083 (8 x Xeon 6736P, 4 x Gold 6442Y, 2 x Gold 6240 pools on tasks/trees_rf_lofomc.txt); only the two 6240 ones started, the rest estimated to start 21:12 .. 06:28 |
| 17:26 | 509070-509081 cancelled before they started (sacct Start None); 509117-509125 on idle Gold 6240 nodes; UGE: forest T1 gate on Gold 6140 (15293201) and a pool held on it (15293203); addendum E (f) |
| 17:41 | 509467 / 509468 (two more Gold 6240 pools on the nodes the finished add_fomc pools freed): 13 jobs, 52 workers, all Gold 6240 |
| 17:5x | add_fomc fix pools 506300-506302 done: 15 of 15 lines, 0 failed |
| 17:59 | forest lo_fomc at 47.1 s a refit on Gold 6240 (52 pieces, 1,890 refits), against lo_rm's ~28: ~3,550 CPU-hours, over the 2,500 cap -- the projection of addendum E (c) used the wrong comparator (lo_fomc drops cheap calendar columns, lo_rm the costly realized moments).  lo_fomc design identity: X on its 528 columns md5 774b1fc1... in both caches |
| 18:00 | UGE 15293203 (the Gold 6140 pool, not started) on user hold; coordinator told, with options (continue on 6240 ~3,550; unstarted pieces to 6442Y / 6736P ~2,900-3,000; stop); default: run inside the allowed 2,500, cancel before the cap if no decision; addendum E (g), (h) |
| 18:17 | merge 510041 (merge_fix.sh WHAT=trees): T2 over the four gate dirs (Gold 6140 now with the forest: max rel 1.79e-15, as before; runs/gate2_final.json), add_fomc merged for LightGBM (3 pieces, Xeon 6736P), XGBoost (3, 6736P + Gold 6240), forest (9, 6736P + Gold 6240), 33,970 rows each -> trees_wide.parquet (md5 79c8049a...) |
| 18:36 | fetched -> forecasts/trees_wide_fix.parquet, runs/tree_merge_fix/; scored --design fix with the forest's patched full set (fomcfix ed43a18 results/pooled_rf_skx_2014start_fomcfix_w3000/forecasts.parquet, md5 4dff304d...): 122 forecasts on 33,527 bars, checks PASS (forest full vs HAR -11.3 %, DM -9.74 = fomcfix's); table / macros regenerated, draft on the patched panel, compile-tested |
| 19:00 | coordinator (author's decision): stop the forest lo_fomc arm; new arm add_rm_fomc (HAR + RM + FOMC) on the patched design, every class |
| 19:06 | forest lo_fomc STOPPED: 13 Slurm pools cancelled, UGE 15293203 deleted (never started); 678.1 allocated CPU-hours; 52 of 101 pieces begun, 0 finished, 6,322 of 33,970 refits checkpointed (kept, resumable on Gold 6240); RULE.txt addendum E (i) |
| 19:15 | RULE.txt addendum F (add_rm_fomc; requested by the author after the channel results, not pre-registered): projection <= ~760 CPU-hours (forest <= 492 on Gold 6240) |
| 19:14 | add_rm_fomc submitted (code fced823 installed on Hoffman2, code/COMMIT; jobids.txt slurm15), all on Hoffman2 Slurm campus (sbatch via hoffman2.oarc.ucla.edu), Gold 6240: prep 510532 (designs_fix/lin/add_rm_fomc; gone from the queue 40 s later -- check logs/prep_rmfomc.510532.log), linear pool 510533 (18 one-thread workers, tasks/linear_rmfomc.txt, after prep; PENDING on Priority at 19:15), tree pools 510534-510542 (9 x 4 workers, tasks/trees_rmfomc.txt: forest 25, XGBoost 5, LightGBM 4 pieces; RUNNING at 19:15 on n1807, n1808, n1810-n1812, n1815-n1818) |
| 19:17 | generator for the new row / macros (407d874), dry-run on stand-in data passed (81 EVENT days; ORDINARY-day full vs HAR = channels item1 (c) exactly), stand-ins removed |

## State at 19:21 PDT, 26 October (laptop reboot; cluster jobs keep running)

**add_rm_fomc (RULE.txt addendum F), Hoffman2 Slurm campus, staging /u/scratch/j/jamesdc1/abl:**

| job | what | state at 19:15 |
|---|---|---|
| 510532 | prep_rmfomc.sh: designs_fix/lin/add_rm_fomc | left the queue (check its log) |
| 510533 | pool3.sh KIND=lin: 18 walks (gate L's 14 + ridge / lasso / enet / OLS add_rm_fomc) -> runs/lin_rmfomc, claims runs/claims_lin_rmfomc | pending (afterok:510532, Priority) |
| 510534-510542 | pool3.sh KIND=trees: 34 pieces -> runs/trees_rmfomc, claims runs/claims_trees_rmfomc | running |

Expected: tree pieces 2.2-3.1 h on Gold 6240, so the trees finish ~22:30-23:00; the linear walks ~1 h after 510533 starts.

**Steps to finish** (B=/u/scratch/j/jamesdc1/abl; sbatch on hoffman2.oarc.ucla.edu; S="sbatch --parsable -p campus --qos=campus24"):
1. Poll: `squeue -u jamesdc1`; `ls -d $B/runs/claims_lin_rmfomc/*/done | wc -l` (18 expected) and `ls -d $B/runs/claims_trees_rmfomc/*/done | wc -l` (34); no `failed`; `python3 $B/tmp/rate_rmfomc_20261026.py` (speed, if uploaded).
2. Linear merge when 18 are done: `$S -C intel-gold-6240 -c 1 --mem=16G -t 1:00:00 -J abl_mlinrmf --export=ALL,WHAT=lin -o $B/logs/mlinrmf.%j.log $B/code/cluster/ablate/merge_rmfomc.sh` -> gate/gateLF_rmfomc.json (must PASS), runs/lin_rmfomc/linear_wide.parquet (4 columns ridge / lasso / enet / ols|add_rm_fomc).
3. Tree merge when 34 are done: the same with `-J abl_mtrrmf --export=ALL,WHAT=trees -o $B/logs/mtrrmf.%j.log` -> runs/trees_rmfomc/trees_wide.parquet (lgbm / rf_skx / xgb|add_rm_fomc), runs/trees_rmfomc/forecasts/*.json.
4. Fetch only the compact files (the laptop's C: is nearly full): runs/lin_rmfomc/linear_wide.parquet -> forecasts/linear_wide_rmfomc.parquet; runs/trees_rmfomc/trees_wide.parquet -> forecasts/trees_wide_rmfomc.parquet; gate/gateLF_rmfomc.json -> gate/; runs/trees_rmfomc/forecasts/*.json -> runs/tree_merge_rmfomc/.
5. Score: `python experiments/ablate_score.py --design fix` (the forest's patched full set by default) -> scores_fomcfix/ (checks PASS; day_types: 81 EVENT days).
6. Outputs: `python writeup/make_rehaul_ablations.py` (Panel D row "vs HAR + RM + FOMC", \rhab<Cls>FullVsAddRmFomc{All,Ord,Ev}{Pct,Dm,P}, AddRmFomcVsHar, AddRmFomcVsOlsHar, FullVsOlsAddRmFomc, OlsAddRmFomcVsHar, NEventDays); compile-test; a labelled sentence in writeup/drafts/2026-10-25_ablations.tex; STATUS; commit, push; report full vs HAR+RM+FOMC, HAR+RM+FOMC vs HAR, vs OLS-HAR (all / ORDINARY / EVENT).

**Forest lo_fomc (stopped 19:06, addendum E (i)):** checkpoints in $B/runs/trees_lofomc/pieces/lo_fomc_rf_skx_<i>of101.{parquet,ckpt.json} (52 pieces, 6,322 refits, all Gold 6240), claims $B/runs/claims_rf_lofomc; resumable on Gold 6240 only (pool3.sh KIND=trees TASKS=tasks/trees_rf_lofomc.txt CL=$B/runs/claims_rf_lofomc TOUT=$B/runs/trees_lofomc TFULL=$B/designs/full; a stale claim is taken over by a worker on the same CPU model).  Not to be resumed without a new decision.
| 19:48 | after the laptop reboot (read-only poll): prep 510532 COMPLETED (19:14:45, designs_fix/lin/add_rm_fomc, 132 columns); linear pool 510533 COMPLETED (19:15-19:31, 18 of 18 walks, 0 failed); tree pools 510534-510542 running, 34 of 34 pieces in progress, 0 failed: forest 6.66 s a refit (arm ~503 CPU-hours, pieces 2.5 h), LightGBM 1.16 (~88), XGBoost 1.27 (~96) on Gold 6240 -- total ~690 CPU-hours; trees expected ~22:00 |
| 20:04 | linear merge 510755 (merge_rmfomc.sh WHAT=lin): gate LF again PASS (gate/gateLF_rmfomc.json: the 3 patched full-set walks = fomcfix lin_B, the 4 HAR walks = w3000, bit for bit, Gold 6240); 11 (arm, model) walks merged; compact linear_wide.parquet (4 add_rm_fomc walks x 33,970 bars, md5 aced5275...) |
| 20:21 | fetched -> forecasts/linear_wide_rmfomc.parquet, gate/gateLF_rmfomc.json; trees 0 of 34 done (pieces ~45 % through), expected ~21:45-22:00 |
| 22:23 | last tree piece done (510540): 34 of 34, 0 failed, 0 take-overs; measured forest 500 CPU-hours (6.62 s a refit), LightGBM 87, XGBoost 94, all Gold 6240 |
| 22:24 | tree merge 512306 (merge_rmfomc.sh WHAT=trees): T2 as before (LightGBM / XGBoost bit-identical across models; forest 1.79e-15); add_rm_fomc merged for the three trees (4 / 5 / 25 pieces, Gold 6240 only, 33,970 rows each) -> trees_wide.parquet (md5 e1741bd4...); fetched -> forecasts/trees_wide_rmfomc.parquet, runs/tree_merge_rmfomc/ |
| 22:30 | scored --design fix: 129 forecasts on 33,527 bars, checks PASS; day types 81 EVENT / 2,498 ORDINARY days (1,053 / 32,474 bars); tab:rh_ablation Panel D rows now "\quad vs HAR + RM / + FOMC / + RM + FOMC*" (the * row and note: requested by the author after the results by day type, not pre-specified); \rhab add_rm_fomc macros (generator asserts every macro unique); draft paragraph; compile-tested |
| 22:45 | coordinator (author, "let's do 3 on the clusters"): arms add_rm_fomc_<g> for the six remaining groups, LightGBM / XGBoost / ridge / lasso / elastic net, no forest; budget 2,500 without asking |
| 23:40 | RULE.txt addendum G (requested after the HAR+RM+FOMC results; not pre-registered): 6 arms, 73 tree pieces + 32 linear walks; projection 1,579 CPU-hours by the cost model, <= ~2,000 with margins; group membership checked (the constituents' order flow is in liq) |
| 23:40 | addendum G submitted (code b297046 installed, code/COMMIT; jobids.txt slurm18 / uge7 / slurm19): prep 513108 (done 23:40:40, designs_fix/lin/add_rm_fomc_<g>), linear pools 513109 / 513110 (16 workers each, 32 walks, Gold 6240), tree pools 513111-513123 (13 x 4 workers, Gold 6240) and 3 more (slurm19, start as the linear pools free the 16-node cap), UGE 15298489 (25 x 8 slots, Gold 6140; 11 running at 23:41).  At 23:41: 67 of 73 tree pieces and all 32 linear walks claimed.  Expected: linear ~00:15, trees ~03:00-03:30 (pieces 2.5-3.4 h) |

## State at 23:45 PDT, 26 October: addendum G (beyond HAR + RM + FOMC) running

**Steps to finish** (B=/u/scratch/j/jamesdc1/abl; Slurm on hoffman2.oarc.ucla.edu; S="sbatch --parsable -p campus --qos=campus24"):
1. Poll: `ls -d $B/runs/claims_lin_beyond/*/done | wc -l` (32) and `ls -d $B/runs/claims_trees_beyond/*/done | wc -l` (73), no `failed`.
2. `$S -C intel-gold-6240 -c 1 --mem=16G -t 1:00:00 -J abl_mlinbey --export=ALL,WHAT=lin -o $B/logs/mlinbey.%j.log $B/code/cluster/ablate/merge_beyond.sh` (gate/gateLF_beyond.json must PASS) and the same with `-J abl_mtrbey --export=ALL,WHAT=trees -o $B/logs/mtrbey.%j.log`.
3. Fetch only runs/lin_beyond/linear_wide.parquet -> forecasts/linear_wide_beyond.parquet, runs/trees_beyond/trees_wide.parquet -> forecasts/trees_wide_beyond.parquet, gate/gateLF_beyond.json, runs/trees_beyond/forecasts/*.json -> runs/tree_merge_beyond/.
4. `python experiments/ablate_score.py --design fix`; `python writeup/make_rehaul_ablations.py` (writes writeup/generated/rehaul_ablation_beyond_table.tex, tab:rh_ablation_beyond, and the \rhab beyond macros); compile-test; commit; push; report (i) base + g vs base, (ii) full vs base + g, (iii) the overlap ratio, every class, every bar / ORDINARY / EVENT.
| 03:24 | (27 Oct) addendum G done: 73 of 73 tree pieces (Gold 6240 and UGE Gold 6140; each arm's pieces from one model or several, T2 bit-identical), 32 of 32 linear walks, 0 failed; merges 514146 (gate LF again PASS: gate/gateLF_beyond.json; 18 walks compact) and 514147 (12 (class, arm) pairs, 33,970 rows each); fetched -> forecasts/linear_wide_beyond.parquet (md5 873d7531...), forecasts/trees_wide_beyond.parquet (md5 76ac1aba...), runs/tree_merge_beyond/ |
| 03:35 | scored --design fix: 159 forecasts, checks PASS; tab:rh_ablation_beyond (writeup/generated/rehaul_ablation_beyond_table.tex) and the \rhab beyond macros (the overlap ratio only where the full set's change against the base is significant); compile-tested |
| (27 Oct) | coordinator: the Hoffman2 leave-one-out campaign was proposed and cancelled before anything was launched (one read-only poll, 12:08) |
| (27 Oct) | coordinator (author approved): the within-block ladder (results/beyond_trees_2026-10-27/RULE.txt, 7bd70bb cherry-picked); RULE.txt addendum I: rankings recorded, 10 rungs, BASE gate per CPU model, 94 pieces, projection 1,427 CPU-hours (<= ~1,800) |
| 12:31 | addendum I submitted (code f7d71b2 installed, code/COMMIT; jobids.txt slurm21 / uge8): BASE gates 518246 (Gold 6442Y), 518251 (Xeon 6736P), 518254 (Gold 6240) running, UGE 15308728 (Gold 6140) queued; pools, each after its model's gate (afterok / hold_jid): 518247-518250 (6442Y, 4 x 6 workers), 518252-518253 (6736P, 2 x 9), 518255-518262 (6240, 8 x 4), UGE 15308729 (25 x 1); 94 pieces; expected ~15:00-15:30 |

## State at 12:35 PDT, 27 October: addendum I (the ladder) running

**Steps to finish** (B=/u/scratch/j/jamesdc1/abl; Slurm on hoffman2.oarc.ucla.edu):
1. Gates: `cat $B/gate/rb/*/gatebase.json` (pass per model; a failed model's pools never start); then `ls -d $B/runs/claims_trees_ladder/*/done | wc -l` (94), no `failed`.
2. `sbatch --parsable -p campus --qos=campus24 -C intel-gold-6240 -c 1 --mem=16G -t 1:00:00 -J abl_mlad -o $B/logs/mlad.%j.log $B/code/cluster/ablate/merge_ladder.sh`
3. Fetch only runs/trees_ladder/trees_wide.parquet -> forecasts/trees_wide_ladder.parquet, runs/trees_ladder/forecasts/*.json -> runs/tree_merge_ladder/, gate/rb/*/gatebase.json -> gate/rb/.
4. `python experiments/ablate_score.py --design fix`; `python writeup/make_rehaul_ablations.py`; `python writeup/make_rehaul_beyond_ladder.py` (tab:rh_beyond_ladder, \rhbyL macros, the section 2 readings); compile-test; commit; push; report.
| 12:32-12:37 | the three Slurm BASE gates 518246 / 518251 / 518254 FAILED (exit 1): the walks ran (7 BASE refits per model and class, pieces written), then ablate_trees.py gatebase raised KeyError 'true_adj' -- the merged BASE file (runs/trees_rmfomc/forecasts/<est>_add_rm_fomc.parquet) holds date and pred_adj only.  The 14 dependent pools never started (DependencyNeverSatisfied); nothing ran |
| 13:08 | infrastructure fix (gate code only; no change to the design, configs, rankings or model code): gatebase compares pred_adj with the merged BASE file and true_adj / B with the templates' all-features rows, as cmd_merge does; tested locally on the templates.  Plan: scancel the 14 dead pools, resubmit the 3 gates (they reuse their own walked BASE pieces, same code and CPU model) and the pools; the UGE gate 15308728 had not started and runs the fixed code |
| 13:08 | (one session) scancel 518247-518250, 518252-518253, 518255-518262 (never started); resubmitted with the fixed code (jobids.txt slurm22 / slurm23): BASE gates 518575 (6442Y), 518580 (6736P), 518583 (6240) all PASS (7 of 7 origins text-identical, true_adj / B equal, both models; gate/rb/<model>/gatebase.json); pools 518576-518579 (6442Y), 518581-518582 (6736P) pending on Priority, 518584-518591 (6240, 32 workers) running; UGE gate 15308728 still queued (pool 15308729 held on it).  Expected: ~15:15 if the 6442Y / 6736P / UGE pools start soon, up to ~19:00 on the 6240 pools alone |
| 13:37 | 56 of 94 pieces claimed (32 Gold 6240, 24 Gold 6442Y: every XGBoost piece and 3 LightGBM top-20 pieces), 0 done, 0 failed; XGBoost 1.27 s a refit on 6442Y, 2.93 on 6240 (the top-20 / top-10 rungs); 6442Y pools 518576-518579 running since 13:17-13:28; 6736P pools 518581-518582 pending (Slurm's start estimate 00:20 on 28 Oct); UGE gate 15308728 running.  Added 518771 / 518772 (Gold 6240, 4 workers each; jobids.txt slurm24).  The 38 LightGBM pieces left go to UGE (after its gate), the new pools and the first freed workers: expected ~16:00-16:30 |
| 17:49 | (read-only) ladder 94 of 94 pieces done, 0 failed; no job of ours left in either queue (the three UGE pp_pooltre jobs are another run's, untouched) |
| 17:50 | merge 521871 (merge_ladder.sh): T2 bit-identical (LightGBM / XGBoost), the 10 rungs merged (pieces from Gold 6442Y, Xeon 6736P, Gold 6240, UGE Gold 6140; all four BASE gates PASS, gate/rb/), trees_wide.parquet md5 fc6bbe09...; fetched -> forecasts/trees_wide_ladder.parquet, runs/tree_merge_ladder/, gate/rb/*/gatebase.json |
| 17:55 | scored --design fix (169 forecasts, checks PASS); writeup/make_rehaul_beyond_ladder.py: readings (ORDINARY days, RULE section 2) LightGBM INTERMEDIATE (R(5) 0.72, R(20) 0.94), XGBoost CONCENTRATED (R(5) 0.91): split; tab:rh_beyond_ladder and the \rhbyL macros, compile-tested |
| (28 Oct) 16:40 | author ("do 1", after the readings by day type): the boosted trees' removals lo_<g>, g != fomc, on the patched design; RULE.txt addendum J (5b1b842, pushed before any job): 14 arms (LightGBM, XGBoost x rm, liq, ew, vw, sent, iv, vd), the 171 pieces of tasks/trees.txt verbatim -> tasks/trees_lofix.txt, ~3,900 CPU-hours (approved above 2,500); merge_lofix.sh |
| 16:42 | installed on Hoffman2 (code_lofix_5b1b842.tar extracted into code/, LF, code/COMMIT appended); submitted (jobids.txt slurm26 / uge9): Slurm 541089-541096 (Gold 6240, 8 x 4 workers; RUNNING at 16:43), 541097-541100 (Gold 6442Y, 4 x 6; pending), 541101-541102 (Xeon 6736P, 2 x 9; pending); UGE 15339286 (25 x 1 worker, 8 slots, Gold 6140; queued).  Claims runs/claims_trees_lofix, output runs/trees_lofix, TFULL designs_fix/full |

## State at 16:45 PDT, 28 October: addendum J (the trees' removals on the patched design) running

**Steps to finish** (B=/u/scratch/j/jamesdc1/abl; Slurm on hoffman2.oarc.ucla.edu):
1. Poll: `ls $B/runs/claims_trees_lofix/*/done | wc -l` (171), no `failed`.
2. `sbatch --parsable -p campus --qos=campus24 -C intel-gold-6240 -c 1 --mem=16G -t 1:00:00 -J abl_mlofix -o $B/logs/mlofix.%j.log $B/code/cluster/ablate/merge_lofix.sh`
3. Fetch only runs/trees_lofix/trees_wide.parquet -> forecasts/trees_wide_lofix.parquet and runs/trees_lofix/forecasts/*.json -> runs/tree_merge_lofix/.
4. ablate_score.py --design fix with the patched lo_<g> against the patched full set (replacing the pre-patch pairs, the dagger cells); make_rehaul_ablations.py, make_rehaul_channels.py (readings recomputed); compile-test; report each removal pre-patch -> patched and every reading that changes.
