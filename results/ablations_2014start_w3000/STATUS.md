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
