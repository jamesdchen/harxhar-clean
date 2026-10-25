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
