# Close-only window sweep, pooled linear models and trees (status)

Rule: `RULE.txt` (committed and pushed before any job).  Code: `experiments/closewin.py` (constants, design stage and
checks), `experiments/closewin_linear.py` (the linear three-way's CLOSE arm at window W; gate L),
`experiments/closewin_trees.py` (the trees three-way's walk / merge with arms CLOSE_w<W>; gate T; CPU check),
`experiments/closewin_score.py` (CARC, 16:00 scores and DM per W), `experiments/closewin_econ.py` (local, D6 and the
part-2-layout tables), `code/` (CARC job scripts: `env.sh`, `stage.sbatch`, `lin.sbatch`, `tgate.sbatch`,
`twalk.sbatch`, `fin.sbatch`, `submit.sh`).  Staging: `/scratch1/jc_905/closewin` on CARC Discovery; designs
`/home1/jc_905/closewin/designs`.

Local tests before the commit (scratchpad, the three-way's CLOSE forecasts standing in for every W): the scorer's W =
3,000 checks pass for both families (linear QLIKE OLS HAR 0.12228624582318119 vs 0.1222862458231811, enet full
0.11438473820141823 vs 0.1143847382014182; the trees' qlike_16 / gap rows); the econ's W = 3,000 D6 check passes for
both families.

| Time (PDT, 2026-10-24) | Event |
|---|---|
| ~09:40 | RULE.txt and code committed (62cfc8c) and pushed; no job before it |
| 09:44 | chain submitted (`code/submit.sh 62cfc8c`; code archive `/scratch1/jc_905/closewin/code_62cfc8c`) |

Jobs (CARC, account pollok_1603, partition main):

| Job | Name | What | Depends on |
|---|---|---|---|
| 13380316 | cw_stage | the six W designs copied to /home1 (md5) + checks -> design_check.json (EPYC) | - |
| 13380317 | cw_lin3k | linear W = 3,000: derive / pretune / select / 7 walks / export + gate L (EPYC, 4 CPUs) | stage |
| 13380318 | cw_linW | linear W = 1,000, 1,500, 2,000, 2,500, 3,500 (EPYC, 4 CPUs) | lin3k |
| 13380319 | cw_linS | linear 16:00 scores -> scores/linear (last linear job) | linW |
| 13380320 | cw_tgate | trees gate T (W = 3,000, 20 origins x 6 class/set, EPYC, 8 threads) + reuse of the three-way CLOSE forecasts | stage |
| 13380321 | cw_twgb | tree walks LightGBM / XGBoost, 20 lines (W ascending), EPYC, 8 threads, %8 | tgate |
| 13380322 | cw_twrf | tree walks, the forest, 10 lines, EPYC or Xeon 4116 (bad nodes excluded), 8 threads, %4 | tgate |
| 13380323 | cw_tfin | merge (5 W) + CPU check + tree 16:00 scores -> scores/trees (last job of the chain) | twgb, twrf (afterany) |

To finish: pull scores/{linear,trees} (csv, tables_qlike.txt, f16.parquet, check.json), design_check.json, the gate
jsons, the frozen penalties per W and the piece metas; run `experiments/closewin_econ.py <main checkout> <fam> ...`
locally for each family (tables.txt); sacct CPU-hours.

## Addendum A: the 10:15 failures (RULE.txt addendum A, commit 050a6a6, before its jobs)

| Time (PDT) | Event |
|---|---|
| 09:44-09:49 | cw_stage 13380316 (design check PASS), cw_lin3k 13380317 (gate L PASS: W = 3,000 = the three-way CLOSE arm), cw_tgate 13380320 (gate T PASS; the three-way CLOSE forecasts reused as CLOSE_w3000) COMPLETED |
| 09:47 | cw_linW 13380318 FAILED after 21 s: W = 1,000, every penalized walk on `assert tw > 2 * (STD_MIN_EVIDENCE + TUNE_PER)` (tw = 1,000) in pool_linear_pretune._seed_std (STANDARDIZE_PENALTY's tracker needs a window > 1,000 rows); not I/O; W = 1,500 .. 3,500 never started |
| 09:57 | cw_twrf 13380322_1 (CLOSE_w1000 forest HAR, d05-39, Xeon 4116) FAILED: BrokenPipeError errno 108 reading data/core_stats.parquet on /scratch1 before any fit (no piece) -- a failing /scratch1 mount; the only I/O error in this sweep's logs |
| 10:21 | `code/submit_a.sh 050a6a6`: code archive on /home1 (`/home1/jc_905/closewin/code_050a6a6`; the walk code cmp-identical to 62cfc8c's, src identical but __pycache__; git diff 62cfc8c..050a6a6 touches only the two scorers and results/); cw_linW2 13381257 (W = 1,500, 2,000, 2,500, 3,500, EPYC); cw_twrf2 13381258 (the failed forest line, EPYC only, d05-39 excluded); 13380322_9 / _10 (pending, W = 3,500 forest) restricted to EPYC 7513 / 7542 with the exclusions; cw_linS 13380319 cancelled, cw_linS2 13381259 (afterok 13381257; the new scorer) = the last linear job; cw_tfin 13380323 kept, re-pointed afterany 13380321, 13380322, 13381258 and restricted to EPYC = the last tree job |

Linear W = 1,000 waited for the owner (RULE.txt addendum A options a-d) until addendum B.

## Addendum B: linear W = 1,000 at 1,001 closes (owner option b; RULE.txt addendum B, commit a3e87f3, before its jobs)

From the coordinator's 10:45 standing rule on, every state-changing command is written verbatim in the SSH tool call
(no submission scripts); submit.sh and submit_a.sh (above) were piped scripts.

| Time (PDT) | Event |
|---|---|
| 10:27-10:30 | cw_linW2 13381257 (W = 1,500 .. 3,500), cw_linS2 13381259 (scores without W = 1,000) and cw_twrf2 13381258 COMPLETED |
| 10:36 | code archive `/home1/jc_905/closewin/code_a3e87f3` (linear_train3way{,_runner}.py, pool_linear_pretune.py, the orchestrator, closewin.py cmp-identical to 62cfc8c's); the failed W = 1,000 walks set aside (`mv runs/linear/w1000/_work/walk runs/linear/w1000/_work/walk_failed_0947`); cw_lin1k 13381484 (sbatch --wrap: OLS walk at 1,000 on w1000; derive_shift -> w1000_r1001; pretune / select / walks 1-6 at --rows 1001; export_1k; EPYC); `scancel 13381259` a no-op (it had completed); the first linS3 submission refused (afterok on the completed, purged 13381257: "Job dependency problem") |
| 10:37 | the scores without W = 1,000 moved to `scores/linear_addA_noW1000` (mv to a new name); cw_linS3 13381487 (afterok 13381484) = the last linear job, -> scores/linear with W = 1,000 |
