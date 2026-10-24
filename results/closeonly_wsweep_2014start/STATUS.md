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
