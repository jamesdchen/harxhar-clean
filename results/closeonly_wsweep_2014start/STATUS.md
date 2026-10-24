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
