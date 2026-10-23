#!/bin/bash
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --constraint=epyc-7513
# One task = one line of $TASKS: "<window_days> <fold> <est> <set>": the trees' frozen configuration fitted on the
# fold's fit rows of the 2016-start design cache of that window, scored on the fold's validation rows
# (fold_fit16.py).  Environment and threads as results/tree_window_presample_final_2026-10-22 (harxhar-r13, 8 threads,
# EPYC 7513); code = the sweep's code tree (f71449e: experiments/pooled_trees_rerun_13.py and the configs unchanged
# from 9d3fc05).
set -euo pipefail
B=/scratch1/jc_905/wsweep
: "${TASKS:?}"
PY=/scratch1/jc_905/envs/harxhar-r13/bin/python
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK CODE_COMMIT=$(cat $B/code_commit.txt)
read -r W F EST SET <<< "$(sed -n "${SLURM_ARRAY_TASK_ID}p" $TASKS)"
DZ=$B/code/results/window_sweep_2016start_2026-10-22/w$W/_work/design_$SET
echo "tree16 w${W}_f${F}_${EST}_${SET} ($DZ) on $(hostname) [$(lscpu | grep 'Model name' | sed 's/.*: *//')] $(date)"
$PY $B/trees16/fold_fit16.py --repo $B/code --design $DZ --folds-json $B/trees16/folds16.json --window-days $W \
    --fold $F --est $EST --set $SET --threads $SLURM_CPUS_PER_TASK --out $B/trees16/out
echo "tree16 done $(date)"
