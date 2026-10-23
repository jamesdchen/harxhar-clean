#!/bin/bash
# The rf_skx walks of one window (CARC login node), everything on /home1: the task file from the rf_skx time probe by
# the printed run's rule (tasks --refits bar --target-hours 3), one Slurm array through
# cluster/slurm/pooltree_rerun13.sbatch (MODE=walk) of the code archive, python = the /home1 venv_skx; the array is
# registered in $HB/arrays_skx.txt for resubmit.sh (failed tasks resubmitted, resumed from their checkpoints).
# usage: bash submit_skx.sh <W> <threads> <throttle> <constraint> <code dir> [--hold]
set -euo pipefail
HB=/home1/jc_905/trees_final
W=$1; NT=$2; TH=$3; CN=$4; CODE=$5; XF=${6:-}
PY=$HB/venv_skx/bin/python
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
R=$HB/runs_skx/w$W
mkdir -p $R
cd $CODE
if [ ! -f $R/tasks.txt ]; then
  $PY experiments/pooled_trees_rerun_13.py tasks --design $HB/designs/w$W --window-days $W --est rf_skx \
    --sec-table $R/timing/sec_refit.csv --refits bar --span 2014-01-02:2024-04-30 --target-hours 3 \
    --taskfile $R/tasks.txt | tee $R/tasks_info.txt
fi
n=$(wc -l < $R/tasks.txt)
OC=/home1/jc_905/envs/harxhar-r13/ssl/openssl.cnf
X=()
[ -n "$XF" ] && X=("$XF")
j=$(sbatch --parsable -J tfs_w$W --array=1-$n%$TH -c $NT --mem=16G -t 8:00:00 --constraint="$CN" "${X[@]}" \
  -o "$HB/logs/%x.%A_%a.out" \
  --export=ALL,MODE=walk,TASKFILE=$R/tasks.txt,HARXHAR_PY=$PY,DESIGN=$HB/designs/w$W,OUT=$R,WINDOW_DAYS=$W,OPENSSL_CONF=$OC \
  cluster/slurm/pooltree_rerun13.sbatch)
echo "$j $R/tasks.txt $W $R $CN $CODE $NT" >> $HB/arrays_skx.txt
echo "$(date +%F_%T) rf_skx w$W: $n tasks, $NT threads, throttle $TH, constraint $CN, code $CODE, array $j" | tee -a $HB/jobids.txt
