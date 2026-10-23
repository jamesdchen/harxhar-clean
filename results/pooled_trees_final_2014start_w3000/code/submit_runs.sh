#!/bin/bash
# The final tree walks on CARC (login node): the task file of a window from its time probe (the printed run's rule:
# experiments/pooled_trees_rerun_13.py tasks --refits bar --target-hours 3, arms cheapest first), split by group, and
# one Slurm array per (window, group) through cluster/slurm/pooltree_rerun13.sbatch (MODE=walk; resumable pieces).
# usage: bash submit_runs.sh <W> <gb|rf> <threads> <throttle> <constraint> [dependency]
#   gb = LightGBM + XGBoost (both sets), rf = random forest (both sets)
set -euo pipefail
BASE=/scratch1/jc_905/trees_final
W=$1; G=$2; NT=$3; TH=$4; CN=$5; DEP=${6:-}
PY=/scratch1/jc_905/envs/harxhar-r13/bin/python
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
R=$BASE/runs/w$W
cd $BASE/repo
if [ ! -f $R/tasks.txt ]; then
  $PY experiments/pooled_trees_rerun_13.py tasks --design $BASE/designs/w$W --window-days $W \
    --sec-table $R/timing/sec_refit.csv --refits bar --span 2014-01-02:2024-04-30 --target-hours 3 \
    --taskfile $R/tasks.txt | tee $R/tasks_info.txt
fi
case $G in gb) P='^(lgbm|xgb) ' ;; rf) P='^rf ' ;; *) echo "group gb or rf"; exit 2 ;; esac
grep -E "$P" $R/tasks.txt > $R/tasks_$G.txt
n=$(wc -l < $R/tasks_$G.txt)
D=()
[ -n "$DEP" ] && D=(--dependency="$DEP")
j=$(sbatch --parsable -J tf_${G}_w$W --array=1-$n%$TH -c $NT --mem=8G -t 8:00:00 --constraint="$CN" "${D[@]}" \
  -o "$BASE/logs/%x.%A_%a.out" \
  --export=ALL,MODE=walk,TASKFILE=$R/tasks_$G.txt,HARXHAR_PY=$PY,DESIGN=$BASE/designs/w$W,OUT=$R,WINDOW_DAYS=$W \
  cluster/slurm/pooltree_rerun13.sbatch)
echo "$(date +%F_%T) $G w$W: $n tasks, $NT threads, throttle $TH, constraint $CN, array $j" | tee -a $BASE/jobids.txt
