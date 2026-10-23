#!/bin/bash
# After the gate (CARC login node): the gate's comparison and sacct, the repo clone moved to <commit> (the runner is the
# gate's), then the LightGBM / XGBoost walks of both windows (bar refits) and the one-month day-refit check (W = 3,000,
# 2014-01, LightGBM and XGBoost full set).  No random-forest walk.
# usage: bash after_gate.sh <commit> <gate job ids, comma-separated>
set -euo pipefail
BASE=/scratch1/jc_905/trees_final
C=$1; GJ=$2
PY=/scratch1/jc_905/envs/harxhar-r13/bin/python
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
PR=/scratch1/jc_905/harxhar-trees13/results/pooled_trees_rerun13_2026-10-20/w1000
if squeue -u jc_905 -h -o "%j" | grep -q '^tf'; then echo "ABORT: tf jobs still queued"; squeue -u jc_905 | grep tf; exit 3; fi
cd $BASE
H0=$(git -C repo rev-parse HEAD)
git -C repo fetch -q origin insurance-selling-2026-09-19-e34-trees-final
git -C repo checkout -q --detach "$C"
git -C repo diff --quiet "$H0" HEAD -- experiments cluster src data || { echo "ABORT: runner tree changed since the gate"; exit 4; }
echo "runner, configs, src, data unchanged from the gate's $H0"
echo "repo at $(git -C repo log --oneline -1)"
CD=$BASE/repo/results/pooled_trees_final_2014start_w3000/code
$PY $CD/gate_cmp.py $PR/pieces gate/gate_check.csv A=gate/A B=gate/B B7542=gate/B7542 Brf16=gate/Brf16
sacct -j "$GJ" -P --format=JobID,JobName%20,State,ExitCode,Elapsed,Start,End,NodeList,AllocCPUS,ReqMem,TotalCPU,CPUTimeRAW \
  | tr '|' ',' > gate/sacct_gate.csv
CN="epyc-7513|epyc-7542"
bash $CD/submit_runs.sh 3000 gb 8 49 "$CN"
bash $CD/submit_runs.sh 3500 gb 8 32 "$CN"
for W in 3000 3500; do echo "== w$W"; cat $BASE/runs/w$W/tasks_info.txt; done
# one month of day refits, W = 3,000 (owner's question): LightGBM and XGBoost, full set
DC=$BASE/runs/w3000_daycheck
mkdir -p $DC
printf "lgbm all_features day 2014-01-02 2014-01-31 1/1\nxgb all_features day 2014-01-02 2014-01-31 1/1\n" > $DC/tasks.txt
j=$(cd $BASE/repo && sbatch --parsable -J tf_day_w3000 --array=1-2 -c 8 --mem=8G -t 2:00:00 --constraint="$CN" \
  -o "$BASE/logs/%x.%A_%a.out" \
  --export=ALL,MODE=walk,TASKFILE=$DC/tasks.txt,HARXHAR_PY=$PY,DESIGN=$BASE/designs/w3000,OUT=$DC,WINDOW_DAYS=3000 \
  cluster/slurm/pooltree_rerun13.sbatch)
echo "$(date +%F_%T) daycheck w3000 2014-01: array $j" | tee -a $BASE/jobids.txt
sleep 3
squeue -u jc_905 -h -o "%j %T" | grep tf | sort | uniq -c
