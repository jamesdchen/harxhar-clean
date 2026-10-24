#!/bin/bash
# Phase 2 of the trees' three-way comparison (CARC login node): the ALL48 (3,000-day) walks, after gates 1-2 passed
# (gate/g1/gate1.json, gate/g2/gate2.json) and the projection (runs/tasks/projection.json) is within the cap; the task
# files w3gb.txt (LightGBM / XGBoost, 8 threads) and w3rf.txt (the forest, 20 threads) were written by phase 1's tasks
# job from the time probe.  Then the ALL48 merge and the final scores (every arm) -> $BASE/scores.
# usage: bash submit2.sh <code dir> <throttle gb> <throttle rf> [dependency for the final score, e.g. afterany:<s2>]
set -euo pipefail
CODE=$1; TG=$2; TR=$3; DEP=${4:-}
BASE=/scratch1/jc_905/trees3way
R=$BASE/runs
L=$BASE/lines
J=$CODE/results/trees_train3way_2014start_w3000/code
X="experiments/trees_train3way.py"
D="--design-root $BASE/designs"
if [ -f $BASE/jobids.txt ] && grep -q "^phase2 " $BASE/jobids.txt; then
  echo "ABORT: phase 2 already submitted"
  exit 3
fi
for f in $BASE/gate/g1/gate1.json $BASE/gate/g2/gate2.json $R/tasks/projection.json; do
  grep -q -E '"(pass|within_cap)": true' $f || { echo "ABORT: $f does not pass"; exit 3; }
done
ng=$(grep -c . $R/tasks/w3gb.txt)
nr=$(grep -c . $R/tasks/w3rf.txt)
A=(--parsable --account=pollok_1603 --partition=main --constraint="epyc-7513|epyc-7542")
cd $CODE
w3g=$(sbatch "${A[@]}" -J t3t_w3g -c 8 --mem=16G -t 12:00:00 --array=1-$ng%$TG -o "$BASE/logs/%x.%A_%a.out" \
  --export=ALL,CODE=$CODE,TASKFILE=$R/tasks/w3gb.txt $J/walk.sbatch)
w3r=$(sbatch "${A[@]}" -J t3t_w3r -c 20 --mem=16G -t 12:00:00 --array=1-$nr%$TR -o "$BASE/logs/%x.%A_%a.out" \
  --export=ALL,CODE=$CODE,TASKFILE=$R/tasks/w3rf.txt $J/walk.sbatch)
echo "gb $X merge --arm ALL48 $D --out $R --require-all" > $L/m3.txt
echo "sweep experiments/trees_train3way_score.py $R $BASE/scores" > $L/s3.txt
m3=$(sbatch "${A[@]}" -J t3t_m3 -c 1 --mem=16G -t 1:00:00 --dependency=afterany:$w3g:$w3r \
  -o "$BASE/logs/%x.%A_%a.out" --export=ALL,LINES=$L/m3.txt,CODE=$CODE $J/line.sbatch)
s3=$(sbatch "${A[@]}" -J t3t_s3 -c 1 --mem=16G -t 1:00:00 --dependency=afterok:$m3${DEP:+,$DEP} \
  -o "$BASE/logs/%x.%A_%a.out" --export=ALL,LINES=$L/s3.txt,CODE=$CODE $J/line.sbatch)
echo "phase2 $(date +%F_%T) code=$CODE w3g=$w3g ($ng tasks, %$TG) w3r=$w3r ($nr tasks, %$TR) m3=$m3 s3=$s3" | tee -a $BASE/jobids.txt
