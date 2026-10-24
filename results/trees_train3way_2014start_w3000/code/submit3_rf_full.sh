#!/bin/bash
# PREPARED, NOT SUBMITTED (RULE.txt addendum B): the forest's full set on ALL48 (3,000 days), the task lines of
# runs/tasks/w3rf_full.txt (from the time probe: 145.9 s a refit at 20 threads, 368 tasks, 27,527 CPU-hours projected),
# 20 threads; requires the forest's gate 2 on ALL48 to have passed (gate/g2/gate2_ALL48_rf_skx.json); then ALL48 merged
# again with every class and the final scores again (scores_full).  Run only on the owner's go.
# usage: bash submit3_rf_full.sh <code dir> <throttle> [dependency for the scores, e.g. afterany:<s3>]
set -euo pipefail
CODE=$1
TH=$2
DEP=${3:-}
BASE=/scratch1/jc_905/trees3way
R=$BASE/runs
L=$BASE/lines
J=$CODE/results/trees_train3way_2014start_w3000/code
X="experiments/trees_train3way.py"
if ! grep -q '"pass": true' $BASE/gate/g2/gate2_ALL48_rf_skx.json; then
  echo "ABORT: the forest's gate 2 on ALL48 has not passed"
  exit 3
fi
if grep -q "^phase3 " $BASE/jobids.txt; then
  echo "ABORT: already submitted"
  exit 3
fi
n=$(grep -c . $R/tasks/w3rf_full.txt)
A=(--parsable --account=pollok_1603 --partition=main --constraint="epyc-7513|epyc-7542")
cd $CODE
w=$(sbatch "${A[@]}" -J t3t_w3f -c 20 --mem=16G -t 12:00:00 --array=1-$n%$TH -o "$BASE/logs/%x.%A_%a.out" \
  --export=ALL,CODE=$CODE,TASKFILE=$R/tasks/w3rf_full.txt $J/walk.sbatch)
echo "gb $X merge --arm ALL48 --design-root $BASE/designs --out $R --require-all" > $L/m4.txt
echo "sweep experiments/trees_train3way_score.py $R $BASE/scores_full" > $L/s4.txt
m=$(sbatch "${A[@]}" -J t3t_m4 -c 1 --mem=16G -t 1:00:00 --dependency=afterany:$w -o "$BASE/logs/%x.%A_%a.out" \
  --export=ALL,LINES=$L/m4.txt,CODE=$CODE $J/line.sbatch)
s=$(sbatch "${A[@]}" -J t3t_s4 -c 1 --mem=16G -t 1:00:00 --dependency="afterok:$m${DEP:+,$DEP}" \
  -o "$BASE/logs/%x.%A_%a.out" --export=ALL,LINES=$L/s4.txt,CODE=$CODE $J/line.sbatch)
echo "phase3 $(date +%F_%T) code=$CODE w3f=$w ($n tasks, %$TH) m4=$m s4=$s" | tee -a $BASE/jobids.txt
