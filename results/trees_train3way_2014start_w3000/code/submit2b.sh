#!/bin/bash
# Phase 2 under RULE.txt addendum B (CARC login node): gate C (LightGBM force_col_wise on the 48-period arms), then
#   ALL48_N39k: LightGBM (afterok gate C), XGBoost (afterok gate 2 on ALL48's XGBoost), the forest at 20 threads
#               (afterok gate 2 on ALL48's forest, i.e. after array <gate-2 forest> ends);
#   ALL48:      LightGBM (afterok gate C), XGBoost, the forest's HAR set (as above); NOT its full set (task file
#               runs/tasks/w3rf_full.txt and code/submit3_rf_full.sh prepared, unsubmitted);
#   merges and scores: m2 -> s2 (scores_w2, after the cheap arms' s1), m3 -> s3 (scores, after s2).
# usage: bash submit2b.sh <code dir> <s1 job id> <gate-2 forest array id>
set -euo pipefail
CODE=$1
S1=$2
G2B=$3
BASE=/scratch1/jc_905/trees3way
R=$BASE/runs
L=$BASE/lines
T=$R/tasks
J=$CODE/results/trees_train3way_2014start_w3000/code
X="experiments/trees_train3way.py"
D="--design-root $BASE/designs"
if grep -q "^phase2b " $BASE/jobids.txt; then
  echo "ABORT: phase 2b already submitted"
  exit 3
fi
grep -E '^ALL48_N39k lgbm ' $T/w2gb.txt > $T/w2lgbm.txt
grep -E '^ALL48_N39k xgb ' $T/w2gb.txt > $T/w2xgb.txt
grep -E '^ALL48 lgbm ' $T/w3gb.txt > $T/w3lgbm.txt
grep -E '^ALL48 xgb ' $T/w3gb.txt > $T/w3xgb.txt
grep -E '^ALL48 rf_skx baseline ' $T/w3rf.txt > $T/w3rf_har.txt
grep -E '^ALL48 rf_skx all_features ' $T/w3rf.txt > $T/w3rf_full.txt
wc -l $T/w2lgbm.txt $T/w2xgb.txt $T/w2rf.txt $T/w3lgbm.txt $T/w3xgb.txt $T/w3rf_har.txt $T/w3rf_full.txt
# gate C lines (gate/gc) and the record (gate/gcb)
GC=$BASE/gate/gc
W48="--from 2014-01-02T15:30 --to 2014-01-03T10:00"
ONE="--from 2014-01-03T10:00 --to 2014-01-03T10:00"
: > $L/gc.txt
for s in baseline all_features; do
  {
    echo "gb $X walk --arm ALL48 --est lgbm --set $s $W48 --every-row --tag every $D --out $GC"
    echo "gb $X walk --arm ALL48 --est lgbm --set $s $W48 --tag scored $D --out $GC"
    echo "gb $X walk --arm ALL48 --est lgbm --set $s $W48 --tag scored2 $D --out $GC"
    for k in 1 2 3 4; do
      echo "gb $X walk --arm ALL48 --est lgbm --set $s $ONE --tag one$k $D --out $GC"
    done
    echo "gb $X walk --arm ALL48_N39k --est lgbm --set $s $W48 --tag scored $D --out $GC"
    echo "gb $X walk --arm ALL48_N39k --est lgbm --set $s $W48 --tag scored2 $D --out $GC"
    for k in 1 2; do
      echo "gb $X walk --arm ALL48_N39k --est lgbm --set $s $ONE --tag one$k $D --out $GC"
    done
    echo "gb $X walk --arm RTH13 --est lgbm --set $s --ends 10 --lgbm-col-wise on --tag ends10_col $D --out $BASE/gate/gcb"
  } >> $L/gc.txt
done
printf "%s\n" "gb $X gate2 --arm ALL48 --est lgbm --out $GC" "gb $X gaterep --out $GC" > $L/gcc.txt
echo "gb $X gate1 --no-fail --out $BASE/gate/gcb" > $L/gcb.txt
echo "gb $X gate2 --arm ALL48 --est xgb --out $BASE/gate/g2" > $L/g2x.txt
echo "gb $X gate2 --arm ALL48 --est rf_skx --out $BASE/gate/g2" > $L/g2r.txt
echo "gb $X merge --arm ALL48_N39k $D --out $R --require-all" > $L/m2.txt
echo "gb $X merge --arm ALL48 $D --out $R --require-all --skip rf_skx/all_features" > $L/m3.txt
echo "sweep experiments/trees_train3way_score.py $R $BASE/scores_w2" > $L/s2.txt
echo "sweep experiments/trees_train3way_score.py $R $BASE/scores" > $L/s3.txt
A=(--parsable --account=pollok_1603 --partition=main --constraint="epyc-7513|epyc-7542")
lj() {  # <name> <lines> <cpus> <mem> <time> <array or -> [dependency]
  local arr=()
  local dep=()
  if [ "$6" != - ]; then arr=(--array="$6"); fi
  if [ -n "${7:-}" ]; then dep=(--dependency="$7"); fi
  sbatch "${A[@]}" -J "$1" -c "$3" --mem="$4" -t "$5" "${arr[@]}" "${dep[@]}" -o "$BASE/logs/%x.%A_%a.out" \
    --export=ALL,LINES=$2,CODE=$CODE $J/line.sbatch
}
wj() {  # <name> <taskfile> <cpus> <mem> <throttle> <dependency>
  local n
  n=$(grep -c . "$2")
  sbatch "${A[@]}" -J "$1" -c "$3" --mem="$4" -t 12:00:00 --array="1-$n%$5" --dependency="$6" \
    -o "$BASE/logs/%x.%A_%a.out" --export=ALL,CODE=$CODE,TASKFILE=$2 $J/walk.sbatch
}
cd $CODE
gc=$(lj t3t_gc $L/gc.txt 8 16G 2:00:00 "1-$(grep -c . $L/gc.txt)")
gcc=$(lj t3t_gcc $L/gcc.txt 1 8G 0:30:00 1-2 afterok:$gc)
gcb=$(lj t3t_gcb $L/gcb.txt 1 8G 0:30:00 - afterany:$gc)
g2x=$(lj t3t_g2x $L/g2x.txt 1 8G 0:30:00 -)
g2r=$(lj t3t_g2r $L/g2r.txt 1 8G 0:30:00 - afterok:$G2B)
w2l=$(wj t3t_w2l $T/w2lgbm.txt 8 8G 12 afterok:$gcc)
w2x=$(wj t3t_w2x $T/w2xgb.txt 8 8G 15 afterok:$g2x)
w3l=$(wj t3t_w3l $T/w3lgbm.txt 8 16G 15 afterok:$gcc)
w3x=$(wj t3t_w3x $T/w3xgb.txt 8 16G 15 afterok:$g2x)
w2r=$(wj t3t_w2r $T/w2rf.txt 20 12G 20 afterok:$g2r)
w3h=$(wj t3t_w3h $T/w3rf_har.txt 20 16G 4 afterok:$g2r)
m2=$(lj t3t_m2 $L/m2.txt 1 16G 1:00:00 - afterany:$w2l:$w2x:$w2r)
s2=$(lj t3t_s2 $L/s2.txt 1 16G 1:00:00 - afterok:$m2,afterany:$S1)
m3=$(lj t3t_m3 $L/m3.txt 1 16G 1:00:00 - afterany:$w3l:$w3x:$w3h)
s3=$(lj t3t_s3 $L/s3.txt 1 16G 1:00:00 - afterok:$m3,afterany:$s2)
echo "phase2b $(date +%F_%T) code=$CODE gc=$gc gcc=$gcc gcb=$gcb g2x=$g2x g2r=$g2r w2l=$w2l w2x=$w2x w3l=$w3l w3x=$w3x w2r=$w2r w3h=$w3h m2=$m2 s2=$s2 m3=$m3 s3=$s3" | tee -a $BASE/jobids.txt
sleep 2
squeue -u jc_905 -h -o "%.16i %.10j %.8T %.4C %R" | grep -E "t3t_(gc|g2|w2|w3|m2|m3|s2|s3)" | head -30
