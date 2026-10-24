#!/bin/bash
# Phase 1 of the trees' three-way comparison on CARC Discovery (login node; RULE.txt sections 5 and 8):
#   code      git archive of <commit> (tracked tree without results/, plus the two templates' and this run's results
#             directories) -> $BASE/code_<sha7>
#   stage     the design caches copied (md5) + experiments/trees_train3way.py check
#   g1a/g1b   gate 1: RTH13, the first and last 10 origins of each class and set (8 threads; the forest also at 20)
#   g2a/g2b   gate 2: scored-only against every row, ALL48 (W 144,000; 2014-01-02 15:30 .. 01-03 10:00) and ALL48_N3k
#             (W 3,000; 2014-01-02 15:30 .. 01-06 10:30), each class and set (the forest of ALL48 at 20 threads)
#   tm8/tm20  the time probe: 3 refits at the first origins of each arm, class and set (8 threads; the forest of
#             ALL48_N39k / ALL48 and a RTH13 calibration at 20)
#   gate      gate1 + gate2 comparisons (exit 3 on a failure; every walk is afterok on it)
#   tasks     task files of every arm from the probe (target 3 h a task) and their CPU-hour projections
#   cap       the projection of all new walks against 25,000 CPU-hours (exit 3 above it)
#   w1g/w1r   walks CLOSE, RTH13_N3k, ALL48_N3k: LightGBM / XGBoost, then the forest; 8 threads
#   w2g/w2r   walks ALL48_N39k: LightGBM / XGBoost (8 threads), then the forest (20 threads); afterok cap
#   m1/s1     merge of the w1 arms, interim scores -> $BASE/scores_w1
#   m2/s2     merge of ALL48_N39k, interim scores -> $BASE/scores_w2
# ALL48 (3,000 days) is phase 2 (after the projection is read).  Refuses a second submission (jobids.txt).
# usage: bash submit1.sh <commit>
set -euo pipefail
C=$1
BASE=/scratch1/jc_905/trees3way
BR=insurance-selling-2026-09-19-e34-train3way-trees
if [ -f $BASE/jobids.txt ] && grep -q "^phase1 " $BASE/jobids.txt; then
  echo "ABORT: phase 1 already submitted"
  exit 3
fi
mkdir -p $BASE/logs $BASE/lines $BASE/runs/tasks $BASE/gate/g1 $BASE/gate/g2 $BASE/timing $BASE/designs
cd $BASE
if [ ! -d repo/.git ]; then
  git clone -q --reference-if-able /scratch1/jc_905/trees_final/repo --no-checkout \
    https://github.com/jamesdchen/harxhar-clean.git repo
fi
git -C repo fetch -q origin $BR
S=$(git -C repo rev-parse --short=7 "$C^{commit}")
CODE=$BASE/code_$S
if [ ! -d $CODE ]; then
  mkdir -p $CODE
  P=$(git -C repo ls-tree --name-only "$C" | grep -v '^results$')
  git -C repo archive "$C" $P results/pooled_trees_final_2014start_w3000 results/pooled_rf_skx_2014start_w3000 \
    results/trees_train3way_2014start_w3000 | tar -x -C $CODE
  git -C repo rev-parse "$C^{commit}" > $CODE/COMMIT
fi
echo "code $CODE at $(cat $CODE/COMMIT), $(du -sh $CODE | cut -f1)"
J=$CODE/results/trees_train3way_2014start_w3000/code
D="--design-root $BASE/designs"
X="experiments/trees_train3way.py"
L=$BASE/lines
R=$BASE/runs
# gate 1 (RTH13, --ends 10)
: > $L/g1a.txt
: > $L/g1b.txt
for e in lgbm xgb rf_skx; do
  for s in baseline all_features; do
    py=gb
    if [ $e = rf_skx ]; then py=skx; fi
    echo "$py $X walk --arm RTH13 --est $e --set $s --ends 10 --tag ends10_t8 $D --out $BASE/gate/g1" >> $L/g1a.txt
    if [ $e = rf_skx ]; then
      echo "$py $X walk --arm RTH13 --est $e --set $s --ends 10 --tag ends10_t20 $D --out $BASE/gate/g1" >> $L/g1b.txt
    fi
  done
done
# gate 2 (every row against the scored origins only)
: > $L/g2a.txt
: > $L/g2b.txt
for e in lgbm xgb rf_skx; do
  for s in baseline all_features; do
    for how in every scored; do
      py=gb
      if [ $e = rf_skx ]; then py=skx; fi
      ev=""
      if [ $how = every ]; then ev="--every-row"; fi
      l48="$py $X walk --arm ALL48 --est $e --set $s --from 2014-01-02T15:30 --to 2014-01-03T10:00 $ev --tag $how $D --out $BASE/gate/g2"
      if [ $e = rf_skx ]; then echo "$l48" >> $L/g2b.txt; else echo "$l48" >> $L/g2a.txt; fi
      echo "$py $X walk --arm ALL48_N3k --est $e --set $s --from 2014-01-02T15:30 --to 2014-01-06T10:30 $ev --tag $how $D --out $BASE/gate/g2" >> $L/g2a.txt
    done
  done
done
# time probes
: > $L/tm8.txt
: > $L/tm20.txt
for a in RTH13 CLOSE RTH13_N3k ALL48_N3k ALL48_N39k ALL48; do
  echo "gb $X time --arm $a --est lgbm --est xgb --k 3 $D --out $BASE" >> $L/tm8.txt
  case $a in
    ALL48_N39k | ALL48) echo "skx $X time --arm $a --est rf_skx --k 3 $D --out $BASE" >> $L/tm20.txt ;;
    RTH13)
      echo "skx $X time --arm $a --est rf_skx --k 3 $D --out $BASE" >> $L/tm8.txt
      echo "skx $X time --arm $a --est rf_skx --k 3 $D --out $BASE" >> $L/tm20.txt
      ;;
    *) echo "skx $X time --arm $a --est rf_skx --k 3 $D --out $BASE" >> $L/tm8.txt ;;
  esac
done
printf "%s\n" "gb $X gate1 --out $BASE/gate/g1" "gb $X gate2 --out $BASE/gate/g2" > $L/gate.txt
TA="$D --sec-dir $BASE/timing --target-hours 3"
{
  echo "gb $X tasks --arm CLOSE --arm RTH13_N3k --arm ALL48_N3k --est lgbm --est xgb --threads-of lgbm=8,xgb=8 $TA --max-lines 50 --taskfile $R/tasks/w1gb.txt"
  echo "gb $X tasks --arm CLOSE --arm RTH13_N3k --arm ALL48_N3k --est rf_skx --threads-of rf_skx=8 $TA --max-lines 50 --taskfile $R/tasks/w1rf.txt"
  echo "gb $X tasks --arm ALL48_N39k --est lgbm --est xgb --threads-of lgbm=8,xgb=8 $TA --max-lines 40 --taskfile $R/tasks/w2gb.txt"
  echo "gb $X tasks --arm ALL48_N39k --est rf_skx --threads-of rf_skx=20 $TA --max-lines 150 --taskfile $R/tasks/w2rf.txt"
  echo "gb $X tasks --arm ALL48 --est lgbm --est xgb --threads-of lgbm=8,xgb=8 $TA --max-lines 150 --taskfile $R/tasks/w3gb.txt"
  echo "gb $X tasks --arm ALL48 --est rf_skx --threads-of rf_skx=20 $TA --max-lines 400 --taskfile $R/tasks/w3rf.txt"
} > $L/tasks.txt
echo "gb $X project --taskdir $R/tasks --cap 25000" > $L/cap.txt
{
  echo "gb $X merge --arm CLOSE $D --out $R --require-all"
  echo "gb $X merge --arm RTH13_N3k $D --out $R --require-all"
  echo "gb $X merge --arm ALL48_N3k $D --out $R --require-all"
} > $L/m1.txt
echo "gb $X merge --arm ALL48_N39k $D --out $R --require-all" > $L/m2.txt
echo "sweep experiments/trees_train3way_score.py $R $BASE/scores_w1" > $L/s1.txt
echo "sweep experiments/trees_train3way_score.py $R $BASE/scores_w2" > $L/s2.txt
A=(--parsable --account=pollok_1603 --partition=main --constraint="epyc-7513|epyc-7542")
lj() {  # <name> <lines> <cpus> <mem> <time> <array or -> <dependency>
  local arr=()
  if [ "$6" != - ]; then arr=(--array="$6"); fi
  sbatch "${A[@]}" -J "$1" -c "$3" --mem="$4" -t "$5" "${arr[@]}" --dependency="$7" -o "$BASE/logs/%x.%A_%a.out" \
    --export=ALL,LINES=$2,CODE=$CODE $J/line.sbatch
}
wj() {  # <name> <taskfile> <cpus> <mem> <array> <dependency>
  sbatch "${A[@]}" -J "$1" -c "$3" --mem="$4" -t 12:00:00 --array="$5" --dependency="$6" -o "$BASE/logs/%x.%A_%a.out" \
    --export=ALL,CODE=$CODE,TASKFILE=$2 $J/walk.sbatch
}
cd $CODE
st=$(sbatch "${A[@]}" -J t3t_stage -o "$BASE/logs/%x.%j.out" --export=ALL,CODE=$CODE $J/stage.sbatch)
g1a=$(lj t3t_g1a $L/g1a.txt 8 16G 3:00:00 "1-$(wc -l < $L/g1a.txt)" afterok:$st)
g1b=$(lj t3t_g1b $L/g1b.txt 20 16G 3:00:00 "1-$(wc -l < $L/g1b.txt)" afterok:$st)
g2a=$(lj t3t_g2a $L/g2a.txt 8 16G 4:00:00 "1-$(wc -l < $L/g2a.txt)" afterok:$st)
g2b=$(lj t3t_g2b $L/g2b.txt 20 16G 4:00:00 "1-$(wc -l < $L/g2b.txt)" afterok:$st)
t8=$(lj t3t_tm8 $L/tm8.txt 8 16G 2:00:00 "1-$(wc -l < $L/tm8.txt)" afterok:$st)
t20=$(lj t3t_tm20 $L/tm20.txt 20 16G 2:00:00 "1-$(wc -l < $L/tm20.txt)" afterok:$st)
gt=$(lj t3t_gate $L/gate.txt 1 8G 0:30:00 1-2 afterok:$g1a:$g1b:$g2a:$g2b)
tk=$(lj t3t_tasks $L/tasks.txt 1 8G 0:30:00 1-6 afterok:$t8:$t20)
cp_=$(lj t3t_cap $L/cap.txt 1 4G 0:15:00 - afterok:$tk)
w1g=$(wj t3t_w1g $R/tasks/w1gb.txt 8 8G 1-50%30 afterok:$gt:$tk)
w1r=$(wj t3t_w1r $R/tasks/w1rf.txt 8 8G 1-50%30 afterok:$gt:$tk)
w2g=$(wj t3t_w2g $R/tasks/w2gb.txt 8 8G 1-40%15 afterok:$gt:$cp_)
w2r=$(wj t3t_w2r $R/tasks/w2rf.txt 20 12G 1-150%25 afterok:$gt:$cp_)
m1=$(lj t3t_m1 $L/m1.txt 1 16G 1:00:00 1-3 afterany:$w1g:$w1r)
s1=$(lj t3t_s1 $L/s1.txt 1 16G 1:00:00 - afterok:$m1)
m2=$(lj t3t_m2 $L/m2.txt 1 16G 1:00:00 - afterany:$w2g:$w2r)
s2=$(lj t3t_s2 $L/s2.txt 1 16G 1:00:00 - afterok:$m2:$s1)
echo "phase1 $(date +%F_%T) code=$S stage=$st g1a=$g1a g1b=$g1b g2a=$g2a g2b=$g2b tm8=$t8 tm20=$t20 gate=$gt tasks=$tk cap=$cp_ w1g=$w1g w1r=$w1r w2g=$w2g w2r=$w2r m1=$m1 s1=$s1 m2=$m2 s2=$s2" | tee -a $BASE/jobids.txt
sleep 2
squeue -u jc_905 -h -o "%.14i %.10j %.8T %.4C %R" | grep t3t_ | head -30
