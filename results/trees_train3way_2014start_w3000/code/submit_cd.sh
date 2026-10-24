#!/bin/bash
# RULE.txt addenda C and D (CARC login node):
#   D: gate D (LightGBM deterministic + force_col_wise on ALL48 / ALL48_N39k; fresh single fits on EPYC 7513 and 7542),
#      the addendum-B LightGBM walks and their merges / scores cancelled (never started) and resubmitted on this code
#      afterok gate D; interim scores scores_w2 (ALL48_N39k) and scores_w3 (ALL48 without the forest's full set).
#   C: the forest's full set on ALL48, lines 1-239 of runs/tasks/w3rf_full.txt on CARC (20 threads, throttled); lines
#      240-368 for Hoffman2 (runs/tasks/h2_assignment.txt); the ALL48 merge with every class (intake of the Hoffman2
#      pieces) and the final scores as a HELD job, released when the Hoffman2 pieces are in.
# usage: bash submit_cd.sh <code dir> <commit sha>
set -euo pipefail
CODE=$1
SHA=$2
BASE=/scratch1/jc_905/trees3way
R=$BASE/runs
L=$BASE/lines
T=$R/tasks
J=$CODE/results/trees_train3way_2014start_w3000/code
X="experiments/trees_train3way.py"
D="--design-root $BASE/designs"
if grep -q "^phase_cd " $BASE/jobids.txt; then
  echo "ABORT: already submitted"
  exit 3
fi
grep -q '"pass": true' $BASE/gate/g2/gate2_ALL48_rf_skx.json || { echo "ABORT: the forest's gate 2 on ALL48"; exit 3; }
mkdir -p $R/pieces_h2 $BASE/gate/h2 $BASE/gate/gd $BASE/gate/gdb
[ -f $T/h2_assignment.txt ] || cp $J/h2_assignment.txt $T/h2_assignment.txt
echo "$SHA" > $T/h2_code_commit.txt
# gate D lines: 7513 (a) and 7542 (b)
W48="--from 2014-01-02T15:30 --to 2014-01-03T10:00"
ONE="--from 2014-01-03T10:00 --to 2014-01-03T10:00"
GD=$BASE/gate/gd
: > $L/gda.txt
: > $L/gdb_.txt
for s in baseline all_features; do
  {
    echo "gb $X walk --arm ALL48 --est lgbm --set $s $W48 --every-row --tag every $D --out $GD"
    echo "gb $X walk --arm ALL48 --est lgbm --set $s $W48 --tag scored $D --out $GD"
    for k in 1 2 3 4; do echo "gb $X walk --arm ALL48 --est lgbm --set $s $ONE --tag one$k $D --out $GD"; done
    echo "gb $X walk --arm ALL48_N39k --est lgbm --set $s $W48 --tag scored $D --out $GD"
    for k in 1 2; do echo "gb $X walk --arm ALL48_N39k --est lgbm --set $s $ONE --tag one$k $D --out $GD"; done
    echo "gb $X walk --arm RTH13 --est lgbm --set $s --ends 10 --lgbm-col-wise on --lgbm-deterministic on --tag ends10_cd $D --out $BASE/gate/gdb"
  } >> $L/gda.txt
  {
    echo "gb $X walk --arm ALL48 --est lgbm --set $s $W48 --tag scored2 $D --out $GD"
    for k in 5 6 7 8; do echo "gb $X walk --arm ALL48 --est lgbm --set $s $ONE --tag one$k $D --out $GD"; done
    echo "gb $X walk --arm ALL48_N39k --est lgbm --set $s $W48 --tag scored2 $D --out $GD"
    for k in 3 4; do echo "gb $X walk --arm ALL48_N39k --est lgbm --set $s $ONE --tag one$k $D --out $GD"; done
  } >> $L/gdb_.txt
done
printf "%s\n" "gb $X gate2 --arm ALL48 --est lgbm --out $GD" "gb $X gaterep --min-hosts 2 --out $GD" > $L/gdc.txt
echo "gb $X gate1 --no-fail --out $BASE/gate/gdb" > $L/gdr.txt
echo "gb $X merge --arm ALL48_N39k $D --out $R --require-all" > $L/m2d.txt
echo "gb $X merge --arm ALL48 $D --out $R --require-all --skip rf_skx/all_features" > $L/m3d.txt
echo "gb $X merge --arm ALL48 $D --out $R --require-all --intake $R/pieces_h2 --accept-cpus $R/pieces_h2/accepted_cpus.json" > $L/m4.txt
echo "sweep experiments/trees_train3way_score.py $R $BASE/scores_w2" > $L/s2d.txt
echo "sweep experiments/trees_train3way_score.py $R $BASE/scores_w3" > $L/s3d.txt
echo "sweep experiments/trees_train3way_score.py $R $BASE/scores" > $L/s4.txt
A=(--parsable --account=pollok_1603 --partition=main)
C2="--constraint=epyc-7513|epyc-7542"
lj() {  # <name> <lines> <cpus> <mem> <time> <array or -> <constraint> [dependency] [extra]
  local arr=()
  local dep=()
  local xf=()
  if [ "$6" != - ]; then arr=(--array="$6"); fi
  if [ -n "${8:-}" ]; then dep=(--dependency="$8"); fi
  if [ -n "${9:-}" ]; then xf=("$9"); fi
  sbatch "${A[@]}" "$7" -J "$1" -c "$3" --mem="$4" -t "$5" "${arr[@]}" "${dep[@]}" "${xf[@]}" \
    -o "$BASE/logs/%x.%A_%a.out" --export=ALL,LINES=$2,CODE=$CODE $J/line.sbatch
}
wj() {  # <name> <taskfile> <cpus> <mem> <array> <dependency or ->
  local dep=()
  if [ "$6" != - ]; then dep=(--dependency="$6"); fi
  sbatch "${A[@]}" "$C2" -J "$1" -c "$3" --mem="$4" -t 12:00:00 --array="$5" "${dep[@]}" \
    -o "$BASE/logs/%x.%A_%a.out" --export=ALL,CODE=$CODE,TASKFILE=$2 $J/walk.sbatch
}
cd $CODE
# cancel the addendum-B LightGBM walks and their merges / scores (never started): merges / scores first
scancel 13371791 13371792 13371793 13371794
scancel 13371785 13371787
gda=$(lj t3t_gda $L/gda.txt 8 16G 2:00:00 "1-$(grep -c . $L/gda.txt)" --constraint=epyc-7513)
gdb=$(lj t3t_gdb $L/gdb_.txt 8 16G 2:00:00 "1-$(grep -c . $L/gdb_.txt)" --constraint=epyc-7542)
gdc=$(lj t3t_gdc $L/gdc.txt 1 8G 0:30:00 1-2 "$C2" afterok:$gda:$gdb)
gdr=$(lj t3t_gdr $L/gdr.txt 1 8G 0:30:00 - "$C2" afterany:$gda)
w2l=$(wj t3t_w2l $T/w2lgbm.txt 8 8G "1-$(grep -c . $T/w2lgbm.txt)%12" afterok:$gdc)
w3l=$(wj t3t_w3l $T/w3lgbm.txt 8 16G "1-$(grep -c . $T/w3lgbm.txt)%15" afterok:$gdc)
m2=$(lj t3t_m2 $L/m2d.txt 1 16G 1:00:00 - "$C2" afterany:$w2l:13371786:13371789)
s2=$(lj t3t_s2 $L/s2d.txt 1 16G 1:00:00 - "$C2" afterok:$m2,afterany:13369760)
m3=$(lj t3t_m3 $L/m3d.txt 1 16G 1:00:00 - "$C2" afterany:$w3l:13371788:13371790)
s3=$(lj t3t_s3 $L/s3d.txt 1 16G 1:00:00 - "$C2" afterok:$m3,afterany:$s2)
# addendum C: the forest's full set on ALL48, CARC lines 1-239
w3f=$(wj t3t_w3f $T/w3rf_full.txt 20 16G "1-239%25" -)
m4=$(lj t3t_m4 $L/m4.txt 1 16G 2:00:00 - "$C2" afterany:$w3f:$w3l:13371788:13371790 --hold)
s4=$(lj t3t_s4 $L/s4.txt 1 16G 1:00:00 - "$C2" afterok:$m4,afterany:$s3)
echo "phase_cd $(date +%F_%T) code=$CODE gda=$gda gdb=$gdb gdc=$gdc gdr=$gdr w2l=$w2l w3l=$w3l m2=$m2 s2=$s2 m3=$m3 s3=$s3 w3f=$w3f m4=$m4(held) s4=$s4; cancelled 13371785 13371787 13371791-13371794" | tee -a $BASE/jobids.txt
sleep 2
squeue -u jc_905 -h -o "%j %T %R" | grep t3t_ | sort | uniq -c
