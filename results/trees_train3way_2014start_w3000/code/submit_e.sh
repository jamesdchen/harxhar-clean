#!/bin/bash
# RULE.txt addendum E (CARC login node): the forest's full set on ALL48 beyond the 20-thread EPYC arrays.
#   gates of two further CARC CPU models, one tag each (gate/carc_<tag>/g1, g2): x4116 (Xeon Silver 4116; main and
#   oneweek) and x2640v4 (Xeon E5-2640 v4; oneweek): gate 1 at 8 threads, gate 2 every row vs scored at 20 threads on
#   the ALL48 forest full set (+ a scored-only repeat at 8 threads) and at 8 threads on its HAR set and ALL48_N3k;
#   checked by accept --require-tag (exit 3 unless the tag passes).
#   the not-yet-started lines (5-348 of runs/tasks/w3rf_full.txt; 1-4 run in 13372432; 349-368 are Hoffman2's) go to a
#   claim pool (code/pool.sbatch; one atomic claim a line): EPYC 7513 / 7542 at 8 threads now, Xeon 4116 at 8 threads
#   and Xeon E5-2640 v4 at 20 threads after their gates; the pending tasks of 13372432 / 13372801 are cancelled.
#   the held final merge's dependency is extended to the pools.
# usage: bash submit_e.sh <code dir> <held merge job id>
set -euo pipefail
CODE=$1
M4=$2
BASE=/scratch1/jc_905/trees3way
R=$BASE/runs
L=$BASE/lines
T=$R/tasks
G=$BASE/gate
J=$CODE/results/trees_train3way_2014start_w3000/code
X="experiments/trees_train3way.py"
D="--design-root $BASE/designs"
CL=$R/claims_w3rf_full
if grep -q "^phase_e " $BASE/jobids.txt; then
  echo "ABORT: already submitted"
  exit 3
fi
[ ! -e $CL ] || { echo "ABORT: $CL exists"; exit 3; }
echo "partition time limits: $(sinfo -h -o '%P=%l' | sort -u | tr '\n' ' ')"
W48="--from 2014-01-02T15:30 --to 2014-01-03T10:00"
W3K="--from 2014-01-02T15:30 --to 2014-01-06T10:30"
for tag in carc_x4116 carc_x2640v4; do
  mkdir -p $G/$tag/g1 $G/$tag/g2
  O1="--out $G/$tag/g1"
  O2="--out $G/$tag/g2"
  {
    for s in baseline all_features; do echo "skx $X walk --arm RTH13 --est rf_skx --set $s --ends 10 --tag ends10_t8 $D $O1"; done
    echo "skx $X walk --arm ALL48 --est rf_skx --set all_features $W48 --tag scored2 $D $O2"
    for how in every scored; do
      ev=""
      if [ $how = every ]; then ev="--every-row"; fi
      echo "skx $X walk --arm ALL48 --est rf_skx --set baseline $W48 $ev --tag $how $D $O2"
      for s in baseline all_features; do echo "skx $X walk --arm ALL48_N3k --est rf_skx --set $s $W3K $ev --tag $how $D $O2"; done
    done
  } > $L/ge_${tag}_t8.txt
  {
    echo "skx $X walk --arm ALL48 --est rf_skx --set all_features $W48 --every-row --tag every $D $O2"
    echo "skx $X walk --arm ALL48 --est rf_skx --set all_features $W48 --tag scored $D $O2"
  } > $L/ge_${tag}_t20.txt
  echo "gb $X accept --gate-root $G --require-tag $tag --out $R" > $L/ge_${tag}_chk.txt
done
cd $CODE
sb() { sbatch --parsable --account=pollok_1603 "$@"; }
gl() {  # <name> <partition> <constraint> <cpus> <mem> <time> <lines> [dependency]
  local n
  n=$(grep -c . "$7")
  local dep=()
  if [ -n "${8:-}" ]; then dep=(--dependency="$8"); fi
  sb -p "$2" --constraint="$3" -J "$1" -c "$4" --mem="$5" -t "$6" --array="1-$n" "${dep[@]}" \
    -o "$BASE/logs/%x.%A_%a.out" --export=ALL,LINES=$7,CODE=$CODE $J/line.sbatch
}
a1=$(gl t3t_ge41a main,oneweek xeon-4116 8 16G 4:00:00 $L/ge_carc_x4116_t8.txt)
b1=$(gl t3t_ge41b main,oneweek xeon-4116 20 24G 8:00:00 $L/ge_carc_x4116_t20.txt)
c1=$(gl t3t_ge41c main xeon-4116 1 8G 0:30:00 $L/ge_carc_x4116_chk.txt afterok:$a1:$b1)
a2=$(gl t3t_ge26a oneweek xeon-2640v4 8 16G 4:00:00 $L/ge_carc_x2640v4_t8.txt)
b2=$(gl t3t_ge26b oneweek xeon-2640v4 20 24G 8:00:00 $L/ge_carc_x2640v4_t20.txt)
c2=$(gl t3t_ge26c oneweek xeon-2640v4 1 8G 0:30:00 $L/ge_carc_x2640v4_chk.txt afterok:$a2:$b2)
# move the not-yet-started forest lines: the pools are submitted HELD, then the pending tasks of the two EPYC arrays
# are cancelled (running tasks untouched), then the pools are released -- no line can run twice
mkdir -p $CL
PE=(--export=ALL,CODE=$CODE,TASKFILE=$T/w3rf_full.txt,LO=5,HI=348,CLAIMS=$CL)
pE=$(sb -H -p main --constraint="epyc-7513|epyc-7542" -J t3t_pE -c 8 --mem=16G -t 24:00:00 --array=1-150%20   -o "$BASE/logs/%x.%A_%a.out" "${PE[@]}" $J/pool.sbatch)
p41=$(sb -H -p main,oneweek --constraint=xeon-4116 -J t3t_p41 -c 8 --mem=16G -t 24:00:00 --array=1-200%40   --dependency=afterok:$c1 -o "$BASE/logs/%x.%A_%a.out" "${PE[@]}" $J/pool.sbatch)
p26=$(sb -H -p oneweek --constraint=xeon-2640v4 -J t3t_p26 -c 20 --mem=24G -t 24:00:00 --array=1-60%10   --dependency=afterok:$c2 -o "$BASE/logs/%x.%A_%a.out" "${PE[@]}" $J/pool.sbatch)
scancel --state=PENDING 13372432 13372801
sleep 3
left=$(squeue -h -r -j 13372432,13372801 -t PD -o %i | wc -l)
echo "13372432 running: $(squeue -h -r -j 13372432 -t R -o %i | paste -sd' '); pending left in 13372432 / 13372801: $left"
if [ "$left" -ne 0 ]; then
  echo "ABORT: pending tasks remain; the pools stay held"
  exit 3
fi
scontrol release $pE $p41 $p26
scontrol update JobId=$M4 Dependency=afterany:13372432:$pE:$p41:$p26:13372427:13371788:13371790 && echo "m4 $M4 now waits on the pools too"
scontrol show job $M4 | grep -o -E "JobState=[A-Z]+|Reason=[A-Za-z]+" | tr '\n' ' '
echo
echo "phase_e $(date +%F_%T) code=$CODE ge41a=$a1 ge41b=$b1 ge41c=$c1 ge26a=$a2 ge26b=$b2 ge26c=$c2 pE=$pE p41=$p41 p26=$p26 claims=$CL lines 5-348; cancelled pending tasks of 13372432 and 13372801" | tee -a $BASE/jobids.txt
