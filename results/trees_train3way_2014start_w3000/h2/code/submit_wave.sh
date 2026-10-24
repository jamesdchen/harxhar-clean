#!/bin/bash
# Hoffman2 login node (bash -l): one wave of forest walk pieces.  Owner (02:40 PDT, relayed): never queue more lines
# than can start in the current round (unstarted lines stay reassignable to CARC; only started lines are Hoffman2's),
# and before each wave run only lines still assigned to Hoffman2.  So, after runs/tasks/h2_assignment.txt has been
# re-relayed from CARC:
#   the n highest-numbered lines of CARC's runs/tasks/w3rf_full.txt (carc/runs/tasks/, transfer 1) inside an ASSIGNED
#   range of the current assignment, with no piece, meta or checkpoint on Hoffman2 and in no earlier wave
#   -> runs/tasks/h2_wave<k>.txt; one UGE array of those n tasks with -tc n (h2/code/walk.sh: threads = slots, re-checks the
#   assignment, a gate-passed CPU model and the piece's CPU model at its start), arch and h_rt from h2/walk_resources.txt
#   (the gate summary; h_rt override as the second argument), $NT threads = slots (default 20; 8 where Hoffman2 does
#   not schedule 20-slot jobs, the gated thread count), h_data per slot $HD (default 1536M at 20 slots, 2G at 8).
# usage: bash submit_wave.sh <n lines> [h_rt]
set -euo pipefail
B=/u/scratch/j/jamesdc1/t3t
N=${1:?n lines}
NT=${NT:-20}
if [ "$NT" -ge 16 ]; then HD=${HD:-1536M}; else HD=${HD:-2G}; fi
cd $B
[ -s h2/passed_cpus.txt ] && [ -s h2/walk_resources.txt ] || { echo "ABORT: no gate-passed CPU model"; exit 3; }
. h2/walk_resources.txt
HRT=${2:-$HRT}
k=1
while [ -e runs/tasks/h2_wave$k.txt ]; do k=$((k + 1)); done
F=runs/tasks/h2_wave$k.txt
SRC=carc/runs/tasks/w3rf_full.txt
cat runs/tasks/h2_wave*.txt 2>/dev/null | tr -d '\r' > h2/.prev_waves || true
awk '$1 == "ASSIGNED" && $2 == "w3rf_full.txt" {split($3, r, "-"); for (i = r[1]; i <= r[2]; i++) print i}' \
  runs/tasks/h2_assignment.txt | sort -rn -u > h2/.assigned_lines
n=0
: > $F.tmp
while read -r L; do
  [ $n -lt $N ] || break
  LINE=$(sed -n "${L}p" $SRC | tr -d '\r')
  read -r ARM EST SET PART <<< "$LINE"
  [ "${PART%/*}" = "$L" ] || { echo "ABORT: line $L of $SRC is '$LINE'"; exit 3; }
  P=runs/pieces/${ARM}_${EST}_${SET}_${L}of${PART#*/}
  if [ -e $P.parquet ] || [ -e $P.json ] || [ -e $P.ckpt.json ] || grep -qxF "$LINE" h2/.prev_waves; then continue; fi
  echo "$LINE" >> $F.tmp
  n=$((n + 1))
done < h2/.assigned_lines
[ $n -gt 0 ] || { rm -f $F.tmp; echo "no assigned line left to start"; exit 0; }
mv $F.tmp $F
echo w3rf_full.txt > $F.source
j=$(qsub -terse -N t3t_w$k -t 1-$n -tc $n -pe shared $NT -l "h_rt=$HRT,h_data=$HD,arch=$ARCH" \
  -o "$B/logs/w$k.\$TASK_ID.log" -v "TASKFILE=$B/$F" jobs/walk.sh | cut -d. -f1)
echo "wave $k $(date +%F_%T) job=$j lines=$n ($(head -1 $F | awk '{print $4}') .. $(tail -1 $F | awk '{print $4}')) tc=$n threads=$NT h_rt=$HRT h_data=$HD arch=$ARCH" | tee -a jobids.txt
