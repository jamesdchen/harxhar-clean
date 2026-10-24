#!/bin/bash
# Note to addendum C/E (CARC login node), 2026-10-24 ~10:40 PDT: throughput-balanced split of the ALL48 forest full-set
# lines not yet claimed, after the Hoffman2 gates passed (coordinator 10:25: Hoffman2 ~14 lines/h at a cap of ~120
# 8-thread tasks) and the Xeon 4116 pool's /scratch1 failures (it moves to /home1).  Replaces move_home1.sh, which
# stopped (set -e on an empty grep) after releasing lines 7-9 and before changing anything else.
#   1. the old Xeon 4116 pool's failed tasks: claimed lines without a written piece released; its pending tasks cancelled
#   2. U = the unclaimed lines of 5-348 (the /scratch1 claim namespace), n = |U|; CARC takes the lowest nC of U and
#      Hoffman2 the rest from the top down, nC from equal finish times (Hoffman2's 20-line wave 1 counted), Hoffman2's
#      share rounded down to a multiple of 10:  nC = ceil(rC (n + 20) / (rC + rH)), rH = 14.0, rC = 4.5 lines/h
#   3. Hoffman2's lines pre-claimed in the /scratch1 namespace (mkdir, marker owner_h2) so no CARC pool takes them; a
#      line claimed meanwhile stays CARC's
#   4. CARC's block: its lowest 30 % for the /scratch1 pools (EPYC, E5-2640 v4), the rest (B .. topC) for a new Xeon
#      4116 pool on /home1 (code, claims, pieces, logs off /scratch1): reserved in the /scratch1 namespace
#      (reserved_home1); a line of B .. topC claimed on /scratch1 is marked taken_scratch1 in the /home1 namespace
#   5. runs/tasks/h2_assignment.txt rewritten (rows ASSIGNED: 349-368 wave 1, and the new Hoffman2 lines as ranges);
#      the held final merge waits on the new pool
# usage: bash rebalance_h2.sh <commit sha> <old Xeon 4116 pool id> <held merge id> <EPYC pool id> <E5-2640 v4 pool id>
set -uo pipefail
C=$1
OLD=$2
M4=$3
PE=$4
P26=$5
SB=/scratch1/jc_905/trees3way
HB=/home1/jc_905/trees3way
CLS=$SB/runs/claims_w3rf_full
CLH=$HB/claims_w3rf_full
TF=$SB/runs/tasks/w3rf_full.txt
RH=14.0
RC=4.5
WAVE1=20
if grep -q "^rebalance_h2 " $SB/jobids.txt; then
  echo "ABORT: already done"
  exit 3
fi
S=$(git -C $SB/repo rev-parse --short=7 "$C^{commit}") || exit 3
HC=$HB/code_$S
if [ ! -d $HC ]; then
  mkdir -p $HC
  P=$(git -C $SB/repo ls-tree --name-only "$C" | grep -v '^results$')
  git -C $SB/repo archive "$C" $P results/trees_train3way_2014start_w3000 | tar -x -C $HC || exit 3
  git -C $SB/repo rev-parse "$C^{commit}" > $HC/COMMIT
fi
mkdir -p $HB/runs/pieces $HB/runs/tasks $HB/logs $CLH
cp $TF $HB/runs/tasks/w3rf_full.txt && cmp $TF $HB/runs/tasks/w3rf_full.txt || exit 3
piece() { echo $SB/runs/pieces/$(sed -n "${1}p" $TF | awk '{split($4,a,"/"); print $1 "_" $2 "_" $3 "_" a[1] "of" a[2]}'); }
# 1. the old pool
while IFS='|' read -r jid st ex node; do
  k=$(grep -l -E "^[^ ]+ ${jid} " $CLS/*/claim 2> /dev/null | head -1 | xargs -r dirname | xargs -r basename)
  if [ -z "$k" ]; then continue; fi
  pc=$(piece $k)
  if [ -e $CLS/$k/done ] || [ -e $pc.parquet ] || [ -e $pc.json ] || [ -e $pc.ckpt.json ]; then  # 2026-10-24 fix: was `ls a b c`, false when any one is absent
    echo "$jid line $k: a piece exists or done -> left claimed"
    continue
  fi
  rm -f $pc.part
  echo "$(date +%F_%T) released line $k (claim of $jid on $node, $st $ex; no piece written): $(cat $CLS/$k/claim 2> /dev/null)" >> $SB/runs/claims_released.log
  rm -rf "$CLS/$k"
  echo "$jid line $k on $node: released"
done < <(sacct -j $OLD -X -n -P -s F,NF,OOM,TO,BF -o JobID,State,ExitCode,NodeList)
bad=$(sacct -j $OLD -X -n -P -s F,NF,OOM,TO,BF -o NodeList | sort -u | paste -sd,)
BAD=$(printf "%s\n" d05-06 d06-25 d11-23 d11-24 d18-17 d18-26 d18-28 d18-29 ${bad//,/ } | sort -u | paste -sd,)
scancel --state=PENDING $OLD
sleep 2
echo "old pool $OLD: running $(squeue -h -r -j $OLD -t R | wc -l), pending $(squeue -h -r -j $OLD -t PD | wc -l); bad nodes $BAD"
# 2. the split
U=()
for ((k = 5; k <= 348; k++)); do [ -e $CLS/$k ] || U+=("$k"); done
n=${#U[@]}
nC=$(python3 -c "import math; print(math.ceil($RC * ($n + $WAVE1) / ($RC + $RH)))")
nH=$(((n - nC) / 10 * 10))
nC=$((n - nH))
nS=$(python3 -c "print(round($nC * 0.3))")
echo "unclaimed n=$n -> CARC nC=$nC (scratch pools nS=$nS, /home1 pool $((nC - nS))), Hoffman2 nH=$nH"
# 3. Hoffman2's lines (the top nH of U), pre-claimed
H=()
for ((i = nC; i < n; i++)); do
  k=${U[$i]}
  if mkdir $CLS/$k 2> /dev/null; then
    echo "$(date +%F_%T) owner h2 (Hoffman2; runs/tasks/h2_assignment.txt)" > $CLS/$k/owner_h2
    H+=("$k")
  else
    echo "line $k claimed meanwhile on CARC: stays CARC's"
  fi
done
# 4. CARC's block: U[0 .. nS-1] for the /scratch1 pools, U[nS .. nC-1] for the /home1 Xeon 4116 pool
B=${U[$nS]}
TOPC=${U[$((nC - 1))]}
res=0
for ((k = B; k <= TOPC; k++)); do
  if mkdir $CLS/$k 2> /dev/null; then
    echo "$(date +%F_%T) reserved for the /home1 Xeon 4116 pool ($CLH)" > $CLS/$k/reserved_home1
    res=$((res + 1))
  else
    mkdir -p $CLH/$k
    echo "$(date +%F_%T) claimed in $CLS: $(ls $CLS/$k | paste -sd' ') $(cat $CLS/$k/claim 2> /dev/null)" > $CLH/$k/taken_scratch1
  fi
done
J=$HC/results/trees_train3way_2014start_w3000/code
nt=$((res + 5))
p=$(cd $HC && sbatch --parsable --account=pollok_1603 -p main --constraint=xeon-4116 --exclude=$BAD -J t3t_p41h -c 8 \
  --mem=16G -t 20:00:00 --array=1-$nt%40 -o "$HB/logs/%x.%A_%a.out" \
  --export=ALL,CODE=$HC,T3T_BASE=$HB,TASKFILE=$HB/runs/tasks/w3rf_full.txt,LO=$B,HI=$TOPC,CLAIMS=$CLH,DESIGNS=$HB/designs \
  $J/pool.sbatch) || p=""
echo "/home1 pool: ${p:-REFUSED} ($nt tasks %40) lines $B-$TOPC ($res reserved)"
# 5. the assignment file
ranges=$(printf "%s\n" "${H[@]}" | sort -n | awk 'NR==1{s=$1;e=$1;next} $1==e+1{e=$1;next} {printf "%s,", (s==e?s:s"-"e); s=$1;e=$1} END{if (NR) printf "%s", (s==e?s:s"-"e)}')
A=$SB/runs/tasks/h2_assignment.txt
cp $A $A.bak_$(date +%H%M)
{
  grep "^#" $A
  echo "# $(date +%F_%T) rebalance_h2.sh (throughput-balanced, coordinator 10:25): Hoffman2 ~14 lines/h at <= ~120"
  echo "#   concurrent 8-thread tasks (Gold 6342 ~35 at 5.4 h a line; Gold 6140 / 6240 / E5-2650 v4 ~85 at ~11.3 h), CARC"
  echo "#   ~4.5 lines/h for new lines; the second row's lines are pre-claimed 'owner_h2' in runs/claims_w3rf_full/."
  echo "ASSIGNED w3rf_full.txt 349-368 ALL48 rf_skx all_features 8"
  echo "ASSIGNED w3rf_full.txt $ranges ALL48 rf_skx all_features 8"
} > $A.new && mv $A.new $A
grep -v "^#" $A
[ -n "$p" ] && scontrol update JobId=$M4 Dependency=afterany:13372432:$PE:$OLD:$P26:$p:13372427:13371788:13371790 && echo "m4 $M4 waits on $p too"
echo "rebalance_h2 $(date +%F_%T) n=$n nC=$nC nS=$nS nH=${#H[@]} B=$B topC=$TOPC h2_lines=$ranges p41h=${p:-refused} bad=$BAD old_pending_cancelled=$OLD" | tee -a $SB/jobids.txt
