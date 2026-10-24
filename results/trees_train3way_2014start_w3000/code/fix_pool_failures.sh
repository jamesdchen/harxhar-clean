#!/bin/bash
# Failed tasks of a claim pool (CARC login node): for each FAILED / NODE_FAIL / OUT_OF_MEMORY / TIMEOUT task of the pool
# array, its node, exit code, log tail, the line it claimed and that line's claim / piece state; a failure is
# 'infrastructure' when the log or state shows a bus error, an I/O or Lustre transport error, a node failure or memory,
# else 'code'.  Only when every failure is infrastructure: their nodes are added to the exclusion of the pool's pending
# tasks (scontrol update ExcNodeList), and each failed line is resumed from its checkpoint on the same CPU model by a
# one-line job (code/pool.sbatch with LO = HI = the line in a separate claim namespace, so no other line is touched);
# the line's original claim stays, marked 'resumed'.  Otherwise nothing is changed.
# usage: bash fix_pool_failures.sh <code dir> <pool array id> <constraint> <cpus> <mem> <time> <partition> <known bad nodes>
set -uo pipefail
CODE=$1
P=$2
CN=$3
NC=$4
MEM=$5
TL=$6
PART=$7
BAD0=$8
BASE=/scratch1/jc_905/trees3way
CL=$BASE/runs/claims_w3rf_full
CR=$BASE/runs/claims_w3rf_full_resume
T=$BASE/runs/tasks
J=$CODE/results/trees_train3way_2014start_w3000/code
LOGN=$(squeue -h -j $P -o %j | head -1)
[ -n "$LOGN" ] || LOGN=$(sacct -j $P -X -n -P -o JobName | head -1)
infra=1
lines=()
nodes=()
while IFS='|' read -r jid st ex el node rss; do
  t=${jid#*_}
  f=$(ls $BASE/logs/${LOGN}.${P}_${t}.out 2> /dev/null | head -1)
  k=$(grep -o -E "line [0-9]+:" "$f" 2> /dev/null | head -1 | grep -o -E "[0-9]+")
  echo "==== ${P}_$t $st $ex $el node=$node maxrss=$rss line=${k:-?}"
  tail -n 4 "$f" 2> /dev/null | cut -c1-240
  cls=code
  if grep -q -E "Bus error|Errno 108|Input/output error|Transport endpoint|Stale file handle|Cannot send after transport" "$f" 2> /dev/null; then cls=infrastructure; fi
  if [[ "$st" =~ NODE_FAIL|OUT_OF_ME|BOOT_FAIL ]] || [ "$ex" = "0:9" ]; then cls=infrastructure; fi
  echo "     class: $cls"
  if [ -n "$k" ]; then
    echo "     claim: $(ls $CL/$k 2> /dev/null | paste -sd' ') | $(cat $CL/$k/claim 2> /dev/null)"
    pc=$BASE/runs/pieces/$(sed -n "${k}p" $T/w3rf_full.txt | awk '{split($4,a,"/"); print $1 "_" $2 "_" $3 "_" a[1] "of" a[2]}')
    echo "     piece: $(ls $pc.parquet $pc.ckpt.json $pc.json 2> /dev/null | sed 's|.*/||' | paste -sd' ') $(cat $pc.ckpt.json 2> /dev/null | cut -c1-160)"
  fi
  [ "$cls" = infrastructure ] || infra=0
  lines+=("${k:-}")
  nodes+=("$node")
done < <(sacct -j $P -X -n -P -s F,NF,OOM,TO,BF -o JobID,State,ExitCode,Elapsed,NodeList,MaxRSS)
echo "failed tasks: ${#lines[@]}; all infrastructure: $infra"
if [ ${#lines[@]} -eq 0 ] || [ $infra -eq 0 ]; then
  echo "no change made"
  exit 0
fi
BAD=$(printf "%s\n" ${BAD0//,/ } "${nodes[@]}" | sort -u | paste -sd,)
scontrol update JobId=$P ExcNodeList=$BAD && echo "$P pending tasks now exclude $BAD"
mkdir -p $CR
for k in "${lines[@]}"; do
  [ -n "$k" ] || { echo "a failed task has no claimed line: left for a look"; continue; }
  if [ -f $CL/$k/done ]; then echo "line $k done meanwhile"; continue; fi
  echo "$(date +%F_%T) resumed by fix_pool_failures.sh on $CN" > $CL/$k/resumed
  j=$(cd $CODE && sbatch --parsable --account=pollok_1603 -p $PART --constraint=$CN --exclude=$BAD -J t3t_rs -c $NC \
    --mem=$MEM -t $TL -o "$BASE/logs/%x.%j.out" \
    --export=ALL,CODE=$CODE,TASKFILE=$T/w3rf_full.txt,LO=$k,HI=$k,CLAIMS=$CR,DESIGNS=/home1/jc_905/trees3way/designs \
    $J/pool.sbatch)
  echo "line $k: resume job $j"
  echo "$(date +%F_%T) line $k resumed on $CN by job $j (pool $P; excluded $BAD)" >> $BASE/jobids.txt
done
