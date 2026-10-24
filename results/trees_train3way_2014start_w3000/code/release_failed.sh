#!/bin/bash
# The failed tasks of a claim pool whose failures were /scratch1 Lustre errors on their nodes (I/O error, transport
# endpoint shutdown, or a log that could not be written on such a node): their nodes are added to the exclusion of the
# pool's pending tasks; each line they had claimed is released (claim removed, logged) only if no piece of it was
# written (no parquet, json or checkpoint: a stray .part is removed), so any pool may run it from scratch.  A line
# with a partial piece is left claimed for a look.
# usage: bash release_failed.sh <pool array id> <known bad nodes>
set -uo pipefail
P=$1
BAD0=$2
BASE=/scratch1/jc_905/trees3way
CL=$BASE/runs/claims_w3rf_full
T=$BASE/runs/tasks
LOG=$BASE/runs/claims_released.log
nodes=()
while IFS='|' read -r jid st ex node; do
  nodes+=("$node")
  k=$(grep -l -E "^[^ ]+ ${jid} " $CL/*/claim 2> /dev/null | head -1 | xargs -r dirname | xargs -r basename)
  if [ -z "$k" ]; then
    echo "$jid ($st $ex, $node): no claim found"
    continue
  fi
  pc=$BASE/runs/pieces/$(sed -n "${k}p" $T/w3rf_full.txt | awk '{split($4,a,"/"); print $1 "_" $2 "_" $3 "_" a[1] "of" a[2]}')
  if [ -e $CL/$k/done ] || ls $pc.parquet $pc.json $pc.ckpt.json > /dev/null 2>&1; then
    echo "$jid line $k: a piece exists or done -> left claimed ($(ls $pc.* 2> /dev/null | sed 's|.*/||' | paste -sd' '))"
    continue
  fi
  rm -f $pc.part
  echo "$(date +%F_%T) released line $k (claim of $jid on $node, $st $ex; no piece written): $(cat $CL/$k/claim 2> /dev/null)" >> $LOG
  rm -rf "$CL/$k"
  echo "$jid line $k on $node: released"
done < <(sacct -j $P -X -n -P -s F,NF,OOM,TO,BF -o JobID,State,ExitCode,NodeList)
BAD=$(printf "%s\n" ${BAD0//,/ } "${nodes[@]}" | sort -u | paste -sd,)
scontrol update JobId=$P ExcNodeList=$BAD && echo "$P pending tasks now exclude $BAD"
echo "$(date +%F_%T) pool $P: failed lines released, pending tasks exclude $BAD" >> $BASE/jobids.txt
