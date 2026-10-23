#!/bin/bash
# Failed rf_skx tasks resubmitted (CARC login node, each poll): every array of $HB/arrays_skx.txt; a task that ended
# FAILED / NODE_FAIL / OUT_OF_MEMORY / TIMEOUT / BOOT_FAIL / PREEMPTED (an I/O error or SIGBUS ends the python with a
# non-zero code: pieces are written atomically and get their .json meta only after the row check, so a failed task
# never leaves a finished piece) is resubmitted once into a new array (the same task file and line, resumed from its
# checkpoint), excluding every node a task failed on; at most 3 resubmissions a (window, line).  New arrays are
# registered as well.
set -uo pipefail
HB=/home1/jc_905/trees_final
REG=$HB/arrays_skx.txt
DONE=$HB/resubmitted.txt
BAD=$HB/bad_nodes.txt
touch $REG $DONE $BAD
OC=/home1/jc_905/envs/harxhar-r13/ssl/openssl.cnf
PY=$HB/venv_skx/bin/python
cp $REG $REG.snap
while read -r A TF W R CN CODE NT; do
  idx=()
  while IFS='|' read -r jid st nodes; do
    i=${jid#*_}
    [[ "$i" =~ ^[0-9]+$ ]] || continue
    grep -qx "$A $i" $DONE && continue
    n=$(grep -c " w$W:$i\$" $DONE || true)
    echo "$A $i" >> $DONE
    if [[ "$st" =~ FAILED|NODE_FAIL|BOOT_FAIL ]] && [ -n "$nodes" ] && [ "$nodes" != "None" ]; then
      echo "$nodes" >> $BAD  # a node a task failed on (not a time limit or memory)
    fi
    if [ "$n" -ge 3 ]; then echo "w$W line $i: 3 resubmissions already, left for a look"; continue; fi
    echo "resub w$W:$i" >> $DONE.log
    echo "x w$W:$i" >> $DONE
    idx+=("$i")
  done < <(sacct -j "$A" -X -n -P -o JobID,State,NodeList 2>/dev/null | grep -E "FAILED|NODE_FAIL|OUT_OF_ME|TIMEOUT|BOOT_FAIL|PREEMPTED")
  if [ ${#idx[@]} -gt 0 ]; then
    ex=$(sort -u $BAD | paste -sd, -)
    arr=$(IFS=,; echo "${idx[*]}")
    T=6:00:00; [ "${NT:-8}" -ge 20 ] && T=3:30:00  # 2026-10-23 08:55: 4 GB and tight time limits (were 16G, 8:00:00)
    j=$(cd $CODE && sbatch --parsable -J tfs_w${W}r --array=$arr -c ${NT:-8} --mem=4G -t $T --constraint="$CN" \
      ${ex:+--exclude=$ex} -o "$HB/logs/%x.%A_%a.out" \
      --export=ALL,MODE=walk,TASKFILE=$TF,HARXHAR_PY=$PY,DESIGN=$HB/designs/w$W,OUT=$R,WINDOW_DAYS=$W,OPENSSL_CONF=$OC \
      cluster/slurm/pooltree_rerun13.sbatch)
    echo "$j $TF $W $R $CN $CODE ${NT:-8}" >> $REG
    echo "$(date +%F_%T) resubmitted w$W lines $arr (from $A, excluding $ex): array $j" | tee -a $HB/jobids.txt
  fi
done < $REG.snap
