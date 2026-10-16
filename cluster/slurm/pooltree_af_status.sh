#!/bin/bash
# Pooled forest all_features (2026-10-09): compact CARC status -- pieces done (npz), merged chunks, this user's task
# count / running pooled tasks / CPUs, and per array (every all_features piece array in logs/submitted_pooltrees.txt:
# waves w*, largemem) the running / pending counts and the end states (COMPLETED, FAILED, TIMEOUT, OUT_OF_MEMORY ...).
#   bash cluster/slurm/pooltree_af_status.sh            (on CARC, from /scratch1/jc_905/harxhar-pooltrees)
cd "$(dirname "$0")/../.."
P=results/pooled_trees_2026-10-07/chunks/pooltree_rf_all_features
echo "$(date +%F_%T) all_features npz $(ls $P/pieces/*.npz 2>/dev/null | grep -vc part) merged chunks $(ls $P/chunk_*.npz 2>/dev/null | wc -l)/100;" \
  "user tasks $(squeue -u "$USER" -h -r | wc -l) (cap 5000); ptrp running $(squeue -u "$USER" -h -t R -o %j | grep -c ptrp)," \
  "CPUs $(squeue -u "$USER" -h -t R -o '%j %C' | awk '/ptrp/ {s += $2} END {print s + 0}')"
for j in $(awk -F= '/pieces_pooltree_pieces_rf_all_features_(w[0-9]+|carc_largemem|lmsweep[0-9]+|wsweep[0-9]+)/ {split($2, a, " "); print a[1]}' logs/submitted_pooltrees.txt); do
  echo "  $j: $(squeue -j $j -h -r -o %T 2>/dev/null | sort | uniq -c | tr '\n' ' ')| $(sacct -X -n -P -j $j -o State 2>/dev/null | grep -v 'RUNNING\|PENDING' | sed 's/ by.*//' | sort | uniq -c | tr '\n' ' ')"
done
