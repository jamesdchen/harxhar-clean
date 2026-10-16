#!/bin/bash
# 2026-10-06 (moments_vix per-bar LSTM, deployment root /scratch1/jc_905/harxhar-momvix-lstm): held
# afterany on the per-bar fleet.  The LSTM part of cluster/slurm/trees_mask_merge.sbatch for this
# root's one arm: lists the chunks without DONE, merges with the unedited reducer
# experiments/reduce_lstm_chunks.py (reduce_gates.csv / reduce.log in the root), records every
# chunk's CPU class (NODE) in $L/chunk_nodes.txt and touches $L/MERGED_ALL whatever merged.
#SBATCH --job-name=mvl_merge
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x.%j.out
set -eo pipefail
module load conda >/dev/null 2>&1
source activate harxhar
set -u
export OMP_NUM_THREADS=1 TQDM_DISABLE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$PWD:$PWD/experiments"
L=results/linear_subsection_lstm_mask_momvix
LT=cluster/lstm_momvix_tasks_fleet.txt
: > "$L/chunk_nodes.txt"
while read -r BUCKET MODEL TW SEG CHUNK START END HALO; do
  [ -n "${BUCKET:-}" ] || continue
  D="$L/$BUCKET/$SEG/$MODEL/tw$TW/chunks/c$CHUNK"
  echo "c$CHUNK $(tr '\n' ' ' < "$D/NODE" 2>/dev/null || echo 'no NODE')" >> "$L/chunk_nodes.txt"
  [ -f "$D/DONE" ] && continue
  if [ -f "$D/run.log" ]; then echo "NOT DONE: $D: $(tail -1 "$D/run.log" | cut -c1-200)"; else echo "NOT DONE: $D: never started"; fi
done < "$LT"
python -u experiments/reduce_lstm_chunks.py --root "$L" --tasks "$LT" > "$L/reduce.log" 2>&1 \
  || echo "REDUCE $L: $(tail -2 "$L/reduce.log")"
echo "$L $(grep -c '^merged ' "$L/reduce.log" || true) merged; $(tail -1 "$L/reduce.log")"
cat "$L/chunk_nodes.txt"
echo "$(date)" > "$L/MERGED_ALL"
