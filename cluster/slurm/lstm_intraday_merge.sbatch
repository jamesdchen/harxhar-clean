#!/bin/bash
# I10 (2026-09-29): held on the stage-2 fleet with afterany.  Lists the chunks of the fleet task file
# without DONE, merges every arm whose chunks are all DONE with
# experiments/reduce_lstm_intraday_chunks.py (NOT MERGED for the rest) and touches MERGED_ALL
# whatever happened.  No scoring on the cluster: the stacks are built and scored locally after the pull.
#SBATCH --job-name=lstmi_merge
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --output=logs/%x.%j.out
set -eo pipefail
module load conda >/dev/null 2>&1
source activate harxhar
set -u
export OMP_NUM_THREADS=1 TQDM_DISABLE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$PWD"
R="${RESULTS_ROOT:-results/linear_subsection_lstm_intraday}"
FLEET="${FLEET:-cluster/lstm_intraday_tasks_fleet.txt}"
EXPECTED=0
FINISHED=0
while read -r BUCKET MODEL TW SEG CHUNK START END HALO STAGE; do
  [ -n "${BUCKET:-}" ] || continue
  EXPECTED=$((EXPECTED + 1))
  D="$R/$BUCKET/$SEG/$MODEL/tw$TW/chunks/c$CHUNK"
  if [ -f "$D/DONE" ]; then
    FINISHED=$((FINISHED + 1))
  elif [ -f "$D/run.log" ]; then
    echo "NOT DONE: $D [$START, $END): $(tail -1 "$D/run.log" | cut -c1-200)"
  else
    echo "NOT DONE: $D [$START, $END): never started (no run.log)"
  fi
done < "$FLEET"
echo "$R: chunks finished $FINISHED of $EXPECTED"
python -u experiments/reduce_lstm_intraday_chunks.py --root "$R" --tasks "$FLEET" > "$R/reduce.log" 2>&1 \
  || echo "REDUCE FAILED for $R: $(tail -2 "$R/reduce.log")"
grep 'NOT MERGED\|GATE FAIL\|merged ' "$R/reduce.log" || true
echo "chunks finished $FINISHED of $EXPECTED  $(date)" > "$R/MERGED_ALL"
