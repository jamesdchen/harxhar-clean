#!/bin/bash
# Merge, then (best effort) score, the per-bar LSTM arms
# (results/linear_subsection_lstm).  Held on the LSTM fleet with afterany, so a
# failed chunk does not block the rest.  The twin of treestuned_score.sbatch:
#   1. every chunk of the fleet task file is walked: the ones without DONE are
#      listed (NOT DONE + the last line of their run.log, or "never started");
#   2. experiments/reduce_lstm_chunks.py merges every arm whose chunks are all
#      DONE into the unchunked layout
#      $R/<bucket>/<seg>/lstm/tw<TW>/causal_tune_lstm/lstm/<bucket>/
#      and prints NOT MERGED for the rest;
#   3. experiments/score_lstm_subsection.py runs on what merged (best effort:
#      the definitive scoring runs locally after the pull, against the local
#      linear and tree arms); SCORED is touched whatever happened.
# RESULTS_ROOT overrides the root.
#SBATCH --job-name=lstm_score
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
export OMP_NUM_THREADS=1 TQDM_DISABLE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD:$PWD/experiments:$PWD/notebooks"
R="${RESULTS_ROOT:-results/linear_subsection_lstm}"
FLEET=cluster/lstm_tasks_fleet.txt
mkdir -p "$R"

EXPECTED=0
FINISHED=0
while read -r BUCKET MODEL TW SEG CHUNK START END HALO; do
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

python -u experiments/reduce_lstm_chunks.py --root "$R" --tasks "$FLEET" > "$R/reduce.log" 2>&1 \
  || echo "REDUCE FAILED for $R: $(tail -2 "$R/reduce.log")"
grep 'NOT MERGED\|GATE FAIL\|merged ' "$R/reduce.log" || true

python -u experiments/score_lstm_subsection.py --root "$R" --out "$R/score_carc" \
  > "$R/score.log" 2>&1 \
  || echo "SCORER FAILED for $R (best effort on the cluster): $(tail -2 "$R/score.log")"
touch "$R/SCORED"
