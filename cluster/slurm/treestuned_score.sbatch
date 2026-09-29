#!/bin/bash
# Merge, then score, the per-bar TUNED tree arms (results/linear_subsection_trees_tuned)
# beside the untuned tree arms (results/linear_subsection_trees) and the
# per-bar linear arms (results/linear_subsection, _lassofix).  Held on both
# tuned-tree fleets with afterany, so a failed chunk does not block the rest.
#   1. every chunk-arm of the task files (one per line x model) is walked: the
#      ones without DONE are listed (NOT DONE + the last line of their run.log,
#      or "never started");
#   2. experiments/reduce_trees_tuned_chunks.py merges every arm whose chunks
#      are all DONE into the unchunked layout
#      $R/<bucket>/<seg>/<model>/tw<TW>/causal_tune_trees/<model>/<bucket>/
#      and prints NOT MERGED for the rest (echoed here from reduce.log);
#   3. the merged arms are counted against the task files' (bucket, model, bar)
#      arms and the scorer runs on what merged.
# RESULTS_ROOT overrides the tuned root.
# 4 h / 32 G (vs the untuned scorer's 3 h / 16 G): each tuned arm also carries
# its candidate tables, so the reduce and the scorer read more per arm.
#SBATCH --job-name=tt_score
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x.%j.out
set -eo pipefail
module load conda >/dev/null 2>&1
source activate harxhar
set -u
export OMP_NUM_THREADS=1 TQDM_DISABLE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD:$PWD/experiments:$PWD/notebooks:$PWD/pylib_trees"
R="${RESULTS_ROOT:-results/linear_subsection_trees_tuned}"
GBT=cluster/treestuned_tasks_gb.txt
RFT=cluster/treestuned_tasks_rf.txt
mkdir -p "$R"

# 1. chunk-arms, walked from the task files (so a chunk that never started is
# listed too)
EXPECTED=0
FINISHED=0
while read -r BUCKET MODELS TW SEG CHUNK START END HALO; do
  [ -n "${BUCKET:-}" ] || continue
  for MODEL in ${MODELS//,/ }; do
    EXPECTED=$((EXPECTED + 1))
    D="$R/$BUCKET/$SEG/$MODEL/tw$TW/chunks/c$CHUNK"
    if [ -f "$D/DONE" ]; then
      FINISHED=$((FINISHED + 1))
    elif [ -f "$D/run.log" ]; then
      echo "NOT DONE: $D [$START, $END): $(tail -1 "$D/run.log" | cut -c1-200)"
    else
      echo "NOT DONE: $D [$START, $END): never started (no run.log)"
    fi
  done
done < <(cat "$GBT" "$RFT")
echo "$R: chunk-arms finished $FINISHED of $EXPECTED"

# 2. merge the arms whose chunks are all DONE
python -u experiments/reduce_trees_tuned_chunks.py --root "$R" --tasks "$GBT" --tasks "$RFT" \
  > "$R/reduce.log" 2>&1 \
  || echo "REDUCE FAILED for $R: $(tail -2 "$R/reduce.log")"
grep 'NOT MERGED' "$R/reduce.log" || true

# 3. merged arms vs the (bucket, model, bar) arms of the task files, then score
ARMS=$(cat "$GBT" "$RFT" \
  | awk 'NF {m = split($2, a, ","); for (i = 1; i <= m; i++) seen[$1 " " a[i] " " $4] = 1}
         END {n = 0; for (k in seen) n++; print n}')
shopt -s nullglob
MERGED=("$R"/*/*/*/tw*/causal_tune_trees/*/*/results_bar*.csv)
shopt -u nullglob
echo "$R: arms merged ${#MERGED[@]} of $ARMS"
python -u experiments/score_trees_tuned.py --root "$R" \
  --untuned-root results/linear_subsection_trees \
  --linear-root results/linear_subsection --linear-root results/linear_subsection_lassofix \
  > "$R/score.log" 2>&1 \
  || echo "SCORER FAILED for $R: $(tail -2 "$R/score.log")"
touch "$R/SCORED"
