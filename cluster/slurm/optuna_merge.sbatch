#!/bin/bash
# Merge one stage of the Optuna per-bar tree campaign (specs/causal_tune_trees_optuna.py).
# Held on the stage's fleet with afterany, so a failed chunk does not block the rest.
#   1. every chunk-arm of the stage's task file is walked: the ones without DONE are
#      listed (NOT DONE + the last line of their run.log, or "never started");
#   2. experiments/reduce_trees_optuna.py --stage $MERGE_STAGE merges every arm whose
#      chunks are all DONE (stage 1: + the completeness gate that writes the arm's
#      STAGE1_COMPLETE flag, which each stage-2 task requires; stage 2: the untuned
#      layout per path under $R/paths/);
#   3. writes $R/STAGE<k>_MERGED (the next stage's hold flag) even when some arm is
#      incomplete: stage-2 tasks check their own arm's STAGE1_COMPLETE.
# Exported: MERGE_STAGE (1 | 2), TASKFILE (the stage's task file), RESULTS_ROOT.
#SBATCH --job-name=optuna_merge
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x.%j.out
set -eo pipefail
module load conda >/dev/null 2>&1
source activate harxhar
set -u
export OMP_NUM_THREADS=1 TQDM_DISABLE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$PWD"
R="${RESULTS_ROOT:-results/linear_subsection_trees_optuna_mask}"
K="${MERGE_STAGE:?MERGE_STAGE must be 1 or 2}"
mkdir -p "$R"

EXPECTED=0
FINISHED=0
while read -r BUCKET MODELS TW SEG CHUNK START END HALO; do
  [ -n "${BUCKET:-}" ] || continue
  for MODEL in ${MODELS//,/ }; do
    EXPECTED=$((EXPECTED + 1))
    D="$R/stage$K/$BUCKET/$SEG/$MODEL/tw$TW/chunks/c$CHUNK"
    if [ -f "$D/DONE" ]; then
      FINISHED=$((FINISHED + 1))
    elif [ -f "$D/run.log" ]; then
      echo "NOT DONE: $D [$START, $END): $(tail -1 "$D/run.log" | cut -c1-200)"
    else
      echo "NOT DONE: $D [$START, $END): never started (no run.log)"
    fi
  done
done < "$TASKFILE"
echo "$R stage $K: chunk-arms finished $FINISHED of $EXPECTED"

python -u experiments/reduce_trees_optuna.py --stage "$K" --root "$R" --tasks "$TASKFILE" \
  > "$R/reduce_stage$K.log" 2>&1 \
  || echo "REDUCE stage $K reported a failure: $(tail -2 "$R/reduce_stage$K.log")"
grep -E 'NOT MERGED|GATE FAIL' "$R/reduce_stage$K.log" || true
tail -1 "$R/reduce_stage$K.log"
touch "$R/STAGE${K}_MERGED"
