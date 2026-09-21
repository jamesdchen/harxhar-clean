#!/bin/bash
# Block ridge with per-clock HAR deltas: one "<arm> <chunk list>" line per SGE
# array task (Hoffman2).  The batch runner loads the panel once and skips chunks
# whose npz already exists, so a resubmitted task resumes.
#$ -cwd
#$ -j y
#$ -o logs/
#$ -l h_rt=12:00:00,h_data=24G
set -eo pipefail  # -u only after conda: its activate scripts read unset variables

source /u/local/apps/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate hpc-pi
set -u

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_${JOB_ID}_${SGE_TASK_ID}"
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD:$PWD/experiments"
# the prepared incumbent panel (prep_cache_all_features_b2.npz) is staged here;
# without it the loader would try to rebuild the panel in-task
export UNIFY_CACHE_DIR="$PWD/results"
[ -f "$UNIFY_CACHE_DIR/prep_cache_all_features_b2.npz" ] || { echo "panel cache missing"; exit 1; }

if [ -n "${CANARY_FLAG:-}" ] && [ ! -f "$CANARY_FLAG" ]; then
  echo "canary did not finish ($CANARY_FLAG missing); refusing to run"
  exit 1
fi

LINE=$(sed -n "${SGE_TASK_ID}p" "$TASKFILE")
read -r ARM CHUNKS <<< "$LINE"
echo "task $SGE_TASK_ID on $(hostname): $ARM chunks $CHUNKS  $(date)"
python -u experiments/run_unification_batch.py --arm "$ARM" --chunks "$CHUNKS" \
  --output-dir results/unification_clock
echo "done $ARM $CHUNKS  $(date)"
