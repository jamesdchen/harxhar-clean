#!/bin/bash
# Wave 2: a PACK of subsection arms per SGE array task (Hoffman2).
#
# TASKFILE holds one "<bucket> <estimator> <train_win_days> <seg1,seg2,...>"
# line per task; the arms of a line run one after another (a one-bar arm is
# about a minute, so a pack of 17 beats 17 scheduler round-trips).  The pooled
# arm is the pack "none".  An arm that already wrote DONE is skipped, so a
# resubmitted task resumes.
#$ -cwd
#$ -j y
#$ -o logs/
#$ -l h_rt=6:00:00,h_data=16G
set -eo pipefail  # -u only after conda: its activate scripts read unset variables

source /u/local/apps/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate hpc-pi
set -u

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_${JOB_ID}_${SGE_TASK_ID}"
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD"

# The fleet is held on the canary job; SGE releases a hold whatever the exit
# status, so a fleet task refuses to run unless the canary actually finished.
if [ -n "${CANARY_FLAG:-}" ] && [ ! -f "$CANARY_FLAG" ]; then
  echo "canary did not finish ($CANARY_FLAG missing); refusing to run"
  exit 1
fi

LINE=$(sed -n "${SGE_TASK_ID}p" "$TASKFILE")
read -r BUCKET EST TW SEGS <<< "$LINE"
echo "task $SGE_TASK_ID on $(hostname): $BUCKET $EST tw$TW [$SEGS]  $(date)"
FAIL=0
for SEG in ${SEGS//,/ }; do
  OUT="results/linear_subsection/$BUCKET/$SEG/$EST/tw$TW"
  [ -f "$OUT/DONE" ] && continue
  mkdir -p "$OUT"
  if HPC_KW_SEGMENT="$SEG" HPC_KW_LAG_SCOPE=global HPC_KW_ESTIMATOR="$EST" \
     HPC_KW_EXOG_BUCKET="$BUCKET" HPC_KW_TRAIN_WIN="$TW" HPC_RESULT_DIR="$OUT" \
     python -u specs/causal_tune_linear.py > "$OUT/run.log" 2>&1; then
    touch "$OUT/DONE"
    echo "done $BUCKET $SEG $EST tw$TW  $(date)"
  else
    FAIL=1
    echo "FAILED $BUCKET $SEG $EST tw$TW: $(tail -1 "$OUT/run.log" | cut -c1-200)"
  fi
done
exit $FAIL
