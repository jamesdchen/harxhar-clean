#!/bin/bash
# CARC Discovery (Slurm) twin of cluster/subsection_pack.sh: a PACK of subsection
# arms per array task.  TASKFILE holds one
#   "<bucket> <estimator> <train_win_days> <seg1,seg2,...>"
# line per task; the arms of a line run one after another.  The pooled arm is the
# pack "none".  An arm that already wrote DONE is skipped, so a resubmitted task
# resumes.  Exported at submit time: TASKFILE, RESULTS_ROOT (default
# results/linear_subsection), CANARY_FLAG (a DONE file the fleet requires).
#SBATCH --job-name=subsec
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x.%A.%a.out
set -eo pipefail
module load conda >/dev/null 2>&1
source activate harxhar
set -u

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD"

# Slurm's afterok dependency already refuses to start the fleet if the canary
# failed; the flag check is the belt to that suspender.
if [ -n "${CANARY_FLAG:-}" ] && [ ! -f "$CANARY_FLAG" ]; then
  echo "canary did not finish ($CANARY_FLAG missing); refusing to run"
  exit 1
fi

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID:-1}p" "$TASKFILE")
read -r BUCKET EST TW SEGS <<< "$LINE"
echo "task ${SLURM_ARRAY_TASK_ID:-1} on $(hostname): $BUCKET $EST tw$TW [$SEGS]  $(date)"
FAIL=0
for SEG in ${SEGS//,/ }; do
  OUT="${RESULTS_ROOT:-results/linear_subsection}/$BUCKET/$SEG/$EST/tw$TW"
  [ -f "$OUT/DONE" ] && continue
  mkdir -p "$OUT"
  if HPC_KW_SEGMENT="$SEG" HPC_KW_LAG_SCOPE=global HPC_KW_ESTIMATOR="$EST" \
     HPC_KW_EXOG_BUCKET="$BUCKET" HPC_KW_TRAIN_WIN="$TW" HPC_RESULT_DIR="$OUT" \
     python -u specs/causal_tune_linear.py > "$OUT/run.log" 2>&1; then
    touch "$OUT/DONE"
    echo "  $SEG done $(date)"
  else
    FAIL=1
    echo "  $SEG FAILED: $(tail -1 "$OUT/run.log" | cut -c1-200)"
  fi
done
exit $FAIL
