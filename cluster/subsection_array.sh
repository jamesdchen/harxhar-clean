#!/bin/bash
# One subsection arm of specs/causal_tune_linear.py per SGE array task (Hoffman2).
#
#   qsub -N subsec -t 1-100 -v TASKFILE=cluster/subsection_tasks_segments.txt \
#        cluster/subsection_array.sh
#
# TASKFILE holds one "<segment> <estimator> <train_win_days>" line per task;
# OFFSET shifts the line read, because Hoffman2 caps an array at 100 tasks.
#$ -cwd
#$ -j y
#$ -o logs/
#$ -l h_rt=2:00:00,h_data=8G
set -eo pipefail  # -u only after conda: its activate scripts read unset variables

source /u/local/apps/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate hpc-pi
set -u

# one core a task: no BLAS/numba fan-out under SGE's memory accounting, and a
# task-private numba cache so concurrent tasks cannot tear the shared one
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_${JOB_ID}_${SGE_TASK_ID}"
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD"

# The fleet is held on the canary job; SGE releases a hold whatever the exit
# status, so a fleet task refuses to run unless the canary actually finished.
if [ -n "${CANARY_FLAG:-}" ] && [ ! -f "$CANARY_FLAG" ]; then
  echo "canary did not finish ($CANARY_FLAG missing); refusing to run"
  exit 1
fi

LINE=$(sed -n "$(( SGE_TASK_ID + ${OFFSET:-0} ))p" "$TASKFILE")
read -r SEG EST TW <<< "$LINE"
OUT="results/linear_subsection/baseline/$SEG/$EST/tw$TW"
mkdir -p "$OUT"
echo "task $SGE_TASK_ID on $(hostname): $SEG $EST tw$TW  $(date)"

HPC_KW_SEGMENT="$SEG" HPC_KW_LAG_SCOPE=global HPC_KW_ESTIMATOR="$EST" \
HPC_KW_EXOG_BUCKET=baseline HPC_KW_TRAIN_WIN="$TW" HPC_RESULT_DIR="$OUT" \
  python -u specs/causal_tune_linear.py > "$OUT/run.log" 2>&1

echo "done $SEG $EST tw$TW  $(date)"
touch "$OUT/DONE"
