#!/bin/bash
# I1 (2026-09-29): a PACK of one-bar arms of specs/causal_tune_linear.py per SGE array task
# (Hoffman2), on the de-duplicated per-bar design (commit 47f7f9c).  Twin of
# cluster/subsection_pack.sh with every spec axis on the task line, so ONE array covers every
# per-bar linear forecast the closing-strategy master table scores
# (experiments/linear_dedup_plan.py writes the task files):
#   "<results_root> <bucket> <estimator> <train_win_days> <har_base|-> <lag_scope> <seg1,seg2,...>"
# An arm runs into <results_root>/<bucket>/<seg>/<estimator>/tw<tw> (the old layouts, under
# results/linear_subsection_dedup); an arm that already wrote DONE is skipped, so a resubmitted
# task resumes.  Exported at submit time: TASKFILE, CANARY_FLAG (a file the fleet requires).
#$ -cwd
#$ -j y
#$ -o logs/
#$ -l h_rt=2:00:00,h_data=16G
set -eo pipefail  # -u only after conda: its activate scripts read unset variables

source /u/local/apps/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate hpc-pi
set -u

# one core a task: no BLAS/numba fan-out under SGE's memory accounting, a task-private numba
# cache, and the masked-scaling pool held to the slot grant (the executor's UNIF_SCALE_PROCS;
# its workers are exact, so the count changes scheduling only)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export UNIF_SCALE_PROCS="${NSLOTS:-1}" TQDM_DISABLE=1
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_${JOB_ID:-0}_${SGE_TASK_ID:-0}"
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD"

# SGE releases a hold whatever the exit status: a fleet task refuses to run without the flag
if [ -n "${CANARY_FLAG:-}" ] && [ ! -f "$CANARY_FLAG" ]; then
  echo "canary did not pass ($CANARY_FLAG missing); refusing to run"
  exit 1
fi

TASK="${SGE_TASK_ID:-1}"
[ "$TASK" = "undefined" ] && TASK=1
LINE=$(sed -n "${TASK}p" "$TASKFILE")
read -r R BUCKET EST TW HAR LAG SEGS <<< "$LINE"
if [ -z "${SEGS:-}" ]; then
  echo "malformed line $TASK of $TASKFILE (want 7 fields): '$LINE'"
  exit 1
fi
[ "$HAR" = "-" ] && HAR=""
echo "task $TASK on $(hostname): $R | $BUCKET $EST tw$TW har_base='${HAR}' lag=$LAG [$SEGS]  $(date)"
FAIL=0
for SEG in ${SEGS//,/ }; do
  OUT="$R/$BUCKET/$SEG/$EST/tw$TW"
  [ -f "$OUT/DONE" ] && { echo "  $SEG already DONE"; continue; }
  mkdir -p "$OUT"
  T0=$(date +%s)
  if HPC_KW_SEGMENT="$SEG" HPC_KW_LAG_SCOPE="$LAG" HPC_KW_ESTIMATOR="$EST" \
     HPC_KW_HAR_BASE="$HAR" HPC_KW_EXOG_BUCKET="$BUCKET" HPC_KW_TRAIN_WIN="$TW" \
     HPC_RESULT_DIR="$OUT" python -u specs/causal_tune_linear.py > "$OUT/run.log" 2>&1; then
    echo "$(( $(date +%s) - T0 ))" > "$OUT/SECONDS"
    touch "$OUT/DONE"
    echo "  $SEG done in $(cat "$OUT/SECONDS") s: $(grep -h 'masked cols' "$OUT/run.log" | cut -c1-160)"
  else
    FAIL=1
    echo "  $SEG FAILED: $(tail -2 "$OUT/run.log" | cut -c1-240)"
  fi
done
exit $FAIL
