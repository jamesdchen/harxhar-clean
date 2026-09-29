#!/bin/bash
# CARC Discovery (Slurm) twin of cluster/slurm/trees_pack.sbatch and
# cluster/slurm/subsection_pack.sbatch for the DIRECT rest-of-day variance arms
# (specs/causal_tune_rest_of_day.py): a PACK of arms per array task.
# TASKFILE holds one
#   "<bucket> <estimator[,estimator...]> <train_win_days> <clock[,clock...]>"
# line per task (clock = the entry clock HHMM, 0930 .. 1530); the arms of a
# line (estimators x clocks) run one after another, one python process each,
# single-threaded.  An arm that already wrote DONE is skipped, so a
# resubmitted task resumes.  Arm dir:
#   $RESULTS_ROOT/<bucket>/rod<clock>/<estimator>/tw<train_win>
# (HPC_RESULT_DIR; the spec writes causal_tune_rest_of_day/<est>/<bucket>/
# results_bar<HHMM>.csv + targets_bar<HHMM>.csv under it; run.log and DONE sit
# in the arm dir).
# Exported at submit time: TASKFILE, RESULTS_ROOT (default
# results/linear_subsection_restofday), CANARY_FLAG (a file the fleet
# requires), WRITE_FLAG (the canary only: after every arm of its line
# succeeded it runs the 15:30 == bar1600 identity check and touches this file
# only if that check passes too).
# Before any arm: cluster/restofday_manifest.md5 (written by
# cluster/slurm/ship_restofday_carc.sh) pins the exact code + data the local
# gates ran with; a task refuses to run if any pinned file differs here.
#SBATCH --job-name=restofday
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
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$PWD:$PWD/experiments"
ROOT="${RESULTS_ROOT:-results/linear_subsection_restofday}"
MANIFEST=cluster/restofday_manifest.md5

# Slurm's afterok dependency already refuses to start the fleet if the canary
# failed; the flag check is the belt to that suspender.
if [ -n "${CANARY_FLAG:-}" ] && [ ! -f "$CANARY_FLAG" ]; then
  echo "canary did not finish ($CANARY_FLAG missing); refusing to run"
  exit 1
fi

# Fail fast (before any arm) if any pinned code or data file differs from the
# bytes the local gates ran with.
if [ ! -f "$MANIFEST" ]; then
  echo "$MANIFEST missing in $PWD: run the ship script first; refusing to run"
  exit 1
fi
if ! md5sum -c --quiet "$MANIFEST"; then
  echo "manifest check failed in $PWD (the files above differ from $MANIFEST); refusing to run"
  exit 1
fi
echo "manifest ok: $(wc -l < "$MANIFEST") pinned files"

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID:-1}p" "$TASKFILE")
read -r BUCKET ESTS TW CLOCKS <<< "$LINE"
echo "task ${SLURM_ARRAY_TASK_ID:-1} on $(hostname): $BUCKET [$ESTS] tw$TW [$CLOCKS]  $(date)"
FAIL=0
for EST in ${ESTS//,/ }; do
  for CLOCK in ${CLOCKS//,/ }; do
    OUT="$ROOT/$BUCKET/rod$CLOCK/$EST/tw$TW"
    [ -f "$OUT/DONE" ] && continue
    mkdir -p "$OUT"
    if HPC_KW_EXOG_BUCKET="$BUCKET" HPC_KW_ESTIMATOR="$EST" HPC_KW_CLOCK="$CLOCK" \
       HPC_KW_TRAIN_WIN="$TW" HPC_KW_START=0 HPC_KW_END=-1 HPC_KW_HALO=0 \
       HPC_RESULT_DIR="$OUT" \
       python -u specs/causal_tune_rest_of_day.py > "$OUT/run.log" 2>&1; then
      touch "$OUT/DONE"
      echo "  $EST rod$CLOCK done $(date)"
    else
      FAIL=1
      echo "  $EST rod$CLOCK FAILED: $(tail -1 "$OUT/run.log" | cut -c1-200)"
    fi
  done
done

if [ "$FAIL" -eq 0 ] && [ -n "${WRITE_FLAG:-}" ]; then
  # The canary's gate: the 15:30 direct arm must reproduce the per-bar bar1600
  # forecast (exit 0 = identity holds).  The flag is written only if it does.
  echo "identity check $(date)"
  if python -u experiments/check_restofday_identity.py --root "$ROOT" \
       --ref results/linear_subsection_restofday/ref/yhat_sub_ridge_live_feasible.parquet \
       > "$ROOT/identity_check.log" 2>&1; then
    mkdir -p "$(dirname "$WRITE_FLAG")"
    touch "$WRITE_FLAG"
    echo "identity check passed; wrote $WRITE_FLAG"
  else
    FAIL=1
    echo "identity check FAILED ($ROOT/identity_check.log): $(tail -2 "$ROOT/identity_check.log" | cut -c1-300)"
    echo "$WRITE_FLAG not written; the fleet stays held"
  fi
fi
exit $FAIL
