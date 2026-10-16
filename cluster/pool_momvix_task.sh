#!/bin/bash
# One POOLED arm of specs/causal_tune_linear.py per SGE array task (Hoffman2), 2026-10-06.
#
# The pooled twin: segment "none" = one coefficient vector over all 48 half-hour bars (the HAR +
# calendar backbone keeps its session-edge interactions on this path), global lags, a 2000-session
# window (96,000 bars), coefficients re-solved every bar, the penalty re-chosen every 250 solves on
# the window's validation tail -- the spec's defaults, run exactly as the stored pooled arms were
# (cluster/subsection_pack.sh with the line "<bucket> <est> 2000 none": one thread, the spec's
# environment axes only).  New here: the slice fields, so the canary can re-run one slice of a
# stored arm (the fleet passes 0 -1 0 = the whole series, the run_executor defaults).
#
# TASKFILE line: "<tag> <bucket> <estimator> <train_win_days> <start> <end> <halo>"
# Output: results/pool_momvix/<tag>/<bucket>/none/<estimator>/tw<tw>/ (+ DONE, SECONDS, host.txt).
# REQUIRE_FLAG (optional): a file that must exist (the gate's GATE_OK; SGE releases a hold whatever
# the held job's exit status).  ARCH_REQ: the -l arch the submit script pinned (recorded only).
#$ -cwd
#$ -j y
#$ -o logs/
#$ -l h_rt=6:00:00,h_data=16G
set -eo pipefail  # -u only after conda: its activate scripts read unset variables

source /u/local/apps/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate hpc-pi
set -u

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_${JOB_ID:-0}_${SGE_TASK_ID:-0}"
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD"

if [ -n "${REQUIRE_FLAG:-}" ] && [ ! -f "$REQUIRE_FLAG" ]; then
  echo "required flag $REQUIRE_FLAG missing; refusing to run"
  exit 1
fi

TASK="${SGE_TASK_ID:-1}"
[ "$TASK" = "undefined" ] && TASK=1
LINE=$(sed -n "${TASK}p" "$TASKFILE")
read -r TAG BUCKET EST TW START END HALO <<< "$LINE"
if [ -z "${HALO:-}" ]; then
  echo "malformed line $TASK of $TASKFILE (want 7 fields): '$LINE'"
  exit 1
fi
OUT="results/pool_momvix/$TAG/$BUCKET/none/$EST/tw$TW"
[ -f "$OUT/DONE" ] && { echo "$OUT already DONE"; exit 0; }
mkdir -p "$OUT"
echo "task $TASK on $(hostname) (arch ${ARCH_REQ:-?}): $TAG $BUCKET $EST tw$TW start=$START end=$END halo=$HALO  $(date)"
{
  echo "host $(hostname) arch_requested ${ARCH_REQ:-?} job ${JOB_ID:-?}.${SGE_TASK_ID:-?}"
  lscpu | grep -i "model name"
  python -c "import numpy; from threadpoolctl import threadpool_info as t; print('numpy', numpy.__version__, [(i.get('internal_api'), i.get('version'), i.get('architecture')) for i in t()])"
} > "$OUT/host.txt" 2>&1 || true

T0=$(date +%s)
if HPC_KW_SEGMENT=none HPC_KW_LAG_SCOPE=global HPC_KW_ESTIMATOR="$EST" \
   HPC_KW_EXOG_BUCKET="$BUCKET" HPC_KW_TRAIN_WIN="$TW" \
   HPC_KW_START="$START" HPC_KW_END="$END" HPC_KW_HALO="$HALO" HPC_RESULT_DIR="$OUT" \
   python -u specs/causal_tune_linear.py > "$OUT/run.log" 2>&1; then
  echo "$(( $(date +%s) - T0 ))" > "$OUT/SECONDS"
  touch "$OUT/DONE"
  echo "done in $(cat "$OUT/SECONDS") s: $(grep -h 'OOS rows' "$OUT/run.log" | cut -c1-200)  $(date)"
else
  echo "FAILED: $(tail -2 "$OUT/run.log" | cut -c1-240)"
  exit 1
fi
