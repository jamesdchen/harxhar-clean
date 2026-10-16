#!/bin/bash
# One task of the pooled pre-tune study on CARC, single-threaded (twin of cluster/pool_pretune_task.sh).
# Arg: one TASKFILE line "<tag> <mode> <bucket> <estimator> <r> <sched> <start> <end> <halo>"
# (experiments/pool_linear_pretune.py).  Output: results/pool_pretune/<tag>/<bucket>/<est>/r<r>/s<start>/
# with DONE and SECONDS on success.
set -eo pipefail
read -r TAG MODE BUCKET EST R SCHED START END HALO <<< "$1"
[ -n "${HALO:-}" ] || { echo "malformed line: '$1'"; exit 2; }
OUT="results/pool_pretune/$TAG/$BUCKET/$EST/r$R/s$START"
[ -f "$OUT/DONE" ] && { echo "$OUT already DONE"; exit 0; }
mkdir -p "$OUT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_${SLURM_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-0}_$$"
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD"
{
  echo "host $(hostname) job ${SLURM_JOB_ID:-?}.${SLURM_ARRAY_TASK_ID:-?}"
  scontrol show node "$(hostname -s)" 2>/dev/null | grep -o 'AvailableFeatures=[^ ]*'
  lscpu | grep -i "model name"
  python -c "import numpy; from threadpoolctl import threadpool_info as t; print('numpy', numpy.__version__, [(i.get('internal_api'), i.get('version'), i.get('architecture')) for i in t()])"
} > "$OUT/host.txt" 2>&1 || true
T0=$(date +%s)
if python -u experiments/pool_linear_pretune.py "$MODE" --bucket "$BUCKET" --est "$EST" --r "$R" \
     --sched "$SCHED" --start "$START" --end "$END" --halo "$HALO" --out "$OUT" > "$OUT/run.log" 2>&1; then
  echo "$(( $(date +%s) - T0 ))" > "$OUT/SECONDS"
  touch "$OUT/DONE"
  echo "done $OUT in $(cat "$OUT/SECONDS") s"
else
  echo "FAILED $OUT: $(tail -3 "$OUT/run.log" | cut -c1-300)"
  exit 1
fi
