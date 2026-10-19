#!/bin/bash
# One task of the pooled linear rerun on all 48 intervals (2026-10-19; results/pooled_linear_rerun_2026-10-19_48/),
# SGE array task, Hoffman2.  cluster/pool_pretune_task.sh with --rerun (experiments/pool_linear_pretune.py
# docstring: base-2 HAR ladder, DROP_ACTIVITY_MA + CALENDAR_THROUGH_OVERNIGHT, no FOMC, backbone = HAR benchmark).
#
# TASKFILE line (cluster/pooled_linear_rerun_48_tasks.txt):
#   "<tag> <mode> <bucket> <estimator> <r> <sched> <start> <end> <halo> <design>"
#   mode walk:   <sched> = a JSON file holding ONE number (the frozen penalty, frozen_penalties.json),
#                START 246,000 (the re-choice boundary before 2020-01-03), END -1, HALO 96,000
#   <design>:    the design cache (mode design --rerun; all_features: design --part K/4 x 4 + merge)
# Output: results/pooled_linear_rerun_2026-10-19_48/_work/walk/<bucket>/<estimator>/ (+ DONE, SECONDS, host.txt):
# the layout experiments/pooled_linear_rerun_2026_10_19.py score reads.
#
# Submit (from the repo root, after copying the _work/design_* caches there; sched/ is tracked):
#   qsub -t 1-7 -v TASKFILE=cluster/pooled_linear_rerun_48_tasks.txt cluster/pooled_linear_rerun_48_task.sh
#$ -cwd
#$ -j y
#$ -o logs/
#$ -l h_rt=24:00:00,h_data=16G
set -eo pipefail

source /u/local/apps/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate hpc-pi
set -u

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_${JOB_ID:-0}_${SGE_TASK_ID:-0}"
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD"

TASK="${SGE_TASK_ID:-1}"
[ "$TASK" = "undefined" ] && TASK=1
LINE=$(sed -n "${TASK}p" "$TASKFILE")
read -r TAG MODE BUCKET EST R SCHED START END HALO DESIGN <<< "$LINE"
if [ -z "${DESIGN:-}" ] || [ "$MODE" != walk ]; then
  echo "malformed line $TASK of $TASKFILE (want 10 fields, mode walk): '$LINE'"
  exit 1
fi
OUT="results/pooled_linear_rerun_2026-10-19_48/_work/walk/$BUCKET/$EST"
[ -f "$OUT/DONE" ] && { echo "$OUT already DONE"; exit 0; }
mkdir -p "$OUT"
echo "task $TASK on $(hostname): $LINE  $(date)"
{
  echo "host $(hostname) job ${JOB_ID:-?}.${SGE_TASK_ID:-?} tag $TAG"
  lscpu | grep -i "model name"
  python -c "import numpy; print('numpy', numpy.__version__)"
} > "$OUT/host.txt" 2>&1 || true

T0=$(date +%s)
if python -u experiments/pool_linear_pretune.py walk --rerun --bucket "$BUCKET" --est "$EST" --r "$R" \
     --sched "$SCHED" --start "$START" --end "$END" --halo "$HALO" --design "$DESIGN" --out "$OUT" > "$OUT/run.log" 2>&1; then
  echo "$(( $(date +%s) - T0 ))" > "$OUT/SECONDS"
  touch "$OUT/DONE"
  echo "done in $(cat "$OUT/SECONDS") s  $(date)"
else
  echo "FAILED: $(tail -3 "$OUT/run.log" | cut -c1-300)"
  exit 1
fi
