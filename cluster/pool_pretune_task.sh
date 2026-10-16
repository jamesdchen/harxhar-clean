#!/bin/bash
# One task of the pooled pre-tune study (experiments/pool_linear_pretune.py) per SGE array task, Hoffman2,
# 2026-10-06.  The runner executes the spec's own estimator section (specs/causal_tune_linear.py, segment
# "none": one coefficient vector over all 48 half-hour bars, 96,000-bar window, re-solved every bar,
# penalty re-chosen every 250 solves) through the spec's run_executor invocation, one thread, with the
# backbone penalty factor r and either the spec's causal re-choice or a fixed {block: alpha} schedule.
#
# TASKFILE line: "<tag> <mode> <bucket> <estimator> <r> <sched> <start> <end> <halo> [<design>]"
#   mode walk:    START / END / HALO = run_executor's slice (START a re-choice boundary, HALO 96,000)
#   mode pretune: START = the first row of the first trade day, HALO = 145,200 (4 folds + embargo + window)
#   mode design:  the bucket's design cache into <design> (the whole-series data prep, once per CPU type)
#   <design>:     walk / pretune read that cache instead of redoing the data prep (bit-identical arrays)
# Output: results/pool_pretune/<tag>/<bucket>/<estimator>/r<r>/s<start>/ (+ DONE, SECONDS, host.txt);
# mode design: <design>/ (+ DONE).
# REQUIRE_FLAG (optional): a file that must exist (the gate's GATE_OK; SGE releases a hold whatever the
# held job's exit status).  ARCH_REQ: the -l arch the submit script pinned (recorded only).
#$ -cwd
#$ -j y
#$ -o logs/
#$ -l h_rt=8:00:00,h_data=16G
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
read -r TAG MODE BUCKET EST R SCHED START END HALO DESIGN <<< "$LINE"
if [ -z "${HALO:-}" ]; then
  echo "malformed line $TASK of $TASKFILE (want 9 or 10 fields): '$LINE'"
  exit 1
fi
OUT="results/pool_pretune/$TAG/$BUCKET/$EST/r$R/s$START"
DARG=()
if [ "$MODE" = design ]; then
  OUT="$DESIGN"
elif [ -n "${DESIGN:-}" ]; then
  [ -f "$DESIGN/DONE" ] || { echo "design cache $DESIGN not DONE; refusing to run"; exit 1; }
  DARG=(--design "$DESIGN")
fi
[ -f "$OUT/DONE" ] && { echo "$OUT already DONE"; exit 0; }
mkdir -p "$OUT"
echo "task $TASK on $(hostname) (arch ${ARCH_REQ:-?}): $LINE  $(date)"
{
  echo "host $(hostname) arch_requested ${ARCH_REQ:-?} job ${JOB_ID:-?}.${SGE_TASK_ID:-?}"
  lscpu | grep -i "model name"
  python -c "import numpy; from threadpoolctl import threadpool_info as t; print('numpy', numpy.__version__, [(i.get('internal_api'), i.get('version'), i.get('architecture')) for i in t()])"
} > "$OUT/host.txt" 2>&1 || true

T0=$(date +%s)
if python -u experiments/pool_linear_pretune.py "$MODE" --bucket "$BUCKET" --est "$EST" --r "$R" \
     --sched "$SCHED" --start "$START" --end "$END" --halo "$HALO" --out "$OUT" ${DARG[@]+"${DARG[@]}"} > "$OUT/run.log" 2>&1; then
  echo "$(( $(date +%s) - T0 ))" > "$OUT/SECONDS"
  touch "$OUT/DONE"
  echo "done in $(cat "$OUT/SECONDS") s: $(grep -h 'done:' "$OUT/run.log" | cut -c1-300)  $(date)"
else
  echo "FAILED: $(tail -3 "$OUT/run.log" | cut -c1-300)"
  exit 1
fi
