#!/bin/bash
# I6 (2026-09-29): the DIRECT rest-of-day arms (specs/causal_tune_rest_of_day.py) on the
# de-duplicated per-bar design (commit 47f7f9c), on Hoffman2 (SGE).  SGE twin of
# cluster/slurm/restofday_pack.sbatch: a PACK of arms per array task.  TASKFILE holds one
#   "<bucket> <estimator[,estimator...]> <train_win_days> <clock[,clock...]>"
# line per task (clock = the entry clock HHMM, 0930 .. 1530); the arms of a line run one after
# another, one single-threaded python process each.  An arm that already wrote DONE is skipped, so
# a resubmitted task resumes.  Arm dir (the old campaign's layout, new root):
#   results/linear_subsection_restofday_dedup/<bucket>/rod<clock>/<estimator>/tw<train_win>
# holding run.log, HOST (host + CPU model), SECONDS, DONE and the spec's
# causal_tune_rest_of_day/<est>/<bucket>/{results,targets}_bar<HHMM>.csv.
# Exported at submit time: TASKFILE, CANARY_FLAG (a file the fleet requires; empty for the canary).
# Before any arm: cluster/restofday_dedup_manifest.md5 (written by
# cluster/ship_restofday_dedup_hoffman2.sh) pins the shipped code, references and data; a task
# refuses to run if any pinned file differs here.
#$ -cwd
#$ -j y
#$ -o logs/
#$ -l h_rt=1:00:00,h_data=16G
set -eo pipefail  # -u only after conda: its activate scripts read unset variables

source /u/local/apps/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate hpc-pi
set -u

# one core a task (cluster/linear_dedup_pack.sh's settings): no BLAS/numba fan-out, a
# task-private numba cache, the executor's masked-scaling pool held to the slot grant
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export UNIF_SCALE_PROCS="${NSLOTS:-1}" TQDM_DISABLE=1
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_${JOB_ID:-0}_${SGE_TASK_ID:-0}"
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD"
R=results/linear_subsection_restofday_dedup
MANIFEST=cluster/restofday_dedup_manifest.md5

# SGE releases a hold whatever the exit status: a fleet task refuses to run without the flag
if [ -n "${CANARY_FLAG:-}" ] && [ ! -f "$CANARY_FLAG" ]; then
  echo "canary did not pass ($CANARY_FLAG missing); refusing to run"
  exit 1
fi
if [ ! -f "$MANIFEST" ] || ! md5sum -c --quiet "$MANIFEST"; then
  echo "manifest check failed in $PWD ($MANIFEST); refusing to run"
  exit 1
fi

TASK="${SGE_TASK_ID:-1}"
[ "$TASK" = "undefined" ] && TASK=1
LINE=$(sed -n "${TASK}p" "$TASKFILE")
read -r BUCKET ESTS TW CLOCKS <<< "$LINE"
if [ -z "${CLOCKS:-}" ]; then
  echo "malformed line $TASK of $TASKFILE (want 4 fields): '$LINE'"
  exit 1
fi
CPU=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')
echo "task $TASK on $(hostname -s) [$CPU]: $BUCKET [$ESTS] tw$TW [$CLOCKS]  $(date)"
FAIL=0
for EST in ${ESTS//,/ }; do
  for CLOCK in ${CLOCKS//,/ }; do
    OUT="$R/$BUCKET/rod$CLOCK/$EST/tw$TW"
    [ -f "$OUT/DONE" ] && { echo "  $EST rod$CLOCK already DONE"; continue; }
    mkdir -p "$OUT"
    echo "$(hostname -s) $CPU" > "$OUT/HOST"
    T0=$(date +%s)
    if HPC_KW_EXOG_BUCKET="$BUCKET" HPC_KW_ESTIMATOR="$EST" HPC_KW_CLOCK="$CLOCK" \
       HPC_KW_TRAIN_WIN="$TW" HPC_KW_START=0 HPC_KW_END=-1 HPC_KW_HALO=0 \
       HPC_RESULT_DIR="$OUT" python -u specs/causal_tune_rest_of_day.py > "$OUT/run.log" 2>&1; then
      echo "$(( $(date +%s) - T0 ))" > "$OUT/SECONDS"
      touch "$OUT/DONE"
      echo "  $EST rod$CLOCK done in $(cat "$OUT/SECONDS") s: $(grep -h 'masked cols' "$OUT/run.log" | cut -c1-200)"
    else
      FAIL=1
      echo "  $EST rod$CLOCK FAILED: $(tail -2 "$OUT/run.log" | cut -c1-240)"
    fi
  done
done
exit $FAIL
