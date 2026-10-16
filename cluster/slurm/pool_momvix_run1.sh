#!/bin/bash
# One run of specs/causal_tune_linear.py's POOLED arm (segment "none") on CARC, single-threaded.
# Args: <outdir> <bucket> <estimator> <train_win_days> <start> <end> <halo>
# start / end / halo are run_executor's slice fields (0 -1 0 = the whole series); every other
# setting is the spec's default, as the stored pooled arms ran (cluster/slurm/subsection_pack.sbatch
# with the line "<bucket> <est> 2000 none").  Writes <outdir>/DONE and SECONDS on success.
set -eo pipefail
OUT=$1 BUCKET=$2 EST=$3 TW=$4 START=$5 END=$6 HALO=$7
[ -n "$HALO" ] || { echo "usage: pool_momvix_run1.sh outdir bucket est tw start end halo"; exit 2; }
[ -f "$OUT/DONE" ] && exit 0
mkdir -p "$OUT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_${SLURM_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-0}_$$"
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD"
TIMECMD=()
[ -x /usr/bin/time ] && TIMECMD=(/usr/bin/time -v)  # max RSS into run.log (memory sizing only)
T0=$(date +%s)
if HPC_KW_SEGMENT=none HPC_KW_LAG_SCOPE=global HPC_KW_ESTIMATOR="$EST" \
   HPC_KW_EXOG_BUCKET="$BUCKET" HPC_KW_TRAIN_WIN="$TW" \
   HPC_KW_START="$START" HPC_KW_END="$END" HPC_KW_HALO="$HALO" HPC_RESULT_DIR="$OUT" \
   "${TIMECMD[@]}" python -u specs/causal_tune_linear.py > "$OUT/run.log" 2>&1; then
  echo "$(( $(date +%s) - T0 ))" > "$OUT/SECONDS"
  touch "$OUT/DONE"
  echo "done $OUT in $(cat "$OUT/SECONDS") s, max RSS $(grep -h 'Maximum resident' "$OUT/run.log" | awk '{print $NF}') kB"
else
  echo "FAILED $OUT: $(grep -v '^\s' "$OUT/run.log" | tail -2 | cut -c1-240)"
  exit 1
fi
