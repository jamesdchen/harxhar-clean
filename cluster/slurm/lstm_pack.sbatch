#!/bin/bash
# CARC Discovery (Slurm) twin of cluster/slurm/treestuned_pack.sbatch for the
# per-bar LSTM arms (specs/causal_tune_lstm.py), split into TIME CHUNKS: one
# array task runs one chunk of one arm.
# TASKFILE holds one
#   "<bucket> lstm <train_win_days> <seg> <chunk> <start> <end> <halo>"
# line per task (see submit_lstm.sh); the chunk runs into
#   $RESULTS_ROOT/<bucket>/<seg>/lstm/tw<TW>/chunks/c<chunk>
# (experiments/reduce_lstm_chunks.py merges an arm's chunks later).  A chunk
# that already wrote DONE is skipped, so a resubmitted task resumes.
# Exported at submit time: TASKFILE, RESULTS_ROOT (default
# results/linear_subsection_lstm), CANARY_FLAG (a file the fleet requires),
# WRITE_FLAG (touched when the line's chunk succeeded -- the canary writes
# CANARY_FLAG this way), REPEAT_CHECK (1 = run the chunk a second time into
# c<chunk>_repeat and require byte-identical results tables: the determinism
# check on the cluster's own hardware; the canary sets it), QSEL (default 1).
# Parallelism is across PROCESSES: the spec runs a pool of SLURM_CPUS_PER_TASK
# workers (specs/causal_tune_lstm_jobs.py), every network fit inside a worker is
# single-threaded (torch.set_num_threads(1) there), so the BLAS/OpenMP thread
# counts are pinned to 1 here too.  torch: the harxhar env's own CPU build; if
# that import fails, ./pylib_lstm (pip --target, installed by
# cluster/slurm/ship_lstm_carc.sh only when the env lacks torch) is on PYTHONPATH.
# The #SBATCH cpus/mem/time below are fallbacks for a hand-submitted task only:
# submit_lstm.sh passes --cpus-per-task/--mem/--time per job.
#SBATCH --job-name=lstm
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x.%A.%a.out
set -eo pipefail
module load conda >/dev/null 2>&1
source activate harxhar
set -u

NT="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$PWD:$PWD/pylib_lstm"

if [ -n "${CANARY_FLAG:-}" ] && [ ! -f "$CANARY_FLAG" ]; then
  echo "canary did not finish ($CANARY_FLAG missing); refusing to run"
  exit 1
fi
if ! python -c "import torch, numpy, pandas; print('torch', torch.__version__, torch.__file__)"; then
  echo "torch import failed (PYTHONPATH=$PYTHONPATH); refusing to run"
  exit 1
fi

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID:-1}p" "$TASKFILE")
read -r BUCKET MODEL TW SEG CHUNK START END HALO <<< "$LINE"
if [ -z "$HALO" ] || [ "$MODEL" != "lstm" ]; then
  echo "malformed line ${SLURM_ARRAY_TASK_ID:-1} of $TASKFILE (want 8 fields, model lstm): '$LINE'"
  exit 1
fi
R="${RESULTS_ROOT:-results/linear_subsection_lstm}"
echo "task ${SLURM_ARRAY_TASK_ID:-1} on $(hostname), pool of $NT: $BUCKET tw$TW $SEG chunk $CHUNK rows [$START, $END) halo $HALO qsel=${QSEL:-1} $(date)"

run_chunk() {  # $1 = output dir
  HPC_KW_EXOG_BUCKET="$BUCKET" HPC_KW_SEGMENT="$SEG" HPC_KW_TRAIN_WIN="$TW" \
  HPC_KW_LAG_SCOPE="${LAG_SCOPE:-global}" \
  HPC_KW_START="$START" HPC_KW_END="$END" HPC_KW_HALO="$HALO" HPC_KW_QSEL="${QSEL:-1}" \
  HPC_RESULT_DIR="$1" \
  python -u specs/causal_tune_lstm.py > "$1/run.log" 2>&1
}

OUT="$R/$BUCKET/$SEG/lstm/tw$TW/chunks/c$CHUNK"
FAIL=0
if [ -f "$OUT/DONE" ]; then
  echo "  $SEG c$CHUNK already DONE"
else
  mkdir -p "$OUT"
  if run_chunk "$OUT"; then
    touch "$OUT/DONE"
    echo "  $SEG c$CHUNK done $(date): $(tail -1 "$OUT/run.log" | cut -c1-300)"
  else
    FAIL=1
    echo "  $SEG c$CHUNK FAILED: $(tail -3 "$OUT/run.log" | cut -c1-300)"
  fi
fi
if [ "$FAIL" -eq 0 ] && [ "${REPEAT_CHECK:-0}" = "1" ]; then
  REP="$R/$BUCKET/$SEG/lstm/tw$TW/chunks/c${CHUNK}_repeat"
  mkdir -p "$REP"
  if run_chunk "$REP"; then
    IN="causal_tune_lstm/lstm/$BUCKET"
    for f in "results_$SEG.csv" "results_qsel_$SEG.csv"; do  # (the tune tables carry timings)
      if cmp -s "$OUT/$IN/$f" "$REP/$IN/$f"; then
        echo "  repeat check: $f byte-identical"
      else
        echo "  repeat check FAILED: $f differs between two runs of the same chunk"
        FAIL=1
      fi
    done
  else
    echo "  repeat run FAILED: $(tail -3 "$REP/run.log" | cut -c1-300)"
    FAIL=1
  fi
fi
if [ "$FAIL" -eq 0 ] && [ -n "${WRITE_FLAG:-}" ]; then
  mkdir -p "$(dirname "$WRITE_FLAG")"
  touch "$WRITE_FLAG"
  echo "wrote $WRITE_FLAG"
fi
exit $FAIL
