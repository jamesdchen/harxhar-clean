#!/bin/bash
# I1 (2026-09-29): twin of cluster/slurm/lstm_pack.sbatch for the per-bar LSTM arms on the
# DE-DUPLICATED per-bar design (commit 47f7f9c: the segment path no longer builds the twelve
# har_ma_*_x_open / _x_close columns; per-bar p 34/244/640 -> 22/232/628).  Unchanged code path:
# one array task runs one TIME CHUNK of one arm of specs/causal_tune_lstm.py (read-only), a pool of
# SLURM_CPUS_PER_TASK spawned workers, every fit single-threaded (numbers independent of the
# pool size).  TASKFILE lines: "<bucket> lstm <train_win_days> <seg> <chunk> <start> <end> <halo>";
# the chunk runs into $RESULTS_ROOT/<bucket>/<seg>/lstm/tw<TW>/chunks/c<chunk> (default root
# results/linear_subsection_lstm_dedup).  A chunk that already wrote DONE is skipped.  Exported
# at submit time: TASKFILE, RESULTS_ROOT, CANARY_FLAG (a file the fleet requires), WRITE_FLAG
# (touched when the line's chunk succeeded), REPEAT_CHECK (1 = run the chunk twice and require
# byte-identical results tables: determinism on this hardware), QSEL (default 1).
#SBATCH --job-name=lstm_dedup
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
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
R="${RESULTS_ROOT:-results/linear_subsection_lstm_dedup}"
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
