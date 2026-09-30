#!/bin/bash
# I10 (2026-09-29): one array task of the intraday-sequence LSTM (specs/causal_tune_lstm_intraday.py,
# read-only), a pool of SLURM_CPUS_PER_TASK spawned workers, every fit single-threaded (numbers
# independent of the pool size).  TASKFILE lines:
#   "<bucket> lstm_intraday <train_win_days> <seg> <chunk> <start> <end> <halo> <stage>"
#   stage tune: the tuning point(s) inside the line's rows -> $RESULTS_ROOT/tune_cache/<bucket>/
#               (STAGE=tune; output dir .../stage_tune/r<chunk>)
#   stage all : a refit chunk -> $RESULTS_ROOT/<bucket>/<seg>/lstm_intraday/tw<TW>/chunks/c<chunk>,
#               reading its tuning point(s) from the cache (REQUIRE_CACHE, default 1)
# A line that already wrote DONE is skipped.  Exported at submit time: TASKFILE, RESULTS_ROOT,
# REQUIRE_FLAGS (space-separated files that must exist, e.g. the gates' and canary's flags),
# WRITE_FLAG (touched when the line succeeded), REQUIRE_CACHE (0 lets a chunk tune for itself).
#SBATCH --job-name=lstmi
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x.%A.%a.out
set -eo pipefail
module load conda >/dev/null 2>&1
source activate harxhar
set -u

NT="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$PWD"

for f in ${REQUIRE_FLAGS:-}; do
  if [ ! -f "$f" ]; then echo "required flag $f missing (gates / canary not passed); refusing to run"; exit 1; fi
done

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID:-1}p" "$TASKFILE")
read -r BUCKET MODEL TW SEG CHUNK START END HALO STAGE <<< "$LINE"
if [ -z "${STAGE:-}" ] || [ "$MODEL" != "lstm_intraday" ]; then
  echo "malformed line ${SLURM_ARRAY_TASK_ID:-1} of $TASKFILE (want 9 fields, model lstm_intraday): '$LINE'"
  exit 1
fi
R="${RESULTS_ROOT:-results/linear_subsection_lstm_intraday}"
CACHE="$R/tune_cache"
ARM="$R/$BUCKET/$SEG/$MODEL/tw$TW"
if [ "$STAGE" = "tune" ]; then OUT="$ARM/stage_tune/r$CHUNK"; else OUT="$ARM/chunks/c$CHUNK"; fi
echo "task ${SLURM_ARRAY_TASK_ID:-1} on $(hostname) ($(lscpu | grep -m1 'Model name' | sed 's/.*: *//')), pool of $NT: $BUCKET $STAGE $SEG chunk $CHUNK rows [$START, $END) halo $HALO $(date)"

if [ -f "$OUT/DONE" ]; then
  echo "  $STAGE c$CHUNK already DONE"
else
  mkdir -p "$OUT"
  if HPC_KW_EXOG_BUCKET="$BUCKET" HPC_KW_SEGMENT="$SEG" HPC_KW_TRAIN_WIN="$TW" HPC_KW_LAG_SCOPE=global \
     HPC_KW_START="$START" HPC_KW_END="$END" HPC_KW_HALO="$HALO" HPC_KW_QSEL=1 HPC_KW_WINDOW_MASK=1 \
     HPC_KW_STAGE="$STAGE" HPC_KW_TUNE_CACHE="$CACHE" \
     HPC_KW_REQUIRE_CACHE="$([ "$STAGE" = tune ] && echo 0 || echo "${REQUIRE_CACHE:-1}")" \
     HPC_RESULT_DIR="$OUT" python -u specs/causal_tune_lstm_intraday.py > "$OUT/run.log" 2>&1; then
    touch "$OUT/DONE"
    echo "$(hostname) $(lscpu | grep -m1 'Model name' | sed 's/.*: *//') $(date)" > "$OUT/NODE"
    echo "  $STAGE c$CHUNK done $(date): $(tail -1 "$OUT/run.log" | cut -c1-400)"
  else
    echo "  $STAGE c$CHUNK FAILED: $(tail -3 "$OUT/run.log" | cut -c1-400)"
    exit 1
  fi
fi
if [ -n "${WRITE_FLAG:-}" ]; then
  mkdir -p "$(dirname "$WRITE_FLAG")"
  touch "$WRITE_FLAG"
  echo "wrote $WRITE_FLAG"
fi
