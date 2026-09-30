#!/bin/bash
# CARC Discovery (Slurm): one PACK of UNTUNED per-bar tree chunk processes of the 16:00
# cadence campaign (checklist I2, rungs T10 and T1; specs/causal_tune_trees.py on the
# de-duplicated per-bar design of commit 47f7f9c).
# TASKFILE (cluster/trees_cadence_tasks_untuned.txt) holds one
#   "<pack> <rung> <bucket> <model> <tw> <seg> <chunk> <start> <end> <halo>"
# line per chunk PROCESS; this task runs every line of pack PACK (default: the array
# index) AT ONCE, one single-threaded python process per line, on the task's cores:
#   rung t10: REFIT_EVERY 10, IMPORTANCE_EVERY 1   (the shipped cadence)
#   rung t1 : REFIT_EVERY 1,  IMPORTANCE_EVERY 10  (a refit every session)
# TreeSHAP at 16:00 (SHAP_SEGMENTS = the line's segment) at every recorded refit.
# SLURM_CPUS_PER_TASK is overridden to 1 PER PROCESS: the spec passes it to the models
# as their thread count (LightGBM num_threads, XGBoost / RF n_jobs), so every fit is
# single-threaded and no number depends on how many processes the task hosts (gate g4).
# Output of a line: $RESULTS_ROOT/<rung>/<bucket>/<seg>/<model>/tw<tw>/chunks/c<chunk>
# (experiments/trees_cadence_reduce_chunks.py merges an arm's chunks).  A chunk with
# DONE is skipped, so a resubmitted pack resumes.  Processes start STAGGER seconds apart
# (the data preparation is each process's memory peak; see submit_trees_cadence.sh).
# Exported at submit time: TASKFILE, RESULTS_ROOT (default
# results/linear_subsection_trees_dedup), PACK (canary: 0), CANARY_FLAG (a file the
# fleet requires), WRITE_FLAG (touched when every line succeeded and, if ALONE_CHECK is
# set, the alone re-run matched), ALONE_CHECK ("<rung>:<bucket>:<model>:<chunk>": after
# the pack, that chunk is run again ALONE into <its dir>_alone and compared bit for bit).
#SBATCH --job-name=trees_cadence
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=80G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x.%A.%a.out
set -eo pipefail
module load conda >/dev/null 2>&1
source activate harxhar
set -u

# one thread per fit: the task's processes are the parallelism
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$PWD:$PWD/pylib_trees"

if [ -n "${CANARY_FLAG:-}" ] && [ ! -f "$CANARY_FLAG" ]; then
  echo "canary did not finish ($CANARY_FLAG missing); refusing to run"
  exit 1
fi
if ! python -c "import shap, lightgbm, xgboost, sklearn; print('shap', shap.__version__, 'lightgbm', lightgbm.__version__, 'xgboost', xgboost.__version__, 'sklearn', sklearn.__version__)"; then
  echo "tree stack import failed (PYTHONPATH=$PYTHONPATH); refusing to run"
  exit 1
fi

ROOT="${RESULTS_ROOT:-results/linear_subsection_trees_dedup}"
PACK="${PACK:-${SLURM_ARRAY_TASK_ID:-0}}"
STAGGER="${STAGGER:-10}"
NCPU="${SLURM_CPUS_PER_TASK:-1}"
mapfile -t LINES < <(awk -v p="$PACK" 'NF && $1 == p' "$TASKFILE")
if [ "${#LINES[@]}" -eq 0 ] || [ "${#LINES[@]}" -gt "$NCPU" ]; then
  echo "pack $PACK of $TASKFILE has ${#LINES[@]} lines for $NCPU cores; refusing to run"
  exit 1
fi
echo "pack $PACK on $(hostname): ${#LINES[@]} processes on $NCPU cores, stagger ${STAGGER}s  $(date)"

chunk_dir() {  # rung bucket seg model tw chunk -> the chunk's output dir
  echo "$ROOT/$1/$2/$3/$4/tw$5/chunks/c$6"
}

run_line() {  # "<line>" [out-dir override]
  local PK RUNG BUCKET MODEL TW SEG CHUNK START END HALO RE IE OUT
  read -r PK RUNG BUCKET MODEL TW SEG CHUNK START END HALO <<< "$1"
  case "$RUNG" in
    t10) RE=10; IE=1 ;;
    t1) RE=1; IE=10 ;;
    *) echo "  unknown rung '$RUNG'"; return 1 ;;
  esac
  OUT="${2:-$(chunk_dir "$RUNG" "$BUCKET" "$SEG" "$MODEL" "$TW" "$CHUNK")}"
  if [ -f "$OUT/DONE" ]; then
    echo "  skip (DONE) $RUNG $BUCKET $MODEL c$CHUNK"
    return 0
  fi
  mkdir -p "$OUT"
  local T0=$SECONDS
  if SLURM_CPUS_PER_TASK=1 \
     HPC_KW_MODEL="$MODEL" HPC_KW_EXOG_BUCKET="$BUCKET" HPC_KW_SEGMENT="$SEG" \
     HPC_KW_TRAIN_WIN="$TW" HPC_KW_LAG_SCOPE=global HPC_KW_HAR_BASE="" \
     HPC_KW_START="$START" HPC_KW_END="$END" HPC_KW_HALO="$HALO" \
     HPC_KW_REFIT_EVERY="$RE" HPC_KW_IMPORTANCE_EVERY="$IE" HPC_KW_SHAP_SEGMENTS="$SEG" \
     HPC_RESULT_DIR="$OUT" \
     python -u specs/causal_tune_trees.py > "$OUT/run.log" 2>&1; then
    touch "$OUT/DONE"
    echo "  done $RUNG $BUCKET $MODEL c$CHUNK [$START, $END) in $((SECONDS - T0))s  $(date +%T)"
  else
    echo "  FAILED $RUNG $BUCKET $MODEL c$CHUNK: $(tail -1 "$OUT/run.log" | cut -c1-200)"
    return 1
  fi
}

PIDS=()
n=0
for L in "${LINES[@]}"; do
  run_line "$L" &
  PIDS+=("$!")
  n=$((n + 1))
  if [ "$n" -lt "${#LINES[@]}" ]; then sleep "$STAGGER"; fi
done
FAIL=0
for p in "${PIDS[@]}"; do
  wait "$p" || FAIL=1
done
echo "pack $PACK: processes finished, fail=$FAIL  $(date)"

if [ "$FAIL" -eq 0 ] && [ -n "${ALONE_CHECK:-}" ]; then
  IFS=: read -r AR AB AM AC <<< "$ALONE_CHECK"
  L=$(awk -v p="$PACK" -v r="$AR" -v b="$AB" -v m="$AM" -v c="$AC" '$1 == p && $2 == r && $3 == b && $4 == m && $7 == c' "$TASKFILE" | head -1)
  [ -n "$L" ] || { echo "ALONE_CHECK '$ALONE_CHECK' is not a line of pack $PACK"; exit 1; }
  read -r _ _ _ _ TW SEG _ _ _ _ <<< "$L"
  A=$(chunk_dir "$AR" "$AB" "$SEG" "$AM" "$TW" "$AC")
  B="${A}_alone"
  rm -rf "$B"
  echo "alone check: $ALONE_CHECK again, the only process of the task"
  run_line "$L" "$B" || FAIL=1
  if [ "$FAIL" -eq 0 ]; then
    python - "$A/causal_tune_trees/$AM/$AB/trees_$SEG.npz" "$B/causal_tune_trees/$AM/$AB/trees_$SEG.npz" <<'EOF' || FAIL=1
import sys
import numpy as np
a, b = (np.load(p, allow_pickle=False) for p in sys.argv[1:3])
keys = ("date", "pred_adj", "true_adj", "true_raw", "refit_row", "importance", "shap")
same = {k: bool(np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind == "f")) for k in keys}
print("alone vs in-pack (bit for bit):", same)
sys.exit(0 if all(same.values()) else 1)
EOF
  fi
fi
if [ "$FAIL" -eq 0 ] && [ -n "${WRITE_FLAG:-}" ]; then
  mkdir -p "$(dirname "$WRITE_FLAG")"
  touch "$WRITE_FLAG"
  echo "wrote $WRITE_FLAG"
fi
exit $FAIL
