#!/bin/bash
# CARC Discovery (Slurm): one PACK of the 16:00-bar FEATURE-IMPORTANCE tree refits of the C1
# re-run on the de-duplicated per-bar design (checklist I7; experiments/
# feature_importance_1530_dedup.py, refits by experiments/feature_importance_1530_trees.py
# stage `trees`, the first pass's code).  TASKFILE holds one
#   "<pack> <variant> <bucket> <model[,model...]> <k0> <k1>"
# line per PROCESS (refits k0..k1-1, every 10 sessions, of the untuned per-bar LightGBM /
# XGBoost / random forest of that bucket: MDI, split count, permutation importance on the
# causal held-out tail, TreeSHAP, noise probes).  This task runs every line of pack PACK
# (default: the array index) through a pool of SLURM_CPUS_PER_TASK single-threaded workers,
# in file order (the task files list the longest lines first).
#   variant mask   : FEATIMP_WINDOW_MASK=1, root results/feature_importance_1530_dedup
#                    (every refit uses only src/models/window_mask.window_keep's columns of
#                    its own training window; the user decision of 2026-09-29)
#   variant nomask : FEATIMP_WINDOW_MASK=0, root results/feature_importance_1530_dedup_nomask
# Every fit is single-threaded (the input file's stored thread count is 1 for every model:
# the cadence campaign's T10 runs), so no number depends on how many processes share the
# node.  Input: <root>/_work/input_<bucket>.npz (shipped); output:
# <root>/_work/trees/trees_<bucket>_<model>_<k0>_<k1>.npz (a model whose output exists is
# skipped, so a resubmitted pack resumes).
# Exported at submit time: TASKFILE, PACK (canary 0), CANARY_FLAG (a file the fleet
# requires), WRITE_FLAG (touched when every line of the pack succeeded), CHECK_GATE=1 (the
# canary: per line, the refit-vs-stored forecast gaps into $WRITE_FLAG.<variant>.<n>.gate --
# a report: the stored forecasts are the UNMASKED T10 runs, so for the mask variant the gap
# is the mask's effect, for the nomask variant it is the reproduction gate).
#SBATCH --job-name=featimp_dedup
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x.%A.%a.out
set -eo pipefail
module load conda >/dev/null 2>&1
source activate harxhar
set -u

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$PWD:$PWD/pylib_trees"

if [ -n "${CANARY_FLAG:-}" ] && [ ! -f "$CANARY_FLAG" ]; then
  echo "canary did not finish ($CANARY_FLAG missing); refusing to run"
  exit 1
fi
if ! python -c "import shap, lightgbm, xgboost, sklearn; from src.models.window_mask import window_keep; print('shap', shap.__version__, 'lightgbm', lightgbm.__version__, 'xgboost', xgboost.__version__, 'sklearn', sklearn.__version__)"; then
  echo "tree stack / window_mask import failed (PYTHONPATH=$PYTHONPATH); refusing to run"
  exit 1
fi

PACK="${PACK:-${SLURM_ARRAY_TASK_ID:-0}}"
NCPU="${SLURM_CPUS_PER_TASK:-1}"
LOGD="logs/featimp_dedup_${SLURM_JOB_ID}_${PACK}"
mkdir -p "$LOGD"
N=$(awk -v p="$PACK" 'NF && $1 == p' "$TASKFILE" | wc -l)
[ "$N" -gt 0 ] || { echo "pack $PACK of $TASKFILE is empty; refusing to run"; exit 1; }
echo "pack $PACK on $(hostname): $N lines through a pool of $NCPU single-threaded workers  $(date)"

run_line() {  # "<n> <pack> <variant> <bucket> <models> <k0> <k1>"
  local NL PK VARIANT BUCKET MODELS K0 K1 ROOT WM T0
  read -r NL PK VARIANT BUCKET MODELS K0 K1 <<< "$1"
  case "$VARIANT" in
    mask) ROOT=results/feature_importance_1530_dedup; WM=1 ;;
    nomask) ROOT=results/feature_importance_1530_dedup_nomask; WM=0 ;;
    *) echo "  line $NL: unknown variant '$VARIANT'"; return 1 ;;
  esac
  T0=$SECONDS
  if FEATIMP_OUT="$ROOT" FEATIMP_WINDOW_MASK="$WM" \
     python -u experiments/feature_importance_1530_trees.py trees "$BUCKET" "$MODELS" "$K0" "$K1" \
     > "$LOGD/line_${NL}_${VARIANT}_${BUCKET}_${K0}_${K1}.log" 2>&1; then
    echo "  done line $NL $VARIANT $BUCKET [$MODELS] [$K0, $K1) in $((SECONDS - T0))s  $(date +%T)"
  else
    echo "  FAILED line $NL $VARIANT $BUCKET [$MODELS] [$K0, $K1): $(tail -1 "$LOGD/line_${NL}_${VARIANT}_${BUCKET}_${K0}_${K1}.log" | cut -c1-200)"
    return 1
  fi
  if [ "${CHECK_GATE:-0}" = "1" ]; then
    FEATIMP_OUT="$ROOT" FEATIMP_WINDOW_MASK="$WM" \
      python experiments/feature_importance_1530_trees.py gate "$BUCKET" "$MODELS" "$K0" "$K1" \
      > "${WRITE_FLAG:-gate}.${VARIANT}.${NL}.gate" 2>&1 || true
    sed "s/^/  [$VARIANT] /" "${WRITE_FLAG:-gate}.${VARIANT}.${NL}.gate"
  fi
}
export -f run_line
export LOGD CHECK_GATE="${CHECK_GATE:-0}" WRITE_FLAG="${WRITE_FLAG:-}"

set +e
awk -v p="$PACK" 'NF && $1 == p {print NR, $0}' "$TASKFILE" \
  | xargs -P "$NCPU" -I{} bash -c 'run_line "$1"' _ {}
FAIL=$?
set -e
echo "pack $PACK: pool finished, xargs exit $FAIL  $(date)"
if [ "$FAIL" -eq 0 ] && [ -n "${WRITE_FLAG:-}" ]; then
  mkdir -p "$(dirname "$WRITE_FLAG")"
  touch "$WRITE_FLAG"
  echo "wrote $WRITE_FLAG"
fi
exit $FAIL
