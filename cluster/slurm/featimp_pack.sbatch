#!/bin/bash
# CARC Discovery (Slurm) array task of the 16:00-bar FEATURE-IMPORTANCE refits
# (experiments/feature_importance_1530_trees.py, stage `trees`): one task = one line of
# TASKFILE, "<bucket> <model[,model...]> <k0> <k1>", = refits k0..k1-1 (every 10
# sessions) of the untuned per-bar LightGBM / XGBoost / random forest of that
# bucket, with MDI, split count, permutation importance on the causal held-out tail,
# TreeSHAP and the noise-probe fits.  Input: the shipped
# results/feature_importance_1530/_work/input_<bucket>.npz; output:
# results/feature_importance_1530/_work/trees/trees_<bucket>_<model>_<k0>_<k1>.npz
# (a model whose output exists is skipped, so a resubmitted task resumes).
# CARC, not the lighter cluster: the stored untuned tree runs this reproduces were
# fitted here (xgboost 3.2.0, shap 0.51.0 in ./pylib_trees); the other cluster's env
# has other LightGBM / XGBoost / sklearn versions and no shap.
# Exported at submit time: TASKFILE; CANARY_FLAG + CANARY_N (the fleet requires
# CANARY_N files $CANARY_FLAG.<task>, one per canary line); WRITE_FLAG (the canary
# touches $WRITE_FLAG.<task> when every model of its line wrote its output);
# CHECK_GATE=1 (the canary) prints, per model, the largest gap between the refit
# forecasts and the stored forecasts into $WRITE_FLAG.<task>.gate -- a report, not
# a stop: a refit that does not reproduce the stored forecast bit for bit is still
# the shipped configuration, and the gap is carried into the tables.
# Model threads: the thread count of the stored run (in the input file); the
# task's 4 cores bound them (4 = the largest stored count).
#SBATCH --job-name=featimp
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x.%A.%a.out
set -eo pipefail
module load conda >/dev/null 2>&1
source activate harxhar
set -u

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$PWD:$PWD/pylib_trees"

if [ -n "${CANARY_FLAG:-}" ]; then
  NFLAG=0
  for f in "$CANARY_FLAG".*; do
    case "$f" in *.gate) ;; *) [ -f "$f" ] && NFLAG=$((NFLAG + 1)) ;; esac
  done
  if [ "$NFLAG" -lt "${CANARY_N:-1}" ]; then
    echo "canary did not finish ($NFLAG of ${CANARY_N:-1} flags $CANARY_FLAG.<task>); refusing to run"
    exit 1
  fi
fi
if ! python -c "import shap, lightgbm, xgboost, sklearn; print('shap', shap.__version__, 'lightgbm', lightgbm.__version__, 'xgboost', xgboost.__version__, 'sklearn', sklearn.__version__)"; then
  echo "tree stack import failed (PYTHONPATH=$PYTHONPATH); refusing to run"
  exit 1
fi

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID:-1}p" "$TASKFILE")
read -r BUCKET MODELS K0 K1 <<< "$LINE"
if [ -z "${K1:-}" ]; then
  echo "malformed line ${SLURM_ARRAY_TASK_ID:-1} of $TASKFILE (want 4 fields): '$LINE'"
  exit 1
fi
echo "task ${SLURM_ARRAY_TASK_ID:-1} on $(hostname): $BUCKET [$MODELS] refits [$K0, $K1)  $(date)"
python -u experiments/feature_importance_1530_trees.py trees "$BUCKET" "$MODELS" "$K0" "$K1"

if [ "${CHECK_GATE:-0}" = "1" ]; then
  GATE_FILE="${WRITE_FLAG:-gate}.${SLURM_ARRAY_TASK_ID:-1}.gate"
  mkdir -p "$(dirname "$GATE_FILE")"
  python experiments/feature_importance_1530_trees.py gate "$BUCKET" "$MODELS" "$K0" "$K1" > "$GATE_FILE"
  cat "$GATE_FILE"
fi
if [ -n "${WRITE_FLAG:-}" ]; then
  mkdir -p "$(dirname "$WRITE_FLAG")"
  touch "$WRITE_FLAG.${SLURM_ARRAY_TASK_ID:-1}"
  echo "wrote $WRITE_FLAG.${SLURM_ARRAY_TASK_ID:-1}"
fi
