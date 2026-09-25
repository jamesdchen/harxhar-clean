#!/bin/bash
# CARC Discovery (Slurm) twin of cluster/slurm/subsection_pack.sbatch for the
# per-bar TREE arms (specs/causal_tune_trees.py): a PACK of arms per array task.
# TASKFILE holds one
#   "<bucket> <model[,model...]> <train_win_days> <seg1,seg2,...>"
# line per task; the arms of a line (models x segments) run one after another.
# An arm that already wrote DONE is skipped, so a resubmitted task resumes.
# Exported at submit time: TASKFILE, RESULTS_ROOT (default
# results/linear_subsection_trees), CANARY_FLAG (a file the fleet requires),
# WRITE_FLAG (a file this task touches when every arm of its line succeeded --
# the canary writes CANARY_FLAG this way), and the optional axes HAR_BASE
# (empty = production base-5) and LAG_SCOPE (global | intra; default global).
# Unlike the linear pack, the model uses the task's cores: the spec reads
# SLURM_CPUS_PER_TASK for LightGBM num_threads / XGBoost and RF n_jobs.
# shap (RF TreeSHAP) is not in the harxhar env: it lives in ./pylib_trees
# (installed there by cluster/slurm/ship_trees_carc.sh), on PYTHONPATH only here.
#SBATCH --job-name=trees
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x.%A.%a.out
set -eo pipefail
module load conda >/dev/null 2>&1
source activate harxhar
set -u

NT="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="$NT" MKL_NUM_THREADS="$NT" OPENBLAS_NUM_THREADS="$NT" NUMBA_NUM_THREADS="$NT"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$PWD:$PWD/pylib_trees"

# Slurm's afterok dependency already refuses to start the fleet if the canary
# failed; the flag check is the belt to that suspender.
if [ -n "${CANARY_FLAG:-}" ] && [ ! -f "$CANARY_FLAG" ]; then
  echo "canary did not finish ($CANARY_FLAG missing); refusing to run"
  exit 1
fi

# Fail fast (before any arm) if the tree stack does not import on this node.
if ! python -c "import shap, lightgbm, xgboost, sklearn; print('shap', shap.__version__, shap.__file__); print('lightgbm', lightgbm.__version__, 'xgboost', xgboost.__version__, 'sklearn', sklearn.__version__)"; then
  echo "tree stack import failed (shap/lightgbm/xgboost/sklearn; PYTHONPATH=$PYTHONPATH); refusing to run"
  exit 1
fi

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID:-1}p" "$TASKFILE")
read -r BUCKET MODELS TW SEGS <<< "$LINE"
echo "task ${SLURM_ARRAY_TASK_ID:-1} on $(hostname) with $NT cpus: $BUCKET [$MODELS] tw$TW [$SEGS]  $(date)"
FAIL=0
for MODEL in ${MODELS//,/ }; do
  for SEG in ${SEGS//,/ }; do
    OUT="${RESULTS_ROOT:-results/linear_subsection_trees}/$BUCKET/$SEG/$MODEL/tw$TW"
    [ -f "$OUT/DONE" ] && continue
    mkdir -p "$OUT"
    if HPC_KW_MODEL="$MODEL" HPC_KW_EXOG_BUCKET="$BUCKET" HPC_KW_SEGMENT="$SEG" \
       HPC_KW_TRAIN_WIN="$TW" HPC_KW_LAG_SCOPE="${LAG_SCOPE:-global}" \
       HPC_KW_HAR_BASE="${HAR_BASE:-}" HPC_RESULT_DIR="$OUT" \
       python -u specs/causal_tune_trees.py > "$OUT/run.log" 2>&1; then
      touch "$OUT/DONE"
      echo "  $MODEL $SEG done $(date)"
    else
      FAIL=1
      echo "  $MODEL $SEG FAILED: $(tail -1 "$OUT/run.log" | cut -c1-200)"
    fi
  done
done
if [ "$FAIL" -eq 0 ] && [ -n "${WRITE_FLAG:-}" ]; then
  mkdir -p "$(dirname "$WRITE_FLAG")"
  touch "$WRITE_FLAG"
  echo "wrote $WRITE_FLAG"
fi
exit $FAIL
