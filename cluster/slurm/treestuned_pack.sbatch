#!/bin/bash
# CARC Discovery (Slurm) twin of cluster/slurm/trees_pack.sbatch for the
# per-bar TREE arms with causal periodic hyperparameter tuning
# (specs/causal_tune_trees_tuned.py), split into TIME CHUNKS: one array task
# runs one chunk of one arm per model named on its line.
# TASKFILE holds one
#   "<bucket> <model[,model...]> <train_win_days> <seg> <chunk> <start> <end> <halo>"
# line per task (see submit_treestuned.sh for how start/end/halo are set); the
# models of a line run one after another, each into
#   $RESULTS_ROOT/<bucket>/<seg>/<model>/tw<TW>/chunks/c<chunk>
# (experiments/reduce_trees_tuned_chunks.py merges an arm's chunks into the
# unchunked layout later).  A chunk-arm that already wrote DONE is skipped, so a
# resubmitted task resumes.
# Exported at submit time: TASKFILE, RESULTS_ROOT (default
# results/linear_subsection_trees_tuned), CANARY_FLAG (a file the fleet
# requires), WRITE_FLAG (a file this task touches when every chunk-arm of its
# line succeeded -- the canary writes CANARY_FLAG this way), the optional axes
# HAR_BASE (empty = production base-5) and LAG_SCOPE (global | intra; default
# global), and the optional tuning switches, forwarded as HPC_KW_TUNE_IDENTITY
# / HPC_KW_QSEL (see the spec's header): TUNE_IDENTITY (0 | 1, default 0; 1 =
# the grid reduced to the shipped configuration, early stopping off -- the
# identity gate, which must reproduce the untuned arm bit for bit) and QSEL
# (0 | 1, default 1; 1 = also walk the QLIKE-argmin choice forward, written as
# results_qsel_<seg>.csv).
# Parallelism is across PROCESSES: the spec runs a pool of SLURM_CPUS_PER_TASK
# workers (specs/causal_tune_trees_tuned_jobs.py) and every model fit inside a
# worker is single-threaded, so the BLAS/OpenMP/numba thread counts are pinned
# to 1 here (a thread count above 1 would oversubscribe the task's cores).
# shap (RF TreeSHAP) is not in the harxhar env: it lives in ./pylib_trees
# (installed there by cluster/slurm/ship_trees_carc.sh for the untuned
# campaign; ship_treestuned_carc.sh only checks it), on PYTHONPATH only here.
# The #SBATCH cpus/mem/time below are fallbacks for a hand-submitted task only:
# submit_treestuned.sh passes --cpus-per-task/--mem/--time per job; they match
# its provisional defaults (a pool of 8 workers; one chunk well under 3 h).
#SBATCH --job-name=treestuned
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=logs/%x.%A.%a.out
set -eo pipefail
module load conda >/dev/null 2>&1
source activate harxhar
set -u

NT="${SLURM_CPUS_PER_TASK:-1}"
# one thread per fit: the pool of NT worker processes is the parallelism
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
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
read -r BUCKET MODELS TW SEG CHUNK START END HALO <<< "$LINE"
if [ -z "$HALO" ]; then
  echo "malformed line ${SLURM_ARRAY_TASK_ID:-1} of $TASKFILE (want 8 fields): '$LINE'"
  exit 1
fi
echo "task ${SLURM_ARRAY_TASK_ID:-1} on $(hostname), pool of $NT: $BUCKET [$MODELS] tw$TW $SEG chunk $CHUNK rows [$START, $END) halo $HALO  tune_identity=${TUNE_IDENTITY:-0} qsel=${QSEL:-1}  $(date)"
FAIL=0
for MODEL in ${MODELS//,/ }; do
  OUT="${RESULTS_ROOT:-results/linear_subsection_trees_tuned}/$BUCKET/$SEG/$MODEL/tw$TW/chunks/c$CHUNK"
  [ -f "$OUT/DONE" ] && continue
  mkdir -p "$OUT"
  if HPC_KW_MODEL="$MODEL" HPC_KW_EXOG_BUCKET="$BUCKET" HPC_KW_SEGMENT="$SEG" \
     HPC_KW_TRAIN_WIN="$TW" HPC_KW_LAG_SCOPE="${LAG_SCOPE:-global}" \
     HPC_KW_HAR_BASE="${HAR_BASE:-}" \
     HPC_KW_START="$START" HPC_KW_END="$END" HPC_KW_HALO="$HALO" \
     HPC_KW_TUNE_IDENTITY="${TUNE_IDENTITY:-0}" HPC_KW_QSEL="${QSEL:-1}" \
     HPC_RESULT_DIR="$OUT" \
     python -u specs/causal_tune_trees_tuned.py > "$OUT/run.log" 2>&1; then
    touch "$OUT/DONE"
    echo "  $MODEL $SEG c$CHUNK done $(date)"
  else
    FAIL=1
    echo "  $MODEL $SEG c$CHUNK FAILED: $(tail -1 "$OUT/run.log" | cut -c1-200)"
  fi
done
if [ "$FAIL" -eq 0 ] && [ -n "${WRITE_FLAG:-}" ]; then
  mkdir -p "$(dirname "$WRITE_FLAG")"
  touch "$WRITE_FLAG"
  echo "wrote $WRITE_FLAG"
fi
exit $FAIL
