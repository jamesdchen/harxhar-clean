#!/bin/bash
# CARC array task: the drop-column / permutation walk of the per-bar LASSO on the
# all_features design (experiments/feature_importance_1530.py, stage `linear_part`)
# for ONE refit per task.  Only the refits where the causally chosen penalty is the
# grid's smaller value (0.001: 115-120 active columns, ~220 units to re-solve with the
# spec's own homotopy, ~5 min per refit) run here; the rest run locally.  TASKFILE
# holds one "<bucket> <estimator> <k0> <k1>" line per task; output
# results/feature_importance_1530/_work/linear/linear_<bucket>_<est>_part<k0>_<k1>.npz.
# Reads: the shipped input_<bucket>.npz, the coefficient capture
# results/model_diagnostics_1530/capture_bar1600_all_features_reclasso.npz, the stored
# research forecasts (gate), experiments/model_diagnostics_1530.py (executes the
# spec's estimator section from specs/causal_tune_linear.py, already here).
# CANARY_FLAG (a file the fleet requires) / WRITE_FLAG (the canary touches it).
#SBATCH --job-name=featimp_lin
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x.%A.%a.out
set -eo pipefail
module load conda >/dev/null 2>&1
source activate harxhar
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$PWD"
if [ -n "${CANARY_FLAG:-}" ] && [ ! -f "$CANARY_FLAG" ]; then
  echo "canary did not finish ($CANARY_FLAG missing); refusing to run"
  exit 1
fi
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID:-1}p" "$TASKFILE")
read -r BUCKET EST K0 K1 <<< "$LINE"
if [ -z "${K1:-}" ]; then
  echo "malformed line ${SLURM_ARRAY_TASK_ID:-1} of $TASKFILE: '$LINE'"
  exit 1
fi
echo "task ${SLURM_ARRAY_TASK_ID:-1} on $(hostname): $BUCKET $EST refits [$K0, $K1)  $(date)"
python -u experiments/feature_importance_1530.py linear_part "$BUCKET" "$EST" "$K0" "$K1"
if [ -n "${WRITE_FLAG:-}" ]; then
  touch "$WRITE_FLAG"
  echo "wrote $WRITE_FLAG"
fi
