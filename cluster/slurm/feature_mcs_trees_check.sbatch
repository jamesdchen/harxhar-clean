#!/bin/bash
# Feature MCS, the tree candidates (2026-10-08): one check step of experiments/feature_mcs_trees.py on one CPU --
# MODE canary_check | gate (EST lgbm | xgb | rf) | collect.  The exit status is the step's verdict, so a fleet held
# afterok on a gate runs only when the gate passed.
#SBATCH --job-name=mct_check
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --output=logs/%x.%j.out
set -eo pipefail
module load conda >/dev/null 2>&1
source activate harxhar
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
MODE="${MODE:?MODE canary_check | gate | collect}"
echo "check $MODE ${EST:-} on $(hostname)  $(date)"
if [ "$MODE" = gate ]; then
  python experiments/feature_mcs_trees.py gate --est "${EST:?EST}"
else
  python experiments/feature_mcs_trees.py "$MODE"
fi
