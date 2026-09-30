#!/bin/bash
# CARC Discovery (Slurm): the density inputs of the C2 dense-vs-sparse re-run on the
# de-duplicated per-bar design (checklist I8, experiments/dvs_dedup_1530.py): the masked T10
# walk of LightGBM, XGBoost and random forest (specs/causal_tune_trees.py shipped
# configuration, refit every 10 sessions, one thread per fit, per-window column mask of
# src/models/window_mask.py) on all columns of the live_feasible and all_features designs,
# with TreeSHAP at every refit, split into the six 250-session blocks.
# Exported at submit time (cluster/slurm/submit_dvs_dedup.sh): DVS_MODE = canary | fleet | gate
# (gate: the walk WITHOUT the mask on block 5 of live_feasible vs the stored unmasked T10 run).
# The canary runs block 5 of live_feasible for the three models and writes CANARY_OK; the
# fleet refuses to run without it, runs every other block chunk in a pool of
# SLURM_CPUS_PER_TASK single-threaded worker processes, assembles the runs and writes DONE.
# Chunks already on disk are skipped, so a resubmitted task resumes.
# shap (random forest TreeSHAP) lives in ./pylib_trees (installed there for the untuned tree
# campaign), on PYTHONPATH only here.
#SBATCH --job-name=dvs_dedup
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x.%j.out
set -eo pipefail
module load conda >/dev/null 2>&1
source activate harxhar
set -u

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_${SLURM_JOB_ID}"
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$PWD:$PWD/pylib_trees"
export DVS_WORK="$PWD/results/dense_vs_sparse_dedup/_work_carc"
MODE=${DVS_MODE:?DVS_MODE must be canary, fleet or gate}

if [ "$MODE" = fleet ] && [ ! -f "$DVS_WORK/CANARY_OK" ]; then
  echo "canary did not finish ($DVS_WORK/CANARY_OK missing); refusing to run"
  exit 1
fi
echo "dvs_dedup $MODE on $(hostname), ${SLURM_CPUS_PER_TASK:-1} CPUs, job ${SLURM_JOB_ID}"
python -u experiments/dvs_dedup_1530.py carc "$MODE"
