#!/bin/bash
#SBATCH --job-name=harx-analysis
#SBATCH --array=0-11
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%A_%a.out
# One array element = one experiment cell.  Sized from
#   python analysis/taskrunner.py --run <run> --manifest
# which prints the exact --array bound.  Tasks are idempotent, so a requeue
# after preemption re-runs only the cells actually lost.
set -euo pipefail
source "${CONDA_SOURCE:-/apps/conda/miniforge3/25.3.0/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-harxhar}"
cd "${REMOTE_PATH:-$SLURM_SUBMIT_DIR}"
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
python analysis/taskrunner.py --run "${RUN:-cadence_x_design}" \
                              --task-id "${SLURM_ARRAY_TASK_ID}"
