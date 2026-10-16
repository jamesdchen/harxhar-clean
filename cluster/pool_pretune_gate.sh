#!/bin/bash
# Gate of the pooled pre-tune study (Hoffman2): the canary's arm-a slices of nine stored pooled arms vs the
# stored arms, and the smoke tasks (cluster/pool_pretune_check.py gate); writes
# results/pool_pretune/GATE_OK on a pass.
#$ -cwd
#$ -j y
#$ -o logs/
#$ -l h_rt=1:00:00,h_data=8G
set -eo pipefail  # -u only after conda: its activate scripts read unset variables
source /u/local/apps/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate hpc-pi
set -u
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH="$PWD"
echo "gate on $(hostname)  $(date)"
python -u cluster/pool_pretune_check.py gate
