#!/bin/bash
#SBATCH --job-name=unif_warmup
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch1/jc_905/harxhar-clean/logs/unif_warmup_%j.out
# Builds the panel cache once, on a compute node (login-node thread ulimit
# kills the build). All unification arrays depend on this job.
cd /scratch1/jc_905/harxhar-clean
export PYTHONPATH="$PWD:$PWD/experiments${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/home1/jc_905/.conda/envs/harxhar/bin/python
$PY -c "from src.unification import panel_length; print('panel rows:', panel_length())"
