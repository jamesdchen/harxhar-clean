#!/bin/bash
#SBATCH --job-name=vrpst2
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/scratch1/jc_905/harxhar-clean/logs/vrpst2_%A_%a.out
# array size set at submit time:  sbatch --array=0-$((2**k-1)) jobs/slurm/vrp_stage2.slurm
# requires results/vrp_stage2_spec.json (written by the stage-1 analysis)
cd /scratch1/jc_905/harxhar-clean
export TQDM_DISABLE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=/scratch1/jc_905/harxhar-clean
PY=/home1/jc_905/.conda/envs/harxhar/bin/python
export LD_PRELOAD=/home1/jc_905/.conda/envs/harxhar/lib/libstdc++.so.6
$PY -u experiments/vrp_stage2_worker.py
