#!/bin/bash
#SBATCH --job-name=vrpdes
#SBATCH --array=0-15
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/scratch1/jc_905/harxhar-clean/logs/vrpdes_%A_%a.out
cd /scratch1/jc_905/harxhar-clean
export TQDM_DISABLE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=/scratch1/jc_905/harxhar-clean
PY=/home1/jc_905/.conda/envs/harxhar/bin/python
export LD_PRELOAD=/home1/jc_905/.conda/envs/harxhar/lib/libstdc++.so.6
$PY -u vrp_design_worker.py
