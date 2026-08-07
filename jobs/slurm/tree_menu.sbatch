#!/bin/bash
#SBATCH --job-name=tree_menu
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-7
#SBATCH --output=/scratch1/jc_905/harxhar-clean/logs/unif_%x_%A_%a.out
# Dev-span LightGBM menu study worker (tree expert bank). W=8 workers share
# one Optuna JournalStorage on scratch (file backend — NOT sqlite on NFS).
cd /scratch1/jc_905/harxhar-clean
export PYTHONPATH="$PWD:$PWD/experiments${PYTHONPATH:+:$PYTHONPATH}"
export TQDM_DISABLE=1 OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
PY=/home1/jc_905/.conda/envs/harxhar/bin/python
$PY experiments/tree_menu_dev.py \
    --journal-lgbm /scratch1/jc_905/harxhar-clean/results/tree_menu_journal_lgbm.log \
    --journal-xgb /scratch1/jc_905/harxhar-clean/results/tree_menu_journal_xgb.log \
    --n-trials 100
