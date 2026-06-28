#!/bin/bash
#SBATCH --job-name=knn_local
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --time=1:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/scratch1/jc_905/harxhar-clean/logs/knn_local_%j.out
cd /scratch1/jc_905/harxhar-clean
export TQDM_DISABLE=1
PY=/home1/jc_905/.conda/envs/harxhar/bin/python
$PY knn_local.py 2>&1 | grep -vE 'reloaded|version change'
echo KNN_LOCAL_SBATCH_DONE
