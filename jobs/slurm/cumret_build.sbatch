#!/bin/bash
#SBATCH --job-name=cumret_build
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch1/jc_905/harxhar-clean/logs/cumret_build_%j.out
cd /scratch1/jc_905/harxhar-clean
export PYTHONPATH="$PWD:$PWD/experiments${PYTHONPATH:+:$PYTHONPATH}"
export TQDM_DISABLE=1
PY=/home1/jc_905/.conda/envs/harxhar/bin/python
$PY experiments/build_intraday_cumret.py 2>&1 | grep -E 'SAVED|causality|DONE|Error|Traceback'
