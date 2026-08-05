#!/bin/bash
#SBATCH --job-name=ivmag_prep
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --time=1:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=44G
#SBATCH --output=/scratch1/jc_905/harxhar-clean/logs/ivmag_prep_%j.out
cd /scratch1/jc_905/harxhar-clean
export PYTHONPATH="$PWD:$PWD/experiments${PYTHONPATH:+:$PYTHONPATH}"
export TQDM_DISABLE=1
PY=/home1/jc_905/.conda/envs/harxhar/bin/python
echo "=== enetreg2_ivmag prep (model=xgb): enetreg2 + rolling per-slot standardized vol innovations (z_har_*) ==="
$PY experiments/resid_amortized.py prep xgb all_buckets 1000 1 480 slim enetreg2_ivmag 2>&1 | grep -E 'PREP|Error|Traceback'
echo REL_PREP_DONE
