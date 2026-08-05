#!/bin/bash
#SBATCH --job-name=harunpen_prep
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --time=1:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=44G
#SBATCH --output=/scratch1/jc_905/harxhar-clean/logs/harunpen_prep_%j.out
cd /scratch1/jc_905/harxhar-clean
export PYTHONPATH="$PWD:$PWD/experiments${PYTHONPATH:+:$PYTHONPATH}"
export TQDM_DISABLE=1
PY=/home1/jc_905/.conda/envs/harxhar/bin/python
echo "=== enetreg2_harunpen (6 HAR mains UNPENALIZED via FWL/OLS profile-out; L1+L2 on rest) vs 0.12314 ==="
$PY experiments/resid_amortized.py prep ebm all_buckets 1000 1 480 slim enetreg2_harunpen 2>&1 | grep -E 'PREP|CANON_COEFS|COEF|Error|Traceback'
echo HARUNPEN_PREP_DONE
