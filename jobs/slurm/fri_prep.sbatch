#!/bin/bash
#SBATCH --job-name=fri_prep
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=44G
#SBATCH --output=/scratch1/jc_905/harxhar-clean/logs/fri_prep_%j.out
cd /scratch1/jc_905/harxhar-clean
export PYTHONPATH="$PWD:$PWD/experiments${PYTHONPATH:+:$PYTHONPATH}"
export TQDM_DISABLE=1
PY=/home1/jc_905/.conda/envs/harxhar/bin/python
echo "=== enetreg2_fri prep: enetreg2 + WITHIN-WEEK Friday-close path block ==="
echo "    (week vol path + week-mean voldemand value + pre-Friday demand setup; value not active-flag) ==="
$PY experiments/resid_amortized.py prep xgb all_buckets 1000 1 480 slim enetreg2_fri 2>&1 | grep -E 'PREP|FRI|Error|Traceback'
echo FRI_PREP_DONE
echo "=== enetreg2_frirel prep: Friday-week block + rolling-relative regime (stack test) ==="
$PY experiments/resid_amortized.py prep xgb all_buckets 1000 1 480 slim enetreg2_frirel 2>&1 | grep -E 'PREP|FRI|Error|Traceback'
echo FRIREL_PREP_DONE
