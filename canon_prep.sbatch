#!/bin/bash
#SBATCH --job-name=canon_prep
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --time=1:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=44G
#SBATCH --output=/scratch1/jc_905/harxhar-clean/logs/canon_prep_%j.out
cd /scratch1/jc_905/harxhar-clean
export TQDM_DISABLE=1
PY=/home1/jc_905/.conda/envs/harxhar/bin/python
echo "=== PREP enetreg2_canon (banded HAR basis; legibility benchmark vs 0.12314) ==="
$PY resid_amortized.py prep ebm all_buckets 1000 1 480 slim enetreg2_canon 2>&1 | grep -E 'PREP|CANON_COEFS|COEF|Error|Traceback'
echo "=== PREP enetreg2_canon_spline (canon + clean har_ma_5 magnitude spline x close; does -0.00021 reappear?) ==="
$PY resid_amortized.py prep ebm all_buckets 1000 1 480 slim enetreg2_canon_spline 2>&1 | grep -E 'PREP|CANON_COEFS|COEF|Error|Traceback'
echo CANON_PREP_DONE
