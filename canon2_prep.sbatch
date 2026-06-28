#!/bin/bash
#SBATCH --job-name=canon2_prep
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --time=1:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=44G
#SBATCH --output=/scratch1/jc_905/harxhar-clean/logs/canon2_prep_%j.out
cd /scratch1/jc_905/harxhar-clean
export TQDM_DISABLE=1
PY=/home1/jc_905/.conda/envs/harxhar/bin/python
echo "=== (E) har5rank_decomp: is har5rank static curvature or rolling/relative? ==="
$PY har5rank_decomp.py 2>&1 | grep -E 'E-TEST|R2|win_slot|DONE|Error|Traceback'
echo "=== (1) enetreg2_canon_scaled (robust-scaled bands; should recover ~0.12314) ==="
$PY resid_amortized.py prep ebm all_buckets 1000 1 480 slim enetreg2_canon_scaled 2>&1 | grep -E 'PREP|CANON_COEFS|COEF|Error|Traceback'
echo "=== (2a) enetreg2_har5spl (ISOLATED dense static spline on enetreg2; vs 0.12314 / har5rank 0.12293) ==="
$PY resid_amortized.py prep ebm all_buckets 1000 1 480 slim enetreg2_har5spl 2>&1 | grep -E 'PREP|CANON_COEFS|COEF|Error|Traceback'
echo "=== (2b) enetreg2_har5rank_spl (STACKING: enetreg2 + har5rank + dense spline) ==="
$PY resid_amortized.py prep ebm all_buckets 1000 1 480 slim enetreg2_har5rank_spl 2>&1 | grep -E 'PREP|CANON_COEFS|COEF|Error|Traceback'
echo CANON2_PREP_DONE
