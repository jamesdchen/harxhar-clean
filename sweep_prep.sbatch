#!/bin/bash
#SBATCH --job-name=sweep_prep
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=44G
#SBATCH --output=/scratch1/jc_905/harxhar-clean/logs/sweep_prep_%j.out
cd /scratch1/jc_905/harxhar-clean
export TQDM_DISABLE=1
PY=/home1/jc_905/.conda/envs/harxhar/bin/python
# CLOSE_LO/CLOSE_HI arrive via sbatch --export -> in the env -> resid_amortized reads them at import,
# so cell_id auto-suffixes _cw<LO>-<HI> and every close gate shifts to this window.
echo "=== enetreg2 prep, close window $CLOSE_LO-$CLOSE_HI ==="
$PY resid_amortized.py prep xgb all_buckets 1000 1 480 slim enetreg2 2>&1 | grep -E 'PREP|Error|Traceback'
echo SWEEP_PREP_DONE "$CLOSE_LO-$CLOSE_HI"
