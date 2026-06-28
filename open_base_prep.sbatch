#!/bin/bash
#SBATCH --job-name=open_base_prep
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --time=1:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=44G
#SBATCH --output=/scratch1/jc_905/harxhar-clean/logs/open_base_prep_%j.out
cd /scratch1/jc_905/harxhar-clean
export TQDM_DISABLE=1
PY=/home1/jc_905/.conda/envs/harxhar/bin/python
# enetreg2_open = locked enetreg2 base (HAR x {open,close} + cumrv x close) + cross-boundary
# OVERNIGHT cumrv gated to the open (hour==9). Base-alone QLIKE vs locked enetreg2 0.12314.
$PY resid_amortized.py prep ebm all_buckets 1000 1 480 slim enetreg2_open 2>&1 | grep -E 'PREP|Error|Traceback'
echo OPEN_BASE_PREP_DONE
