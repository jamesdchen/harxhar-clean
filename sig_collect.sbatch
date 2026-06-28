#!/bin/bash
#SBATCH --job-name=sig_col
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=/scratch1/jc_905/harxhar-clean/logs/sig_col_%j.out
cd /scratch1/jc_905/harxhar-clean
PY=/home1/jc_905/.conda/envs/harxhar/bin/python
$PY resid_amortized.py chunk_collect "$CID" "$ARM" "$LBL" 2>&1 | grep -E 'COLLECT|INCOMPLETE|Error'
echo SIG_COL_DONE "$CID" "$ARM"
