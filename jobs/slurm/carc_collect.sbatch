#!/bin/bash
#SBATCH --job-name=ggcol
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=44G
#SBATCH --output=/scratch1/jc_905/harxhar-clean/logs/ggcol_%A.out
cd /scratch1/jc_905/harxhar-clean
export PYTHONPATH="$PWD:$PWD/experiments${PYTHONPATH:+:$PYTHONPATH}"
PY=/home1/jc_905/.conda/envs/harxhar/bin/python
export LD_PRELOAD=/home1/jc_905/.conda/envs/harxhar/lib/libstdc++.so.6
$PY experiments/resid_amortized.py chunk_collect "$CID" "$ARM" "$LBL" 2>&1 | grep -E 'COLLECT|INCOMPLETE|Error'
echo SIG_COL_DONE "$CID" "$ARM" "$LBL"
