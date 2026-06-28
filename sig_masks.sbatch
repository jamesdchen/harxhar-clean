#!/bin/bash
#SBATCH --job-name=sig_masks
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=44G
#SBATCH --output=/scratch1/jc_905/harxhar-clean/logs/sig_masks_%j.out
cd /scratch1/jc_905/harxhar-clean
export TQDM_DISABLE=1
PY=/home1/jc_905/.conda/envs/harxhar/bin/python
$PY resid_amortized.py enet_masks "$CID" 2>&1 | grep -E 'ENETMASKS|Error|Traceback'
echo MASKS_DONE "$CID"
