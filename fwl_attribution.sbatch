#!/bin/bash
#SBATCH --job-name=fwl_attr
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=44G
#SBATCH --output=/scratch1/jc_905/harxhar-clean/logs/fwl_attr_%j.out
cd /scratch1/jc_905/harxhar-clean
export TQDM_DISABLE=1
PY=/home1/jc_905/.conda/envs/harxhar/bin/python
$PY fwl_attribution.py 2>&1 | grep -vE '^\s*$|reloaded|version change'
echo FWL_SBATCH_DONE
