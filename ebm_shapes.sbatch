#!/bin/bash
#SBATCH --job-name=sig_shapes
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/scratch1/jc_905/harxhar-clean/logs/sig_shapes_%j.out
cd /scratch1/jc_905/harxhar-clean
export TQDM_DISABLE=1
PY=/home1/jc_905/.conda/envs/harxhar/bin/python
echo "=== EBM shape probe: CELL=$CELL SUB=$SUB OUT=$OUT ==="
$PY ebm_shapes.py "$CELL" "$SUB" "$OUT" 2>&1
echo SIG_SHAPES_DONE "$CELL"
