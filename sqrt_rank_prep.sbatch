#!/bin/bash
#SBATCH --job-name=sqrt_rank_prep
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --time=1:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=44G
#SBATCH --output=/scratch1/jc_905/harxhar-clean/logs/sqrt_rank_prep_%j.out
cd /scratch1/jc_905/harxhar-clean
export TQDM_DISABLE=1
PY=/home1/jc_905/.conda/envs/harxhar/bin/python
echo "=== BUILD cumrv_realsqrt.npy (sqrt-scale physical cumrv; isolates the sqrt) ==="
$PY build_intraday_cumrv.py 2>&1 | grep -E 'SAVE|SQRT|PASS|FAIL|sliced array len|cumrv_realsqrt|Error|Traceback'
echo "=== PREP enetreg2_realsqrt (cumsum of sqrt(adj RV) x close; vs real 0.12436 / proxy 0.12314) ==="
$PY resid_amortized.py prep ebm all_buckets 1000 1 480 slim enetreg2_realsqrt 2>&1 | grep -E 'PREP|Error|Traceback'
echo "=== PREP enetreg2_har5rank (rank-Gauss har_ma_5 x close added to enetreg2; vs 0.12314) ==="
$PY resid_amortized.py prep ebm all_buckets 1000 1 480 slim enetreg2_har5rank 2>&1 | grep -E 'PREP|Error|Traceback'
echo SQRT_RANK_PREP_DONE
