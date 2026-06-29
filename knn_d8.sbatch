#!/bin/bash
#SBATCH --job-name=knn_d8
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/scratch1/jc_905/harxhar-clean/logs/knn_d8_%j.out
# Temporal kNN on the SAME d8@cs0.5 leftover the MoE/EBM regime stages model (linbest Hero-A
# fa_d8c5 chunks). Two leakage-guarded variants, both with shuffle placebos:
#   knn_analog  = causal kNN analog (gaussian/uniform weighted MEAN of past close analogs)
#   knn_local   = large-K local-linear RIDGE relevance regression (the "proper" variant, -0.00018 prior)
# Reads saved Hero-A preds + the Xs embedding cols; no training, fast (CPU). Independent of the MoE chain.
cd /scratch1/jc_905/harxhar-clean
export TQDM_DISABLE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/home1/jc_905/.conda/envs/harxhar/bin/python
export KNN_CELL=xgb_all_buckets_tw1000_enetreg2_linbest_rf480_slim
export KNN_LBL=resid_subset_fa_d8c5
echo "=== knn_analog (causal kNN analog, weighted mean) ==="
$PY knn_analog.py 2>&1 | grep -E 'OOS|Hero A|K=|DONE|Error|Traceback' || true
echo "=== knn_local (large-K local-linear ridge relevance) ==="
$PY knn_local.py 2>&1 | grep -E 'close=|Hero A|intercept|^  har|^  V|^  all|SHUFFLE|DONE|Error|Traceback' || true
echo KNN_D8_ALLDONE
