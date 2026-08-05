#!/bin/bash
#SBATCH --job-name=heroBxgb
#SBATCH --array=0-89
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=44G
#SBATCH --output=/scratch1/jc_905/harxhar-clean/logs/heroBxgb_%A_%a.out
cd /scratch1/jc_905/harxhar-clean
export PYTHONPATH="$PWD:$PWD/experiments${PYTHONPATH:+:$PYTHONPATH}"
export TQDM_DISABLE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/home1/jc_905/.conda/envs/harxhar/bin/python
# Hero B PURE POWER: d8@cs0.3 global (the proven Hero-A frontier) + a TUNED XGB regime stage (no EBM).
export GLOBAL_CFG="{\"n_estimators\":200,\"learning_rate\":0.03,\"subsample\":0.7,\"colsample_bytree\":0.3,\"min_child_weight\":50.0,\"reg_lambda\":2.0,\"n_jobs\":4,\"max_depth\":8}"
export REGIME_CFG="{\"n_estimators\":${RNEST},\"learning_rate\":${RLR},\"max_depth\":${RDEPTH},\"subsample\":0.8,\"colsample_bytree\":0.8,\"min_child_weight\":${RMCW},\"reg_lambda\":2.0,\"n_jobs\":4}"
export REGIME_MODEL=xgb
$PY experiments/resid_amortized.py chunk_task "$CID" resid_regime "$SLURM_ARRAY_TASK_ID" 90 "$LBL" 2>&1 | grep -E 'CHUNK|Error|Traceback'
