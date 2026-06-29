#!/bin/bash
#SBATCH --job-name=moe_dig
#SBATCH --array=0-89
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=44G
#SBATCH --output=/scratch1/jc_905/harxhar-clean/logs/moe_%A_%a.out
# Hero-B regime stage = interpretable regime MoE (REGIME_MODEL=moe), gated h16-19.
# GLOBAL stage = d8@cs0.5 (the retuned Hero-A frontier that produced the 0.12033 EBM ceiling).
# Env in: CID, RDEPTH (gate depth; 0=single global NAM baseline), RDIAG (true/false high-order probe), LBL.
cd /scratch1/jc_905/harxhar-clean
export TQDM_DISABLE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/home1/jc_905/.conda/envs/harxhar/bin/python
export GLOBAL_CFG="{\"n_estimators\":200,\"learning_rate\":0.03,\"subsample\":0.7,\"colsample_bytree\":0.5,\"min_child_weight\":50.0,\"reg_lambda\":2.0,\"n_jobs\":4,\"max_depth\":8}"
DIAG=${RDIAG:-false}
export REGIME_CFG="{\"depth\":${RDEPTH},\"epochs\":150,\"hidden_dim\":16,\"dropout\":0.15,\"n_bags\":8,\"colsample\":0.5,\"lr\":0.001,\"weight_decay\":${WD:-0.1},\"aux_weight\":0.5,\"val_frac\":0.2,\"patience\":12,\"use_diagnostic_mlp\":${DIAG}}"
export REGIME_MODEL=moe
$PY resid_amortized.py chunk_task "$CID" resid_regime "$SLURM_ARRAY_TASK_ID" 90 "$LBL" 2>&1 | grep -E 'CHUNK|Error|Traceback'
