#!/bin/bash
#SBATCH --job-name=ctrl_ebm
#SBATCH --array=0-89
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=44G
#SBATCH --output=/scratch1/jc_905/harxhar-clean/logs/ctrlebm_%A_%a.out
# EBM-regime control on d8@cs0.5 global = the established 0.12033 ceiling. Validates the
# whole harness path (cache, global cfg, collect, smearing) independent of the MoE, so the
# MoE rungs are directly comparable. Everything hardcoded; only CID/LBL via --export.
cd /scratch1/jc_905/harxhar-clean
export TQDM_DISABLE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/home1/jc_905/.conda/envs/harxhar/bin/python
export GLOBAL_CFG="{\"n_estimators\":200,\"learning_rate\":0.03,\"subsample\":0.7,\"colsample_bytree\":0.5,\"min_child_weight\":50.0,\"reg_lambda\":2.0,\"n_jobs\":4,\"max_depth\":8}"
export REGIME_CFG="{\"learning_rate\":0.02,\"max_leaves\":3,\"interactions\":10,\"max_bins\":256,\"min_samples_leaf\":10,\"max_rounds\":500,\"outer_bags\":4,\"n_jobs\":4}"
export REGIME_MODEL=ebm
$PY resid_amortized.py chunk_task "$CID" resid_regime "$SLURM_ARRAY_TASK_ID" 90 "$LBL" 2>&1 | grep -E 'CHUNK|Error|Traceback'
