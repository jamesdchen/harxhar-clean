#!/bin/bash
#SBATCH --job-name=sig_dig
#SBATCH --array=0-89
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=44G
#SBATCH --output=/scratch1/jc_905/harxhar-clean/logs/sig_dig_%A_%a.out
cd /scratch1/jc_905/harxhar-clean
export TQDM_DISABLE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/home1/jc_905/.conda/envs/harxhar/bin/python
# cfgs built INSIDE the script (no JSON through sbatch --export, which comma-splits). d8 Hero-A/B
# configs verbatim from heroB_d8.sbatch. resid_regime reads GLOBAL_CFG+REGIME_CFG; resid_subset
# reads TREE_CFG (set = the d8 global cfg so the +d8 rung uses the SAME global tree).
export GLOBAL_CFG="{\"n_estimators\":200,\"learning_rate\":0.03,\"subsample\":0.7,\"colsample_bytree\":0.3,\"min_child_weight\":50.0,\"reg_lambda\":2.0,\"n_jobs\":4,\"max_depth\":8}"
export REGIME_CFG="{\"learning_rate\":0.02,\"max_leaves\":3,\"interactions\":10,\"max_bins\":256,\"min_samples_leaf\":10,\"max_rounds\":500,\"outer_bags\":4,\"n_jobs\":4}"
export TREE_CFG="$GLOBAL_CFG"
$PY resid_amortized.py chunk_task "$CID" "$ARM" "$SLURM_ARRAY_TASK_ID" 90 "$LBL" 2>&1 | grep -E 'CHUNK|Error|Traceback'
