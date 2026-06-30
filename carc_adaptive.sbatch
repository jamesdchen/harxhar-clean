#!/bin/bash
#SBATCH --job-name=adapt
#SBATCH --array=0-89
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=44G
#SBATCH --output=/scratch1/jc_905/harxhar-clean/logs/adapt_%A_%a.out
# CARC adaptive-omega regime: per-block omega tuned on a PURGED inner-val (AW_MODE = fixed|scalar|context).
# fixed@0.8 = sanity (should ~reproduce the ggrid auniform_w80 0.12022). Established 0.12033 EBM cfg + libstdc++ preload.
cd /scratch1/jc_905/harxhar-clean
export TQDM_DISABLE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/home1/jc_905/.conda/envs/harxhar/bin/python
export LD_PRELOAD=/home1/jc_905/.conda/envs/harxhar/lib/libstdc++.so.6
export GLOBAL_CFG="{\"n_estimators\":200,\"learning_rate\":0.03,\"subsample\":0.7,\"colsample_bytree\":0.5,\"min_child_weight\":50.0,\"reg_lambda\":2.0,\"n_jobs\":4,\"max_depth\":8}"
export EBM_CFG="{\"learning_rate\":0.02,\"max_leaves\":3,\"interactions\":10,\"max_bins\":256,\"min_samples_leaf\":10,\"max_rounds\":500,\"outer_bags\":4,\"n_jobs\":4}"
export MT_AUXW=0.3 MT_NBAGS=8 MT_RANK=4 MT_EPOCHS=250 MT_WD=0.1
export AW_MODE=${AW_MODE:-scalar} AW_FIXEDW=${AW_FIXEDW:-0.8} AW_K=${AW_K:-4}
$PY resid_amortized.py chunk_task_adaptive "${CID:-xgb_all_buckets_tw1000_enetreg2_linbest_rf480_slim}" "$SLURM_ARRAY_TASK_ID" 90 "${BASE:-adapt}" 2>&1 | grep -E 'CHUNK|Error|Traceback'
