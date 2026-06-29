#!/bin/bash
# Cluster-side controller for the Hero-B XGB-REGIME Optuna tune (resid_regime, REGIME_MODEL=xgb,
# linbest base, frozen d8@cs0.5 global). Runs as a long SLURM job so it survives disconnects; it
# keeps K=24 trial jobs in flight (TPE + constant_liar, 5 curated warm-starts), tell-as-each-lands,
# until 120 trials are TOLD. Study journal: .hpc/campaigns/heroBxgb_regime_linbest/optuna.db.
# Target: beat the curated XGB-regime 0.12072, challenge the EBM ceiling 0.12033.
#SBATCH --job-name=heroBxgb_ctrl
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/scratch1/jc_905/harxhar-clean/logs/heroBxgb_ctrl_%j.out
cd /scratch1/jc_905/harxhar-clean
export TQDM_DISABLE=1
PY=/home1/jc_905/.conda/envs/harxhar/bin/python
$PY -u async_tune.py --regime --no-dry-run --n-trials 120 --k 24 \
  --cpus 4 --walltime 5:00:00 --mem 44G --poll 120
echo "CONTROLLER_DONE rc=$?"
