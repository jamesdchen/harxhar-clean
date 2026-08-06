#!/bin/bash
# Submit ALL CARC arms of the unification campaign at once — no waves.
# Run ON the CARC login node from /scratch1/jc_905/harxhar-clean.
# Arm list = penalized + block ladders + diagnostics + a0 float-parity canary.
set -e
cd /scratch1/jc_905/harxhar-clean
# Login-node thread ulimit kills the panel build, so the cache warm-up runs
# as its own queued job and every array depends on it.
WID=$(sbatch --parsable jobs/slurm/unif_warmup.sbatch)
echo "warmup job: $WID"
# The ENTIRE campaign — single cluster (Hoffman2 abandoned 2026-08-06).
ARMS_CARC=(
  a0_ols_har
  a_bucket_moments
  a_bucket_liquidity
  a_bucket_market_ew
  a_bucket_market_vw
  a_bucket_sentiment
  a_bucket_implied_vol
  a_bucket_vol_demand
  a_bucket_all_features
  b1_ridge
  b2_lasso
  blk2_user blk3_user blk4_user
  blk2_doc  blk3_doc  blk4_doc
  c4_product_alone_user c4_product_alone_doc
  d3_transmission_alone_user d3_transmission_alone_doc
)
for arm in "${ARMS_CARC[@]}"; do
  sbatch --dependency=afterok:"$WID" --export=ALL,ARM="$arm" \
    --job-name="unif_$arm" jobs/slurm/unification.sbatch
done
squeue -u "$USER" -h | wc -l
