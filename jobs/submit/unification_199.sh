#!/bin/bash
# FULL campaign rerun under the 1/99 winsorization contract + solve-verified
# OLS path. Run ON the CARC login node from /scratch1/jc_905/harxhar-clean.
# Panel cache was purged — warmup rebuilds it; every array depends on warmup.
set -e
cd /scratch1/jc_905/harxhar-clean
WID=$(sbatch --parsable jobs/slurm/unif_warmup.sbatch)
echo "warmup job: $WID"
ARMS_ALL=(
  a0_ols_har
  a_bucket_moments a_bucket_liquidity a_bucket_market_ew a_bucket_market_vw
  a_bucket_sentiment a_bucket_implied_vol a_bucket_vol_demand
  a_bucket_all_features
  b1_ridge b2_lasso
  b1_ridge_tuned b2_lasso_tuned
  b1_ridge_a0p1 b1_ridge_a0p3 b1_ridge_a3 b1_ridge_a10
  blk2_user blk3_user blk4_user
  blk2_doc blk3_doc blk4_doc
  c4_product_alone_user c4_product_alone_doc
  d3_transmission_alone_user d3_transmission_alone_doc
)
for arm in "${ARMS_ALL[@]}"; do
  sbatch --dependency=afterok:"$WID" --export=ALL,ARM="$arm" \
    --job-name="unif_$arm" jobs/slurm/unification.sbatch
done
squeue -u "$USER" -h | wc -l
