#!/bin/bash
# Rerun wave 2: bucket arms under full-support masking + fixed-ridge jiggle.
# Run ON the CARC login node from /scratch1/jc_905/harxhar-clean.
set -e
cd /scratch1/jc_905/harxhar-clean
ARMS_FIX2=(
  a_bucket_moments
  a_bucket_liquidity
  a_bucket_market_ew
  a_bucket_market_vw
  a_bucket_sentiment
  a_bucket_implied_vol
  a_bucket_vol_demand
  a_bucket_all_features
  b1_ridge_a0p1
  b1_ridge_a0p3
  b1_ridge_a3
  b1_ridge_a10
)
for arm in "${ARMS_FIX2[@]}"; do
  sbatch --export=ALL,ARM="$arm" --job-name="unif_$arm" jobs/slurm/unification.sbatch
done
squeue -u "$USER" -h | wc -l
