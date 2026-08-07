#!/bin/bash
# Submit the 9 OLS-family arms under the MINIMUM-NORM estimator (author
# directive 2026-08-07: pinv-based min-norm least squares, the alpha->0+
# ridge limit; see src/unification.py _walk_ols). CARC, no warmup — the
# panel cache is hot. Pattern of unification_tuned.sh.
# Run ON the CARC login node from /scratch1/jc_905/harxhar-clean.
set -e
cd /scratch1/jc_905/harxhar-clean
for arm in a0_ols_har a_bucket_moments a_bucket_liquidity a_bucket_market_ew \
           a_bucket_market_vw a_bucket_sentiment a_bucket_implied_vol \
           a_bucket_vol_demand a_bucket_all_features; do
  sbatch --export=ALL,ARM="$arm" --job-name="unif_$arm" jobs/slurm/unification.sbatch
done
squeue -u "$USER" -h | wc -l
