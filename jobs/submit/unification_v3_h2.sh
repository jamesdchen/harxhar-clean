#!/bin/bash
# Hoffman2 leg of the v3 campaign: the light OLS family (a0 + 8 buckets).
# Run ON the Hoffman2 login node from /u/scratch/j/jamesdc1/harxhar-clean
# under a login shell (qsub on PATH). Warmup runs as a job (login ulimits
# kill panel builds); every array holds on it.
set -e
cd /u/scratch/j/jamesdc1/harxhar-clean
qsub jobs/sge/unif_warmup.sge
ARMS_H2=(
  a0_ols_har
  a_bucket_moments
  a_bucket_liquidity
  a_bucket_market_ew
  a_bucket_market_vw
  a_bucket_sentiment
  a_bucket_implied_vol
  a_bucket_vol_demand
  a_bucket_all_features
)
for arm in "${ARMS_H2[@]}"; do
  qsub -hold_jid unif_warmup -N "unif_$arm" -v ARM="$arm" jobs/sge/unification.sge
done
qstat -u "$USER" 2>/dev/null | tail -n +3 | wc -l
