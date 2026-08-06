#!/bin/bash
# Submit ALL Hoffman2 arms of the unification campaign at once — no waves.
# Run ON the Hoffman2 login node from /u/scratch/j/jamesdc1/harxhar-clean.
# Arm list = the OLS family (benchmark + bucket table + negative control).
# NOTE: bucket arm names below are placeholders — replaced by the canonical
# names from the executor's ARMS registry before submission.
set -e
cd /u/scratch/j/jamesdc1/harxhar-clean
export PYTHONPATH="$PWD:$PWD/experiments${PYTHONPATH:+:$PYTHONPATH}"
PY=/u/home/j/jamesdc1/.conda/envs/hpc-pi/bin/python
# Warm the panel cache ONCE before the arrays fire.
$PY -c "from src.unification import panel_length; print('panel rows:', panel_length())"
ARMS_H2=(
  a0_ols_har
  a_bucket_moments
  a_bucket_liquidity
  a_bucket_market_ew
  a_bucket_market_vw
  a_bucket_sentiment
  a_bucket_implied_vol
  a_bucket_vol_demand
  a_bucket_all_features   # IS the joint arm (canonical enumeration)
  # a10_noexog COLLAPSED into a0 — alias verified computationally
  # (designs + 300-bar predictions bit-identical, max|diff|=0.0)
)
for arm in "${ARMS_H2[@]}"; do
  qsub -N "unif_$arm" -v ARM="$arm" jobs/sge/unification.sge
done
qstat -u "$USER" | tail -n +3 | wc -l
