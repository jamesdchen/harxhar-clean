#!/bin/bash
# Submit the merged-section bucket grid on CARC (author directive 2026-08-07):
# per bucket design, causally tuned ridge (br_tuned_*) and causally tuned
# free-l1 elastic net (be_tuned_*). 16 arms = 2 estimators x 8 designs
# (7 single buckets + the joint all_features, the canonical SUBGROUPS
# enumeration). See src/unification.py. No warmup — panel cache hot.
# Run ON the CARC login node from /scratch1/jc_905/harxhar-clean.
set -e
cd /scratch1/jc_905/harxhar-clean
for b in moments liquidity market_ew market_vw sentiment implied_vol vol_demand all_features; do
  for fam in br_tuned be_tuned; do
    arm="${fam}_${b}"
    sbatch --export=ALL,ARM="$arm" --job-name="unif_$arm" jobs/slurm/unification.sbatch
  done
done
squeue -u "$USER" -h | wc -l
