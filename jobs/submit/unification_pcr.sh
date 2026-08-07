#!/bin/bash
# Submit the genuine-PCR arms on CARC (2026-08-07): the principal components
# REPLACE the wide exogenous design instead of augmenting it — HAR backbone +
# K trailing-standardized frozen-frame factor scores, nothing else (no exog,
# no product, no operator columns). K=20 and K=40.
# Questions: can a 20-column PCA summary replace the 526-column exogenous
# panel, and does the K=40 collapse reproduce when the factor block does NOT
# share a penalty with a wide block alongside it?
# 2 arms x 100 chunks; no warmup (cache hot). Run ON the CARC login node from
# /scratch1/jc_905/harxhar-clean.
set -e
cd /scratch1/jc_905/harxhar-clean
for arm in blk2_pcr_tuned blk2_pcrForty_tuned; do
  sbatch --export=ALL,ARM="$arm" --job-name="unif_$arm" jobs/slurm/unification.sbatch
done
squeue -u "$USER" -h | wc -l
