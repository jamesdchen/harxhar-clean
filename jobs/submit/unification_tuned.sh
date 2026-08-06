#!/bin/bash
# Submit the two causally-tuned head-to-head control arms on CARC.
# Run ON the CARC login node from /scratch1/jc_905/harxhar-clean.
# Panel cache already exists — no warmup dependency needed.
set -e
cd /scratch1/jc_905/harxhar-clean
for arm in b1_ridge_tuned b2_lasso_tuned; do
  sbatch --export=ALL,ARM="$arm" --job-name="unif_$arm" jobs/slurm/unification.sbatch
done
squeue -u "$USER" -h | wc -l
