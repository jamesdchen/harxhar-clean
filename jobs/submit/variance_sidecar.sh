#!/bin/bash
# Submit the exog-conditional second-moment sidecar (2026-08-07) for the two
# target arms: bench = a0_ols_har, champ = blk4_trail_tuned. 100 chunks each,
# mirroring the arm chunking; requires the arms' npzs in results/unification
# (present on CARC). Run ON the CARC login node from
# /scratch1/jc_905/harxhar-clean.
set -e
cd /scratch1/jc_905/harxhar-clean
for arm in a0_ols_har blk4_trail_tuned; do
  [ -d "results/unification/$arm" ] || { echo "ABORT: results/unification/$arm absent"; exit 1; }
  sbatch --export=ALL,ARM="$arm" --job-name="var_$arm" jobs/slurm/variance_sidecar.sbatch
done
squeue -u "$USER" -h | wc -l
