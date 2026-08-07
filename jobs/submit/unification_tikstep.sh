#!/bin/bash
# Submit blk3_tikhonovStep_tuned on CARC (hard-vs-soft tilt, 2026-08-07):
# identical to blk3_tikhonov_tuned except the exogenous tilt grid is the UNION
# of the power family (alpha*i**gamma, smooth) and the step family (alpha, then
# alpha*1e4 beyond rank K in {20,40,80} — PCR's hard threshold with a finite
# multiplier). 21 grid points on one block; the causal tuner picks the SHAPE
# FAMILY per retune, which is the exhibit. Single arm x 100 chunks; no warmup.
# Run ON the CARC login node from /scratch1/jc_905/harxhar-clean.
set -e
cd /scratch1/jc_905/harxhar-clean
sbatch --export=ALL,ARM=blk3_tikhonovStep_tuned \
    --job-name=unif_blk3_tikhonovStep_tuned jobs/slurm/unification.sbatch
squeue -u "$USER" -h | wc -l
