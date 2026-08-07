#!/bin/bash
# Submit the five causally-tuned BLOCK arms on CARC (author directive
# 2026-08-06: per-block alphas re-selected jointly every 250 solves over
# BLOCK_TUNE_GRIDS; see src/unification.py _walk_blocks_tuned).
# Run ON the CARC login node from /scratch1/jc_905/harxhar-clean.
# Panel cache already exists — no warmup dependency needed.
set -e
cd /scratch1/jc_905/harxhar-clean
for arm in blk2_tuned blk3_tuned blk4_tuned c4_product_alone_tuned d3_transmission_alone_tuned; do
  sbatch --export=ALL,ARM="$arm" --job-name="unif_$arm" jobs/slurm/unification.sbatch
done
squeue -u "$USER" -h | wc -l
