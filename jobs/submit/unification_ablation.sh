#!/bin/bash
# Submit the transmission ablation triple on CARC (author directive
# 2026-08-07): G-only / Ghat-only / per-block-tuned variants of the
# trailing-standardized four-block ridge. See src/unification.py
# (_transmission_block parts= + blk4_trailG/blk4_trailGhat/blk4_trail_tuned).
# Run ON the CARC login node from /scratch1/jc_905/harxhar-clean.
# Panel cache is hot — no warmup dependency needed.
set -e
cd /scratch1/jc_905/harxhar-clean
for arm in blk4_trailG blk4_trailGhat blk4_trail_tuned; do
  sbatch --export=ALL,ARM="$arm" --job-name="unif_$arm" jobs/slurm/unification.sbatch
done
squeue -u "$USER" -h | wc -l
