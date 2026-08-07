#!/bin/bash
# Submit blk4_trailG_tuned on CARC (parsimony test, 2026-08-07): the champion
# blk4_trail_tuned construction with the transmission block reduced to the
# trailing factor SCORES ONLY (20 cols, no Ghat, no operator machinery).
# The parsimony head-to-head vs blk4_trail_tuned decides whether the operator
# stays in the headline model. Single arm x 100 chunks; no warmup (cache hot).
# Run ON the CARC login node from /scratch1/jc_905/harxhar-clean.
set -e
cd /scratch1/jc_905/harxhar-clean
sbatch --export=ALL,ARM=blk4_trailG_tuned --job-name=unif_blk4_trailG_tuned \
    jobs/slurm/unification.sbatch
squeue -u "$USER" -h | wc -l
