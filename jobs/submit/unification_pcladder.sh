#!/bin/bash
# Submit blk_pcladder_tuned on CARC (2026-08-07): ridge on LADDER-EXPANDED
# principal components — the moving-average ladder applied to the eigenvector
# SERIES, not the raw features, so every column carries a (PC rank x horizon)
# identity and ONE rank-tilted penalty is shared across each rank's rungs.
# Justified by ma_j(V'x) == V' ma_j(x): the same subspace as rotating the
# expanded design, organized so the tilt is a prior over DIRECTIONS.
# Single arm x 100 chunks; no warmup (cache hot). Run ON the CARC login node
# from /scratch1/jc_905/harxhar-clean.
set -e
cd /scratch1/jc_905/harxhar-clean
sbatch --export=ALL,ARM=blk_pcladder_tuned --job-name=unif_blk_pcladder_tuned \
    jobs/slurm/unification.sbatch
squeue -u "$USER" -h | wc -l
