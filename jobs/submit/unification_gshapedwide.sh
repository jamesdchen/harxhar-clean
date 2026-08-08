#!/bin/bash
# Submit blk4_trailGShapedWide on CARC (endpoint relief for the spectral rung,
# 2026-08-07): identical to blk4_trailGShaped in every respect EXCEPT that the
# rank-shape exponent grid runs to gamma=4 instead of stopping at 2 —
# lambda_i = lambda0 * i**gamma over an 18-point (lambda0, gamma) block grid,
# still selected by the cyclic causal tuner, still nesting the flat penalty
# exactly at gamma=0.
# WHY: across 1092 retunes blk4_trailGShaped picks gamma=2, the TOP of its
# grid, ~45% of the time. By this campaign's own >20% rule that is not a valid
# selection — the grid boundary is setting the tilt, not the validation tail.
# The wide grid STRICTLY CONTAINS the original 12 points, so the increment
# against blk4_trailGShaped isolates the two added exponents alone.
# blk4_trailGShaped itself is untouched (TRANS_SHAPE_GAMMAS unchanged, separate
# block-grid key "trans_shaped_wide"), so its on-disk chunks stay reproducible.
# Single arm x 100 chunks; no warmup (cache hot). Run ON the CARC login node
# from /scratch1/jc_905/harxhar-clean.
set -e
cd /scratch1/jc_905/harxhar-clean
sbatch --export=ALL,ARM=blk4_trailGShapedWide \
    --job-name=unif_blk4_trailGShapedWide jobs/slurm/unification.sbatch
squeue -u "$USER" -h | wc -l
