#!/bin/bash
# Submit blk4_trailGShapedWide on CARC (endpoint relief for the spectral rung,
# 2026-08-07): identical to blk4_trailGShaped in every respect EXCEPT that the
# rank-shape exponent grid is BIPOLAR, running from -1 to 4 instead of
# stopping at 0 and 2 — lambda_i = lambda0 * i**gamma over a 24-point
# (lambda0, gamma) block grid, still selected by the cyclic causal tuner, still
# nesting the flat penalty bit-exactly at gamma=0.
# WHY: the pinning is TWO-SIDED. Across blk4_trailGShaped's 1092 retunes 45%
# sit at the exponent grid FLOOR (gamma=0) and 41% at the CEILING (gamma=2) —
# 86% at an endpoint, against this campaign's own >20% rule. An upward-only
# extension would have relieved the smaller half.
# Negative exponents are coherent here, not exotic: lambda_i stays strictly
# positive and monotone for gamma<0, it simply shrinks the LEADING directions
# hardest — which is what the trailing standardization already does
# implicitly, so gamma=0 is an interior point of the mechanism rather than a
# natural boundary.
# The wide grid STRICTLY CONTAINS the original 12 points, so the increment
# against blk4_trailGShaped isolates the added exponents alone.
# blk4_trailGShaped itself is untouched (TRANS_SHAPE_GAMMAS unchanged, separate
# block-grid key "trans_shaped_wide"), so its on-disk chunks stay reproducible.
# Single arm x 100 chunks; no warmup (cache hot). Run ON the CARC login node
# from /scratch1/jc_905/harxhar-clean.
set -e
cd /scratch1/jc_905/harxhar-clean
sbatch --export=ALL,ARM=blk4_trailGShapedWide \
    --job-name=unif_blk4_trailGShapedWide jobs/slurm/unification.sbatch
squeue -u "$USER" -h | wc -l
