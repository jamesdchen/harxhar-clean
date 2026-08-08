#!/bin/bash
# Submit blk_pcladderPerRung_tuned on CARC (2026-08-07): PER-RUNG eigenbases.
# WHY: every PC construction in this campaign uses ONE eigenbasis — that of the
# raw base features — and applies it at every horizon. ma_j is a linear
# smoother, so ma_j(X) has its own cross-sectional correlation structure and
# the leading directions of the fast panel need not be the leading directions
# of the slow panel. Here each rung is SMOOTHED FIRST and rotated into its own
# frozen first-window frame second (same frame window, same _DEGENERATE_SD
# liveness rule, descending eigenvalues, top K=20).
# The penalty parameterization is UNCHANGED from blk_pcladder_tuned (pcrank
# family, 12-point (lambda0, gamma) grid, one shape shared across all rungs),
# so the BASIS is the only difference and the head-to-head is clean.
# The build log prints the fast-vs-slow subspace alignment (principal-angle
# cosines + mean |dot| of rank-matched eigenvectors): if the bases are
# effectively identical this arm is expected to TIE blk_pcladder_tuned, and
# that alignment number is the explanation, not a null result.
# Single arm x 100 chunks; no warmup (cache hot). Run ON the CARC login node
# from /scratch1/jc_905/harxhar-clean.
set -e
cd /scratch1/jc_905/harxhar-clean
sbatch --export=ALL,ARM=blk_pcladderPerRung_tuned \
    --job-name=unif_blk_pcladderPerRung_tuned jobs/slurm/unification.sbatch
squeue -u "$USER" -h | wc -l
