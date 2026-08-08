#!/bin/bash
# Submit the TUNED-RIDGE PCR twins (2026-08-08): 7 arms x 100 chunks.
# The estimator-corrected rerun of the proper-PCR ladder.
#
# WHY THE RERUN. The _js ladder came back CONFOUNDED. James-Stein under-shrinks
# on this design (factor ~0.77-0.83 everywhere, consistent with
# blk3_js_tuned's 0.24631 / DM +26.6 verdict), so its K-curve is a
# REGULARIZATION curve, not an information curve: FullK — which discards
# nothing — was the family's WORST arm (0.2407), performance improved
# monotonically as K FELL in BOTH orderings, and every arm lost to
# blk2_gated_tuned (best rotPredFiveK 0.22389 vs 0.21997, DM +6.87).
# Under-shrinkage makes a wide design look bad and a narrow one look good
# regardless of what the discarded directions actually carry. The only clean
# readout was ordering at matched K: predictive beat variance at K=30
# (-0.00107, DM -2.94), null at K=20 (+1.63).
#
# WHAT THESE SETTLE. With shrinkage restored to the level the design demands,
# the K-curve becomes interpretable:
#   * if truncation still IMPROVES or HOLDS under proper tuning, compression is
#     real and the panel's leading directions carry its forecasting content;
#   * if the curve FLATTENS toward blk2_gated_tuned as K grows, the panel needs
#     its full span and the earlier "compression" was regularization in
#     disguise.
#
# DESIGNS ARE BIT-IDENTICAL to the _js twins — same frozen V on the ma_1 slab
# (the base-quantity vector), same per-slab input standardization, scores NEVER
# standardized, same frozen predictive ordering, indicators unrotated. The
# verify script asserts this against the _js CONSTRUCTION PATH itself rather
# than a re-derivation. The ONLY change is the estimator.
#
# ESTIMATOR: two blocks, cartesian per-block tuning exactly as the other
# 2-block tuned arms use.
#   backbone            (0.1, 1, 10) — unchanged.
#   rotated exog set    ONE block holding the rotated VALUE tensor AND the
#                       unrotated availability indicators together (the same
#                       set the _js arms shrank jointly), on
#                       logspace(-1,4,6) = 0.1, 1, 10, 100, 1000, 10000.
# The extra decade is deliberate: the tuned-ridge family pins its modal alpha
# at 1000 in 8/8 bucket designs, so a grid stopping there could not distinguish
# "1000 is optimal" from "1000 is the ceiling". If 1e4 pins too, the endpoint
# diagnostic reports it and the grid moves again.
# NOTE the two column groups share ONE alpha by construction — they are a
# single segment, because the block tuner keys its search on segment
# alpha-keys and two segments with the same key would collapse in the combo
# dict and silently waste that axis.
#
# ARMS (all oos_mult=2, legality range(9), window 24000, per-bar refit):
#   blk2_rotVarFullK_tuned    blk2_rotVarThirtyK_tuned   blk2_rotVarTwentyK_tuned
#   blk2_rotPredThirtyK_tuned blk2_rotPredTwentyK_tuned
#   blk2_rotPredTenK_tuned    blk2_rotPredFiveK_tuned
#
# ONE-CHUNK CANARY FIRST (standing rule). A healthy first chunk shows:
#   meta.tuned_alphas   NON-EMPTY, ~11 entries for a 2763-bar chunk (one per
#                       TUNE_PER=250 boundary), each with alphas =
#                       {"backbone": <0.1|1|10>, "rotexog_wide": <0.1..1e4>}.
#   n_tail_evals        18 (3 backbone x 6 exogenous, cartesian).
#   meta.tuned_grids    {"backbone": [0.1,1,10],
#                        "rotexog_wide": [0.1,1,10,100,1000,10000]}.
#   meta.n_design_cols  backbone + K*12 + 576 indicators. For K=full that is
#                       the live rank (<=43) x 12 + 576 + backbone; for K=5,
#                       60 + 576 + backbone. A count far from that means the
#                       rotation or the truncation is wrong.
#   The ENDPOINT diagnostic on rotexog_wide is the thing to read first: if
#   1e4 pins, the grid needs another decade before any K-curve is quotable.
#
# HOFFMAN2 (intended target; h_data 16G):
#   cd /u/scratch/j/jamesdc1/harxhar-clean
#   for arm in blk2_rotVarFullK_tuned blk2_rotVarThirtyK_tuned \
#       blk2_rotVarTwentyK_tuned blk2_rotPredThirtyK_tuned \
#       blk2_rotPredTwentyK_tuned blk2_rotPredTenK_tuned \
#       blk2_rotPredFiveK_tuned; do
#     qsub -N "unif_$arm" -v ARM="$arm" -tc 60 jobs/sge/unification_serial.sge
#   done
# This script is the CARC-shaped equivalent, kept for parity with the rest of
# jobs/submit/. Run ON the CARC login node from /scratch1/jc_905/harxhar-clean.
set -e
cd /scratch1/jc_905/harxhar-clean
for arm in blk2_rotVarFullK_tuned blk2_rotVarThirtyK_tuned \
    blk2_rotVarTwentyK_tuned blk2_rotPredThirtyK_tuned \
    blk2_rotPredTwentyK_tuned blk2_rotPredTenK_tuned blk2_rotPredFiveK_tuned; do
  sbatch --export=ALL,ARM="$arm" --job-name="unif_$arm" jobs/slurm/unification.sbatch
done
squeue -u "$USER" -h | wc -l
