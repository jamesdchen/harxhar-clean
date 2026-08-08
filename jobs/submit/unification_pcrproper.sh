#!/bin/bash
# Submit the PROPER PCR family (2026-08-07): 9 arms x 100 chunks.
# Supersedes the blk3_rotK sketch. The earlier PC families could never answer
# the replacement question: blk_pcladder_* standardized each SCORE (a
# non-orthogonal, time-varying rescale that broke the equivalence — measured at
# 0.796 relative fitted-value difference — and capped the family ~1.6pp below
# the panel at every K), and blk2_pcr_* dropped the product block AND the
# indicators, so they differed from the comparator in more than the
# representation. This family fixes both.
#
# TENSOR STRUCTURE — VERIFIED against the real name list, not assumed
# (results/panel_columns.json, 1144 columns; the check is section M of
# experiments/verify_unification_shapes.py and it reconciles EXACTLY):
#     value      43 base quantities x 12 MA windows (1,2,4,...,2048) =  516
#     indicator  48 prefixes        x 12 MA windows                  =  576
#     extras     12 har + 24 regime + 16 calendar                    =   52
#                                                            total     1144
# Every value stem appears at every window and every indicator prefix appears
# at every window — no ragged columns, so the shared rotation is well posed
# (and a ragged tensor now fails LOUDLY rather than misaligning silently).
# TWO CORRECTIONS TO THE BRIEF, both load-bearing:
#   * the rotatable set is 12 x 43, NOT 12 x 92. The 48 indicator prefixes
#     exceed the 43 value stems because five quantities carry BOTH an _avail_
#     and an _active_ column (numobs and the four voldemand families). So
#     K_max = 43 and the requested K = 92/80/60 do not exist.
#   * the extras are 52 (12 har + 24 regime + 16 calendar), not 40; and 516 is
#     the VALUE count, not the indicator count (indicators are 576).
# The ladder is therefore FULL/40/30/20 for the variance family and 30/20/10/5
# for the predictive one — which also yields matched-K ordering pairs at BOTH
# 30 and 20 instead of only one.
#
# SHARED DESIGN (only ORDERING and K vary):
#   backbone       UNROTATED, UNSHRUNK — the benchmark's own basis.
#   rotated VALUE  one frozen V from the frame window (rows [W,2W), same window
#                  and same _DEGENERATE_SD liveness rule as the transmission
#                  frame), computed on the ma_1 slab's correlation matrix and
#                  applied to EVERY MA window's slab. ma_1 is the right slab:
#                  generate_har_features builds it as rolling(1).mean().shift(1),
#                  i.e. it IS the base quantity lagged one bar, the only slab
#                  that is not a smoothing of something else.
#   availability   UNROTATED, own block (a linear combination of binary
#                  indicators indicates nothing) but SHRUNK with the rotated
#                  block as "the exogenous set".
#   NO PRODUCT BLOCK — the comparison is against the two-block model.
#   INPUTS ARE STANDARDIZED (per slab, frame-window statistics — that is what
#   makes it a correlation-matrix PCA and keeps the twelve slabs commensurate
#   under one shared shrinkage). SCORES ARE NEVER STANDARDIZED. The synthetic
#   asserts the score sds are NOT all 1, so the defect that broke the PC-ladder
#   cannot silently return.
#
# ESTIMATOR: the exact positive-part James-Stein of blk3_js_tuned. No tuned
# penalty anywhere. THIS IS LOAD-BEARING: exact JS is INVARIANT under an
# orthogonal rotation of the shrunk block (proved, and asserted to 1e-8), so at
# FULL RANK the rotation is provably a no-op and the ONLY thing this family
# varies is what the truncation DISCARDS — a clean isolation of truncation and
# ordering rather than a confound of representation with penalty.
#
# PREDICTIVE ORDERING, and the statistic: directions are ranked on the FRAME
# WINDOW ONLY (rows [W,2W), which precede every scored bar, so no scored
# outcome can enter) by the MULTIPLE R^2 of regressing the frame-window target
# on that direction's OWN 12 ladder columns, with an intercept, ties broken by
# direction index. A direction is a 12-column object here, so a literal
# per-column univariate R^2 would force an arbitrary choice of which MA window
# speaks for the direction; the block R^2 treats every direction identically.
# The ordering is frozen permanently, exactly as V is, and a synthetic asserts
# that perturbing every post-frame-window row leaves it bit-identical.
#
# THE GATE (hard): at K=full with variance ordering the rotated design is an
# orthogonal reparameterization of the standardized unrotated design, and exact
# JS is rotation-invariant, so blk2_rotVarFullK_js must reproduce the unrotated
# JS fit to machine precision. MEASURED END TO END ON A FIXTURE:
#     max relative fitted-value difference 3.991e-15  (max abs 6.795e-14)
# Also asserted: at K=full the two orderings keep bitwise-identical column
# multisets (a free correctness check on the ranking code), and at K<full they
# genuinely select different directions.
#
# blk2_gated_tuned is the comparator, and it exists because blk2_tuned runs at
# oos_mult=1 and is scored on 273,554 rows while every frozen-construction arm
# is scored on 248,686 — a row-set confound that has already bitten this
# campaign once. It is blk2_tuned's design at oos_mult=2. DELIBERATELY NOT
# grid-free: it is the incumbent two-block ridge on the ordinary per-block
# grids, and it is what the PCR arms must beat; putting it on the grid-free
# estimator would change what is being compared.
#
# HOFFMAN2 (intended target; h_data now 16G, sized from measured peak):
#   cd /u/scratch/j/jamesdc1/harxhar-clean
#   for arm in blk2_rotVarFullK_js blk2_rotVarFortyK_js blk2_rotVarThirtyK_js \
#       blk2_rotVarTwentyK_js blk2_rotPredThirtyK_js blk2_rotPredTwentyK_js \
#       blk2_rotPredTenK_js blk2_rotPredFiveK_js blk2_gated_tuned; do
#     qsub -N "unif_$arm" -v ARM="$arm" -tc 60 jobs/sge/unification_serial.sge
#   done
# This script is the CARC-shaped equivalent, kept for parity with the rest of
# jobs/submit/. Run ON the CARC login node from /scratch1/jc_905/harxhar-clean.
set -e
cd /scratch1/jc_905/harxhar-clean
for arm in blk2_rotVarFullK_js blk2_rotVarFortyK_js blk2_rotVarThirtyK_js \
    blk2_rotVarTwentyK_js blk2_rotPredThirtyK_js blk2_rotPredTwentyK_js \
    blk2_rotPredTenK_js blk2_rotPredFiveK_js blk2_gated_tuned; do
  sbatch --export=ALL,ARM="$arm" --job-name="unif_$arm" jobs/slurm/unification.sbatch
done
squeue -u "$USER" -h | wc -l
