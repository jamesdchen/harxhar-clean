#!/bin/bash
# Submit the ADAPTIVE-VS-FIXED TILT control + the two ENDPOINT-RELIEF twins
# (2026-08-07): 3 arms x 100 chunks.
#
# ---- blk4_trailGShapedFrozen : adaptive vs fixed spectral tilt --------------
# NOT a standardization-convention arm. Standardizing factor score i divides it
# by ~sqrt(d_i), so a penalty lambda_i on the standardized coefficient is
# lambda_i * d_i on the raw eigen-direction: STANDARDIZATION IS ITSELF A
# SPECTRAL TILT (gamma_eff = gamma - 1.176 under the measured spectrum
# d_i ~ i^-1.176). The two conventions differ in whether that tilt ADAPTS:
#   TRAILING (blk4_trailGShapedWide) divides by a ROLLING sd -> time-varying
#     tilt that tracks the CURRENT spectrum; -1.176 is only its average.
#   FROZEN   (this arm)              divides by a CONSTANT frame-window sd ->
#     a FIXED tilt of the same average magnitude.
# The paper currently attributes trailing-beats-frozen to "causal scale
# stability". If that is the whole story the two arms should nearly tie once
# both can reach the same gamma_eff. If trailing still wins, the claim becomes
# stronger and more interesting: an ADAPTIVE spectral prior beats a fixed one.
# Both arms carry the SAME wide exponent grid (0,0.5,1,2,3,4) — essential, not
# incidental: the two standardizations sit at different points of one tilt
# family, so a narrow grid would confound the mechanism with grid REACH.
#
# ---- the two endpoint-relief twins -----------------------------------------
# The shape-endpoint diagnostic added to the scorer fired on real data. The
# extension DIRECTION for each was read off that arm's own 1092 persisted
# retunes, not assumed:
#   blk_pcladderWide_tuned        pcrank pinned at BOTH ends — gamma=0 49%,
#                                 gamma=2 32% (81% at an endpoint), LOW end
#                                 dominant. Grid made BIPOLAR (-1..4, 24 pts).
#                                 Negative exponents are the right relief, not
#                                 an exotic prior: these scores are trailing-
#                                 standardized, so gamma=0 is ALREADY an
#                                 effective -1.176 tilt and a tuner sitting
#                                 there half the time wants to go further.
#   blk3_tikhonovStepWide_tuned   the two families pin in OPPOSITE directions:
#                                 power at the gamma=0 FLOOR (56%), step at the
#                                 K0=80 CEILING (48%). Power extended DOWN to
#                                 -1 (this block is an orthogonal rotation with
#                                 no rescaling, so gamma_eff == gamma and
#                                 gamma=0 is literally plain scalar ridge);
#                                 step extended both ways to (5,10,20,40,80,100).
# Both change ONLY their shape grid; the narrow originals are untouched and
# stay byte-reproducible, and each narrow grid is a bit-exact STRICT SUBSET of
# its widened twin, so the increment is attributable to the added points alone.
#
# TWO DELIBERATE DEVIATIONS, both measured and recorded in the source:
#  1. power gamma=4 is EXCLUDED from the 526-wide exog_rot axis (capped at 3).
#     That family spans the whole rotated design, so its range is
#     lambda0*526**g: gamma=4 gives 7.65e14 and a penalized-gram condition
#     number of 5.3e16, which EXCEEDS 1/eps = 4.5e15 — numerically singular to
#     working precision. It also buys nothing: lambda_1/lambda_526 = 7.6e10 is
#     not a tilt, it is hard truncation at rank ~10, and hard truncation is
#     already in the same block grid as the step family. gamma=3 is safe
#     (cond 1.2e11, fitted agreement 9.3e-14). The exponent ceiling is a
#     function of BLOCK WIDTH — the 40-wide transmission block carries gamma=4
#     safely (cond 2.6e11). No cap constant was introduced; a clip that bites
#     would destroy the signal-carrying extreme.
#  2. the step axis is (5,10,20,40,80,100), not the requested
#     (5,10,20,30,60,100), which would have DROPPED K0=40 and K0=80 — the two
#     points the shipped arm selects most (29% and 48%). Dropping the incumbent
#     optimum would make the wide-vs-narrow increment uninterpretable. Same
#     cardinality, same both-ended widening, superset restored.
#
# Cyclic cost per retune: Frozen 70, pcladderWide 88, tikhonovStepWide 127
# (shipped counterparts 52 / 61 / 73) — all inside the ~150 ceiling.
#
# HOFFMAN2 (intended target; CARC is saturated by the tree bank):
#   cd /u/scratch/j/jamesdc1/harxhar-clean
#   for arm in blk4_trailGShapedFrozen blk_pcladderWide_tuned \
#       blk3_tikhonovStepWide_tuned; do
#     qsub -N "unif_$arm" -v ARM="$arm" -tc 60 jobs/sge/unification_serial.sge
#   done
# This script is the CARC-shaped equivalent, kept for parity with the rest of
# jobs/submit/. Run ON the CARC login node from /scratch1/jc_905/harxhar-clean.
set -e
cd /scratch1/jc_905/harxhar-clean
for arm in blk4_trailGShapedFrozen blk_pcladderWide_tuned \
    blk3_tikhonovStepWide_tuned; do
  sbatch --export=ALL,ARM="$arm" --job-name="unif_$arm" jobs/slurm/unification.sbatch
done
squeue -u "$USER" -h | wc -l
