#!/bin/bash
# Submit the REACH-MATCHED elastic-net arms (2026-08-07): 8 designs x 100
# chunks. THIS IS A GRID DEFECT REPAIR, NOT A NEW HYPOTHESIS — the paper's
# "shrinkage beats selection" conclusion currently rests on a head-to-head
# between two families whose penalty grids do not span the same shrinkage.
#
# THE ARITHMETIC. reclasso's sklearn-compatible mapping (src/models/reclasso_har
# module docstring) over N = 24000 window rows is
#     mu   = N * alpha * l1_ratio        (L1)
#     lam2 = N * alpha * (1 - l1_ratio)  (L2, the ridge-equivalent)
# be_tuned's grid tops at alpha = 1e-2, so its LARGEST expressible ridge-
# equivalent penalty is 24000 * 1e-2 * 0.75 = 180 at l1_ratio=0.25 and less at
# every heavier mixing. The tuned RIDGE grid reaches alpha = 1000 and SELECTS
# that top point in 41.3% of retunes. The elastic net cannot reach the
# shrinkage ridge picks; the two arms are not comparable at the margin where
# ridge actually operates.
#
# THE MEASURED SYMPTOM: the enet's pooled deficit is entirely at its own grid
# ceiling. Where the tuner picked alpha <= 1e-3 (72.1% of bars) the ENET WINS
# (d = -0.00039, DM -6.70, 8/8 designs); where it picked alpha = 1e-2 (27.9%)
# it loses by +0.00250 (DM +6.85, 8/8 designs, 18/24 years). Alpha-endpoint DiD
# +0.00289, stacked z = +7.78. Controlling alpha halves the l1_ratio effect and
# leaves it significant in 1 of 8 designs with a sign reversal at 1e-4 — mixing
# is a symptom, not the channel.
#
# THE GRID TOP IS DERIVED, NOT ROUNDED. Every mixing value must be able to
# express ridge's selected lam2 = 1000; inverting the mapping:
#     l1=0.25 -> alpha = 1000/(24000*0.75) = 0.0556
#     l1=0.50 -> alpha = 1000/(24000*0.50) = 0.0833
#     l1=0.75 -> alpha = 1000/(24000*0.25) = 0.1667   <- the BINDING case
#     l1=1.00 -> unreachable at any alpha (no L2 term at pure lasso)
# so a top of 1e-1 would NOT suffice. Top = 1e0, alphas = logspace(-6,0,7).
# Maximum expressible lam2 at that top (= 24000 * 1 * (1 - l1)):
#     l1=0.25 -> 18000 (18.0x ridge)   l1=0.50 -> 12000 (12.0x)
#     l1=0.75 ->  6000 ( 6.0x)         l1=1.00 ->     0 (structural)
# Past ridge's reach at every mixing where the comparison is defined, so an
# endpoint selection at 1e0 now MEANS the enet wants more shrinkage than
# ridge's own grid offers, rather than that it ran out of room.
#
# STRUCTURAL LIMIT AT PURE LASSO: at l1_ratio=1.0 the L2 term is identically
# zero for every alpha, so pure lasso can never match ridge's shrinkage however
# far the axis extends. That is the definition of the family. This grid removes
# the reach confound at l1_ratio < 1 only; a residual pure-lasso deficit that
# survives is a genuine family limitation and may be reported as one.
#
# DEGENERATE CORNER, VERIFIED end-to-end through the warm homotopy on a fixture
# (experiments/verify_unification_shapes.py section K): alpha=1e0 at
# l1_ratio=1.0 gives mu = 24000, far above mu_max = max|X'y|, and the homotopy
# DOES return an empty active set. It is safe by construction, not by luck —
# the intercept is a locked augmented column taking neither the L1 penalty nor
# the L2 ridge, so it never leaves the active set and the forecast degrades to
# the intercept-only limit (window mean of y, matched to 3e-18). Not a bug, and
# no longer silent: every retune's active-set size is persisted and
# meta.tuned_penalty_summary carries frac_intercept_only / min_n_active /
# mean_n_active, surfaced by the scorer as estimator_penalty_summary.csv.
#
# Everything else is identical to be_tuned_<bucket>: same per-bucket design,
# free l1_ratio {0.25,0.5,0.75,1.0}, warm enet_online homotopy, identifiability
# mask, window 24000, TUNE_PER=250, 25-embargo/125-tail, per-bar refit, same
# (alpha, l1_ratio) persistence. The original 20 grid points are carried
# through as the SAME tuples in the SAME order, so be_tunedWide nests
# be_tuned's choice set bit-exactly and preserves its tie-break.
# These are full-length arms (no frozen construction) — all 100 chunks legal,
# no legal-missing set, matching be_tuned_*.
#
# HOFFMAN2 (intended target; CARC is saturated by the tree bank):
#   cd /u/scratch/j/jamesdc1/harxhar-clean
#   for arm in be_tunedWide_moments be_tunedWide_liquidity \
#       be_tunedWide_market_ew be_tunedWide_market_vw be_tunedWide_sentiment \
#       be_tunedWide_implied_vol be_tunedWide_vol_demand \
#       be_tunedWide_all_features; do
#     qsub -N "unif_$arm" -v ARM="$arm" -tc 60 jobs/sge/unification_serial.sge
#   done
# This script is the CARC-shaped equivalent, kept for parity with the rest of
# jobs/submit/. Run ON the CARC login node from /scratch1/jc_905/harxhar-clean.
set -e
cd /scratch1/jc_905/harxhar-clean
for arm in be_tunedWide_moments be_tunedWide_liquidity be_tunedWide_market_ew \
    be_tunedWide_market_vw be_tunedWide_sentiment be_tunedWide_implied_vol \
    be_tunedWide_vol_demand be_tunedWide_all_features; do
  sbatch --export=ALL,ARM="$arm" --job-name="unif_$arm" jobs/slurm/unification.sbatch
done
squeue -u "$USER" -h | wc -l
