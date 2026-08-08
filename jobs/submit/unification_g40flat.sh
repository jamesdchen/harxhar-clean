#!/bin/bash
# Submit blk4_trailG40_tuned on CARC (2026-08-07): the DE-CONFOUNDING CONTROL.
# 1 arm x 100 chunks. This arm exists to fix a published comparison, not to win.
#
# THE PROBLEM. The paper reports blk4_trailGShaped beating blk4_trailG_tuned
# (-1.836e-4, DM -2.019) and attributes the gain to the RANK-SHAPED penalty.
# The persisted trajectories do not support that attribution, twice over:
#  (1) gamma=0 is BIT-EXACTLY the flat penalty (lambda0 * i**0 == lambda0) and
#      is selected in 45% of retunes. The whole aggregate edge sits in the
#      gamma<=0.5 subset — mean diff -4.697e-4, DM -3.604, better in 19 of 21
#      years — while where a tilt is actually applied (gamma>=2) the shaped arm
#      LOSES: +2.079e-4, DM +2.113. Difference-in-differences z = -4.15,
#      p = 6.7e-5. The edge is concentrated exactly where the "shaped" arm is
#      not shaped.
#  (2) The two arms also differ in FRAME WIDTH: blk4_trailGShaped uses
#      trans_trailG40 (K=40), blk4_trailG_tuned uses trans_trailG (K=20).
#      Every other block and grid is identical. The pair confounds TILT with K.
#
# THE CONTROL: K=40 trailing factor LEVELS with the ORDINARY FLAT `trans` grid
# (1e2, 1e3, 1e4) — no shape axis at all. It splits the confounded comparison:
#     vs blk4_trailG_tuned  -> pure K effect at flat penalty (K=40 vs K=20)
#     vs blk4_trailGShaped  -> pure SHAPE effect at matched K=40
# The second is the comparison that decides whether rank-shaping earns
# anything, and the campaign never ran it.
#
# Everything else is identical to both neighbours: kind blocks_tuned, cyclic
# selection, window 24000, TUNE_PER=250, 25-embargo/125-tail, oos_mult=2,
# legality range(9), same persistence contract.
#
# HOFFMAN2 (intended target; CARC is saturated by the tree bank):
#   cd /u/scratch/j/jamesdc1/harxhar-clean
#   qsub -N unif_blk4_trailG40_tuned -v ARM=blk4_trailG40_tuned \
#       -tc 60 jobs/sge/unification_serial.sge
# This script is the CARC-shaped equivalent, kept for parity with the rest of
# jobs/submit/. Run ON the CARC login node from /scratch1/jc_905/harxhar-clean.
set -e
cd /scratch1/jc_905/harxhar-clean
sbatch --export=ALL,ARM=blk4_trailG40_tuned \
    --job-name=unif_blk4_trailG40_tuned jobs/slurm/unification.sbatch
squeue -u "$USER" -h | wc -l
