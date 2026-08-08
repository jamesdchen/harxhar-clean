#!/bin/bash
# Submit the FIXED-PENALTY ENVELOPE arms (2026-08-07): 4 lasso jiggle rungs +
# 3 ridge grid-extension rungs, 7 arms x 100 chunks.
#
# WHY. b2_lasso at a HAND-PICKED alpha=1e-4 scores 0.22950 (DM -8.39) and beats
# every other single-estimator arm — tuned ridge 0.23040 (-4.45), tuned enet
# 0.23056 (-2.58), tuned lasso 0.23134 (-2.03). Because 1e-4 was chosen with
# hindsight, that number is currently UNINTERPRETABLE. Two explanations with
# opposite implications for the paper have to be separated:
#   (a) 1e-4 is an oracle/lucky point -> the headline is an artifact;
#   (b) the lasso family genuinely wins on this design and the CAUSAL TUNER is
#       what costs performance (independent evidence: the tuner's selected
#       l1_ratio is bimodal, flipping between 0.25 and 1.0 across ~1200
#       retunes — what happens when a 125-bar validation tail cannot identify
#       the parameter).
# The readout is QLIKE vs log(alpha): a SHARP peak at 1e-4 with much worse
# neighbours says luck; a BROAD flat optimum spanning decades says the family
# wins and the tuner underperforms. Both are publishable; the point is to know.
#
# Separately, the fixed-RIDGE jiggle grid is monotone across its whole range
# (0.1: 0.23360, 0.3: 0.23312, 1: 0.23243, 3: 0.23169, 10: 0.23087) — still
# improving at its top endpoint, so it never located its own optimum and the
# fixed-ridge envelope is unmeasured. alpha in {30, 100, 300} brackets it.
#
# b2_lasso (alpha=1e-4) and b1_ridge (alpha=1.0) are NOT rebuilt: they already
# ARE rungs of these envelopes and their chunks must stay byte-reproducible.
#
# These are FULL-LENGTH arms (no 24000-bar frame block), so all 100 chunks are
# legal — no legal-missing set.
#
# CLUSTER NOTE: CARC is saturated by the tree bank, so these are intended for
# HOFFMAN2 (SGE). The generic H2 entry point already takes ARM:
#   cd /u/scratch/j/jamesdc1/harxhar-clean
#   for arm in <the 7 names below>; do
#     qsub -N "unif_$arm" -v ARM="$arm" -tc 40 jobs/sge/unification.sge
#   done
# This script is the CARC-shaped equivalent, kept for parity with the rest of
# jobs/submit/. Run ON the CARC login node from /scratch1/jc_905/harxhar-clean.
set -e
cd /scratch1/jc_905/harxhar-clean
for arm in b2_lasso_a1em6 b2_lasso_a1em5 b2_lasso_a1em3 b2_lasso_a1em2 \
    b1_ridge_a30 b1_ridge_a100 b1_ridge_a300; do
  sbatch --export=ALL,ARM="$arm" --job-name="unif_$arm" jobs/slurm/unification.sbatch
done
squeue -u "$USER" -h | wc -l
