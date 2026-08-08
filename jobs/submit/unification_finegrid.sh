#!/bin/bash
# Submit the GRID-RESOLUTION arms (2026-08-07): 2 arms x 100 chunks.
#
# THE ARITHMETIC. The fixed-ridge envelope moves ~0.0009 in QLIKE per DECADE of
# alpha (0.1: 0.23360, 0.3: 0.23312, 1: 0.23243, 3: 0.23169, 10: 0.23087), so a
# HALF-decade error in the selected penalty costs ~0.0004 — larger than
# increments this paper reports as significant (the transmission increment is
# 0.00028 at DM -2.06). The tuned-ridge grid is logspace(-2,3,6), i.e. decade
# spacing; the block grids are worse at 3 points each, and the paper's BEST
# MODEL is tuned on those. We are resolving the hyperparameter more coarsely
# than the effects we measure.
#
# COUNTER-CONSIDERATION, which is why this is an empirical trade and not an
# obvious win: a finer grid is also more opportunities to overfit the 125-bar
# validation tail, and we have direct evidence the tail is a noisy selector —
# the enet arms' chosen l1_ratio is BIMODAL, flipping between 0.25 and 1.0
# across ~1200 retunes. These arms are allowed to LOSE.
#
# ARMS (each identical to its parent except grid RESOLUTION; every coarse grid
# is a strict, bit-exact subset of its refinement, so the increment is
# attributable to the interstitial points alone):
#   b1_ridge_tuned_fine     wide all_features basis, logspace(-2,3,11)
#   blk4_trailGShaped_fine  best model, every block grid half-decade
#                           (backbone/exog/product 5 points each; shaped
#                           transmission lambda0 5 points x 4 gammas = 20).
#                           Cyclic cost 94 tail evals/retune vs the shipped
#                           arm's 52 — under 2x, inside the ~150 ceiling.
#
# THE DIAGNOSTIC that makes either outcome publishable: each chunk persists
# meta.coarse_grids, and the scorer emits results/fine_grid_usage.csv — the
# fraction of retunes selecting an INTERSTITIAL point. Rarely selected means
# resolution was never binding and the coarse grid is vindicated; selected
# constantly with no QLIKE gain means selection noise, i.e. evidence FOR the
# coarse grid as implicit regularization.
#
# CLUSTER NOTE: CARC is saturated by the tree bank, so these are intended for
# HOFFMAN2. Both are single-threaded numpy, so use the SERIAL entry point (no
# parallel environment — the 4-slot PE request was making them queue):
#   cd /u/scratch/j/jamesdc1/harxhar-clean
#   for arm in b1_ridge_tuned_fine blk4_trailGShaped_fine; do
#     qsub -N "unif_$arm" -v ARM="$arm" -tc 50 jobs/sge/unification_serial.sge
#   done
# This script is the CARC-shaped equivalent, kept for parity with the rest of
# jobs/submit/. Run ON the CARC login node from /scratch1/jc_905/harxhar-clean.
set -e
cd /scratch1/jc_905/harxhar-clean
for arm in b1_ridge_tuned_fine blk4_trailGShaped_fine; do
  sbatch --export=ALL,ARM="$arm" --job-name="unif_$arm" jobs/slurm/unification.sbatch
done
squeue -u "$USER" -h | wc -l
