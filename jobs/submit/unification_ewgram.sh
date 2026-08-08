#!/bin/bash
# Submit the DISCOUNTED-GRAM family (2026-08-08): 3 arms x 100 chunks.
#
# MOTIVATION, from the JS verdict. blk3_js_tuned scored 0.24631 (DM +26.6 vs
# blk3_tuned) with its estimator VERIFIED firing correctly — mean shrinkage
# factor 0.77, stable, zero singular bars. So risk-optimal IN-SAMPLE shrinkage
# (23%) is orders of magnitude below forecast-optimal shrinkage. The reading:
# beta DRIFTS, and in-sample precision is a bad guide to forecast risk when the
# coefficients are not constant over the window. Heavy ridge on a flat
# 24000-bar window is a crude patch — it distrusts ALL evidence equally when
# the real problem is that OLD evidence is STALE. Window length is the one
# hyperparameter this campaign has never varied.
#
# THE OBJECT: exponentially weighted sufficient statistics
#     G_t = sum_s w^(t-s) x_s x_s',   c_t = sum_s w^(t-s) x_s y_s
# with w = 2^(-1/H) for half-life H in bars.
#
# TRUNCATED, NOT INFINITE-MEMORY — a deliberate deviation from the brief:
#  (1) NESTING. An infinite-memory recursion at w=1 is a CUMULATIVE sum over
#      all history, not the 24000-bar window, so H=infinity would NOT nest
#      blk3_tuned. Summing over the same trailing W bars does, term for term.
#  (2) COMPARABILITY. Holding the span at W=24000 means these arms vary the
#      WEIGHTING and nothing else — same rows, same legality, same first legal
#      OOS row as every other frozen-construction arm.
# Cost: the recurrence keeps a subtraction,
#     G_{t+1} = w G_t + x_{t+1} x_{t+1}' - w^W x_{t-W+1} x_{t-W+1}'
# which at w=1 IS the existing sliding-window update. So it is the same price
# as the incumbent, not cheaper; the brief's "no subtraction" saving belongs
# only to the non-nesting infinite-memory variant. Verified: H=inf reproduces
# RollingLeastSquares' centered gram to 1.1e-13 and its rhs to 1.8e-15.
#
# NO POWERS ARE ACCUMULATED: the recurrence runs per bar and the single
# boundary weight w^W is computed in closed form. Measured at W=24000 —
# H=1000: 5.96e-08, H=3000: 3.91e-03, H=6000: 6.25e-02, H=12000: 2.50e-01,
# H=24000: 5.00e-01. No underflow anywhere on the shipped grid.
#
# INITIALIZATION: the gram is built EXACTLY from the full trailing 24000-bar
# window at the first bar and rebuilt exactly at every retune boundary (every
# TUNE_PER=250 bars, which is also when H may change). There is therefore no
# burn-in period during which the statistics are under-filled — the window is
# always complete, and the effective sample is whatever the weighting implies.
#
# EFFECTIVE SAMPLE SIZE (Kish), used for the dof of any sigma^2 on this path
# and persisted per boundary as a diagnostic:
#     n_eff = (sum_k w^k)^2 / sum_k w^(2k),  k = 0 .. W-1
# The FINITE sum is used because the window is finite. At W=24000:
#     H=1000: 2885   H=3000: 8589   H=12000: 20775   H=24000: 23083
#     H=inf : 24000 (exactly the flat window's own count)
# NOTE the infinite-horizon form (1+w)/(1-w) quoted in the brief gives 69249 at
# H=24000 — three times the truncated value — so quoting it on a truncated
# window would overstate the evidence by that factor.
#
# ARMS (all blk3 design: backbone + exog + product, per-block penalty grids
# exactly as blk3_tuned, window 24000, oos_mult=2, legality range(9)):
#   blk3_ew_tuned        half-life selected causally from
#                        {1000,3000,6000,12000,24000,inf} JOINTLY with the
#                        per-block penalties (cartesian, 6 x 27 = 162 tail
#                        evaluations per retune). THE EXHIBIT is the selected-H
#                        trajectory: a finite H chosen WITH lighter penalties
#                        than the flat arm selects is direct evidence that the
#                        campaign has been paying for drift with shrinkage.
#   blk3_ewFixed_H3000   fixed half-life, penalties tuned as usual.
#   blk3_ewFixed_H12000  ditto. These two bound the selection-noise question:
#                        the H grid adds a dial the 125-bar validation tail
#                        must resolve, and this campaign has repeatedly found
#                        that tail to be a noisy selector. They show what a
#                        half-life is worth WITHOUT paying for its selection.
#
# ONE-CHUNK CANARY FIRST (standing rule). Submit a single mid-history chunk and
# read its meta before the fleet. A healthy first chunk shows:
#   meta.tuned_alphas   NON-EMPTY, ~11 entries for a 2763-bar chunk (one per
#                       TUNE_PER=250 boundary), each carrying
#                       half_life / decay / n_eff / alphas / n_tail_evals.
#   half_life           one of {1000,3000,6000,12000,24000,inf}; JSON writes
#                       infinity as null, so an `inf` selection appears as a
#                       null half_life with decay 1.0 — that is the flat-window
#                       selection, not a bug.
#   decay               2^(-1/H), i.e. 0.99930 at H=1000 ... 1.0 at H=inf.
#   n_eff               matches the table above for the selected H; anything
#                       outside [2885, 24000] means the weighting is wrong.
#   n_tail_evals        162 for blk3_ew_tuned, 27 for the fixed-H arms.
#   meta.tuned_grids    carries the block grids PLUS a "half_life" entry.
#
# HOFFMAN2 (intended target; h_data 16G):
#   cd /u/scratch/j/jamesdc1/harxhar-clean
#   for arm in blk3_ew_tuned blk3_ewFixed_H3000 blk3_ewFixed_H12000; do
#     qsub -N "unif_$arm" -v ARM="$arm" -tc 60 jobs/sge/unification_serial.sge
#   done
# This script is the CARC-shaped equivalent, kept for parity with the rest of
# jobs/submit/. Run ON the CARC login node from /scratch1/jc_905/harxhar-clean.
set -e
cd /scratch1/jc_905/harxhar-clean
for arm in blk3_ew_tuned blk3_ewFixed_H3000 blk3_ewFixed_H12000; do
  sbatch --export=ALL,ARM="$arm" --job-name="unif_$arm" jobs/slurm/unification.sbatch
done
squeue -u "$USER" -h | wc -l
