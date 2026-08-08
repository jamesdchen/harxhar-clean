#!/bin/bash
# Submit blk4_trailGZoo on CARC (2026-08-07): the SHAPE ZOO. Identical to
# blk4_trailGShaped (four blocks, trans_trailG40 = 40 trailing-standardized
# frozen-frame factor levels) except that the transmission block's grid is the
# UNION of three penalty families, each crossed with lambda0 in (1e2,1e3,1e4):
#   power        lambda_i = lambda0 * i**gamma,          gamma in 0,.5,1,2,3,4
#   exponential  lambda_i = lambda0 * exp(kappa*(i-1)),  kappa in .02,.05,.10
#   step         lambda0 up to rank K0, lambda0*1e4 after, K0 in 10,20,30
# 12 shape points x 3 levels = 36 points on ONE block, selected per retune by
# the same cyclic causal tuner. power/gamma=0 still nests the flat penalty
# exactly, so the arm remains honestly comparable to the flat-penalty rung.
# NO eigenvalue-based profile is run, deliberately: the frame's spectrum is a
# clean power law (d_i ~ c*i^-1.176, log-log R^2 = 0.981 over the top 40,
# d_1/d_40 = 83.6), so lambda_i proportional to d_i^-theta IS the rank power
# law under gamma = 1.176*theta — a reparameterization, not an experiment.
# What a power-law spectrum does not already contain is a geometric tail
# (exponential) or a hard cutoff on this basis (step); those are the two real
# alternatives and they are what this arm offers.
# THE EXHIBIT is which FAMILY the tuner selects, per retune and by year
# (penalty_shape_summary.csv: shape_family, frac_of_year). A dominant family is
# this panel's revealed prior; a family that flips by era is a statement about
# regime-dependent effective dimension.
# Single arm x 100 chunks; no warmup (cache hot). Run ON the CARC login node
# from /scratch1/jc_905/harxhar-clean.
set -e
cd /scratch1/jc_905/harxhar-clean
sbatch --export=ALL,ARM=blk4_trailGZoo --job-name=unif_blk4_trailGZoo \
    jobs/slurm/unification.sbatch
squeue -u "$USER" -h | wc -l
