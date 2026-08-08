#!/bin/bash
# Submit the PC-ladder K-sweep on CARC (2026-08-07): blk_pcladder_tuned's
# construction at K=40, K=80 and at the FULL live rank of the frozen frame.
# WHY: {ma_j(G_i)} spans the eigen-projection of the ladder-expanded base
# design, because moving averages commute with a linear projection. At K=20
# that projection DISCARDS most of the space, so the arm measures compression.
# At full rank nothing is discarded — the two designs differ by a block-
# diagonal orthogonal rotation, a flat-penalty ridge on either is identical,
# and the only thing that can move the score is the penalty PROFILE. So the
# sweep converts the arm from a compression experiment into a
# REPARAMETERIZATION experiment: gamma=0 at full rank should reproduce the flat
# result, and anything better is pure gain from the tilt.
# K for the full-rank arm is READ from the frame's own liveness rule
# (src/unification.py::_frame_live_rank) — never hardcoded — and K=40/K=80 fail
# LOUDLY in _frame_of if they exceed the available spectrum.
# Exponent grid is the WIDE one (gamma up to 4): at full rank the tail
# directions are precisely what the tilt has to suppress.
# 3 arms x 100 chunks; no warmup (cache hot). Run ON the CARC login node from
# /scratch1/jc_905/harxhar-clean.
set -e
cd /scratch1/jc_905/harxhar-clean
for arm in blk_pcladder_fortyK_tuned blk_pcladder_eightyK_tuned \
    blk_pcladder_fullK_tuned; do
  sbatch --export=ALL,ARM="$arm" --job-name="unif_$arm" jobs/slurm/unification.sbatch
done
squeue -u "$USER" -h | wc -l
