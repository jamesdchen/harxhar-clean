#!/bin/bash
# Submit the transmission dig on CARC (author directive 2026-08-07): 7
# variants of the trailing-standardized transmission block — symmetric-part
# control, undecomposed C, causally refreshed frame, K=5/10/40 frame widths,
# lag-2 operator. Fixed user penalties, same window/legality as blk4_trail.
# See src/unification.py (_transmission_block dig knobs). No warmup — the
# panel cache is hot. Run ON the CARC login node from
# /scratch1/jc_905/harxhar-clean.
set -e
cd /scratch1/jc_905/harxhar-clean
for arm in blk4_trailSym blk4_trailFullC blk4_trailRefresh \
           blk4_trailKFive blk4_trailKTen blk4_trailKForty blk4_trailLagTwo; do
  sbatch --export=ALL,ARM="$arm" --job-name="unif_$arm" jobs/slurm/unification.sbatch
done
squeue -u "$USER" -h | wc -l
