#!/bin/bash
# Submit blk4_trailGShaped on CARC (spectral penalty-allocation rung,
# 2026-08-07): trailing factor LEVELS at K=40 with a RANK-SHAPED transmission
# penalty lambda_i = lambda0 * i**gamma, (lambda0, gamma) causally selected as
# one 12-point block grid by the cyclic tuner. Tests whether the K=40 collapse
# is a penalty-allocation artifact rather than a signal boundary; gamma=0
# nests the flat penalty exactly. Single arm x 100 chunks; no warmup.
# Run ON the CARC login node from /scratch1/jc_905/harxhar-clean.
set -e
cd /scratch1/jc_905/harxhar-clean
sbatch --export=ALL,ARM=blk4_trailGShaped --job-name=unif_blk4_trailGShaped \
    jobs/slurm/unification.sbatch
squeue -u "$USER" -h | wc -l
