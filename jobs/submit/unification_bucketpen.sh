#!/bin/bash
# Submit blk_bucketpen_tuned on CARC (penalty-allocation ladder, 2026-08-07):
# the champion structure with the single exogenous penalty replaced by one
# penalty per canonical family, selected by deterministic cyclic coordinate
# descent (3 passes, fixed order, 90 tail evaluations per retune vs 59,049
# for the cartesian product). Single arm x 100 chunks; no warmup (cache hot).
# Run ON the CARC login node from /scratch1/jc_905/harxhar-clean.
set -e
cd /scratch1/jc_905/harxhar-clean
sbatch --export=ALL,ARM=blk_bucketpen_tuned --job-name=unif_blk_bucketpen_tuned \
    jobs/slurm/unification.sbatch
squeue -u "$USER" -h | wc -l
