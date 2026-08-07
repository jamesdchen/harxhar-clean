#!/bin/bash
# Submit blk4_trailDropHet on CARC (era-dig surgical test, 2026-08-07): the
# blk4_trail construction with the cadence-heterogeneous sumret3_{ew,vw}stock
# families excluded from the TRANSMISSION BASE ONLY (exog ridge block keeps
# them). Separate script so re-running never resubmits the original dig 7.
# Hypothesis: pre-2015 flow harm disappears, post-2014 gain survives.
# Run ON the CARC login node from /scratch1/jc_905/harxhar-clean.
set -e
cd /scratch1/jc_905/harxhar-clean
sbatch --export=ALL,ARM=blk4_trailDropHet --job-name=unif_blk4_trailDropHet \
    jobs/slurm/unification.sbatch
squeue -u "$USER" -h | wc -l
