#!/bin/bash
# Submit blk3_tikhonov_tuned on CARC (generalized Tikhonov, 2026-08-07): three
# blocks, NO transmission columns, exogenous penalty spectrum-tilted as
# Gamma = V diag(alpha * i**gamma) V' in the frozen eigenframe — the
# principled form of what the transmission block achieves by duplication.
# gamma=0 nests plain scalar ridge (blk3_tuned). If gamma>0 reproduces or
# beats blk4_trailG_tuned, the four-block story collapses into "ridge with a
# spectrum-tilted penalty". Single arm x 100 chunks; no warmup (cache hot).
# Run ON the CARC login node from /scratch1/jc_905/harxhar-clean.
set -e
cd /scratch1/jc_905/harxhar-clean
sbatch --export=ALL,ARM=blk3_tikhonov_tuned --job-name=unif_blk3_tikhonov_tuned \
    jobs/slurm/unification.sbatch
squeue -u "$USER" -h | wc -l
