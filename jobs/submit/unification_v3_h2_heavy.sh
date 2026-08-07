#!/bin/bash
# Hoffman2 race leg: the lighter-heavy arms, run in parallel with CARC's
# copies — whichever cluster completes an arm first supplies the scores
# (the scorer picks the complete root per arm). Cache already exists on H2
# (herd winner built it); no warmup hold needed. Run under a login shell.
set -e
cd /u/scratch/j/jamesdc1/harxhar-clean
ARMS_H2_HEAVY=(
  b1_ridge b2_lasso
  b1_ridge_a0p1 b1_ridge_a0p3 b1_ridge_a3 b1_ridge_a10
  blk2_doc blk3_doc blk4_doc
  c4_product_alone_doc d3_transmission_alone_doc
)
for arm in "${ARMS_H2_HEAVY[@]}"; do
  qsub -N "unif_$arm" -v ARM="$arm" jobs/sge/unification.sge
done
qstat -u "$USER" 2>/dev/null | tail -n +3 | wc -l
