#!/bin/bash
# Submit ALL Hoffman2 arms of the unification campaign at once — no waves.
# Run ON the Hoffman2 login node from /u/scratch/j/jamesdc1/harxhar-clean.
# Arm list = the OLS family (benchmark + bucket table + negative control).
# NOTE: bucket arm names below are placeholders — replaced by the canonical
# names from the executor's ARMS registry before submission.
set -e
cd /u/scratch/j/jamesdc1/harxhar-clean
ARMS_H2=(
  a0_ols_har
  __BUCKET_ARMS__      # 8 canonical bucket arms, filled from ARMS registry
  a10_noexog           # dropped if the alias check collapses it into a0
)
for arm in "${ARMS_H2[@]}"; do
  qsub -N "unif_$arm" -v ARM="$arm" jobs/sge/unification.sge
done
qstat -u "$USER" | tail -n +3 | wc -l
