#!/bin/bash
# CARC leg of the v3 campaign: a0 (own-root benchmark) + all heavy arms.
# The OLS bucket family runs on Hoffman2. Run ON the CARC login node.
# Caller must purge results/prep_cache_* first; warmup rebuilds it.
set -e
cd /scratch1/jc_905/harxhar-clean
WID=$(sbatch --parsable jobs/slurm/unif_warmup.sbatch)
echo "warmup job: $WID"
ARMS_CARC=(
  a0_ols_har
  b1_ridge b2_lasso
  b1_ridge_tuned b2_lasso_tuned b3_enet_tuned
  b1_ridge_a0p1 b1_ridge_a0p3 b1_ridge_a3 b1_ridge_a10
  blk2_user blk3_user blk4_user
  blk2_doc blk3_doc blk4_doc
  blk2_tuned blk3_tuned blk4_tuned
  blk4_trail
  c4_product_alone_user c4_product_alone_doc c4_product_alone_tuned
  d3_transmission_alone_user d3_transmission_alone_doc
  d3_transmission_alone_tuned d3_transmission_alone_trail
)
for arm in "${ARMS_CARC[@]}"; do
  sbatch --dependency=afterok:"$WID" --export=ALL,ARM="$arm" \
    --job-name="unif_$arm" jobs/slurm/unification.sbatch
done
squeue -u "$USER" -h | wc -l
