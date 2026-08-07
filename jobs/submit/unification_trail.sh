#!/bin/bash
# Late-join transmission-revival arms for the running v2 campaign:
# trailing-standardized transmission construction. Depends on the v2 warmup
# if still queued; otherwise the cache exists. Run ON the CARC login node.
set -e
cd /scratch1/jc_905/harxhar-clean
WID=$(squeue -u "$USER" -h -n unif_warmup -o %i | head -1)
DEP=()
if [ -n "$WID" ]; then
  DEP=(--dependency=afterok:"$WID")
  echo "depending on warmup $WID"
fi
for arm in d3_transmission_alone_trail blk4_trail; do
  sbatch "${DEP[@]}" --export=ALL,ARM="$arm" \
    --job-name="unif_$arm" jobs/slurm/unification.sbatch
done
squeue -u "$USER" -h | wc -l
