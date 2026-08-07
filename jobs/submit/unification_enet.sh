#!/bin/bash
# Late-join arm for the running v2 campaign: b3_enet_tuned. Depends on the
# v2 warmup if it is still in the queue; otherwise the cache exists and the
# array runs immediately. Run ON the CARC login node.
set -e
cd /scratch1/jc_905/harxhar-clean
WID=$(squeue -u "$USER" -h -n unif_warmup -o %i | head -1)
DEP=()
if [ -n "$WID" ]; then
  DEP=(--dependency=afterok:"$WID")
  echo "depending on warmup $WID"
fi
sbatch "${DEP[@]}" --export=ALL,ARM=b3_enet_tuned \
  --job-name=unif_b3_enet_tuned jobs/slurm/unification.sbatch
squeue -u "$USER" -h | wc -l
