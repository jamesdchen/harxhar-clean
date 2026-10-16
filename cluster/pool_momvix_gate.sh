#!/bin/bash
# Gate of the pooled moments_vix campaign (Hoffman2): the canary's slices of six stored pooled arms
# vs the stored arms (cluster/pool_momvix_check.py gate); writes results/pool_momvix/GATE_OK on a pass.
#$ -cwd
#$ -j y
#$ -o logs/
#$ -l h_rt=1:00:00,h_data=8G
set -eo pipefail  # -u only after conda: its activate scripts read unset variables
source /u/local/apps/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate hpc-pi
set -u
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH="$PWD"
echo "gate on $(hostname)  $(date)"
for f in results/pool_momvix/gate/*/none/*/tw2000; do
  echo "$f: $([ -f "$f/DONE" ] && echo DONE || echo NOT-DONE) $(cat "$f/SECONDS" 2>/dev/null) s $(head -1 "$f/host.txt" 2>/dev/null)"
done
python -u cluster/pool_momvix_check.py gate
