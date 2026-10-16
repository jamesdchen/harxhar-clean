#!/bin/bash
# Collect step of the pooled moments_vix campaign (Hoffman2): checks the three whole-series arms and
# writes their regular-hours files (cluster/pool_momvix_check.py collect).
#$ -cwd
#$ -j y
#$ -o logs/
#$ -l h_rt=1:00:00,h_data=8G
set -eo pipefail  # -u only after conda: its activate scripts read unset variables
source /u/local/apps/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate hpc-pi
set -u
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH="$PWD"
echo "collect on $(hostname)  $(date)"
for f in results/pool_momvix/fleet/${BUCKET:-moments_vix}/none/*/tw2000; do
  echo "$f: $([ -f "$f/DONE" ] && echo DONE || echo NOT-DONE) $(cat "$f/SECONDS" 2>/dev/null) s $(head -1 "$f/host.txt" 2>/dev/null)"
  grep -h "OOS rows" "$f/run.log" 2>/dev/null | cut -c1-200
done
python -u cluster/pool_momvix_check.py collect ${BUCKET:-}  # BUCKET: a fleet bucket other than moments_vix
