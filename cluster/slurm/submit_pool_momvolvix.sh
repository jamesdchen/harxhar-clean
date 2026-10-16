#!/bin/bash
# Pooled twins on the moments_vol_vix bucket (2026-10-06, user: moments_vix plus ES volume) on CARC,
# exactly as the moments_vix fleet of cluster/slurm/submit_pool_momvix.sh: epyc-7513, each arm as 20
# tune-aligned chunks of 10,250 solves on 20 CPUs, merged.  The gate of that campaign stands
# (results/pool_momvix/GATE_OK, 3/3 stored slices bit for bit): the code shipped since differs only by
# the new SUBGROUPS line in src/data/loading.py (checked below), and every fleet task still refuses to
# run without GATE_OK.  From the deployment root:
#
#   cd /scratch1/jc_905/harxhar-pool-momvix && bash cluster/slurm/submit_pool_momvolvix.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
STAMP=logs/submitted_pool_momvolvix_carc.txt
[ -f "$STAMP" ] && { echo "already submitted ($(cat "$STAMP")); delete $STAMP by hand to resubmit"; exit 1; }
FLAG=results/pool_momvix/GATE_OK
[ -f "$FLAG" ] || { echo "$FLAG missing: run the moments_vix gate first"; exit 1; }
set +u; module load conda >/dev/null 2>&1; source activate harxhar; set -u
PYTHONPATH=$PWD python -c "
from src.data.loading import SUBGROUPS as S
assert S['moments_vol_vix'] == [*S['moments'], 'sumvolume', 'vix'] and len(S['moments_vol_vix']) == 9
assert S['moments_vix'] == [*S['moments'], 'vix']
print('moments_vol_vix:', S['moments_vol_vix'])"
S="sbatch --parsable --constraint=epyc-7513"
F=$($S -J pmvv_fleet --array=1-3 --cpus-per-task=20 --mem=120G --time=3:00:00 \
      --export=ALL,MODE=fleet,TASKFILE=cluster/pool_momvolvix_tasks_fleet_carc.txt,REQUIRE_FLAG=$FLAG \
      cluster/slurm/pool_momvix_carc.sbatch)
K=$($S -J pmvv_collect --dependency=afterany:"$F" --export=ALL,MODE=check,STEP=collect,CBUCKET=moments_vol_vix \
      cluster/slurm/pool_momvix_carc.sbatch)
echo "fleet=$F collect=$K  $(date)" | tee "$STAMP"
squeue -u "$USER" -o "%.12i %.12j %.4t %.10M %.5C %R" | grep pmvv || true
