#!/bin/bash
# Pooled twins on the moments_vol_vix bucket (2026-10-06, user: moments_vix plus ES volume,
# SUBGROUPS["moments_vol_vix"] = the 7 ES return moments + sumvolume + vix) on Hoffman2, exactly as the
# moments_vix fleet of cluster/submit_pool_momvix.sh: whole series, ridge / elastic net on intel-6736p,
# lasso on intel-gold-6140.  The gate of that campaign stands (results/pool_momvix/GATE_OK, 6/6 stored
# slices bit for bit): the code shipped since differs only by the new SUBGROUPS line in
# src/data/loading.py (checked below: the bucket resolves to 9 columns, moments_vix is unchanged), and
# every fleet task still refuses to run without GATE_OK.  From the deployment root:
#
#   cd /u/scratch/j/jamesdc1/harxhar-pool-momvix && bash cluster/submit_pool_momvolvix.sh
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -z "${SGE_ROOT:-}" ]; then set +u; source /u/systems/UGE8.6.4/hoffman2/common/settings.sh; set -u; fi
STAMP=logs/submitted_pool_momvolvix.txt
[ -f "$STAMP" ] && { echo "already submitted ($(cat "$STAMP")); delete $STAMP by hand to resubmit"; exit 1; }
FLAG=results/pool_momvix/GATE_OK
[ -f "$FLAG" ] || { echo "$FLAG missing: run the moments_vix gate first"; exit 1; }
set +u; source /u/local/apps/anaconda3/2023.03/etc/profile.d/conda.sh; conda activate hpc-pi; set -u
PYTHONPATH=$PWD python -c "
from src.data.loading import SUBGROUPS as S
assert S['moments_vol_vix'] == [*S['moments'], 'sumvolume', 'vix'] and len(S['moments_vol_vix']) == 9
assert S['moments_vix'] == [*S['moments'], 'vix']
print('moments_vol_vix:', S['moments_vol_vix'])"

submit () {  # the scheduler's submission verifier times out now and then: retry
  local out n=0
  until out=$(qsub -terse "$@" 2>&1); do
    n=$((n + 1))
    echo "submit attempt $n failed: $out" >&2
    [ "$n" -ge 8 ] && return 1
    sleep 30
  done
  echo "${out%%.*}"
}

FLEET=""
for A in intel-6736p intel-gold-6140; do
  F=cluster/pool_momvolvix_tasks_fleet_$A.txt
  J=$(submit -N pmvv_fleet -t 1-"$(wc -l < "$F")" -l "h_rt=6:00:00,h_data=16G,arch=$A" \
        -v TASKFILE="$F",ARCH_REQ="$A",REQUIRE_FLAG="$FLAG" cluster/pool_momvix_task.sh)
  FLEET="${FLEET:+$FLEET,}$J"
done
K=$(submit -N pmvv_collect -hold_jid "$FLEET" -v BUCKET=moments_vol_vix cluster/pool_momvix_collect.sh)
echo "fleet=$FLEET collect=$K  $(date)" | tee "$STAMP"
qstat -u "$USER" | grep pmvv || true
