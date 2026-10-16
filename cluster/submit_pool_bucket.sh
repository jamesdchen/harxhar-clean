#!/bin/bash
# Pooled twins on one more bucket on Hoffman2 (2026-10-06), exactly as the moments_vix fleet of
# cluster/submit_pool_momvix.sh: whole series, ridge / elastic net on intel-6736p, lasso on
# intel-gold-6140 (the CPU types of the stored baseline / all_features pooled arms).  The gate of that
# campaign stands (results/pool_momvix/GATE_OK, 6/6 stored slices bit for bit): the code shipped since
# differs only by new SUBGROUPS lines in src/data/loading.py (checked below against the definitions the
# user gave), and every fleet task still refuses to run without GATE_OK.  Task files:
# cluster/pool_<bucket>_tasks_fleet_<arch>.txt.  From the deployment root:
#
#   cd /u/scratch/j/jamesdc1/harxhar-pool-momvix && bash cluster/submit_pool_bucket.sh lean_vol_vix
set -euo pipefail
cd "$(dirname "$0")/.."
B=${1:?usage: submit_pool_bucket.sh <bucket>}
if [ -z "${SGE_ROOT:-}" ]; then set +u; source /u/systems/UGE8.6.4/hoffman2/common/settings.sh; set -u; fi
STAMP=logs/submitted_pool_$B.txt
[ -f "$STAMP" ] && { echo "already submitted ($(cat "$STAMP")); delete $STAMP by hand to resubmit"; exit 1; }
FLAG=results/pool_momvix/GATE_OK
[ -f "$FLAG" ] || { echo "$FLAG missing: run the moments_vix gate first"; exit 1; }
set +u; source /u/local/apps/anaconda3/2023.03/etc/profile.d/conda.sh; conda activate hpc-pi; set -u
PYTHONPATH=$PWD python - "$B" <<'EOF'
import sys
from src.data.loading import SUBGROUPS as S
m = S["moments"]
want = {
    "moments_vix": [*m, "vix"],
    "moments_vol_vix": [*m, "sumvolume", "vix"],
    "lean_vol_vix": ["sumret", "sumabsret", "sumret4", "sumautocov", "sumvolume", "vix"],
}
for k, v in want.items():
    assert S.get(k) == v, (k, S.get(k))
print(sys.argv[1], S[sys.argv[1]])
EOF

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
  F=cluster/pool_${B}_tasks_fleet_$A.txt
  J=$(submit -N "pb_${B:0:6}" -t 1-"$(wc -l < "$F")" -l "h_rt=6:00:00,h_data=16G,arch=$A" \
        -v TASKFILE="$F",ARCH_REQ="$A",REQUIRE_FLAG="$FLAG" cluster/pool_momvix_task.sh)
  FLEET="${FLEET:+$FLEET,}$J"
done
K=$(submit -N "pbc_${B:0:6}" -hold_jid "$FLEET" -v BUCKET="$B" cluster/pool_momvix_collect.sh)
echo "$B fleet=$FLEET collect=$K  $(date)" | tee "$STAMP"
qstat -u "$USER" | grep "pb" || true
