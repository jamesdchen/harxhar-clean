#!/bin/bash
# Pooled twins on the moments_vix bucket (2026-10-06, user: every model also on the return moments +
# the VIX level alone): canary -> gate -> fleet -> collect, on Hoffman2.  Run from the deployment root:
#
#   cd /u/scratch/j/jamesdc1/harxhar-pool-momvix && bash cluster/submit_pool_momvix.sh
#
# canary   one slice (solves 199,250 .. 204,316, a tune boundary of the whole-series run) of six STORED
#          pooled arms -- moments and baseline x ridge / lasso / elastic net -- each pinned to the CPU
#          architecture its stored arm ran on (ridge and elastic net: intel-6736p; the lassofix lasso
#          arms: intel-gold-6140), so the comparison can be bit for bit
# gate     held on the canary: cluster/pool_momvix_check.py gate -> results/pool_momvix/GATE_OK
# fleet    held on the gate, and every task refuses to run without GATE_OK: the three whole-series
#          moments_vix arms, each on its estimator's architecture
# collect  held on the fleet: cluster/pool_momvix_check.py collect
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -z "${SGE_ROOT:-}" ]; then set +u; source /u/systems/UGE8.6.4/hoffman2/common/settings.sh; set -u; fi
mkdir -p logs results/pool_momvix
STAMP=logs/submitted_pool_momvix.txt
[ -f "$STAMP" ] && { echo "already submitted ($(cat "$STAMP")); delete $STAMP by hand to resubmit"; exit 1; }
FLAG=results/pool_momvix/GATE_OK
rm -f "$FLAG"

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

CAN=""
for A in intel-6736p intel-gold-6140; do
  F=cluster/pool_momvix_tasks_canary_$A.txt
  J=$(submit -N pmv_canary -t 1-"$(wc -l < "$F")" -l "h_rt=2:00:00,h_data=16G,arch=$A" \
        -v TASKFILE="$F",ARCH_REQ="$A" cluster/pool_momvix_task.sh)
  CAN="${CAN:+$CAN,}$J"
done
G=$(submit -N pmv_gate -hold_jid "$CAN" cluster/pool_momvix_gate.sh)
FLEET=""
for A in intel-6736p intel-gold-6140; do
  F=cluster/pool_momvix_tasks_fleet_$A.txt
  J=$(submit -N pmv_fleet -hold_jid "$G" -t 1-"$(wc -l < "$F")" -l "h_rt=6:00:00,h_data=16G,arch=$A" \
        -v TASKFILE="$F",ARCH_REQ="$A",REQUIRE_FLAG="$FLAG" cluster/pool_momvix_task.sh)
  FLEET="${FLEET:+$FLEET,}$J"
done
K=$(submit -N pmv_collect -hold_jid "$FLEET" cluster/pool_momvix_collect.sh)

echo "canary=$CAN gate=$G fleet=$FLEET collect=$K  $(date)" | tee "$STAMP"
qstat -u "$USER" | tail -10
