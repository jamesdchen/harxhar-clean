#!/bin/bash
# Pooled linear models pre-tuned, backbone 1/100 (2026-10-06): canary -> gate -> fleet on Hoffman2.
# Runner: experiments/pool_linear_pretune.py.  From the deployment root:
#
#   cd /u/scratch/j/jamesdc1/harxhar-pool-pretune && bash cluster/submit_pool_pretune.sh [stage]
#
# stage 1 (default)
#   canary  arm a (the spec, through the runner's penalty-factor class at r = 1) on one slice of nine
#           stored pooled arms (all_features / baseline / lean_vol_vix x ridge / lasso / elastic net:
#           solves 199,250 .. 204,316), each pinned to the CPU architecture its stored arm ran on
#           (ridge and elastic net intel-6736p, lasso intel-gold-6140); smoke: r = 0.01 walks on that
#           slice (lean_vol_vix lasso, elastic net) and one pre-tune
#   gate    held on the canary: cluster/pool_pretune_check.py gate -> results/pool_pretune/GATE_OK
#   fleet   held on the gate, every task refuses to run without GATE_OK: arm b (r = 0.01, causal) on
#           all_features / lean_vol_vix over blocks 600 .. 817 (the re-choice boundary before
#           2020-01-03 to the end, tune-aligned chunks), the causal r = 1 walk of baseline over the same
#           blocks (arm a there, and the record arm f's light retunes are decided from), and the pre-tunes
# stage rerunN  canary lines again (cluster/pool_pretune_tasks_rerunN_<arch>.txt, h_data HDATA, default 32G: the
#          all_features data prep peaks near 10.5 GB private and was killed under h_vmem 16G), then the gate again
# stage cacheN  the all_features design cache per CPU type, the gatec slices through it, the gate again (CACHE_OK)
# stage X  (any other name) the task files cluster/pool_pretune_tasks_X_<arch>.txt, h_data HDATA (default 16G),
#          REQUIRE_FLAG REQ (default GATE_OK; REQ=results/pool_pretune/CACHE_OK for walks that read the cache).
# As run on 2026-10-06 (the stage-1 fleet found no GATE_OK and refused; its work was resubmitted as):
#   small      arm b on lean_vol_vix, arm a on baseline (the record of baseline's arm f), their pre-tunes
#   afulllean  arm a on lean_vol_vix over blocks 600 .. 817 (full-period gate, penalty path)
#   afc6736 / afcgold  through the all_features cache: arm b (tag bc), arm a (tag ac), the pre-tunes (prec)
#   f / f2     arm f walks (fixed schedules cluster/pool_pretune_sched/h2_*.json, the local plan step); all_features
#              through the cache (tag fc)
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -z "${SGE_ROOT:-}" ]; then set +u; source /u/systems/UGE8.6.4/hoffman2/common/settings.sh; set -u; fi
STAGE=${1:-1}
mkdir -p logs results/pool_pretune
STAMP=logs/submitted_pool_pretune_$STAGE.txt
[ -f "$STAMP" ] && { echo "already submitted ($(cat "$STAMP")); delete $STAMP by hand to resubmit"; exit 1; }
FLAG=results/pool_pretune/GATE_OK

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

if [ "$STAGE" = 1 ]; then
  rm -f "$FLAG"
  CAN=""
  for A in intel-6736p intel-gold-6140; do
    F=cluster/pool_pretune_tasks_canary_$A.txt
    J=$(submit -N plp_canary -t 1-"$(wc -l < "$F")" -l "h_rt=4:00:00,h_data=16G,arch=$A" \
          -v TASKFILE="$F",ARCH_REQ="$A" cluster/pool_pretune_task.sh)
    CAN="${CAN:+$CAN,}$J"
  done
  G=$(submit -N plp_gate -hold_jid "$CAN" cluster/pool_pretune_gate.sh)
  FLEET=""
  for A in intel-6736p intel-gold-6140; do
    F=cluster/pool_pretune_tasks_fleet_$A.txt
    J=$(submit -N plp_fleet -hold_jid "$G" -t 1-"$(wc -l < "$F")" -l "h_rt=12:00:00,h_data=16G,arch=$A" \
          -v TASKFILE="$F",ARCH_REQ="$A",REQUIRE_FLAG="$FLAG" cluster/pool_pretune_task.sh)
    FLEET="${FLEET:+$FLEET,}$J"
  done
  echo "canary=$CAN gate=$G fleet=$FLEET  $(date)" | tee "$STAMP"
elif [ "${STAGE#cache}" != "$STAGE" ]; then  # cache, cache2, ...: the same task files
  # the all_features design cache per CPU type (cluster/pool_pretune_tasks_design_<arch>.txt, 32G), then
  # arm a through the cache on the gate slice (cluster/pool_pretune_tasks_cgate_<arch>.txt, tag gatec),
  # then the gate again (it compares the gatec slices with the stored arms as well; CACHE_OK on a pass)
  CJ=""
  for A in intel-6736p intel-gold-6140; do
    F=cluster/pool_pretune_tasks_design_$A.txt
    J=$(submit -N plp_design -t 1-"$(wc -l < "$F")" -l "h_rt=4:00:00,h_data=32G,arch=$A" \
          -v TASKFILE="$F",ARCH_REQ="$A" cluster/pool_pretune_task.sh)
    F2=cluster/pool_pretune_tasks_cgate_$A.txt
    J2=$(submit -N plp_cgate -hold_jid "$J" -t 1-"$(wc -l < "$F2")" -l "h_rt=4:00:00,h_data=16G,arch=$A" \
          -v TASKFILE="$F2",ARCH_REQ="$A" cluster/pool_pretune_task.sh)
    CJ="${CJ:+$CJ,}$J,$J2"
  done
  G=$(submit -N plp_regate -hold_jid "$CJ" cluster/pool_pretune_gate.sh)
  echo "stage cache=$CJ gate=$G  $(date)" | tee "$STAMP"
elif [ "${STAGE#rerun}" != "$STAGE" ]; then
  # rerun<N>: canary lines killed by the scheduler, again with 32G (cluster/pool_pretune_tasks_<stage>_<arch>.txt
  # repeat lines of the canary task files, so they write the same task directories), then the gate again
  CAN=""
  for A in intel-6736p intel-gold-6140; do
    F=cluster/pool_pretune_tasks_${STAGE}_$A.txt
    [ -s "$F" ] || continue
    J=$(submit -N "plp_$STAGE" -t 1-"$(wc -l < "$F")" -l "h_rt=4:00:00,h_data=${HDATA:-32G},arch=$A" \
          -v TASKFILE="$F",ARCH_REQ="$A" cluster/pool_pretune_task.sh)
    CAN="${CAN:+$CAN,}$J"
  done
  [ -n "$CAN" ] || { echo "no task files cluster/pool_pretune_tasks_${STAGE}_<arch>.txt"; exit 1; }
  G=$(submit -N plp_regate -hold_jid "$CAN" cluster/pool_pretune_gate.sh)
  echo "stage $STAGE=$CAN gate=$G  $(date)" | tee "$STAMP"
else  # any other stage name X: the task files cluster/pool_pretune_tasks_X_<arch>.txt, gated by GATE_OK
  REQ=${REQ:-$FLAG}  # REQ=results/pool_pretune/CACHE_OK for the walks that read the design cache
  [ -f "$REQ" ] || { echo "$REQ missing: the gate has not passed"; exit 1; }
  FJ=""
  for A in intel-6736p intel-gold-6140; do
    F=cluster/pool_pretune_tasks_${STAGE}_$A.txt
    [ -s "$F" ] || continue
    J=$(submit -N "plp_$STAGE" -t 1-"$(wc -l < "$F")" -l "h_rt=12:00:00,h_data=${HDATA:-16G},arch=$A" \
          -v TASKFILE="$F",ARCH_REQ="$A",REQUIRE_FLAG="$REQ" cluster/pool_pretune_task.sh)
    FJ="${FJ:+$FJ,}$J"
  done
  [ -n "$FJ" ] || { echo "no task files cluster/pool_pretune_tasks_${STAGE}_<arch>.txt"; exit 1; }
  echo "stage $STAGE=$FJ  $(date)" | tee "$STAMP"
fi
qstat -u "$USER" | grep plp | tail -12 || true
