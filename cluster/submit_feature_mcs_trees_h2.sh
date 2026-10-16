#!/bin/bash
# Feature MCS trees, the 16 bucket designs: XGBoost and forest chunks on Hoffman2 AVX-only nodes (2026-10-13), from the
# root /u/scratch/j/jamesdc1/harxhar-mcstrees-h2 (cluster/ship_feature_mcs_trees_h2.sh), one single-slot SGE task per line
# of cluster/feature_mcs_trees_h2_v3_lines.txt (cluster/feature_mcs_trees_h2_task.sh), pinned to the arch ARCHES and the
# CPU check REQUIRE_CPU.  Chunks land in results/feature_mcs_trees_work/chunks2/<est>/<design>/c<chunk> (the CARC layout);
# cluster/pull_feature_mcs_trees_h2.sh brings them home for the collect step.
#   bash cluster/submit_feature_mcs_trees_h2.sh canary    the first NC lines (default 2), each writing an ok flag
#   bash cluster/submit_feature_mcs_trees_h2.sh fleet     the other lines, -hold_jid the canary, refusing to run unless
#                                                         the NC flags exist
# Submit from a shell with no conda env active.  A stamp refuses a second identical submission.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -z "${SGE_ROOT:-}" ]; then set +u; source /u/systems/UGE8.6.4/hoffman2/common/settings.sh; set -u; fi
case "${CONDA_PREFIX:-}" in */envs/*) echo "conda env ${CONDA_PREFIX} active; submit from a clean shell"; exit 1 ;; esac
MODE=${1:?usage: submit_feature_mcs_trees_h2.sh canary | fleet | lines <range> | check <arch> <cpu> <range>}
TF=cluster/feature_mcs_trees_h2_v3_lines.txt
R=results/feature_mcs_trees_work/chunks2
CF=results/feature_mcs_trees_work/h2_canary_ok
ARCHES=${ARCHES:-intel-E5-2670|intel-E5-2650v2}
REQUIRE_CPU=${REQUIRE_CPU:-E5-2670:E5-2650_v2}
H_RT=${H_RT:-2:00:00}
H_DATA=${H_DATA:-7G}  # canary 15145174 ran out of memory at 5G; 15145215 peaked at 5.03G virtual (3.8G resident)
NC=${NC:-2}
mkdir -p logs "$R"
for f in "$TF" cluster/feature_mcs_trees_h2_task.sh specs/causal_tune_trees_mcs.py specs/causal_tune_trees_xgbtop.py \
         specs/causal_tune_trees_rf400.py cluster/feature_mcs_trees_designs.json cluster/feature_mcs_trees_rows.json; do
  [ -f "$f" ] || { echo "$f missing (cluster/ship_feature_mcs_trees_h2.sh locally)"; exit 1; }
done
STAMP=logs/submitted_feature_mcs_trees_h2_$MODE.txt
case "$MODE" in lines|check) STAMP=/dev/null/none ;; esac  # no stamp: re-runs are idempotent through DONE
[ -f "$STAMP" ] && { echo "already submitted ($(cat "$STAMP"))"; exit 1; }
submit () {
  local out n=0
  until out=$(qsub -terse -S /bin/bash "$@" 2>&1); do
    n=$((n + 1)); echo "submit attempt $n failed: $out" >&2
    [ "$n" -ge 8 ] && return 1
    sleep 30
  done
  echo "${out%%.*}"
}
N=$(grep -c . "$TF")
ENVV="TASKFILE=$TF,RESULTS_ROOT=$R,REQUIRE_CPU=$REQUIRE_CPU"
case "$MODE" in
  canary)
    C=$(submit -N mcb_h2c -t 1-"$NC" -l "h_rt=$H_RT,h_data=$H_DATA,arch=$ARCHES" -v "$ENVV,CANARY_OUT=$CF" cluster/feature_mcs_trees_h2_task.sh)
    echo "canary: $C (1-$NC, arch $ARCHES) $(date)" | tee "$STAMP"
    ;;
  fleet)
    C=$(grep -o "canary: [0-9]*" logs/submitted_feature_mcs_trees_h2_canary.txt | awk '{print $2}')
    [ -n "$C" ] || { echo "no canary recorded"; exit 1; }
    F=$(submit -N mcb_h2f -hold_jid "$C" -t "$((NC + 1))-$N" -l "h_rt=$H_RT,h_data=$H_DATA,arch=$ARCHES" \
          -v "$ENVV,CANARY_DIR=$CF,CANARY_N=$NC" cluster/feature_mcs_trees_h2_task.sh)
    echo "fleet: $F ($((NC + 1))-$N, hold $C, arch $ARCHES) $(date)" | tee "$STAMP"
    ;;
  lines)
    # lines <range>: re-run task lines (e.g. 3-8: fleet tasks that refused before the canary flags existed); a line
    # whose chunk has DONE exits at once
    RG=${2:?a task range, e.g. 3-8}
    F=$(submit -N mcb_h2r -t "$RG" -l "h_rt=$H_RT,h_data=$H_DATA,arch=$ARCHES" \
          -v "$ENVV,CANARY_DIR=$CF,CANARY_N=$NC" cluster/feature_mcs_trees_h2_task.sh)
    echo "lines $RG: $F (arch $ARCHES, h_data $H_DATA) $(date)" | tee -a logs/submitted_feature_mcs_trees_h2_lines.txt
    ;;
  check)
    # check <arch> <cpu substring> <range>: re-run task lines on another CPU class into results/feature_mcs_trees_work/
    # xcheck_<arch> for a byte-for-byte comparison with the E5 chunks (2026-10-13: the AVX2 Zen 2 classes amd-epyc-7642,
    # amd-7542); never part of the fleet's results
    XA=${2:?arch}; XC=${3:?cpu substring, spaces as underscores}; RG=${4:?task range}
    F=$(submit -N "mcb_x${XA##*-}" -t "$RG" -l "h_rt=$H_RT,h_data=$H_DATA,arch=$XA" \
          -v "TASKFILE=$TF,RESULTS_ROOT=results/feature_mcs_trees_work/xcheck_$XA,REQUIRE_CPU=$XC" cluster/feature_mcs_trees_h2_task.sh)
    echo "check $XA $RG: $F (cpu $XC) $(date)" | tee -a logs/submitted_feature_mcs_trees_h2_lines.txt
    ;;
  *) echo "MODE must be canary, fleet, lines or check"; exit 1 ;;
esac
qstat -u "$USER" | tail -4
