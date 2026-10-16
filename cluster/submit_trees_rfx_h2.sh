#!/bin/bash
# rfx on Hoffman2 (2026-10-07): the Hoffman2 share of a dev-span forest's FIXED-SETTINGS chunks (KIND fx = masked, nm =
# unmasked; token TOKEN), pinned to the AVX CPU models that reproduced CARC's epyc-7513 forest byte for byte in the
# cross-CPU check (results/h2check_*: E5-2670, E5-2650 v2) -- SGE arch request ARCHES (default
# intel-E5-2670|intel-E5-2650v2) AND, inside every task, REQUIRE_CPU (default E5-2670:E5-2650_v2: any other CPU model refuses
# to run).  One SGE task = one chunk line of cluster/trees_<TOKEN>_<KIND>_h2lines.txt (the lines the split gave Hoffman2;
# experiments/trees_rfx_split.py), the CARC run's code / configuration / chunk geometry, single-threaded fits, into the CARC
# layout (results/linear_subsection_trees_{mask,nomask}_<TOKEN>/...), relayed to CARC before the merge
# (cluster/relay_trees_rfx_h2.sh).  Task script cluster/trees_rfprev_h2_task.sh (CARC's own Python stack, read only).
#   TOKEN=rf400 bash cluster/submit_trees_rfx_h2.sh canary fx   the first 2 lines, each run twice (bit-identical repeat)
#   TOKEN=rf400 bash cluster/submit_trees_rfx_h2.sh fleet fx    the other lines, -hold_jid the canary, each refusing to run
#                                                               unless both canary flags exist
#   TOKEN=rf400 LINES=<file> bash cluster/submit_trees_rfx_h2.sh extra nm   every line of LINES (e.g. the still-pending CARC
#                                                               packs, duplicated here; a chunk with DONE is skipped, and
#                                                               cluster/sync_trees_rfx_done.sh shares DONE both ways), each
#                                                               refusing to run unless the kind's canary flags exist
# Submit from a shell with no conda env active.  A stamp refuses a second identical submission.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -z "${SGE_ROOT:-}" ]; then set +u; source /u/systems/UGE8.6.4/hoffman2/common/settings.sh; set -u; fi
case "${CONDA_PREFIX:-}" in */envs/*) echo "conda env ${CONDA_PREFIX} active; submit from a clean shell"; exit 1 ;; esac
MODE=${1:?usage: submit_trees_rfx_h2.sh canary|fleet fx|nm}
KIND=${2:?kind fx or nm}
TOKEN=${TOKEN:?TOKEN}
case "$KIND" in
  fx) WM=1; R=results/linear_subsection_trees_mask_$TOKEN ;;
  nm) WM=0; R=results/linear_subsection_trees_nomask_$TOKEN ;;
  *) echo "kind must be fx or nm"; exit 1 ;;
esac
TF=${LINES:-cluster/trees_${TOKEN}_${KIND}_h2lines.txt}
CFG=cluster/${TOKEN}_rf_config.json
ARCHES=${ARCHES:-intel-E5-2670|intel-E5-2650v2}
REQUIRE_CPU=${REQUIRE_CPU:-E5-2670:E5-2650_v2}
H_RT=${H_RT:-4:00:00}
H_DATA=${H_DATA:-6G}
NC=${NC:-2}
mkdir -p logs "$R"
for f in "$TF" "$CFG" cluster/trees_rfprev_h2_task.sh specs/causal_tune_trees.py specs/causal_tune_trees_rfdev.py src/data/loading.py; do
  [ -f "$f" ] || { echo "$f missing (TOKEN=$TOKEN cluster/ship_trees_rfx_h2.sh locally)"; exit 1; }
done
STAMP=logs/submitted_trees_rfx_h2_${TOKEN}_${MODE}_$KIND${LINES:+_$(basename "$LINES" .txt)}.txt
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
ENVV="MODE=chunk,TASKFILE=$TF,RESULTS_ROOT=$R,HPC_KW_RF_CONFIG=$CFG,WINDOW_MASK=$WM,REQUIRE_CPU=$REQUIRE_CPU"
case "$MODE" in
  canary)
    C=$(submit -N "rfx_h2c_$KIND" -t 1-"$NC" -l "h_rt=$H_RT,h_data=$H_DATA,arch=$ARCHES" -v "$ENVV,REPEAT=1,CANARY_OUT=$R/h2_canary_ok" cluster/trees_rfprev_h2_task.sh)
    echo "canary $TOKEN $KIND: $C (1-$NC, repeat, arch $ARCHES) $(date)" | tee "$STAMP"
    ;;
  fleet)
    C=$(grep -o "canary $TOKEN $KIND: [0-9]*" "logs/submitted_trees_rfx_h2_${TOKEN}_canary_$KIND.txt" | awk '{print $4}')
    [ -n "$C" ] || { echo "no canary recorded"; exit 1; }
    F=$(submit -N "rfx_h2f_$KIND" -hold_jid "$C" -t "$((NC + 1))-$N" -l "h_rt=$H_RT,h_data=$H_DATA,arch=$ARCHES" \
          -v "$ENVV,CANARY_DIR=$R/h2_canary_ok,CANARY_N=$NC" cluster/trees_rfprev_h2_task.sh)
    echo "fleet $TOKEN $KIND: $F ($((NC + 1))-$N, hold $C, arch $ARCHES) $(date)" | tee "$STAMP"
    ;;
  extra)
    [ -n "${LINES:-}" ] || { echo "extra needs LINES"; exit 1; }
    ls "$R/h2_canary_ok"/ok_* >/dev/null 2>&1 || { echo "no canary flags in $R/h2_canary_ok"; exit 1; }
    F=$(submit -N "rfx_h2x_$KIND" -t 1-"$N" -l "h_rt=$H_RT,h_data=$H_DATA,arch=$ARCHES"           -v "$ENVV,CANARY_DIR=$R/h2_canary_ok,CANARY_N=$NC" cluster/trees_rfprev_h2_task.sh)
    echo "extra $TOKEN $KIND: $F (1-$N of $TF, arch $ARCHES) $(date)" | tee "$STAMP"
    ;;
  *) echo "MODE must be canary, fleet or extra"; exit 1 ;;
esac
qstat -u "$USER" | tail -5
