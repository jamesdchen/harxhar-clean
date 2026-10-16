#!/bin/bash
# Cross-CPU reproducibility check of the dev forest on Hoffman2 (2026-10-07, D.19-style): the six per-bar 16:00
# random-forest cells (T1 / T10 x baseline / all_features / lean_vol_vix) re-computed on Hoffman2 CPUs with the CARC
# run's code, chunking and configuration, to be compared with the stored CARC epyc-7513 tables.  Two sets:
#   rfdev      rf_opt_00 (cluster/rfdev_rf_config.json), the chunks of cluster/trees_rfdev_tasks_untuned.txt
#   rfprev163  trial 163 (cluster/rfprev163_rf_config.json), the chunks of cluster/trees_rfprev_tasks_untuned.txt
# Task script cluster/trees_rfprev_h2_task.sh (CARC's own Python stack from agent D's runtime, read only; one SGE task =
# one chunk process, single-threaded, WINDOW_MASK=1).  No CPU pin: every Hoffman2 class the scheduler offers (the class
# of every chunk is recorded; the comparison is reported per class).  Results: results/h2check_<set>/{t1,t10}/...
#   bash cluster/submit_trees_rfh2.sh canary <set>   the 6 pack-0 chunks, each run twice (bit-identical repeat)
#   bash cluster/submit_trees_rfh2.sh fleet <set>    every other chunk, -hold_jid the canary, refusing to run unless
#                                                    the 6 canary flags exist; then the merge, -hold_jid the fleet
# Submit from a shell with no conda env active.  A stamp (logs/submitted_trees_rfh2_<mode>_<set>.txt) refuses a
# second identical submission (qdel is not available to this session); DONE chunks are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -z "${SGE_ROOT:-}" ]; then set +u; source /u/systems/UGE8.6.4/hoffman2/common/settings.sh; set -u; fi
case "${CONDA_PREFIX:-}" in */envs/*) echo "conda env ${CONDA_PREFIX} active; submit from a clean shell"; exit 1 ;; esac
MODE=${1:?usage: submit_trees_rfh2.sh canary|fleet <set>}
SET=${2:?set: rfdev or rfprev163}
case "$SET" in
  rfdev) TF=cluster/trees_rfdev_tasks_untuned.txt; CFG=cluster/rfdev_rf_config.json ;;
  rfprev163) TF=cluster/trees_rfprev_tasks_untuned.txt; CFG=cluster/rfprev163_rf_config.json ;;
  *) echo "set must be rfdev or rfprev163"; exit 1 ;;
esac
R=results/h2check_$SET
H_RT=${H_RT:-3:00:00}
H_DATA=${H_DATA:-6G}
mkdir -p logs "$R"
for f in "$TF" "$CFG" cluster/trees_rfprev_h2_task.sh specs/causal_tune_trees.py specs/causal_tune_trees_rfdev.py \
         experiments/trees_cadence_reduce_chunks.py src/data/loading.py; do
  [ -f "$f" ] || { echo "$f missing (run cluster/ship_trees_rfh2_h2.sh locally)"; exit 1; }
done
STAMP=logs/submitted_trees_rfh2_${MODE}_$SET.txt
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
N0=$(awk '$1 == 0' "$TF" | wc -l)
NALL=$(grep -c . "$TF")
[ "$(head -"$N0" "$TF" | awk '$1 != 0' | wc -l)" = 0 ] || { echo "$TF: the pack-0 lines are not first"; exit 1; }
case "$MODE" in
  canary)
    C=$(submit -N "rfh2c_$SET" -t 1-"$N0" -l "h_rt=$H_RT,h_data=$H_DATA" \
          -v "MODE=chunk,TASKFILE=$TF,RESULTS_ROOT=$R,HPC_KW_RF_CONFIG=$CFG,REPEAT=1,CANARY_OUT=$R/canary_ok" cluster/trees_rfprev_h2_task.sh)
    echo "canary $SET: $C (1-$N0, repeat) $(date)" | tee "$STAMP"
    ;;
  fleet)
    C=$(grep -o 'canary [a-z0-9]*: [0-9]*' "logs/submitted_trees_rfh2_canary_$SET.txt" | awk '{print $3}')
    [ -n "$C" ] || { echo "no canary job recorded for $SET"; exit 1; }
    F=$(submit -N "rfh2f_$SET" -hold_jid "$C" -t "$((N0 + 1))-$NALL" -l "h_rt=$H_RT,h_data=$H_DATA" \
          -v "MODE=chunk,TASKFILE=$TF,RESULTS_ROOT=$R,HPC_KW_RF_CONFIG=$CFG,CANARY_DIR=$R/canary_ok,CANARY_N=$N0" cluster/trees_rfprev_h2_task.sh)
    M=$(submit -N "rfh2m_$SET" -hold_jid "$F" -l "h_rt=1:00:00,h_data=8G" \
          -v "MODE=merge,TASKFILE=$TF,RESULTS_ROOT=$R,HPC_KW_RF_CONFIG=$CFG" cluster/trees_rfprev_h2_task.sh)
    echo "fleet $SET: $F ($((N0 + 1))-$NALL, hold $C) merge $M $(date)" | tee "$STAMP"
    ;;
  *) echo "MODE must be canary or fleet"; exit 1 ;;
esac
qstat -u "$USER" | tail -6
