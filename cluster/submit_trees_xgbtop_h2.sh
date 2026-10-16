#!/bin/bash
# xgbtop on Hoffman2 (2026-10-07): the per-bar XGBoost (xgb_opt_00) runs on Hoffman2 AVX-only nodes beside CARC, through
# cluster/trees_xgbtop_h2_task.sh (CARC's Python stack; every task refuses a CPU outside ALLOW_CPU and records its model).
#   bash cluster/submit_trees_xgbtop_h2.sh gate <arch>     the reproduction gate on one pinned class (qsub -l arch=<arch>:
#        intel-E5-2670 or intel-E5-2650v2): specs/causal_tune_trees_xgb06.py must reproduce the stored CARC epyc-7513
#        XGBoost chunks of cluster/trees_xgbtop_h2_gate_lines.txt (copied under ./ref) and the published table cells byte
#        for byte -> results/trees_xgbtop_gates/h2_<arch>/GATE_OK
#   bash cluster/submit_trees_xgbtop_h2.sh chunks <taskfile> <results root> <mask 0|1> <arch>   an array over the lines of
#        the task file (one chunk process each), refusing to run unless every given gate passed (GATES, default both
#        AVX-only gates' GATE_OK)
#   bash cluster/submit_trees_xgbtop_h2.sh optuna <taskfile> <results root> <arch>   Optuna stage-1 lines (a subset of
#        cluster/xgbtop_optuna_tasks_s1.txt), a spawned pool of SLOTS (default 16) single-threaded workers each, same gates
#   bash cluster/submit_trees_xgbtop_h2.sh tuned <taskfile> <results root> <rs10|rs1> <arch>   random-search chunk-arms
#        (a subset of cluster/trees_xgbtop_tasks_<rung>.txt), a spawned pool of SLOTS workers each, same gates
#   bash cluster/submit_trees_xgbtop_h2.sh merge <taskfile> <results root> [hold job id]   the fixed-settings reducer
# Submit from a shell with no conda env active.  A stamp (logs/submitted_trees_xgbtop_h2_<mode>_<tag>.txt) refuses a
# second identical submission (qdel is not available to this session); DONE chunks are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -z "${SGE_ROOT:-}" ]; then set +u; source /u/systems/UGE8.6.4/hoffman2/common/settings.sh; set -u; fi
case "${CONDA_PREFIX:-}" in */envs/*) echo "conda env ${CONDA_PREFIX} active; submit from a clean shell"; exit 1 ;; esac
MODE=${1:?usage: submit_trees_xgbtop_h2.sh gate <arch> | chunks <taskfile> <root> <mask> <arch>}
mkdir -p logs
submit () {
  local out n=0
  until out=$(qsub -terse -S /bin/bash "$@" 2>&1); do
    n=$((n + 1)); echo "submit attempt $n failed: $out" >&2
    [ "$n" -ge 8 ] && return 1
    sleep 30
  done
  echo "${out%%.*}"
}
for f in cluster/trees_xgbtop_h2_task.sh specs/causal_tune_trees.py specs/causal_tune_trees_xgbtop.py specs/causal_tune_trees_xgb06.py \
         experiments/trees_xgbtop_gate.py experiments/trees_momvix_gate.py experiments/trees_xgbtop_make_specs.py src/data/loading.py; do
  [ -f "$f" ] || { echo "$f missing (run cluster/ship_trees_xgbtop_h2.sh locally)"; exit 1; }
done
case "$MODE" in
  gate)
    ARCH=${2:?arch}
    # GATE_SLOTS (default 1): the first submission (-pe shared 8, 15098179 / 15098180) sat queued for over an hour on
    # these nodes; one slot runs the 8 lines one after another into a separate folder (h2_<arch>_s1)
    GS=${GATE_SLOTS:-1}
    SUF=$([ "$GS" = 8 ] && echo "" || echo "_s$GS")
    STAMP=logs/submitted_trees_xgbtop_h2_gate_$ARCH$SUF.txt
    [ -f "$STAMP" ] && { echo "already submitted ($(cat "$STAMP"))"; exit 1; }
    OUT=results/trees_xgbtop_gates/h2_$ARCH$SUF
    mkdir -p "$OUT"; rm -f "$OUT/GATE_OK"
    PE=(); [ "$GS" -gt 1 ] && PE=(-pe shared "$GS")
    HD=12G; [ "$GS" -gt 1 ] && HD=6G
    J=$(submit -N "xth2g_${ARCH#intel-}" ${PE[@]+"${PE[@]}"} -l "h_rt=2:30:00,h_data=$HD,arch=$ARCH" \
          -v "MODE=gate,OUT=$OUT,GATE_LINES=cluster/trees_xgbtop_h2_gate_lines.txt" cluster/trees_xgbtop_h2_task.sh)
    echo "gate $ARCH ($GS slots): $J $(date)" | tee "$STAMP"
    ;;
  chunks)
    TF=${2:?taskfile}; R=${3:?results root}; MASK=${4:?mask}; ARCH=${5:?arch}
    [ -f "$TF" ] || { echo "$TF missing"; exit 1; }
    GATES=${GATES:-results/trees_xgbtop_gates/h2_intel-E5-2670_s1/GATE_OK results/trees_xgbtop_gates/h2_intel-E5-2650v2_s1/GATE_OK}
    for g in $GATES; do [ -f "$g" ] || { echo "gate flag $g missing; refusing"; exit 1; }; done
    TAG="$(basename "$TF" .txt)_m${MASK}_$ARCH"
    STAMP=logs/submitted_trees_xgbtop_h2_chunks_$TAG.txt
    [ -f "$STAMP" ] && { echo "already submitted ($(cat "$STAMP"))"; exit 1; }
    N=$(grep -c . "$TF")
    J=$(submit -N "xth2c_${ARCH#intel-}" -t 1-"$N" -l "h_rt=${H_RT:-2:00:00},h_data=${H_DATA:-8G},arch=$ARCH" \
          -v "MODE=chunk,TASKFILE=$TF,RESULTS_ROOT=$R,MASK=$MASK,TOKEN=xgbtop,XGB_CONFIG=cluster/xgbtop_xgb_config.json" \
          cluster/trees_xgbtop_h2_task.sh)
    echo "chunks $TF -> $R (mask $MASK, $ARCH): $J (1-$N) $(date)" | tee "$STAMP"
    ;;
  optuna)
    TF=${2:?taskfile}; R=${3:?results root}; ARCH=${4:?arch}
    [ -f "$TF" ] || { echo "$TF missing"; exit 1; }
    GATES=${GATES:-results/trees_xgbtop_gates/h2_intel-E5-2670_s1/GATE_OK results/trees_xgbtop_gates/h2_intel-E5-2650v2_s1/GATE_OK}
    for g in $GATES; do [ -f "$g" ] || { echo "gate flag $g missing; refusing"; exit 1; }; done
    TAG="$(basename "$TF" .txt)_$ARCH"
    STAMP=logs/submitted_trees_xgbtop_h2_optuna_$TAG.txt
    [ -f "$STAMP" ] && { echo "already submitted ($(cat "$STAMP"))"; exit 1; }
    N=$(grep -c . "$TF")
    SLOTS=${SLOTS:-16}
    J=$(submit -N "xth2o_${ARCH#intel-}" -t 1-"$N" -pe shared "$SLOTS" -l "h_rt=${H_RT:-6:00:00},h_data=${H_DATA:-2G},arch=$ARCH" \
          -v "MODE=optuna,TASKFILE=$TF,RESULTS_ROOT=$R,TOKEN=xgbtop,XGB_CONFIG=cluster/xgbtop_xgb_config.json" \
          cluster/trees_xgbtop_h2_task.sh)
    echo "optuna stage 1 $TF -> $R ($ARCH, $SLOTS slots): $J (1-$N) $(date)" | tee "$STAMP"
    ;;
  tuned)
    TF=${2:?taskfile}; R=${3:?results root}; RUNG=${4:?rs10 or rs1}; ARCH=${5:?arch}
    [ -f "$TF" ] || { echo "$TF missing"; exit 1; }
    GATES=${GATES:-results/trees_xgbtop_gates/h2_intel-E5-2670_s1/GATE_OK results/trees_xgbtop_gates/h2_intel-E5-2650v2_s1/GATE_OK}
    for g in $GATES; do [ -f "$g" ] || { echo "gate flag $g missing; refusing"; exit 1; }; done
    TAG="$(basename "$TF" .txt)_${RUNG}_$ARCH"
    STAMP=logs/submitted_trees_xgbtop_h2_tuned_$TAG.txt
    [ -f "$STAMP" ] && { echo "already submitted ($(cat "$STAMP"))"; exit 1; }
    N=$(grep -c . "$TF")
    SLOTS=${SLOTS:-16}
    J=$(submit -N "xth2t_${RUNG}" -t 1-"$N" -pe shared "$SLOTS" -l "h_rt=${H_RT:-6:00:00},h_data=${H_DATA:-3G},arch=$ARCH" \
          -v "MODE=tuned,RUNG=$RUNG,TASKFILE=$TF,RESULTS_ROOT=$R,TOKEN=xgbtop,XGB_CONFIG=cluster/xgbtop_xgb_config.json" \
          cluster/trees_xgbtop_h2_task.sh)
    echo "tuned $RUNG $TF -> $R ($ARCH, $SLOTS slots): $J (1-$N) $(date)" | tee "$STAMP"
    ;;
  merge)
    TF=${2:?taskfile}; R=${3:?results root}; HOLD=${4:-}
    STAMP=logs/submitted_trees_xgbtop_h2_merge_$(basename "$R").txt
    [ -f "$STAMP" ] && { echo "already submitted ($(cat "$STAMP"))"; exit 1; }
    J=$(submit -N xth2m ${HOLD:+-hold_jid "$HOLD"} -l "h_rt=1:00:00,h_data=8G" \
          -v "MODE=merge,TASKFILE=$TF,RESULTS_ROOT=$R" cluster/trees_xgbtop_h2_task.sh)
    echo "merge $R: $J${HOLD:+ (hold $HOLD)} $(date)" | tee "$STAMP"
    ;;
  *) echo "MODE must be gate, chunks, optuna, tuned or merge"; exit 1 ;;
esac
qstat -u "$USER" | tail -6
