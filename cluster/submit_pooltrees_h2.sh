#!/bin/bash
# Pooled tree walks on Hoffman2 AVX-only nodes (2026-10-07), through cluster/pooltree_h2_task.sh (one slot a task).
#   bash cluster/submit_pooltrees_h2.sh gate <arch>                bank arm 10 (XGBoost) rows at 1 thread on that class
#        (intel-E5-2670 | intel-E5-2650v2) vs the stored bank chunks -> results/pooltree_gates/h2_<arch>/GATE_OK
#   [REFIT_EVERY=K] bash cluster/submit_pooltrees_h2.sh chunk <taskfile> <arch>   an array over the task file's lines
#        (xgb / rf only), refusing to run unless that class's gate passed
#   H_RT=<h:m:s> [ARRAY=a-b] bash cluster/submit_pooltrees_h2.sh pieces <taskfile> '<arch>[|<arch>]'   row pieces
#        (experiments/pooled_trees_pieces.py), one a task, after each class's gate and forest cross-check
# Submit from a shell with no conda env active; stamps (logs/submitted_pooltrees_h2_*.txt) refuse resubmission.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -z "${SGE_ROOT:-}" ]; then set +u; source /u/systems/UGE8.6.4/hoffman2/common/settings.sh; set -u; fi
case "${CONDA_PREFIX:-}" in */envs/*) echo "conda env ${CONDA_PREFIX} active; submit from a clean shell"; exit 1 ;; esac
MODE=${1:?usage: submit_pooltrees_h2.sh gate <arch> | chunk <taskfile> <arch> | pieces <taskfile> <arch[|arch]>}
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
for f in cluster/pooltree_h2_task.sh experiments/pooled_trees_walk.py experiments/pooled_trees_gate.py cache/prep_cache_all_features_b2.npz; do
  [ -f "$f" ] || { echo "$f missing (ship first)"; exit 1; }
done
case "$MODE" in
  gate)
    ARCH=${2:?arch}
    S=logs/submitted_pooltrees_h2_gate_$ARCH.txt; [ -f "$S" ] && { echo "already submitted ($(cat "$S"))"; exit 1; }
    OUT=results/pooltree_gates/h2_$ARCH; mkdir -p "$OUT"; rm -f "$OUT/GATE_OK"
    J=$(submit -N "ptrh2g_${ARCH#intel-}" -l "h_rt=3:00:00,h_data=8G,arch=$ARCH" \
          -v "MODE=gate,OUT=$OUT,GATE_LINES=cluster/pooltree_h2_gate_lines.txt" cluster/pooltree_h2_task.sh)
    echo "gate $ARCH: $J $(date)" | tee "$S"
    ;;
  chunk)
    TF=${2:?taskfile}; ARCH=${3:?arch}
    [ -f "$TF" ] || { echo "$TF missing"; exit 1; }
    [ -f "results/pooltree_gates/h2_$ARCH/GATE_OK" ] || { echo "the $ARCH gate has not passed; refusing"; exit 1; }
    K=${REFIT_EVERY:-1}
    S=logs/submitted_pooltrees_h2_chunk_$(basename "$TF" .txt)_r${K}_$ARCH.txt; [ -f "$S" ] && { echo "already submitted ($(cat "$S"))"; exit 1; }
    N=$(grep -c . "$TF")
    J=$(submit -N "ptrh2_${ARCH#intel-}" -t 1-"$N" -l "h_rt=${H_RT:-24:00:00},h_data=${H_DATA:-8G},arch=$ARCH" \
          -v "MODE=chunk,TASKFILE=$TF,REFIT_EVERY=$K" cluster/pooltree_h2_task.sh)
    echo "chunk $TF (refit every $K) $ARCH: $J (1-$N) $(date)" | tee "$S"
    ;;
  pieces)
    # ONE slot a task (multi-slot jobs share a 200-slot cap here; single-slot ones do not), a tight h_rt (H_RT, required),
    # h_data 8G: the panel load peaks at 5.6 GB RSS / 6.0 GB VM (qacct, cross-check 15099759) and Hoffman2 sets h_vmem =
    # h_data, so 6G killed every task at load (15100609-11, 2026-10-08); E5 hosts have little free h_data, so no more than 8G.
    # ARCH may list classes ("intel-E5-2670|intel-E5-2650v2"): each needs its XGBoost gate (GATE_OK) and, for the forest,
    # its one-row cross-check against the CARC canary (results/pooltree_xcheck/h2_<arch>/XCHECK_OK, written after the
    # byte comparison).  An index range of the task file: ARRAY (default all).  ALLOW_CPU (regex on the CPU model)
    # overrides the task's E5-2670 / E5-2650 v2 default -- for canaries on other classes and, once they match, the fleet.
    TF=${2:?taskfile}; ARCH=${3:?arch}
    [ -f "$TF" ] || { echo "$TF missing"; exit 1; }
    EST=$(head -1 "$TF" | cut -d' ' -f1)
    for A1 in ${ARCH//|/ }; do
      # the forest is checked by its own byte cross-check below; the XGBoost gate binds the boosted arms only (2026-10-14:
      # E5-2650v4 passed the forest cross-check, 3 of 3 sets, and has no XGBoost gate)
      [ "${SKIP_GATE:-0}" = 1 ] || [ "$EST" = rf ] || [ -f "results/pooltree_gates/h2_$A1/GATE_OK" ] || { echo "the $A1 gate has not passed; refusing"; exit 1; }
      [ "${SKIP_GATE:-0}" = 1 ] || [ "$EST" != rf ] || [ -f "results/pooltree_xcheck/h2_$A1/XCHECK_OK" ] || { echo "the $A1 forest cross-check has not passed; refusing"; exit 1; }
    done
    AT=${ARCH//|/+}
    S=logs/submitted_pooltrees_h2_pieces_$(basename "$TF" .txt)_$AT${TAG:+_$TAG}.txt; [ -f "$S" ] && { echo "already submitted ($(cat "$S"))"; exit 1; }
    N=$(grep -c . "$TF")
    NM=$(basename "$TF" .txt); NM=${NM#pooltree_pieces_}
    J=$(submit -N "pp_${NM:0:12}" -t "${ARRAY:-1-$N}" -l "h_rt=${H_RT:?H_RT},h_data=${H_DATA:-8G},arch=$ARCH" \
          -v "MODE=pieces,TASKFILE=$TF,RESULTS_ROOT=${RROOT:-results/pooled_trees_2026-10-07}${ALLOW_CPU:+,ALLOW_CPU=$ALLOW_CPU}" cluster/pooltree_h2_task.sh)
    echo "pieces $TF $ARCH: $J (${ARRAY:-1-$N}, h_rt $H_RT, h_data ${H_DATA:-8G}) $(date)" | tee "$S"
    ;;
  *) echo "MODE must be gate, chunk or pieces"; exit 1 ;;
esac
qstat -u "$USER" | tail -5
