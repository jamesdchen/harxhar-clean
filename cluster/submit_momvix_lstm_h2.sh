#!/bin/bash
# 2026-10-06 (moments_vix LSTMs): Hoffman2 submissions.  Hoffman2 may host a moments_vix task only on a CPU
# class shown bit-identical to the stored CARC (epyc-7513) refits; this decides it.
#   check   the reproduction-gate lines of both LSTMs (cluster/lstm_momvix_tasks_gate.txt,
#           cluster/lstmi_momvix_tasks_gate.txt: stored refits of baseline / all_features) on each CPU
#           class in ARCHES, one array per (class, model), into results/momvix_gates_h2/<class>/<model>;
#           compared with the stored chunks by experiments/momvix_lstm_gate_compare.py (after the pull).
#   fleet   CLASS=<arch> TASKFILE=<file> RESULTS_ROOT=<dir>: task lines on ONE class that passed the
#           check (refuses unless results/momvix_gates_h2/<class>/PASS exists, written after the compare).
#   cd /u/scratch/j/jamesdc1/harxhar-momvix-lstm && bash cluster/submit_momvix_lstm_h2.sh check|fleet
set -euo pipefail
cd "$(dirname "$0")/.."
[ "$(basename "$PWD")" = harxhar-momvix-lstm ] || { echo "run in /u/scratch/j/jamesdc1/harxhar-momvix-lstm"; exit 1; }
mkdir -p logs results
[ -x carc_env/harxhar/bin/python3.11 ] || { echo "runtime missing: bash cluster/momvix_lstm_h2_setup.sh"; exit 1; }
MODE="${1:-}"
SLOTS=${SLOTS:-8}
HDATA=${HDATA:-5G}   # per slot
case "$MODE" in
  check)
    ARCHES=${ARCHES:-"amd-7542 amd-epyc-7642 amd-5955 intel-E5-2650v4"}
    IDS=""
    for A in $ARCHES; do
      for M in perbar intraday; do
        if [ $M = perbar ]; then TF=cluster/lstm_momvix_tasks_gate.txt; else TF=cluster/lstmi_momvix_tasks_gate.txt; fi
        J=$(qsub -terse -N "mvl_h2_${M}" -t 1-"$(grep -c . "$TF")" -pe shared "$SLOTS" \
              -l h_rt=2:00:00,h_data="$HDATA",arch="$A" \
              -v TASKFILE="$TF",RESULTS_ROOT="results/momvix_gates_h2/$A/$M",REQUIRE_CACHE=0 cluster/momvix_lstm_h2_task.sh)
        IDS="$IDS $A/$M=${J%%.*}"
      done
    done
    echo "$(date +%F_%T) check$IDS" | tee -a logs/submitted_momvix_lstm_h2.txt
    ;;
  fleet)
    : "${CLASS:?CLASS}" "${TASKFILE:?TASKFILE}" "${RESULTS_ROOT:?RESULTS_ROOT}"
    [ -f "results/momvix_gates_h2/$CLASS/PASS" ] || { echo "class $CLASS has not passed the check"; exit 1; }
    J=$(qsub -terse -N mvl_h2_fleet -t 1-"$(grep -c . "$TASKFILE")" -pe shared "$SLOTS" \
          -l h_rt="${HRT:-4:00:00}",h_data="$HDATA",arch="$CLASS" \
          -v TASKFILE="$TASKFILE",RESULTS_ROOT="$RESULTS_ROOT",REQUIRE_CACHE=1 cluster/momvix_lstm_h2_task.sh)
    echo "$(date +%F_%T) fleet class=$CLASS tasks=$TASKFILE root=$RESULTS_ROOT job=${J%%.*}" | tee -a logs/submitted_momvix_lstm_h2.txt
    ;;
  *) echo "usage: bash cluster/submit_momvix_lstm_h2.sh check|fleet"; exit 1 ;;
esac
qstat -u "$USER" | head -30
