#!/bin/bash
# I1 attribution control (Hoffman2): the arms of cluster/linear_dedup_control_tasks.txt with the
# PRE-dedup executor, in the control root cluster/ship_linear_dedup_control_hoffman2.sh fills:
#
#   cd /u/scratch/j/jamesdc1/harxhar-dedup-control && bash cluster/submit_linear_dedup_control.sh
#
# One array (cluster/linear_dedup_pack.sh, the fleet's own task script) and a collector.  No canary:
# the code path is the de-dup fleet's, canaried, and the executor is the one that made the
# pre-dedup arms.  Refuses a second submission while logs/submitted_linear_dedup_control.txt exists.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -z "${SGE_ROOT:-}" ]; then set +u; source /u/systems/UGE8.6.4/hoffman2/common/settings.sh; set -u; fi
mkdir -p logs results
TASKS=cluster/linear_dedup_control_tasks.txt
STAMP=logs/submitted_linear_dedup_control.txt
[ -f "$STAMP" ] && { echo "already submitted ($(cat "$STAMP")); delete $STAMP by hand to resubmit"; exit 1; }
if grep -q "session_edge" src/backtest/executor.py; then
  echo "src/backtest/executor.py is the de-dup executor: this root must hold the pre-dedup one"; exit 1
fi
submit () {
  local out n=0
  until out=$(qsub -terse "$@" 2>&1); do
    n=$((n + 1)); echo "submit attempt $n failed: $out" >&2
    [ "$n" -ge 8 ] && return 1
    sleep 30
  done
  echo "${out%%.*}"
}
N=$(wc -l < "$TASKS")
A=$(submit -N lin_dedup_ctrl -t 1-"$N" -v TASKFILE=$TASKS cluster/linear_dedup_pack.sh)
echo "control=$A (1-$N) $(date)" | tee "$STAMP"
qstat -u "$USER" | tail -4
