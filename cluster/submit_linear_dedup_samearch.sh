#!/bin/bash
# I1 attribution, same CPU architecture (Hoffman2).  experiments/linear_dedup_control_plan.py
# --samearch writes, per node architecture, the 16:00 (forecast) pairs whose control run landed on
# that architecture:
#   cluster/linear_dedup_samearch_<arch>.txt  -> run with the DE-DUP executor  (the de-dup root)
#   cluster/linear_dedup_ctrlrep_<arch>.txt   -> run with the PRE-dedup executor (the control root)
# Each file becomes one array pinned to its architecture (qsub -l arch=<arch>), through the fleet's
# own task script.  TAG = samearch (run in /u/scratch/j/jamesdc1/harxhar-dedup) or ctrlrep (run in
# /u/scratch/j/jamesdc1/harxhar-dedup-control); the script refuses the wrong root.
#
#   cd /u/scratch/j/jamesdc1/harxhar-dedup && bash cluster/submit_linear_dedup_samearch.sh samearch
#   cd /u/scratch/j/jamesdc1/harxhar-dedup-control && bash cluster/submit_linear_dedup_samearch.sh ctrlrep
set -euo pipefail
cd "$(dirname "$0")/.."
TAG=${1:?usage: submit_linear_dedup_samearch.sh samearch|ctrlrep}
if [ -z "${SGE_ROOT:-}" ]; then set +u; source /u/systems/UGE8.6.4/hoffman2/common/settings.sh; set -u; fi
case "$TAG" in
  samearch) grep -q "session_edge" src/backtest/executor.py || { echo "samearch needs the de-dup executor"; exit 1; } ;;
  ctrlrep) ! grep -q "session_edge" src/backtest/executor.py || { echo "ctrlrep needs the pre-dedup executor"; exit 1; } ;;
  *) echo "TAG must be samearch or ctrlrep"; exit 1 ;;
esac
mkdir -p logs results
STAMP=logs/submitted_linear_dedup_$TAG.txt
[ -f "$STAMP" ] && { echo "already submitted ($(cat "$STAMP")); delete $STAMP by hand to resubmit"; exit 1; }
submit () {
  local out n=0
  until out=$(qsub -terse "$@" 2>&1); do
    n=$((n + 1)); echo "submit attempt $n failed: $out" >&2
    [ "$n" -ge 8 ] && return 1
    sleep 30
  done
  echo "${out%%.*}"
}
IDS=""
for F in cluster/linear_dedup_${TAG}_*.txt; do
  ARCH=$(basename "$F" .txt); ARCH=${ARCH#linear_dedup_${TAG}_}
  N=$(wc -l < "$F")
  J=$(submit -N "lin_dedup_$TAG" -t 1-"$N" -l "h_rt=1:00:00,h_data=16G,arch=$ARCH" -v TASKFILE="$F" cluster/linear_dedup_pack.sh)
  IDS="$IDS $ARCH:$J(1-$N)"
done
echo "$TAG:$IDS $(date)" | tee "$STAMP"
qstat -u "$USER" | tail -6
