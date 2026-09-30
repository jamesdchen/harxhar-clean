#!/bin/bash
# I6 attribution control (Hoffman2): six rest-of-day arms re-run with the PRE-dedup executor
# (src/backtest/executor.py of git 47f7f9c^, everything else = the I6 root), each pinned to the CPU
# architecture its I6 arm ran on -- the arm that moved most against the pre-dedup CARC arm, per
# estimator x bucket where any row moved by more than 1e-8 (experiments/restofday_dedup_gates.py,
# arms.csv).  Control vs I6 isolates the design change on one architecture; control vs the old CARC
# arm isolates the machine.  One array per file cluster/restofday_dedup_control_<arch>.txt, through
# the fleet's own task script (no canary flag: the pipeline passed its canary in the I6 run).
# Run in the control root cluster/ship_restofday_dedup_control_hoffman2.sh fills:
#
#   cd /u/scratch/j/jamesdc1/harxhar-restofday-dedup-control && bash cluster/submit_restofday_dedup_control.sh
#
# REFUSES a second submission while logs/submitted_restofday_dedup_control.txt exists.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -z "${SGE_ROOT:-}" ]; then set +u; source /u/systems/UGE8.6.4/hoffman2/common/settings.sh; set -u; fi
STAMP=logs/submitted_restofday_dedup_control.txt
MANIFEST=cluster/restofday_dedup_manifest.md5
mkdir -p logs
[ -f "$STAMP" ] && { echo "already submitted ($(cat "$STAMP")); delete $STAMP by hand to resubmit"; exit 1; }
! grep -q "session_edge" src/backtest/executor.py || { echo "the control needs the pre-dedup executor"; exit 1; }
md5sum -c --quiet "$MANIFEST" || { echo "manifest check failed: nothing submitted"; exit 1; }
echo "manifest ok: $(wc -l < "$MANIFEST") pinned files (pre-dedup executor)"
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
for F in cluster/restofday_dedup_control_*.txt; do
  ARCH=$(basename "$F" .txt); ARCH=${ARCH#restofday_dedup_control_}
  N=$(wc -l < "$F")
  J=$(submit -N rod_dedup_ctrl -t 1-"$N" -l "h_rt=1:00:00,h_data=16G,arch=$ARCH" -v TASKFILE="$F" \
        cluster/restofday_dedup_pack.sh)
  IDS="$IDS control/$ARCH=$J(1-$N)"
done
echo "${IDS# } $(date)" | tee "$STAMP"
qstat -u "$USER" | tail -5
