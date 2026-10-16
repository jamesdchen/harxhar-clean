#!/bin/bash
# Pooled forest all_features (2026-10-14): keep CARC near the 100-running-job cap.  The 20/16-CPU tasks go first (nice 95),
# but main's idle cores are mostly memory-stranded (2-8 GB free a node), so 7G small tasks are needed to fill the gaps.
# The pending 4/8-CPU tasks are HELD in reserve; when fewer than TARGET (97) of my jobs run and fewer than BUF (50)
# unheld small (<= 8 CPU) tasks are pending on main, release held small tasks, lowest index first, to restore BUF.
#   bash cluster/slurm/pooltree_af_topup.sh          (on CARC; HELD_ARRAYS overrides the reserve arrays, first first)
set -euo pipefail
cd "$(dirname "$0")/../.."
TARGET=${TARGET:-97}; BUF=${BUF:-50}
HELD_ARRAYS=${HELD_ARRAYS:-13074358 13166916 13174440 13077718 13080577}   # w7, w24, w28 (8 CPUs; reserves added 2026-10-15), w9, w10 (4 CPUs)
RUN=$( (squeue -u "$USER" -h -t R -o %i || true) | wc -l)
SMALL=$( (squeue -u "$USER" -h -r -t PD -p main -o "%C %r" || true) | awk '$1 <= 8 && $2 != "JobHeldUser"' | wc -l)
echo "$(date +%T) running $RUN; unheld small pending on main $SMALL"
{ [ "$RUN" -lt "$TARGET" ] && [ "$SMALL" -lt "$BUF" ]; } || exit 0
N=$((BUF - SMALL))
for J in $HELD_ARRAYS; do
  [ "$N" -gt 0 ] || break
  IDX=$( (squeue -j "$J" -h -r -t PD -o "%i %r" 2>/dev/null || true) | awk '$2 == "JobHeldUser" {print $1}' | sed 's/.*_//' | sort -n | awk -v n="$N" 'NR <= n' | paste -sd, -)   # awk, not head: head + pipefail = SIGPIPE exit 141 (2026-10-14)
  [ -n "$IDX" ] || continue
  scontrol release "${J}_[$IDX]"
  K=$(echo "$IDX" | tr , '\n' | wc -l)
  echo "released $K of $J: ${IDX%%,*}..${IDX##*,}"
  N=$((N - K))
done
