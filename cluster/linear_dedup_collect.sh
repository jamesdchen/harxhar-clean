#!/bin/bash
# I1 (Hoffman2): held on the fleet (any exit status).  Walks every arm of the fleet task file,
# lists the ones without DONE (last line of their run.log), sums the arms' own run seconds, and
# writes results/linear_subsection_dedup/FLEET_DONE with the counts.  No scoring here: the gates
# and the tables are built locally after the pull (experiments/linear_dedup_gates.py,
# experiments/rebuild_yhat_dedup.py).
#$ -cwd
#$ -j y
#$ -o logs/
#$ -l h_rt=0:20:00,h_data=2G
set -uo pipefail
R=results/linear_subsection_dedup
EXPECTED=0
FINISHED=0
SECS=0
while read -r ROOT BUCKET EST TW HAR LAG SEGS; do
  [ -n "${SEGS:-}" ] || continue
  for SEG in ${SEGS//,/ }; do
    EXPECTED=$((EXPECTED + 1))
    D="$ROOT/$BUCKET/$SEG/$EST/tw$TW"
    if [ -f "$D/DONE" ]; then
      FINISHED=$((FINISHED + 1))
      [ -f "$D/SECONDS" ] && SECS=$((SECS + $(cat "$D/SECONDS")))
    elif [ -f "$D/run.log" ]; then
      echo "NOT DONE: $D: $(tail -1 "$D/run.log" | cut -c1-200)"
    else
      echo "NOT DONE: $D: never started"
    fi
  done
done < cluster/linear_dedup_tasks_fleet.txt
echo "arms finished $FINISHED of $EXPECTED; arm run seconds $SECS  $(date)" | tee "$R/FLEET_DONE"
