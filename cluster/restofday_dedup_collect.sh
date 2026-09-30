#!/bin/bash
# I6 (Hoffman2): held on every I6 array (any exit status).  Walks every arm of the canary, 15:30
# and fleet task files, lists the ones without DONE (last line of their run.log), sums the arms'
# own run seconds, and writes results/linear_subsection_restofday_dedup/FLEET_DONE with the
# counts.  No scoring here: the gates and the tables are built locally after the pull
# (experiments/restofday_dedup_gates.py, experiments/build_restofday_yhat.py).
#$ -cwd
#$ -j y
#$ -o logs/
#$ -l h_rt=0:20:00,h_data=2G
set -uo pipefail
R=results/linear_subsection_restofday_dedup
EXPECTED=0
FINISHED=0
SECS=0
while read -r BUCKET ESTS TW CLOCKS; do
  [ -n "${CLOCKS:-}" ] || continue
  for EST in ${ESTS//,/ }; do
    for CLOCK in ${CLOCKS//,/ }; do
      EXPECTED=$((EXPECTED + 1))
      D="$R/$BUCKET/rod$CLOCK/$EST/tw$TW"
      if [ -f "$D/DONE" ]; then
        FINISHED=$((FINISHED + 1))
        [ -f "$D/SECONDS" ] && SECS=$((SECS + $(cat "$D/SECONDS")))
      elif [ -f "$D/run.log" ]; then
        echo "NOT DONE: $D: $(tail -1 "$D/run.log" | cut -c1-200)"
      else
        echo "NOT DONE: $D: never started"
      fi
    done
  done
done < <(cat cluster/restofday_dedup_tasks_canary.txt cluster/restofday_dedup_tasks_1530_*.txt \
             cluster/restofday_dedup_tasks_fleet.txt)
echo "arms finished $FINISHED of $EXPECTED; arm run seconds $SECS  $(date)" | tee "$R/FLEET_DONE"
