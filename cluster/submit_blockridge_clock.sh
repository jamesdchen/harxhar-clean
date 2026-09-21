#!/bin/bash
# Block ridge with per-clock HAR deltas, on the incumbent panel, on Hoffman2.
# Run from the deployment root:
#
#   cd /u/scratch/j/jamesdc1/harxhar-subsection && bash cluster/submit_blockridge_clock.sh
#
# Canary: one chunk of the middle-penalty clock arm.  The fleet (the pooled arm
# and the three clock arms, ten tasks of ten chunks each) is held on it and
# refuses to run unless the canary chunk exists; the reducer/scorer is held on
# the fleet.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results/unification_clock
TASKS=cluster/blockridge_tasks.txt
CANARY=cluster/blockridge_tasks_canary.txt
FLAG=results/unification_clock/blk2_clockhar_a100/chunk_095.npz
N=$(wc -l < "$TASKS")

CAN=$(qsub -terse -N br_canary -t 1-1 -v TASKFILE=$CANARY cluster/blockridge_batch.sh | cut -d. -f1)
A=$(qsub -terse -N br_fleet -hold_jid "$CAN" -t 1-"$N" \
      -v TASKFILE=$TASKS,CANARY_FLAG=$FLAG cluster/blockridge_batch.sh | cut -d. -f1)
S=$(qsub -terse -N br_score -hold_jid "$A" cluster/blockridge_score.sh | cut -d. -f1)

echo "canary=$CAN fleet=$A score=$S" | tee logs/submitted_blockridge.txt
qstat -u "$USER" | tail -6
