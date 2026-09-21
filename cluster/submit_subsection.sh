#!/bin/bash
# Submit the subsection-regression wave on Hoffman2. RUN BY A HUMAN, on the
# cluster, from the deployment root:
#
#   cd /u/scratch/j/jamesdc1/harxhar-subsection && bash cluster/submit_subsection.sh
#
# One canary task (15:30-16:00 bar, ridge, 500-day window) goes first; the
# fleet is held on it and every fleet task refuses to run unless the canary
# wrote its DONE flag.  A scoring job is held on the whole fleet.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
SEGS=cluster/subsection_tasks_segments.txt
POOL=cluster/subsection_tasks_pooled.txt
FLAG=results/linear_subsection/baseline/bar1600/ridge/tw500/DONE
rm -f "$FLAG"

CAN=$(qsub -terse -N subsec_canary -t 1-1 -v TASKFILE=$SEGS cluster/subsection_array.sh | cut -d. -f1)
A=$(qsub -terse -N subsec_a -hold_jid "$CAN" -t 1-100 \
      -v TASKFILE=$SEGS,OFFSET=0,CANARY_FLAG=$FLAG cluster/subsection_array.sh | cut -d. -f1)
B=$(qsub -terse -N subsec_b -hold_jid "$CAN" -t 1-2 \
      -v TASKFILE=$SEGS,OFFSET=100,CANARY_FLAG=$FLAG cluster/subsection_array.sh | cut -d. -f1)
# the six pooled full-series arms are the long ones
P=$(qsub -terse -N subsec_pool -hold_jid "$CAN" -t 1-6 -l h_rt=12:00:00,h_data=16G \
      -v TASKFILE=$POOL,OFFSET=0,CANARY_FLAG=$FLAG cluster/subsection_array.sh | cut -d. -f1)
S=$(qsub -terse -N subsec_score -hold_jid "$A,$B,$P" cluster/subsection_score.sh | cut -d. -f1)

echo "canary=$CAN fleet=$A,$B pooled=$P score=$S" | tee logs/submitted.txt
qstat -u "$USER" | tail -8
