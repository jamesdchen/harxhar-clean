#!/bin/bash
# Wave 2 of the subsection regressions: the nine wider feature buckets.
# Run on Hoffman2 from the deployment root:
#
#   cd /u/scratch/j/jamesdc1/harxhar-subsection && bash cluster/submit_subsection_wave2.sh
#
# Canary first (all_features x recursive lasso x the 15:30-16:00 bar, the widest
# design on the most fragile estimator); the packs and the pooled arms are held
# on it and refuse to run unless it wrote DONE; the scorer is held on both.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
PACKS=cluster/wave2_tasks_packs.txt
POOL=cluster/wave2_tasks_pooled.txt
CANARY=cluster/wave2_tasks_canary.txt
FLAG=results/linear_subsection/all_features/bar1600/reclasso/tw500/DONE
rm -f "$FLAG"
NP=$(wc -l < "$PACKS")
NQ=$(wc -l < "$POOL")

CAN=$(qsub -terse -N w2_canary -t 1-1 -v TASKFILE=$CANARY cluster/subsection_pack.sh | cut -d. -f1)
A=$(qsub -terse -N w2_packs -hold_jid "$CAN" -t 1-"$NP" \
      -v TASKFILE=$PACKS,CANARY_FLAG=$FLAG cluster/subsection_pack.sh | cut -d. -f1)
P=$(qsub -terse -N w2_pooled -hold_jid "$CAN" -t 1-"$NQ" -l h_rt=24:00:00,h_data=16G \
      -v TASKFILE=$POOL,CANARY_FLAG=$FLAG cluster/subsection_pack.sh | cut -d. -f1)
S=$(qsub -terse -N w2_score -hold_jid "$A,$P" cluster/subsection_score_wave2.sh | cut -d. -f1)

echo "canary=$CAN packs=$A pooled=$P score=$S" | tee logs/submitted_wave2.txt
qstat -u "$USER" | tail -8
