#!/bin/bash
# The regular session as ONE segment ("rth": the 13 bars 09:30-16:00 pooled), beside
# the one-bar and block arms already on disk.  Canary first (baseline ridge 500);
# the ridge / elastic-net packs run under the wave-2 root and the fixed lasso under
# the lassofix root, both held on the canary and refusing to run unless it wrote DONE.
# Score afterwards with the two scorers of the campaign.
#
#   cd /u/scratch/j/jamesdc1/harxhar-subsection && bash cluster/submit_subsection_rth.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
RE=cluster/rth_tasks_ridge_enet.txt
LA=cluster/rth_tasks_lasso.txt
CANARY=cluster/rth_tasks_canary.txt
FLAG=results/linear_subsection/baseline/rth/ridge/tw500/DONE
rm -f "$FLAG"
SUBMIT=qsub
CAN=$($SUBMIT -terse -N rth_canary -t 1-1 -v TASKFILE=$CANARY cluster/subsection_pack.sh | cut -d. -f1)
A=$($SUBMIT -terse -N rth_re -hold_jid "$CAN" -t 1-"$(wc -l < "$RE")" \
      -v TASKFILE=$RE,CANARY_FLAG=$FLAG cluster/subsection_pack.sh | cut -d. -f1)
L=$($SUBMIT -terse -N rth_lasso -hold_jid "$CAN" -t 1-"$(wc -l < "$LA")" \
      -v TASKFILE=$LA,CANARY_FLAG=$FLAG,RESULTS_ROOT=results/linear_subsection_lassofix cluster/subsection_pack.sh | cut -d. -f1)
echo "canary=$CAN ridge_enet=$A lasso=$L" | tee logs/submitted_rth.txt
qstat -u "$USER" | tail -6
