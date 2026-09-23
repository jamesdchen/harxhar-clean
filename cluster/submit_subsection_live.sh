#!/bin/bash
# The LIVE-FEASIBLE bucket (src.data.loading.SUBGROUPS["live_feasible"]: the 16
# columns a 15:30 forecaster can rebuild from ES minute bars, the Cboe volatility
# indices and the release calendar) through the subsection campaign: the 13
# one-bar arms + the whole-session arm ("rth") + the pooled arm, for ridge and the
# elastic net under the wave-2 root and the fixed recursive lasso under the
# lassofix root.  Canary first (lasso x 500 x the 15:30-16:00 bar, the fragile
# estimator on the bar that matters); everything else is held on it and refuses
# to run unless it wrote DONE; the scorer is held on all of it.
#
#   cd /u/scratch/j/jamesdc1/harxhar-subsection && bash cluster/submit_subsection_live.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
CANARY=cluster/livefeasible_tasks_canary.txt
RE=cluster/livefeasible_tasks_ridge_enet.txt
LA=cluster/livefeasible_tasks_lasso.txt
LROOT=results/linear_subsection_lassofix
FLAG=$LROOT/live_feasible/bar1600/reclasso/tw500/DONE
rm -f "$FLAG"
SUBMIT=qsub
CAN=$($SUBMIT -terse -N lf_canary -t 1-1 -v TASKFILE=$CANARY,RESULTS_ROOT=$LROOT cluster/subsection_pack.sh | cut -d. -f1)
A=$($SUBMIT -terse -N lf_re -hold_jid "$CAN" -t 1-"$(wc -l < "$RE")" -l h_rt=24:00:00,h_data=16G \
      -v TASKFILE=$RE,CANARY_FLAG=$FLAG cluster/subsection_pack.sh | cut -d. -f1)
L=$($SUBMIT -terse -N lf_lasso -hold_jid "$CAN" -t 1-"$(wc -l < "$LA")" -l h_rt=24:00:00,h_data=16G \
      -v TASKFILE=$LA,CANARY_FLAG=$FLAG,RESULTS_ROOT=$LROOT cluster/subsection_pack.sh | cut -d. -f1)
S=$($SUBMIT -terse -N lf_score -hold_jid "$A,$L" cluster/subsection_score_live.sh | cut -d. -f1)
echo "canary=$CAN ridge_enet=$A lasso=$L score=$S" | tee logs/submitted_live.txt
qstat -u "$USER" | tail -8
