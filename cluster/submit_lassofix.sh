#!/bin/bash
# Re-run EVERY recursive-lasso arm of the subsection regressions with the
# between-tune identifiability fix, into a new results root.  Run on Hoffman2
# from the deployment root:
#
#   cd /u/scratch/j/jamesdc1/harxhar-subsection && bash cluster/submit_lassofix.sh
#
# canary (worst offender, must reproduce the fixed-code reference) -> packs of
# one-bar and block arms + pooled arms, all refusing to run without the
# canary's CANARY_HEALTHY flag -> an early scorer for the nine buckets whose
# pooled arms are quick -> a final scorer once the 47-source pooled arms land.
set -euo pipefail
cd "$(dirname "$0")/.."
R=results/linear_subsection_lassofix
mkdir -p logs "$R"
PACKS=cluster/lassofix_tasks_packs.txt
POOL_FAST=cluster/lassofix_tasks_pooled_fast.txt
POOL_SLOW=cluster/lassofix_tasks_pooled_slow.txt
FLAG=$R/CANARY_HEALTHY
V="RESULTS_ROOT=$R,CANARY_FLAG=$FLAG"

submit () {  # the scheduler's submission verifier times out now and then: retry
  local out n=0
  until out=$(qsub -terse "$@" 2>&1); do
    n=$((n + 1))
    echo "submit attempt $n failed: $out" >&2
    [ "$n" -ge 8 ] && return 1
    sleep 30
  done
  echo "${out%%.*}"
}

CAN=$(submit -N lf_canary cluster/lassofix_canary.sh)
A=$(submit -N lf_packs -hold_jid "$CAN" -t 1-"$(wc -l < $PACKS)" \
      -v TASKFILE=$PACKS,$V cluster/subsection_pack.sh)
P1=$(submit -N lf_poolfast -hold_jid "$CAN" -t 1-"$(wc -l < $POOL_FAST)" -l h_rt=12:00:00,h_data=16G \
      -v TASKFILE=$POOL_FAST,$V cluster/subsection_pack.sh)
P2=$(submit -N lf_poolslow -hold_jid "$CAN" -t 1-"$(wc -l < $POOL_SLOW)" -l h_rt=24:00:00,h_data=16G \
      -v TASKFILE=$POOL_SLOW,$V cluster/subsection_pack.sh)
S1=$(submit -N lf_score1 -hold_jid "$A,$P1" \
      -v TAG=early,BUCKETS=baseline:moments:liquidity:market_ew:market_vw:sentiment:implied_vol:vol_demand:fomc \
      cluster/lassofix_score.sh)
S2=$(submit -N lf_score2 -hold_jid "$S1,$P2" -v TAG=final,BUCKETS=all_features \
      cluster/lassofix_score.sh)

echo "canary=$CAN packs=$A pooled_fast=$P1 pooled_slow=$P2 score_early=$S1 score_final=$S2" | tee logs/submitted_lassofix.txt
qstat -u "$USER" | tail -8
