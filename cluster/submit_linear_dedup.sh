#!/bin/bash
# I1 (2026-09-29): every per-bar 16:00 linear forecast the closing-strategy master table scores,
# re-run on the de-duplicated per-bar design, on Hoffman2 (SGE), from the deployment root
# cluster/ship_linear_dedup_hoffman2.sh fills:
#
#   cd /u/scratch/j/jamesdc1/harxhar-dedup && bash cluster/submit_linear_dedup.sh
#
# canary (cluster/linear_dedup_canary.sh: baseline x lasso x bar1600 must reproduce the pre-dedup
# lasso and OLS forecasts; writes CANARY_OK) -> the fleet array (one task per line of
# cluster/linear_dedup_tasks_fleet.txt), held on the canary and refusing to run without CANARY_OK
# (the canary's arm carries DONE, so the fleet skips it) -> a collector held on the fleet.
# REFUSES a second submission while logs/submitted_linear_dedup.txt exists (jobs cannot be
# cancelled from here): delete it by hand to resubmit; DONE arms are skipped, so it resumes.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -z "${SGE_ROOT:-}" ]; then set +u; source /u/systems/UGE8.6.4/hoffman2/common/settings.sh; set -u; fi
mkdir -p logs results/linear_subsection_dedup
FLEET=cluster/linear_dedup_tasks_fleet.txt
FLAG=results/linear_subsection_dedup/CANARY_OK
STAMP=logs/submitted_linear_dedup.txt
if [ -f "$STAMP" ]; then
  echo "already submitted ($(cat "$STAMP")); refusing a second submission -- delete $STAMP by hand to resubmit"
  exit 1
fi
for f in specs/causal_tune_linear.py src/backtest/executor.py cluster/linear_dedup_pack.sh \
         cluster/linear_dedup_canary.sh cluster/linear_dedup_collect.sh "$FLEET" \
         cluster/linear_dedup_tasks_canary.txt cluster/linear_dedup_ref_lasso_baseline_bar1600.csv \
         cluster/linear_dedup_ref_ols_baseline_bar1600.csv; do
  [ -f "$f" ] || { echo "$f missing: run cluster/ship_linear_dedup_hoffman2.sh (locally) first"; exit 1; }
done
rm -f "$FLAG" results/linear_subsection_dedup/FLEET_DONE

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

N=$(wc -l < "$FLEET")
CAN=$(submit -N lin_dedup_canary cluster/linear_dedup_canary.sh)
FL=$(submit -N lin_dedup_fleet -hold_jid "$CAN" -t 1-"$N" \
      -v TASKFILE=$FLEET,CANARY_FLAG=$FLAG cluster/linear_dedup_pack.sh)
CO=$(submit -N lin_dedup_collect -hold_jid "$FL" cluster/linear_dedup_collect.sh)
echo "canary=$CAN fleet=$FL (1-$N) collect=$CO $(date)" | tee "$STAMP"
qstat -u "$USER" | tail -6
