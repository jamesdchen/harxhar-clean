#!/bin/bash
# I6 (2026-09-29): the direct rest-of-day model (specs/causal_tune_rest_of_day.py) re-run on the
# de-duplicated per-bar design (commit 47f7f9c), on Hoffman2 (SGE), from the deployment root
# cluster/ship_restofday_dedup_hoffman2.sh fills:
#
#   cd /u/scratch/j/jamesdc1/harxhar-restofday-dedup && bash cluster/submit_restofday_dedup.sh
#
# 117 arms = 13 entry clocks (09:30 .. 15:30) x {ridge, reclasso, reclasticnet} x {all_features,
# baseline, live_feasible} at 2000 sessions: the master table's check rows need only the nine
# 15:30 arms, but experiments/build_restofday_yhat.py writes a yhat_restofday_* table only when
# all 13 clocks of an estimator x bucket exist (the canonical tables carry all 13 for the
# hold-to-close trading test), so the minimum that rebuilds the tables is every clock.
#
# canary (cluster/restofday_dedup_canary.sh: live_feasible x ridge x 15:30 must reproduce agent A's
# de-duplicated per-bar 16:00 arm; writes CANARY_OK) -> three 15:30 arrays and one fleet array,
# held on the canary and refusing to run without CANARY_OK -> a collector held on all four.
#
# The nine 15:30 arms run on the CPU architecture their per-bar 16:00 twin ran on in agent A's
# fleet (14970158/14970159: the host in results/linear_subsection_dedup/logs/lin_dedup_*.o* ->
# qhost -F arch): lasso / elastic-net float paths depend on the machine (A's control_attribution:
# up to ~1e-1 relative across architectures, bit-identical on one), so the check rows then measure
# the model, not the machine.  One file per architecture, cluster/restofday_dedup_tasks_1530_<arch>.txt
# (the canary's twin ran on intel-gold-6140).  The other 108 arms have no twin and run anywhere.
# REFUSES a second submission while logs/submitted_restofday_dedup.txt exists (jobs cannot be
# cancelled from here): delete it by hand to resubmit; DONE arms are skipped, so it resumes.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -z "${SGE_ROOT:-}" ]; then set +u; source /u/systems/UGE8.6.4/hoffman2/common/settings.sh; set -u; fi
R=results/linear_subsection_restofday_dedup
FLEET=cluster/restofday_dedup_tasks_fleet.txt
FLAG=$R/CANARY_OK
STAMP=logs/submitted_restofday_dedup.txt
MANIFEST=cluster/restofday_dedup_manifest.md5
CANARY_ARCH=intel-gold-6140
RES="h_rt=1:00:00,h_data=16G"
mkdir -p logs "$R"
if [ -f "$STAMP" ]; then
  echo "already submitted ($(cat "$STAMP")); refusing a second submission -- delete $STAMP by hand to resubmit"
  exit 1
fi
for f in specs/causal_tune_rest_of_day.py specs/causal_tune_linear.py src/backtest/executor.py \
         experiments/check_restofday_identity.py cluster/restofday_dedup_pack.sh \
         cluster/restofday_dedup_canary.sh cluster/restofday_dedup_collect.sh "$FLEET" \
         cluster/restofday_dedup_tasks_canary.txt "$MANIFEST"; do
  [ -f "$f" ] || { echo "$f missing: run cluster/ship_restofday_dedup_hoffman2.sh (locally) first"; exit 1; }
done
grep -q "session_edge" src/backtest/executor.py || { echo "this root does not hold the de-dup executor"; exit 1; }
md5sum -c --quiet "$MANIFEST" || { echo "manifest check failed: nothing submitted"; exit 1; }
echo "manifest ok: $(wc -l < "$MANIFEST") pinned files"
rm -f "$FLAG" "$R/FLEET_DONE"

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

CAN=$(submit -N rod_dedup_canary -l "$RES,arch=$CANARY_ARCH" cluster/restofday_dedup_canary.sh)
HOLD=""
IDS=""
for F in cluster/restofday_dedup_tasks_1530_*.txt; do
  ARCH=$(basename "$F" .txt); ARCH=${ARCH#restofday_dedup_tasks_1530_}
  N=$(wc -l < "$F")
  J=$(submit -N rod_dedup_1530 -hold_jid "$CAN" -t 1-"$N" -l "$RES,arch=$ARCH" \
        -v TASKFILE="$F",CANARY_FLAG="$FLAG" cluster/restofday_dedup_pack.sh)
  IDS="$IDS 1530/$ARCH=$J(1-$N)"
  HOLD="$HOLD,$J"
done
N=$(wc -l < "$FLEET")
FL=$(submit -N rod_dedup_fleet -hold_jid "$CAN" -t 1-"$N" -l "$RES" \
      -v TASKFILE="$FLEET",CANARY_FLAG="$FLAG" cluster/restofday_dedup_pack.sh)
CO=$(submit -N rod_dedup_collect -hold_jid "${HOLD#,},$FL" -l "h_rt=0:20:00,h_data=2G" \
      cluster/restofday_dedup_collect.sh)
echo "canary=$CAN$IDS fleet=$FL(1-$N) collect=$CO $(date)" | tee "$STAMP"
qstat -u "$USER" | tail -8
