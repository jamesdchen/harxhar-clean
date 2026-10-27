#!/bin/bash
# 2026-10-27 addendum I: the ladder's accuracy gate on one CPU model (results/beyond_trees_2026-10-27/RULE.txt section 0):
# the runner on the BASE arm (add_rm_fomc, the added columns switched off) at 7 origins spread over the span, LightGBM
# and XGBoost (8 threads each, one after the other), compared bit for bit with the merged BASE forecasts
# (runs/trees_rmfomc/forecasts) -> gate/rb/<TAG>/passed_<est>; exit 3 if any class fails.  WANT: a regex the node's
# CPU model must match (else exit 99).
set -uo pipefail
B=/u/scratch/j/jamesdc1/abl
TAG=${TAG:?}; WANT=${WANT:?}
CPU=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//; s/ *$//')
echo "gatebase $TAG job ${SLURM_JOB_ID:-${JOB_ID:-}} on $(hostname) [$CPU] $(date)"
echo "$CPU" | grep -qE "$WANT" || { echo "CPU '$CPU' is not '$WANT': exit 99"; exit 99; }
O=$B/gate/rb/$TAG
mkdir -p $O/logs
for e in lgbm xgb; do
  NT=8 bash $B/code/cluster/ablate/py.sh gb experiments/ablate_trees.py walk --arm add_rm_fomc --est $e --spread 7 \
    --full $B/designs_fix/full --out $O --threads 8 > $O/logs/$e.log 2>&1
  echo "$e walk rc $? $(date)"
done
NT=1 bash $B/code/cluster/ablate/py.sh gb experiments/ablate_trees.py gatebase --out $O --est lgbm,xgb --spread 7 \
  --base $B/runs/trees_rmfomc/forecasts
rc=$?
echo "gatebase $TAG done rc $rc $(date)"
exit $rc
