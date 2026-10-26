#!/bin/bash
# Tree gates T1 / T2 pieces on one CPU model (RULE.txt section 5) -> gate/<TAG>/: PHASE=A LightGBM and XGBoost, PHASE=B
# the forest.  Per class, in parallel at 8 threads each (3 x 8 = 24 CPUs): the templates' first and last 10 origins of
# the full set and of the HAR benchmark (the 12 HAR columns of the full cache), and lo_rm's first 10 origins; then
# ablate_trees.py gate1 for the phase's classes (gate/<TAG>/passed_<est> for each class that passes; exit 3 if any
# fails).  WANT: a regex the node's CPU model must match (else exit 99).
set -uo pipefail
B=/u/scratch/j/jamesdc1/abl
TAG=${TAG:?}; PHASE=${PHASE:?}; WANT=${WANT:?}
CPU=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//; s/ *$//')
echo "tgate $TAG phase $PHASE job ${SLURM_JOB_ID:-${JOB_ID:-}} on $(hostname) [$CPU] $(date)"
echo "$CPU" | grep -qE "$WANT" || { echo "CPU '$CPU' is not '$WANT': exit 99"; exit 99; }
O=$B/gate/$TAG
mkdir -p $O/logs
if [ "$PHASE" = A ]; then ESTS="xgb lgbm"; ENV=gb; else ESTS="rf_skx"; ENV=skx; fi
for e in $ESTS; do
  for spec in "full --ends 10" "har --ends 10" "lo_rm --first 10"; do
    set -- $spec
    NT=8 bash $B/code/cluster/ablate/py.sh $ENV experiments/ablate_trees.py walk --arm $1 --est $e $2 $3 \
      --full $B/designs/full --out $O --threads 8 > $O/logs/$e.$1.log 2>&1 &
  done
  wait
  echo "$e lines done $(date)"
done
NT=1 bash $B/code/cluster/ablate/py.sh gb experiments/ablate_trees.py gate1 --out $O --est ${ESTS// /,}
rc=$?
echo "tgate $TAG phase $PHASE done rc $rc $(date)"
exit $rc
