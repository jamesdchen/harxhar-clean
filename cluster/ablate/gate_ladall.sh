#!/bin/bash
# 2026-10-28 addendum K: the every-bar ladder's gate on one Hoffman2 CPU model.  The rung code path (ablate_trees.py
# walk on lrung_<est>_0 = the HAR set and lrung_<est>_46 = the full set, column subsets of designs_fix/full) at the
# first 10 origins, LightGBM and XGBoost (8 threads, one after the other), against Table 1's every-bar walks: the HAR
# walk (the templates' baseline rows) and the FOMC-patched full walk (fomcfix forecasts_gb.parquet): pred_adj
# text-identical, true_adj and B equal (ablate_ladall.py gate) -> gate/ladall/<TAG>/passed_<est>; exit 3 if a class
# fails.  WANT: a regex the node's CPU model must match (else exit 99).
set -uo pipefail
B=/u/scratch/j/jamesdc1/abl
TAG=${TAG:?}; WANT=${WANT:?}
CPU=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//; s/ *$//')
echo "gate_ladall $TAG job ${SLURM_JOB_ID:-${JOB_ID:-}} on $(hostname) [$CPU] $(date)"
echo "$CPU" | grep -qE "$WANT" || { echo "CPU '$CPU' is not '$WANT': exit 99"; exit 99; }
O=$B/gate/ladall/$TAG
mkdir -p $O/logs
for e in lgbm xgb; do
  for k in 0 46; do
    NT=8 bash $B/code/cluster/ablate/py.sh gb experiments/ablate_trees.py walk --arm lrung_${e}_$k --est $e --first 10 \
      --full $B/designs_fix/full --out $O --threads 8 > $O/logs/${e}_$k.log 2>&1
    echo "$e k=$k walk rc $? $(date)"
  done
done
NT=1 bash $B/code/cluster/ablate/py.sh gb experiments/ablate_ladall.py gate --out $O --est lgbm,xgb
rc=$?
echo "gate_ladall $TAG done rc $rc $(date)"
exit $rc
