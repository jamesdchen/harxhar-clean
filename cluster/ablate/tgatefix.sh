#!/bin/bash
# Addendum D, gate FT: on the FOMC-patched full cache, the first and last 10 origins of LightGBM and XGBoost (full set,
# and the HAR benchmark, which reads no FOMC column) at 8 threads, three lines in parallel per class -> gate/fix_<TAG>/,
# then ablate_trees.py gate1 against fomcfix's forecasts_gb.parquet (full) and the templates (HAR).  WANT: CPU regex.
set -uo pipefail
B=/u/scratch/j/jamesdc1/abl
TAG=${TAG:?}; WANT=${WANT:?}
CPU=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//; s/ *$//')
echo "tgatefix $TAG job ${SLURM_JOB_ID:-${JOB_ID:-}} on $(hostname) [$CPU] $(date)"
echo "$CPU" | grep -qE "$WANT" || { echo "CPU '$CPU' is not '$WANT': exit 99"; exit 99; }
O=$B/gate/fix_$TAG
mkdir -p $O/logs
for e in xgb lgbm; do
  for a in full har; do
    NT=8 bash $B/code/cluster/ablate/py.sh gb experiments/ablate_trees.py walk --arm $a --est $e --ends 10 \
      --full $B/designs_fix/full --out $O --threads 8 > $O/logs/$e.$a.log 2>&1 &
  done
  wait
done
NT=1 bash $B/code/cluster/ablate/py.sh gb experiments/ablate_trees.py gate1 --out $O --est xgb,lgbm \
  --template-gb $B/code/results/fomcfix_2026-10-26/forecasts/forecasts_gb.parquet
rc=$?
echo "tgatefix $TAG done rc $rc $(date)"
exit $rc
