#!/bin/bash
# 2026-10-28 addendum J (e), gate FC: tgatefix.sh (gate FT) for the CARC share, LightGBM only (the one class walked on
# CARC): on the FOMC-patched full cache, LightGBM's first and last 10 origins (the full set, and the HAR benchmark, which
# reads no FOMC column) at 8 threads, one walk after the other -> gate/<TAG>/, then ablate_trees.py gate1 --est lgbm
# against fomcfix's forecasts_gb.parquet (full set: Table 1's patched LightGBM, itself walked on CARC EPYC) and the
# templates (HAR): text-identical pred_adj, equal true_adj and B -> gate/<TAG>/passed_lgbm (the CPU model), exit 3 if not.
# env: ABL_B (staging), TFULL (patched full cache), TAG, WANT (CPU regex), PYSH (default code/cluster/ablate/py_carc.sh)
set -uo pipefail
B=${ABL_B:?}; TFULL=${TFULL:?}; TAG=${TAG:?}; WANT=${WANT:?}
PYSH=${PYSH:-$B/code/cluster/ablate/py_carc.sh}
CPU=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//; s/ *$//')
echo "tgate_carc $TAG job ${SLURM_JOB_ID:-} on $(hostname) [$CPU] $(date); design $TFULL"
echo "$CPU" | grep -qE "$WANT" || { echo "CPU '$CPU' is not '$WANT': exit 99"; exit 99; }
O=$B/gate/$TAG
mkdir -p $O/logs
for a in full har; do
  NT=8 bash $PYSH gb experiments/ablate_trees.py walk --arm $a --est lgbm --ends 10 --full $TFULL --out $O \
    --threads 8 > $O/logs/lgbm.$a.log 2>&1 || echo "walk $a rc $?"
done
NT=1 bash $PYSH gb experiments/ablate_trees.py gate1 --out $O --est lgbm \
  --template-gb $B/code/results/fomcfix_2026-10-26/forecasts/forecasts_gb.parquet
rc=$?
echo "tgate_carc $TAG done rc $rc $(date)"
exit $rc
