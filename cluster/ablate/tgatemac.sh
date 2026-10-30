#!/bin/bash
# 2026-10-29 addendum L, gate GMT on one Hoffman2 CPU model (as tgatefix.sh, gate FT): on the macro cache
# (designs_mac/full_mac; its X md5 = MACX checked first), for each class of EST, ablate_trees.py walk of the arms full
# (the 564 columns) and har at the first and last 10 origins and full_mac (780) at the first 10 (the T2 piece of the
# macro columns), 8 threads, one after the other -> gate/mac/<TAG>/, then ablate_trees.py gatemac: the pieces read the
# macro cache; full / har pred_adj text-identical to fomcfix forecasts_gb.parquet / the templates, true_adj and B equal;
# full_mac finite, true_adj and B equal -> gate/mac/<TAG>/passed_<est> (the CPU model); exit 3 if a class fails.  The
# tree pools of addendum L are pool4.sh with REQGATE=$B/gate/mac (a line needs the node's CPU model in T1's
# gate/*/passed_<est> and in this gate).  env: TAG, WANT (CPU regex, else exit 99), MACX (default: the X.npy line of
# the cache's MD5SUMS, CARC's manifest), EST ("+"-separated, default
# xgb+lgbm: both classes, so a class can move between clusters without a new gate), TFULL (default designs_mac/full_mac).
set -uo pipefail
B=/u/scratch/j/jamesdc1/abl
TAG=${TAG:?}; WANT=${WANT:?}
EST=${EST:-xgb+lgbm}
TFULL=${TFULL:-$B/designs_mac/full_mac}
MACX=${MACX:-$(awk '$2 == "X.npy" {print $1}' "$TFULL/MD5SUMS")}
[ -n "$MACX" ] || { echo "no MACX and no X.npy line in $TFULL/MD5SUMS: exit 2"; exit 2; }
CPU=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//; s/ *$//')
echo "tgatemac $TAG job ${SLURM_JOB_ID:-${JOB_ID:-}} on $(hostname) [$CPU] $(date); design $TFULL; classes $EST"
echo "$CPU" | grep -qE "$WANT" || { echo "CPU '$CPU' is not '$WANT': exit 99"; exit 99; }
m=$(md5sum "$TFULL/X.npy" | cut -d' ' -f1)
[ "$m" = "$MACX" ] || { echo "X md5 $m != MACX $MACX: exit 2"; exit 2; }
O=$B/gate/mac/$TAG
mkdir -p $O/logs
# 2026-10-30 addendum L (iii): EST=rf_skx is the forest's gate, run alone: env skx, pieces har and add_fomc at the
# first and last 10 origins and full_mac at the first 10 (ablate_trees.py MAC_GATE_RF), then gatemac --est rf_skx
GATE_ARMS="full:ends har:ends full_mac:first"; ENV=gb
[ "$EST" = rf_skx ] && { GATE_ARMS="har:ends add_fomc:ends full_mac:first"; ENV=skx; }
for e in ${EST//+/ }; do
  for at in $GATE_ARMS; do
    a=${at%%:*}
    sel=--${at##*:}
    NT=8 bash $B/code/cluster/ablate/py.sh $ENV experiments/ablate_trees.py walk --arm $a --est $e $sel 10 \
      --full $TFULL --out $O --threads 8 > $O/logs/$e.$a.log 2>&1
    echo "$e $a $sel 10 walk rc $? $(date)"
  done
done
NT=1 bash $B/code/cluster/ablate/py.sh $ENV experiments/ablate_trees.py gatemac --out $O --est ${EST//+/,} \
  --template-gb $B/code/results/fomcfix_2026-10-26/forecasts/forecasts_gb.parquet
rc=$?
echo "tgatemac $TAG done rc $rc $(date)"
exit $rc
