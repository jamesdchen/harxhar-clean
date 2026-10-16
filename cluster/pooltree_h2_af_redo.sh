#!/bin/bash
# Pooled forest all_features on Hoffman2 (2026-10-09): re-run whatever Hoffman2 owes -- every piece of its plans
# (cluster/pooltree_pieces_rf_all_features_{h2,h2r,h2r2,h2r3,h2b}.txt: chunks 85-92 and 98-99) that has no npz here
# and is not in a live task.  Run it when the arrays listed in LIVE have no queued / running task for those pieces;
# by default it REFUSES while any task of $LIVE is still queued or running.  Writes cluster/pooltree_pieces_rf_
# all_features_h2redo<N>.txt (one 5-row piece a line) and submits it at 8G / 4:30 on E5-2670 | E5-2650v2.
#   N=1 bash cluster/pooltree_h2_af_redo.sh      (on a Hoffman2 login node, no conda env active)
set -euo pipefail
cd "$(dirname "$0")/.."
N=${N:?N (redo number)}
LIVE=${LIVE:-"15100611 15101135 15101593 15112859 15114407"}
P=results/pooled_trees_2026-10-07/chunks/pooltree_rf_all_features/pieces
OUT=cluster/pooltree_pieces_rf_all_features_h2redo$N.txt
[ -f $OUT ] && { echo "$OUT exists; refusing"; exit 1; }
for j in $LIVE; do
  q=$(qstat -u "$USER" 2>/dev/null | awk -v j=$j '$1 == j' | wc -l)
  [ "$q" = 0 ] || [ "${FORCE:-0}" = 1 ] || { echo "array $j still has queued / running tasks; refusing (FORCE=1 overrides)"; exit 1; }
done
cat cluster/pooltree_pieces_rf_all_features_h2.txt cluster/pooltree_pieces_rf_all_features_h2b.txt | awk '{print $3}' | sort -u > logs/h2redo_plan_$N
: > $OUT
while IFS=: read -r k a b; do
  [ -f $(printf '%s/chunk_%03d_%04d-%04d.npz' $P $k $a $b) ] || echo "rf all_features $k:$a:$b" >> $OUT
done < logs/h2redo_plan_$N
echo "$OUT: $(wc -l < $OUT) pieces without an npz"; md5sum $OUT
[ -s $OUT ] || exit 0
[ "${DRY:-0}" = 1 ] && { echo "DRY: not submitted"; rm -f $OUT; exit 0; }
H_DATA=8G H_RT=4:30:00 bash cluster/submit_pooltrees_h2.sh pieces $OUT "intel-E5-2670|intel-E5-2650v2"
