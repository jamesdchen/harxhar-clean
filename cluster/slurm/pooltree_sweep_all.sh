#!/bin/bash
# Pooled forest all_features (2026-10-13): sweep EVERY all_features piece array listed in logs/submitted_pooltrees.txt
# (waves w*, re-runs wsweep*, scored wscore* / wrest*, lmsweep*) with cluster/slurm/pooltree_sweep.sh, drop pieces already
# re-run by an earlier sweep (logs/swept_registry.txt), and submit what is left as ONE re-run array: one piece a task,
# 20 CPUs, 7G, 1:00 (re-runs are often the pieces whose alignment-gate widening made them slow), nice NICE (default 95).
#   NAME=wsweep9 bash cluster/slurm/pooltree_sweep_all.sh      (on CARC, no conda env)
set -euo pipefail
cd "$(dirname "$0")/../.."
NAME=${NAME:?NAME, e.g. wsweep9}
R=results/pooled_trees_2026-10-07
OUT=cluster/pooltree_pieces_rf_all_features_$NAME.txt
[ -f $OUT ] && { echo "$OUT exists; refusing"; exit 1; }
REG=logs/swept_registry.txt; touch $REG
W=logs/sweepall_$NAME; mkdir -p $W; : > $W/all
awk -F'[= ]' '/pieces_pooltree_pieces_rf_all_features_/ {n = $2; sub(/^pieces_pooltree_pieces_rf_all_features_/, "", n); sub(/_[^_]*$/, "", n); print n, $3}' logs/submitted_pooltrees.txt \
  | awk '$1 ~ /^w|^lmsweep/' | sort -u -k2,2 > $W/arrays   # only the 2026-10-09+ arrays (older ones were re-queued by waves)
while read -r n j; do
  tf=cluster/pooltree_pieces_rf_all_features_$n.txt
  [ -f $tf ] || continue
  rm -f $W/s_$j.txt
  bash cluster/slurm/pooltree_sweep.sh $j $tf $W/s_$j.txt 1 > /dev/null 2>&1 || true
  [ -s $W/s_$j.txt ] && awk '{print $3}' $W/s_$j.txt | tr , '\n' >> $W/all
done < $W/arrays
sort -u $W/all | grep -vxF -f <(awk '{print $3}' $REG 2>/dev/null || true) > $W/new || true
# waves run at 6G since 2026-10-14 00:20 (max MaxRSS 4.7 GB); re-runs at 7G, so an OOM piece goes back at 7G -- report OOMs
OOM=$(while read -r n j; do sacct -X -n -P -j $j -o JobID,State 2>/dev/null | grep -c OUT_OF_ME || true; done < $W/arrays | paste -sd+ - | bc)
echo "out-of-memory tasks in the swept arrays: ${OOM:-0}"
echo "swept $(wc -l < $W/arrays) arrays; unfinished pieces $(sort -u $W/all | wc -l); new (not re-run before) $(wc -l < $W/new)"
[ -s $W/new ] || exit 0
sed 's/^/rf all_features /' $W/new > $OUT
cat $OUT >> $REG
md5sum $OUT
CLASSES="epyc-7513|epyc-7542" CPUS=20 MEM=${MEM:-7G} TIME=01:00:00 NICE=${NICE:-95} CANARY_FLAG=$R/canary_ok_rf_all_features TAG=$NAME \
  bash cluster/slurm/submit_pooltrees.sh pieces rf all_features $OUT > /dev/null 2>&1 || true
tail -1 logs/submitted_pooltrees.txt
