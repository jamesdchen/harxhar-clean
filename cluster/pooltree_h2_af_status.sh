#!/bin/bash
# Pooled forest all_features on Hoffman2 (2026-10-09): pieces done (npz with .node), tasks running / queued per array,
# and task logs that end in FAILED (one line each).  Run on a Hoffman2 login node (qstat), e.g. from this machine:
#   ssh h2dtn "ssh login2 'bash -lc \"bash /u/scratch/j/jamesdc1/harxhar-pooltrees-h2/cluster/pooltree_h2_af_status.sh\"'"
cd "$(dirname "$0")/.."
P=results/pooled_trees_2026-10-07/chunks/pooltree_rf_all_features/pieces
ARRAYS=${ARRAYS:-"15100611 15101135 15101593 15112859 15114407"}
echo "$(date +%F_%T) h2 all_features pieces done: $(ls $P/*.node 2>/dev/null | wc -l)"
qstat -u "$USER" 2>/dev/null | tail -n +3 | awk '{print $1, $5}' | sort | uniq -c | sed 's/^/  /'
for j in $ARRAYS; do
  n=$(grep -l '^FAILED' logs/*.o$j.* 2>/dev/null | wc -l)
  [ "$n" -gt 0 ] && echo "  $j: $n task logs end in FAILED (the 6G / old-runner ones are known and re-run)"
done
