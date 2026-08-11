#!/bin/bash
# Submit all 20 tree expert arms on Hoffman2 (100 chunks each, idempotent
# resume). REQUIRES the frozen menu experiments/tree_menu.json.
set -e
cd /u/scratch/j/jamesdc1/harxhar-clean
[ -f experiments/tree_menu.json ] || { echo "ABORT: experiments/tree_menu.json absent — freeze first"; exit 1; }
for k in $(seq -w 0 19); do
  qsub -N unif_tree_$k -v ARM=tree_expert_$k jobs/sge/unification_tree.sge
done
echo submitted 20 tree expert arrays
