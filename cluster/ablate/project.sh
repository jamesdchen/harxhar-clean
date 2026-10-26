#!/bin/bash
# After the tree gates: T2 across the gated CPU models and the projection of every tree line (RULE.txt section 6) ->
# runs/gate2.json, runs/projection.json, runs/rf_skip.txt; on a projection within the cap, runs/projection.done (the
# forest lines become eligible).  GATES: comma-separated gate dirs.
set -uo pipefail
B=/u/scratch/j/jamesdc1/abl
GATES=${GATES:?}
echo "project job ${SLURM_JOB_ID:-${JOB_ID:-}} on $(hostname) $(date)"
P="bash $B/code/cluster/ablate/py.sh gb experiments/ablate_trees.py"
NT=1 $P gate2 --gates $GATES --out $B/runs/gate2.json
NT=1 $P project --gates $GATES --tasks $B/code/results/ablations_2014start_w3000/tasks/trees.txt \
  --out $B/runs/projection.json --cap 15000
rc=$?
[ $rc -eq 0 ] && date > $B/runs/projection.done
echo "project done rc $rc $(date)"
exit $rc
