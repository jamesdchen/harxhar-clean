#!/bin/bash
# The new-series design (bucket abl_new) and gate 0b, the linear arms' column-subset caches, the late-start windows
# (RULE.txt sections 2, 5, 7).  Exit 3 on a failed gate (later steps do not run).
set -euo pipefail
B=/u/scratch/j/jamesdc1/abl
P="bash $B/code/cluster/ablate/py.sh gb experiments/ablate_design.py"
echo "prep1 job ${SLURM_JOB_ID:-${JOB_ID:-}} on $(hostname) $(date)"
export NT=8 UNIF_SCALE_PROCS=8
$P newseries --out $B/designs/newser
$P cmpnew --new $B/designs/newser --full $B/designs/full --out $B/gate/gate0b.json
$P subsets --full $B/designs/full --new $B/designs/newser --root $B/designs/lin
$P latestart --full $B/designs/full --out $B/runs/latestart_w3000.csv \
  --rule $B/code/results/window_sweep_2014start_2026-10-22/latestart/latestart_rule_all.csv
echo "prep1 done $(date)"
