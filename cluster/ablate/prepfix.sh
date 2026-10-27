#!/bin/bash
# Addendum D: gate F0 on the FOMC-patched full cache (designs_fix/full, fomcfix cell B) against the final one, then the
# linear arms' caches from it (designs_fix/lin; classical arms: HAR, calendar, RS+ from it, the new series from abl_new).
set -euo pipefail
B=/u/scratch/j/jamesdc1/abl
P="bash $B/code/cluster/ablate/py.sh gb experiments/ablate_design.py"
echo "prepfix job ${SLURM_JOB_ID:-${JOB_ID:-}} on $(hostname) $(date)"
NT=8 $P checkfix --full $B/designs/full --new $B/designs_fix/full --out $B/gate/gateF0.json
NT=8 $P subsets --full $B/designs_fix/full --new $B/designs/newser --root $B/designs_fix/lin
echo "prepfix done $(date)"
