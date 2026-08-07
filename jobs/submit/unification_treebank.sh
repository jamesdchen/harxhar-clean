#!/bin/bash
# Submit the 20 tree expert arms (frozen menu; see src/unification.py and
# experiments/tree_menu.json). 20 arms x 100 chunks, 8 cpus/task per the QOS
# ruling. REQUIRES the frozen menu — refuses to submit without it, matching
# the executor's own loud failure. Run ON the CARC login node from
# /scratch1/jc_905/harxhar-clean.
set -e
cd /scratch1/jc_905/harxhar-clean
[ -f experiments/tree_menu.json ] || { echo "ABORT: experiments/tree_menu.json absent — freeze the menu first"; exit 1; }
for k in $(seq -w 0 19); do
  arm="tree_expert_$k"
  sbatch --export=ALL,ARM="$arm" --job-name="unif_$arm" jobs/slurm/unification_tree.sbatch
done
squeue -u "$USER" -h | wc -l
