#!/bin/bash
# Launch the dev-span LightGBM menu study (tree expert bank): one 8-task
# array, each worker 8 cpus x 6h, all appending to the shared Optuna journal.
# After completion: freeze the menu with
#   python experiments/tree_menu_dev.py --freeze \
#       --journal /scratch1/jc_905/harxhar-clean/results/tree_menu_journal.log
# Run ON the CARC login node from /scratch1/jc_905/harxhar-clean.
set -e
cd /scratch1/jc_905/harxhar-clean
sbatch jobs/slurm/tree_menu.sbatch
squeue -u "$USER" -h | wc -l
