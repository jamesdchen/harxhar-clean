#!/bin/bash
# Launch the mixed-family tree menu study on Hoffman2: warmup job builds the
# panel cache on a compute node (mandatory — login ulimits), then the 8-worker
# array holds on it. NOTE the hold is advisory only: SGE releases holds even
# when the held-on job fails, so the workers re-verify the cache themselves
# and exit nonzero rather than rebuild (see jobs/sge/tree_menu.qsub).
# Freeze afterwards (any node):
#   python experiments/tree_menu_dev.py --freeze \
#     --journal-lgbm results/tree_menu_journal_lgbm.log \
#     --journal-xgb  results/tree_menu_journal_xgb.log
# Run ON the Hoffman2 login node from /u/scratch/j/jamesdc1/harxhar-clean.
set -e
cd /u/scratch/j/jamesdc1/harxhar-clean
mkdir -p logs
J0=$(qsub -terse jobs/sge/tree_menu_warmup.qsub | cut -d. -f1)
qsub -hold_jid "$J0" jobs/sge/tree_menu.qsub
echo "SGE_SUBMITTED tree_menu warmup=$J0 (workers hold on it + self-verify)"
qstat -u "$USER" | tail -n +3 | wc -l
