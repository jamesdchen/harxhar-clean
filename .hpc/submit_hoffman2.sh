#!/bin/bash
# Submit an analysis run to Hoffman2 as two dependent arrays.
#
#   ./.hpc/submit_hoffman2.sh cadence_x_design jc_905@hoffman2.idre.ucla.edu \
#        /u/scratch/j/jc_905/harxhar-clean
#
# Phase 1 picks lambda once per design-by-cadence; phase 2 runs the chunked
# walk with that lambda fixed.  Selecting inside each chunk would let lambda
# vary along the panel, which is a different estimator from the one being
# compared, so the dependency is a correctness requirement and not just
# scheduling.
set -euo pipefail
RUN="${1:-cadence_x_design}"
TARGET="${2:?usage: submit_hoffman2.sh <run> <user@host> <remote_path>}"
RPATH="${3:?remote path}"

NSEL=$(python analysis/taskrunner.py --run "$RUN" --phase sel --manifest | sed -n 's/.*cells=\([0-9]*\).*/\1/p' | head -1)
NWALK=$(python analysis/taskrunner.py --run "$RUN" --phase walk --manifest | sed -n 's/.*cells=\([0-9]*\).*/\1/p' | head -1)
echo "run=$RUN  sel tasks=$NSEL  walk tasks=$NWALK"

rsync -az --delete \
  --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
  --exclude 'logs' --exclude 'results/tasks' \
  ./ "$TARGET:$RPATH/"
# results/ holds the multi-GB caches; send only what the cells actually read
rsync -az --info=progress2 results/b2_mmap_warm "$TARGET:$RPATH/results/"

ssh "$TARGET" bash -lc "'
  cd $RPATH && mkdir -p logs
  JID=\$(qsub -terse -N ${RUN}_sel -t 1-$NSEL \
        -v RUN=$RUN,PHASE=sel,REMOTE_PATH=$RPATH \
        .hpc/templates/analysis_array.uge | cut -d. -f1)
  echo \"sel array \$JID\"
  qsub -N ${RUN}_walk -t 1-$NWALK -hold_jid \$JID \
       -v RUN=$RUN,PHASE=walk,REMOTE_PATH=$RPATH \
       .hpc/templates/analysis_array.uge
'"
echo "submitted. when both arrays finish:"
echo "  rsync -az $TARGET:$RPATH/results/tasks results/"
echo "  python analysis/taskrunner.py --run $RUN --harvest"
