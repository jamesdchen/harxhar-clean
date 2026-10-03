#!/bin/bash
# 2026-10-03 close study "trees and training-set size": submit the tree arms this container could not
# afford (refit every session; XGBoost / random forest; any arm of the plan left over) as one SGE
# array on Hoffman2.  NOT SUBMITTED by the agent that wrote it (the cloud container cannot reach the
# cluster); run from the laptop.
#
# 1. ship (laptop, Git Bash; code by tar over ssh, the 27 MB design through the data-transfer node):
#      bash cluster/submit_close_trees_datasize_h2.sh ship
# 2. submit (on Hoffman2, in a login shell so the scheduler and conda load):
#      /c/Windows/System32/OpenSSH/ssh.exe -o BatchMode=yes hoffman2 \
#        "bash -lc 'cd /u/scratch/j/jamesdc1/harxhar-subsection && bash cluster/submit_close_trees_datasize_h2.sh submit <tag>'"
# 3. pull the pieces back (laptop), then merge and score locally:
#      bash cluster/submit_close_trees_datasize_h2.sh pull
#      TDS_REFIT_EVERY=1 python experiments/close_trees_datasize.py merge <arm ...>
#      TDS_REFIT_EVERY=1 python experiments/close_trees_datasize.py analyze
#
# The task list is cluster/close_trees_datasize_h2_tasks.txt ("<arm> <refit_every> <chunk> <n_chunks>").
# Scratch (/u/scratch/j/jamesdc1/) is purged: `submit` checks every input against the md5 list the
# ship step wrote (cluster/close_trees_datasize_h2.md5) and refuses to run if one is missing or differs.
# Each task holds one slot (every fit single-threaded).  Env: H_RT (default 6:00:00), H_DATA (default 4G).
# REFUSES a second submission of the same tag (logs/submitted_close_trees_datasize_h2_<tag>.txt).
set -euo pipefail
cd "$(dirname "$0")/.."
H2=/u/scratch/j/jamesdc1/harxhar-subsection
SSH=/c/Windows/System32/OpenSSH/ssh.exe
TASKS=cluster/close_trees_datasize_h2_tasks.txt
MD5=cluster/close_trees_datasize_h2.md5
CODE=(experiments/close_trees_datasize.py experiments/dense_vs_sparse_1530.py experiments/model_diagnostics_1530.py
      specs/causal_tune_trees.py specs/causal_tune_linear.py cluster/close_trees_datasize_h2_task.sh
      cluster/submit_close_trees_datasize_h2.sh "$TASKS")
DESIGN=(results/close_design/_work/design_bar1600_all_features.npz results/close_design/_work/design_last30_all_features.npz)
MODE=${1:?usage: submit_close_trees_datasize_h2.sh ship | submit <tag> | pull}

case "$MODE" in
  ship)
    mapfile -t SRC < <(find src -name '*.py' -not -path '*/__pycache__/*' | sort)
    ALL=("${CODE[@]}" "${SRC[@]}" "${DESIGN[@]}")
    md5sum "${ALL[@]}" > "$MD5"
    tar czf - "${CODE[@]}" "${SRC[@]}" "$MD5" | "$SSH" -o BatchMode=yes hoffman2 "mkdir -p $H2 && cd $H2 && tar xzf -"
    tar cf - "${DESIGN[@]}" | "$SSH" -o BatchMode=yes h2dtn "mkdir -p $H2 && cd $H2 && tar xf -"
    "$SSH" -o BatchMode=yes hoffman2 "cd $H2 && md5sum -c --quiet $MD5" && echo "shipped ${#ALL[@]} files, Hoffman2 md5 == local"
    ;;
  submit)
    TAG=${2:?usage: submit_close_trees_datasize_h2.sh submit <tag>}
    if [ -z "${SGE_ROOT:-}" ]; then set +u; source /u/systems/UGE8.6.4/hoffman2/common/settings.sh; set -u; fi
    mkdir -p logs
    STAMP=logs/submitted_close_trees_datasize_h2_$TAG.txt
    [ -f "$STAMP" ] && { echo "already submitted ($(cat "$STAMP")); delete $STAMP by hand to resubmit"; exit 1; }
    [ -f "$MD5" ] || { echo "$MD5 missing: run the ship step from the laptop"; exit 1; }
    md5sum -c --quiet "$MD5" || { echo "inputs missing or changed on scratch (purged?): re-run the ship step"; exit 1; }
    N=$(grep -c . "$TASKS")
    J=$(qsub -terse -S /bin/bash -N tds_h2 -t 1-"$N" -l "h_rt=${H_RT:-6:00:00},h_data=${H_DATA:-4G}" \
          -v TASKFILE=$TASKS cluster/close_trees_datasize_h2_task.sh)
    echo "close_trees_datasize $TAG: job ${J%%.*}, $N tasks, h_rt=${H_RT:-6:00:00} h_data=${H_DATA:-4G} $(date)" | tee "$STAMP"
    qstat -u "$USER" | tail -5
    ;;
  pull)
    R=results/close_studies_2026-10-03/trees_datasize
    "$SSH" -o BatchMode=yes h2dtn "cd $H2 && find $R -name '*.part*of*.npz' | sort | tar cf - -T -" | tar xf -
    "$SSH" -o BatchMode=yes hoffman2 "cd $H2 && find $R -name '*.part*of*.npz' | sort | xargs md5sum" > "$R/h2_pull.md5"
    md5sum -c --quiet "$R/h2_pull.md5" && echo "pulled $(wc -l < "$R/h2_pull.md5") pieces, md5 == Hoffman2"
    ;;
  *) echo "MODE must be ship, submit or pull"; exit 1 ;;
esac
