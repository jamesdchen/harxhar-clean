#!/bin/bash
# 2026-10-24 MinGRU v2: submit the every-bar walk (RULE ADDENDUM 2; released by the owner ~10:15 PDT).
# The slice check first (P100; after the two W 3,000 w000 bar-init tasks if they are still queued), then every arm's
# lanes, each file afterok on the slice and on its bar-init task (when still queued).  Run once:
#   bash /project2/pollok_1603/jc_905/pmf2/code/go_walk.sh
# 10:20 PDT fix: a dependency on an already finished (purged) job fails with "Job dependency problem": finished init
# tasks are checked with sacct (COMPLETED: no dependency; any other end state: stop).
set -euo pipefail
P=/project2/pollok_1603/jc_905; P3=$P/pmf2; P5=$P/pmf2_35
[ -f $P3/inputs/walk_go.done ] && { echo "already submitted"; exit 1; }
SD=""
for s in baseline all_features; do
  t=$(cat $P3/inputs/walkinit_$s.jobid)_0; st=$(sacct -j "$t" -X -n -P -o State | head -1)
  case "$st" in COMPLETED) ;; PENDING|RUNNING|"") SD="${SD:+$SD,}afterok:$t" ;; *) echo "init $t $st"; exit 1 ;; esac
done
SL=$(sbatch --parsable -o $P3/logs/%x_%j.out ${SD:+--dependency=$SD} \
  --export=NONE,ROOT=$P3,JOBS=v2wslice.txt,CONC=12,GPUTYPE=P100 $P3/code/pooled_mingru_v2_slice_carc.sbatch)
echo "v2wslice.txt $SL gpu slice-check P100" | tee -a $P3/inputs/submitted.txt
B="bash $P3/code/pooled_mingru_v2_walk_submit_carc.sh"
$B $P3 baseline gpu P100 0 afterok:$SL
$B $P3 all_features gpu A40 0 afterok:$SL
$B $P5 baseline gpu A40 100 afterok:$SL
$B $P5 all_features gpu L40S 100 afterok:$SL
date > $P3/inputs/walk_go.done
squeue -u jc_905 -h -p gpu -o "%j/%T/%f" | sort | uniq -c
