#!/bin/bash
# Four forest walk lines in parallel on one Slurm node (the campus QOS counts nodes per job, 16 a user): lines
# 4j-3 .. 4j of TASKFILE for array task j, each through jobs/walk_slurm.sh at 8 threads with its own log; waits.
# (Written on Hoffman2 2026-10-24 12:55 PDT from the session heredoc, md5 d9ad198c6f23674c2f8b06e6c23a3667.)
set -uo pipefail
B=/u/scratch/j/jamesdc1/t3t
J=${SLURM_ARRAY_TASK_ID:?}
N=$(grep -c . "${TASKFILE:?}")
for i in 0 1 2 3; do
  L=$((4 * (J - 1) + 1 + i))
  [ $L -le $N ] || continue
  SLURM_ARRAY_TASK_ID=$L SLURM_CPUS_PER_TASK=8 bash $B/jobs/walk_slurm.sh > $B/logs/slurm_w6240.$L.log 2>&1 &
done
wait
echo "job task $J done $(date)"
