#!/bin/bash
# 2026-10-27 addendum I: the ladder's merge -- T2 over the gate dirs, the 10 (model, rung) arms merged ->
# runs/trees_ladder/forecasts/, the compact runs/trees_ladder/trees_wide.parquet.
set -uo pipefail
B=/u/scratch/j/jamesdc1/abl
echo "merge_ladder job ${SLURM_JOB_ID:-${JOB_ID:-}} on $(hostname) $(date)"
NT=1 bash $B/code/cluster/ablate/py.sh gb experiments/ablate_widen.py --out $B/runs/trees_ladder \
  --gates $B/gate/s6240,$B/gate/s6736p,$B/gate/s6442y,$B/gate/u6140 \
  --tasks $B/code/results/ablations_2014start_w3000/tasks/trees_ladder.txt --to $B/runs/trees_ladder/trees_wide.parquet
echo "merge_ladder done rc $? $(date)"
