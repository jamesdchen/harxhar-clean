#!/bin/bash
# 2026-10-28 addendum J merge: T2 over the gate dirs (LightGBM / XGBoost bit-identical across the gated CPU models),
# the 14 (class, arm) pairs lo_<g> (g = rm, liq, ew, vw, sent, iv, vd; LightGBM, XGBoost) on the FOMC-patched design
# merged -> runs/trees_lofix/forecasts/, runs/trees_lofix/trees_wide.parquet.
set -uo pipefail
B=/u/scratch/j/jamesdc1/abl
TD=$B/code/results/ablations_2014start_w3000/tasks
P="bash $B/code/cluster/ablate/py.sh gb"
echo "merge_lofix job ${SLURM_JOB_ID:-${JOB_ID:-}} on $(hostname) $(date)"
NT=1 $P experiments/ablate_widen.py --out $B/runs/trees_lofix --gates $B/gate/s6240,$B/gate/s6736p,$B/gate/s6442y,$B/gate/u6140 \
  --tasks $TD/trees_lofix.txt --to $B/runs/trees_lofix/trees_wide.parquet
echo "merge_lofix done rc $? $(date)"
