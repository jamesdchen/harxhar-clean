#!/bin/bash
# Gate L of the ablation run (RULE.txt section 5): the final w3000 run's 14 linear walks (lines 1-14 of the linear task
# file: the full cache, and the HAR benchmark as the 12-column subset of the full cache) run by this code path in
# parallel, one thread each (14 CPUs), then compared bit for bit with w3000/forecasts.parquet -> gate/gateL.json and,
# on a pass, gate/passed_lin (this node's CPU model; the linear pool runs only on it).  Exit 3 on a failure.
set -uo pipefail
B=/u/scratch/j/jamesdc1/abl
TASKS=$B/code/results/ablations_2014start_w3000/tasks/linear.txt
echo "lingate job ${SLURM_JOB_ID:-${JOB_ID:-}} on $(hostname) $(date)"
mkdir -p $B/logs/lin
for n in $(seq 1 14); do
  NT=1 bash $B/code/cluster/ablate/py.sh gb experiments/ablate_linear.py walk1 --tasks $TASKS --line $n \
    --root $B/runs/lin --designs $B/designs/lin --full $B/designs/full > $B/logs/lin/gate.$n.log 2>&1 &
done
wait
NT=1 bash $B/code/cluster/ablate/py.sh gb experiments/ablate_linear.py gate --root $B/runs/lin --out $B/gate/gateL.json
rc=$?
if [ $rc -eq 0 ]; then
  grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//; s/ *$//' > $B/gate/passed_lin
fi
echo "lingate done rc $rc $(date)"
exit $rc
