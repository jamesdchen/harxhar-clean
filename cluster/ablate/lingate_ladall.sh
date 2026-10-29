#!/bin/bash
# 2026-10-28 addendum K: the linear side's caches and gate (Hoffman2 Gold 6240, the linear-gated model).  (1) The rung
# caches designs_fix/lin/lrung_<m>_<k> (k = 1, 3, 5, 10, 20 and the gate rung 46; m = ridge, lasso, enet, each with its
# own ranking) by ablate_design.py subsets from the FOMC-patched full cache, one call per model (no other cache is
# touched).  (2) The gate: chunk 0 of lrung_<m>_46 (the full set through the rung code path, with the full set's
# chunks) by ablate_linear.py walk1 with the cell-B flags, the three in parallel, one thread each.  (3) ablate_ladall.py
# lingate: each chunk's rows = Table 1's patched walk (fomcfix lin_B.parquet) bit for bit -> gate/ladall_lin.json;
# exit 3 on a failure (the linear pool runs afterok on this job).
set -uo pipefail
B=/u/scratch/j/jamesdc1/abl
TD=$B/code/results/ablations_2014start_w3000/tasks
P="bash $B/code/cluster/ablate/py.sh gb"
CELL=FOMC_SCHEDULE_2020_FIX=1+FOMC_PROXIMITY_NO_WINSOR=1+DROP_HOLIDAY_SESSIONS=0
echo "lingate_ladall job ${SLURM_JOB_ID:-} on $(hostname) $(date)"
for m in ridge lasso enet; do
  NT=4 $P experiments/ablate_design.py subsets --full $B/designs_fix/full --new $B/designs/newser \
    --root $B/designs_fix/lin --arms lrung_${m}_1,lrung_${m}_3,lrung_${m}_5,lrung_${m}_10,lrung_${m}_20,lrung_${m}_46 \
    || { echo "subsets $m failed $(date)"; exit 2; }
  echo "subsets $m done $(date)"
done
G=$B/runs/lin_ladall_gate
mkdir -p $G
for n in 1 2 3; do
  NT=1 $P experiments/ablate_linear.py walk1 --tasks $TD/linear_ladall_gate.txt --line $n --root $G \
    --designs $B/designs_fix/lin --full $B/designs_fix/full --cell "$CELL" > $G/walk1_$n.log 2>&1 &
done
wait
echo "gate walks done $(date)"
NT=1 $P experiments/ablate_ladall.py lingate --root $G --tasks $TD/linear_ladall_gate.txt --out $B/gate/ladall_lin.json
rc=$?
echo "lingate_ladall done rc $rc $(date)"
exit $rc
