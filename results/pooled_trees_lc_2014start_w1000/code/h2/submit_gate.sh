#!/bin/bash
# Hoffman2 login node (bash -l): the env job, then (held on it) the gate arrays and the time probes.
#   gate/<tag>   the printed 2012-start w1000 pieces minus their first and last k days (gate_prep.py, made off the
#                cluster), re-walked on the printed design ($B/printed, CARC's caches, md5-checked) by this branch's
#                runner: T8_<arch> 8 threads (the printed run's) on one CPU class; T1_<arch> one thread
#   probe/w<W>_t8      experiments/pooled_trees_rerun_13.py time, 3 refits of each arm at the first 2014-01-02 rows,
#                      8 threads, on the final design of window W (intel-gold-6240)
#   probe/w<W>_rf_t1   the same for the random forest, 2 refits, one thread
# Refuses a second submission (jobids.txt).
set -euo pipefail
B=/u/scratch/j/jamesdc1/tlc
cd $B
if [ -f jobids.txt ] && grep -q "^gate " jobids.txt; then echo "ABORT: gate already submitted"; cat jobids.txt; exit 3; fi
L=$B/logs
mkdir -p $L $B/probe
chmod +x jobs/*.sh
q() { qsub -terse "$@" | cut -d. -f1; }
e=$(q -N tlc_env -o $L/env.log jobs/env.sh)
declare -A ARCH=([6240]=intel-gold-6240 [6140]=intel-gold-6140 [6342]=intel-gold-6342 [E5v4]=intel-E5-2650v4)
G=""
for t in T8_6240 T8_6140 T8_6342 T8_E5v4 T1_6240 T1_E5v4; do
  a=${ARCH[${t#*_}]}
  n=$(wc -l < gate/$t/tasks.txt)
  if [ "${t%%_*}" = T8 ]; then R=(-pe shared 8 -l "h_rt=4:00:00,h_data=2G,arch=$a"); else R=(-l "h_rt=8:00:00,h_data=6G,arch=$a"); fi
  j=$(q -N tlcg_$t -hold_jid $e -t 1-$n "${R[@]}" -o "$L/gate_$t.\$TASK_ID.log" \
        -v TASKFILE=$B/gate/$t/tasks.txt,DESIGN=$B/printed,OUT=$B/gate/$t,WINDOW_DAYS=1000 jobs/walk.sh)
  G="$G $t=$j"
done
P=""
for W in 1000 2000; do
  j=$(q -N tlct_w$W -hold_jid $e -pe shared 8 -l h_rt=3:00:00,h_data=2G,arch=intel-gold-6240 -o $L/time_w${W}_t8.log \
        jobs/py.sh experiments/pooled_trees_rerun_13.py time --design $B/designs/w$W --out $B/probe/w${W}_t8 \
        --window-days $W --k 3 --threads 8 --span 2014-01-02:2024-04-30)
  k=$(q -N tlct1_w$W -hold_jid $e -l h_rt=4:00:00,h_data=6G,arch=intel-gold-6240 -o $L/time_w${W}_rf_t1.log \
        jobs/py.sh experiments/pooled_trees_rerun_13.py time --design $B/designs/w$W --out $B/probe/w${W}_rf_t1 \
        --window-days $W --est rf --k 2 --threads 1 --span 2014-01-02:2024-04-30)
  P="$P w${W}_t8=$j w${W}_rf_t1=$k"
done
echo "gate $(date +%F_%T) env=$e$G$P" | tee -a jobids.txt
echo "== campus rqs"; qconf -srqs campus 2>/dev/null | head -30
echo "== rqs naming jamesdc1 / mcucurin"; qconf -srqs 2>/dev/null | grep -n -i -E "jamesdc1|mcucurin" | head
echo "== queues mcucurin"; qconf -sql | grep -i mcucurin
sleep 2
qstat -u jamesdc1 | head -30
