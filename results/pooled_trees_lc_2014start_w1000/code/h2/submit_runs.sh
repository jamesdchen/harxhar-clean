#!/bin/bash
# Hoffman2 login node (bash -l): one UGE array per (window, group) over $B/runs/w<W>/tasks_<group>.txt (made off the
# cluster by experiments/pooled_trees_rerun_13.py tasks --refits bar --target-hours H from this cluster's time probe,
# the printed run's rule; uploaded through the data-transfer node), through h2/walk.sh (resumable pieces).
# usage: submit_runs.sh <W> <group> <threads> <h_rt> <h_data per slot> <arch list> <priority> [max concurrent tasks]
#   group: a task file suffix (gb = LightGBM + XGBoost both sets, rf = the random forest both sets); threads 1 = a
#   single-slot job, else -pe shared <threads>; priority 0 (highest a user may set) .. -1023.
# HOLD (env): job ids the array waits for (-hold_jid).  Refuses a second submission of the same (W, group) (jobids.txt).
set -euo pipefail
B=/u/scratch/j/jamesdc1/tlc
W=$1; G=$2; NT=$3; HRT=$4; HD=$5; AR=$6; PR=$7; TC=${8:-}
cd $B
R=$B/runs/w$W
TF=$R/tasks_$G.txt
[ -f $TF ] || { echo "no $TF"; exit 2; }
if grep -q "^run w$W $G " jobids.txt 2>/dev/null; then echo "ABORT: w$W $G already submitted"; grep "^run w$W $G " jobids.txt; exit 3; fi
[ -f ENV_OK ] || { echo "no env"; exit 2; }
n=$(grep -c . $TF)
mkdir -p $R/pieces $B/logs
PE=(); [ "$NT" -gt 1 ] && PE=(-pe shared "$NT")
T=(); [ -n "$TC" ] && T=(-tc "$TC")
H=(); [ -n "${HOLD:-}" ] && H=(-hold_jid "$HOLD")
j=$(qsub -terse -N tlc_${G}_w$W -t 1-$n ${T[@]+"${T[@]}"} ${H[@]+"${H[@]}"} ${PE[@]+"${PE[@]}"} -p "$PR" -l "h_rt=$HRT,h_data=$HD,arch=$AR" \
      -o "$B/logs/run_w${W}_$G.\$TASK_ID.log" \
      -v TASKFILE=$TF,DESIGN=$B/designs/w$W,OUT=$R,WINDOW_DAYS=$W,THREADS=$NT jobs/walk.sh | cut -d. -f1)
echo "run w$W $G $(date +%F_%T) job=$j tasks=$n threads=$NT h_rt=$HRT h_data=$HD arch=$AR priority=$PR tc=${TC:-none} hold=${HOLD:-none}" | tee -a jobids.txt
