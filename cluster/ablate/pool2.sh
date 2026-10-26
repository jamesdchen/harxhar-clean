#!/bin/bash
# Claim-pool workers of the ablation run (RULE.txt section 6): NW workers on this node, each taking the lowest eligible
# unclaimed line of the task file, running it to the end, then the next, until no eligible line is left or the job's
# time is nearly up.  KIND=trees: lines "<est> <arm> <i>/<n> <hours at EPYC speed>" -> ablate_trees.py walk (8 threads);
# KIND=lin: lines "<arm> <model> <chunk> <n>" from line 15 on (1-14 are gate L's) -> ablate_linear.py walk1 (1 thread).
# Eligible: the node's CPU model in gate/*/passed_<est> (trees) or gate/passed_lin (linear); a forest line also needs
# runs/projection.done and no "rf_skx <arm>" line in runs/rf_skip.txt; a tree line is not started unless 1.3 x its hours
# fit before the job's end.  Claims: runs/claims_<KIND>/<line>[.r<j>]/ (mkdir is the atomic claim; files claim, cpu, hb
# touched every 5 minutes while the line runs, done or failed).  A claim with neither done nor failed whose hb is older
# than STALE minutes is stale: a worker on the SAME CPU model takes it over as <line>.r<j+1> (at most 9 times) and
# resumes it (tree pieces resume from their checkpoints; a forest piece refuses another CPU family anyway).  A failed
# line is left for a look.  Env: KIND, NW, TLIMIT (seconds of the job), STALE (minutes, default 25).  Writes only under
# runs/ and logs/ of the staging directory.
# 2026-10-25 addendum C: pool.sh with the eligibility test fixed -- it was "cat gate/*/passed_<est> | grep -q" under
# pipefail, which fails whenever grep exits on its first match before cat has written the rest (SIGPIPE), so workers
# found "no eligible line" at random and exited; now grep reads the files itself.  Nothing else changed.
set -uo pipefail
B=/u/scratch/j/jamesdc1/abl
KIND=${KIND:?}; NW=${NW:?}; TLIMIT=${TLIMIT:?}; STALE=${STALE:-25}
JOBID=${SLURM_JOB_ID:-${JOB_ID:-x}}.${SGE_TASK_ID:-0}
CPU=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//; s/ *$//')
T0=$(date +%s)
END=$((T0 + TLIMIT - 900))
if [ "$KIND" = trees ]; then TF=trees; else TF=linear; fi
TASKS=$B/code/results/ablations_2014start_w3000/tasks/$TF.txt
CL=$B/runs/claims_$KIND
LG=$B/logs/pool_$KIND
mkdir -p $CL $LG
echo "pool $KIND job $JOBID on $(hostname) [$CPU]: $NW workers, start $(date), stop claiming at $(date -d @$END)"

eligible() {  # <field1> <field2>: 0 if this node may run the line
  if [ "$KIND" = trees ]; then
    grep -qxF "$CPU" $B/gate/*/passed_$1 2> /dev/null || return 1
    if [ "$1" = rf_skx ]; then
      [ -f $B/runs/projection.done ] || return 1
      grep -qxF "rf_skx $2" $B/runs/rf_skip.txt 2> /dev/null && return 1
    fi
    return 0
  fi
  grep -qxF "$CPU" $B/gate/passed_lin 2> /dev/null
}

state() {  # <line>: free | done | failed | running | stale <next generation dir> <cpu of the stale claim>
  local n=$1 g=$CL/$1 j=0 k d hb
  [ -d $g ] || { echo free; return; }
  for k in 1 2 3 4 5 6 7 8 9; do [ -d $CL/$n.r$k ] && { j=$k; g=$CL/$n.r$k; }; done
  for d in $CL/$n $CL/$n.r[1-9]; do [ -e "$d/done" ] && { echo done; return; }; done
  [ -e $g/failed ] && { echo failed; return; }
  hb=$g/hb
  [ -e $hb ] || hb=$g
  if [ $(( $(date +%s) - $(stat -c %Y $hb) )) -gt $((STALE * 60)) ] && [ $j -lt 9 ]; then
    echo "stale $CL/$n.r$((j + 1)) $(cat $g/cpu 2> /dev/null)"
  else
    echo running
  fi
}

run_line() {  # <gen dir> <line> <field1> <field2> <field3> <worker>
  local g=$1 n=$2 f1=$3 f2=$4 f3=$5 w=$6 rc hbp lg env
  echo "$(hostname) job $JOBID worker $w $(date +%F_%T)" > $g/claim
  echo "$CPU" > $g/cpu
  touch $g/hb
  ( while sleep 300; do touch $g/hb; done ) &
  hbp=$!
  lg=$LG/$n.$(basename $g).log
  echo "[$(date +%T)] w$w line $n ($(basename $g)): $f1 $f2 $f3"
  if [ "$KIND" = trees ]; then
    env=gb
    [ "$f1" = rf_skx ] && env=skx
    NT=8 bash $B/code/cluster/ablate/py.sh $env experiments/ablate_trees.py walk --arm $f2 --est $f1 --part $f3 \
      --full $B/designs/full --out $B/runs/trees --threads 8 > $lg 2>&1
  else
    NT=1 bash $B/code/cluster/ablate/py.sh gb experiments/ablate_linear.py walk1 --tasks $TASKS --line $n \
      --root $B/runs/lin --designs $B/designs/lin --full $B/designs/full > $lg 2>&1
  fi
  rc=$?
  kill $hbp 2> /dev/null
  if [ $rc -eq 0 ]; then date +%F_%T > $g/done; else echo "rc $rc $(date +%F_%T)" > $g/failed; fi
  echo "[$(date +%T)] w$w line $n exit $rc"
}

worker() {
  local w=$1 n f1 f2 f3 f4 st s1 s2 s3 gen need
  while [ $(date +%s) -lt $END ]; do
    gen=""
    n=0
    while read -r f1 f2 f3 f4; do
      n=$((n + 1))
      [ -n "$f1" ] || continue
      if [ "$KIND" = lin ] && [ $n -le 14 ]; then continue; fi
      eligible $f1 $f2 || continue
      if [ "$KIND" = trees ]; then
        need=$(awk -v h="$f4" 'BEGIN {printf "%d", h * 3600 * 1.3}')
        [ $(( $(date +%s) + need )) -lt $END ] || continue
      fi
      st=$(state $n)
      read -r s1 s2 s3 <<< "$st"
      if [ "$s1" = free ]; then
        if mkdir $CL/$n 2> /dev/null; then gen=$CL/$n; break; fi
      elif [ "$s1" = stale ] && [ "$s3" = "$CPU" ]; then
        if mkdir $s2 2> /dev/null; then gen=$s2; echo "[$(date +%T)] w$w takes over stale line $n -> $s2"; break; fi
      fi
    done < $TASKS
    if [ -z "$gen" ]; then echo "[$(date +%T)] w$w: no eligible line left"; return 0; fi
    run_line $gen $n $f1 $f2 $f3 $w
  done
  echo "[$(date +%T)] w$w: time"
}

for w in $(seq 1 $NW); do
  worker $w > $LG/w.$JOBID.$w.log 2>&1 &
  sleep 2
done
wait
echo "pool $KIND job $JOBID done $(date)"
