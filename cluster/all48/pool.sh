#!/bin/bash
# Claim-pool workers of the ALL48 trees (results/all48_2026-10-30/RULE.txt sections 2.3 and 6), a job payload (no
# sbatch, copy or move in here): cluster/ablate/pool5.sh's claim logic, unchanged, with the trees' walk of this study
# (and, for a task file of ~10,000 lines, a fast skip of lines done by their first claim and the classes' eligibility
# read once a scan).
# NW workers on this node, each taking the lowest eligible unclaimed line of the task file ("<est> <set> <i>/<n> <hours
# at EPYC speed>", experiments/all48_trees.py tasks), running it to the end (experiments/all48_trees.py walk, 8
# threads), then the next, until no eligible line is left or the job's time is nearly up.
# Eligible: the node's CPU model in $B/gate/*/passed_<est> (gates T1 + T2), and 1.3 x the line's hours fit before the
# job's end (backfill: short jobs take short lines).  Claims: $CL/<line>[.r<j>]/ (mkdir is the atomic claim; files
# claim, cpu, hb touched every 5 minutes while the line runs, done or failed).  A claim with neither done nor failed
# whose hb is older than STALE minutes is stale: a worker on the SAME CPU model takes it over as <line>.r<j+1> (at most
# 9 times) and resumes it from its checkpoint (a forest piece refuses another CPU family anyway).  A failed line is left
# for a look.  A line reserved for the other cluster (cluster/ablate/claim_reserve.sh, a marker that is not a CPU model)
# is never taken.  Writes only under runs/ and logs/ of the staging directory.
# env: A48_B (staging holding code/), NW (workers), TLIMIT (seconds of the job), STALE (minutes, default 25),
#      PYSH (python wrapper; default $A48_B/code/cluster/all48/py_h2.sh; on CARC $A48_B/code/cluster/ablate/py_carc.sh),
#      TASKS (default $A48_B/tasks/trees48.txt), CL (default $A48_B/runs/claims_trees), TOUT (pieces root; default
#      $A48_B/runs/trees), TDESIGN (the 48-interval run dir holding _work/design_<set>; default $A48_B/runs/lin)
set -uo pipefail
B=${A48_B:?}
export ABL_B=$B  # cluster/ablate/py_carc.sh reads its staging from ABL_B
PYSH=${PYSH:-$B/code/cluster/all48/py_h2.sh}
NW=${NW:?}; TLIMIT=${TLIMIT:?}; STALE=${STALE:-25}
JOBID=${SLURM_JOB_ID:-${JOB_ID:-x}}.${SGE_TASK_ID:-0}
CPU=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//; s/ *$//')
T0=$(date +%s)
END=$((T0 + TLIMIT - 900))
TASKS=${TASKS:-$B/tasks/trees48.txt}
CL=${CL:-$B/runs/claims_trees}
TOUT=${TOUT:-$B/runs/trees}
TDESIGN=${TDESIGN:-$B/runs/lin}
LG=$B/logs/pool_trees
mkdir -p $CL $LG
echo "pool all48 job $JOBID on $(hostname) [$CPU]: staging $B, python $PYSH; tasks $TASKS claims $CL; design $TDESIGN; $NW workers, start $(date), stop claiming at $(date -d @$END)"

eligible() {  # <est>: 0 if this node's CPU model passed the gates for the class
  grep -qxF "$CPU" $B/gate/*/passed_$1 2> /dev/null
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

run_line() {  # <gen dir> <line> <est> <set> <part> <worker>
  local g=$1 n=$2 f1=$3 f2=$4 f3=$5 w=$6 rc hbp lg env
  echo "$(hostname) job $JOBID worker $w $(date +%F_%T)" > $g/claim
  echo "$CPU" > $g/cpu
  touch $g/hb
  ( while sleep 300; do touch $g/hb; done ) &
  hbp=$!
  lg=$LG/$n.$(basename $g).log
  echo "[$(date +%T)] w$w line $n ($(basename $g)): $f1 $f2 $f3"
  env=gb
  [ "$f1" = rf_skx ] && env=skx
  NT=8 bash $PYSH $env experiments/all48_trees.py walk --design $TDESIGN --est $f1 --set $f2 --part $f3 \
    --out $TOUT --threads 8 > $lg 2>&1
  rc=$?
  kill $hbp 2> /dev/null
  if [ $rc -eq 0 ]; then date +%F_%T > $g/done; else echo "rc $rc $(date +%F_%T)" > $g/failed; fi
  echo "[$(date +%T)] w$w line $n exit $rc"
}

worker() {
  local w=$1 n f1 f2 f3 f4 st s1 s2 s3 gen need
  local -A el
  while [ $(date +%s) -lt $END ]; do
    gen=""
    n=0
    el=()  # eligibility of each class, read once a scan (the task file has ~10,000 lines)
    while read -r f1 f2 f3 f4; do
      n=$((n + 1))
      [ -n "$f1" ] || continue
      [ -e $CL/$n/done ] && continue  # fast path: a line done by its first claim (state() would say done)
      if [ -z "${el[$f1]:-}" ]; then
        if eligible $f1; then el[$f1]=1; else el[$f1]=0; fi
      fi
      [ "${el[$f1]}" = 1 ] || continue
      need=$(awk -v h="$f4" 'BEGIN {printf "%d", h * 3600 * 1.3}')
      [ $(( $(date +%s) + need )) -lt $END ] || continue
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
echo "pool all48 job $JOBID done $(date)"
