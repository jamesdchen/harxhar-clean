#!/bin/bash
# Running slots of the final tree arrays within a job cap and a CPU cap (the per-user QOS is 100 running jobs and
# 2,000 CPUs, shared with the MinGRU runs; the coordinator asked to leave room for its few CPU jobs).  Arrays get
# slots in the priority order given, each min(remaining tasks, jobs left, CPUs left / its CPUs a task); running tf jobs
# outside these arrays (gates, resubmissions) count against both caps first.  An array with no slot is held (Slurm
# reads ArrayTaskThrottle=0 as "no limit"), one with slots released and its throttle set; running tasks untouched.
# usage: CAP_JOBS=90 CAP_CPU=1880 bash throttle2.sh <array id>:<cpus a task> ...   (priority order, highest first)
set -uo pipefail
CJ=${CAP_JOBS:-90}
CC=${CAP_CPU:-1880}
ids=" "
for x in "$@"; do ids="$ids${x%%:*} "; done
oj=0; oc=0
while read -r jid st c; do
  a=${jid%%_*}
  if [[ "$ids" != *" $a "* ]] && [ "$st" = RUNNING ]; then oj=$((oj + 1)); oc=$((oc + c)); fi
done < <(squeue -u jc_905 -r -h -o "%i %T %C %j" | awk '$4 ~ /^tf/ {print $1, $2, $3}')
aj=$((CJ - oj)); ac=$((CC - oc))
out="caps $CJ/$CC other $oj/$oc"
for x in "$@"; do
  a=${x%%:*}; c=${x##*:}
  rem=$(squeue -r -h -j "$a" -o "%T" 2>/dev/null | grep -c -E "RUNNING|PENDING|CONFIGURING")
  run=$(squeue -r -h -j "$a" -t RUNNING -o "%T" 2>/dev/null | wc -l)
  [ "$rem" -eq 0 ] && { out="$out | $a done"; continue; }
  n=$rem
  [ "$n" -gt "$aj" ] && n=$aj
  k=$((ac / c)); [ "$n" -gt "$k" ] && n=$k
  [ "$n" -lt 0 ] && n=0
  held=$(squeue -r -h -j "$a" -t PENDING -o "%r" 2>/dev/null | grep -c -i "held")
  if [ "$n" -eq 0 ]; then
    [ "$held" -eq 0 ] && [ "$rem" -gt "$run" ] && scontrol hold "$a" >/dev/null 2>&1
    out="$out | $a rem $rem run $run HELD"
  else
    [ "$held" -gt 0 ] && scontrol release "$a" >/dev/null 2>&1
    cur=$(scontrol show job "$a" 2>/dev/null | grep -o "ArrayTaskThrottle=[0-9]*" | head -n 1 | cut -d= -f2)
    [ "$cur" != "$n" ] && scontrol update JobId="$a" ArrayTaskThrottle="$n" >/dev/null 2>&1
    out="$out | $a rem $rem run $run thr $n"
  fi
  aj=$((aj - n)); ac=$((ac - n * c))
done
echo "$out"
