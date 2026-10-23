#!/bin/bash
# Running slots of the final tree arrays within a job cap and a CPU cap (the per-user QOS is 100 running jobs and
# 2,000 CPUs, shared with the MinGRU runs; the coordinator asked to leave room for its few CPU jobs).  Running tf jobs
# outside these arrays (gates, resubmissions) count against both caps first, then every array's running tasks; the
# headroom left goes to pending tasks one at a time, round-robin over the arrays in the priority order given, so a
# later array gets slots while an earlier one still has pending tasks.  (To 2026-10-23 08:55 each array in turn got
# min(remaining tasks, jobs left, CPUs left / its CPUs a task), so pending tasks that could not be placed held slots.)
# An array with pending tasks and no slot is held (Slurm reads ArrayTaskThrottle=0 as "no limit"); one with slots
# gets throttle = running + slots, then is released; running tasks untouched.
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
A=(); C=(); REM=(); RUN=(); PEND=(); GIVE=()
for x in "$@"; do
  a=${x%%:*}; c=${x##*:}
  rem=$(squeue -r -h -j "$a" -o "%T" 2>/dev/null | grep -c -E "RUNNING|PENDING|CONFIGURING")
  run=$(squeue -r -h -j "$a" -o "%T" 2>/dev/null | grep -c -E "RUNNING|CONFIGURING")
  A+=("$a"); C+=("$c"); REM+=("$rem"); RUN+=("$run"); PEND+=($((rem - run))); GIVE+=(0)
  aj=$((aj - run)); ac=$((ac - run * c))
done
more=1
while [ "$more" -eq 1 ]; do
  more=0
  for i in "${!A[@]}"; do
    if [ "${GIVE[$i]}" -lt "${PEND[$i]}" ] && [ "$aj" -ge 1 ] && [ "$ac" -ge "${C[$i]}" ]; then
      GIVE[$i]=$((GIVE[$i] + 1)); aj=$((aj - 1)); ac=$((ac - C[$i])); more=1
    fi
  done
done
for i in "${!A[@]}"; do
  a=${A[$i]}; rem=${REM[$i]}; run=${RUN[$i]}; g=${GIVE[$i]}
  [ "$rem" -eq 0 ] && { out="$out | $a done"; continue; }
  held=$(squeue -r -h -j "$a" -t PENDING -o "%r" 2>/dev/null | grep -c -i "held")
  if [ "$g" -eq 0 ]; then
    if [ "${PEND[$i]}" -gt 0 ]; then
      [ "$held" -eq 0 ] && scontrol hold "$a" >/dev/null 2>&1
      out="$out | $a rem $rem run $run HELD"
    else
      out="$out | $a rem $rem run $run"
    fi
  else
    n=$((run + g))
    cur=$(scontrol show job "$a" 2>/dev/null | grep -o "ArrayTaskThrottle=[0-9]*" | head -n 1 | cut -d= -f2)
    [ "$cur" != "$n" ] && scontrol update JobId="$a" ArrayTaskThrottle="$n" >/dev/null 2>&1
    [ "$held" -gt 0 ] && scontrol release "$a" >/dev/null 2>&1
    out="$out | $a rem $rem run $run thr $n"
  fi
done
echo "$out"
