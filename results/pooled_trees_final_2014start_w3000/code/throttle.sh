#!/bin/bash
# Keep the final tree walks within CAP running jobs (the per-user limit of 100 is shared with the MinGRU runs; the
# coordinator asked to leave ~10 slots): the arrays get running slots in the priority order given, each its remaining
# tasks up to what is left of CAP (minus the running tf jobs outside these arrays).  An array with no slot left is held,
# one with slots is released and its ArrayTaskThrottle set (Slurm reads 0 as "no limit", so 0 is a hold).  Running
# tasks are never touched.  Run on the CARC login node, idempotent (each poll).
# usage: CAP=90 bash throttle.sh <array id> [<array id> ...]   (priority order, highest first)
set -uo pipefail
CAP=${CAP:-90}
ids=" $* "
other=0
while read -r jid st; do
  a=${jid%%_*}
  if [[ "$ids" != *" $a "* ]] && [ "$st" = RUNNING ]; then other=$((other + 1)); fi
done < <(squeue -u jc_905 -r -h -o "%i %T %j" | awk '$3 ~ /^tf/ {print $1, $2}')
avail=$((CAP - other))
out="cap $CAP other $other"
for a in "$@"; do
  rem=$(squeue -r -h -j "$a" -o "%T" 2>/dev/null | grep -c -E "RUNNING|PENDING|CONFIGURING")
  run=$(squeue -r -h -j "$a" -t RUNNING -o "%T" 2>/dev/null | wc -l)
  [ "$rem" -eq 0 ] && { out="$out | $a done"; continue; }
  n=$rem; [ "$n" -gt "$avail" ] && n=$avail
  [ "$n" -lt 0 ] && n=0
  held=$(squeue -r -h -j "$a" -t PENDING -o "%r" 2>/dev/null | grep -c -i "held")
  if [ "$n" -eq 0 ]; then
    [ "$held" -eq 0 ] && scontrol hold "$a" >/dev/null 2>&1
    out="$out | $a rem $rem run $run HELD"
  else
    [ "$held" -gt 0 ] && scontrol release "$a" >/dev/null 2>&1
    cur=$(scontrol show job "$a" 2>/dev/null | grep -o "ArrayTaskThrottle=[0-9]*" | head -n 1 | cut -d= -f2)
    [ "$cur" != "$n" ] && scontrol update JobId="$a" ArrayTaskThrottle="$n" >/dev/null 2>&1
    out="$out | $a rem $rem run $run thr $n"
  fi
  avail=$((avail - n))
done
echo "$out"
