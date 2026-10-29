#!/bin/bash
# 2026-10-28 addendum J (e): reserve, release or complete pool3.sh / pool5.sh claim lines by hand, for a share of a task
# file walked on another cluster (the CARC share of addendum J).  Runs where the claims dir is; from the laptop:
#   git -c core.autocrlf=false show <rev>:cluster/ablate/claim_reserve.sh |
#     ssh <host> "bash -s -- $(printf '%q ' <mode> <claims dir> '<marker>' <first> <last>)"
# reserve: all or nothing -- every line n in [first, last] must be free (no n or n.r<j> dir), then mkdir n (the pool's
#   atomic claim) with claim, cpu = <marker>, hb.  The pool's workers skip such a line for good: state() gives "running",
#   later "stale <n>.r1 <marker>", and a stale claim is taken over only by a worker whose /proc/cpuinfo model name equals
#   the claim's cpu string, which <marker> (not a CPU model name) never does; done / failed are never written by them.
# release: undo a reservation (cpu = <marker>, no done or failed, no n.r<j>): rm its files, rmdir.
# done: date +%F_%T > n/done in each reservation (cpu = <marker>, no done yet), as the pool's run_line on exit 0.
set -euo pipefail
mode=${1:?mode} CL=${2:?claims dir} MARK=${3:?marker} a=${4:?first} b=${5:?last}
[ -d "$CL" ] || { echo "no claims dir $CL"; exit 2; }
case $MARK in Intel* | AMD* | *Xeon* | *EPYC*) echo "marker '$MARK' looks like a CPU model"; exit 2 ;; esac
lines=$(seq "$a" "$b")
bad=0
for n in $lines; do
  r=$(ls -d "$CL/$n".r[1-9] 2> /dev/null | head -1 || true)
  case $mode in
    reserve) [ ! -e "$CL/$n" ] && [ -z "$r" ] || { echo "line $n: already claimed ($(cat "$CL/$n/cpu" 2> /dev/null))"; bad=1; } ;;
    release | done)
      [ -d "$CL/$n" ] && [ -z "$r" ] && [ "$(cat "$CL/$n/cpu" 2> /dev/null)" = "$MARK" ] && [ ! -e "$CL/$n/failed" ] \
        && [ ! -e "$CL/$n/done" ] || { echo "line $n: not an open reservation '$MARK'"; bad=1; } ;;
    *) echo "unknown mode $mode"; exit 2 ;;
  esac
done
[ $bad -eq 0 ] || { echo "$mode $a-$b: refused, nothing changed"; exit 3; }
for n in $lines; do
  case $mode in
    reserve)
      mkdir "$CL/$n" || { echo "line $n: mkdir lost a race -- stop"; exit 4; }
      echo "$(hostname) reservation for addendum J (e) $(date +%F_%T): '$MARK'" > "$CL/$n/claim"
      echo "$MARK" > "$CL/$n/cpu"
      touch "$CL/$n/hb" ;;
    release) rm -f "$CL/$n/claim" "$CL/$n/cpu" "$CL/$n/hb" && rmdir "$CL/$n" ;;
    done) date +%F_%T > "$CL/$n/done" ;;
  esac
done
echo "$mode $a-$b: $(echo $lines | wc -w) lines, $(date +%F_%T), $CL, marker '$MARK'"
