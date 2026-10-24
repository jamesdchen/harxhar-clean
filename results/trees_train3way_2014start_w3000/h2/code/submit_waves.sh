#!/bin/bash
# Hoffman2 login node (bash -l): waves of the lines still ASSIGNED, across the gate-passed CPU models, one array per
# model (a piece never crosses CPU families: arch pinned, walk.sh refuses another model), 8 threads.  Coordinator,
# 10:25 PDT: at most about 120 concurrent 8-thread tasks on Hoffman2, e.g. Gold 6342 ~35, the rest on Gold 6140 /
# E5-2650 v4 / Gold 6240; h_rt from the measured hours a line with margin (submit_wave.sh SEC).  Run after the
# assignment has been re-relayed from CARC into runs/tasks/h2_assignment.txt.
#   per-model caps: CAPS (default "gold6342:35 gold6140:50 e5v4:20 gold6240:15"), minus that model's wave tasks still
#   queued or running; the total also capped at $1 (default 120) minus all wave tasks queued or running.
#   seconds a refit at 8 threads: h2/h2_throughput.txt (the gate pieces).
# usage: bash submit_waves.sh [total cap]
set -euo pipefail
B=/u/scratch/j/jamesdc1/t3t
cd $B
CAP=${1:-120}
CAPS=${CAPS:-"gold6342:35 gold6140:50 e5v4:20 gold6240:15"}
qstat -u jamesdc1 -g d > h2/.qgd
declare -A ACT
tot=0
while read -r J A; do
  n=$(awk -v j=$J 'NR > 2 && $1 == j' h2/.qgd | wc -l)
  ACT[$A]=$(( ${ACT[$A]:-0} + n ))
  tot=$((tot + n))
done < <(grep '^wave ' jobids.txt | sed -n 's/.* job=\([0-9]*\) .* arch=\([^ ]*\).*/\1 \2/p')
echo "wave tasks queued or running: $tot ($(for a in "${!ACT[@]}"; do printf '%s=%s ' $a ${ACT[$a]}; done))"
for tc in $CAPS; do
  t=${tc%%:*}
  c=${tc#*:}
  arch=$(awk -v t=$t '$1 == t {print $2}' h2/tags.txt)
  grep -qxF "$(awk -F' [|] ' -v t=$t '$1 == t {print $2}' h2/h2_throughput.txt)" h2/passed_cpus.txt || { echo "$t: not gate-passed, skipped"; continue; }
  sec=$(awk -F' [|] ' -v t=$t '$1 == t {print $5}' h2/h2_throughput.txt)
  free=$(( c - ${ACT[$arch]:-0} ))
  [ $free -le $(( CAP - tot )) ] || free=$(( CAP - tot ))
  [ $free -gt 0 ] || { echo "$t: no room (cap $c, active ${ACT[$arch]:-0}, total $tot of $CAP)"; continue; }
  out=$(NT=8 ARCH_ONLY=$arch SEC=$sec WTAG=$t bash jobs/submit_wave.sh $free)
  echo "$t: $out"
  m=$(echo "$out" | sed -n 's/.* lines=\([0-9]*\) .*/\1/p')
  tot=$(( tot + ${m:-0} ))
done
