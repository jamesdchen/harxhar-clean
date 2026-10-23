#!/bin/bash
# Hoffman2 login node (bash -l), read only: the tree learning-curve jobs (qstat by name and state), finished pieces by
# directory and arm, errors in the logs since the last call, the env log's end.  usage: bash status.sh [n log lines]
B=/u/scratch/j/jamesdc1/tlc
N=${1:-4}
echo "== $(hostname) $(date)"
echo "== qstat jamesdc1 (count name state), array tasks expanded"
qstat -u jamesdc1 -g d 2>/dev/null | awk 'NR > 2 {print $3, $5}' | sort | uniq -c
echo "== tlc slots running: $(qstat -u jamesdc1 -s r 2>/dev/null | awk 'NR > 2 && $3 ~ /^tlc/ {s += $9} END {print s + 0}')"
for d in $B/gate/* $B/runs/*; do
  [ -d $d/pieces ] || continue
  done_=$(ls $d/pieces 2>/dev/null | grep -E '\.json$' | grep -vc '\.ckpt\.json$')
  ck=$(ls $d/pieces 2>/dev/null | grep -c '\.ckpt\.json$')
  echo "$(basename $d): $done_ pieces done, $ck with checkpoints, $(wc -l < $d/tasks.txt 2>/dev/null) tasks"
done
for d in $B/runs/*; do
  [ -d $d/pieces ] || continue
  ls $d/pieces | grep -E '\.json$' | grep -v ckpt | sed -E 's/_(bar|day)_.*//' | sort | uniq -c | sed "s|^|  $(basename $d) |"
done
for p in $B/probe/*/timing/sec_refit.csv; do [ -f $p ] && { echo "== $p"; cat $p | cut -d, -f1-7; }; done
echo "== env log"; tail -n $N $B/logs/env.log 2>/dev/null
M=$B/logs/.status_mark
echo "== log errors since the last status"
NEW=(); [ -f $M ] && NEW=(-newer $M)
find $B/logs -name '*.log' "${NEW[@]}" 2>/dev/null | xargs -r grep -l -E "Traceback|Error|error:|Killed|refusing|exceeded" 2>/dev/null | head -20 | while read -r f; do
  echo "-- $f"; grep -E "Traceback|Error|error:|Killed|refusing|exceeded" "$f" | tail -n $N
done
touch $M
echo "== done $(date)"
