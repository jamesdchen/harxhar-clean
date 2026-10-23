#!/bin/bash
# 2026-10-23 final pooled MinGRU: keep the gpu partition's per-user submit limit (QOS gpu: 100 submitted, 36 running)
# filled with the tune files still waiting to be submitted.  Each run submits the next lines of the queue file
# (<root> <jobs file> <conc> <time limit> <nice> <debug-eligible 0|1>) while fewer than LIMIT of our jobs are in the
# gpu partition; a debug-eligible file is also appended to the debug feeder's queue (pooled_mingru_final_debug_feed.sh
# moves it to the debug partition if it is still pending there when a debug slot frees).
#   usage (CARC login node, repeatedly): bash pooled_mingru_final_gpu_feed.sh [queue file]
set -uo pipefail
Q=${1:-/home1/jc_905/pmf/gpu_queue.txt}
DONE=/home1/jc_905/pmf/gpu_queue_done.txt
LIMIT=${LIMIT:-96}
G=/home1/jc_905/pmf/pooled_mingru_final_gpu_carc.sbatch
touch "$DONE"
n=$(squeue -u jc_905 -p gpu -h | wc -l)
while [ "$n" -lt "$LIMIT" ]; do
  line=$(grep -v -x -F -f "$DONE" "$Q" | head -1)
  [ -z "$line" ] && break
  read -r root file conc tl nice dbg <<< "$line"
  tag=${file%.txt}; tag=${tag//./_}
  id=$(sbatch --parsable --job-name=pmft --mem=24G --time="$tl" --nice="$nice" -o "$root/logs/%x_%j.out" \
       --export=NONE,ROOT=$root,JOBS=$file,SUB=tune,TAG=$tag,CONC=$conc,GPUTYPE=P100,POST=pooled_mingru_final_post_check.py $G) || break
  echo "$file $id" >> "$root/inputs/submitted.txt"
  [ "$dbg" = 1 ] && echo "$root $file $conc $id" >> /home1/jc_905/pmf/debug_queue.txt
  echo "$line" >> "$DONE"
  echo "submitted $file $id"
  n=$((n + 1))
done
echo "gpu partition: $(squeue -u jc_905 -p gpu -h | wc -l) jobs; queue left: $(grep -v -x -F -f "$DONE" "$Q" | wc -l)"
