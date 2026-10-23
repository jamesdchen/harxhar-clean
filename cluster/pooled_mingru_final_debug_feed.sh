#!/bin/bash
# 2026-10-23 final pooled MinGRU: keep the debug partition's P100s (e23-02, 2 GPUs; QOS debug: 1 h, 5 jobs a user)
# busy with tune shards that are still PENDING in the gpu partition (where our fairshare keeps them waiting).  Each run
# takes the next lines of the queue file (<root> <jobs file> <conc> <pending gpu-partition job id or ->), submits the
# file(s) (a heavy shard: <half a>+<half b>, whole configurations each) to the debug partition (1 h limit, GPUTYPE P100, POST check-run) and cancels its pending gpu-partition twin, so a
# shard runs exactly once.  A line whose twin already started (or finished) in the gpu partition is skipped.
# Every move is appended to <root>/inputs/submitted.txt and to the moved file.
#   usage (CARC login node, repeatedly): bash pooled_mingru_final_debug_feed.sh [queue file]
set -uo pipefail
Q=${1:-/home1/jc_905/pmf/debug_queue.txt}
MOVED=/home1/jc_905/pmf/debug_moved.txt
MAXJ=${MAXJ:-4}  # one debug slot (QOS: 5 a user) left for the walk probe
G=/home1/jc_905/pmf/pooled_mingru_final_gpu_carc.sbatch
touch "$MOVED"
n=$(squeue -u jc_905 -p debug -h | wc -l)
while [ "$n" -lt "$MAXJ" ]; do
  line=$(grep -v -x -F -f "$MOVED" "$Q" | head -1)
  [ -z "$line" ] && break
  read -r root file conc twin <<< "$line"
  ok=1
  if [ "$twin" != - ]; then
    st=$(squeue -h -j "$twin" -o %T 2>/dev/null)
    [ "$st" = PENDING ] || ok=0
  fi
  files=(${file//+/ })  # a heavy shard split in two halves (whole configurations each): both move together
  if [ $ok = 1 ] && [ $((n + ${#files[@]})) -gt "$MAXJ" ]; then break; fi
  if [ $ok = 1 ]; then
    case $root in */pmf35) WD=3500 ;; *) WD=3000 ;; esac
    for f in "${files[@]}"; do
      tag=${f%.txt}; tag=${tag//./_}
      id=$(sbatch --parsable -p debug --time=1:00:00 --mem=24G --job-name=pmft -o "$root/logs/%x_%j.out"            --export=NONE,ROOT=$root,JOBS=$f,SUB=tune,TAG=$tag,CONC=$conc,GPUTYPE=P100,POST=pooled_mingru_final_post_check.py $G) || exit 1
      echo "$f $id" >> "$root/inputs/submitted.txt"
      echo "moved W$WD $f -> debug $id"
      n=$((n + 1))
    done
    [ "$twin" != - ] && scancel "$twin" && echo "  twin $twin cancelled"
  else
    echo "skip $file: twin $twin is ${st:-gone}"
  fi
  echo "$line" >> "$MOVED"
done
squeue -u jc_905 -p debug -h -o "%i %j %T %M %R" | sed 's/^/debug: /'
