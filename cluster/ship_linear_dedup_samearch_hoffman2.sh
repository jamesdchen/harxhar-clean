#!/bin/bash
# I1 attribution, same CPU architecture: ship the per-architecture task files and
# cluster/submit_linear_dedup_samearch.sh to the de-dup root and the control root on Hoffman2
# (tar over native Windows OpenSSH), and FAIL unless every file's remote md5 equals the local one.
#   bash cluster/ship_linear_dedup_samearch_hoffman2.sh
set -euo pipefail
cd "$(dirname "$0")/.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
mapfile -t FILES < <(ls cluster/linear_dedup_samearch_*.txt cluster/linear_dedup_ctrlrep_*.txt)
FILES+=(cluster/submit_linear_dedup_samearch.sh cluster/linear_dedup_pack.sh)
sums() { awk '{p = $2; sub(/^\*/, "", p); print p, $1}' | sort; }
LOCAL=$(md5sum "${FILES[@]}" | sums)
for REMOTE in /u/scratch/j/jamesdc1/harxhar-dedup /u/scratch/j/jamesdc1/harxhar-dedup-control; do
  tar czf - "${FILES[@]}" | "$SSH" -o BatchMode=yes hoffman2 "cd $REMOTE && tar xzf -"
  R=$(printf '%s\n' "${FILES[@]}" | "$SSH" -o BatchMode=yes hoffman2 "cd $REMOTE && xargs md5sum" | sums)
  BAD=$(diff <(printf '%s\n' "$LOCAL") <(printf '%s\n' "$R") || true)
  [ -z "$BAD" ] || { echo "SHIP FAILED at $REMOTE:"; printf '%s\n' "$BAD"; exit 1; }
  echo "md5 OK: ${#FILES[@]} files -> $REMOTE"
done
