#!/bin/bash
# Pooled tree walks (2026-10-07): pull the dumped pooled tables (yhat_pooltree_<est>_<set>.parquet), the chunk CPU census
# and DUMP_DONE from CARC into results/pooled_trees_2026-10-07/.  An existing local table is never replaced by different
# bytes (refused, listed); md5 compared after the copy.
#   bash cluster/slurm/pull_pooltrees.sh [<glob> (default yhat_pooltree_*.parquet)]
set -euo pipefail
cd "$(dirname "$0")/../.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
CH=${CARC_HOST:-usc-discovery1}
CARC=/scratch1/jc_905/harxhar-pooltrees
R=results/pooled_trees_2026-10-07
G=${1:-yhat_pooltree_*.parquet}
mkdir -p "$R"
REMOTE=$("$SSH" -o BatchMode=yes "$CH" "cd $CARC/$R && md5sum $G class_census.csv 2>/dev/null" 2>/dev/null | grep -v "reloaded\|python/3" || true)
[ -n "$REMOTE" ] || { echo "nothing matching $G on CARC"; exit 0; }
TODO=()
while read -r m f; do
  f=${f#\*}
  if [ -f "$R/$f" ]; then
    l=$(md5sum "$R/$f" | cut -d' ' -f1)
    if [ "$l" = "$m" ]; then echo "  same      $f"; continue; fi
    [ "$f" = class_census.csv ] || { echo "  REFUSED   $f (local copy differs; not replaced)"; continue; }
  fi
  TODO+=("$f")
done <<< "$REMOTE"
if [ "${#TODO[@]}" -gt 0 ]; then
  "$SSH" -o BatchMode=yes "$CH" "cd $CARC/$R && tar cf - ${TODO[*]} \$(ls DUMP_DONE 2>/dev/null)" 2>/dev/null | tar xf - -C "$R"
  for f in "${TODO[@]}"; do
    m=$(printf '%s\n' "$REMOTE" | awk -v f="$f" '{g = $2; sub(/^\*/, "", g)} g == f {print $1}')
    l=$(md5sum "$R/$f" | cut -d' ' -f1)
    [ "$l" = "$m" ] && echo "  pulled    $f ($l)" || { echo "  MD5 DIFFERS $f"; exit 1; }
  done
fi
