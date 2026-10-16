#!/bin/bash
# xgbtop (2026-10-07): copy FINISHED Hoffman2 chunk directories (DONE; AVX-only nodes, gated byte-identical to CARC
# epyc-7513) into the CARC deployment root at the same relative path, so the CARC pack tasks of those chunks skip them
# (every pack script skips a chunk whose DONE exists) and the CARC merges see every chunk in one root.
#   bash cluster/transfer_xgbtop_h2_to_carc.sh <relative root> [<relative root> ...]
#     e.g. results/linear_subsection_trees_optuna_mask_xgbtop/stage1 results/linear_subsection_trees_tuned_mask_xgbtop
# A chunk is copied only when its CARC directory does NOT exist (a CARC task that already started keeps its own
# computation); DONE is extracted last; every copied file's md5 is compared on both ends; the NODE file (Hoffman2 host and
# CPU model) travels with the chunk.  Streams Hoffman2 (DTN) -> this machine -> CARC; nothing is written locally but the log.
# Run from Git Bash locally.
set -euo pipefail
cd "$(dirname "$0")/.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
H2=/u/scratch/j/jamesdc1/harxhar-xgbtop-h2
CARC=/scratch1/jc_905/harxhar-xgbtop
CH=${CARC_HOST:-usc-discovery1}
LOG=logs/transfer_xgbtop_h2_to_carc.log
mkdir -p logs
[ "$#" -gt 0 ] || { echo "usage: transfer_xgbtop_h2_to_carc.sh <relative root> ..."; exit 1; }
DONE_H2=$("$SSH" -o BatchMode=yes h2dtn "cd $H2 && for r in $*; do [ -d \$r ] && find \$r -path '*/chunks/c*' -name DONE -printf '%h\n'; done | sort")
[ -n "$DONE_H2" ] || { echo "no finished Hoffman2 chunks under $*"; exit 0; }
EXIST=$("$SSH" -o BatchMode=yes "$CH" "cd $CARC && for d in $(printf '%s ' $DONE_H2); do [ -e \$d ] && echo \$d; done" 2>/dev/null | grep -v "reloaded\|python/3" || true)
TODO=$(comm -23 <(printf '%s\n' $DONE_H2 | sort) <(printf '%s\n' $EXIST | grep . | sort))
echo "$(date +%F_%T) Hoffman2 DONE chunks: $(printf '%s\n' $DONE_H2 | grep -c .); already on CARC (started or done there): $(printf '%s\n' $EXIST | grep -c . || true); to copy: $(printf '%s\n' $TODO | grep -c . || true)" | tee -a "$LOG"
[ -n "$TODO" ] || exit 0
for d in $TODO; do
  # every file but DONE first, DONE last (a CARC task starting mid-copy finds no DONE and recomputes the same bytes)
  "$SSH" -o BatchMode=yes h2dtn "cd $H2 && { find $d -type f ! -name DONE; echo $d/DONE; } | tar cf - -T -" \
    | "$SSH" -o BatchMode=yes "$CH" "cd $CARC && tar xf -" 2>/dev/null
  A=$("$SSH" -o BatchMode=yes h2dtn "cd $H2 && find $d -type f -exec md5sum {} + | sort -k2")
  B=$("$SSH" -o BatchMode=yes "$CH" "cd $CARC && find $d -type f -exec md5sum {} + | sort -k2" 2>/dev/null | grep -v "reloaded\|python/3")
  if [ "$A" = "$B" ]; then
    echo "  copied $d ($(printf '%s\n' "$A" | wc -l) files, md5 identical; $(grep -h '^cpu' <<< "$("$SSH" -o BatchMode=yes h2dtn "cat $H2/$d/NODE")" | head -1))" | tee -a "$LOG"
  else
    echo "  MD5 DIFFERS after copying $d (CARC copy left in place; check by hand)" | tee -a "$LOG"
  fi
done
