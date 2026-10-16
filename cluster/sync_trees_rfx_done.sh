#!/bin/bash
# rfx (2026-10-07): the piece manifest shared by CARC and Hoffman2 when the same fixed-settings chunks are on both clusters'
# lists (still-pending CARC packs duplicated on Hoffman2 AVX, cluster/submit_trees_rfx_h2.sh extra): the DONE flags of the two
# roots, synchronised both ways, so a chunk finished on one cluster is skipped by the other from then on (the CARC pack and
# the Hoffman2 task both skip a chunk folder with DONE; nothing is cancelled; the forest is byte-identical on the two CPU
# classes, results/trees_rfdev_preview163/xcpu_carc/SUMMARY.md).
#   1. Hoffman2 -> CARC: cluster/relay_trees_rfx_h2.sh (md5-checked; copied only where CARC has no DONE and is not running it)
#   2. CARC -> Hoffman2: every chunk DONE on CARC and not on Hoffman2 gets a marker folder in the Hoffman2 root (DONE +
#      FROM_CARC: skipped by the Hoffman2 tasks, never relayed back)
# Run from Git Bash locally, repeatedly while both run:  TOKEN=rf400 KIND=nm bash cluster/sync_trees_rfx_done.sh
set -euo pipefail
cd "$(dirname "$0")/.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
TOKEN=${TOKEN:?TOKEN}
KIND=${KIND:?KIND fx or nm}
case "$KIND" in fx) RR=results/linear_subsection_trees_mask_$TOKEN ;; nm) RR=results/linear_subsection_trees_nomask_$TOKEN ;; *) exit 1 ;; esac
H2=/u/scratch/j/jamesdc1/harxhar-$TOKEN-h2
CARC=/scratch1/jc_905/harxhar-$TOKEN-trees
bash cluster/relay_trees_rfx_h2.sh || echo "relay: nothing relayed"
C_DONE=$("$SSH" -o BatchMode=yes usc-discovery "cd $CARC && find $RR -path '*/chunks/c*' -name DONE -printf '%h\n' | grep -E '/c[0-9]+\$' | sort" 2>/dev/null | tr -d '\r' | grep -v "reloaded with a version change\|python/3\|^$" || true)
H_DONE=$("$SSH" -o BatchMode=yes hoffman2 "cd $H2 && find $RR -path '*/chunks/c*' -name DONE -printf '%h\n' | grep -E '/c[0-9]+\$' | sort" | tr -d '\r' || true)
NEW=$(comm -23 <(printf '%s\n' "$C_DONE" | grep . | sort) <(printf '%s\n' "$H_DONE" | grep . | sort) || true)
NN=$(printf '%s\n' "$NEW" | grep -c . || true)
if [ "$NN" -gt 0 ]; then
  printf '%s\n' "$NEW" | "$SSH" -o BatchMode=yes hoffman2 "cd $H2 && while read -r D; do [ -n \"\$D\" ] || continue; mkdir -p \"\$D\"; echo carc > \"\$D/FROM_CARC\"; touch \"\$D/DONE\"; done"
fi
echo "sync $(date +%T): CARC DONE $(printf '%s\n' "$C_DONE" | grep -c . || true), Hoffman2 DONE $(printf '%s\n' "$H_DONE" | grep -c . || true); $NN CARC chunks marked on Hoffman2"
