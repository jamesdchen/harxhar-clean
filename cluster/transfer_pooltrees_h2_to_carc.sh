#!/bin/bash
# Pooled tree walks (2026-10-07): copy FINISHED Hoffman2 chunks and row pieces (finished = its .node exists; files: npz,
# .meta.json, .node, and a chunk's .log / a piece's .claim) into the CARC deployment at the same path when CARC has no
# npz there yet -- the CARC tasks skip an existing npz, pooled_trees_merge.py joins the pieces into chunks and the dump
# reads one complete root.  Only classes whose gate reproduced the bank byte for byte run there.  Streams Hoffman2 (DTN)
# -> this machine -> CARC (file lists on stdin, so thousands of pieces are fine); md5 compared on both ends.
#   bash cluster/transfer_pooltrees_h2_to_carc.sh     (every arm under the Hoffman2 chunks folder)
set -euo pipefail
cd "$(dirname "$0")/.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
H2=/u/scratch/j/jamesdc1/harxhar-pooltrees-h2
CARC=${CARC_ROOT:-/scratch1/jc_905/harxhar-pooltrees}   # the new root after the 2026-10-13 /scratch1 incident: CARC_ROOT=...-v2
CH=${CARC_HOST:-usc-discovery1}
C=results/pooled_trees_2026-10-07/chunks
LOG=logs/transfer_pooltrees_h2_to_carc.log
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
mkdir -p logs
"$SSH" -o BatchMode=yes "${H2HOST:-h2dtn}" "cd $H2 && find $C -name 'chunk_*.node' 2>/dev/null | sed 's/\.node\$//' | sort" > "$TMP/h2"
[ -s "$TMP/h2" ] || { echo "no finished Hoffman2 chunks or pieces"; exit 0; }
# the CARC listing must END with the marker: a dropped connection (USC VPN, 2026-10-14 01:28) gave an empty list, and
# every Hoffman2 piece then looked missing on CARC
"$SSH" -o BatchMode=yes "$CH" "cd $CARC && find $C -name 'chunk_*.npz' ! -name '*.part.npz' 2>/dev/null | sed 's/\.npz\$//' | sort; echo __LIST_END__" 2>/dev/null \
  | grep -v "reloaded\|python/3" > "$TMP/carc.raw" || true
[ "$(tail -1 "$TMP/carc.raw")" = __LIST_END__ ] || { echo "CARC listing incomplete (connection?); nothing copied"; exit 1; }
grep -v '^__LIST_END__$' "$TMP/carc.raw" > "$TMP/carc" || true
comm -23 "$TMP/h2" "$TMP/carc" > "$TMP/todo.all"
head -n "${MAXN:-100000}" "$TMP/todo.all" > "$TMP/todo"   # MAXN: a batch that fits one call (no pipe: SIGPIPE + pipefail)
echo "$(date +%F_%T) Hoffman2 finished: $(wc -l < "$TMP/h2"); missing on CARC: $(wc -l < "$TMP/todo.all"); copying now: $(wc -l < "$TMP/todo")" | tee -a "$LOG"
[ -s "$TMP/todo" ] || exit 0
while read -r s; do
  printf '%s.npz\n%s.meta.json\n%s.node\n' "$s" "$s" "$s"
  case "$s" in */pieces/*) printf '%s.claim\n' "$s" ;; *) printf '%s.log\n' "$s" ;; esac
done < "$TMP/todo" > "$TMP/files"
# staged extraction (cluster/slurm/relay_extract.sh): files are renamed into place after the whole stream arrived, npz last
"$SSH" -o BatchMode=yes "${H2HOST:-h2dtn}" "cd $H2 && tar cf - -T -" < "$TMP/files" | "$SSH" -o BatchMode=yes "$CH" "cd $CARC && bash cluster/slurm/relay_extract.sh" 2>/dev/null
"$SSH" -o BatchMode=yes "${H2HOST:-h2dtn}" "cd $H2 && xargs md5sum" < "$TMP/files" | sort -k2 > "$TMP/a"
"$SSH" -o BatchMode=yes "$CH" "cd $CARC && xargs md5sum" < "$TMP/files" 2>/dev/null | grep -v "reloaded\|python/3" | sort -k2 > "$TMP/b"
if cmp -s "$TMP/a" "$TMP/b"; then
  echo "  copied $(wc -l < "$TMP/todo") chunks / pieces ($(wc -l < "$TMP/files") files), md5 identical" | tee -a "$LOG"
else
  echo "  MD5 DIFFERS after the copy; check by hand" | tee -a "$LOG"; exit 1
fi
