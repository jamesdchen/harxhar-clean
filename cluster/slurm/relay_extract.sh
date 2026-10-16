#!/bin/bash
# Pooled tree walks (2026-10-15): CARC side of cluster/transfer_pooltrees_h2_to_carc.sh -- extract the tar stream on stdin
# into a staging folder first, then rename each file into place (same filesystem: atomic), the npz files LAST.  A relay
# killed mid-copy (2026-10-15 01:07: a client-side timeout) then leaves no partial npz in the pieces folder (an npz there
# means "done" to the waves, the sweeps, the coverage and the merge).
#   ... | ssh carc "cd <root> && bash cluster/slurm/relay_extract.sh"
set -euo pipefail
ST=.relay_stage_$$_$(date +%s)
mkdir "$ST"
trap 'rm -rf "$ST" "$ST.list"' EXIT
tar xf - -C "$ST"
( cd "$ST" && find . -type f ! -name '*.npz' && find . -type f -name '*.npz' ) > "$ST.list"
while read -r f; do
  mkdir -p "$(dirname "$f")"
  mv -f "$ST/$f" "$f"
done < "$ST.list"
