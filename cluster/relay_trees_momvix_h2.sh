#!/bin/bash
# moments_vix tree workstream (2026-10-06): relay what Hoffman2 computed into the CARC root, so the
# CARC merge (cluster/slurm/trees_momvix_merge.sbatch) reduces Hoffman2 chunks with the CARC ones.
#   1. Hoffman2 (/u/scratch/j/jamesdc1/harxhar-momvix-trees): every FINISHED chunk dir (DONE present) of
#      results/linear_subsection_trees_mask_momvix and results/linear_subsection_trees_tuned_mask_momvix,
#      the gate outputs (results/trees_momvix_gates/h2_*: CSVs and flags) and the audit run
#      (results/trees_momvix_h2audit: the new bucket's chunk 0 recomputed on Hoffman2) -> a local
#      tarball (a direct ssh-to-ssh pipe truncates here) -> also extracted locally (the record)
#   2. CARC (/scratch1/jc_905/harxhar-momvix-trees): the tarball into results/trees_momvix_h2relay/, md5
#      of every relayed file == Hoffman2's, then each relayed chunk dir is copied into the results tree
#      ONLY where CARC has no DONE for it (a CARC chunk is never overwritten); the gate / audit files are
#      copied to their own paths (no CARC file has those paths)
# Run from Git Bash locally:  bash cluster/relay_trees_momvix_h2.sh
set -euo pipefail
cd "$(dirname "$0")/.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
H2=/u/scratch/j/jamesdc1/harxhar-momvix-trees
CARC=/scratch1/jc_905/harxhar-momvix-trees
STAGE_DIR=${STAGE_DIR:-${TMPDIR:-/tmp}/relay_trees_momvix_h2}
mkdir -p "$STAGE_DIR"
sums() { awk '{p = $2; sub(/^\*/, "", p); print p, $1}' | sort; }
LIST=$("$SSH" -o BatchMode=yes hoffman2 "cd $H2 && {
  for D in \$(find results/linear_subsection_trees_mask_momvix results/linear_subsection_trees_tuned_mask_momvix -path '*/chunks/c*' -name DONE -printf '%h\n' 2>/dev/null); do find \$D -type f; done
  find results/trees_momvix_gates -maxdepth 2 -type f \( -name '*.csv' -o -name GATE_OK -o -name 'PASS_*' -o -name H2_HOST \) 2>/dev/null
  find results/trees_momvix_h2audit -type f 2>/dev/null; } | sort" | tr -d '\r')
N=$(printf '%s\n' "$LIST" | grep -c . || true)
echo "relaying $N files from Hoffman2"
[ "$N" -gt 0 ] || exit 1
printf '%s\n' "$LIST" > "$STAGE_DIR/list.txt"
"$SSH" -o BatchMode=yes hoffman2 "cd $H2 && tar czf - -T -" < "$STAGE_DIR/list.txt" > "$STAGE_DIR/h2.tgz"
H2_SUMS=$("$SSH" -o BatchMode=yes hoffman2 "cd $H2 && xargs md5sum" < "$STAGE_DIR/list.txt" | sums)
# local record of the gate / audit / host files (small); the chunk npz stay in the tarball
tar xzf "$STAGE_DIR/h2.tgz" --wildcards 'results/trees_momvix_gates/*' 'results/trees_momvix_h2audit/*' 2>/dev/null || true
"$SSH" -o BatchMode=yes usc-discovery "mkdir -p $CARC/results/trees_momvix_h2relay && cd $CARC/results/trees_momvix_h2relay && tar xzf -" < "$STAGE_DIR/h2.tgz" 2>/dev/null
C_SUMS=$("$SSH" -o BatchMode=yes usc-discovery "cd $CARC/results/trees_momvix_h2relay && xargs md5sum" < "$STAGE_DIR/list.txt" 2>/dev/null | sums)
[ "$H2_SUMS" = "$C_SUMS" ] || { echo "RELAY FAILED: CARC md5 != Hoffman2 md5"; diff <(printf '%s\n' "$H2_SUMS") <(printf '%s\n' "$C_SUMS") | head; exit 1; }
echo "relay md5 OK: $N files on CARC under results/trees_momvix_h2relay == Hoffman2"
"$SSH" -o BatchMode=yes usc-discovery "cd $CARC && set -e
for D in \$(cd results/trees_momvix_h2relay && find results -path '*/chunks/c*' -name DONE -printf '%h\n' | sort); do
  if [ -f \$D/DONE ]; then echo \"  CARC has \$D (DONE): kept, the Hoffman2 copy stays in the relay dir\"; continue; fi
  mkdir -p \$(dirname \$D); rm -rf \$D; cp -rp results/trees_momvix_h2relay/\$D \$D; echo \"  copied \$D\"
done
cd results/trees_momvix_h2relay && for F in \$(find results/trees_momvix_gates results/trees_momvix_h2audit -type f 2>/dev/null); do mkdir -p ../\$(dirname \${F#results/}); cp -p \$F ../\${F#results/}; done
echo 'relay into the results tree done'" 2>&1 | grep -v "reloaded with a version change\|python/3\|^$"
