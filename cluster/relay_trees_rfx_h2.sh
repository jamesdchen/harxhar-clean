#!/bin/bash
# rfx (2026-10-07): relay the Hoffman2 share of a forest's fixed-settings chunks (KIND fx = masked, nm = unmasked) into the
# CARC root, so the CARC merge reduces them with the CARC chunks (twin of cluster/relay_trees_momvix_h2.sh).
#   1. Hoffman2 (/u/scratch/j/jamesdc1/harxhar-<TOKEN>-h2): every FINISHED chunk dir (DONE) of the kind's results root
#      -> a local tarball (a direct ssh-to-ssh pipe truncates)
#   2. CARC (/scratch1/jc_905/harxhar-<TOKEN>-trees): into results/h2relay_<KIND>/, md5 of every file == Hoffman2's, then each
#      chunk dir copied into the results tree ONLY where CARC has no DONE for it (a CARC chunk is never overwritten); the
#      Hoffman2 host record (H2_HOST: host, CPU class and model) travels with the chunk.  A chunk a CARC pack is running now
#      (run.log written in the last 200 min, no DONE) is left to CARC; Hoffman2 chunk folders that only carry CARC's DONE
#      marker (FROM_CARC; cluster/sync_trees_rfx_done.sh) are not relayed.
# Run from Git Bash locally:  TOKEN=rf400 KIND=fx bash cluster/relay_trees_rfx_h2.sh
set -euo pipefail
cd "$(dirname "$0")/.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
TOKEN=${TOKEN:?TOKEN}
KIND=${KIND:?KIND fx or nm}
case "$KIND" in fx) RR=results/linear_subsection_trees_mask_$TOKEN ;; nm) RR=results/linear_subsection_trees_nomask_$TOKEN ;; *) exit 1 ;; esac
H2=/u/scratch/j/jamesdc1/harxhar-$TOKEN-h2
CARC=/scratch1/jc_905/harxhar-$TOKEN-trees
STAGE_DIR=${STAGE_DIR:-${TMPDIR:-/tmp}/relay_trees_rfx_h2}
mkdir -p "$STAGE_DIR"
sums() { awk '{p = $2; sub(/^\*/, "", p); print p, $1}' | sort; }
LIST=$("$SSH" -o BatchMode=yes hoffman2 "cd $H2 && for D in \$(find $RR -path '*/chunks/c*' -name DONE -printf '%h\n' 2>/dev/null | grep -v '_repeat\$'); do [ -f \$D/FROM_CARC ] || find \$D -type f; done | sort" | tr -d '\r')
N=$(printf '%s\n' "$LIST" | grep -c . || true)
echo "relaying $N files from Hoffman2 ($RR)"
[ "$N" -gt 0 ] || exit 1
printf '%s\n' "$LIST" > "$STAGE_DIR/list_$KIND.txt"
"$SSH" -o BatchMode=yes hoffman2 "cd $H2 && tar czf - -T -" < "$STAGE_DIR/list_$KIND.txt" > "$STAGE_DIR/h2_$KIND.tgz"
H2_SUMS=$("$SSH" -o BatchMode=yes hoffman2 "cd $H2 && xargs md5sum" < "$STAGE_DIR/list_$KIND.txt" | sums)
"$SSH" -o BatchMode=yes usc-discovery "mkdir -p $CARC/results/h2relay_$KIND && cd $CARC/results/h2relay_$KIND && tar xzf -" < "$STAGE_DIR/h2_$KIND.tgz" 2>/dev/null
C_SUMS=$("$SSH" -o BatchMode=yes usc-discovery "cd $CARC/results/h2relay_$KIND && xargs md5sum" < "$STAGE_DIR/list_$KIND.txt" 2>/dev/null | sums)
[ "$H2_SUMS" = "$C_SUMS" ] || { echo "RELAY FAILED: CARC md5 != Hoffman2 md5"; diff <(printf '%s\n' "$H2_SUMS") <(printf '%s\n' "$C_SUMS") | head; exit 1; }
echo "relay md5 OK: $N files"
"$SSH" -o BatchMode=yes usc-discovery "cd $CARC && n=0; k=0; r=0
for D in \$(cd results/h2relay_$KIND && find $RR -path '*/chunks/c*' -name DONE -printf '%h\n' | sort); do
  if [ -f \$D/DONE ]; then k=\$((k + 1)); continue; fi
  if [ -f \$D/run.log ] && [ -n \"\$(find \$D/run.log -mmin -200)\" ]; then r=\$((r + 1)); continue; fi
  mkdir -p \$(dirname \$D); rm -rf \$D; cp -rp results/h2relay_$KIND/\$D \$D; n=\$((n + 1))
done
echo \"copied \$n Hoffman2 chunks into $RR; \$k already DONE on CARC (kept); \$r running on CARC now (left to it)\"" 2>&1 | grep -v "reloaded with a version change\|python/3\|^$"
