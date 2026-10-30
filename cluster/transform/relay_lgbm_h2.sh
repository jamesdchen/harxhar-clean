#!/bin/bash
# 2026-10-30 transform ablation, LightGBM confirmation: relay finished Hoffman2 pieces of a variant (tx_lgbm_h2.sh) into
# CARC's lgbm_conf/pieces, where CARC's own array task for that part then finds the piece complete and exits.  Run on
# the PC (native OpenSSH; one stream per call, nothing processed locally).  A piece is relayed only when
#   - the Hoffman2 gate passed ($TX/gate/passed_lgbm_h2: V6 part 2/32 bit-identical to CARC's piece), and
#   - it is complete on Hoffman2 (its .json sidecar is written last), and
#   - CARC has no file of that piece yet (no CARC task has started it; a started part is left to CARC).
# Each piece goes into pieces/.relay/ first, is md5-checked there, then moved in place (parquet first, sidecar last).
# usage: bash cluster/transform/relay_lgbm_h2.sh V7 [--apply]      (default: dry run)
set -euo pipefail
V=${1:?variant}
APPLY=${2:-}
SSH=/c/Windows/System32/OpenSSH/ssh.exe
O="-o BatchMode=yes -o ConnectTimeout=60"
HT=/u/scratch/j/jamesdc1/tx
CT=/scratch1/jc_905/tx
HP=$HT/runs/$V/lgbm_conf/pieces
CP=$CT/runs/$V/lgbm_conf/pieces
gate=$("$SSH" $O h2dtn "cat $HT/gate/passed_lgbm_h2 2> /dev/null || true" 2> /dev/null)
[ -n "$gate" ] || { echo "relay $V: Hoffman2 gate not passed (yet); nothing relayed"; exit 0; }
done_h2=$("$SSH" $O h2dtn "cd $HP 2> /dev/null && ls *.json 2> /dev/null | grep -v ckpt.json | sed 's/\.json$//' || true" 2> /dev/null)
[ -n "$done_h2" ] || { echo "relay $V: no complete Hoffman2 piece yet"; exit 0; }
have_carc=$("$SSH" $O usc-discovery "cd $CP && ls" 2> /dev/null | sed -E 's/\.(ckpt\.json|json|parquet)$//' | sort -u)
todo=$(comm -23 <(echo "$done_h2" | sort -u) <(echo "$have_carc"))
echo "relay $V: gate passed on [$gate]; Hoffman2 complete $(echo "$done_h2" | wc -l), CARC already has $(comm -12 <(echo "$done_h2" | sort -u) <(echo "$have_carc") | wc -l) of those, to relay $(echo "$todo" | grep -c . || true)"
[ -n "$todo" ] || exit 0
echo "$todo" | sed 's/^/  /'
[ "$APPLY" = --apply ] || { echo "dry run (pass --apply)"; exit 0; }
files=$(echo "$todo" | sed 's/$/.parquet/'; echo "$todo" | sed 's/$/.json/')
# shellcheck disable=SC2086
"$SSH" $O h2dtn "cd $HP && md5sum $(echo $files) > /tmp/RELAY_MD5_$V && tar -cf - $(echo $files) -C /tmp RELAY_MD5_$V; rm -f /tmp/RELAY_MD5_$V" 2> /dev/null |
  "$SSH" $O usc-discovery "set -e; mkdir -p $CP/.relay && cd $CP/.relay && tar -xf - && md5sum -c --quiet RELAY_MD5_$V && rm RELAY_MD5_$V &&
    for p in $(echo $todo); do
      if [ -e $CP/\$p.json ] || [ -e $CP/\$p.parquet ] || [ -e $CP/\$p.ckpt.json ]; then echo \"  skip \$p (CARC started it meanwhile)\"; rm -f \$p.parquet \$p.json; continue; fi
      mv -f \$p.parquet $CP/ && mv -f \$p.json $CP/ && echo \"  relayed \$p\"
    done; echo \"relay $V: md5 OK; CARC pieces now \$(ls $CP/*.json | grep -vc ckpt.json) complete\"" 2>&1 | grep -v -E "reloaded|python/3"
