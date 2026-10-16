#!/bin/bash
# 2026-10-06 (moments_vix LSTMs): build the Hoffman2 root /u/scratch/j/jamesdc1/harxhar-momvix-lstm from
# the CARC deployment root /scratch1/jc_905/harxhar-momvix-lstm (cluster/slurm/ship_momvix_lstm_carc.sh):
#   * code: src/, specs/, experiments/, cluster/ of the CARC root (its bytes, relayed through this machine);
#   * data: data/*.parquet copied on Hoffman2 from the I5 root harxhar-optuna, md5 == the CARC root's;
#   * torch: TORCH_TGZ (CARC's site-packages/torch + dist-info, downloaded before) + its md5;
#   * then cluster/momvix_lstm_h2_setup.sh builds CARC's runtime with torch.
# One ssh session at a time.  Run locally:  TORCH_TGZ=<path> bash cluster/momvix_lstm_h2_ship.sh
set -euo pipefail
cd "$(dirname "$0")/.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
CARC=/scratch1/jc_905/harxhar-momvix-lstm
H2=/u/scratch/j/jamesdc1/harxhar-momvix-lstm
OPT=/u/scratch/j/jamesdc1/harxhar-optuna
: "${TORCH_TGZ:?TORCH_TGZ=<local torch.tgz>}"
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
MINE=(cluster/momvix_lstm_h2_setup.sh cluster/momvix_lstm_h2_task.sh cluster/submit_momvix_lstm_h2.sh)

# 1. CARC: the code and every md5 (code + data)
"$SSH" -o BatchMode=yes usc-discovery "cd $CARC && tar czf - --exclude=__pycache__ src specs experiments cluster" > "$STAGE/code.tgz"
"$SSH" -o BatchMode=yes usc-discovery "cd $CARC && find src specs experiments cluster data -name __pycache__ -prune -o -type f -print | sort | xargs md5sum" > "$STAGE/carc.md5"
md5sum "$TORCH_TGZ" | awk '{print $1 "  torch.tgz"}' > "$STAGE/torch.tgz.md5"
cp -p "$TORCH_TGZ" "$STAGE/torch.tgz"
cp -p "${MINE[@]}" "$STAGE/"
echo "code $(du -h "$STAGE/code.tgz" | cut -f1), torch $(du -h "$STAGE/torch.tgz" | cut -f1), $(wc -l < "$STAGE/carc.md5") md5 lines"

# 2. Hoffman2 (data-transfer node): one tar of everything
"$SSH" -o BatchMode=yes hoffman2 "mkdir -p $H2/logs $H2/results $H2/data && test ! -e $H2/carc_env"
( cd "$STAGE" && tar cf - code.tgz carc.md5 torch.tgz torch.tgz.md5 "$(basename "${MINE[0]}")" "$(basename "${MINE[1]}")" "$(basename "${MINE[2]}")" ) \
  | "$SSH" -o BatchMode=yes h2dtn "cd $H2 && tar xf -"

# 3. Hoffman2: unpack, data from the I5 root, md5 of everything against CARC's, the runtime
"$SSH" -o BatchMode=yes hoffman2 "bash -lc 'set -e; cd $H2
tar xzf code.tgz && mv -f momvix_lstm_h2_setup.sh momvix_lstm_h2_task.sh submit_momvix_lstm_h2.sh cluster/
for f in \$(grep \" data/\" carc.md5 | awk \"{print \\\$2}\"); do [ -f \$f ] || cp -p $OPT/\$f \$f; done
md5sum -c --quiet carc.md5 && echo \"md5 OK: \$(wc -l < carc.md5) files == CARC root\"
extra=\$(ls data/*.parquet | grep -v -x -F -f <(grep \" data/\" carc.md5 | awk \"{print \\\$2}\") || true)
[ -z \"\$extra\" ] || { echo \"extra data files: \$extra\"; exit 1; }
bash cluster/momvix_lstm_h2_setup.sh 2>&1 | tail -12'"
echo "Hoffman2 root ready: $H2"
