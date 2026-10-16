#!/bin/bash
# Pull the pooled moments_vix campaign from CARC (2026-10-06) into results/linear_subsection_pool_momvix/carc/
# (the cluster layout under results/pool_momvix/, minus the chunk result tables), check the merged
# files' md5 against the cluster's MD5SUMS, and (unless NO_FLAT=1) copy each arm's regular-hours file into
# the flat layout the yhat builder reads: results/linear_subsection/arms_hoffman2/<bucket>/<est>/tw2000/
# results_none_rth.csv (a new bucket subdirectory).  2026-10-06: the flat layout holds the Hoffman2 arms
# (the arms of record; experiments/build_pool_moments_vix_yhat.py), so pass NO_FLAT=1 when Hoffman2 has
# them; an existing different file is never overwritten.  BUCKET=moments_vol_vix pulls that fleet (its
# collect files carry the suffix _moments_vol_vix).  Run from Git Bash locally:
#   [BUCKET=moments_vol_vix] [NO_FLAT=1] [HOST=usc-discovery1] bash cluster/slurm/pull_pool_momvix_carc.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
HOST=${HOST:-usc-discovery}
REMOTE=/scratch1/jc_905/harxhar-pool-momvix
DEST=results/linear_subsection_pool_momvix/carc
B=${BUCKET:-moments_vix}
X=""
[ "$B" = moments_vix ] || X="_$B"
mkdir -p "$DEST"
"$SSH" -o BatchMode=yes "$HOST" "cd $REMOTE && [ -f results/pool_momvix/COLLECTED$X ] && tar czf - \
  results/pool_momvix/gate.csv results/pool_momvix/rth_filter_check.csv results/pool_momvix/collect$X.csv \
  results/pool_momvix/MD5SUMS$X results/pool_momvix/GATE_OK results/pool_momvix/COLLECTED$X \
  \$(ls logs/submitted_pool_momv*_carc.txt logs/pmv*_*.out) \
  \$(find results/pool_momvix/gate results/pool_momvix/fleet/$B -type f) \
  \$(find results/pool_momvix/fleet_chunks/$B -type f \( -name run.log -o -name SECONDS -o -name DONE -o -name metrics_table.csv \))" \
  | tar xzf - -C "$DEST"
(cd "$DEST" && md5sum -c "results/pool_momvix/MD5SUMS$X")
for e in ridge reclasso reclasticnet; do
  [ "${NO_FLAT:-0}" = 1 ] && break
  src="$DEST/results/pool_momvix/fleet/$B/none/$e/tw2000/causal_tune_linear/$e/$B/results_none_rth.csv"
  dst="results/linear_subsection/arms_hoffman2/$B/$e/tw2000/results_none_rth.csv"
  if [ -e "$dst" ]; then
    if cmp -s "$src" "$dst"; then echo "$dst already there (same bytes)"; continue; fi
    echo "$dst exists and differs: not overwritten"; exit 1
  fi
  mkdir -p "$(dirname "$dst")"
  cp -p "$src" "$dst"
  echo "copied $dst ($(wc -l < "$dst") lines)"
done
echo "pull OK ($B) -> $DEST"
