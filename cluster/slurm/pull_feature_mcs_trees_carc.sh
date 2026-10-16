#!/bin/bash
# Pull the tree agent's feature-MCS forecasts (2026-10-08) from CARC: results/feature_mcs_2026-10-08/trees/
# (<est>/<model>.npz [+ .json] for est in lgbm / xgb / rf, and the TREES_DONE flag; layout: the docstring of
# experiments/feature_mcs_report.py), md5-checked.  The CARC deployment root holding TREES_DONE is found under
# /scratch1/jc_905/*/ unless TREES_ROOT names it.  Then: python experiments/feature_mcs_report.py --workers 3
#
#   [TREES_ROOT=/scratch1/jc_905/<root>] bash cluster/slurm/pull_feature_mcs_trees_carc.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
T=results/feature_mcs_2026-10-08/trees
if [ -z "${TREES_ROOT:-}" ]; then
  FOUND=$("$SSH" -o BatchMode=yes usc-discovery "ls -d /scratch1/jc_905/*/$T/TREES_DONE 2>/dev/null" || true)
  [ "$(printf '%s\n' "$FOUND" | grep -c . || true)" -eq 1 ] || { echo "TREES_DONE found at: '${FOUND:-nowhere}' (want exactly one; set TREES_ROOT)"; exit 1; }
  TREES_ROOT=${FOUND%/$T/TREES_DONE}
fi
echo "trees root: $TREES_ROOT"
N=$("$SSH" -o BatchMode=yes usc-discovery "cd $TREES_ROOT && ls $T/*/*.npz | wc -l")
echo "remote: $N tree forecast files"
mkdir -p "$T"
"$SSH" -o BatchMode=yes usc-discovery "cd $TREES_ROOT && tar czf - $T" | tar xzf -
sums() { awk '{p = $2; sub(/^\*/, "", p); print p, $1}' | sort; }
REMOTE_SUMS=$("$SSH" -o BatchMode=yes usc-discovery "cd $TREES_ROOT && find $T -type f | sort | xargs md5sum" | sums)
LOCAL_SUMS=$(find "$T" -type f | sort | xargs md5sum | sums)
BAD=$(diff <(printf '%s\n' "$REMOTE_SUMS") <(printf '%s\n' "$LOCAL_SUMS") || true)
[ -z "$BAD" ] || { echo "PULL FAILED: md5 differs:"; printf '%s\n' "$BAD" | head; exit 1; }
echo "pull OK: $(ls $T/*/*.npz | wc -l) tree forecast files, md5 checked"
