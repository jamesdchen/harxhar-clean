#!/bin/bash
# Feature MCS trees, the bucket designs (2026-10-13): pull every FINISHED Hoffman2 chunk (a folder with DONE) of
# /u/scratch/j/jamesdc1/harxhar-mcstrees-h2/results/feature_mcs_trees_work/chunks2 into the same local path, through the
# data-transfer node (H2HOST, default h2dtn: the same /u/scratch), md5 of every pulled file against the remote.  The local
# collect step (python experiments/feature_mcs_trees.py collect_buckets_xgb_rf) then reads them.
# Run from Git Bash locally:   [H2HOST=hoffman2] bash cluster/pull_feature_mcs_trees_h2.sh
set -euo pipefail
cd "$(dirname "$0")/.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
H2HOST=${H2HOST:-h2dtn}
H2=/u/scratch/j/jamesdc1/harxhar-mcstrees-h2
R=results/feature_mcs_trees_work/chunks2
sums() { awk '{p = $2; sub(/^\*/, "", p); print p, $1}' | sort; }
LIST=$("$SSH" -o BatchMode=yes "$H2HOST" "cd $H2 && for d in \$(find $R -mindepth 3 -maxdepth 3 -type d -name 'c*' 2>/dev/null); do [ -f \$d/DONE ] && find \$d -type f; done | sort" | tr -d '\r')
N=$(printf '%s\n' "$LIST" | grep -c . || true)
echo "pulling $N files ($(printf '%s\n' "$LIST" | grep -c '/DONE$' || true) finished chunks)"
[ "$N" -gt 0 ] || exit 1
mkdir -p "$R"
printf '%s\n' "$LIST" | "$SSH" -o BatchMode=yes "$H2HOST" "cd $H2 && tar czf - -T -" | tar xzf -
REMOTE_SUMS=$(printf '%s\n' "$LIST" | "$SSH" -o BatchMode=yes "$H2HOST" "cd $H2 && xargs md5sum" | sums)
LOCAL_SUMS=$(printf '%s\n' "$LIST" | xargs md5sum | sums)
BAD=$(diff <(printf '%s\n' "$REMOTE_SUMS") <(printf '%s\n' "$LOCAL_SUMS") || true)
[ -z "$BAD" ] || { echo "PULL FAILED: md5 differs:"; printf '%s\n' "$BAD" | head; exit 1; }
echo "pull OK: $N files, md5 identical to Hoffman2"
