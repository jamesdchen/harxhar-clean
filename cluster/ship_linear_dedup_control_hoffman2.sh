#!/bin/bash
# I1 attribution control (2026-09-29): a second Hoffman2 deployment root, identical to the de-dup
# root (cluster/ship_linear_dedup_hoffman2.sh: same src/, spec, data, pack script, environment)
# EXCEPT src/backtest/executor.py, which is the PRE-dedup file (git 47f7f9c^, the only file of src/
# that commit touched).  The arms listed by experiments/linear_dedup_control_plan.py run there, so
# control vs old isolates the cluster / environment and control vs new isolates the design change.
# FAILS unless every file's remote md5 equals the expected one.  Run from Git Bash locally:
#   bash cluster/ship_linear_dedup_control_hoffman2.sh
set -euo pipefail
cd "$(dirname "$0")/.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
REMOTE=/u/scratch/j/jamesdc1/harxhar-dedup-control
DEPLOY=/u/scratch/j/jamesdc1/harxhar-dedup
PRE=47f7f9c^

"$SSH" -o BatchMode=yes hoffman2 "mkdir -p $REMOTE/logs $REMOTE/results $REMOTE/cluster && cd $DEPLOY && cp -rp src specs data $REMOTE/"
TMP=$(mktemp -d)
mkdir -p "$TMP/src/backtest" "$TMP/cluster"
git show "$PRE:src/backtest/executor.py" > "$TMP/src/backtest/executor.py"
cp cluster/linear_dedup_pack.sh cluster/linear_dedup_control_tasks.txt cluster/submit_linear_dedup_control.sh \
   cluster/linear_dedup_collect.sh "$TMP/cluster/"
( cd "$TMP" && tar czf - src/backtest/executor.py cluster ) | "$SSH" -o BatchMode=yes hoffman2 "cd $REMOTE && tar xzf -"

mapfile -t SRC < <(git ls-files src | grep -v '/__pycache__/' | grep -v '^src/backtest/executor.py$')
mapfile -t DATA < <(find data -maxdepth 1 -name '*.parquet' -not -name 'spxw_chain.parquet' -not -name 'spxw_spot.parquet' | sort)
SAME=("${SRC[@]}" specs/causal_tune_linear.py cluster/linear_dedup_pack.sh cluster/linear_dedup_control_tasks.txt
      cluster/submit_linear_dedup_control.sh cluster/linear_dedup_collect.sh "${DATA[@]}")
sums() { awk '{p = $2; sub(/^\*/, "", p); print p, $1}' | sort; }
LOCAL=$( { md5sum "${SAME[@]}" | sums; printf 'src/backtest/executor.py %s\n' "$(md5sum < "$TMP/src/backtest/executor.py" | cut -d' ' -f1)"; } | sort)
REMOTE_SUMS=$(printf '%s\n' "${SAME[@]}" src/backtest/executor.py | "$SSH" -o BatchMode=yes hoffman2 "cd $REMOTE && xargs md5sum" | sums)
rm -rf "$TMP"
BAD=$(diff <(printf '%s\n' "$LOCAL") <(printf '%s\n' "$REMOTE_SUMS") || true)
[ -z "$BAD" ] || { echo "SHIP FAILED:"; printf '%s\n' "$BAD"; exit 1; }
echo "md5 OK: $(printf '%s\n' "$LOCAL" | wc -l) files; executor = $PRE ($(git rev-parse --short "$PRE"))"
"$SSH" -o BatchMode=yes hoffman2 "cd $REMOTE && grep -c 'session_edge' src/backtest/executor.py || true; bash -n cluster/submit_linear_dedup_control.sh && echo 'bash -n ok'"
echo "ship OK -> hoffman2:$REMOTE"
