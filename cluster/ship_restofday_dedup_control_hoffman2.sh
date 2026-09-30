#!/bin/bash
# I6 attribution control (2026-09-29): a second Hoffman2 deployment root, identical to the I6 root
# (cluster/ship_restofday_dedup_hoffman2.sh: same src/, specs, data, pack script, environment) EXCEPT
# src/backtest/executor.py, which is the PRE-dedup file (git 47f7f9c^, the only file of src/ that
# commit touched), with the control task files and submit script and a manifest of its own (at the
# path the pack script checks).  FAILS unless every file's remote md5 equals the expected one.
# Run from Git Bash locally:
#   bash cluster/ship_restofday_dedup_control_hoffman2.sh
set -euo pipefail
cd "$(dirname "$0")/.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
REMOTE=/u/scratch/j/jamesdc1/harxhar-restofday-dedup-control
DEPLOY=/u/scratch/j/jamesdc1/harxhar-restofday-dedup
PRE=47f7f9c^

git diff --quiet HEAD -- src specs/causal_tune_rest_of_day.py specs/causal_tune_linear.py \
  || { echo "src/ or a spec differs from HEAD: commit first"; exit 1; }
"$SSH" -o BatchMode=yes hoffman2 "mkdir -p $REMOTE/logs $REMOTE/results $REMOTE/cluster && cd $DEPLOY && cp -rp src specs data $REMOTE/"
TMP=$(mktemp -d)
mkdir -p "$TMP/src/backtest" "$TMP/cluster"
git show "$PRE:src/backtest/executor.py" > "$TMP/src/backtest/executor.py"
mapfile -t CTRL < <(ls cluster/restofday_dedup_control_*.txt)
cp cluster/restofday_dedup_pack.sh cluster/submit_restofday_dedup_control.sh "${CTRL[@]}" "$TMP/cluster/"

mapfile -t SRC < <(git ls-files src | grep -v '/__pycache__/' | grep -v '^src/backtest/executor.py$')
mapfile -t DATA < <(find data -maxdepth 1 -name '*.parquet' -not -name 'spxw_chain.parquet' -not -name 'spxw_spot.parquet' | sort)
SAME=("${SRC[@]}" specs/causal_tune_rest_of_day.py specs/causal_tune_linear.py cluster/restofday_dedup_pack.sh
      cluster/submit_restofday_dedup_control.sh "${CTRL[@]}" "${DATA[@]}")
# the control root's manifest (the pack script checks cluster/restofday_dedup_manifest.md5)
{ md5sum "${SAME[@]}"; ( cd "$TMP" && md5sum src/backtest/executor.py ); } \
  | sed 's/^\([0-9a-f]\{32\}\) \*/\1  /' | sort -k2 > "$TMP/cluster/restofday_dedup_manifest.md5"
( cd "$TMP" && tar czf - src/backtest/executor.py cluster ) | "$SSH" -o BatchMode=yes hoffman2 "cd $REMOTE && tar xzf -"

sums() { awk '{p = $2; sub(/^\*/, "", p); print p, $1}' | sort; }
LOCAL=$(sums < "$TMP/cluster/restofday_dedup_manifest.md5")
REMOTE_SUMS=$(printf '%s\n' "${SAME[@]}" src/backtest/executor.py | "$SSH" -o BatchMode=yes hoffman2 "cd $REMOTE && xargs md5sum" | sums)
rm -rf "$TMP"
BAD=$(diff <(printf '%s\n' "$LOCAL") <(printf '%s\n' "$REMOTE_SUMS") || true)
[ -z "$BAD" ] || { echo "SHIP FAILED:"; printf '%s\n' "$BAD"; exit 1; }
echo "md5 OK: $(printf '%s\n' "$LOCAL" | wc -l) files; executor = $PRE ($(git rev-parse --short "$PRE"))"
"$SSH" -o BatchMode=yes hoffman2 "cd $REMOTE && md5sum -c --quiet cluster/restofday_dedup_manifest.md5 && echo 'manifest OK'; echo \"session_edge in executor: \$(grep -c session_edge src/backtest/executor.py || true)\"; bash -n cluster/submit_restofday_dedup_control.sh && echo 'bash -n ok'"
echo "ship OK -> hoffman2:$REMOTE"
