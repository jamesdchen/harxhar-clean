#!/bin/bash
# rfx on Hoffman2 (2026-10-07): ship the Hoffman2 share of a dev-span forest's fixed-settings chunks (cluster/submit_trees_rfx_h2.sh)
# to /u/scratch/j/jamesdc1/harxhar-$TOKEN-h2 -- TOKEN-driven copy of cluster/ship_trees_rfh2_h2.sh: HEAD blobs for the campaign's
# unchanged code, the worktree for src/data/loading.py (added lines only), the wrapper, the task script, TOKEN's Hoffman2 line
# files and configuration; data copied on Hoffman2 from agent D's root where the md5 equals the local file.
# Run from Git Bash locally:  TOKEN=rf400 bash cluster/ship_trees_rfx_h2.sh
set -euo pipefail
TOKEN=${TOKEN:?TOKEN}
cd "$(dirname "$0")/.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
H2=/u/scratch/j/jamesdc1/harxhar-$TOKEN-h2
RT=/u/scratch/j/jamesdc1/harxhar-optuna
MASK=/scratch1/jc_905/harxhar-mask
STAGE_DIR=${STAGE_DIR:-${TMPDIR:-/tmp}/ship_trees_rfx_h2}
mkdir -p "$STAGE_DIR"
sums() { awk '{p = $2; sub(/^\*/, "", p); print p, $1}' | sort; }

mapfile -t SRC < <(git ls-tree -r --name-only HEAD src | grep -v '/__pycache__/' | grep -v -x 'src/data/loading.py')
FROM_HEAD=(
  "${SRC[@]}"
  specs/causal_tune_trees.py
  experiments/trees_cadence_reduce_chunks.py experiments/reduce_trees_tuned_chunks.py
)
FROM_WORKTREE=(
  src/data/loading.py
  specs/causal_tune_trees_rfdev.py
  cluster/trees_rfprev_h2_task.sh cluster/submit_trees_rfx_h2.sh cluster/ship_trees_rfx_h2.sh
  "cluster/${TOKEN}_rf_config.json"
)
for f in "cluster/trees_${TOKEN}_fx_h2lines.txt" "cluster/trees_${TOKEN}_nm_h2lines.txt"; do [ -f "$f" ] && FROM_WORKTREE+=("$f"); done
git diff --quiet HEAD -- "${FROM_HEAD[@]}" || { echo "SHIP REFUSED: a file shipped from HEAD differs from HEAD:"; git diff --stat HEAD -- "${FROM_HEAD[@]}"; exit 1; }
NUM=$(git diff --numstat HEAD -- src/data/loading.py | cut -f2)
[ "$NUM" = 0 ] || { echo "SHIP REFUSED: src/data/loading.py removes $NUM lines of HEAD"; exit 1; }
BADADD=$(git diff -U0 HEAD -- src/data/loading.py | grep '^+[^+]' | grep -v -E '^\+#|^\+SUBGROUPS\["[a-z_]+"\] = ' || true)
[ -z "$BADADD" ] || { echo "SHIP REFUSED: added loading.py lines other than comments / SUBGROUPS lines"; exit 1; }
for f in "${FROM_WORKTREE[@]}"; do
  [ -f "$f" ] || { echo "SHIP REFUSED: $f missing"; exit 1; }
  ! grep -q $'\r' "$f" || { echo "SHIP REFUSED: $f has CRLF line endings"; exit 1; }
done
mapfile -t DATA < <(find data -maxdepth 1 -name '*.parquet' -not -name 'spxw_chain.parquet' -not -name 'spxw_spot.parquet' | sort)

# 1. root, data, runtime manifest
LOCAL_DATA=$(md5sum "${DATA[@]}" | sums)
# RUNTIME_MD5=0 skips the (slow) manifest check on a re-ship; every task still checks in-process
# glibc 2.28 and that no shared object comes from outside the runtime root
if [ "${RUNTIME_MD5:-0}" = 1 ]; then
  "$SSH" -o BatchMode=yes hoffman2 "mkdir -p $H2/data $H2/logs $H2/results $H2/ref && cd $RT && md5sum -c --quiet runtime_manifest.md5 && echo \"runtime $RT: \$(wc -l < runtime_manifest.md5) files match their manifest\""
else
  "$SSH" -o BatchMode=yes hoffman2 "mkdir -p $H2/data $H2/logs $H2/results $H2/ref"
fi
RT_DATA=$("$SSH" -o BatchMode=yes hoffman2 "cd $RT && md5sum data/*.parquet" | sums)
UPLOAD=()
CP=""
for f in "${DATA[@]}"; do
  l=$(printf '%s\n' "$LOCAL_DATA" | awk -v f="$f" '$1 == f {print $2}')
  o=$(printf '%s\n' "$RT_DATA" | awk -v f="$f" '$1 == f {print $2}')
  if [ -n "$o" ] && [ "$o" = "$l" ]; then CP="$CP [ -f $H2/$f ] || cp -p $RT/$f $H2/$f;"; else UPLOAD+=("$f"); fi
done
[ -z "$CP" ] || "$SSH" -o BatchMode=yes hoffman2 "$CP true"
if [ "${#UPLOAD[@]}" -gt 0 ]; then
  echo "uploading ${#UPLOAD[@]} data files: ${UPLOAD[*]}"
  tar czf - "${UPLOAD[@]}" | "$SSH" -o BatchMode=yes hoffman2 "cd $H2 && tar xzf -"
fi

# 2. code
git -c core.autocrlf=false -c core.eol=lf archive --format=tar HEAD "${FROM_HEAD[@]}" > "$STAGE_DIR/head.tar"
tar cf "$STAGE_DIR/wt.tar" "${FROM_WORKTREE[@]}"
"$SSH" -o BatchMode=yes hoffman2 "cd $H2 && tar xf -" < "$STAGE_DIR/head.tar"
"$SSH" -o BatchMode=yes hoffman2 "cd $H2 && tar xf -" < "$STAGE_DIR/wt.tar"
LOCAL_ALL=$( { for f in "${FROM_HEAD[@]}"; do printf '%s %s\n' "$f" "$(git show "HEAD:$f" | md5sum | cut -d' ' -f1)"; done
               md5sum "${FROM_WORKTREE[@]}" "${DATA[@]}" | sums; } | sort)
REMOTE_ALL=$(printf '%s\n' "${FROM_HEAD[@]}" "${FROM_WORKTREE[@]}" "${DATA[@]}" | "$SSH" -o BatchMode=yes hoffman2 "cd $H2 && xargs md5sum" | sums)
BAD=$(diff <(printf '%s\n' "$LOCAL_ALL") <(printf '%s\n' "$REMOTE_ALL") || true)
[ -z "$BAD" ] || { echo "SHIP FAILED: remote md5 differs:"; printf '%s\n' "$BAD"; exit 1; }
echo "md5 OK: $(printf '%s\n' "$LOCAL_ALL" | wc -l) files (${#FROM_HEAD[@]} from HEAD $(git rev-parse --short HEAD), ${#FROM_WORKTREE[@]} from the worktree, ${#DATA[@]} data)"
# 4. scripts parse; the bucket is in the shipped loader (the runtime import runs in every task: the
#    login node's memory limit kills a numba / shap import there)
"$SSH" -o BatchMode=yes hoffman2 "cd $H2 && set -e
for s in cluster/trees_rfprev_h2_task.sh cluster/submit_trees_rfx_h2.sh; do bash -n \$s && echo \"bash -n ok: \$s\"; done
grep -n 'SUBGROUPS\[\"lean_vol_vix\"\]' src/data/loading.py; ls -d $RT/carc_env/harxhar/bin/python3.11"
echo "ship OK -> hoffman2:$H2"
