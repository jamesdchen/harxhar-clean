#!/bin/bash
# I6 (2026-09-29): ship the COMMITTED src/ (the de-duplicating executor, commit 47f7f9c),
# specs/causal_tune_rest_of_day.py and the per-bar spec it executes (specs/causal_tune_linear.py),
# the identity check, the I6 Hoffman2 scripts and task files, and the canary's references (agent A's
# de-duplicated per-bar table results/spxw_pnl/yhat_sub_ridge_live_feasible.parquet and its
# live_feasible ridge bar1600 arm files, at their own repo paths) to a deployment root of their own
# on Hoffman2 (tar over native Windows OpenSSH); write cluster/restofday_dedup_manifest.md5 (every
# shipped file + the data, md5sum format; every task re-checks it) and FAIL unless every shipped
# file's remote md5 equals the local one.  Data: the bar-keyed data/*.parquet the loader reads (the
# local set minus the option-chain exports, = agent A's I1 root's set); a file already in A's root
# with the same md5 is copied there instead of uploaded.  Then, on the login node under hpc-pi: the
# three buckets resolve, the scripts parse.  Run from Git Bash locally:
#   bash cluster/ship_restofday_dedup_hoffman2.sh
set -euo pipefail
cd "$(dirname "$0")/.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
REMOTE=/u/scratch/j/jamesdc1/harxhar-restofday-dedup
AROOT=/u/scratch/j/jamesdc1/harxhar-dedup
MANIFEST=cluster/restofday_dedup_manifest.md5
TWIN=results/linear_subsection_dedup/live_feasible/bar1600/ridge/tw2000/causal_tune_linear/ridge/live_feasible

# committed == shipped for the code the arms and the canary run
git diff --quiet HEAD -- src specs/causal_tune_rest_of_day.py specs/causal_tune_linear.py \
    experiments/check_restofday_identity.py \
  || { echo "src/, a spec or the identity check differs from HEAD: commit first"; exit 1; }
[ -z "$(git ls-files --others --exclude-standard src)" ] || { echo "untracked files under src/"; exit 1; }

mapfile -t SRC < <(git ls-files src | grep -v '/__pycache__/')
mapfile -t TASKS < <(ls cluster/restofday_dedup_tasks_*.txt)
CODE=(
  "${SRC[@]}"
  specs/causal_tune_rest_of_day.py
  specs/causal_tune_linear.py
  experiments/check_restofday_identity.py
  cluster/restofday_dedup_pack.sh
  cluster/restofday_dedup_canary.sh
  cluster/restofday_dedup_collect.sh
  cluster/submit_restofday_dedup.sh
  "${TASKS[@]}"
)
REFS=(
  results/spxw_pnl/yhat_sub_ridge_live_feasible.parquet
  "$TWIN/results_bar1600.csv"
  "$TWIN/results_bar1600_feature_health.csv"
)
mapfile -t DATA < <(find data -maxdepth 1 -name '*.parquet' -not -name 'spxw_chain.parquet' -not -name 'spxw_spot.parquet' | sort)
for f in "${CODE[@]}" "${REFS[@]}" "${DATA[@]}"; do [ -f "$f" ] || { echo "missing locally: $f"; exit 1; }; done

md5sum "${CODE[@]}" "${REFS[@]}" "${DATA[@]}" | sed 's/^\([0-9a-f]\{32\}\) \*/\1  /' > "$MANIFEST"
echo "wrote $MANIFEST: $(wc -l < "$MANIFEST") files (${#CODE[@]} code/scripts/tasks, ${#REFS[@]} references, ${#DATA[@]} data)"
FILES=("${CODE[@]}" "${REFS[@]}" "$MANIFEST")

sums() { awk '{p = $2; sub(/^\*/, "", p); print p, $1}' | sort; }

# 1. data: copy from agent A's root where the md5 matches, upload the rest
"$SSH" -o BatchMode=yes hoffman2 "mkdir -p $REMOTE/data $REMOTE/logs $REMOTE/results"
LOCAL_DATA=$(md5sum "${DATA[@]}" | sums)
A_DATA=$("$SSH" -o BatchMode=yes hoffman2 "cd $AROOT && md5sum data/*.parquet 2>/dev/null" | sums || true)
UPLOAD=()
for f in "${DATA[@]}"; do
  l=$(printf '%s\n' "$LOCAL_DATA" | awk -v f="$f" '$1 == f {print $2}')
  o=$(printf '%s\n' "$A_DATA" | awk -v f="$f" '$1 == f {print $2}')
  if [ -n "$o" ] && [ "$o" = "$l" ]; then
    "$SSH" -o BatchMode=yes hoffman2 "[ -f $REMOTE/$f ] || cp -p $AROOT/$f $REMOTE/$f"
  else
    UPLOAD+=("$f")
  fi
done
if [ "${#UPLOAD[@]}" -gt 0 ]; then
  echo "uploading ${#UPLOAD[@]} data files: ${UPLOAD[*]}"
  tar czf - "${UPLOAD[@]}" | "$SSH" -o BatchMode=yes hoffman2 "cd $REMOTE && tar xzf -"
fi

# 2. code, scripts, task files, references, manifest
tar czf - "${FILES[@]}" | "$SSH" -o BatchMode=yes hoffman2 "cd $REMOTE && tar xzf -"

# 3. every shipped file and every data file: remote md5 == local md5; the manifest checks there
LOCAL_ALL=$(md5sum "${FILES[@]}" "${DATA[@]}" | sums)
REMOTE_ALL=$(printf '%s\n' "${FILES[@]}" "${DATA[@]}" \
  | "$SSH" -o BatchMode=yes hoffman2 "cd $REMOTE && xargs md5sum" | sums)
BAD=$(diff <(printf '%s\n' "$LOCAL_ALL") <(printf '%s\n' "$REMOTE_ALL") || true)
N=$(printf '%s\n' "$LOCAL_ALL" | wc -l)
if [ -n "$BAD" ]; then
  echo "SHIP FAILED: remote md5 differs from local:"; printf '%s\n' "$BAD"; exit 1
fi
echo "md5 OK: $N files (${#SRC[@]} src, 2 specs, 1 check, 4 scripts, ${#TASKS[@]} task files, ${#REFS[@]} refs, manifest, ${#DATA[@]} data)"
EXTRA=$("$SSH" -o BatchMode=yes hoffman2 "cd $REMOTE && ls data/*.parquet" | grep -v -x -F -f <(printf '%s\n' "${DATA[@]}") || true)
[ -z "$EXTRA" ] || { echo "FAILED: remote data/ holds files the local set does not: $EXTRA"; exit 1; }

# 4. login node: the manifest holds, the buckets resolve, the scripts parse
"$SSH" -o BatchMode=yes hoffman2 "cd $REMOTE && source /u/local/apps/anaconda3/2023.03/etc/profile.d/conda.sh && conda activate hpc-pi && set -e
md5sum -c --quiet $MANIFEST && echo \"manifest OK on hoffman2 (\$(wc -l < $MANIFEST) files)\"
for s in cluster/restofday_dedup_pack.sh cluster/restofday_dedup_canary.sh cluster/restofday_dedup_collect.sh cluster/submit_restofday_dedup.sh; do bash -n \$s && echo \"bash -n ok: \$s\"; done
PYTHONPATH=\$PWD python - <<'EOF'
import ast, sys, numpy, pandas
from src.data.loading import get_bucket
for p in ('specs/causal_tune_rest_of_day.py', 'experiments/check_restofday_identity.py'):
    ast.parse(open(p).read())
    print('parses:', p)
print('python', sys.version.split()[0], 'numpy', numpy.__version__, 'pandas', pandas.__version__)
for b in ('all_features', 'baseline', 'live_feasible'):
    print(f'  bucket {b}: {len(get_bucket(b))} exog columns')
print('ref table rows', len(pandas.read_parquet('results/spxw_pnl/yhat_sub_ridge_live_feasible.parquet')))
EOF"
echo "ship OK -> hoffman2:$REMOTE"
