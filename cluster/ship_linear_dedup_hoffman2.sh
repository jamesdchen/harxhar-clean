#!/bin/bash
# I1 (2026-09-29): ship the COMMITTED src/ (the de-duplicating executor, commit 47f7f9c) and
# specs/causal_tune_linear.py, the I1 Hoffman2 scripts, task files and canary references to a
# deployment root of their own on Hoffman2 (tar over native Windows OpenSSH), and FAIL unless
# every shipped file's remote md5 equals the local one.  Data: the bar-keyed data/*.parquet the
# loader reads (the local set minus the option-chain exports, = the CARC root's set); a file
# already in the older deployment root with the same md5 is copied there instead of uploaded.
# Then, on the login node under hpc-pi: every bucket of the task file resolves, and the scripts
# parse.  Run from Git Bash locally:
#   bash cluster/ship_linear_dedup_hoffman2.sh
set -euo pipefail
cd "$(dirname "$0")/.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
REMOTE=/u/scratch/j/jamesdc1/harxhar-dedup
OLDROOT=/u/scratch/j/jamesdc1/harxhar-subsection

# committed == shipped for the code the arms run
git diff --quiet HEAD -- src specs/causal_tune_linear.py \
  || { echo "src/ or the spec differs from HEAD: commit first"; exit 1; }
[ -z "$(git ls-files --others --exclude-standard src)" ] || { echo "untracked files under src/"; exit 1; }

# the canary references: the pre-dedup arms the master table scores
cp results/linear_subsection/arms_hoffman2/baseline/reclasso/tw2000/results_bar1600.csv \
   cluster/linear_dedup_ref_lasso_baseline_bar1600.csv
cp results/linear_subsection/baseline/bar1600/reclasso/tw2000/causal_tune_linear/incumbent_ols/results_bar1600.csv \
   cluster/linear_dedup_ref_ols_baseline_bar1600.csv

mapfile -t SRC < <(git ls-files src | grep -v '/__pycache__/')
FILES=(
  "${SRC[@]}"
  specs/causal_tune_linear.py
  cluster/linear_dedup_pack.sh
  cluster/linear_dedup_canary.sh
  cluster/linear_dedup_collect.sh
  cluster/submit_linear_dedup.sh
  cluster/linear_dedup_tasks_canary.txt
  cluster/linear_dedup_tasks_fleet.txt
  cluster/linear_dedup_ref_lasso_baseline_bar1600.csv
  cluster/linear_dedup_ref_ols_baseline_bar1600.csv
)
mapfile -t DATA < <(find data -maxdepth 1 -name '*.parquet' -not -name 'spxw_chain.parquet' -not -name 'spxw_spot.parquet' | sort)

sums() { awk '{p = $2; sub(/^\*/, "", p); print p, $1}' | sort; }

# 1. data: copy from the older root where the md5 matches, upload the rest
"$SSH" -o BatchMode=yes hoffman2 "mkdir -p $REMOTE/data $REMOTE/logs $REMOTE/results"
LOCAL_DATA=$(md5sum "${DATA[@]}" | sums)
OLD_DATA=$("$SSH" -o BatchMode=yes hoffman2 "cd $OLDROOT && md5sum data/*.parquet 2>/dev/null" | sums || true)
UPLOAD=()
for f in "${DATA[@]}"; do
  l=$(printf '%s\n' "$LOCAL_DATA" | awk -v f="$f" '$1 == f {print $2}')
  o=$(printf '%s\n' "$OLD_DATA" | awk -v f="$f" '$1 == f {print $2}')
  if [ -n "$o" ] && [ "$o" = "$l" ]; then
    "$SSH" -o BatchMode=yes hoffman2 "[ -f $REMOTE/$f ] || cp -p $OLDROOT/$f $REMOTE/$f"
  else
    UPLOAD+=("$f")
  fi
done
if [ "${#UPLOAD[@]}" -gt 0 ]; then
  echo "uploading ${#UPLOAD[@]} data files: ${UPLOAD[*]}"
  tar czf - "${UPLOAD[@]}" | "$SSH" -o BatchMode=yes hoffman2 "cd $REMOTE && tar xzf -"
fi

# 2. code, scripts, task files, references
tar czf - "${FILES[@]}" | "$SSH" -o BatchMode=yes hoffman2 "cd $REMOTE && tar xzf -"

# 3. every shipped file and every data file: remote md5 == local md5
LOCAL_ALL=$(md5sum "${FILES[@]}" "${DATA[@]}" | sums)
REMOTE_ALL=$(printf '%s\n' "${FILES[@]}" "${DATA[@]}" \
  | "$SSH" -o BatchMode=yes hoffman2 "cd $REMOTE && xargs md5sum" | sums)
BAD=$(diff <(printf '%s\n' "$LOCAL_ALL") <(printf '%s\n' "$REMOTE_ALL") || true)
N=$(printf '%s\n' "$LOCAL_ALL" | wc -l)
if [ -n "$BAD" ]; then
  echo "SHIP FAILED: remote md5 differs from local:"; printf '%s\n' "$BAD"; exit 1
fi
echo "md5 OK: $N files (${#SRC[@]} src, 1 spec, 8 I1 files, ${#DATA[@]} data)"
EXTRA=$("$SSH" -o BatchMode=yes hoffman2 "cd $REMOTE && ls data/*.parquet" | grep -v -x -F -f <(printf '%s\n' "${DATA[@]}") || true)
[ -z "$EXTRA" ] || { echo "FAILED: remote data/ holds files the local set does not: $EXTRA"; exit 1; }

# 4. login node: the buckets resolve, the scripts parse
"$SSH" -o BatchMode=yes hoffman2 "cd $REMOTE && source /u/local/apps/anaconda3/2023.03/etc/profile.d/conda.sh && conda activate hpc-pi && set -e
for s in cluster/linear_dedup_pack.sh cluster/linear_dedup_canary.sh cluster/linear_dedup_collect.sh cluster/submit_linear_dedup.sh; do bash -n \$s && echo \"bash -n ok: \$s\"; done
PYTHONPATH=\$PWD python - <<'EOF'
import sys, numpy, pandas
from src.data.loading import get_bucket
print('python', sys.version.split()[0], 'numpy', numpy.__version__, 'pandas', pandas.__version__)
bk = sorted({l.split()[1] for l in open('cluster/linear_dedup_tasks_fleet.txt') if l.strip()})
for b in bk:
    print(f'  bucket {b}: {len(get_bucket(b))} exog columns')
EOF"
echo "ship OK -> hoffman2:$REMOTE"
