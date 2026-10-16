#!/bin/bash
# Ship the pooled pre-tune study (2026-10-06) to a deployment root of its own on Hoffman2 and FAIL unless
# every shipped file's remote md5 equals the local one.  Mirrors cluster/pool_momvix_ship_h2.sh.
#
# Code: every tracked src/*.py and specs/causal_tune_linear.py as git stores them (LF; a file with no
# change against HEAD is taken from HEAD; the only working-tree change under src/ is the 2026-10-06
# buckets in src/data/loading.py), the runner experiments/pool_linear_pretune.py, the campaign's
# cluster/pool_pretune_* files and submit script (and the arm-f schedules once written).
# Data: the seven bar-keyed parquets the stored pooled arms read, copied on the cluster from the old root
# after its md5 there equals the local file's.
# Then, on the login node under hpc-pi: the scripts parse, the runner parses its arguments, the buckets
# resolve, the stored arms the gate compares with exist.
#
#   STAGE=<empty scratch dir> bash cluster/pool_pretune_ship_h2.sh
set -euo pipefail
cd "$(dirname "$0")/.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
REMOTE=/u/scratch/j/jamesdc1/harxhar-pool-pretune
OLDROOT=/u/scratch/j/jamesdc1/harxhar-subsection
STAGE=${STAGE:?set STAGE to an empty local scratch directory}
mkdir -p "$STAGE"
[ -z "$(ls -A "$STAGE")" ] || { echo "STAGE $STAGE is not empty"; exit 1; }

mapfile -t SRC < <(git ls-files src | grep '\.py$')
CODE=("${SRC[@]}" specs/causal_tune_linear.py)
OURS=(
  experiments/pool_linear_pretune.py
  cluster/pool_pretune_task.sh
  cluster/pool_pretune_gate.sh
  cluster/pool_pretune_check.py
  cluster/submit_pool_pretune.sh
)
mapfile -t TASKS < <(ls cluster/pool_pretune_tasks_*.txt)
mapfile -t SCHED < <(ls cluster/pool_pretune_sched/*.json 2>/dev/null || true)
OURS+=("${TASKS[@]}" "${SCHED[@]}")
DATA=(core_stats ewstock_stats releases spy_and_sentiment time_categories vix_and_voldemand vwstock_stats)

# 1. stage the code as git stores it, and our files as they are (LF)
for f in "${CODE[@]}"; do
  mkdir -p "$STAGE/$(dirname "$f")"
  if git diff --quiet HEAD -- "$f"; then git show "HEAD:$f" > "$STAGE/$f"; else cp "$f" "$STAGE/$f"; fi
done
for f in "${OURS[@]}"; do mkdir -p "$STAGE/$(dirname "$f")"; cp "$f" "$STAGE/$f"; done
CR=$(cd "$STAGE" && cat "${CODE[@]}" "${OURS[@]}" | tr -cd '\r' | wc -c)
[ "$CR" -eq 0 ] || { echo "staged files carry $CR CR bytes"; exit 1; }
grep -q 'SUBGROUPS\["lean_vol_vix"\]' "$STAGE/src/data/loading.py" || { echo "no lean_vol_vix bucket"; exit 1; }

# 2. data: copied on the cluster from the old root after an md5 match with the local file
COPY="set -e; mkdir -p $REMOTE/data $REMOTE/logs $REMOTE/results"
for d in "${DATA[@]}"; do
  f=data/$d.parquet
  l=$(md5sum < "$f" | cut -c1-32)
  COPY="$COPY; o=\$(md5sum < $OLDROOT/$f | cut -c1-32); [ \"\$o\" = $l ] || { echo \"$f: old root md5 \$o != local $l\"; exit 1; }; [ -f $REMOTE/$f ] || cp -p $OLDROOT/$f $REMOTE/$f"
done
"$SSH" -o BatchMode=yes hoffman2 "$COPY"

# 3. code and our files
(cd "$STAGE" && tar czf - "${CODE[@]}" "${OURS[@]}") | "$SSH" -o BatchMode=yes hoffman2 "cd $REMOTE && tar xzf -"

# 4. every shipped file: remote md5 == staged md5; data: remote md5 == local md5; no other data
sums() { awk '{p = $2; sub(/^\*/, "", p); print p, $1}' | sort; }
LOCAL_ALL=$( { (cd "$STAGE" && md5sum "${CODE[@]}" "${OURS[@]}"); for d in "${DATA[@]}"; do md5sum "data/$d.parquet"; done; } | sums)
REMOTE_ALL=$( { printf '%s\n' "${CODE[@]}" "${OURS[@]}"; for d in "${DATA[@]}"; do echo "data/$d.parquet"; done; } \
  | "$SSH" -o BatchMode=yes hoffman2 "cd $REMOTE && xargs md5sum" | sums)
BAD=$(diff <(printf '%s\n' "$LOCAL_ALL") <(printf '%s\n' "$REMOTE_ALL") || true)
[ -z "$BAD" ] || { echo "SHIP FAILED: remote md5 differs from local:"; printf '%s\n' "$BAD"; exit 1; }
EXTRA=$("$SSH" -o BatchMode=yes hoffman2 "cd $REMOTE && ls data/*.parquet" | grep -v -x -F -f <(for d in "${DATA[@]}"; do echo "data/$d.parquet"; done) || true)
[ -z "$EXTRA" ] || { echo "FAILED: remote data/ holds other files: $EXTRA"; exit 1; }
echo "md5 OK: $(printf '%s\n' "$LOCAL_ALL" | wc -l) files (${#SRC[@]} src, 1 spec, ${#OURS[@]} campaign files, ${#DATA[@]} data)"

# 5. login node: scripts parse, the runner parses, buckets resolve, stored arms exist
"$SSH" -o BatchMode=yes hoffman2 "bash -lc 'cd $REMOTE && bash -s'" <<'EOF'
set -e
for s in cluster/pool_pretune_task.sh cluster/pool_pretune_gate.sh cluster/submit_pool_pretune.sh; do
  bash -n "$s" && echo "bash -n ok: $s"
done
source /u/local/apps/anaconda3/2023.03/etc/profile.d/conda.sh && conda activate hpc-pi
PYTHONPATH=$PWD python experiments/pool_linear_pretune.py --help > /dev/null && echo "runner parses"
PYTHONPATH=$PWD python - <<'PY'
import importlib.util
from src.data.loading import get_bucket
for b in ("baseline", "all_features", "lean_vol_vix"):
    print(b, len(get_bucket(b)))
spec = importlib.util.spec_from_file_location("chk", "cluster/pool_pretune_check.py")
chk = importlib.util.module_from_spec(spec); spec.loader.exec_module(chk)
for bk in ("all_features", "baseline", "lean_vol_vix"):
    for e in ("ridge", "reclasso", "reclasticnet"):
        p = chk.stored(bk, e)
        print(f"stored {bk:12s} {e:12s} {'ok' if p.is_file() else 'MISSING'} {p}")
PY
EOF
echo "ship OK -> hoffman2:$REMOTE"
