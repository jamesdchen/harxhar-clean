#!/bin/bash
# Ship the pooled pre-tune study (2026-10-06) to a deployment root of its own on CARC and FAIL unless
# every shipped file's remote md5 equals the local one.  Twin of cluster/pool_pretune_ship_h2.sh.
#
# Code: every tracked src/*.py and specs/causal_tune_linear.py as git stores them (LF; a file with no
# change against HEAD is taken from HEAD), the runner experiments/pool_linear_pretune.py, the campaign's
# CARC files, the gate's reference rows and the arm-f schedules once written.
# Data: the seven bar-keyed parquets, copied on the cluster from the old root after an md5 match.
#
#   STAGE=<empty scratch dir> bash cluster/slurm/ship_pool_pretune_carc.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
REMOTE=/scratch1/jc_905/harxhar-pool-pretune
OLDROOT=/scratch1/jc_905/harxhar-subsection
STAGE=${STAGE:?set STAGE to an empty local scratch directory}
mkdir -p "$STAGE"
[ -z "$(ls -A "$STAGE")" ] || { echo "STAGE $STAGE is not empty"; exit 1; }

mapfile -t SRC < <(git ls-files src | grep '\.py$')
CODE=("${SRC[@]}" specs/causal_tune_linear.py)
OURS=(
  experiments/pool_linear_pretune.py
  cluster/slurm/pool_pretune_run1.sh
  cluster/slurm/pool_pretune_carc.sbatch
  cluster/slurm/pool_pretune_check_carc.py
  cluster/slurm/submit_pool_pretune.sh
)
mapfile -t EXTRA_FILES < <(ls cluster/pool_pretune_tasks_*carc.txt cluster/pool_pretune_ref_*.csv 2>/dev/null)
mapfile -t SCHED < <(ls cluster/pool_pretune_sched/*.json 2>/dev/null || true)
OURS+=("${EXTRA_FILES[@]}" "${SCHED[@]}")
DATA=(core_stats ewstock_stats releases spy_and_sentiment time_categories vix_and_voldemand vwstock_stats)

for f in "${CODE[@]}"; do
  mkdir -p "$STAGE/$(dirname "$f")"
  if git diff --quiet HEAD -- "$f"; then git show "HEAD:$f" > "$STAGE/$f"; else cp "$f" "$STAGE/$f"; fi
done
for f in "${OURS[@]}"; do mkdir -p "$STAGE/$(dirname "$f")"; cp "$f" "$STAGE/$f"; done
CR=$(cd "$STAGE" && cat "${CODE[@]}" "${OURS[@]}" | tr -cd '\r' | wc -c)
[ "$CR" -eq 0 ] || { echo "staged files carry $CR CR bytes"; exit 1; }
grep -q 'SUBGROUPS\["lean_vol_vix"\]' "$STAGE/src/data/loading.py" || { echo "no lean_vol_vix bucket"; exit 1; }

COPY="set -e; mkdir -p $REMOTE/data $REMOTE/logs $REMOTE/results"
for d in "${DATA[@]}"; do
  f=data/$d.parquet
  l=$(md5sum < "$f" | cut -c1-32)
  COPY="$COPY; o=\$(md5sum < $OLDROOT/$f | cut -c1-32); [ \"\$o\" = $l ] || { echo \"$f: old root md5 \$o != local $l\"; exit 1; }; [ -f $REMOTE/$f ] || cp -p $OLDROOT/$f $REMOTE/$f"
done
"$SSH" -o BatchMode=yes usc-discovery "$COPY"
(cd "$STAGE" && tar czf - "${CODE[@]}" "${OURS[@]}") | "$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && tar xzf -"

sums() { awk '{p = $2; sub(/^\*/, "", p); print p, $1}' | sort; }
LOCAL_ALL=$( { (cd "$STAGE" && md5sum "${CODE[@]}" "${OURS[@]}"); for d in "${DATA[@]}"; do md5sum "data/$d.parquet"; done; } | sums)
REMOTE_ALL=$( { printf '%s\n' "${CODE[@]}" "${OURS[@]}"; for d in "${DATA[@]}"; do echo "data/$d.parquet"; done; } \
  | "$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && xargs md5sum" | sums)
BAD=$(diff <(printf '%s\n' "$LOCAL_ALL") <(printf '%s\n' "$REMOTE_ALL") || true)
[ -z "$BAD" ] || { echo "SHIP FAILED: remote md5 differs from local:"; printf '%s\n' "$BAD"; exit 1; }
EXTRA=$("$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && ls data/*.parquet" | grep -v -x -F -f <(for d in "${DATA[@]}"; do echo "data/$d.parquet"; done) || true)
[ -z "$EXTRA" ] || { echo "FAILED: remote data/ holds other files: $EXTRA"; exit 1; }
echo "md5 OK: $(printf '%s\n' "$LOCAL_ALL" | wc -l) files (${#SRC[@]} src, 1 spec, ${#OURS[@]} campaign files, ${#DATA[@]} data)"

"$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && bash -s" <<'REMOTE_EOF'
set -e
for s in cluster/slurm/pool_pretune_run1.sh cluster/slurm/pool_pretune_carc.sbatch cluster/slurm/submit_pool_pretune.sh; do
  bash -n "$s" && echo "bash -n ok: $s"
done
module load conda >/dev/null 2>&1; source activate harxhar
PYTHONPATH=$PWD python experiments/pool_linear_pretune.py --help > /dev/null && echo "runner parses"
PYTHONPATH=$PWD python -c "from src.data.loading import get_bucket; print({b: len(get_bucket(b)) for b in ('baseline', 'all_features', 'lean_vol_vix')})"
REMOTE_EOF
echo "ship OK -> usc-discovery:$REMOTE"
