#!/bin/bash
# Ship the pooled moments_vix campaign to a deployment root of its own on CARC (2026-10-06) and FAIL
# unless every shipped file's remote md5 equals the local one.  Twin of cluster/pool_momvix_ship_h2.sh.
#
# Code: every tracked src/*.py and specs/causal_tune_linear.py as git stores them (LF; a file with no
# change against HEAD is taken from HEAD; the only working-tree change under src/ is the moments_vix
# bucket in src/data/loading.py); the campaign's CARC files.
# Data: the seven bar-keyed parquets the stored pooled live_feasible arms of the CARC root read
# (their run.log: 59 columns loaded), copied on the cluster from that root after an md5 match.
# Then on the login node: the scripts parse, the bucket resolves, the stored arms exist.
#
#   STAGE=<empty scratch dir> bash cluster/slurm/ship_pool_momvix_carc.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
REMOTE=/scratch1/jc_905/harxhar-pool-momvix
OLDROOT=/scratch1/jc_905/harxhar-subsection
STAGE=${STAGE:?set STAGE to an empty local scratch directory}
mkdir -p "$STAGE"
[ -z "$(ls -A "$STAGE")" ] || { echo "STAGE $STAGE is not empty"; exit 1; }

mapfile -t SRC < <(git ls-files src | grep '\.py$')
CODE=("${SRC[@]}" specs/causal_tune_linear.py)
OURS=(
  cluster/slurm/pool_momvix_run1.sh
  cluster/slurm/pool_momvix_carc.sbatch
  cluster/slurm/pool_momvix_check_carc.py
  cluster/slurm/submit_pool_momvix.sh
  cluster/pool_momvix_tasks_canary_carc.txt
  cluster/pool_momvix_tasks_fleet_carc.txt
  cluster/pool_momvix_rth_ref_md5_carc.txt
)
DATA=(core_stats ewstock_stats releases spy_and_sentiment time_categories vix_and_voldemand vwstock_stats)

for f in "${CODE[@]}"; do
  mkdir -p "$STAGE/$(dirname "$f")"
  if git diff --quiet HEAD -- "$f"; then git show "HEAD:$f" > "$STAGE/$f"; else cp "$f" "$STAGE/$f"; fi
done
for f in "${OURS[@]}"; do mkdir -p "$STAGE/$(dirname "$f")"; cp "$f" "$STAGE/$f"; done
CR=$(cd "$STAGE" && cat "${CODE[@]}" "${OURS[@]}" | tr -cd '\r' | wc -c)
[ "$CR" -eq 0 ] || { echo "staged files carry $CR CR bytes"; exit 1; }
grep -q 'SUBGROUPS\["moments_vix"\]' "$STAGE/src/data/loading.py" || { echo "no moments_vix bucket"; exit 1; }

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

"$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && bash -s" <<'EOF'
set -e
for s in cluster/slurm/pool_momvix_run1.sh cluster/slurm/pool_momvix_carc.sbatch cluster/slurm/submit_pool_momvix.sh; do
  bash -n "$s" && echo "bash -n ok: $s"
done
module load conda >/dev/null 2>&1; source activate harxhar
PYTHONPATH=$PWD python - <<'PY'
import importlib.util
from src.data.loading import SUBGROUPS, get_bucket
b = get_bucket("moments_vix")
assert b == [*SUBGROUPS["moments"], "vix"] and len(b) == 8, b
print("moments_vix:", b)
spec = importlib.util.spec_from_file_location("chk", "cluster/slurm/pool_momvix_check_carc.py")
chk = importlib.util.module_from_spec(spec); spec.loader.exec_module(chk)
for e in chk.ESTS:
    p = chk.stored("live_feasible", e)
    print(f"stored live_feasible {e:12s} {'ok' if p.is_file() else 'MISSING'} {p}")
PY
EOF
echo "ship OK -> usc-discovery:$REMOTE"
