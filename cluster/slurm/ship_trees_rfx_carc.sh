#!/bin/bash
# rfx (2026-10-07): ship the dev-span forest TOKEN (e.g. rf400) -- fixed-settings masked / unmasked rungs and the re-centred
# random-search rungs (cluster/slurm/submit_trees_rfx.sh) -- to /scratch1/jc_905/harxhar-$TOKEN-trees.  TOKEN-driven copy of
# cluster/slurm/ship_trees_rfp163_carc.sh: HEAD blobs for the campaign's unchanged code, the worktree for
# src/data/loading.py (added comment / SUBGROUPS lines only), the wrappers, the generated spec
# specs/causal_tune_trees_$TOKEN.py (experiments/trees_rfp163_make_specs.py --config cluster/${TOKEN}_rf_config.json --token $TOKEN),
# the gate / pack / merge / submit scripts, TOKEN's task files and configuration.
# Run from Git Bash locally:   TOKEN=rf400 bash cluster/slurm/ship_trees_rfx_carc.sh
set -euo pipefail
TOKEN=${TOKEN:?TOKEN}
cd "$(dirname "$0")/../.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
REMOTE=/scratch1/jc_905/harxhar-$TOKEN-trees
MASK=/scratch1/jc_905/harxhar-mask

mapfile -t SRC < <(git ls-tree -r --name-only HEAD src | grep -v '/__pycache__/' | grep -v -x 'src/data/loading.py')
FROM_HEAD=(
  "${SRC[@]}"
  specs/causal_tune_trees.py
  specs/causal_tune_trees_tuned.py
  specs/causal_tune_trees_tuned_jobs.py
  specs/causal_tune_linear.py
  experiments/trees_cadence_reduce_chunks.py
  experiments/reduce_trees_tuned_chunks.py
  experiments/trees_mask_kept_counts.py
)
FROM_WORKTREE=(
  src/data/loading.py
  specs/causal_tune_trees_rfdev.py
  "specs/causal_tune_trees_$TOKEN.py"
  specs/causal_tune_trees_tuned_rfp.py
  experiments/trees_rfp163_make_specs.py
  experiments/trees_rfp163_tuned_gate.py
  cluster/slurm/trees_rfdev_pack.sbatch
  cluster/slurm/treestuned_rfp_pack.sbatch
  cluster/slurm/trees_rfdev_nomask_pack.sbatch
  cluster/slurm/trees_rfp163_gates.sbatch
  cluster/slurm/trees_rfx_merge.sbatch
  cluster/slurm/submit_trees_rfx.sh
  cluster/slurm/ship_trees_rfx_carc.sh
  "cluster/trees_${TOKEN}_tasks_untuned.txt"
  "cluster/trees_${TOKEN}_nomask_tasks_untuned.txt"
  "cluster/trees_${TOKEN}_tasks_rs10.txt"
  "cluster/trees_${TOKEN}_tasks_rs1.txt"
  "cluster/trees_${TOKEN}_tasks_rs_canary.txt"
  "cluster/trees_${TOKEN}_tuned_gate_lines.txt"
  "cluster/${TOKEN}_rf_config.json"
)
for f in "${FROM_HEAD[@]}"; do
  git cat-file -e "HEAD:$f" 2>/dev/null || { echo "SHIP REFUSED: $f is not in HEAD"; exit 1; }
done
if ! git diff --quiet HEAD -- "${FROM_HEAD[@]}"; then
  echo "SHIP REFUSED: a file shipped from HEAD differs from HEAD in the worktree:"
  git diff --stat HEAD -- "${FROM_HEAD[@]}"
  exit 1
fi
NUM=$(git diff --numstat HEAD -- src/data/loading.py | cut -f2)
[ "$NUM" = 0 ] || { echo "SHIP REFUSED: src/data/loading.py removes $NUM lines of HEAD"; exit 1; }
BADADD=$(git diff -U0 HEAD -- src/data/loading.py | grep '^+[^+]' | grep -v -E '^\+#|^\+SUBGROUPS\["[a-z_]+"\] = ' || true)
[ -z "$BADADD" ] || { echo "SHIP REFUSED: added loading.py lines other than comments / SUBGROUPS lines:"; echo "$BADADD"; exit 1; }
for f in "${FROM_WORKTREE[@]}"; do
  [ -f "$f" ] || { echo "SHIP REFUSED: $f missing"; exit 1; }
  ! grep -q $'\r' "$f" || { echo "SHIP REFUSED: $f has CRLF line endings"; exit 1; }
done
mapfile -t DATA < <(find data -maxdepth 1 -name '*.parquet' -not -name 'spxw_chain.parquet' -not -name 'spxw_spot.parquet' | sort)
sums() { awk '{p = $2; sub(/^\*/, "", p); print p, $1}' | sort; }
quiet() { grep -v "reloaded with a version change\|python/3\|^$" || true; }

"$SSH" -o BatchMode=yes usc-discovery "mkdir -p $REMOTE/data $REMOTE/logs $REMOTE/results" 2>&1 | quiet

# 1. data: copy on the cluster where harxhar-mask's md5 matches, upload the rest
LOCAL_DATA=$(md5sum "${DATA[@]}" | sums)
OLD_DATA=$("$SSH" -o BatchMode=yes usc-discovery "cd $MASK && md5sum data/*.parquet 2>/dev/null" | sums || true)
UPLOAD=()
CP=""
for f in "${DATA[@]}"; do
  l=$(printf '%s\n' "$LOCAL_DATA" | awk -v f="$f" '$1 == f {print $2}')
  o=$(printf '%s\n' "$OLD_DATA" | awk -v f="$f" '$1 == f {print $2}')
  if [ -n "$o" ] && [ "$o" = "$l" ]; then CP="$CP [ -f $REMOTE/$f ] || cp -p $MASK/$f $REMOTE/$f;"; else UPLOAD+=("$f"); fi
done
[ -z "$CP" ] || "$SSH" -o BatchMode=yes usc-discovery "$CP true"
if [ "${#UPLOAD[@]}" -gt 0 ]; then
  echo "uploading ${#UPLOAD[@]} data files: ${UPLOAD[*]}"
  tar czf - "${UPLOAD[@]}" | "$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && tar xzf -"
fi
"$SSH" -o BatchMode=yes usc-discovery "[ -d $REMOTE/pylib_trees/shap ] || cp -rp $MASK/pylib_trees $REMOTE/pylib_trees" 2>&1 | quiet

# 2. code: HEAD blobs, then the worktree files
git -c core.autocrlf=false -c core.eol=lf archive --format=tar HEAD "${FROM_HEAD[@]}" \
  | "$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && tar xf -"
tar cf - "${FROM_WORKTREE[@]}" | "$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && tar xf -"

# 3. md5: HEAD files against HEAD's blobs, worktree files and data against the local files
LOCAL_ALL=$( { for f in "${FROM_HEAD[@]}"; do printf '%s %s\n' "$f" "$(git show "HEAD:$f" | md5sum | cut -d' ' -f1)"; done
               md5sum "${FROM_WORKTREE[@]}" "${DATA[@]}" | sums; } | sort)
REMOTE_ALL=$(printf '%s\n' "${FROM_HEAD[@]}" "${FROM_WORKTREE[@]}" "${DATA[@]}" \
  | "$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && xargs md5sum" | sums)
BAD=$(diff <(printf '%s\n' "$LOCAL_ALL") <(printf '%s\n' "$REMOTE_ALL") || true)
if [ -n "$BAD" ]; then
  echo "SHIP FAILED: remote md5 differs:"; printf '%s\n' "$BAD"; exit 1
fi
echo "md5 OK: $(printf '%s\n' "$LOCAL_ALL" | wc -l) files (${#FROM_HEAD[@]} from HEAD $(git rev-parse --short HEAD), ${#FROM_WORKTREE[@]} from the worktree, ${#DATA[@]} data)"
printf '%s\n' "$LOCAL_ALL" > "logs/ship_trees_${TOKEN}_carc_manifest.txt"
EXTRA=$("$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && ls data/*.parquet" | grep -v -x -F -f <(printf '%s\n' "${DATA[@]}") || true)
[ -z "$EXTRA" ] || { echo "FAILED: remote data/ holds files the local set does not: $EXTRA"; exit 1; }
# against the masked run's deployment (informational; the gate decides)
MASK_SUMS=$(printf '%s\n' "${FROM_HEAD[@]}" src/data/loading.py | "$SSH" -o BatchMode=yes usc-discovery "cd $MASK && xargs md5sum 2>/dev/null" | sums || true)
echo "files differing from $MASK (or absent there):"
join -a1 <(printf '%s\n' "$REMOTE_ALL") <(printf '%s\n' "$MASK_SUMS") | awk 'NF < 3 || $2 != $3 {print "  " $1}' | grep -v -e '^  data/' -e 'rfdev' -e 'rfprev' -e 'rfp' || true

# 4. login node: libraries, the bucket, parse checks
"$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && module load conda >/dev/null 2>&1 && source activate harxhar && set -e
PYTHONPATH=\$PWD:\$PWD/pylib_trees python -c \"
import sys, shap, lightgbm, xgboost, sklearn
from src.data.loading import get_bucket
print('python', sys.version.split()[0], 'shap', shap.__version__, 'lightgbm', lightgbm.__version__, 'xgboost', xgboost.__version__, 'sklearn', sklearn.__version__)
print('lean_vol_vix:', get_bucket('lean_vol_vix'))
\"
for p in specs/*.py experiments/*.py; do
  python -c \"import ast, sys; ast.parse(open(sys.argv[1], encoding='utf-8').read())\" \$p
done
for s in cluster/slurm/*.sbatch cluster/slurm/submit_trees_rfx.sh; do
  bash -n \$s || { echo \"bash -n FAILED: \$s\"; exit 1; }
done
echo 'remote checks ok'" 2>&1 | quiet
echo "ship OK -> usc-discovery:$REMOTE"
