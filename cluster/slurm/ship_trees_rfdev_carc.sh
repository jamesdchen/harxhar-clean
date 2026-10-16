#!/bin/bash
# rfdev (user decision 2026-10-06): ship the dev-tuned random-forest run (T1 / T10 on baseline, all_features,
# lean_vol_vix; cluster/slurm/submit_trees_rfdev.sh) to a CARC deployment root of its own,
# /scratch1/jc_905/harxhar-rfdev-trees.  Twin of cluster/slurm/ship_trees_momvix_carc.sh:
#   * the campaign's code from the COMMIT (git archive HEAD, LF blobs): src/ (but src/data/loading.py), the
#     tree spec, the unedited reducers / kept-count script; REFUSES when one differs from HEAD in the worktree;
#   * from the WORKTREE: src/data/loading.py (its change against HEAD must be added lines only, each a comment
#     or a SUBGROUPS line: checked), the wrapper spec specs/causal_tune_trees_rfdev.py, this run's gate / pack /
#     merge / submit scripts and task files, the two random-forest configurations (cluster/rfdev_shipped_rf.json:
#     the shipped forest, for the gate; cluster/rfdev_rf_config.json: the dev-span configuration, when present);
#   * data copied on the cluster from harxhar-mask where the md5 equals the local file; pylib_trees likewise.
# FAILS unless every shipped file's remote md5 equals the local source.  The gate then proves the shipped code
# reproduces the stored masked random-forest chunks.
# Run from Git Bash locally (no conda env needed):   bash cluster/slurm/ship_trees_rfdev_carc.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
REMOTE=/scratch1/jc_905/harxhar-rfdev-trees
MASK=/scratch1/jc_905/harxhar-mask

mapfile -t SRC < <(git ls-tree -r --name-only HEAD src | grep -v '/__pycache__/' | grep -v -x 'src/data/loading.py')
FROM_HEAD=(
  "${SRC[@]}"
  specs/causal_tune_trees.py
  experiments/trees_cadence_reduce_chunks.py
  experiments/reduce_trees_tuned_chunks.py
  experiments/trees_mask_kept_counts.py
  cluster/slurm/trees_mask_pack.sbatch
)
FROM_WORKTREE=(
  src/data/loading.py
  specs/causal_tune_trees_rfdev.py
  experiments/trees_momvix_gate.py
  experiments/trees_rfdev_gate.py
  cluster/slurm/trees_rfdev_pack.sbatch
  cluster/slurm/trees_rfdev_gates.sbatch
  cluster/slurm/trees_rfdev_merge.sbatch
  cluster/slurm/submit_trees_rfdev.sh
  cluster/slurm/ship_trees_rfdev_carc.sh
  cluster/trees_rfdev_gate_lines.txt
  cluster/trees_rfdev_tasks_none.txt
  cluster/rfdev_shipped_rf.json
  cluster/slurm/submit_trees_rfdev_xcpu.sh
  cluster/slurm/trees_rfdev_xcpu_merge.sbatch
  experiments/trees_rfdev_xcpu_compare.py
)
for f in cluster/trees_rfdev_tasks_untuned.txt cluster/rfdev_rf_config.json; do [ -f "$f" ] && FROM_WORKTREE+=("$f"); done
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
printf '%s\n' "$LOCAL_ALL" > "logs/ship_trees_rfdev_carc_manifest.txt"
EXTRA=$("$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && ls data/*.parquet" | grep -v -x -F -f <(printf '%s\n' "${DATA[@]}") || true)
[ -z "$EXTRA" ] || { echo "FAILED: remote data/ holds files the local set does not: $EXTRA"; exit 1; }
# against the masked run's deployment (informational; the gate decides)
MASK_SUMS=$(printf '%s\n' "${FROM_HEAD[@]}" src/data/loading.py | "$SSH" -o BatchMode=yes usc-discovery "cd $MASK && xargs md5sum 2>/dev/null" | sums || true)
echo "files differing from $MASK (or absent there):"
join -a1 <(printf '%s\n' "$REMOTE_ALL") <(printf '%s\n' "$MASK_SUMS") | awk 'NF < 3 || $2 != $3 {print "  " $1}' | grep -v -e '^  data/' -e 'rfdev' || true

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
for s in cluster/slurm/*.sbatch cluster/slurm/submit_trees_rfdev.sh; do
  bash -n \$s || { echo \"bash -n FAILED: \$s\"; exit 1; }
done
echo 'remote checks ok'" 2>&1 | quiet
echo "ship OK -> usc-discovery:$REMOTE"
