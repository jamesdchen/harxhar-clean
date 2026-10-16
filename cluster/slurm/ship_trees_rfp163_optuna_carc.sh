#!/bin/bash
# rfp163 (2026-10-07): ship the Optuna rung of the random forest with a dev-span shipped forest
# (cluster/slurm/submit_trees_rfp163_optuna.sh) to a CARC deployment root of its own,
# /scratch1/jc_905/harxhar-<TOKEN>-optuna -- cluster/slurm/ship_trees_rfprev_carc.sh with this run's files.
# PARAMETERIZED by TOKEN (default rfp163, the worked instance: the search's trial 163) and RF_CONFIG (default, for
# rfp163, cluster/rfprev163_rf_config.json; required otherwise): the generated spec specs/causal_tune_trees_<TOKEN>.py
# and the task files cluster/<TOKEN>_optuna_tasks_*.txt must exist (experiments/trees_rfp163_make_specs.py,
# cluster/rfp163_optuna_make_tasks.py with --config / --token).
#   HEAD blobs (git archive)   src/ (except src/data/loading.py), the Optuna spec and its jobs, the tuned jobs, the
#                              untuned tree spec, specs/causal_tune_linear.py (VAL_TAIL / EMBARGO), the reducer --
#                              each must equal HEAD in the worktree
#   worktree files             src/data/loading.py (its diff vs HEAD must be added comment / SUBGROUPS lines only:
#                              the lean_vol_vix bucket), the untuned dev-forest wrapper (the gate's fixed-settings
#                              comparator and rf_params), the generated spec, the Optuna wrapper, the generator, the
#                              gate, the pack / merge / gate / submit / ship scripts, the task files, both
#                              configuration files -- no CRLF
#   data                       copied on the cluster from /scratch1/jc_905/harxhar-mask where the md5 equals the
#                              local file, the rest uploaded (never spxw_chain / spxw_spot); pylib_trees from there
# then an md5 check of every shipped file against HEAD's blobs / the local files, and login-node checks (libraries,
# the bucket, the generated spec = the generator's output, the wrapper's patches resolve, parse checks).
# Run from Git Bash locally:   bash cluster/slurm/ship_trees_rfp163_optuna_carc.sh
#                              TOKEN=rf400 RF_CONFIG=cluster/<config>.json bash cluster/slurm/ship_trees_rfp163_optuna_carc.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
TOKEN=${TOKEN:-rfp163}
if [ "$TOKEN" = rfp163 ]; then RF_CONFIG=${RF_CONFIG:-cluster/rfprev163_rf_config.json}; else RF_CONFIG=${RF_CONFIG:?RF_CONFIG must name the configuration of token $TOKEN}; fi
REMOTE=/scratch1/jc_905/harxhar-$TOKEN-optuna
MASK=/scratch1/jc_905/harxhar-mask

mapfile -t SRC < <(git ls-tree -r --name-only HEAD src | grep -v '/__pycache__/' | grep -v -x 'src/data/loading.py')
FROM_HEAD=(
  "${SRC[@]}"
  specs/causal_tune_trees.py
  specs/causal_tune_trees_optuna.py
  specs/causal_tune_trees_optuna_jobs.py
  specs/causal_tune_trees_tuned_jobs.py
  specs/causal_tune_linear.py
  experiments/reduce_trees_optuna.py
)
FROM_WORKTREE=(
  src/data/loading.py
  specs/causal_tune_trees_rfdev.py
  "specs/causal_tune_trees_$TOKEN.py"
  specs/causal_tune_trees_optuna_rfp.py
  experiments/trees_rfp163_make_specs.py
  experiments/trees_rfp163_optuna_gate.py
  cluster/slurm/optuna_rfp_pack.sbatch
  cluster/slurm/optuna_rfp_merge.sbatch
  cluster/slurm/optuna_rfp_gate.sbatch
  cluster/slurm/submit_trees_rfp163_optuna.sh
  cluster/slurm/ship_trees_rfp163_optuna_carc.sh
  cluster/rfp163_optuna_make_tasks.py
  "cluster/${TOKEN}_optuna_tasks_gate.txt"
  "cluster/${TOKEN}_optuna_tasks_canary.txt"
  "cluster/${TOKEN}_optuna_tasks_s1.txt"
  "cluster/${TOKEN}_optuna_tasks_s2.txt"
  cluster/rfprev163_rf_config.json
  cluster/rfdev_rf_config.json
)
[[ " ${FROM_WORKTREE[*]} " == *" $RF_CONFIG "* ]] || FROM_WORKTREE+=("$RF_CONFIG")
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
  [ "$f" = src/data/loading.py ] || ! grep -q $'\r' "$f" || { echo "SHIP REFUSED: $f has CRLF line endings"; exit 1; }
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

# 2. code: HEAD blobs, then the worktree files (src/data/loading.py with LF endings, as HEAD stores it)
git -c core.autocrlf=false -c core.eol=lf archive --format=tar HEAD "${FROM_HEAD[@]}" \
  | "$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && tar xf -"
STAGE_DIR=$(mktemp -d)
trap 'rm -rf "$STAGE_DIR"' EXIT
tar cf - "${FROM_WORKTREE[@]}" | tar xf - -C "$STAGE_DIR"
tr -d '\r' < src/data/loading.py > "$STAGE_DIR/src/data/loading.py"
tar cf - -C "$STAGE_DIR" "${FROM_WORKTREE[@]}" | "$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && tar xf -"

# 3. md5: HEAD files against HEAD's blobs, worktree files (as staged) and data against the local files
LOCAL_ALL=$( { for f in "${FROM_HEAD[@]}"; do printf '%s %s\n' "$f" "$(git show "HEAD:$f" | md5sum | cut -d' ' -f1)"; done
               (cd "$STAGE_DIR" && md5sum "${FROM_WORKTREE[@]}") | sums
               md5sum "${DATA[@]}" | sums; } | sort)
REMOTE_ALL=$(printf '%s\n' "${FROM_HEAD[@]}" "${FROM_WORKTREE[@]}" "${DATA[@]}" \
  | "$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && xargs md5sum" | sums)
BAD=$(diff <(printf '%s\n' "$LOCAL_ALL") <(printf '%s\n' "$REMOTE_ALL") || true)
if [ -n "$BAD" ]; then
  echo "SHIP FAILED: remote md5 differs:"; printf '%s\n' "$BAD"; exit 1
fi
echo "md5 OK: $(printf '%s\n' "$LOCAL_ALL" | wc -l) files (${#FROM_HEAD[@]} from HEAD $(git rev-parse --short HEAD), ${#FROM_WORKTREE[@]} from the worktree, ${#DATA[@]} data)"
printf '%s\n' "$LOCAL_ALL" > "logs/ship_trees_${TOKEN}_optuna_carc_manifest.txt"
EXTRA=$("$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && ls data/*.parquet" | grep -v -x -F -f <(printf '%s\n' "${DATA[@]}") || true)
[ -z "$EXTRA" ] || { echo "FAILED: remote data/ holds files the local set does not: $EXTRA"; exit 1; }
# against the masked run's deployment (informational; the gate decides)
MASK_SUMS=$(printf '%s\n' "${FROM_HEAD[@]}" src/data/loading.py | "$SSH" -o BatchMode=yes usc-discovery "cd $MASK && xargs md5sum 2>/dev/null" | sums || true)
echo "files differing from $MASK (or absent there):"
join -a1 <(printf '%s\n' "$REMOTE_ALL") <(printf '%s\n' "$MASK_SUMS") | awk 'NF < 3 || $2 != $3 {print "  " $1}' | grep -v -e '^  data/' -e 'rfp163' -e "$TOKEN" -e 'optuna_rfp' -e 'rfdev' -e 'rfprev' || true

# 4. login node: libraries, the bucket, the generated spec, the wrapper's patches, parse checks
"$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && module load conda >/dev/null 2>&1 && source activate harxhar && set -e
PYTHONPATH=\$PWD:\$PWD/pylib_trees python -c \"
import sys, optuna, shap, sklearn
from src.data.loading import get_bucket
print('python', sys.version.split()[0], 'optuna', optuna.__version__, 'shap', shap.__version__, 'sklearn', sklearn.__version__)
print('lean_vol_vix:', get_bucket('lean_vol_vix'))
\"
python experiments/trees_rfp163_make_specs.py --config $RF_CONFIG --token $TOKEN --check
HPC_KW_MODEL=rf HPC_KW_TREE_SPEC=specs/causal_tune_trees_$TOKEN.py PYTHONPATH=\$PWD python -c \"
import sys
sys.path[:0] = ['specs']
import causal_tune_trees_optuna_rfp as w, causal_tune_trees_optuna_jobs as OJ, causal_tune_trees_tuned_jobs as J
assert J.TREE_SPEC.name == 'causal_tune_trees_$TOKEN.py', J.TREE_SPEC
print('wrapper patches: TREE_SPEC', J.TREE_SPEC.name, '| space', OJ.SPACES['rf'], '| trial 0', OJ.shipped_config('rf'))
\" | grep -v '^trees:'
for p in specs/*.py experiments/*.py cluster/*.py; do
  python -c \"import ast, sys; ast.parse(open(sys.argv[1], encoding='utf-8').read())\" \$p
done
for s in cluster/slurm/optuna_rfp_*.sbatch cluster/slurm/submit_trees_rfp163_optuna.sh; do
  bash -n \$s || { echo \"bash -n FAILED: \$s\"; exit 1; }
done
echo 'remote checks ok'" 2>&1 | quiet
echo "ship OK ($TOKEN, $RF_CONFIG) -> usc-discovery:$REMOTE"
