#!/bin/bash
# I9 (2026-09-29 evening): ship the MASKED 16:00 re-run (WINDOW_MASK=1: tree rungs T10 / T1 /
# RS10 / RS1 and the LSTM; plan writeup/CAMPAIGN_16H_2026-09-29.md, "Restart with a per-window
# mask") to a CARC deployment root of its own, /scratch1/jc_905/harxhar-mask, so no shared file
# of another agent's running fleet (harxhar-subsection, harxhar-dedup) is re-extracted.
# Twin of cluster/slurm/ship_lstm_dedup_carc.sh + ship_trees_cadence_carc.sh:
#   * code from the COMMIT (git archive HEAD: src/, the six specs, the gate / reducer / count
#     scripts, the I9 Slurm scripts and task files), so committed == shipped whatever the
#     shared worktree holds; REFUSES when a shipped path differs from HEAD in the worktree;
#   * gates_ref/: src/ and the six specs as of the commit BEFORE the mask edit (PRE_SHA,
#     default f9a19b6), which the gate h1 (WINDOW_MASK=0 = the pre-edit spec) runs;
#   * data: the bar-keyed data/*.parquet the loader reads (never the option chain / spot),
#     copied on the cluster from harxhar-subsection where its md5 equals the local file,
#     uploaded otherwise;
#   * pylib_trees/ (shap for the random forest's TreeSHAP): copied on the cluster from
#     harxhar-subsection/pylib_trees (installed there by cluster/slurm/ship_trees_carc.sh).
# FAILS unless every shipped code file's and every data file's remote md5 equals the local
# one.  Then, on the login node in the harxhar env: the tree stack + shap + torch import, the
# scripts parse.
# Run from Git Bash locally:
#   bash cluster/slurm/ship_trees_mask_carc.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
REMOTE=/scratch1/jc_905/harxhar-mask
OLDROOT=/scratch1/jc_905/harxhar-subsection
PRE_SHA=${PRE_SHA:-f9a19b6}

SPECS=(
  specs/causal_tune_trees.py
  specs/causal_tune_trees_tuned.py
  specs/causal_tune_trees_tuned_jobs.py
  specs/causal_tune_linear.py
  specs/causal_tune_lstm.py
  specs/causal_tune_lstm_jobs.py
)
mapfile -t SRC < <(git ls-tree -r --name-only HEAD src | grep -v '/__pycache__/')
COMMITTED=(
  "${SRC[@]}"
  "${SPECS[@]}"
  experiments/trees_mask_gates.py
  experiments/trees_cadence_gates.py
  experiments/trees_cadence_reduce_chunks.py
  experiments/reduce_trees_tuned_chunks.py
  experiments/reduce_lstm_chunks.py
  experiments/gate_lstm.py
  experiments/trees_mask_kept_counts.py
  cluster/slurm/trees_mask_gates.sbatch
  cluster/slurm/trees_mask_pack.sbatch
  cluster/slurm/treestuned_mask_pack.sbatch
  cluster/slurm/lstm_mask_pack.sbatch
  cluster/slurm/trees_mask_merge.sbatch
  cluster/slurm/submit_trees_mask.sh
  cluster/trees_mask_tasks_untuned.txt
  cluster/trees_mask_tasks_rs10.txt
  cluster/trees_mask_tasks_rs1.txt
  cluster/trees_mask_tasks_rs_canary.txt
  cluster/lstm_mask_tasks_canary.txt
  cluster/lstm_mask_tasks_fleet.txt
)
for f in "${COMMITTED[@]}"; do
  git cat-file -e "HEAD:$f" 2>/dev/null || { echo "SHIP REFUSED: $f is not in HEAD (commit it first)"; exit 1; }
done
if ! git diff --quiet HEAD -- "${COMMITTED[@]}"; then
  echo "SHIP REFUSED: a shipped file differs from HEAD in the worktree:"
  git diff --stat HEAD -- "${COMMITTED[@]}"
  exit 1
fi
mapfile -t DATA < <(find data -maxdepth 1 -name '*.parquet' -not -name 'spxw_chain.parquet' -not -name 'spxw_spot.parquet' | sort)
sums() { awk '{p = $2; sub(/^\*/, "", p); print p, $1}' | sort; }
quiet() { grep -v "reloaded with a version change\|python/3\|^$" || true; }

"$SSH" -o BatchMode=yes usc-discovery "mkdir -p $REMOTE/data $REMOTE/logs $REMOTE/results $REMOTE/gates_ref" 2>&1 | quiet

# 1. data: copy on the cluster where the older root's md5 matches, upload the rest
LOCAL_DATA=$(md5sum "${DATA[@]}" | sums)
OLD_DATA=$("$SSH" -o BatchMode=yes usc-discovery "cd $OLDROOT && md5sum data/*.parquet 2>/dev/null" | sums || true)
UPLOAD=()
CP=""
for f in "${DATA[@]}"; do
  l=$(printf '%s\n' "$LOCAL_DATA" | awk -v f="$f" '$1 == f {print $2}')
  o=$(printf '%s\n' "$OLD_DATA" | awk -v f="$f" '$1 == f {print $2}')
  if [ -n "$o" ] && [ "$o" = "$l" ]; then CP="$CP [ -f $REMOTE/$f ] || cp -p $OLDROOT/$f $REMOTE/$f;"; else UPLOAD+=("$f"); fi
done
[ -z "$CP" ] || "$SSH" -o BatchMode=yes usc-discovery "$CP true"
if [ "${#UPLOAD[@]}" -gt 0 ]; then
  echo "uploading ${#UPLOAD[@]} data files: ${UPLOAD[*]}"
  tar czf - "${UPLOAD[@]}" | "$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && tar xzf -"
fi
# shap for the forest's TreeSHAP (a copy: nothing here reads another root at run time)
"$SSH" -o BatchMode=yes usc-discovery "[ -d $REMOTE/pylib_trees/shap ] || cp -rp $OLDROOT/pylib_trees $REMOTE/pylib_trees" 2>&1 | quiet

# 2. the committed code (git archive HEAD) and the pre-edit reference (git archive PRE_SHA)
git -c core.autocrlf=false -c core.eol=lf archive --format=tar HEAD "${COMMITTED[@]}" \
  | "$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && tar xf -"
mapfile -t PRE_SRC < <(git ls-tree -r --name-only "$PRE_SHA" src | grep -v '/__pycache__/')
git -c core.autocrlf=false -c core.eol=lf archive --format=tar "$PRE_SHA" "${PRE_SRC[@]}" "${SPECS[@]}" \
  | "$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE/gates_ref && tar xf -"

# 3. md5: committed files against HEAD's blobs, the reference against PRE_SHA's, data against local
LOCAL_ALL=$( { for f in "${COMMITTED[@]}"; do printf '%s %s\n' "$f" "$(git show "HEAD:$f" | md5sum | cut -d' ' -f1)"; done
               for f in "${PRE_SRC[@]}" "${SPECS[@]}"; do printf 'gates_ref/%s %s\n' "$f" "$(git show "$PRE_SHA:$f" | md5sum | cut -d' ' -f1)"; done
               md5sum "${DATA[@]}" | sums; } | sort)
REMOTE_ALL=$( { printf '%s\n' "${COMMITTED[@]}" "${DATA[@]}"; for f in "${PRE_SRC[@]}" "${SPECS[@]}"; do echo "gates_ref/$f"; done; } \
  | "$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && xargs md5sum" | sums)
BAD=$(diff <(printf '%s\n' "$LOCAL_ALL") <(printf '%s\n' "$REMOTE_ALL") || true)
if [ -n "$BAD" ]; then
  echo "SHIP FAILED: remote md5 differs:"; printf '%s\n' "$BAD"; exit 1
fi
echo "md5 OK: $(printf '%s\n' "$LOCAL_ALL" | wc -l) files (${#COMMITTED[@]} from HEAD $(git rev-parse --short HEAD), $(( ${#PRE_SRC[@]} + ${#SPECS[@]} )) gates_ref from $PRE_SHA, ${#DATA[@]} data)"
EXTRA=$("$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && ls data/*.parquet" | grep -v -x -F -f <(printf '%s\n' "${DATA[@]}") || true)
[ -z "$EXTRA" ] || { echo "FAILED: remote data/ holds files the local set does not: $EXTRA"; exit 1; }
HEAD_REFIT=$(git show HEAD:specs/causal_tune_trees.py | grep -m1 -E '^REFIT_EVERY = [0-9]+' || true)
echo "the LSTM reads REFIT_EVERY from the trees spec's first literal: '$HEAD_REFIT'"

# 4. login node: libraries, parse checks
"$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && module load conda >/dev/null 2>&1 && source activate harxhar && set -e
PYTHONPATH=\$PWD:\$PWD/pylib_trees python -c \"
import sys, shap, lightgbm, xgboost, sklearn, torch, pyarrow
print('python', sys.version.split()[0], 'shap', shap.__version__, 'lightgbm', lightgbm.__version__, 'xgboost', xgboost.__version__, 'sklearn', sklearn.__version__, 'torch', torch.__version__)
\"
for p in specs/*.py experiments/*.py; do
  python -c \"import ast, sys; ast.parse(open(sys.argv[1], encoding='utf-8').read())\" \$p
done
for s in cluster/slurm/*mask*.sbatch cluster/slurm/submit_trees_mask.sh; do
  bash -n \$s || { echo \"bash -n FAILED: \$s\"; exit 1; }
done
echo 'remote checks ok'" 2>&1 | quiet
echo "ship OK -> usc-discovery:$REMOTE"
