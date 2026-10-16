#!/bin/bash
# moments_vol_vix tree workstream (2026-10-06): ship the masked 16:00 tree rungs on bucket moments_vol_vix to a
# CARC deployment root of its own, /scratch1/jc_905/harxhar-momvolvix-trees, so no file of another agent's
# root (harxhar-mask, harxhar-subsection, harxhar-dedup, harxhar-lstmi) is re-extracted.
# Twin of cluster/slurm/ship_trees_mask_carc.sh:
#   * the campaign's code from the COMMIT (git archive HEAD, LF blobs, as the masked run shipped it):
#     src/ (but src/data/loading.py), the tree specs and the linear spec they read, the unedited
#     reducers / kept-count script and pack scripts; REFUSES when one of them differs from HEAD in
#     the worktree;
#   * from the WORKTREE (this workstream may not commit): src/data/loading.py, whose only change
#     against the gated moments_vix ship must be the moments_vol_vix block alone (checked below), and this
#     workstream's own new files (gate, task generator, task files, gate / merge / submit scripts);
#   * data: the bar-keyed data/*.parquet (never the option chain / spot), copied on the cluster from
#     harxhar-mask where its md5 equals the local file, uploaded otherwise;
#   * pylib_trees/ (shap for the random forest's TreeSHAP): copied from harxhar-mask.
# FAILS unless every shipped file's remote md5 equals the local source (HEAD blob or worktree file).
# Prints which shipped files differ from the masked run's deployment (harxhar-mask) -- the gate
# (experiments/trees_momvix_gate.py) then proves the shipped code reproduces its outputs.
# Run from Git Bash locally:   bash cluster/slurm/ship_trees_momvolvix_carc.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
REMOTE=/scratch1/jc_905/harxhar-momvolvix-trees
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
  experiments/trees_cadence_make_tasks.py
  cluster/slurm/trees_mask_pack.sbatch
  cluster/slurm/treestuned_mask_pack.sbatch
)
FROM_WORKTREE=(
  src/data/loading.py
  experiments/trees_momvix_gate.py
  experiments/trees_momvolvix_make_tasks.py
  cluster/slurm/trees_momvix_gates.sbatch
  cluster/slurm/trees_momvolvix_merge.sbatch
  cluster/slurm/submit_trees_momvolvix.sh
  cluster/slurm/ship_trees_momvolvix_carc.sh
  cluster/trees_momvolvix_tasks_untuned.txt
  cluster/trees_momvolvix_tasks_rs10.txt
  cluster/trees_momvolvix_tasks_rs1.txt
  cluster/trees_momvolvix_tasks_rs_canary.txt
  cluster/trees_momvolvix_tasks_none.txt
  cluster/trees_momvix_gate_lines.txt
)
for f in "${FROM_HEAD[@]}"; do
  git cat-file -e "HEAD:$f" 2>/dev/null || { echo "SHIP REFUSED: $f is not in HEAD"; exit 1; }
done
if ! git diff --quiet HEAD -- "${FROM_HEAD[@]}"; then
  echo "SHIP REFUSED: a file shipped from HEAD differs from HEAD in the worktree:"
  git diff --stat HEAD -- "${FROM_HEAD[@]}"
  exit 1
fi
# the gate of the moments_vix ship (results/trees_momvix_gates/carc_epyc-7513, 21 of 21 stored cells
# identical) stands only if this ship differs from that one by the new SUBGROUPS blocks alone:
#   (a) loading.py without the two new blocks (moments_vol_vix and lean_vol_vix: 3 comment lines + the
#       SUBGROUPS line each, right after the moments_vix block) has the md5 the moments_vix ship recorded
#       for it (logs/ship_trees_momvix_carc_manifest.txt);
#   (b) the removed lines are exactly those 8 added lines and their SUBGROUPS lines are the two buckets;
#   (c) every other file shipped from HEAD has the md5 the moments_vix ship recorded.
GATED=logs/ship_trees_momvix_carc_manifest.txt
[ -f "$GATED" ] || { echo "SHIP REFUSED: $GATED (the gated ship's manifest) missing"; exit 1; }
G_LOAD=$(awk '$1 == "src/data/loading.py" {print $2}' "$GATED")
BLOCK_START=$(grep -n '^# 2026-10-06, user request: moments_vix plus ES volume' src/data/loading.py | cut -d: -f1)
[ -n "$BLOCK_START" ] || { echo "SHIP REFUSED: moments_vol_vix block not found in src/data/loading.py"; exit 1; }
BLOCK=$(sed -n "${BLOCK_START},$((BLOCK_START + 7))p" src/data/loading.py)
[ "$(printf '%s\n' "$BLOCK" | sed -n 4p)" = 'SUBGROUPS["moments_vol_vix"] = [*SUBGROUPS["moments"], "sumvolume", "vix"]' ] \
  || { echo "SHIP REFUSED: line 4 of the block is not the moments_vol_vix SUBGROUPS line"; exit 1; }
[ "$(printf '%s\n' "$BLOCK" | sed -n 8p)" = 'SUBGROUPS["lean_vol_vix"] = ["sumret", "sumabsret", "sumret4", "sumautocov", "sumvolume", "vix"]' ] \
  || { echo "SHIP REFUSED: line 8 of the block is not the lean_vol_vix SUBGROUPS line"; exit 1; }
[ "$(printf '%s\n' "$BLOCK" | sed -n '1,3p;5,7p' | grep -c '^#')" = 6 ] || { echo "SHIP REFUSED: the blocks' other lines are not comments"; exit 1; }
WITHOUT=$(sed "${BLOCK_START},$((BLOCK_START + 7))d" src/data/loading.py | md5sum | cut -d' ' -f1)
[ "$WITHOUT" = "$G_LOAD" ] || { echo "SHIP REFUSED: loading.py minus the blocks ($WITHOUT) != the gated ship's ($G_LOAD)"; exit 1; }
NUM=$(git diff --numstat HEAD -- src/data/loading.py)
[ "$NUM" = "$(printf '13\t0\tsrc/data/loading.py')" ] || { echo "SHIP REFUSED: src/data/loading.py vs HEAD is '$NUM', expected 13 added lines only"; exit 1; }
N_HEAD_DIFF=0
for f in "${FROM_HEAD[@]}"; do
  g=$(awk -v f="$f" '$1 == f {print $2}' "$GATED")
  [ -n "$g" ] && [ "$g" = "$(git show "HEAD:$f" | md5sum | cut -d' ' -f1)" ] || { echo "  differs from the gated ship: $f"; N_HEAD_DIFF=$((N_HEAD_DIFF + 1)); }
done
[ "$N_HEAD_DIFF" = 0 ] || { echo "SHIP REFUSED: $N_HEAD_DIFF HEAD files differ from the gated ship"; exit 1; }
echo "vs the gated moments_vix ship: ${#FROM_HEAD[@]} HEAD files identical; loading.py = the gated file + the moments_vol_vix and lean_vol_vix blocks (8 lines; md5 $G_LOAD without them)"
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
# the moments_vix ship's gate (its flags and CSVs; the canary refuses without GATE_OK)
"$SSH" -o BatchMode=yes usc-discovery "mkdir -p $REMOTE/results/trees_momvix_gates/carc_epyc-7513 && cd /scratch1/jc_905/harxhar-momvix-trees/results/trees_momvix_gates/carc_epyc-7513 && cp -p GATE_OK PASS_* NODE gate_lines.csv gate_rows.csv $REMOTE/results/trees_momvix_gates/carc_epyc-7513/" 2>&1 | quiet

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
printf '%s\n' "$LOCAL_ALL" > "logs/ship_trees_momvolvix_carc_manifest.txt"
EXTRA=$("$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && ls data/*.parquet" | grep -v -x -F -f <(printf '%s\n' "${DATA[@]}") || true)
[ -z "$EXTRA" ] || { echo "FAILED: remote data/ holds files the local set does not: $EXTRA"; exit 1; }
# against the masked run's deployment (informational; the gate decides)
MASK_SUMS=$(printf '%s\n' "${FROM_HEAD[@]}" src/data/loading.py | "$SSH" -o BatchMode=yes usc-discovery "cd $MASK && xargs md5sum 2>/dev/null" | sums || true)
echo "files differing from $MASK (or absent there):"
join -a1 <(printf '%s\n' "$REMOTE_ALL") <(printf '%s\n' "$MASK_SUMS") | awk 'NF < 3 || $2 != $3 {print "  " $1}' | grep -v -e '^  data/' -e 'momvolvix' || true

# 4. login node: libraries, the bucket, parse checks
"$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && module load conda >/dev/null 2>&1 && source activate harxhar && set -e
PYTHONPATH=\$PWD:\$PWD/pylib_trees python -c \"
import sys, shap, lightgbm, xgboost, sklearn
from src.data.loading import get_bucket
print('python', sys.version.split()[0], 'shap', shap.__version__, 'lightgbm', lightgbm.__version__, 'xgboost', xgboost.__version__, 'sklearn', sklearn.__version__)
print('moments_vol_vix:', get_bucket('moments_vol_vix'))
print('lean_vol_vix:', get_bucket('lean_vol_vix'))
\"
for p in specs/*.py experiments/*.py; do
  python -c \"import ast, sys; ast.parse(open(sys.argv[1], encoding='utf-8').read())\" \$p
done
for s in cluster/slurm/*.sbatch cluster/slurm/submit_trees_momvolvix.sh; do
  bash -n \$s || { echo \"bash -n FAILED: \$s\"; exit 1; }
done
echo 'remote checks ok'" 2>&1 | quiet
echo "ship OK -> usc-discovery:$REMOTE"
