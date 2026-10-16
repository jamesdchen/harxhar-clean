#!/bin/bash
# Feature MCS trees, the 16 bucket designs: ship the XGBoost / forest part to Hoffman2 (2026-10-13; CARC /scratch1 storage
# failure) -- /u/scratch/j/jamesdc1/harxhar-mcstrees-h2: src/ and the shipped tree spec (HEAD), src/data/loading.py, the
# generated XGBoost / forest specs, the MCS wrapper, the designs / rows / H2 task lines, the task and submit scripts, and
# data/ (copied on Hoffman2 from the runtime root where the md5 equals the local file, uploaded otherwise).  md5 of every
# file against the local copy.  Copy of cluster/ship_trees_rfx_h2.sh.
# Run from Git Bash locally:   bash cluster/ship_feature_mcs_trees_h2.sh
set -euo pipefail
cd "$(dirname "$0")/.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
H2=/u/scratch/j/jamesdc1/harxhar-mcstrees-h2
RT=/u/scratch/j/jamesdc1/harxhar-optuna
H2HOST=${H2HOST:-hoffman2}  # h2dtn (the data-transfer node, same /u/scratch) when the login nodes do not answer
STAGE_DIR=${STAGE_DIR:-${TMPDIR:-/tmp}/ship_feature_mcs_trees_h2}
mkdir -p "$STAGE_DIR"
sums() { awk '{p = $2; sub(/^\*/, "", p); print p, $1}' | sort; }
mapfile -t SRC < <(git ls-tree -r --name-only HEAD src | grep -v '/__pycache__/' | grep -v -x -e 'src/data/loading.py' -e 'src/evaluation/model_confidence_set.py')
FROM_HEAD=("${SRC[@]}" specs/causal_tune_trees.py)
FROM_WORKTREE=(
  src/data/loading.py
  specs/causal_tune_trees_xgbtop.py specs/causal_tune_trees_rf400.py specs/causal_tune_trees_mcs.py
  cluster/feature_mcs_trees_designs.json cluster/feature_mcs_trees_rows.json cluster/feature_mcs_trees_h2_v3_lines.txt
  cluster/feature_mcs_trees_h2_task.sh cluster/submit_feature_mcs_trees_h2.sh cluster/ship_feature_mcs_trees_h2.sh
)
git diff --quiet HEAD -- "${FROM_HEAD[@]}" || { echo "SHIP REFUSED: a file shipped from HEAD differs from HEAD:"; git diff --stat HEAD -- "${FROM_HEAD[@]}"; exit 1; }
for f in "${FROM_WORKTREE[@]}"; do [ -f "$f" ] || { echo "SHIP REFUSED: $f missing"; exit 1; }; done
mapfile -t DATA < <(find data -maxdepth 1 -name '*.parquet' -not -name 'spxw_chain.parquet' -not -name 'spxw_spot.parquet' | sort)
LOCAL_DATA=$(md5sum "${DATA[@]}" | sums)
"$SSH" -o BatchMode=yes "$H2HOST" "mkdir -p $H2/data $H2/logs $H2/results/feature_mcs_trees_work/chunks2"
RT_DATA=$("$SSH" -o BatchMode=yes "$H2HOST" "cd $RT && md5sum data/*.parquet" | sums)
UPLOAD=()
CP=""
for f in "${DATA[@]}"; do
  l=$(printf '%s\n' "$LOCAL_DATA" | awk -v f="$f" '$1 == f {print $2}')
  o=$(printf '%s\n' "$RT_DATA" | awk -v f="$f" '$1 == f {print $2}')
  if [ -n "$o" ] && [ "$o" = "$l" ]; then CP="$CP [ -f $H2/$f ] || cp -p $RT/$f $H2/$f;"; else UPLOAD+=("$f"); fi
done
[ -z "$CP" ] || "$SSH" -o BatchMode=yes "$H2HOST" "$CP true"
if [ "${#UPLOAD[@]}" -gt 0 ]; then
  echo "uploading ${#UPLOAD[@]} data files: ${UPLOAD[*]}"
  tar czf - "${UPLOAD[@]}" | "$SSH" -o BatchMode=yes "$H2HOST" "cd $H2 && tar xzf -"
fi
git -c core.autocrlf=false -c core.eol=lf archive --format=tar HEAD "${FROM_HEAD[@]}" > "$STAGE_DIR/head.tar"
tar cf "$STAGE_DIR/wt.tar" "${FROM_WORKTREE[@]}"
"$SSH" -o BatchMode=yes "$H2HOST" "cd $H2 && tar xf -" < "$STAGE_DIR/head.tar"
"$SSH" -o BatchMode=yes "$H2HOST" "cd $H2 && tar xf -" < "$STAGE_DIR/wt.tar"
LOCAL_ALL=$( { for f in "${FROM_HEAD[@]}"; do printf '%s %s\n' "$f" "$(git show "HEAD:$f" | md5sum | cut -d' ' -f1)"; done
               md5sum "${FROM_WORKTREE[@]}" "${DATA[@]}" | sums; } | sort)
REMOTE_ALL=$(printf '%s\n' "${FROM_HEAD[@]}" "${FROM_WORKTREE[@]}" "${DATA[@]}" | "$SSH" -o BatchMode=yes "$H2HOST" "cd $H2 && xargs md5sum" | sums)
BAD=$(diff <(printf '%s\n' "$LOCAL_ALL") <(printf '%s\n' "$REMOTE_ALL") || true)
[ -z "$BAD" ] || { echo "SHIP FAILED: remote md5 differs:"; printf '%s\n' "$BAD"; exit 1; }
echo "md5 OK: $(printf '%s\n' "$LOCAL_ALL" | wc -l) files (${#FROM_HEAD[@]} from HEAD $(git rev-parse --short HEAD), ${#FROM_WORKTREE[@]} from the worktree, ${#DATA[@]} data)"
"$SSH" -o BatchMode=yes "$H2HOST" "cd $H2 && set -e
for s in cluster/feature_mcs_trees_h2_task.sh cluster/submit_feature_mcs_trees_h2.sh; do bash -n \$s && echo \"bash -n ok: \$s\"; done
ls -d $RT/carc_env/harxhar/bin/python3.11"
echo "ship OK -> hoffman2:$H2"
