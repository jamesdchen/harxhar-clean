#!/bin/bash
# Pooled tree walks on Hoffman2 (2026-10-07): ship the walk code, the bank's prep cache (the copy the CARC gate proved:
# harxhar-clean/results/prep_cache_all_features_b2.npz, md5 36957e81...; the Hoffman2 copies of that cache have another
# design matrix), the gate's bank chunks and the data to /u/scratch/j/jamesdc1/harxhar-pooltrees-h2 -- through the data
# transfer node (h2dtn; the login node's ssh banner times out under load).  HEAD blobs for src/ and the bank runner's
# helpers (refused when the worktree differs), the worktree for src/data/loading.py (LF) and this run's files; data copied
# on Hoffman2 from agent D's root where the md5 equals the local file.  FAILS unless every md5 matches.
# Run from Git Bash locally:  bash cluster/ship_pooltrees_h2.sh
set -euo pipefail
cd "$(dirname "$0")/.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
H2=/u/scratch/j/jamesdc1/harxhar-pooltrees-h2
RT=/u/scratch/j/jamesdc1/harxhar-optuna
BANK="../harxhar-clean/results/unification_carc"
CACHE="../harxhar-clean/results/prep_cache_all_features_b2.npz"
CACHE_MD5=36957e815818c7ee212ce0d783993de8
sums() { awk '{p = $2; sub(/^\*/, "", p); print p, $1}' | sort; }
mapfile -t SRC < <(git ls-tree -r --name-only HEAD src | grep -v '/__pycache__/' | grep -v -x 'src/data/loading.py')
FROM_HEAD=("${SRC[@]}" experiments/_bootstrap.py experiments/run_geometry_local.py experiments/run_unification.py experiments/dump_unif_yhat.py)
FROM_WORKTREE=(experiments/pooled_trees_walk.py experiments/pooled_trees_gate.py experiments/pooled_trees_merge.py experiments/pooled_trees_pieces.py
  cluster/xgbtop_xgb_config.json cluster/pooltree_rf400_config.json cluster/pooltree_lgbm_config.json
  cluster/pooltree_h2_task.sh cluster/submit_pooltrees_h2.sh cluster/ship_pooltrees_h2.sh cluster/pooltree_h2_gate_lines.txt)
for f in cluster/pooltree_h2_tasks_*.txt cluster/pooltree_tasks_rf_xcheck.txt cluster/pooltree_pieces_*_h2.txt; do [ -f "$f" ] && FROM_WORKTREE+=("$f"); done
git diff --quiet HEAD -- "${FROM_HEAD[@]}" || { echo "SHIP REFUSED: a HEAD file differs in the worktree"; exit 1; }
NUM=$(git diff --numstat HEAD -- src/data/loading.py | cut -f2)
[ "$NUM" = 0 ] || { echo "SHIP REFUSED: src/data/loading.py removes lines of HEAD"; exit 1; }
for f in "${FROM_WORKTREE[@]}"; do
  [ -f "$f" ] || { echo "SHIP REFUSED: $f missing"; exit 1; }
  ! grep -q $'\r' "$f" || { echo "SHIP REFUSED: $f has CRLF line endings"; exit 1; }
done
REFS=(tree_expert_10/chunk_000.npz tree_expert_10/chunk_050.npz)
mapfile -t DATA < <(find data -maxdepth 1 -name '*.parquet' -not -name 'spxw_chain.parquet' -not -name 'spxw_spot.parquet' | sort)
STAGE=$(mktemp -d); trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/wt/src/data" "$STAGE/ref"
tr -d '\r' < src/data/loading.py > "$STAGE/wt/src/data/loading.py"
for r in "${REFS[@]}"; do mkdir -p "$STAGE/ref/$(dirname "$r")"; cp -p "$BANK/$r" "$STAGE/ref/$r"; done
git -c core.autocrlf=false -c core.eol=lf archive --format=tar HEAD "${FROM_HEAD[@]}" > "$STAGE/all.tar"
tar rf "$STAGE/all.tar" "${FROM_WORKTREE[@]}"
tar rf "$STAGE/all.tar" -C "$STAGE/wt" src/data/loading.py
tar rf "$STAGE/all.tar" -C "$STAGE" ref
LOCAL_DATA=$(md5sum "${DATA[@]}" | sums)
RT_DATA=$("$SSH" -o BatchMode=yes h2dtn "cd $RT && md5sum data/*.parquet" | sums)
CP=""; UPLOAD=()
for f in "${DATA[@]}"; do
  l=$(printf '%s\n' "$LOCAL_DATA" | awk -v f="$f" '$1 == f {print $2}'); o=$(printf '%s\n' "$RT_DATA" | awk -v f="$f" '$1 == f {print $2}')
  if [ -n "$o" ] && [ "$o" = "$l" ]; then CP="$CP [ -f $H2/$f ] || cp -p $RT/$f $H2/$f;"; else UPLOAD+=("$f"); fi
done
"$SSH" -o BatchMode=yes h2dtn "mkdir -p $H2/data $H2/logs $H2/cache $H2/results && cd $H2 && tar xf - && $CP true" < "$STAGE/all.tar"
[ "${#UPLOAD[@]}" -eq 0 ] || tar cf - "${UPLOAD[@]}" | "$SSH" -o BatchMode=yes h2dtn "cd $H2 && tar xf -"
RC=$("$SSH" -o BatchMode=yes h2dtn "[ -f $H2/cache/prep_cache_all_features_b2.npz ] && md5sum $H2/cache/prep_cache_all_features_b2.npz | cut -d' ' -f1" || true)
if [ "$RC" != "$CACHE_MD5" ]; then
  echo "uploading the prep cache (2.75 GB)"
  "$SSH" -o BatchMode=yes h2dtn "cat > $H2/cache/prep_cache_all_features_b2.npz.part && mv $H2/cache/prep_cache_all_features_b2.npz.part $H2/cache/prep_cache_all_features_b2.npz" < "$CACHE"
  RC=$("$SSH" -o BatchMode=yes h2dtn "md5sum $H2/cache/prep_cache_all_features_b2.npz | cut -d' ' -f1")
fi
[ "$RC" = "$CACHE_MD5" ] || { echo "SHIP FAILED: cache md5 $RC != $CACHE_MD5"; exit 1; }
LOCAL_ALL=$( { for f in "${FROM_HEAD[@]}"; do printf '%s %s\n' "$f" "$(git show "HEAD:$f" | md5sum | cut -d' ' -f1)"; done
               md5sum "${FROM_WORKTREE[@]}" | sums; printf '%s\n' "$LOCAL_DATA"
               printf 'src/data/loading.py %s\n' "$(md5sum < "$STAGE/wt/src/data/loading.py" | cut -d' ' -f1)"
               for r in "${REFS[@]}"; do printf 'ref/%s %s\n' "$r" "$(md5sum < "$STAGE/ref/$r" | cut -d' ' -f1)"; done; } | sort)
REMOTE_ALL=$( { printf '%s\n' "${FROM_HEAD[@]}" "${FROM_WORKTREE[@]}" "${DATA[@]}" src/data/loading.py; for r in "${REFS[@]}"; do echo "ref/$r"; done; } \
  | "$SSH" -o BatchMode=yes h2dtn "cd $H2 && xargs md5sum" | sums)
BAD=$(diff <(printf '%s\n' "$LOCAL_ALL") <(printf '%s\n' "$REMOTE_ALL") || true)
[ -z "$BAD" ] || { echo "SHIP FAILED: remote md5 differs:"; printf '%s\n' "$BAD"; exit 1; }
echo "md5 OK: $(printf '%s\n' "$LOCAL_ALL" | wc -l) files + cache ($RC) -> hoffman2:$H2"
