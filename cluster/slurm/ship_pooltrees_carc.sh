#!/bin/bash
# Pooled tree walks (user request 2026-10-07): ship the walk code to /scratch1/jc_905/harxhar-pooltrees.
#   * HEAD blobs: src/ (but src/data/loading.py), the bank runner's helpers (experiments/_bootstrap.py,
#     experiments/run_geometry_local.py, experiments/run_unification.py, experiments/dump_unif_yhat.py); REFUSES when one
#     differs from HEAD in the worktree;
#   * worktree: src/data/loading.py (added comment / SUBGROUPS lines only; LF), experiments/pooled_trees_walk.py,
#     experiments/pooled_trees_gate.py, the configurations, gate lines, task files, sbatch / submit scripts;
#   * ref/: the bank chunks the gate reproduces (harxhar-clean/results/unification_carc/tree_expert_{00,10}/chunk_{000,050}
#     .npz, read only);
#   * data/ copied on the cluster from harxhar-mask where the md5 equals the local file;
#   * cache/prep_cache_all_features_b2.npz (the bank's panel, 2.75 GB) is uploaded separately (once) and only md5-checked here.
# FAILS unless every shipped file's remote md5 equals its source.   Run from Git Bash locally.
set -euo pipefail
cd "$(dirname "$0")/../.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
CH=${CARC_HOST:-usc-discovery1}
REMOTE=${CARC_ROOT:-/scratch1/jc_905/harxhar-pooltrees}   # 2026-10-13: a fresh root when /scratch1 storage targets fail
MASK=${DATA_SRC:-/scratch1/jc_905/harxhar-mask}          # where the data parquets are copied from on CARC (md5-checked)
BANK="../harxhar-clean/results/unification_carc"
CACHE_MD5=36957e815818c7ee212ce0d783993de8   # harxhar-clean/results/prep_cache_all_features_b2.npz (local)
sums() { awk '{p = $2; sub(/^\*/, "", p); print p, $1}' | sort; }
mapfile -t SRC < <(git ls-tree -r --name-only HEAD src | grep -v '/__pycache__/' | grep -v -x 'src/data/loading.py')
FROM_HEAD=("${SRC[@]}" experiments/_bootstrap.py experiments/run_geometry_local.py experiments/run_unification.py experiments/dump_unif_yhat.py)
FROM_WORKTREE=(experiments/pooled_trees_walk.py experiments/pooled_trees_gate.py experiments/pooled_trees_merge.py experiments/pooled_trees_pieces.py
  cluster/pooltree_lgbm_config.json cluster/xgbtop_xgb_config.json cluster/pooltree_rf400_config.json
  cluster/pooltree_gate_lines.txt cluster/pooltree_gate_threads_lines.txt cluster/slurm/pooltree_task.sbatch cluster/slurm/pooltree_gate.sbatch
  cluster/slurm/pooltree_dump.sbatch cluster/slurm/pooltree_piece.sbatch cluster/slurm/submit_pooltrees.sh cluster/slurm/ship_pooltrees_carc.sh
  cluster/slurm/pooltree_af_wave.sh cluster/slurm/pooltree_sweep.sh cluster/slurm/pooltree_af_status.sh cluster/slurm/pooltree_af_coverage.sh
  cluster/pooltree_af_exclude_chunks.txt experiments/pooled_trees_xcheck.py experiments/pooled_trees_manifest_wave.py)
for f in cluster/pooltree_tasks_*.txt cluster/pooltree_pieces_*.txt; do [ -f "$f" ] && FROM_WORKTREE+=("$f"); done
# HEAD blobs are shipped either way; ALLOW_WT_DIFF=1 accepts a worktree edit by someone else to a shipped HEAD file
git diff --quiet HEAD -- "${FROM_HEAD[@]}" || [ "${ALLOW_WT_DIFF:-0}" = 1 ] || { echo "SHIP REFUSED: a HEAD file differs in the worktree"; git diff --stat HEAD -- "${FROM_HEAD[@]}"; exit 1; }
NUM=$(git diff --numstat HEAD -- src/data/loading.py | cut -f2)
[ "$NUM" = 0 ] || { echo "SHIP REFUSED: src/data/loading.py removes $NUM lines of HEAD"; exit 1; }
BADADD=$(git diff -U0 HEAD -- src/data/loading.py | grep '^+[^+]' | grep -v -E '^\+#|^\+SUBGROUPS\["[a-z_]+"\] = ' || true)
[ -z "$BADADD" ] || { echo "SHIP REFUSED: added loading.py lines other than comments / SUBGROUPS lines"; exit 1; }
for f in "${FROM_WORKTREE[@]}"; do
  [ -f "$f" ] || { echo "SHIP REFUSED: $f missing"; exit 1; }
  ! grep -q $'\r' "$f" || { echo "SHIP REFUSED: $f has CRLF line endings"; exit 1; }
done
REFS=(tree_expert_00/chunk_000.npz tree_expert_00/chunk_050.npz tree_expert_10/chunk_000.npz tree_expert_10/chunk_050.npz)
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
CP=""
for f in "${DATA[@]}"; do CP="$CP [ -f $REMOTE/$f ] || cp -p $MASK/$f $REMOTE/$f;"; done
"$SSH" -o BatchMode=yes "$CH" "mkdir -p $REMOTE/data $REMOTE/logs $REMOTE/cache $REMOTE/results && cd $REMOTE && tar xf - && $CP true" < "$STAGE/all.tar" 2>&1 | grep -v "reloaded\|python/3\|^$" || true
LOCAL_ALL=$( { for f in "${FROM_HEAD[@]}"; do printf '%s %s\n' "$f" "$(git show "HEAD:$f" | md5sum | cut -d' ' -f1)"; done
               md5sum "${FROM_WORKTREE[@]}" | sums; printf '%s\n' "$LOCAL_DATA"
               printf 'src/data/loading.py %s\n' "$(md5sum < "$STAGE/wt/src/data/loading.py" | cut -d' ' -f1)"
               for r in "${REFS[@]}"; do printf 'ref/%s %s\n' "$r" "$(md5sum < "$STAGE/ref/$r" | cut -d' ' -f1)"; done; } | sort)
REMOTE_ALL=$( { printf '%s\n' "${FROM_HEAD[@]}" "${FROM_WORKTREE[@]}" "${DATA[@]}" src/data/loading.py; for r in "${REFS[@]}"; do echo "ref/$r"; done; } \
  | "$SSH" -o BatchMode=yes "$CH" "cd $REMOTE && xargs md5sum" 2>/dev/null | sums)
BAD=$(diff <(printf '%s\n' "$LOCAL_ALL") <(printf '%s\n' "$REMOTE_ALL") || true)
[ -z "$BAD" ] || { echo "SHIP FAILED: remote md5 differs:"; printf '%s\n' "$BAD"; exit 1; }
echo "md5 OK: $(printf '%s\n' "$LOCAL_ALL" | wc -l) files (${#FROM_HEAD[@]} from HEAD $(git rev-parse --short HEAD), ${#FROM_WORKTREE[@]} worktree, ${#REFS[@]} bank chunks, ${#DATA[@]} data)"
RC=$("$SSH" -o BatchMode=yes "$CH" "cd $REMOTE && [ -f cache/prep_cache_all_features_b2.npz ] && md5sum cache/prep_cache_all_features_b2.npz | cut -d' ' -f1" 2>/dev/null | grep -E '^[0-9a-f]{32}$' || true)
[ "$RC" = "$CACHE_MD5" ] && echo "cache OK ($RC)" || echo "WARNING: cache/prep_cache_all_features_b2.npz missing or md5 '$RC' != $CACHE_MD5"
echo "ship OK -> $CH:$REMOTE"
