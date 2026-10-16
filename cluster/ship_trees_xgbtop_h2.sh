#!/bin/bash
# xgbtop on Hoffman2 (2026-10-07): ship the per-bar XGBoost (xgb_opt_00) runs' code, task files, configurations, the
# reproduction gate's CARC reference chunks and the published tables to /u/scratch/j/jamesdc1/harxhar-xgbtop-h2
# (cluster/submit_trees_xgbtop_h2.sh, cluster/trees_xgbtop_h2_task.sh).  Copy of cluster/ship_trees_rfh2_h2.sh:
#   * HEAD blobs for the campaign's unchanged code (src/ but src/data/loading.py, the tree / tuned / Optuna / linear specs
#     and job modules, the reducers); REFUSES when one differs from HEAD in the worktree;
#   * the worktree for src/data/loading.py (added comment / SUBGROUPS lines only: checked; shipped as LF) and this run's
#     files (generator, generated specs, wrappers, gates, task scripts, task files, configurations);
#   * ref/: the CARC reference chunks of the gate lines (trees_bar1600.npz + results_bar1600.csv of each stored epyc-7513
#     chunk), streamed CARC -> Hoffman2 through this machine (no local copy), md5 checked on both ends;
#   * ref_tables/: results/spxw_pnl/'s yhat_subtree_daily_<set>_xgb / yhat_subtree_xgb_<set> tables;
#   * data copied on Hoffman2 from agent D's root where the md5 equals the local file; agent D's runtime used read only.
# FAILS unless every shipped file's remote md5 equals its source.
# Run from Git Bash locally:  bash cluster/ship_trees_xgbtop_h2.sh
set -euo pipefail
cd "$(dirname "$0")/.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
H2=/u/scratch/j/jamesdc1/harxhar-xgbtop-h2
RT=/u/scratch/j/jamesdc1/harxhar-optuna
sums() { awk '{p = $2; sub(/^\*/, "", p); print p, $1}' | sort; }
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT

mapfile -t SRC < <(git ls-tree -r --name-only HEAD src | grep -v '/__pycache__/' | grep -v -x 'src/data/loading.py')
FROM_HEAD=(
  "${SRC[@]}"
  specs/causal_tune_trees.py specs/causal_tune_trees_tuned.py specs/causal_tune_trees_tuned_jobs.py
  specs/causal_tune_trees_optuna.py specs/causal_tune_trees_optuna_jobs.py specs/causal_tune_linear.py
  experiments/trees_cadence_reduce_chunks.py experiments/reduce_trees_tuned_chunks.py experiments/reduce_trees_optuna.py
)
FROM_WORKTREE=(
  experiments/trees_momvix_gate.py experiments/trees_xgbtop_make_specs.py experiments/trees_xgbtop_gate.py
  experiments/trees_xgbtop_tuned_gate.py experiments/trees_xgbtop_optuna_gate.py
  specs/causal_tune_trees_xgbtop.py specs/causal_tune_trees_xgb06.py
  specs/causal_tune_trees_tuned_xgbtop.py specs/causal_tune_trees_optuna_xgbtop.py
  cluster/xgbtop_xgb_config.json cluster/xgbtop_shipped_xgb.json
  cluster/trees_xgbtop_h2_task.sh cluster/submit_trees_xgbtop_h2.sh cluster/ship_trees_xgbtop_h2.sh
  cluster/trees_xgbtop_h2_gate_lines.txt cluster/trees_xgbtop_tasks_untuned.txt
)
for f in cluster/trees_xgbtop_h2_tasks_*.txt cluster/xgbtop_optuna_tasks_*.txt cluster/trees_xgbtop_tasks_rs*.txt; do
  [ -f "$f" ] && FROM_WORKTREE+=("$f")
done
git diff --quiet HEAD -- "${FROM_HEAD[@]}" || { echo "SHIP REFUSED: a file shipped from HEAD differs from HEAD:"; git diff --stat HEAD -- "${FROM_HEAD[@]}"; exit 1; }
NUM=$(git diff --numstat HEAD -- src/data/loading.py | cut -f2)
[ "$NUM" = 0 ] || { echo "SHIP REFUSED: src/data/loading.py removes $NUM lines of HEAD"; exit 1; }
BADADD=$(git diff -U0 HEAD -- src/data/loading.py | grep '^+[^+]' | grep -v -E '^\+#|^\+SUBGROUPS\["[a-z_]+"\] = ' || true)
[ -z "$BADADD" ] || { echo "SHIP REFUSED: added loading.py lines other than comments / SUBGROUPS lines"; exit 1; }
for f in "${FROM_WORKTREE[@]}"; do
  [ -f "$f" ] || { echo "SHIP REFUSED: $f missing"; exit 1; }
  ! grep -q $'\r' "$f" || { echo "SHIP REFUSED: $f has CRLF line endings"; exit 1; }
done
TABLES=()
for b in baseline all_features lean_vol_vix; do TABLES+=("yhat_subtree_daily_${b}_xgb.parquet" "yhat_subtree_xgb_${b}.parquet"); done
mkdir -p "$STAGE/wt/src/data" "$STAGE/ref_tables"
tr -d '\r' < src/data/loading.py > "$STAGE/wt/src/data/loading.py"
for t in "${TABLES[@]}"; do cp -p "results/spxw_pnl/$t" "$STAGE/ref_tables/$t"; done
mapfile -t DATA < <(find data -maxdepth 1 -name '*.parquet' -not -name 'spxw_chain.parquet' -not -name 'spxw_spot.parquet' | sort)

# 1. root and data (copied from agent D's root where the md5 matches)
"$SSH" -o BatchMode=yes hoffman2 "mkdir -p $H2/data $H2/logs $H2/results $H2/ref"
LOCAL_DATA=$(md5sum "${DATA[@]}" | sums)
RT_DATA=$("$SSH" -o BatchMode=yes hoffman2 "cd $RT && md5sum data/*.parquet" | sums)
UPLOAD=()
CP=""
for f in "${DATA[@]}"; do
  l=$(printf '%s\n' "$LOCAL_DATA" | awk -v f="$f" '$1 == f {print $2}')
  o=$(printf '%s\n' "$RT_DATA" | awk -v f="$f" '$1 == f {print $2}')
  if [ -n "$o" ] && [ "$o" = "$l" ]; then CP="$CP [ -f $H2/$f ] || cp -p $RT/$f $H2/$f;"; else UPLOAD+=("$f"); fi
done
[ -z "$CP" ] || "$SSH" -o BatchMode=yes hoffman2 "$CP true"
if [ "${#UPLOAD[@]}" -gt 0 ]; then
  echo "uploading ${#UPLOAD[@]} data files: ${UPLOAD[*]}"
  tar czf - "${UPLOAD[@]}" | "$SSH" -o BatchMode=yes hoffman2 "cd $H2 && tar xzf -"
fi

# 2. code (one tar: HEAD blobs + worktree + loading.py + tables)
git -c core.autocrlf=false -c core.eol=lf archive --format=tar HEAD "${FROM_HEAD[@]}" > "$STAGE/all.tar"
tar rf "$STAGE/all.tar" "${FROM_WORKTREE[@]}"
tar rf "$STAGE/all.tar" -C "$STAGE/wt" src/data/loading.py
tar rf "$STAGE/all.tar" -C "$STAGE" ref_tables
"$SSH" -o BatchMode=yes hoffman2 "cd $H2 && tar xf -" < "$STAGE/all.tar"

# 3. the gate's CARC reference chunks, streamed CARC -> Hoffman2 (the stored roots of cluster/trees_xgbtop_gate_lines.txt)
REFS=()
while read -r RUNG B C ST N REFROOT STORED; do
  [ -n "${RUNG:-}" ] || continue
  for f in trees_bar1600.npz results_bar1600.csv; do
    REFS+=("$REFROOT|results/$STORED/$RUNG/$B/bar1600/xgb/tw2000/chunks/c$C/causal_tune_trees/xgb/$B/$f")
  done
done < cluster/trees_xgbtop_gate_lines.txt
CARC_SUMS=""
for root in $(printf '%s\n' "${REFS[@]}" | cut -d'|' -f1 | sort -u); do
  mapfile -t RF < <(printf '%s\n' "${REFS[@]}" | awk -F'|' -v r="$root" '$1 == r {print $2}')
  "$SSH" -o BatchMode=yes usc-discovery "cd $root && tar cf - ${RF[*]}" 2>/dev/null | "$SSH" -o BatchMode=yes hoffman2 "cd $H2/ref && tar xf -"
  CARC_SUMS="$CARC_SUMS
$("$SSH" -o BatchMode=yes usc-discovery "cd $root && md5sum ${RF[*]}" 2>/dev/null | sed 's|  |  ref/|')"
done
CARC_SUMS=$(printf '%s\n' "$CARC_SUMS" | grep . | sums)

# 4. md5: HEAD files against HEAD's blobs, worktree / tables / data against the local files, refs against CARC
LOCAL_ALL=$( { for f in "${FROM_HEAD[@]}"; do printf '%s %s\n' "$f" "$(git show "HEAD:$f" | md5sum | cut -d' ' -f1)"; done
               md5sum "${FROM_WORKTREE[@]}" "${DATA[@]}" | sums
               printf 'src/data/loading.py %s\n' "$(md5sum < "$STAGE/wt/src/data/loading.py" | cut -d' ' -f1)"
               for t in "${TABLES[@]}"; do printf 'ref_tables/%s %s\n' "$t" "$(md5sum < "$STAGE/ref_tables/$t" | cut -d' ' -f1)"; done
               printf '%s\n' "$CARC_SUMS"; } | grep . | sort)
REMOTE_ALL=$( { printf '%s\n' "${FROM_HEAD[@]}" "${FROM_WORKTREE[@]}" "${DATA[@]}" src/data/loading.py
                for t in "${TABLES[@]}"; do echo "ref_tables/$t"; done
                printf '%s\n' "${REFS[@]}" | cut -d'|' -f2 | sed 's|^|ref/|'; } | "$SSH" -o BatchMode=yes hoffman2 "cd $H2 && xargs md5sum" | sums)
BAD=$(diff <(printf '%s\n' "$LOCAL_ALL") <(printf '%s\n' "$REMOTE_ALL") || true)
[ -z "$BAD" ] || { echo "SHIP FAILED: remote md5 differs:"; printf '%s\n' "$BAD"; exit 1; }
echo "md5 OK: $(printf '%s\n' "$LOCAL_ALL" | wc -l) files (${#FROM_HEAD[@]} from HEAD $(git rev-parse --short HEAD), ${#FROM_WORKTREE[@]} from the worktree, ${#TABLES[@]} tables, ${#REFS[@]} CARC reference files, ${#DATA[@]} data)"
"$SSH" -o BatchMode=yes hoffman2 "cd $H2 && set -e
for s in cluster/trees_xgbtop_h2_task.sh cluster/submit_trees_xgbtop_h2.sh; do bash -n \$s && echo \"bash -n ok: \$s\"; done
grep -c 'SUBGROUPS\[\"lean_vol_vix\"\]' src/data/loading.py; ls -d $RT/carc_env/harxhar/bin/python3.11"
echo "ship OK -> hoffman2:$H2"
