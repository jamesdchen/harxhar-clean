#!/bin/bash
# moments_vix tree workstream (2026-10-06): ship the masked 16:00 tree rungs on bucket moments_vix to
# their Hoffman2 root (/u/scratch/j/jamesdc1/harxhar-momvix-trees), the twin of
# cluster/slurm/ship_trees_momvix_carc.sh:
#   1. data/*.parquet (the bar-panel files, never the option chain / spot) copied on Hoffman2 from agent
#      D's root (harxhar-optuna) where the md5 equals the local file, uploaded otherwise; agent D's
#      runtime (CARC's Python stack, used read only by cluster/trees_momvix_h2_task.sh) checked against
#      its md5 manifest (scratch is purged)
#   2. code: the SAME bytes as the CARC root (HEAD blobs for the campaign's files, the worktree for
#      src/data/loading.py and this workstream's files; same refusals) + the Hoffman2 scripts;
#      FAIL unless every remote md5 equals the local source
#   3. the gate's references: the stored masked chunk npz files of cluster/trees_momvix_gate_lines.txt,
#      relayed CARC (harxhar-mask) -> a local tarball -> Hoffman2 ./ref (the CARC layout); FAIL unless
#      Hoffman2 md5 == CARC md5
#   4. scripts parse; the shipped loader has the bucket (the runtime import runs in every task)
# Nothing is written into another agent's root.
# Run from Git Bash locally:  bash cluster/ship_trees_momvix_h2.sh
set -euo pipefail
cd "$(dirname "$0")/.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
H2=/u/scratch/j/jamesdc1/harxhar-momvix-trees
RT=/u/scratch/j/jamesdc1/harxhar-optuna
MASK=/scratch1/jc_905/harxhar-mask
STAGE_DIR=${STAGE_DIR:-${TMPDIR:-/tmp}/ship_trees_momvix_h2}
mkdir -p "$STAGE_DIR"
sums() { awk '{p = $2; sub(/^\*/, "", p); print p, $1}' | sort; }

mapfile -t SRC < <(git ls-tree -r --name-only HEAD src | grep -v '/__pycache__/' | grep -v -x 'src/data/loading.py')
FROM_HEAD=(
  "${SRC[@]}"
  specs/causal_tune_trees.py specs/causal_tune_trees_tuned.py specs/causal_tune_trees_tuned_jobs.py specs/causal_tune_linear.py
  experiments/trees_cadence_reduce_chunks.py experiments/reduce_trees_tuned_chunks.py experiments/trees_mask_kept_counts.py
  experiments/trees_cadence_make_tasks.py
)
FROM_WORKTREE=(
  src/data/loading.py
  experiments/trees_momvix_gate.py experiments/trees_momvix_make_tasks.py
  cluster/trees_momvix_tasks_untuned.txt cluster/trees_momvix_tasks_rs10.txt cluster/trees_momvix_tasks_rs1.txt
  cluster/trees_momvix_tasks_rs_canary.txt cluster/trees_momvix_gate_lines.txt
  cluster/trees_momvix_h2_task.sh cluster/submit_trees_momvix_h2.sh cluster/ship_trees_momvix_h2.sh
)
mapfile -t H2TF < <(ls cluster/trees_momvix_h2_tasks_*.txt 2>/dev/null || true)
FROM_WORKTREE+=("${H2TF[@]}")
git diff --quiet HEAD -- "${FROM_HEAD[@]}" || { echo "SHIP REFUSED: a file shipped from HEAD differs from HEAD:"; git diff --stat HEAD -- "${FROM_HEAD[@]}"; exit 1; }
NUM=$(git diff --numstat HEAD -- src/data/loading.py)
[ "$NUM" = "$(printf '5\t0\tsrc/data/loading.py')" ] || { echo "SHIP REFUSED: src/data/loading.py vs HEAD is '$NUM'"; exit 1; }
for f in "${FROM_WORKTREE[@]}"; do
  [ -f "$f" ] || { echo "SHIP REFUSED: $f missing"; exit 1; }
  ! grep -q $'\r' "$f" || { echo "SHIP REFUSED: $f has CRLF line endings"; exit 1; }
done
mapfile -t DATA < <(find data -maxdepth 1 -name '*.parquet' -not -name 'spxw_chain.parquet' -not -name 'spxw_spot.parquet' | sort)

# 1. root, data, runtime manifest
LOCAL_DATA=$(md5sum "${DATA[@]}" | sums)
# RUNTIME_MD5=0 skips the (slow) manifest check on a re-ship; every task still checks in-process
# glibc 2.28 and that no shared object comes from outside the runtime root
if [ "${RUNTIME_MD5:-1}" = 1 ]; then
  "$SSH" -o BatchMode=yes hoffman2 "mkdir -p $H2/data $H2/logs $H2/results $H2/ref && cd $RT && md5sum -c --quiet runtime_manifest.md5 && echo \"runtime $RT: \$(wc -l < runtime_manifest.md5) files match their manifest\""
else
  "$SSH" -o BatchMode=yes hoffman2 "mkdir -p $H2/data $H2/logs $H2/results $H2/ref"
fi
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

# 2. code
git -c core.autocrlf=false -c core.eol=lf archive --format=tar HEAD "${FROM_HEAD[@]}" > "$STAGE_DIR/head.tar"
tar cf "$STAGE_DIR/wt.tar" "${FROM_WORKTREE[@]}"
"$SSH" -o BatchMode=yes hoffman2 "cd $H2 && tar xf -" < "$STAGE_DIR/head.tar"
"$SSH" -o BatchMode=yes hoffman2 "cd $H2 && tar xf -" < "$STAGE_DIR/wt.tar"
LOCAL_ALL=$( { for f in "${FROM_HEAD[@]}"; do printf '%s %s\n' "$f" "$(git show "HEAD:$f" | md5sum | cut -d' ' -f1)"; done
               md5sum "${FROM_WORKTREE[@]}" "${DATA[@]}" | sums; } | sort)
REMOTE_ALL=$(printf '%s\n' "${FROM_HEAD[@]}" "${FROM_WORKTREE[@]}" "${DATA[@]}" | "$SSH" -o BatchMode=yes hoffman2 "cd $H2 && xargs md5sum" | sums)
BAD=$(diff <(printf '%s\n' "$LOCAL_ALL") <(printf '%s\n' "$REMOTE_ALL") || true)
[ -z "$BAD" ] || { echo "SHIP FAILED: remote md5 differs:"; printf '%s\n' "$BAD"; exit 1; }
echo "md5 OK: $(printf '%s\n' "$LOCAL_ALL" | wc -l) files (${#FROM_HEAD[@]} from HEAD $(git rev-parse --short HEAD), ${#FROM_WORKTREE[@]} from the worktree, ${#DATA[@]} data)"
if [ -f logs/ship_trees_momvix_carc_manifest.txt ]; then
  N_DIFF=$(join <(sort logs/ship_trees_momvix_carc_manifest.txt) <(printf '%s\n' "$REMOTE_ALL") | awk '$2 != $3' | wc -l)
  echo "against the CARC root's manifest: $N_DIFF common files differ"
  [ "$N_DIFF" = 0 ] || { echo "SHIP FAILED: Hoffman2 code differs from what CARC runs"; exit 1; }
fi

# 3. the gate's references from the masked campaign's CARC root
REFS=()
while read -r RUNG B M C S N; do
  [ -n "${N:-}" ] || continue
  case "$RUNG" in t1|t10) SR=linear_subsection_trees_mask ;; *) SR=linear_subsection_trees_tuned_mask ;; esac
  REFS+=("results/$SR/$RUNG/$B/bar1600/$M/tw2000/chunks/c$C/causal_tune_trees/$M/$B/trees_bar1600.npz")
done < cluster/trees_momvix_gate_lines.txt
printf '%s\n' "${REFS[@]}" | "$SSH" -o BatchMode=yes usc-discovery "cd $MASK && tar czf - -T -" > "$STAGE_DIR/ref.tgz" 2>/dev/null
CARC_REF=$(printf '%s\n' "${REFS[@]}" | "$SSH" -o BatchMode=yes usc-discovery "cd $MASK && xargs md5sum" 2>/dev/null | sums)
"$SSH" -o BatchMode=yes hoffman2 "cd $H2/ref && tar xzf -" < "$STAGE_DIR/ref.tgz"
H2_REF=$(printf '%s\n' "${REFS[@]}" | "$SSH" -o BatchMode=yes hoffman2 "cd $H2/ref && xargs md5sum" | sums)
[ "$CARC_REF" = "$H2_REF" ] && [ "$(printf '%s\n' "$H2_REF" | wc -l)" = "${#REFS[@]}" ] \
  || { echo "REFERENCE SHIP FAILED"; diff <(printf '%s\n' "$CARC_REF") <(printf '%s\n' "$H2_REF") | head; exit 1; }
echo "references: ${#REFS[@]} stored chunk npz files in $H2/ref, Hoffman2 md5 == CARC md5"

# 4. scripts parse; the bucket is in the shipped loader (the runtime import runs in every task: the
#    login node's memory limit kills a numba / shap import there)
"$SSH" -o BatchMode=yes hoffman2 "cd $H2 && set -e
for s in cluster/trees_momvix_h2_task.sh cluster/submit_trees_momvix_h2.sh; do bash -n \$s && echo \"bash -n ok: \$s\"; done
grep -n 'SUBGROUPS\[\"moments_vix\"\]' src/data/loading.py"
echo "ship OK -> hoffman2:$H2"
