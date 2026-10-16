#!/bin/bash
# Random forest / XGBoost pre-tune study (2026-10-06): ship to a CARC deployment root of its own,
# /scratch1/jc_905/harxhar-rfxgb-pretune (no other agent's root is touched), in ONE ssh session
# (the login node throttles bursts of connections: 2026-09-30, 2026-10-06).
#   * code from the COMMIT (git archive HEAD, LF blobs): src/ and the four specs the study imports
#     or reads (the tree spec, the tuned and Optuna jobs modules, the linear spec's constants);
#     REFUSES when one of them differs from HEAD in the worktree (src/data/loading.py excepted: HEAD's
#     version is shipped; the study reads cached designs and never builds one);
#   * from the WORKTREE (this workstream may not commit): the study's two modules, its sbatch / submit
#     / pull scripts and task files (LF line endings checked);
#   * data: the six cached designs of results/trees_pretune_2026-10-06/_work/design (+ their array
#     hashes, checked by every stage) and the stored masked tables the gate compares with;
#   * pylib_trees/ (the campaign's extra site packages) copied on the cluster from harxhar-mask.
# The two tar streams (HEAD archive, worktree + data + md5 manifest) travel as one gzip stream
# (GNU tar -i reads the concatenation); the same session then checks every shipped file's md5 against
# the manifest (rfxgb_pretune_manifest.md5, re-checked by the submit script), the libraries, parses
# every shipped script and lists the CPU classes of partition main.  FAILS on any mismatch.
# Run from Git Bash locally:   bash cluster/slurm/ship_rfxgb_pretune_carc.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
REMOTE=/scratch1/jc_905/harxhar-rfxgb-pretune
MASK=/scratch1/jc_905/harxhar-mask

mapfile -t SRC < <(git ls-tree -r --name-only HEAD src | grep -v '/__pycache__/')
FROM_HEAD=(
  "${SRC[@]}"
  specs/causal_tune_trees.py
  specs/causal_tune_trees_tuned_jobs.py
  specs/causal_tune_trees_optuna_jobs.py
  specs/causal_tune_linear.py
)
FROM_WORKTREE=(
  experiments/close_trees_pretune_rfxgb.py
  experiments/close_trees_pretune_rfxgb_jobs.py
  cluster/slurm/rfxgb_pretune_task.sbatch
  cluster/slurm/submit_rfxgb_pretune.sh
  cluster/slurm/ship_rfxgb_pretune_carc.sh
  cluster/slurm/pull_rfxgb_pretune_carc.sh
  cluster/slurm/rfxgb_pretune_tasks_pretune.txt
  cluster/slurm/rfxgb_pretune_tasks_plan.txt
  cluster/slurm/rfxgb_pretune_tasks_control.txt
  cluster/slurm/rfxgb_pretune_tasks_arms.txt
)
D=results/trees_pretune_2026-10-06/_work/design
DATA=("$D/design_sha256.json")
# moments_vix dropped (user, 2026-10-06): lean_vol_vix instead; it has no stored tree tables (the gate records that)
for b in all_features baseline lean_vol_vix; do
  DATA+=("$D/design_bar1600_$b.npz" "$D/design_last30_$b.npz")
  for m in rf xgb; do
    for f in "results/spxw_pnl/yhat_subtree_${m}_$b.parquet" "results/spxw_pnl/yhat_subtree_daily_${b}_$m.parquet"; do
      [ -f "$f" ] && DATA+=("$f")
    done
  done
done
for f in "${FROM_HEAD[@]}"; do
  git cat-file -e "HEAD:$f" 2>/dev/null || { echo "SHIP REFUSED: $f is not in HEAD"; exit 1; }
done
CHK=()
for f in "${FROM_HEAD[@]}"; do [ "$f" = src/data/loading.py ] || CHK+=("$f"); done
if ! git diff --quiet HEAD -- "${CHK[@]}"; then
  echo "SHIP REFUSED: a file shipped from HEAD differs from HEAD in the worktree:"
  git diff --stat HEAD -- "${CHK[@]}"
  exit 1
fi
for f in "${FROM_WORKTREE[@]}" "${DATA[@]}"; do
  [ -f "$f" ] || { echo "SHIP REFUSED: $f missing"; exit 1; }
done
for f in "${FROM_WORKTREE[@]}"; do
  ! grep -q $'\r' "$f" || { echo "SHIP REFUSED: $f has CRLF line endings"; exit 1; }
done
sums() { awk '{p = $2; sub(/^\*/, "", p); print p, $1}' | sort; }
quiet() { grep -v "reloaded with a version change\|python/3\|^$" || true; }

# the md5 manifest: HEAD files from HEAD's blobs, worktree files and data from the local files
LOCAL_ALL=$( { for f in "${FROM_HEAD[@]}"; do printf '%s %s\n' "$f" "$(git show "HEAD:$f" | md5sum | cut -d' ' -f1)"; done
               md5sum "${FROM_WORKTREE[@]}" "${DATA[@]}" | sums; } | sort)
mkdir -p logs
MAN=logs/rfxgb_pretune_manifest.md5
printf '%s\n' "$LOCAL_ALL" | awk '{print $2 "  " $1}' > "$MAN"

{ git -c core.autocrlf=false -c core.eol=lf archive --format=tar HEAD "${FROM_HEAD[@]}"
  tar cf - "${FROM_WORKTREE[@]}" "${DATA[@]}" "$MAN"; } | gzip -c \
  | "$SSH" -o BatchMode=yes usc-discovery "mkdir -p $REMOTE/logs && cd $REMOTE && { [ -d pylib_trees ] || cp -rp $MASK/pylib_trees pylib_trees; } \
  && tar xizf - && mv $MAN rfxgb_pretune_manifest.md5 && md5sum -c --quiet rfxgb_pretune_manifest.md5 && echo 'remote md5 OK' \
  && module load conda >/dev/null 2>&1 && source activate harxhar && set -e
export PYTHONPATH=\$PWD:\$PWD/pylib_trees
python -c \"
import sys, numpy, pandas, sklearn, xgboost, optuna
print('python', sys.version.split()[0], 'numpy', numpy.__version__, 'pandas', pandas.__version__, 'sklearn', sklearn.__version__, 'xgboost', xgboost.__version__, 'optuna', optuna.__version__)
\"
for p in experiments/close_trees_pretune_rfxgb.py experiments/close_trees_pretune_rfxgb_jobs.py specs/*.py; do
  python -c \"import ast, sys; ast.parse(open(sys.argv[1], encoding='utf-8').read())\" \$p
done
for s in cluster/slurm/rfxgb_pretune_task.sbatch cluster/slurm/submit_rfxgb_pretune.sh; do bash -n \$s; done
echo 'remote checks ok'
echo 'partition main by feature: state nodes cpus(alloc/idle/other/total)'
sinfo -p main -h -o '%f %T %D %C' | sort | head -40
echo \"jobs of \$USER: \$(squeue -u \$USER -h | wc -l)\"" 2>&1 | quiet
echo "ship done -> usc-discovery:$REMOTE ($(printf '%s\n' "$LOCAL_ALL" | wc -l) files, HEAD $(git rev-parse --short HEAD)); check 'remote md5 OK' and 'remote checks ok' above"
