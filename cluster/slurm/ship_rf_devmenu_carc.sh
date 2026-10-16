#!/bin/bash
# Random-forest dev-span menu study (2026-10-06): ship to a CARC deployment root of its own,
# /scratch1/jc_905/harxhar-rf-devmenu (no other agent's root is touched), in ONE ssh session.
#   * from HEAD (LF blobs): experiments/_bootstrap.py and experiments/tree_menu_dev.py (the procedure
#     the study imports); REFUSES when either differs from HEAD in the worktree;
#   * from the WORKTREE (this workstream may not commit): the study script and its sbatch / submit /
#     ship / pull scripts (LF line endings checked);
#   * data: the dev matrices results/rf_devmenu_2026-10-06/_work/dev/{X,y,names}.npy + dev_manifest.json
#     (built locally by `tree_menu_dev_rf.py extract`; every stage re-checks their sha256).
# The two tar streams travel as one gzip stream (GNU tar -i reads the concatenation); the same session
# checks every shipped file's md5 against the manifest (rf_devmenu_manifest.md5, re-checked by the
# submit script), imports the libraries with the job's interpreter, parses the scripts and prints the
# login environment the jobs would inherit.  FAILS on any mismatch.
# Run from Git Bash locally:   bash cluster/slurm/ship_rf_devmenu_carc.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
REMOTE=/scratch1/jc_905/harxhar-rf-devmenu
PY=/home1/jc_905/.conda/envs/harxhar/bin/python
FROM_HEAD=(experiments/_bootstrap.py experiments/tree_menu_dev.py)
FROM_WORKTREE=(
  experiments/tree_menu_dev_rf.py
  experiments/tree_menu_dev_rf_seqgate.py
  cluster/slurm/rf_devmenu_task.sbatch
  cluster/slurm/submit_rf_devmenu.sh
  cluster/slurm/ship_rf_devmenu_carc.sh
  cluster/slurm/pull_rf_devmenu_carc.sh
)
D=results/rf_devmenu_2026-10-06/_work/dev
DATA=("$D/X.npy" "$D/y.npy" "$D/names.npy" "$D/dev_manifest.json")
for f in "${FROM_HEAD[@]}"; do
  git cat-file -e "HEAD:$f" 2>/dev/null || { echo "SHIP REFUSED: $f is not in HEAD"; exit 1; }
done
if ! git diff --quiet HEAD -- "${FROM_HEAD[@]}"; then
  echo "SHIP REFUSED: a file shipped from HEAD differs from HEAD in the worktree:"
  git diff --stat HEAD -- "${FROM_HEAD[@]}"
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
LOCAL_ALL=$( { for f in "${FROM_HEAD[@]}"; do printf '%s %s\n' "$f" "$(git show "HEAD:$f" | md5sum | cut -d' ' -f1)"; done
               md5sum "${FROM_WORKTREE[@]}" "${DATA[@]}" | sums; } | sort)
mkdir -p logs
# SKIP_DATA=1: code only -- the remote dev matrices stay as they are (running workers memory-map them)
# and the md5 check below still verifies them against the local copies
SHIP_DATA=("${DATA[@]}")
[ "${SKIP_DATA:-0}" = 1 ] && SHIP_DATA=()
MAN=logs/rf_devmenu_manifest.md5
printf '%s\n' "$LOCAL_ALL" | awk '{print $2 "  " $1}' > "$MAN"

{ git -c core.autocrlf=false -c core.eol=lf archive --format=tar HEAD "${FROM_HEAD[@]}"
  tar cf - "${FROM_WORKTREE[@]}" "${SHIP_DATA[@]}" "$MAN"; } | gzip -1 -c \
  | "$SSH" -o BatchMode=yes usc-discovery "mkdir -p $REMOTE/logs && cd $REMOTE \
  && tar xizf - && mv $MAN rf_devmenu_manifest.md5 && md5sum -c --quiet rf_devmenu_manifest.md5 && echo 'remote md5 OK' \
  && set -e
echo \"login env: CONDA_PREFIX='\${CONDA_PREFIX:-}' CONDA_DEFAULT_ENV='\${CONDA_DEFAULT_ENV:-}' PYTHONPATH='\${PYTHONPATH:-}' LD_LIBRARY_PATH='\${LD_LIBRARY_PATH:-}' python=\$(command -v python || true)\"
env -u PYTHONPATH -u PYTHONHOME PYTHONPATH=\$PWD:\$PWD/experiments $PY -c \"
import sys, numpy, sklearn, lightgbm, xgboost, optuna, scipy
print('python', sys.version.split()[0], 'numpy', numpy.__version__, numpy.__file__, 'sklearn', sklearn.__version__, 'lightgbm', lightgbm.__version__, 'xgboost', xgboost.__version__, 'optuna', optuna.__version__, 'scipy', scipy.__version__)
import tree_menu_dev_rf as M
M._check_dev(M.DEV_DIR)
print('dev matrices sha256 OK', M.TMD.WINDOW, M.TMD.SCREEN_REFIT, M.TMD.DEV_END_DATE)
\"
for s in cluster/slurm/rf_devmenu_task.sbatch cluster/slurm/submit_rf_devmenu.sh; do bash -n \$s; done
echo 'remote checks ok'
echo \"jobs of \$USER: \$(squeue -u \$USER -h | wc -l)\"" 2>&1 | quiet
echo "ship done -> usc-discovery:$REMOTE ($(printf '%s\n' "$LOCAL_ALL" | wc -l) files, HEAD $(git rev-parse --short HEAD)); check 'remote md5 OK' and 'remote checks ok' above"
