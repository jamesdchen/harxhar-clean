#!/bin/bash
# Ship the C1 feature-importance re-run on the de-duplicated per-bar design (checklist I7) to the
# CARC root (tar over native Windows OpenSSH) and FAIL if any shipped file's remote md5 differs
# from the local one.
#   trees  (default) the refit script (experiments/feature_importance_1530_trees.py), the
#          per-window mask helper (src/models/window_mask.py, commit f9a19b6), the Slurm scripts,
#          the task files and the input files of both roots (built locally by
#          `python experiments/feature_importance_1530_dedup.py design|inputs` and `--nomask share`);
#          then checks on the login node that ./pylib_trees/shap exists, the tree stack and the
#          mask helper import, and the shipped scripts parse.
#   linear the all_features lasso drop-column refits' inputs: the scripts, the coefficient capture,
#          the stored research forecasts (gate) and the input file; checks that the spec and the
#          src modules its estimator section imports are byte-identical on CARC (not shipped).
# Run from Git Bash locally:  bash cluster/slurm/ship_featimp_dedup_carc.sh [trees|linear]
set -euo pipefail
cd "$(dirname "$0")/../.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
REMOTE=/scratch1/jc_905/harxhar-subsection
MODE="${1:-trees}"
D=results/feature_importance_1530_dedup
DN=results/feature_importance_1530_dedup_nomask
case "$MODE" in
  trees)
    FILES=(
      experiments/feature_importance_1530_trees.py
      src/models/window_mask.py
      cluster/slurm/featimp_dedup_pack.sbatch
      cluster/slurm/featimp_dedup_collect.sbatch
      cluster/slurm/submit_featimp_dedup.sh
      cluster/featimp_dedup_tasks_canary.txt
      cluster/featimp_dedup_tasks.txt
      "$D/_work/input_baseline.npz" "$D/_work/input_live_feasible.npz" "$D/_work/input_all_features.npz"
      "$DN/_work/input_baseline.npz" "$DN/_work/input_live_feasible.npz" "$DN/_work/input_all_features.npz"
    )
    DEPS=()
    ;;
  linear)
    FILES=(
      experiments/feature_importance_1530.py
      experiments/feature_importance_1530_trees.py
      experiments/feature_importance_1530_dedup.py
      experiments/model_diagnostics_1530.py
      src/models/window_mask.py
      cluster/slurm/featimp_dedup_linear.sbatch
      cluster/slurm/submit_featimp_dedup.sh
      cluster/featimp_dedup_linear_tasks.txt
      "$D/_work/input_all_features.npz"
      "$D/_work/linear_captures/capture_bar1600_all_features_reclasso.npz"
      results/linear_subsection_dedup/arms_hoffman2/all_features/reclasso/tw2000/results_bar1600.csv
    )
    DEPS=(specs/causal_tune_linear.py src/models/reclasso_har.py src/backtest/multi_stage.py src/features/transforms/residualizer.py)
    ;;
  *) echo "usage: bash cluster/slurm/ship_featimp_dedup_carc.sh [trees|linear]"; exit 1 ;;
esac
for f in "${FILES[@]}" "${DEPS[@]}"; do
  [ -f "$f" ] || { echo "missing locally: $f"; exit 1; }
done
sums() { awk '{p = $2; sub(/^\*/, "", p); print p, $1}'; }
show() { awk '{printf "%s  %s\n", substr($2, 1, 12), $1}'; }
LOCAL_SHIP=$(md5sum "${FILES[@]}" | sums)
REMOTE_SHIP=$(tar czf - "${FILES[@]}" \
  | "$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && tar xzf - && md5sum ${FILES[*]}" 2>/dev/null | sums) \
  || { echo "SHIP FAILED: tar / ssh / remote md5sum returned non-zero"; exit 1; }
echo "local:";  printf '%s\n' "$LOCAL_SHIP" | show
echo "remote:"; printf '%s\n' "$REMOTE_SHIP" | show
BAD=$(awk 'NR == FNR {r[$1] = $2; next} !($1 in r) || r[$1] != $2 {print $1}' \
  <(printf '%s\n' "$REMOTE_SHIP") <(printf '%s\n' "$LOCAL_SHIP"))
[ -z "$BAD" ] || { echo "SHIP FAILED: remote md5 differs (or no remote line) for:"; printf '  %s\n' $BAD; exit 1; }
if [ "${#DEPS[@]}" -gt 0 ]; then
  LD=$(md5sum "${DEPS[@]}" | sums)
  RD=$("$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && md5sum ${DEPS[*]}" 2>/dev/null | sums)
  DBAD=$(awk 'NR == FNR {r[$1] = $2; next} !($1 in r) || r[$1] != $2 {print $1}' <(printf '%s\n' "$RD") <(printf '%s\n' "$LD"))
  [ -z "$DBAD" ] || { echo "DEPENDENCY DIFFERS on CARC: $DBAD"; exit 1; }
  echo "dependencies byte-identical on CARC: ${DEPS[*]}"
fi
"$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && module load conda >/dev/null 2>&1 && source activate harxhar && set -e
if [ ! -d $REMOTE/pylib_trees/shap ]; then
  echo 'FATAL: $REMOTE/pylib_trees/shap is missing (bash cluster/slurm/ship_trees_carc.sh installs it); this script does not'
  exit 1
fi
PYTHONPATH=\$PWD:\$PWD/pylib_trees python -c \"
import sys, shap, lightgbm, xgboost, sklearn, numpy, scipy
from src.models.window_mask import window_keep, scatter
print('python', sys.version.split()[0], 'numpy', numpy.__version__, 'scipy', scipy.__version__)
print('shap', shap.__version__, 'lightgbm', lightgbm.__version__, 'xgboost', xgboost.__version__, 'sklearn', sklearn.__version__)
\"
for s in ${FILES[*]}; do
  case \$s in
    *.py) python -c \"import ast, sys; ast.parse(open(sys.argv[1], encoding='utf-8').read()); print('parses:', sys.argv[1])\" \$s ;;
    *.sh|*.sbatch) bash -n \$s && echo \"bash -n ok: \$s\" ;;
  esac
done" 2>&1 | grep -v "reloaded\|python/3"
echo "ship OK ($MODE): ${#FILES[@]} files"
