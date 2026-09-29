#!/bin/bash
# Ship what the all_features LASSO drop-column tasks read (see featimp_linear.sbatch) to
# the CARC root over native OpenSSH, FAIL on any md5 mismatch, and check that the spec
# and the src modules its estimator section imports are byte-identical there (they are
# not shipped: CARC's copies are the ones its linear campaigns ran).
#   bash cluster/slurm/ship_featimp_linear_carc.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
REMOTE=/scratch1/jc_905/harxhar-subsection
FILES=(
  experiments/feature_importance_1530.py
  experiments/feature_importance_1530_trees.py
  experiments/model_diagnostics_1530.py
  cluster/slurm/featimp_linear.sbatch
  cluster/slurm/submit_featimp_linear.sh
  cluster/featimp_linear_tasks.txt
  cluster/featimp_linear_tasks_canary.txt
  results/feature_importance_1530/_work/input_all_features.npz
  results/model_diagnostics_1530/capture_bar1600_all_features_reclasso.npz
  results/linear_subsection/arms_hoffman2/all_features/reclasso/tw2000/results_bar1600.csv
)
DEPS=(specs/causal_tune_linear.py src/models/reclasso_har.py src/backtest/multi_stage.py src/features/transforms/residualizer.py)
for f in "${FILES[@]}" "${DEPS[@]}"; do [ -f "$f" ] || { echo "missing locally: $f"; exit 1; }; done
sums() { awk '{p = $2; sub(/^\*/, "", p); print p, $1}'; }
LOCAL=$(md5sum "${FILES[@]}" | sums)
REMOTE_S=$(tar czf - "${FILES[@]}" | "$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && tar xzf - && md5sum ${FILES[*]}" | sums)
BAD=$(awk 'NR == FNR {r[$1] = $2; next} !($1 in r) || r[$1] != $2 {print $1}' <(printf '%s\n' "$REMOTE_S") <(printf '%s\n' "$LOCAL"))
[ -z "$BAD" ] || { echo "SHIP FAILED (md5) for: $BAD"; exit 1; }
LD=$(md5sum "${DEPS[@]}" | sums)
RD=$("$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && md5sum ${DEPS[*]}" | sums)
DBAD=$(awk 'NR == FNR {r[$1] = $2; next} !($1 in r) || r[$1] != $2 {print $1}' <(printf '%s\n' "$RD") <(printf '%s\n' "$LD"))
[ -z "$DBAD" ] || { echo "DEPENDENCY DIFFERS on CARC: $DBAD"; exit 1; }
"$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && bash -n cluster/slurm/featimp_linear.sbatch && bash -n cluster/slurm/submit_featimp_linear.sh && echo 'bash -n ok'"
echo "ship OK: ${#FILES[@]} files shipped (md5 equal), ${#DEPS[@]} dependencies equal"
