#!/bin/bash
# Feature MCS trees, the 16 bucket designs (2026-10-13): a FRESH deployment root on CARC, every file uploaded from the local
# copies (nothing copied on the cluster), for when files of the first root sit on a failed storage target of /scratch1
# (2026-10-12 23:40: reads of /scratch1/jc_905/harxhar-mcstrees/... and harxhar-mask/data hang).  Ships what the bucket
# packs need: src/ and the shipped tree spec (HEAD), src/data/loading.py, the generated XGBoost / forest specs and their
# generators and configurations, the MCS wrapper and driver, the designs / rows / v3 task files, the pack / check / submit
# scripts, the canary and gate flags of the first root (the gates passed there: results/feature_mcs_2026-10-08/trees/
# gate_<est>.csv), and data/ (no TreeSHAP library: the packs run with SHAP off).  Every file is md5-checked AND read end
# to end on the cluster under a timeout (a file on a failing target hangs; it is reported and the ship fails).
# Run from Git Bash locally:   [REMOTE=/scratch1/jc_905/harxhar-mcstrees-b] bash cluster/slurm/ship_feature_mcs_trees_fresh_carc.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
REMOTE=${REMOTE:-/scratch1/jc_905/harxhar-mcstrees-b}
S() {  # one ssh call; a banner-exchange failure (rc 255, nothing ran) is retried up to 3 times, 60 s apart
  local rc=0 i
  for i in 1 2 3 4; do
    rc=0; timeout "${T:-300}" "$SSH" -o BatchMode=yes -o ConnectTimeout=60 usc-discovery "$@" || rc=$?
    [ "$rc" = 255 ] || return "$rc"
    echo "  ssh rc 255 (attempt $i); retrying in 60 s" >&2; sleep 60
  done
  return "$rc"
}
quiet() { grep -v "reloaded with a version change\|python/3\|^$" || true; }
mapfile -t SRC < <(git ls-tree -r --name-only HEAD src | grep -v '/__pycache__/' | grep -v -x -e 'src/data/loading.py' -e 'src/evaluation/model_confidence_set.py')
FROM_HEAD=("${SRC[@]}" specs/causal_tune_trees.py)
FROM_WORKTREE=(
  src/data/loading.py
  specs/causal_tune_trees_xgbtop.py specs/causal_tune_trees_rf400.py specs/causal_tune_trees_rfdev.py specs/causal_tune_trees_mcs.py
  experiments/feature_mcs_trees.py experiments/trees_xgbtop_make_specs.py experiments/trees_rfp163_make_specs.py
  cluster/xgbtop_xgb_config.json cluster/rf400_rf_config.json
  cluster/feature_mcs_trees_designs.json cluster/feature_mcs_trees_rows.json
  cluster/feature_mcs_trees_projection_v3.csv
  cluster/feature_mcs_trees_fleet_lgbm_v3.txt cluster/feature_mcs_trees_fleet_xgb_v3.txt cluster/feature_mcs_trees_fleet_rf_v3.txt
  cluster/slurm/feature_mcs_trees_pack.sbatch cluster/slurm/feature_mcs_trees_check.sbatch
  cluster/slurm/submit_feature_mcs_trees.sh cluster/slurm/ship_feature_mcs_trees_fresh_carc.sh
  results/feature_mcs_trees_work/CANARY_OK
  results/feature_mcs_2026-10-08/trees/gate_lgbm.csv results/feature_mcs_2026-10-08/trees/gate_xgb.csv
  results/feature_mcs_2026-10-08/trees/gate_rf.csv
)
for f in "${FROM_HEAD[@]}"; do git cat-file -e "HEAD:$f" 2>/dev/null || { echo "SHIP REFUSED: $f is not in HEAD"; exit 1; }; done
git diff --quiet HEAD -- "${FROM_HEAD[@]}" || { echo "SHIP REFUSED: a HEAD file differs in the worktree"; git diff --stat HEAD -- "${FROM_HEAD[@]}"; exit 1; }
for f in "${FROM_WORKTREE[@]}"; do [ -f "$f" ] || { echo "SHIP REFUSED: $f missing"; exit 1; }; done
mapfile -t DATA < <(find data -maxdepth 1 -name '*.parquet' -not -name 'spxw_chain.parquet' -not -name 'spxw_spot.parquet' | sort)
sums() { awk '{p = $2; sub(/^\*/, "", p); print p, $1}' | sort; }

S "mkdir -p $REMOTE/data $REMOTE/logs $REMOTE/results/feature_mcs_trees_work" 2>&1 | quiet
echo "uploading ${#DATA[@]} data files, ${#FROM_HEAD[@]} HEAD files, ${#FROM_WORKTREE[@]} worktree files"
T=900 S "cd $REMOTE && tar xzf -" < <(tar czf - "${DATA[@]}")
git -c core.autocrlf=false -c core.eol=lf archive --format=tar HEAD "${FROM_HEAD[@]}" | T=600 S "cd $REMOTE && tar xf -"
tar cf - "${FROM_WORKTREE[@]}" | T=600 S "cd $REMOTE && tar xf -"
# the first root's flags (its canary and the three gates passed; the gate tables travel with them)
S "cd $REMOTE/results/feature_mcs_trees_work && for e in lgbm xgb rf; do printf 'gate passed in /scratch1/jc_905/harxhar-mcstrees (results/feature_mcs_2026-10-08/trees/gate_%s.csv: both anchors bit-identical to the stored T1 runs on the 866 days); re-deployed 2026-10-13\n' \$e > GATE_OK_\$e; done" 2>&1 | quiet

# md5 of every file, each read end to end under a timeout (a file on a failing storage target hangs instead of reading)
LOCAL_ALL=$( { for f in "${FROM_HEAD[@]}"; do printf '%s %s\n' "$f" "$(git show "HEAD:$f" | md5sum | cut -d' ' -f1)"; done
               md5sum "${FROM_WORKTREE[@]}" "${DATA[@]}" | sums; } | sort)
REMOTE_ALL=$(printf '%s\n' "${FROM_HEAD[@]}" "${FROM_WORKTREE[@]}" "${DATA[@]}" \
  | T=900 S "cd $REMOTE && while read -r f; do timeout 60 md5sum \"\$f\" || echo \"HANG \$f\"; done" | sums)
HANG=$(printf '%s\n' "$REMOTE_ALL" | grep '^HANG' || true)
[ -z "$HANG" ] || { echo "SHIP FAILED: files that do not read within 60 s:"; printf '%s\n' "$HANG"; exit 1; }
BAD=$(diff <(printf '%s\n' "$LOCAL_ALL") <(printf '%s\n' "$REMOTE_ALL") || true)
[ -z "$BAD" ] || { echo "SHIP FAILED: remote md5 differs:"; printf '%s\n' "$BAD"; exit 1; }
echo "md5 OK, every file read end to end: $(printf '%s\n' "$LOCAL_ALL" | wc -l) files (${#DATA[@]} data)"
printf '%s\n' "$LOCAL_ALL" > logs/ship_feature_mcs_trees_fresh_carc_manifest.txt

# login node: libraries, the generated specs, parse checks
T=600 S "cd $REMOTE && module load conda >/dev/null 2>&1 && source activate harxhar && set -e
python -c \"import lightgbm, xgboost, sklearn; from src.data.loading import get_bucket; print('lightgbm', lightgbm.__version__, 'xgboost', xgboost.__version__, 'sklearn', sklearn.__version__, 'all_features exog', len(get_bucket('all_features')))\"
python experiments/trees_xgbtop_make_specs.py --config cluster/xgbtop_xgb_config.json --token xgbtop --check
python experiments/trees_rfp163_make_specs.py --config cluster/rf400_rf_config.json --token rf400 --check
for s in cluster/slurm/feature_mcs_trees_*.sbatch cluster/slurm/submit_feature_mcs_trees.sh; do bash -n \$s; done
echo 'remote checks ok'" 2>&1 | quiet
echo "ship OK -> usc-discovery:$REMOTE"
