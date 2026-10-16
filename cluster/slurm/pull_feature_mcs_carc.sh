#!/bin/bash
# Pull the feature Model Confidence Set study (2026-10-08) from CARC: every candidate's forecast file
# (_work/models/<model>.npz + .json), the gate and binding records, the kernel build record and the job logs,
# md5-checked; then the local report:  python experiments/feature_mcs_report.py
#
#   bash cluster/slurm/pull_feature_mcs_carc.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
REMOTE=/scratch1/jc_905/harxhar-feature-mcs
R=results/feature_mcs_2026-10-08
N=$("$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && ls $R/_work/models/*.npz | wc -l")
echo "remote: $N model files"
LIST="$R/_work/models $R/_work/gate.csv $R/_work/GATE_OK $R/_work/bindings.csv $R/_work/kernel/build.json"
# ridge and lasso (2026-10-08), when present
EXTRA=$("$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && ls -d $R/_work/models_ridge $R/_work/models_lasso $R/_work/gate_ridge.csv $R/_work/gate_lasso.csv $R/_work/GATE_OK_ridge $R/_work/GATE_OK_lasso $R/_work/bindings_rl.csv logs/submitted_feature_mcs_rl.txt 2>/dev/null" | tr '\n' ' ')
LIST="$LIST $EXTRA"
mkdir -p "$R/_work/cluster_logs"
"$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && tar czf - $LIST logs/fmcs_*.out logs/submitted_feature_mcs.txt" \
  | tar xzf - --transform "s,^logs/,$R/_work/cluster_logs/,"
sums() { awk '{p = $2; sub(/^\*/, "", p); print p, $1}' | sort; }
CHK="$R/_work/models/* $R/_work/gate.csv $R/_work/bindings.csv"
for d in models_ridge models_lasso; do case "$EXTRA" in *"$d"*) CHK="$CHK $R/_work/$d/*" ;; esac; done
REMOTE_SUMS=$("$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && md5sum $CHK" | sums)
LOCAL_SUMS=$(md5sum $CHK | sums)
BAD=$(diff <(printf '%s\n' "$REMOTE_SUMS") <(printf '%s\n' "$LOCAL_SUMS") || true)
[ -z "$BAD" ] || { echo "PULL FAILED: md5 differs:"; printf '%s\n' "$BAD"; exit 1; }
echo "pull OK: $(ls $R/_work/models*/*.npz | wc -l) model files, md5 checked"
