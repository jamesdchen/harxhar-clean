#!/bin/bash
# Pull the linear bucket designs (2026-10-12/13: addb_* / loob_* for ridge, lasso and elastic net) from a CARC
# root (REMOTE, default the fresh root of 2026-10-13: /scratch1/jc_905/harxhar-feature-mcs2, used because some
# files of the first root hang on read), md5-checked: the 16 bucket npz / json of each estimator, the re-run
# anchors' gate records (gate*.csv, GATE_OK*), the binding record and the job log.  The anchors' npz are NOT
# pulled (the local copies are the ones the MCS used; the job's gate shows the re-run equals the stored arm f).
#
#   [REMOTE=/scratch1/jc_905/<root>] bash cluster/slurm/pull_feature_mcs_buckets_carc.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
REMOTE=${REMOTE:-/scratch1/jc_905/harxhar-feature-mcs2}
R=results/feature_mcs_2026-10-08
mapfile -t FILES < <(timeout 120 "$SSH" -o BatchMode=yes usc-discovery \
  "cd $REMOTE && ls $R/_work/models/addb_* $R/_work/models/loob_* $R/_work/models_ridge/addb_* $R/_work/models_ridge/loob_* $R/_work/models_lasso/addb_* $R/_work/models_lasso/loob_* 2>/dev/null")
[ "${#FILES[@]}" -eq 96 ] || { echo "expected 96 bucket files (3 x 16 x npz + json), found ${#FILES[@]}"; exit 1; }
mkdir -p "$R/_work/buckets_run"
timeout 600 "$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && tar czf - ${FILES[*]}" | tar xzf -
timeout 300 "$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && tar czf - $R/_work/gate.csv $R/_work/gate_ridge.csv $R/_work/gate_lasso.csv $R/_work/GATE_OK $R/_work/GATE_OK_ridge $R/_work/GATE_OK_lasso $R/_work/bindings_bk.csv logs/fmcs_bk.*.out logs/submitted_feature_mcs_buckets.txt" \
  | tar xzf - --transform "s,^$R/_work/,$R/_work/buckets_run/," --transform "s,^logs/,$R/_work/buckets_run/,"
sums() { awk '{p = $2; sub(/^\*/, "", p); print p, $1}' | sort; }
REMOTE_SUMS=$(timeout 300 "$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && md5sum ${FILES[*]}" | sums)
LOCAL_SUMS=$(md5sum "${FILES[@]}" | sums)
BAD=$(diff <(printf '%s\n' "$REMOTE_SUMS") <(printf '%s\n' "$LOCAL_SUMS") || true)
[ -z "$BAD" ] || { echo "PULL FAILED: md5 differs:"; printf '%s\n' "$BAD" | head; exit 1; }
echo "pull OK: ${#FILES[@]} bucket files from $REMOTE, md5 checked; gate records in $R/_work/buckets_run/"
