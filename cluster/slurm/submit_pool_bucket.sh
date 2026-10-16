#!/bin/bash
# Pooled twins on one more bucket on CARC (2026-10-06), exactly as the moments_vix fleet of
# cluster/slurm/submit_pool_momvix.sh: epyc-7513, each arm as 20 tune-aligned chunks of 10,250 solves on
# 20 CPUs, merged.  The gate of that campaign stands (results/pool_momvix/GATE_OK, 3/3 stored slices bit
# for bit): the code shipped since differs only by new SUBGROUPS lines in src/data/loading.py (checked
# below against the definitions the user gave), and every fleet task still refuses to run without
# GATE_OK.  Task file: cluster/pool_<bucket>_tasks_fleet_carc.txt.  From the deployment root:
#
#   cd /scratch1/jc_905/harxhar-pool-momvix && bash cluster/slurm/submit_pool_bucket.sh lean_vol_vix
set -euo pipefail
cd "$(dirname "$0")/../.."
B=${1:?usage: submit_pool_bucket.sh <bucket>}
# _r2: the first submission (18:04) failed in seconds -- the conda activation of the bucket check below
# leaked into sbatch's --export=ALL environment, and the jobs' own activation then found no pandas;
# the check now runs in a subshell
STAMP=logs/submitted_pool_${B}_carc_r2.txt
[ -f "$STAMP" ] && { echo "already submitted ($(cat "$STAMP")); delete $STAMP by hand to resubmit"; exit 1; }
FLAG=results/pool_momvix/GATE_OK
[ -f "$FLAG" ] || { echo "$FLAG missing: run the moments_vix gate first"; exit 1; }
(
set +u; module load conda >/dev/null 2>&1; source activate harxhar; set -u
PYTHONPATH=$PWD python - "$B" <<'EOF'
import sys
from src.data.loading import SUBGROUPS as S
m = S["moments"]
want = {
    "moments_vix": [*m, "vix"],
    "moments_vol_vix": [*m, "sumvolume", "vix"],
    "lean_vol_vix": ["sumret", "sumabsret", "sumret4", "sumautocov", "sumvolume", "vix"],
}
for k, v in want.items():
    assert S.get(k) == v, (k, S.get(k))
print(sys.argv[1], S[sys.argv[1]])
EOF
)  # subshell: the activation must not reach sbatch --export=ALL
S="sbatch --parsable --constraint=epyc-7513"
F=$($S -J "pb_${B:0:6}" --array=1-3 --cpus-per-task=20 --mem=120G --time=3:00:00 \
      --export=ALL,MODE=fleet,TASKFILE=cluster/pool_${B}_tasks_fleet_carc.txt,REQUIRE_FLAG=$FLAG \
      cluster/slurm/pool_momvix_carc.sbatch)
K=$($S -J "pbc_${B:0:6}" --dependency=afterany:"$F" --export=ALL,MODE=check,STEP=collect,CBUCKET=$B \
      cluster/slurm/pool_momvix_carc.sbatch)
echo "$B fleet=$F collect=$K  $(date)" | tee "$STAMP"
squeue -u "$USER" -o "%.12i %.12j %.4t %.10M %.5C %R" | grep "pb" || true
