#!/bin/bash
# Ship the realized vol-of-VIX panel, the four VIX-only buckets and the staging
# to the CARC root (tar over native Windows OpenSSH), then check the files
# landed, the buckets have the expected column counts and the panel loads there
# with the two new columns.  Run from Git Bash locally:
#   bash cluster/slurm/ship_vixonly_carc.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
REMOTE=/scratch1/jc_905/harxhar-subsection
FILES=(
  src/data/loading.py
  cluster/slurm/subsection_score_vixonly.sbatch
  cluster/slurm/submit_vixonly.sh
  cluster/vixonly_tasks_canary.txt
  cluster/vixonly_tasks_ridge_enet.txt
  cluster/vixonly_tasks_lasso.txt
  data/vix_rvol.parquet
  experiments/build_vix_rvol.py
)
echo "local:";  md5sum "${FILES[@]}" | cut -c1-12,34-
tar czf - "${FILES[@]}" | "$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && tar xzf - && echo remote: && md5sum ${FILES[*]} | cut -c1-12,34-"
"$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && module load conda >/dev/null 2>&1 && source activate harxhar && PYTHONPATH=\$PWD python - <<'EOF'
from src.data.loading import SUBGROUPS, load_raw_data
want = {'vix_only': 1, 'vix_rvol': 3, 'live_vix_only': 14, 'live_vix_rvol': 16, 'free_feasible_vol': 15}
for b, n in want.items():
    print(b, len(SUBGROUPS[b]), SUBGROUPS[b][-3:])
    assert len(SUBGROUPS[b]) == n, (b, len(SUBGROUPS[b]), n)
df = load_raw_data('data', allow_missing=True)
cols = ['vix_volofvol_5d', 'vix_volofvol_22d']
missing = [c for c in cols if c not in df.columns]
assert not missing, missing
print('panel rows', len(df), 'first', df['t'].min(), 'last', df['t'].max())
for c in cols:
    ok = df[c].notna()
    print(f'{c:14s} valid {int(ok.sum()):7d}  {df.loc[ok, \"t\"].min()} .. {df.loc[ok, \"t\"].max()}')
EOF"
