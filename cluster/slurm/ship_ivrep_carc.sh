#!/bin/bash
# Ship the VIX-representation panel, the buckets and the staging to the CARC root
# (tar over native Windows OpenSSH), then check the files landed and that the
# panel loads there with the new columns and the six buckets have the expected
# column counts.  Run from Git Bash locally:
#   bash cluster/slurm/ship_ivrep_carc.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
REMOTE=/scratch1/jc_905/harxhar-subsection
FILES=(
  src/data/loading.py
  cluster/slurm/subsection_score_ivrep.sbatch
  cluster/slurm/submit_ivrep.sh
  cluster/ivrep_tasks_canary.txt
  cluster/ivrep_tasks_ridge_enet.txt
  cluster/ivrep_tasks_lasso.txt
  data/vix_representations.parquet
  experiments/build_vix_representations.py
)
echo "local:";  md5sum "${FILES[@]}" | cut -c1-12,34-
tar czf - "${FILES[@]}" | "$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && tar xzf - && echo remote: && md5sum ${FILES[*]} | cut -c1-12,34-"
"$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && module load conda >/dev/null 2>&1 && source activate harxhar && PYTHONPATH=\$PWD python - <<'EOF'
from src.data.loading import SUBGROUPS, load_raw_data
want = {'ivrep_target_scale': 16, 'ivrep_term': 16, 'ivrep_innovations': 19, 'ivrep_vrp': 16, 'ivrep_slice_slope': 17, 'ivrep_all': 25}
for b, n in want.items():
    print(b, len(SUBGROUPS[b]))
    assert len(SUBGROUPS[b]) == n, (b, len(SUBGROUPS[b]), n)
df = load_raw_data('data', allow_missing=True)
cols = ['impl30_perbar_rv', 'vix_slope_3m', 'vvix_over_vix', 'vix_chg_1bar', 'vix_chg_1d', 'vix_chg_5d', 'vix_vrp_1d', 'vix_vrp_5d', 'vix_vrp_22d', 'ivslice_over_vix']
missing = [c for c in cols if c not in df.columns]
assert not missing, missing
print('panel rows', len(df), 'first', df['t'].min(), 'last', df['t'].max())
for c in cols:
    ok = df[c].notna()
    print(f'{c:18s} valid {int(ok.sum()):7d}  {df.loc[ok, \"t\"].min()} .. {df.loc[ok, \"t\"].max()}')
EOF"
