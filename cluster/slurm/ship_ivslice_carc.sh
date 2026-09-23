#!/bin/bash
# Ship the ATM-slice panel, the new buckets and the staging to the CARC root
# (tar over native Windows OpenSSH), then check the files landed and that the
# panel loads there with the new columns.  Run from Git Bash locally:
#   bash cluster/slurm/ship_ivslice_carc.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
REMOTE=/scratch1/jc_905/harxhar-subsection
FILES=(
  src/data/loading.py
  cluster/slurm/subsection_score_ivslice.sbatch
  cluster/slurm/submit_ivslice.sh
  cluster/ivslice_tasks_canary.txt
  cluster/ivslice_tasks_ridge_enet.txt
  cluster/ivslice_tasks_lasso.txt
  data/spxw_ivslice.parquet
  experiments/build_spxw_ivslice_panel.py
)
echo "local:";  md5sum "${FILES[@]}" | cut -c1-12,34-
tar czf - "${FILES[@]}" | "$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && tar xzf - && echo remote: && md5sum ${FILES[*]} | cut -c1-12,34-"
"$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && module load conda >/dev/null 2>&1 && source activate harxhar && PYTHONPATH=\$PWD python - <<'EOF'
from src.data.loading import SUBGROUPS, load_raw_data
for b in ('live_feasible', 'ivslice_only', 'live_feasible_ivslice', 'live_feasible_plus_ivslice'):
    print(b, len(SUBGROUPS[b]))
df = load_raw_data('data', allow_missing=True)
m = df[['t', 'ivslice_annvol', 'ivslice_perbar_rv', 'ivslice_vvix']]
ok = m['ivslice_annvol'].notna()
print('panel rows', len(df), 'first', df['t'].min(), 'last', df['t'].max())
print('ivslice rows', int(ok.sum()), 'first', m.loc[ok, 't'].min(), 'last', m.loc[ok, 't'].max())
print(m[ok].tail(3).to_string(index=False))
EOF"
