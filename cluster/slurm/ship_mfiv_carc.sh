#!/bin/bash
# Ship the MFIV panel, the new buckets, the HAR-ladder axis and the staging to the
# CARC root (tar over native Windows OpenSSH), then check the files landed and
# that the panel loads there with the new columns.  Run from Git Bash locally:
#   bash cluster/slurm/ship_mfiv_carc.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
REMOTE=/scratch1/jc_905/harxhar-subsection
FILES=(
  src/data/loading.py
  specs/causal_tune_linear.py
  cluster/slurm/subsection_pack.sbatch
  cluster/slurm/subsection_score_mfiv.sbatch
  cluster/slurm/submit_mfiv_har.sh
  cluster/mfiv_tasks_canary.txt
  cluster/mfiv_tasks_ridge_enet.txt
  cluster/mfiv_tasks_lasso.txt
  cluster/harlag_tasks.txt
  data/spxw_mfiv.parquet
  experiments/build_spxw_mfiv_panel.py
)
echo "local:";  md5sum "${FILES[@]}" | cut -c1-12,34-
tar czf - "${FILES[@]}" | "$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && tar xzf - && echo remote: && md5sum ${FILES[*]} | cut -c1-12,34-"
"$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && module load conda >/dev/null 2>&1 && source activate harxhar && PYTHONPATH=\$PWD python - <<'EOF'
from src.data.loading import SUBGROUPS, load_raw_data
from src.features.extractors.har import resolve_har_lags
for b in ('live_feasible', 'mfiv_only', 'live_feasible_mfiv', 'live_feasible_plus_mfiv'):
    print(b, len(SUBGROUPS[b]))
print('base2', resolve_har_lags(base=2)); print('base3', resolve_har_lags(base=3))
df = load_raw_data('data', allow_missing=True)
m = df[['t', 'mfiv0_annvol', 'mfiv0_perbar_rv', 'mfiv0_vvix']]
ok = m['mfiv0_annvol'].notna()
print('panel rows', len(df), 'first', df['t'].min(), 'last', df['t'].max())
print('mfiv rows', int(ok.sum()), 'first', m.loc[ok, 't'].min(), 'last', m.loc[ok, 't'].max())
print(m[ok].tail(3).to_string(index=False))
EOF"
