#!/bin/bash
# Ship the free_vix_only bucket and its staging to the CARC root (tar over
# native Windows OpenSSH), check the files landed and that the bucket resolves
# to its 13 columns there.  Run from Git Bash locally:
#   bash cluster/slurm/ship_freevix_carc.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
REMOTE=/scratch1/jc_905/harxhar-subsection
FILES=(
  src/data/loading.py
  cluster/slurm/subsection_score_freevix.sbatch
  cluster/slurm/submit_freevix.sh
  cluster/freevix_tasks_canary.txt
  cluster/freevix_tasks_ridge_enet.txt
  cluster/freevix_tasks_lasso.txt
)
echo "local:";  md5sum "${FILES[@]}" | cut -c1-12,34-
tar czf - "${FILES[@]}" | "$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && tar xzf - && echo remote: && md5sum ${FILES[*]} | cut -c1-12,34-"
"$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && module load conda >/dev/null 2>&1 && source activate harxhar && PYTHONPATH=\$PWD python - <<'EOF'
from src.data.loading import SUBGROUPS
b = SUBGROUPS['free_vix_only']
print('free_vix_only', len(b), b)
assert len(b) == 13, len(b)
assert not any(c in ('numobs', 'vvix', 'vix3m') for c in b)
assert 'sumvolume' in b and 'vix' in b
EOF"
