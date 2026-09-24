#!/bin/bash
# Ship the free-feed bucket and its staging to the CARC root (tar over native
# Windows OpenSSH), check the files landed and that the bucket resolves to its
# 14 columns there.  Run from Git Bash locally:
#   bash cluster/slurm/ship_free_carc.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
REMOTE=/scratch1/jc_905/harxhar-subsection
FILES=(
  src/data/loading.py
  cluster/slurm/subsection_score_free.sbatch
  cluster/slurm/submit_free.sh
  cluster/free_tasks_canary.txt
  cluster/free_tasks_ridge_enet.txt
  cluster/free_tasks_lasso.txt
)
echo "local:";  md5sum "${FILES[@]}" | cut -c1-12,34-
tar czf - "${FILES[@]}" | "$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && tar xzf - && echo remote: && md5sum ${FILES[*]} | cut -c1-12,34-"
"$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && module load conda >/dev/null 2>&1 && source activate harxhar && PYTHONPATH=\$PWD python - <<'EOF'
from src.data.loading import SUBGROUPS
b = SUBGROUPS['free_feasible']
print('free_feasible', len(b), b)
assert len(b) == 14, len(b)
assert not any(c in ('sumvolume', 'numobs') for c in b)
EOF"
