#!/bin/bash
# I10 (2026-09-29): ship the intraday-sequence LSTM of the 15:30-16:00 bar
# (specs/causal_tune_lstm_intraday.py + _jobs.py) to a CARC deployment root of its own
# (/scratch1/jc_905/harxhar-lstmi; the other agents' campaigns ship elsewhere, so no shared
# file is re-extracted under their fleets).  The code is taken from the COMMIT (git archive
# HEAD: src/, the specs the spec runs or reads constants from, the reducers, the gates), so
# committed == shipped whatever the shared worktree holds; the Slurm scripts, task files and the
# stage-2 planner come from the worktree.  Data: the bar-keyed data/*.parquet the loader reads,
# copied on the cluster from an older root where its md5 equals the local file, uploaded otherwise.
# FAILS unless every shipped file's and every data file's remote md5 equals the local one.  Then,
# on the login node in the harxhar env (python 3.11, torch): imports, the scripts parse.
# Run from Git Bash locally:
#   bash cluster/slurm/ship_lstm_intraday_carc.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
REMOTE=/scratch1/jc_905/harxhar-lstmi
OLDROOT=/scratch1/jc_905/harxhar-dedup

mapfile -t SRC < <(git ls-tree -r --name-only HEAD src | grep -v '/__pycache__/')
COMMITTED=(
  "${SRC[@]}"
  specs/causal_tune_lstm_intraday.py
  specs/causal_tune_lstm_intraday_jobs.py
  specs/causal_tune_lstm.py
  specs/causal_tune_lstm_jobs.py
  specs/causal_tune_linear.py
  specs/causal_tune_trees.py
  specs/causal_tune_trees_tuned_jobs.py
  experiments/reduce_lstm_intraday_chunks.py
  experiments/reduce_lstm_chunks.py
  experiments/gate_lstm_intraday.py
  experiments/gate_lstm.py
)
WORKTREE=(
  cluster/slurm/lstm_intraday_pack.sbatch
  cluster/slurm/lstm_intraday_gates.sbatch
  cluster/slurm/lstm_intraday_merge.sbatch
  cluster/slurm/submit_lstm_intraday.sh
  cluster/slurm/submit_lstm_intraday_stage2.sh
  cluster/lstm_intraday_make_stage2.py
  cluster/lstm_intraday_cross_class.py
  cluster/lstm_intraday_tasks_canary.txt
  cluster/lstm_intraday_tasks_tune.txt
)
mapfile -t DATA < <(find data -maxdepth 1 -name '*.parquet' -not -name 'spxw_chain.parquet' -not -name 'spxw_spot.parquet' | sort)
sums() { awk '{p = $2; sub(/^\*/, "", p); print p, $1}' | sort; }

"$SSH" -o BatchMode=yes usc-discovery "mkdir -p $REMOTE/data $REMOTE/logs $REMOTE/results" 2>&1 | grep -v "reloaded with a version change\|python/3\|^$" || true

# 1. data: copy on the cluster where the older root's md5 matches, upload the rest
LOCAL_DATA=$(md5sum "${DATA[@]}" | sums)
OLD_DATA=$("$SSH" -o BatchMode=yes usc-discovery "cd $OLDROOT && md5sum data/*.parquet 2>/dev/null" | sums || true)
UPLOAD=()
CP=""
for f in "${DATA[@]}"; do
  l=$(printf '%s\n' "$LOCAL_DATA" | awk -v f="$f" '$1 == f {print $2}')
  o=$(printf '%s\n' "$OLD_DATA" | awk -v f="$f" '$1 == f {print $2}')
  if [ -n "$o" ] && [ "$o" = "$l" ]; then CP="$CP [ -f $REMOTE/$f ] || cp -p $OLDROOT/$f $REMOTE/$f;"; else UPLOAD+=("$f"); fi
done
[ -z "$CP" ] || "$SSH" -o BatchMode=yes usc-discovery "$CP true"
if [ "${#UPLOAD[@]}" -gt 0 ]; then
  echo "uploading ${#UPLOAD[@]} data files: ${UPLOAD[*]}"
  tar czf - "${UPLOAD[@]}" | "$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && tar xzf -"
fi

# 2. the committed code (git archive HEAD) and the I10 scripts (worktree)
git -c core.autocrlf=false -c core.eol=lf archive --format=tar HEAD "${COMMITTED[@]}" | "$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && tar xf -"
tar czf - "${WORKTREE[@]}" | "$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && tar xzf -"

# 3. md5: committed files against HEAD's blobs, the rest against the local files
LOCAL_ALL=$( { for f in "${COMMITTED[@]}"; do printf '%s %s\n' "$f" "$(git show "HEAD:$f" | md5sum | cut -d' ' -f1)"; done
               md5sum "${WORKTREE[@]}" "${DATA[@]}" | sums; } | sort)
REMOTE_ALL=$(printf '%s\n' "${COMMITTED[@]}" "${WORKTREE[@]}" "${DATA[@]}" \
  | "$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && xargs md5sum" | sums)
BAD=$(diff <(printf '%s\n' "$LOCAL_ALL") <(printf '%s\n' "$REMOTE_ALL") || true)
if [ -n "$BAD" ]; then
  echo "SHIP FAILED: remote md5 differs:"; printf '%s\n' "$BAD"; exit 1
fi
echo "md5 OK: $(printf '%s\n' "$LOCAL_ALL" | wc -l) files (${#COMMITTED[@]} from HEAD $(git rev-parse --short HEAD), ${#WORKTREE[@]} I10 scripts, ${#DATA[@]} data)"
EXTRA=$("$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && ls data/*.parquet" | grep -v -x -F -f <(printf '%s\n' "${DATA[@]}") || true)
[ -z "$EXTRA" ] || { echo "FAILED: remote data/ holds files the local set does not: $EXTRA"; exit 1; }

# 4. login node: torch, parse checks (python 3.11: catches syntax newer than the cluster's)
"$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && module load conda >/dev/null 2>&1 && source activate harxhar && set -e
PYTHONPATH=\$PWD python -c \"
import sys, torch, numpy, pandas
print('python', sys.version.split()[0], 'torch', torch.__version__, 'numpy', numpy.__version__, 'pandas', pandas.__version__)
\"
for p in specs/causal_tune_lstm_intraday.py specs/causal_tune_lstm_intraday_jobs.py experiments/reduce_lstm_intraday_chunks.py experiments/gate_lstm_intraday.py cluster/lstm_intraday_make_stage2.py; do
  python -c \"import ast, sys; ast.parse(open(sys.argv[1], encoding='utf-8').read()); print('parses:', sys.argv[1])\" \$p
done
for s in cluster/slurm/lstm_intraday_pack.sbatch cluster/slurm/lstm_intraday_gates.sbatch cluster/slurm/lstm_intraday_merge.sbatch cluster/slurm/submit_lstm_intraday.sh cluster/slurm/submit_lstm_intraday_stage2.sh; do
  bash -n \$s && echo \"bash -n ok: \$s\"
done" 2>&1 | grep -v "reloaded with a version change\|python/3"
echo "ship OK -> usc-discovery:$REMOTE"
