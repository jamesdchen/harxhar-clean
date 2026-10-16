#!/bin/bash
# 2026-10-06: build the CARC deployment root of the moments_vix LSTMs, /scratch1/jc_905/harxhar-momvix-lstm,
# so that it runs EXACTLY the stored runs' machinery plus the new bucket:
#   * from the canonical per-bar run's root (harxhar-mask, results/linear_subsection_lstm_mask): src/,
#     the per-bar specs, gate_lstm.py, reduce_lstm_chunks.py, lstm_mask_pack.sbatch and data/*.parquet
#     (the stored runs' data; the intraday root's data are md5-identical) -- copied ON the cluster;
#   * from the intraday run's root (harxhar-lstmi): the intraday specs, gate_lstm_intraday.py,
#     lstm_intraday_pack.sbatch / lstm_intraday_merge.sbatch and the stage-2 planner as it ran;
#   * from the local worktree (tar over native ssh): src/data/loading.py (the 2026-10-06 bucket
#     moments_vix; the diff against the stored runs' copy must be exactly that addition, checked below),
#     experiments/reduce_lstm_intraday_chunks.py (the committed reducer: the shipped one could not parse
#     the 9-field task lines, the campaign's merge ran with this one), this campaign's gate / merge /
#     submit scripts and task files, and the per-bar linear moments_vix ridge arm (16:00 results; the
#     per-bar identity gate checks the target and stamps against the linear campaign's stored arm).
# FAILS unless every copied file's md5 equals its source root's and every shipped file's equals the
# local one.  The base copy runs once (a flag); re-running the script only re-ships the local files.
# Run locally from Git Bash:  bash cluster/slurm/ship_momvix_lstm_carc.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
REMOTE=/scratch1/jc_905/harxhar-momvix-lstm
MASK=/scratch1/jc_905/harxhar-mask
LSTMI=/scratch1/jc_905/harxhar-lstmi
RIDGE_LOCAL=results/linear_subsection_dedup/moments_vix/bar1600/ridge/tw2000/causal_tune_linear/ridge/moments_vix/results_bar1600.csv
RIDGE_REMOTE=results/linear_subsection/arms_hoffman2/moments_vix/ridge/tw2000/results_bar1600.csv
FROM_MASK=(specs/causal_tune_lstm.py specs/causal_tune_lstm_jobs.py specs/causal_tune_linear.py specs/causal_tune_trees.py
           specs/causal_tune_trees_tuned_jobs.py experiments/gate_lstm.py experiments/reduce_lstm_chunks.py
           cluster/slurm/lstm_mask_pack.sbatch)
FROM_LSTMI=(specs/causal_tune_lstm_intraday.py specs/causal_tune_lstm_intraday_jobs.py experiments/gate_lstm_intraday.py
            cluster/slurm/lstm_intraday_pack.sbatch cluster/slurm/lstm_intraday_merge.sbatch cluster/lstm_intraday_make_stage2.py)
LOCAL=(src/data/loading.py experiments/reduce_lstm_intraday_chunks.py experiments/momvix_lstm_gate_compare.py
       cluster/slurm/momvix_lstm_check.sbatch cluster/slurm/momvix_lstm_merge.sbatch cluster/slurm/submit_momvix_lstm.sh
       cluster/lstmi_momvix_make_stage2.py cluster/lstm_momvix_tasks_gate.txt cluster/lstm_momvix_tasks_canary.txt
       cluster/lstm_momvix_tasks_fleet.txt cluster/lstmi_momvix_tasks_gate.txt cluster/lstmi_momvix_tasks_canary.txt
       cluster/lstmi_momvix_tasks_tune.txt)
for f in "${LOCAL[@]}" "$RIDGE_LOCAL"; do [ -f "$f" ] || { echo "missing locally: $f"; exit 1; }; done
if grep -l $'\r' "${LOCAL[@]}" >/dev/null 2>&1; then echo "CRLF in a shipped file: $(grep -l $'\r' "${LOCAL[@]}")"; exit 1; fi
sums() { awk '{p = $2; sub(/^\*/, "", p); print p, $1}' | sort; }

# 1. the base copy on the cluster (once): src/ and data/ from harxhar-mask, the named files from each root
"$SSH" -o BatchMode=yes usc-discovery "set -e; mkdir -p $REMOTE/logs $REMOTE/results $REMOTE/data $REMOTE/specs $REMOTE/experiments $REMOTE/cluster/slurm
cd $REMOTE
if [ ! -f .base_copied ]; then
  (cd $MASK && tar cf - --exclude=__pycache__ src data/*.parquet ${FROM_MASK[*]}) | tar xpf -
  (cd $LSTMI && tar cf - --exclude=__pycache__ ${FROM_LSTMI[*]}) | tar xpf -
  date > .base_copied
fi
echo base: \$(cat .base_copied)"

# 2. the local files (worktree bytes, LF) and the ridge arm at the path the identity gate reads
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/$(dirname "$RIDGE_REMOTE")"
cp -p "$RIDGE_LOCAL" "$STAGE/$RIDGE_REMOTE"
tar cf - "${LOCAL[@]}" | "$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && tar xf -"
tar cf - -C "$STAGE" "$RIDGE_REMOTE" | "$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && tar xf -"

# 3. md5: copied files against their source roots (on the cluster), local files against local
CHECK=$("$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE
bad=0
for pair in mask:$MASK lstmi:$LSTMI; do
  tag=\${pair%%:*}; src=\${pair#*:}
  if [ \$tag = mask ]; then files=\$(cd \$src && find src data -name __pycache__ -prune -o -type f \\( -name '*.py' -o -name '*.parquet' -o -name '*.json' -o -name '*.yaml' \\) -print | grep -v '^src/data/loading.py\$'; echo ${FROM_MASK[*]})
  else files='${FROM_LSTMI[*]}'; fi
  for f in \$files; do
    a=\$(md5sum < \$src/\$f | cut -c1-32); b=\$(md5sum < \$f 2>/dev/null | cut -c1-32)
    [ \"\$a\" = \"\$b\" ] || { echo \"COPY MISMATCH \$tag \$f\"; bad=1; }
  done
  echo \"copied from \$tag: \$(echo \$files | wc -w) files checked\"
done
echo \"--- loading.py vs the stored runs' copy\"
diff $MASK/src/data/loading.py src/data/loading.py || true
echo \"--- remote md5\"
md5sum ${LOCAL[*]} $RIDGE_REMOTE
echo \"copy check bad=\$bad\"" || true)
printf '%s\n' "$CHECK" | grep -v '^[0-9a-f]\{32\} ' || true
printf '%s\n' "$CHECK" | grep 'COPY MISMATCH' && { echo "SHIP FAILED: a copied file differs from its source root"; exit 1; }
REMOTE_MD5=$(printf '%s\n' "$CHECK" | grep '^[0-9a-f]\{32\} ' | sums)
LOCAL_MD5=$( { md5sum "${LOCAL[@]}"; printf '%s  %s\n' "$(md5sum < "$RIDGE_LOCAL" | cut -c1-32)" "$RIDGE_REMOTE"; } | sums)
BAD=$(diff <(printf '%s\n' "$LOCAL_MD5") <(printf '%s\n' "$REMOTE_MD5") || true)
[ -z "$BAD" ] || { echo "SHIP FAILED: remote md5 differs from local:"; printf '%s\n' "$BAD"; exit 1; }
# the loading.py diff must be pure additions (the moments_vix block)
DIFF=$(printf '%s\n' "$CHECK" | sed -n '/^--- loading.py/,/^--- remote md5/p' | sed '1d;$d')
if printf '%s\n' "$DIFF" | grep -q '^<'; then echo "SHIP FAILED: loading.py removes or changes a line of the stored runs' copy"; exit 1; fi
printf '%s\n' "$DIFF" | grep -q 'SUBGROUPS\["moments_vix"\]' || { echo "SHIP FAILED: the moments_vix bucket is not in the shipped loading.py"; exit 1; }
echo "md5 OK: ${#LOCAL[@]} local files + the ridge arm; loading.py = stored copy + $(printf '%s\n' "$DIFF" | grep -c '^>') added lines"

# 4. login node: the bucket, the libraries, parse checks
"$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && module load conda >/dev/null 2>&1 && source activate harxhar && set -e
PYTHONPATH=\$PWD python -c \"
import sys, torch, numpy, pandas
from src.data.loading import get_bucket
print('python', sys.version.split()[0], 'torch', torch.__version__, 'numpy', numpy.__version__, 'pandas', pandas.__version__)
print('moments_vix =', get_bucket('moments_vix'))
\"
for p in specs/*.py experiments/*.py cluster/*.py; do python -c \"import ast, sys; ast.parse(open(sys.argv[1], encoding='utf-8').read())\" \$p; done
for s in cluster/slurm/*.sbatch cluster/slurm/*.sh; do bash -n \$s || { echo \"bash -n FAILED: \$s\"; exit 1; }; done
echo 'remote checks ok'" 2>&1 | grep -v "reloaded with a version change\|python/3\|^$" || true
echo "ship OK -> usc-discovery:$REMOTE"
