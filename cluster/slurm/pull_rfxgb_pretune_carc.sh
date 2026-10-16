#!/bin/bash
# Random forest / XGBoost pre-tune study (2026-10-06): pull the run's records from the CARC root
# /scratch1/jc_905/harxhar-rfxgb-pretune (tar over native Windows OpenSSH; every pulled file's md5
# checked against the remote), then the job accounting.  Pulled under results/trees_pretune_2026-10-06/
# _work/ (gitignored): for every cell of tag TAG (default carc) and of the canary: the pre-tune trial
# records (pretune_s*.npz, resume logs, study logs), plan.json, the retune / pick CSVs, the refit parts,
# the flags and the gate CSVs (not the Optuna journals); the job logs to _work/logs_carc/; and
# results/trees_pretune_2026-10-06/sacct.csv (every job of logs/submitted_rfxgb_pretune.txt:
# elapsed x allocated CPUs).  Then, locally:
#   for c in <cells>; do RXP_TAG=carc RXP_MODEL=.. RXP_BUCKET=.. python experiments/close_trees_pretune_rfxgb.py assemble; done
#   RXP_TAG=carc python experiments/close_trees_pretune_rfxgb.py report
# Run from Git Bash locally:  bash cluster/slurm/pull_rfxgb_pretune_carc.sh [tag]
set -euo pipefail
cd "$(dirname "$0")/../.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
REMOTE=/scratch1/jc_905/harxhar-rfxgb-pretune
TAG=${1:-carc}
W=results/trees_pretune_2026-10-06/_work
q() { "$SSH" -o BatchMode=yes usc-discovery "$1" 2>/dev/null | grep -v "reloaded with a version change\|python/3\|^$" || true; }
LIST=$(q "cd $REMOTE && find $W/$TAG $W/canary -type f \( -name 'pretune_s*.npz' -o -name 'pretune_s*_resumes.json' -o -name 'pretune_s*.log' \
  -o -name plan.json -o -name '*.csv' -o -name '*_DONE*' -o -name 'CANARY_OK_*' -o -name 'GATE_OK_*' -o -name gate.log \
  -o -path '*/refits/*_p*.npz' \) ! -name '*.tmp.*' 2>/dev/null | sort")
N=$(printf '%s\n' "$LIST" | grep -c . || true)
echo "pulling $N files"
[ "$N" -gt 0 ] || exit 1
printf '%s\n' $LIST | "$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && tar czf - -T -" 2>/dev/null | tar xzf -
REMOTE_SUMS=$(printf '%s\n' $LIST | "$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && xargs md5sum" 2>/dev/null | awk '{p = $2; sub(/^\*/, "", p); print p, $1}')
LOCAL_SUMS=$(printf '%s\n' $LIST | xargs md5sum | awk '{p = $2; sub(/^\*/, "", p); print p, $1}')
BAD=$(awk 'NR == FNR {r[$1] = $2; next} !($1 in r) || r[$1] != $2 {print $1}' <(printf '%s\n' "$REMOTE_SUMS") <(printf '%s\n' "$LOCAL_SUMS"))
if [ -n "$BAD" ]; then
  echo "PULL FAILED: md5 differs for:"; printf '  %s\n' $BAD; exit 1
fi
echo "md5 OK: $N files"
mkdir -p "$W/logs_carc"
"$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE/logs && tar czf - submitted_rfxgb_pretune* rxp_* 2>/dev/null" 2>/dev/null | tar xzf - -C "$W/logs_carc" || true
IDS=$(grep -oE '=[0-9]+' "$W/logs_carc/submitted_rfxgb_pretune.txt" | tr -d = | sort -u | paste -sd, -)
q "sacct -X -P -n -j $IDS -o JobID,JobName,AllocCPUS,ElapsedRaw,Start,End,State,CPUTimeRAW,NodeList" > "$W/logs_carc/sacct.txt"
awk -F'|' 'BEGIN {print "job_id,job_name,alloc_cpus,elapsed_s,start,end,state,cpu_hours,nodes"}
  {printf "%s,%s,%s,%s,%s,%s,%s,%.4f,%s\n", $1, $2, $3, $4, $5, $6, $7, $8 / 3600, $9}' "$W/logs_carc/sacct.txt" \
  > results/trees_pretune_2026-10-06/sacct.csv
echo "sacct: $(($(wc -l < results/trees_pretune_2026-10-06/sacct.csv) - 1)) job rows"
