#!/bin/bash
# I9: pull the MERGED masked arms of the 16:00 re-run (T10 / T1 / RS10 / RS1 / LSTM, WINDOW_MASK=1)
# from the CARC deployment root /scratch1/jc_905/harxhar-mask (tar over native Windows OpenSSH)
# and check every pulled file's md5 against the remote.  Pulled (CSVs and small JSON / flags;
# no npz, no chunk folders):
#   results/linear_subsection_trees_mask/{t10,t1}/<bucket>/bar1600/<model>/tw2000/causal_tune_trees/...
#   results/linear_subsection_trees_tuned_mask/{rs10,rs1}/<bucket>/bar1600/<model>/tw2000/causal_tune_trees/...
#   results/linear_subsection_lstm_mask/<bucket>/bar1600/lstm/tw2000/causal_tune_lstm/...
#   the reducers' reduce_gates.csv / reduce.log, MERGE_SUMMARY.txt, the flags, the kept-column
#   counts (results/linear_subsection_trees_mask/kept/), the gate outputs (CSV / log / flags;
#   not their scratch runs), the job logs of this campaign
# and writes the campaign's sacct dump (every job id in logs/submitted_trees_mask_carc.txt) to
# results/trees_mask_1600/sacct.txt and sacct_steps.txt (the ladder's cluster-use table).
# Run from Git Bash locally:  bash cluster/slurm/pull_trees_mask_carc.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
REMOTE=/scratch1/jc_905/harxhar-mask
U=results/linear_subsection_trees_mask
T=results/linear_subsection_trees_tuned_mask
L=results/linear_subsection_lstm_mask
L10=results/linear_subsection_lstm_mask_re10
X=results/linear_subsection_trees_mask_xeon
q() { "$SSH" -o BatchMode=yes usc-discovery "$1" 2>/dev/null | grep -v "reloaded with a version change\|python/3\|^$" || true; }
LIST=$(q "cd $REMOTE && { find $U/t10 $U/t1 $T/rs10 $T/rs1 $L $L10 $X/t10 $X/t1 -path '*/chunks' -prune -o -type f \( -path '*/causal_tune_*' \( -name '*.csv' -o -name '*.json' -o -name MERGED \) \) -print 2>/dev/null;
  find $U $T $L $L10 $X -maxdepth 2 -type f \( -name reduce_gates.csv -o -name reduce.log -o -name 'MERGE_SUMMARY*' -o -name CANARY_OK -o -name 'MERGE_DONE*' -o -name kept.log -o -name class.log \) -print 2>/dev/null;
  find $U/kept $U/class -type f -name '*.csv' -print 2>/dev/null;
  find $U/gates -path '*/scratch' -prune -o -type f \( -name '*.csv' -o -name '*.log' -o -name GATES_OK -o -name 'FINISHED_*' \) -print 2>/dev/null; } | sort")
N=$(printf '%s\n' "$LIST" | grep -c . || true)
echo "pulling $N files"
[ "$N" -gt 0 ] || exit 1
# the file list goes over stdin (a Windows command line holds at most 32767 characters)
printf '%s
' $LIST | "$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && tar czf - -T -" 2>/dev/null | tar xzf -
REMOTE_SUMS=$(printf '%s
' $LIST | "$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && xargs md5sum" 2>/dev/null | awk '{p = $2; sub(/^\*/, "", p); print p, $1}')
LOCAL_SUMS=$(md5sum $LIST | awk '{p = $2; sub(/^\*/, "", p); print p, $1}')
BAD=$(awk 'NR == FNR {r[$1] = $2; next} !($1 in r) || r[$1] != $2 {print $1}' <(printf '%s\n' "$REMOTE_SUMS") <(printf '%s\n' "$LOCAL_SUMS"))
if [ -n "$BAD" ]; then
  echo "PULL FAILED: md5 differs for:"; printf '  %s\n' $BAD; exit 1
fi
mkdir -p results/trees_mask_1600/logs
"$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE/logs && tar czf - submitted_trees_mask_carc.txt* tm_* trees_mask* treestuned_mask* lstm_mask* 2>/dev/null" 2>/dev/null \
  | tar xzf - -C results/trees_mask_1600/logs || true
IDS=$(grep -oE '[a-z0-9_]+=[0-9]+' results/trees_mask_1600/logs/submitted_trees_mask_carc.txt | cut -d= -f2 | sort -u | paste -sd, -)
q "sacct -X -P -n -j $IDS -o JobID,JobName,AllocCPUS,ElapsedRaw,Start,End,State,CPUTimeRAW,TotalCPU" > results/trees_mask_1600/sacct.txt
q "sacct -P -n -j $IDS -o JobID,TotalCPU,MaxRSS" | grep '\.batch|' > results/trees_mask_1600/sacct_steps.txt || true
echo "pull OK: $N files, md5 identical to the remote; sacct: $(wc -l < results/trees_mask_1600/sacct.txt) allocations of jobs $IDS"
