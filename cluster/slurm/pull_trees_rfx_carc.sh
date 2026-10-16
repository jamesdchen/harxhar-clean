#!/bin/bash
# rfx (2026-10-07, TOKEN-driven): pull the MERGED fixed / tuned / unmasked arms of token TOKEN (copy of the rfdev pull)
# rfdev (2026-10-06): pull the MERGED dev-tuned random-forest arms (T10, T1; copy of the moments_vix pull)
# moments_vix tree workstream (2026-10-06): pull the MERGED moments_vix arms of the masked 16:00 rungs
# (T10 / T1 / RS10 / RS1) from the CARC deployment root /scratch1/jc_905/harxhar-$TOKEN-trees, the twin
# of cluster/slurm/pull_trees_mask_carc.sh (tar over native Windows OpenSSH, every pulled file's md5
# checked against the remote).  Pulled (CSVs and small JSON / flags; no npz, no chunk folders):
#   results/linear_subsection_trees_mask_momvix/{t10,t1}/moments_vix/bar1600/<model>/tw2000/causal_tune_trees/...
#   results/linear_subsection_trees_tuned_mask_momvix/{rs10,rs1}/moments_vix/bar1600/<model>/tw2000/causal_tune_trees/...
#   the reducers' reduce_gates.csv / reduce.log, MERGE_SUMMARY / MERGE_DONE, CANARY_OK, the kept-column
#   counts, the class census, the gate outputs (CSV / flags; not their run folders), the job logs
# and writes the sacct dump of every job in logs/submitted_trees_${TOKEN}_carc.txt to
# results/trees_momvix_1600/sacct_carc.txt (+ sacct_steps_carc.txt).
# Run from Git Bash locally:  bash cluster/slurm/pull_trees_momvix_carc.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
TOKEN=${TOKEN:?TOKEN}
REMOTE=/scratch1/jc_905/harxhar-$TOKEN-trees
U=results/linear_subsection_trees_mask_$TOKEN
T=results/linear_subsection_trees_tuned_mask_$TOKEN
N=results/linear_subsection_trees_nomask_$TOKEN
G=results/trees_${TOKEN}_gates
O=results/trees_${TOKEN}_1600
q() { "$SSH" -o BatchMode=yes usc-discovery "$1" 2>/dev/null | grep -v "reloaded with a version change\|python/3\|^$" || true; }
LIST=$(q "cd $REMOTE && { find $U/t10 $U/t1 $T/rs10 $T/rs1 $N/t10 $N/t1 -path '*/chunks' -prune -o -type f \( -path '*/causal_tune_*' \( -name '*.csv' -o -name '*.json' -o -name MERGED \) \) -print 2>/dev/null;
  find $U $T $N -maxdepth 2 -type f \( -name reduce_gates.csv -o -name reduce.log -o -name 'MERGE_SUMMARY*' -o -name CANARY_OK -o -name 'MERGE_DONE*' -o -name kept.log -o -name class_census.csv \) -print 2>/dev/null;
  find $U/kept -type f -name '*.csv' -print 2>/dev/null;
  find $G -maxdepth 2 -type f \( -name '*.csv' -o -name GATE_OK -o -name 'PASS_*' -o -name NODE -o -name H2_HOST \) -print 2>/dev/null; } | sort")
NF=$(printf '%s\n' "$LIST" | grep -c . || true)
echo "pulling $NF files"
[ "$NF" -gt 0 ] || exit 1
printf '%s\n' $LIST | "$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && tar czf - -T -" 2>/dev/null | tar xzf -
REMOTE_SUMS=$(printf '%s\n' $LIST | "$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && xargs md5sum" 2>/dev/null | awk '{p = $2; sub(/^\*/, "", p); print p, $1}')
LOCAL_SUMS=$(md5sum $LIST | awk '{p = $2; sub(/^\*/, "", p); print p, $1}')
BAD=$(awk 'NR == FNR {r[$1] = $2; next} !($1 in r) || r[$1] != $2 {print $1}' <(printf '%s\n' "$REMOTE_SUMS") <(printf '%s\n' "$LOCAL_SUMS"))
if [ -n "$BAD" ]; then
  echo "PULL FAILED: md5 differs for:"; printf '  %s\n' $BAD; exit 1
fi
mkdir -p "$O/logs_carc"
"$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE/logs && tar czf - submitted_trees_${TOKEN}* ${TOKEN:0:4}_* 2>/dev/null" 2>/dev/null | tar xzf - -C "$O/logs_carc" || true
IDS=$(grep -oE '[a-z0-9_-]+=[0-9]+' "$O/logs_carc/submitted_trees_${TOKEN}_carc.txt" | cut -d= -f2 | sort -u | paste -sd, -)
q "sacct -X -P -n -j $IDS -o JobID,JobName,AllocCPUS,ElapsedRaw,Start,End,State,CPUTimeRAW,NodeList" > "$O/sacct_carc.txt"
q "sacct -P -n -j $IDS -o JobID,TotalCPU,MaxRSS" | grep '\.batch|' > "$O/sacct_steps_carc.txt" || true
echo "pull OK: $NF files, md5 identical to the remote; sacct: $(wc -l < "$O/sacct_carc.txt") allocations of jobs $IDS"
