#!/bin/bash
# Random-forest dev-span menu study (2026-10-06): pull the run's records from the CARC root
# /scratch1/jc_905/harxhar-rf-devmenu (tar over native Windows OpenSSH; md5 of every pulled file checked
# against the remote) into results/rf_devmenu_2026-10-06/_work/: the study journal, the canary's gate /
# cost / journal / flag, the job logs (_work/logs_carc/) and the job accounting of every job in
# logs/submitted_rf_devmenu.txt (_work/sacct_rf_devmenu.txt, elapsed x allocated CPUs).  Then, locally:
#   python experiments/tree_menu_dev_rf.py freeze --tag canary --bank <harxhar-clean>/results/unification_carc
# Run from Git Bash locally:  bash cluster/slurm/pull_rf_devmenu_carc.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
REMOTE=/scratch1/jc_905/harxhar-rf-devmenu
W=results/rf_devmenu_2026-10-06/_work
q() { "$SSH" -o BatchMode=yes usc-discovery "$1" 2>/dev/null | grep -v "reloaded with a version change\|python/3\|^$" || true; }
LIST=$(q "cd $REMOTE && find $W -maxdepth 2 -type f \( -name 'tree_menu_journal_rf.log' -o -name 'gate.json' -o -name 'cost.json' -o -name 'seqgate.json' \
  -o -name 'journal_rf_canary.log' -o -name 'CANARY_OK' \) 2>/dev/null | sort")
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
"$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE/logs && tar czf - submitted_rf_devmenu* rfdm_* 2>/dev/null" 2>/dev/null | tar xzf - -C "$W/logs_carc" || true
IDS=$(grep -oE '=[0-9]+' "$W/logs_carc/submitted_rf_devmenu.txt" | tr -d = | sort -u | paste -sd, -)
q "sacct -X -P -j $IDS --format=JobID,JobName,State,Elapsed,NCPUS,Start,End,NodeList" > "$W/sacct_rf_devmenu.txt"
echo "sacct: $(($(wc -l < "$W/sacct_rf_devmenu.txt") - 1)) job rows -> $W/sacct_rf_devmenu.txt"
