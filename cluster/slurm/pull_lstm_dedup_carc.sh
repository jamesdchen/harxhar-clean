#!/bin/bash
# I1 (2026-09-29): pull the merged de-duplicated LSTM arms from the CARC deployment root into the
# local results/linear_subsection_lstm_dedup/ (the merged unchunked arm directories, the merge
# log and gates, the flags), plus the job logs into results/linear_subsection_lstm_dedup/logs/.
# The per-chunk directories stay on the cluster except their run logs (small).
# Run from Git Bash locally:
#   bash cluster/slurm/pull_lstm_dedup_carc.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
REMOTE=/scratch1/jc_905/harxhar-dedup
R=results/linear_subsection_lstm_dedup
mkdir -p "$R/logs"
"$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && tar czf - \$(find $R -path '*/chunks' -prune -o -type f -print) \$(find $R -path '*/chunks/*' -name run.log)" 2>/dev/null | tar xzf -
"$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE/logs && tar czf - lstm_dedup_* submitted_lstm_dedup_carc.txt" 2>/dev/null | tar xzf - -C "$R/logs"
echo "merged arms: $(find $R -name MERGED | wc -l); $(cat $R/MERGED_ALL 2>/dev/null || echo 'no MERGED_ALL yet')"
grep -h "repeat check" "$R"/logs/lstm_dedup_canary.*.out 2>/dev/null || true
