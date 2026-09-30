#!/bin/bash
# I10 (2026-09-29): pull the intraday-sequence LSTM from the CARC deployment root into the local
# results/linear_subsection_lstm_intraday/: the merged arm directories, the gates' CSVs, the stage-2
# plan, reduce log / gates, the flags, every chunk's run.log (small), and the job logs + sacct usage
# into logs/.  The tuning cache and the per-chunk result directories stay on the cluster.
#   bash cluster/slurm/pull_lstm_intraday_carc.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
REMOTE=/scratch1/jc_905/harxhar-lstmi
R=results/linear_subsection_lstm_intraday
mkdir -p "$R/logs"
"$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && tar czf - \$(find $R -path '*/chunks' -prune -o -path '*/stage_tune' -prune -o -path '*/tune_cache' -prune -o -path '*/gates/scratch*' -prune -o -path '*mixedclass*' -prune -o -type f -print) \$(find $R -path '*/chunks/*' -name run.log) \$(find $R -path '*/stage_tune/*' -name run.log) cluster/lstm_intraday_tasks_fleet.txt 2>/dev/null" | tar xzf -
"$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE/logs && tar czf - lstmi_* submitted_lstm_intraday*_carc.txt* 2>/dev/null" | tar xzf - -C "$R/logs"
"$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && ids=\$(cat logs/submitted_lstm_intraday*_carc.txt* | grep -o '[0-9]\{8\}' | sort -u | paste -sd,) && sacct -X -P -n -j \$ids -o JobID,JobName,AllocCPUS,ElapsedRaw,Start,End,State,CPUTimeRAW,TotalCPU,Partition && echo '#STEPS' && sacct -P -n -j \$ids -o JobID,TotalCPU,MaxRSS | grep '\.batch'" > "$R/logs/sacct_raw.txt" 2>/dev/null || true
echo "merged arms: $(find $R -name MERGED | wc -l); $(cat $R/MERGED_ALL 2>/dev/null || echo 'no MERGED_ALL yet')"
