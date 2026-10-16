#!/bin/bash
# 2026-10-06: pull the moments_vix LSTMs from the CARC deployment root /scratch1/jc_905/harxhar-momvix-lstm
# into NEW local roots (the canonical roots' flags / reduce logs would collide, so no bucket sub-directory):
#   PART=perbar    results/linear_subsection_lstm_mask_momvix/      (merged arm, chunk run.log / NODE / npz,
#                  reduce files, flags) + gates/ (the reproduction-gate compare.csv, the identity gate)
#   PART=intraday  results/linear_subsection_lstm_intraday_momvix/  (merged arm, chunk run.log / NODE,
#                  stage_tune run.log, stage-2 plan, reduce files, flags, gate set A) + gates/ (compare.csv)
#   PART=refs      the stored chunks the reproduction gates compared against (and the gate runs), for the
#                  Hoffman2 class check: $REFS/{stored_perbar,stored_intraday,carc_perbar,carc_intraday}
# plus logs/ (job logs, submission stamps) and sacct usage (logs/sacct_raw.txt) in each root.
#   PART=perbar|intraday|refs [REFS=<dir>] bash cluster/slurm/pull_momvix_lstm_carc.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
REMOTE=/scratch1/jc_905/harxhar-momvix-lstm
P="${PART:?PART=perbar|intraday|refs}"
q() { grep -v "reloaded with a version change\|python/3\|^$" || true; }
case "$P" in
  perbar)
    L=results/linear_subsection_lstm_mask_momvix; G=results/momvix_gates/perbar; J=mvl_gate_pb
    mkdir -p "$L/gates" "$L/logs"
    "$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && tar czf - \$(find $L -path '*_repeat' -prune -o -type f -print) $G/compare.csv $G/identity 2>/dev/null" | tar xzf - --transform "s|^$G|$L/gates/reproduction|"
    ;;
  intraday)
    L=results/linear_subsection_lstm_intraday_momvix; G=results/momvix_gates/intraday; J=mvl_gate_i
    mkdir -p "$L/gates" "$L/logs"
    "$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && tar czf - \$(find $L -path '*/chunks' -prune -o -path '*/stage_tune' -prune -o -path '*/tune_cache' -prune -o -path '*/gates/scratch*' -prune -o -type f -print) \$(find $L -path '*/chunks/*' \\( -name run.log -o -name NODE \\)) \$(find $L -path '*/stage_tune/*' \\( -name run.log -o -name NODE \\)) $G/compare.csv cluster/lstmi_momvix_tasks_fleet.txt 2>/dev/null" | tar xzf - --transform "s|^$G|$L/gates/reproduction|"
    ;;
  refs)
    R="${REFS:?REFS=<local dir>}"
    mkdir -p "$R"
    "$SSH" -o BatchMode=yes usc-discovery "set -e
M=/scratch1/jc_905/harxhar-mask/results/linear_subsection_lstm_mask; I=/scratch1/jc_905/harxhar-lstmi/results/linear_subsection_lstm_intraday
T=\$(mktemp -d); mkdir -p \$T/stored_perbar \$T/stored_intraday/tune_cache/baseline \$T/stored_intraday/tune_cache/all_features
for x in baseline:1 all_features:4; do b=\${x%%:*}; c=\${x#*:}; d=\$b/bar1600/lstm/tw2000/chunks/c\$c; mkdir -p \$T/stored_perbar/\$d; cp -rp \$M/\$d/causal_tune_lstm \$M/\$d/NODE \$T/stored_perbar/\$d/; done
for x in baseline:1 all_features:4; do b=\${x%%:*}; c=\${x#*:}; d=\$b/bar1600/lstm_intraday/tw2000/chunks/c\$c; mkdir -p \$T/stored_intraday/\$d; cp -rp \$I/\$d/causal_tune_lstm_intraday \$I/\$d/NODE \$T/stored_intraday/\$d/; done
cp -p \$I/tune_cache/baseline/tune_R00250.npz \$T/stored_intraday/tune_cache/baseline/; cp -p \$I/tune_cache/all_features/tune_R00750.npz \$T/stored_intraday/tune_cache/all_features/
cp -rp $REMOTE/results/momvix_gates/perbar \$T/carc_perbar; cp -rp $REMOTE/results/momvix_gates/intraday \$T/carc_intraday
cd \$T && tar czf - --exclude=identity_scratch . ; rm -rf \$T" | tar xzf - -C "$R"
    echo "refs in $R"; exit 0
    ;;
  *) echo "PART=perbar|intraday|refs"; exit 1 ;;
esac
"$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE/logs && tar czf - mvl_* submitted_momvix_lstm_* 2>/dev/null" | tar xzf - -C "$L/logs"
"$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && ids=\$(cat logs/submitted_momvix_lstm_*.txt* | grep -o '[0-9]\{8\}' | sort -u | paste -sd,) && sacct -X -P -n -j \$ids -o JobID,JobName,AllocCPUS,ElapsedRaw,Start,End,State,CPUTimeRAW,TotalCPU,NodeList,Partition && echo '#STEPS' && sacct -P -n -j \$ids -o JobID,TotalCPU,MaxRSS | grep '\.batch'" 2>&1 | q > "$L/logs/sacct_raw.txt"
echo "pulled $P: $(find "$L" -name MERGED | wc -l) merged arm(s); $(cat "$L/MERGED_ALL" 2>/dev/null || echo 'no MERGED_ALL')"
