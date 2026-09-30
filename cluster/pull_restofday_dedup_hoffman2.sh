#!/bin/bash
# I6 (2026-09-29): pull the direct rest-of-day arms re-run on the de-duplicated per-bar design from
# Hoffman2 into the local mirror results/linear_subsection_restofday_dedup/ (the cluster root writes
# the same relative paths), with the job logs, the submission stamp and the accounting (qacct) of
# every I6 job; map every host the arms (and agent A's per-bar 16:00 twins) ran on to its CPU
# architecture (qhost -F arch -> host_arch.txt); the attribution control's arms (if it ran) into
# results/linear_subsection_restofday_dedup_control/; and fetch, read-only, the pre-dedup arms' run.log
# and feature-health tables from the older CARC root into old_carc/ (flattened as
# <bucket>_<est>_<clock>_{run.log,feature_health.csv}: nothing there looks like an arm file to the
# scorer or the stacker).  Run from Git Bash locally:
#   bash cluster/pull_restofday_dedup_hoffman2.sh
set -euo pipefail
cd "$(dirname "$0")/.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
REMOTE=/u/scratch/j/jamesdc1/harxhar-restofday-dedup
CARCOLD=/scratch1/jc_905/harxhar-subsection
R=results/linear_subsection_restofday_dedup
SGE="source /u/systems/UGE8.6.4/hoffman2/common/settings.sh"

mkdir -p "$R/logs"
"$SSH" -o BatchMode=yes hoffman2 "cd $REMOTE && tar czf - $R logs/rod_dedup_* logs/submitted_restofday_dedup.txt" | tar xzf -
mv logs/rod_dedup_* "$R/logs/" 2>/dev/null || true
mv logs/submitted_restofday_dedup.txt "$R/" 2>/dev/null || true
echo "pulled: $(find $R -name DONE | wc -l) arms with DONE, $(find $R -name 'results_bar????.csv' | wc -l) results tables"
cat "$R/FLEET_DONE" 2>/dev/null || echo "no FLEET_DONE yet"

# the attribution control (cluster/submit_restofday_dedup_control.sh: pre-dedup executor, same
# architecture), if it has run: its root writes $R too; locally it lands in $C
CROOT=/u/scratch/j/jamesdc1/harxhar-restofday-dedup-control
C=results/linear_subsection_restofday_dedup_control
if "$SSH" -o BatchMode=yes hoffman2 "[ -d $CROOT/$R ]"; then
  "$SSH" -o BatchMode=yes hoffman2 "cd $CROOT && tar czf - $R logs/rod_dedup_ctrl* logs/submitted_restofday_dedup_control.txt" \
    | tar xzf - --transform="s,^$R,$C," --transform="s,^logs/,$C/logs/,"
  mv "$C/logs/submitted_restofday_dedup_control.txt" "$C/" 2>/dev/null || true
  CIDS=$(grep -o '=[0-9]*' "$C/submitted_restofday_dedup_control.txt" | tr -d '=' | tr '\n' ' ')
  "$SSH" -o BatchMode=yes hoffman2 "$SGE; for j in $CIDS; do qacct -j \$j 2>/dev/null; done" > "$C/qacct.txt" || true
  echo "control: $(find $C -name DONE | wc -l) arms with DONE of $(cat cluster/restofday_dedup_control_*.txt | wc -l)"
fi

# hosts -> CPU architecture (the new arms, the control, and agent A's I1 hosts, where the 15:30
# twins ran)
HOSTS=$( { find "$R" "$C" -name HOST -exec awk '{print $1}' {} \; 2>/dev/null ;
           grep -ho '^task [0-9]* on n[0-9]*' results/linear_subsection_dedup/logs/lin_dedup_*.o* | awk '{print $4}'; } \
         | sort -u | tr '\n' ' ')
"$SSH" -o BatchMode=yes hoffman2 "$SGE; for h in $HOSTS; do echo \"\$h \$(qhost -F arch -h \$h | grep -o 'arch=[^ ]*' | head -1 | cut -d= -f2)\"; done" \
  > "$R/host_arch.txt"
echo "host_arch.txt: $(wc -l < "$R/host_arch.txt") hosts"

# accounting of every I6 job (canary, the 15:30 arrays, the fleet, the collector)
IDS=$(grep -o '=[0-9]*' "$R/submitted_restofday_dedup.txt" | tr -d '=' | tr '\n' ' ')
"$SSH" -o BatchMode=yes hoffman2 "$SGE; for j in $IDS; do qacct -j \$j 2>/dev/null; done" > "$R/qacct.txt" || true
echo "qacct.txt: $(grep -c '^jobnumber' "$R/qacct.txt" || true) job tasks accounted"

# the pre-dedup arms' run.log and feature-health tables (older CARC root, read only; once)
NOLD=$(ls "$R"/old_carc/*_run.log 2>/dev/null | wc -l)
if [ "$NOLD" -ge 117 ]; then
  echo "old_carc: $NOLD pre-dedup arms already here (not re-fetched)"
  exit 0
fi
TMP=$(mktemp -d)
"$SSH" -o BatchMode=yes usc-discovery "cd $CARCOLD && tar czf - results/linear_subsection_restofday/*/rod*/*/tw2000/run.log results/linear_subsection_restofday/*/rod*/*/tw2000/causal_tune_rest_of_day/*/*/results_bar*_feature_health.csv" \
  | tar xzf - -C "$TMP"
mkdir -p "$R/old_carc"
N=0
for f in "$TMP"/results/linear_subsection_restofday/*/rod*/*/tw2000/run.log; do
  IFS=/ read -r -a P <<< "${f#"$TMP"/results/linear_subsection_restofday/}"
  B=${P[0]}; C=${P[1]#rod}; E=${P[2]}
  cp "$f" "$R/old_carc/${B}_${E}_${C}_run.log"
  cp "$(dirname "$f")"/causal_tune_rest_of_day/"$E"/"$B"/results_bar*_feature_health.csv \
     "$R/old_carc/${B}_${E}_${C}_feature_health.csv"
  N=$((N + 1))
done
rm -rf "$TMP"
echo "old_carc: $N pre-dedup arms (run.log + feature health)"
