#!/bin/bash
# I1 (2026-09-29): pull the de-duplicated per-bar linear arms from Hoffman2 into the local mirror
# results/linear_subsection_dedup/ (the cluster root writes the same relative paths), then fetch
# the pre-dedup run logs and feature-health tables of the nine 16:00 arms whose local copies are
# the flat results/linear_subsection/arms_hoffman2/ files (no log beside them): baseline and
# all_features from the older Hoffman2 root, live_feasible from the older CARC root.  Those logs
# carry "masked cols per tune" -- the gate that shows what the twelve dropped columns were doing.
# Run from Git Bash locally:
#   bash cluster/pull_linear_dedup_hoffman2.sh
set -euo pipefail
cd "$(dirname "$0")/.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
REMOTE=/u/scratch/j/jamesdc1/harxhar-dedup
H2OLD=/u/scratch/j/jamesdc1/harxhar-subsection
CARCOLD=/scratch1/jc_905/harxhar-subsection
R=results/linear_subsection_dedup

"$SSH" -o BatchMode=yes hoffman2 "cd $REMOTE && tar czf - $R logs/lin_dedup_*" | tar xzf -
echo "pulled: $(find $R -name DONE | wc -l) arms with DONE, $(find $R -name 'results_bar*.csv' | wc -l) results tables"
cat "$R/FLEET_DONE" 2>/dev/null || echo "no FLEET_DONE yet"
# the attribution control (cluster/ship_linear_dedup_control_hoffman2.sh), if it has run
CTRLROOT=/u/scratch/j/jamesdc1/harxhar-dedup-control
C=results/linear_subsection_dedup_control
if "$SSH" -o BatchMode=yes hoffman2 "[ -d $CTRLROOT/$C ]"; then
  "$SSH" -o BatchMode=yes hoffman2 "cd $CTRLROOT && tar czf - $C logs/lin_dedup_ctrl*" | tar xzf -
  mkdir -p "$C/logs" && mv logs/lin_dedup_ctrl* "$C/logs/" 2>/dev/null || true
  echo "control: $(find $C -name DONE | wc -l) arms with DONE of $(tr ',' '
' < cluster/linear_dedup_control_tasks.txt | wc -l)"
fi
# the same-architecture attribution runs (cluster/submit_linear_dedup_samearch.sh), if they ran
S=results/linear_subsection_dedup_samearch
P=results/linear_subsection_dedup_control_rep
if "$SSH" -o BatchMode=yes hoffman2 "[ -d $REMOTE/$S ]"; then
  "$SSH" -o BatchMode=yes hoffman2 "cd $REMOTE && tar czf - $S" | tar xzf -
  echo "same-arch de-dup: $(find $S -name DONE | wc -l) arms with DONE"
fi
if "$SSH" -o BatchMode=yes hoffman2 "[ -d $CTRLROOT/$P ]"; then
  "$SSH" -o BatchMode=yes hoffman2 "cd $CTRLROOT && tar czf - $P" | tar xzf -
  echo "same-arch control repeat: $(find $P -name DONE | wc -l) arms with DONE"
fi
mkdir -p "$R/logs" && mv logs/lin_dedup_* "$R/logs/" 2>/dev/null || true

O=$R/old_runlogs
mkdir -p "$O"
for B in baseline all_features; do
  for E in ridge reclasticnet reclasso; do
    D=results/linear_subsection
    [ "$E" = reclasso ] && D=results/linear_subsection_lassofix
    "$SSH" -o BatchMode=yes hoffman2 "cat $H2OLD/$D/$B/bar1600/$E/tw2000/run.log" > "$O/${B}_${E}_bar1600.log"
    "$SSH" -o BatchMode=yes hoffman2 \
      "cat $H2OLD/$D/$B/bar1600/$E/tw2000/causal_tune_linear/$E/$B/results_bar1600_feature_health.csv" \
      > "$O/${B}_${E}_bar1600_feature_health.csv"
  done
done
for E in ridge reclasticnet reclasso; do
  D=results/linear_subsection
  [ "$E" = reclasso ] && D=results/linear_subsection_lassofix
  "$SSH" -o BatchMode=yes usc-discovery "cat $CARCOLD/$D/live_feasible/bar1600/$E/tw2000/run.log" 2>/dev/null \
    > "$O/live_feasible_${E}_bar1600.log"
  "$SSH" -o BatchMode=yes usc-discovery \
    "cat $CARCOLD/$D/live_feasible/bar1600/$E/tw2000/causal_tune_linear/$E/live_feasible/results_bar1600_feature_health.csv" 2>/dev/null \
    > "$O/live_feasible_${E}_bar1600_feature_health.csv"
done
grep -h "masked cols" "$O"/*.log | cut -c1-160
