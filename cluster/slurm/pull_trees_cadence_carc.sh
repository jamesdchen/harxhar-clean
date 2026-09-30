#!/bin/bash
# Pull the MERGED arms of the 16:00 tree cadence campaign (checklist I2) from the CARC root
# (tar over native Windows OpenSSH) and check every pulled file's md5 against the remote.
# Pulled (CSVs and small JSON only; no npz, no chunk folders):
#   results/linear_subsection_trees_dedup/{t10,t1}/<bucket>/bar1600/<model>/tw2000/
#       causal_tune_trees/<model>/<bucket>/{results_bar1600.csv, MERGED}
#   results/linear_subsection_trees_tuned_dedup/{rs10,rs1}/<bucket>/bar1600/<model>/tw2000/
#       causal_tune_trees/<model>/<bucket>/{results_bar1600.csv, results_qsel_bar1600.csv,
#       tune_trace_bar1600.csv, tune_candidates_bar1600.csv, grid_bar1600.json, MERGED}
#   every rung root's reduce_gates.csv and reduce.log, MERGE_SUMMARY.txt, CANARY_OK flags
# Run from Git Bash locally:  bash cluster/slurm/pull_trees_cadence_carc.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
REMOTE=/scratch1/jc_905/harxhar-subsection
LIST=$("$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && find results/linear_subsection_trees_dedup/t10 results/linear_subsection_trees_dedup/t1 \
  results/linear_subsection_trees_tuned_dedup/rs10 results/linear_subsection_trees_tuned_dedup/rs1 \
  -path '*/chunks' -prune -o -type f \( -path '*/causal_tune_trees/*' \( -name 'results_bar1600.csv' -o -name 'results_qsel_bar1600.csv' \
  -o -name 'tune_trace_bar1600.csv' -o -name 'tune_candidates_bar1600.csv' -o -name 'grid_bar1600.json' -o -name MERGED \) \) -print; \
  find results/linear_subsection_trees_dedup results/linear_subsection_trees_tuned_dedup -maxdepth 2 -type f \
  \( -name reduce_gates.csv -o -name reduce.log -o -name MERGE_SUMMARY.txt -o -name CANARY_OK -o -name MERGE_DONE \) -print" 2>/dev/null | sort)
N=$(printf '%s\n' "$LIST" | grep -c .)
echo "pulling $N files"
"$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && tar czf - $(printf '%s ' $LIST)" 2>/dev/null | tar xzf -
REMOTE_SUMS=$("$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && md5sum $(printf '%s ' $LIST)" 2>/dev/null | awk '{p = $2; sub(/^\*/, "", p); print p, $1}')
LOCAL_SUMS=$(md5sum $LIST | awk '{p = $2; sub(/^\*/, "", p); print p, $1}')
BAD=$(awk 'NR == FNR {r[$1] = $2; next} !($1 in r) || r[$1] != $2 {print $1}' <(printf '%s\n' "$REMOTE_SUMS") <(printf '%s\n' "$LOCAL_SUMS"))
if [ -n "$BAD" ]; then
  echo "PULL FAILED: md5 differs for:"; printf '  %s\n' $BAD; exit 1
fi
echo "pull OK: $N files, md5 identical to the remote"
