#!/bin/bash
# Pull for the C1 feature-importance re-run on the de-duplicated per-bar design (checklist
# I7; experiments/feature_importance_1530_dedup.py) from the CARC root, tar over native
# Windows OpenSSH, and FAIL unless every pulled file's md5 equals the remote one.
#   stored   the MERGED 16:00 tree runs of the cadence campaign (checklist I2) that the
#            re-run reproduces and reads:
#              results/linear_subsection_trees_dedup/{t10,t1}/<bucket>/bar1600/<model>/tw2000/
#                  causal_tune_trees/<model>/<bucket>/trees_bar1600.npz
#              results/linear_subsection_trees_tuned_dedup/rs10/... (same layout)
#            into results/feature_importance_1530_dedup/_work/stored/{t10,t1,rs10}/<same layout>
#            (T10 = the stored forecasts / params / thread counts the refits must reproduce;
#            T1 = the every-session refit, for the cadence identity; RS10 = the tuned trees'
#            own importance extracts).  Nothing is written into the campaign's own roots.
#   results  the fleet's outputs: results/feature_importance_1530_dedup{,_nomask}/_work/trees/*.npz and
#            gate / flag files, _work/linear/*_part*.npz (the cluster's lasso drop-column
#            refits), and the Slurm logs of the fleet into _work/logs/.
# Run from Git Bash locally:  bash cluster/slurm/pull_featimp_dedup_carc.sh stored|results
set -euo pipefail
cd "$(dirname "$0")/../.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
REMOTE=/scratch1/jc_905/harxhar-subsection
MODE="${1:-}"
D=results/feature_importance_1530_dedup
sums() { awk '{p = $2; sub(/^\*/, "", p); print p, $1}'; }
check() {  # <remote sums> <local sums>
  local BAD
  BAD=$(awk 'NR == FNR {r[$1] = $2; next} !($1 in r) || r[$1] != $2 {print $1}' <(printf '%s\n' "$1") <(printf '%s\n' "$2"))
  if [ -n "$BAD" ]; then
    echo "PULL FAILED: md5 differs (or no remote line) for:"; printf '  %s\n' $BAD; exit 1
  fi
}

case "$MODE" in
  stored)
    mkdir -p "$D/_work/stored"
    for SRC in linear_subsection_trees_dedup:t10 linear_subsection_trees_dedup:t1 linear_subsection_trees_tuned_dedup:rs10; do
      BASE="results/${SRC%%:*}"
      RUNG="${SRC##*:}"
      LIST=$("$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE/$BASE && find $RUNG -path '*/chunks' -prune -o -type f -path '*/causal_tune_trees/*' -name 'trees_bar1600.npz' -print" 2>/dev/null | sort)
      N=$(printf '%s\n' "$LIST" | grep -c .)
      [ "$N" -eq 9 ] || { echo "$BASE/$RUNG: $N merged npz files on CARC, want 9"; exit 1; }
      "$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE/$BASE && tar czf - $(printf '%s ' $LIST)" 2>/dev/null | tar xzf - -C "$D/_work/stored"
      R=$("$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE/$BASE && md5sum $(printf '%s ' $LIST)" 2>/dev/null | sums)
      L=$(cd "$D/_work/stored" && md5sum $LIST | sums)
      check "$R" "$L"
      echo "pull OK: $BASE/$RUNG -> $D/_work/stored/$RUNG ($N npz, md5 identical)"
    done
    ;;
  results)
    LIST=$("$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && find $D/_work/trees $D/_work/linear results/feature_importance_1530_dedup_nomask/_work/trees -maxdepth 1 -type f \( -name '*.npz' -o -name 'CANARY_OK*' -o -name FLEET_DONE -o -name '*.gate' \) -print 2>/dev/null" 2>/dev/null | sort)
    N=$(printf '%s\n' "$LIST" | grep -c .)
    echo "pulling $N files"
    "$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && tar czf - $(printf '%s ' $LIST)" 2>/dev/null | tar xzf -
    R=$("$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && md5sum $(printf '%s ' $LIST)" 2>/dev/null | sums)
    L=$(md5sum $LIST | sums)
    check "$R" "$L"
    mkdir -p "$D/_work/logs"
    "$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE/logs && tar czf - \$(ls | grep -E '^(fid_|featimp_dedup)' || true)" 2>/dev/null | tar xzf - -C "$D/_work/logs" || true
    echo "pull OK: $N files, md5 identical to the remote; logs in $D/_work/logs"
    ;;
  *)
    echo "usage: bash cluster/slurm/pull_featimp_dedup_carc.sh stored|results"
    exit 1
    ;;
esac
