#!/bin/bash
# xgbtop (2026-10-07): pull the MERGED per-bar XGBoost arms with the top dev-span entry (xgb_opt_00) from the CARC
# deployment root /scratch1/jc_905/harxhar-xgbtop -- cluster/slurm/pull_trees_rfdev_carc.sh with this run's roots (tar
# over native Windows OpenSSH, every pulled file's md5 checked against the remote).  Pulled (CSVs and small JSON / flags;
# no npz, no chunk folders):
#   results/linear_subsection_trees_mask_xgbtop/{t10,t1}/...       the masked fixed-settings arms (phase 1)
#   results/linear_subsection_trees_nomask_xgbtop/{t10,t1}/...     the unmasked twin
#   results/linear_subsection_trees_tuned_mask_xgbtop/{rs10,rs1}/... the re-centred random-search rungs (when merged)
#   results/linear_subsection_trees_optuna_mask_xgbtop/paths/...    the re-centred Optuna paths (when merged)
#   results/linear_subsection_trees_mask_xgbtop_xcpu/                the cross-CPU comparison (CSV / flags / logs)
#   the reducers' reduce_gates.csv / reduce.log, MERGE_SUMMARY / MERGE_DONE, CANARY_OK, class census, kept counts,
#   the gate outputs (CSV / flags / NODE), the job logs
# and writes the sacct dump of every job in the submission stamps to results/trees_xgbtop_1600/sacct_carc.txt
# (+ sacct_steps_carc.txt).  PARTS (default "fixed nomask xcpu tuned optuna") selects what is pulled.
# Run from Git Bash locally:  bash cluster/slurm/pull_trees_xgbtop_carc.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
REMOTE=/scratch1/jc_905/harxhar-xgbtop
U=results/linear_subsection_trees_mask_xgbtop
N=results/linear_subsection_trees_nomask_xgbtop
T=results/linear_subsection_trees_tuned_mask_xgbtop
P=results/linear_subsection_trees_optuna_mask_xgbtop
X=results/linear_subsection_trees_mask_xgbtop_xcpu
G=results/trees_xgbtop_gates
O=results/trees_xgbtop_1600
PARTS=${PARTS:-fixed nomask xcpu tuned optuna}
q() { "$SSH" -o BatchMode=yes ${CARC_HOST:-usc-discovery1} "$1" 2>/dev/null | grep -v "reloaded with a version change\|python/3\|^$" || true; }
FIND=""
for part in $PARTS; do
  case "$part" in
    fixed|nomask)
      R=$U; [ "$part" = nomask ] && R=$N
      FIND="$FIND find $R/t10 $R/t1 -path '*/chunks' -prune -o -type f \( -path '*/causal_tune_*' \( -name '*.csv' -o -name '*.json' -o -name MERGED \) \) -print 2>/dev/null;
        find $R -maxdepth 2 -type f \( -name reduce_gates.csv -o -name reduce.log -o -name 'MERGE_SUMMARY*' -o -name CANARY_OK -o -name 'MERGE_DONE*' -o -name kept.log -o -name class_census.csv \) -print 2>/dev/null;
        find $R/kept -type f -name '*.csv' -print 2>/dev/null;" ;;
    tuned)
      FIND="$FIND find $T/rs10 $T/rs1 -path '*/chunks' -prune -o -type f \( -path '*/causal_tune_*' \( -name '*.csv' -o -name '*.json' -o -name MERGED \) \) -print 2>/dev/null;
        find $T -maxdepth 2 -type f \( -name reduce_gates.csv -o -name reduce.log -o -name 'MERGE_*' -o -name CANARY_OK -o -name class_census.csv \) -print 2>/dev/null;" ;;
    optuna)
      FIND="$FIND find $P/paths -type f \( -name '*.csv' -o -name MERGED \) -print 2>/dev/null;
        find $P/stage1 -maxdepth 5 -path '*/chunks' -prune -o -type f \( -name 'trials_*.csv' -o -name STAGE1_COMPLETE \) -print 2>/dev/null;
        find $P -maxdepth 1 -type f \( -name '*.log' -o -name 'GATE_OK_*' -o -name 'canary_ok_*' -o -name 'STAGE*_MERGED' -o -name '*.csv' \) -print 2>/dev/null;
        find $P/gate -maxdepth 2 -type f -name 'gate_*.csv' -print 2>/dev/null;" ;;
    xcpu)
      FIND="$FIND find $X -maxdepth 1 -type f \( -name '*.csv' -o -name '*.log' -o -name 'XCPU_*' \) -print 2>/dev/null;
        find $X -mindepth 2 -path '*/chunks/c*' -name NODE -print 2>/dev/null;" ;;
    *) echo "unknown part $part"; exit 1 ;;
  esac
done
LIST=$(q "cd $REMOTE && { $FIND find $G -maxdepth 2 -type f \( -name '*.csv' -o -name GATE_OK -o -name NODE \) -print 2>/dev/null; } | sort -u")
N_FILES=$(printf '%s\n' "$LIST" | grep -c . || true)
echo "pulling $N_FILES files"
[ "$N_FILES" -gt 0 ] || exit 1
printf '%s\n' $LIST | "$SSH" -o BatchMode=yes ${CARC_HOST:-usc-discovery1} "cd $REMOTE && tar czf - -T -" 2>/dev/null | tar xzf -
REMOTE_SUMS=$(printf '%s\n' $LIST | "$SSH" -o BatchMode=yes ${CARC_HOST:-usc-discovery1} "cd $REMOTE && xargs md5sum" 2>/dev/null | awk '{p = $2; sub(/^\*/, "", p); print p, $1}')
LOCAL_SUMS=$(md5sum $LIST | awk '{p = $2; sub(/^\*/, "", p); print p, $1}')
BAD=$(awk 'NR == FNR {r[$1] = $2; next} !($1 in r) || r[$1] != $2 {print $1}' <(printf '%s\n' "$REMOTE_SUMS") <(printf '%s\n' "$LOCAL_SUMS"))
if [ -n "$BAD" ]; then
  echo "PULL FAILED: md5 differs for:"; printf '  %s\n' $BAD; exit 1
fi
mkdir -p "$O/logs_carc"
"$SSH" -o BatchMode=yes ${CARC_HOST:-usc-discovery1} "cd $REMOTE/logs && tar czf - submitted_trees_xgbtop* submitted_xgbtop* txt_* xgbtop_* op_xgbtop* 2>/dev/null" 2>/dev/null | tar xzf - -C "$O/logs_carc" || true
IDS=$(cat "$O"/logs_carc/submitted_*xgbtop*.txt 2>/dev/null | grep -oE '[=: ][0-9]{7,9}' | tr -d '=: ' | sort -u | paste -sd, -)
if [ -n "$IDS" ]; then
  q "sacct -X -P -n -j $IDS -o JobID,JobName,AllocCPUS,ElapsedRaw,Start,End,State,CPUTimeRAW,NodeList" > "$O/sacct_carc.txt"
  q "sacct -P -n -j $IDS -o JobID,TotalCPU,MaxRSS" | grep '\.batch|' > "$O/sacct_steps_carc.txt" || true
fi
echo "pull OK: $N_FILES files, md5 identical to the remote; sacct: $(wc -l < "$O/sacct_carc.txt" 2>/dev/null || echo 0) allocations"
