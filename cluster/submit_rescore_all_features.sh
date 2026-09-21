#!/bin/bash
# Re-run the causal re-score of all_features only (array task 10), held on the job
# that finishes its pooled arms' scoring.
#   bash cluster/submit_rescore_all_features.sh <results_root> <hold_jid>
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
ROOT_DIR="${1:?results root}"
HOLD="${2:?job id to hold on}"
J=$(qsub -terse -N rescore_af -t 10-10 -hold_jid "$HOLD" -v ROOT_DIR="$ROOT_DIR" cluster/rescore_causal.sh | cut -d. -f1)
echo "rescore_af=$J root=$ROOT_DIR hold=$HOLD" | tee -a logs/submitted_rescore.txt
qstat -u "$USER" | tail -4
