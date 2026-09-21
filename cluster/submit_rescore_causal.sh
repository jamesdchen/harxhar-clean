#!/bin/bash
# Submit the causal re-score of the ten feature buckets (one array task each).
#   bash cluster/submit_rescore_causal.sh [results_root]
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
ROOT_DIR="${1:-results/linear_subsection}"
J=$(qsub -terse -N rescore -t 1-10 -v ROOT_DIR="$ROOT_DIR" cluster/rescore_causal.sh | cut -d. -f1)
echo "rescore=$J root=$ROOT_DIR" | tee -a logs/submitted_rescore.txt
qstat -u "$USER" | tail -4
