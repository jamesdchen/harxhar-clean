#!/bin/bash
# Gate 0a of the ablation run (RULE.txt section 5): the Hoffman2 copies of the final full and baseline caches against
# the templates' md5, the HAR columns of the full cache = the baseline X, group coverage, E-mini availability constant.
# -> gate/gate0a.json; exit 3 on a failure.
set -euo pipefail
B=/u/scratch/j/jamesdc1/abl
echo "prep0 job ${SLURM_JOB_ID:-${JOB_ID:-}} on $(hostname) $(date)"
NT=2 bash $B/code/cluster/ablate/py.sh gb experiments/ablate_design.py check --full $B/designs/full --har $B/designs/har \
  --out $B/gate/gate0a.json
echo "prep0 done $(date)"
