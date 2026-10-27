#!/bin/bash
# 2026-10-26 addendum G: the linear caches of add_rm_fomc_<g> (HAR + RM + FOMC + g, g = ew vw liq iv sent vd) from the
# FOMC-patched full cache designs_fix/full -> designs_fix/lin/<arm> (ablate_design.py subsets --arms; no other cache
# is touched).
set -euo pipefail
B=/u/scratch/j/jamesdc1/abl
echo "prep_beyond job ${SLURM_JOB_ID:-${JOB_ID:-}} on $(hostname) $(date)"
NT=8 bash $B/code/cluster/ablate/py.sh gb experiments/ablate_design.py subsets --full $B/designs_fix/full \
  --new $B/designs/newser --root $B/designs_fix/lin \
  --arms add_rm_fomc_ew,add_rm_fomc_vw,add_rm_fomc_liq,add_rm_fomc_iv,add_rm_fomc_sent,add_rm_fomc_vd
echo "prep_beyond done $(date)"
