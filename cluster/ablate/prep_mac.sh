#!/bin/bash
# 2026-10-29 addendum L, Hoffman2 (Gold 6240, the container of py.sh): the macro cache copied from CARC
# (designs_mac/full_mac) checked, then the linear caches.  (1) md5 of the five files = the CARC manifest (MD5SUMS, written
# by carc_build_mac.sh next to them) and X md5 = MACX (gate GM0 on CARC); (2) gate GM0 again on this copy against
# designs_fix/full (the patched cache, X md5 725eb201...; ablate_design.py checkmac) -> gate/gateGM0_h2.json; (3)
# ablate_design.py submac -> designs_mac/lin/{har_sub, full, <every macro arm>} (column subsets of the macro cache whose
# config records MACRO_CALENDARS; full's X md5 must be 725eb201..., har_sub's 05da4b5b...).  No other cache is touched.
# Exit 2 (copy) / 3 (gate or subset md5) on a failure: the linear pool runs afterok on this job.  env: MACX, MAC (default
# designs_mac/full_mac), ROOT_LIN (default designs_mac/lin).
set -uo pipefail
B=/u/scratch/j/jamesdc1/abl
MACX=${MACX:?MACX must be the macro cache X md5 of gate GM0}
MAC=${MAC:-$B/designs_mac/full_mac}
ROOT_LIN=${ROOT_LIN:-$B/designs_mac/lin}
P="bash $B/code/cluster/ablate/py.sh gb experiments/ablate_design.py"
echo "prep_mac job ${SLURM_JOB_ID:-${JOB_ID:-}} on $(hostname) $(date); cache $MAC"
(cd "$MAC" && md5sum -c MD5SUMS) || { echo "md5 of the copy != the CARC manifest: exit 2"; exit 2; }
m=$(md5sum "$MAC/X.npy" | cut -d' ' -f1)
[ "$m" = "$MACX" ] || { echo "X md5 $m != MACX $MACX: exit 2"; exit 2; }
NT=8 $P checkmac --mac "$MAC" --full $B/designs_fix/full --out $B/gate/gateGM0_h2.json || { echo "GM0 (Hoffman2) FAIL: exit 3"; exit 3; }
NT=8 $P submac --mac "$MAC" --root "$ROOT_LIN" || { echo "submac FAIL: exit 3"; exit 3; }
echo "prep_mac done $(date)"
