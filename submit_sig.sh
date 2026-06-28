#!/bin/bash
# Full higher-order path-signature dig, dependency-chained (no polling):
#   prep (both cells) -> enet_masks (per cell) -> {resid_subset=+d8, resid_regime=+EBM} arrays -> collect
# Final QLIKE lands in logs/sig_col_*.out: COLLECT ... qlike=... and (regime) qlike_h16-19=...
cd /scratch1/jc_905/harxhar-clean
LBL=sig
PREP=$(sbatch --parsable sig_prep.sbatch)
echo "prep=$PREP (enetreg2_sig + enetreg2_sigrel)"
for CID in xgb_all_buckets_tw1000_enetreg2_sig_rf480_slim xgb_all_buckets_tw1000_enetreg2_sigrel_rf480_slim; do
  MASK=$(sbatch --parsable --dependency=afterok:$PREP --export=ALL,CID=$CID sig_masks.sbatch)
  echo "  $CID masks=$MASK"
  for ARM in resid_subset resid_regime; do
    DIG=$(sbatch --parsable --dependency=afterok:$MASK --export=ALL,CID=$CID,ARM=$ARM,LBL=$LBL sig_dig.sbatch)
    COL=$(sbatch --parsable --dependency=afterany:$DIG --export=ALL,CID=$CID,ARM=$ARM,LBL=$LBL sig_collect.sbatch)
    echo "    $ARM dig=$DIG collect=$COL"
  done
done
echo "SUBMIT_SIG_DONE"
