#!/bin/bash
# Friday-week path-block dig, dependency-chained (reuses the generic sig_masks/sig_dig/sig_collect.sbatch,
# which are CID/ARM/LBL-parametrized): prep -> enet_masks -> {resid_subset=+d8, resid_regime=+EBM} -> collect
# Headline = enetreg2_frirel +EBM (resid_regime) vs deployed 0.12035. Results in logs/sig_col_*.out.
cd /scratch1/jc_905/harxhar-clean
LBL=fri
PREP=$(sbatch --parsable fri_prep.sbatch)
echo "prep=$PREP (enetreg2_fri + enetreg2_frirel)"
for CID in xgb_all_buckets_tw1000_enetreg2_fri_rf480_slim xgb_all_buckets_tw1000_enetreg2_frirel_rf480_slim; do
  MASK=$(sbatch --parsable --dependency=afterok:$PREP --export=ALL,CID=$CID sig_masks.sbatch)
  echo "  $CID masks=$MASK"
  for ARM in resid_subset resid_regime; do
    DIG=$(sbatch --parsable --dependency=afterok:$MASK --export=ALL,CID=$CID,ARM=$ARM,LBL=$LBL sig_dig.sbatch)
    COL=$(sbatch --parsable --dependency=afterany:$DIG --export=ALL,CID=$CID,ARM=$ARM,LBL=$LBL sig_collect.sbatch)
    echo "    $ARM dig=$DIG collect=$COL"
  done
done
echo "SUBMIT_FRI_DONE"
