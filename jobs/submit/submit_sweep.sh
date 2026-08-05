#!/bin/bash
# Gate-window sensitivity sweep: enetreg2 + d8 + EBM-regime across close windows, to test whether the
# close edge is robust to the (un-swept) 16-19 choice. Baseline 16-19 reuses the existing deployed
# enetreg2 cell (dig only); each other window gets its own _cw cell (full prep->masks->dig->collect),
# with CLOSE_LO/CLOSE_HI carried in the job env so the features AND the regime-EBM gate both shift.
cd /scratch1/jc_905/harxhar-clean
export PYTHONPATH="$PWD:$PWD/experiments${PYTHONPATH:+:$PYTHONPATH}"
LBL=sweep
BCID=xgb_all_buckets_tw1000_enetreg2_rf480_slim
BDIG=$(sbatch --parsable --export=ALL,CID=$BCID,ARM=resid_regime,LBL=$LBL jobs/slurm/sig_dig.sbatch)
BCOL=$(sbatch --parsable --dependency=afterany:$BDIG --export=ALL,CID=$BCID,ARM=resid_regime,LBL=$LBL jobs/slurm/sig_collect.sbatch)
echo "W=16-19 (baseline, existing cell) dig=$BDIG col=$BCOL"
for W in "15 19" "17 19" "16 20" "16 18"; do
  set -- $W; LO=$1; HI=$2
  CID=xgb_all_buckets_tw1000_enetreg2_rf480_slim_cw${LO}-${HI}
  E=ALL,CLOSE_LO=$LO,CLOSE_HI=$HI
  P=$(sbatch --parsable --export=$E jobs/slurm/sweep_prep.sbatch)
  M=$(sbatch --parsable --dependency=afterok:$P --export=$E,CID=$CID jobs/slurm/sig_masks.sbatch)
  D=$(sbatch --parsable --dependency=afterok:$M --export=$E,CID=$CID,ARM=resid_regime,LBL=$LBL jobs/slurm/sig_dig.sbatch)
  C=$(sbatch --parsable --dependency=afterany:$D --export=$E,CID=$CID,ARM=resid_regime,LBL=$LBL jobs/slurm/sig_collect.sbatch)
  echo "W=$LO-$HI CID=$CID prep=$P mask=$M dig=$D col=$C"
done
echo SUBMIT_SWEEP_DONE
