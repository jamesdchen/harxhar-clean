#!/bin/bash
# Hoffman2/SGE: staged reg_d0 (input-saturation + shrunk wd=0.1 + feature-subsampled 8 bags) to
# validate the Hoffman2 pipeline end-to-end before launching the sweep.
cd /u/scratch/j/jamesdc1/harxhar-clean
export PYTHONPATH="$PWD:$PWD/experiments${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs
CID=xgb_all_buckets_tw1000_enetreg2_linbest_rf480_slim
J0=$(qsub -terse -v CID=$CID,RDEPTH=0,RDIAG=false,WD=0.1,LBL=reg_d0 jobs/sge/moe_dig.sge | cut -d. -f1)
C0=$(qsub -terse -hold_jid "$J0" -v CID=$CID,ARM=resid_regime,LBL=reg_d0 jobs/sge/sig_collect.sge)
echo "SGE_SUBMITTED reg_d0 dig=$J0 col=$C0"
