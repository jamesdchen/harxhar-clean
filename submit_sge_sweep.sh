#!/bin/bash
# Hoffman2/SGE: the regularization SWEEP — shrinkage (weight_decay) × gate depth, all arrays in
# parallel (Hoffman2's pool gives the capacity that CARC's 100-cap doesn't). Run AFTER submit_sge.sh
# validates reg_d0. Each prints full-OOS + h16-19 QLIKE at collect.
cd /u/scratch/j/jamesdc1/harxhar-clean
mkdir -p logs
CID=xgb_all_buckets_tw1000_enetreg2_linbest_rf480_slim
PFX=${1:-reg}   # label prefix (reg2 = NAM resubmit; fm = Factorization-Machine expert)
EXPERT=${EXPERT:-fm}   # fm (native pairwise) | nam (additive)
sub () {  # depth wd diag label
  local J C
  J=$(qsub -terse -v CID=$CID,RDEPTH=$1,WD=$2,RDIAG=$3,EXPERT=$EXPERT,LBL=$4 moe_dig.sge | cut -d. -f1)
  C=$(qsub -terse -hold_jid "$J" -v CID=$CID,ARM=resid_regime,LBL=$4 sig_collect.sge)
  echo "  $4 (d$1 wd$2 diag$3) dig=$J col=$C"
}
echo "SWEEP_SUBMITTED prefix=$PFX"
sub 0 0.03 false ${PFX}_d0_wd03
sub 0 0.1  false ${PFX}_d0_wd10
sub 0 0.3  false ${PFX}_d0_wd30
sub 1 0.1  false ${PFX}_d1_wd10
sub 2 0.1  false ${PFX}_d2_wd10
