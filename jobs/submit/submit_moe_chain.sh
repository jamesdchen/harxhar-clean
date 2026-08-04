#!/bin/bash
# Overnight regime-MoE ladder on the linbest cell (resid_regime cascade), all on the SAME
# d8@cs0.5 global stage that produced the 0.12033 EBM ceiling. Chained with afterany deps so
# only one 90-task array runs at a time (stays under the 100-job cap). Each collect prints both
# full-OOS qlike= and the regime-window qlike_h16-19=.
#   ctrl_ebm   : EBM regime control  -> should reproduce ~0.12033 (path sanity anchor)
#   moe_d0     : depth-0 single global NAM (EBM-equivalent baseline) -> can any DL match the EBM?
#   moe_d1     : depth-1 soft-tree gate (2 regimes) -> does a LEARNED partition beat doing-nothing?
#   moe_d2     : depth-2 gate (4 regimes, hierarchical coarse->fine)
#   moe_d1diag : depth-1 gate + high-order MLP probe -> is >=3-way structure the missing gap?
set -u
cd /scratch1/jc_905/harxhar-clean
export PYTHONPATH="$PWD:$PWD/experiments${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs
CID=xgb_all_buckets_tw1000_enetreg2_linbest_rf480_slim

E=$(sbatch --parsable --export=ALL,CID=$CID,LBL=ctrl_ebm jobs/slurm/ctrl_ebm.sbatch)
CE=$(sbatch --parsable --dependency=afterany:$E --export=ALL,CID=$CID,ARM=resid_regime,LBL=ctrl_ebm jobs/slurm/sig_collect.sbatch)

J0=$(sbatch --parsable --dependency=afterany:$CE --export=ALL,CID=$CID,RDEPTH=0,RDIAG=false,LBL=moe_d0 jobs/slurm/moe_dig.sbatch)
C0=$(sbatch --parsable --dependency=afterany:$J0 --export=ALL,CID=$CID,ARM=resid_regime,LBL=moe_d0 jobs/slurm/sig_collect.sbatch)

J1=$(sbatch --parsable --dependency=afterany:$C0 --export=ALL,CID=$CID,RDEPTH=1,RDIAG=false,LBL=moe_d1 jobs/slurm/moe_dig.sbatch)
C1=$(sbatch --parsable --dependency=afterany:$J1 --export=ALL,CID=$CID,ARM=resid_regime,LBL=moe_d1 jobs/slurm/sig_collect.sbatch)

J2=$(sbatch --parsable --dependency=afterany:$C1 --export=ALL,CID=$CID,RDEPTH=2,RDIAG=false,LBL=moe_d2 jobs/slurm/moe_dig.sbatch)
C2=$(sbatch --parsable --dependency=afterany:$J2 --export=ALL,CID=$CID,ARM=resid_regime,LBL=moe_d2 jobs/slurm/sig_collect.sbatch)

J3=$(sbatch --parsable --dependency=afterany:$C2 --export=ALL,CID=$CID,RDEPTH=1,RDIAG=true,LBL=moe_d1diag jobs/slurm/moe_dig.sbatch)
C3=$(sbatch --parsable --dependency=afterany:$J3 --export=ALL,CID=$CID,ARM=resid_regime,LBL=moe_d1diag jobs/slurm/sig_collect.sbatch)

echo "MOE_CHAIN_SUBMITTED"
echo "ctrl_ebm   dig=$E   col=$CE"
echo "moe_d0     dig=$J0  col=$C0"
echo "moe_d1     dig=$J1  col=$C1"
echo "moe_d2     dig=$J2  col=$C2"
echo "moe_d1diag dig=$J3  col=$C3"
