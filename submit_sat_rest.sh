#!/bin/bash
# Input-saturation MoE gate ladder d1/d2/d1diag — submit AFTER sat_d0 validates the principled fix
# (lands near/below the no-regime 0.12081). Chained afterany, one 90-task array at a time.
set -u
cd /scratch1/jc_905/harxhar-clean
CID=xgb_all_buckets_tw1000_enetreg2_linbest_rf480_slim
J1=$(sbatch --parsable --export=ALL,CID=$CID,RDEPTH=1,RDIAG=false,LBL=sat_d1 moe_dig.sbatch)
C1=$(sbatch --parsable --dependency=afterany:$J1 --export=ALL,CID=$CID,ARM=resid_regime,LBL=sat_d1 sig_collect.sbatch)
J2=$(sbatch --parsable --dependency=afterany:$C1 --export=ALL,CID=$CID,RDEPTH=2,RDIAG=false,LBL=sat_d2 moe_dig.sbatch)
C2=$(sbatch --parsable --dependency=afterany:$J2 --export=ALL,CID=$CID,ARM=resid_regime,LBL=sat_d2 sig_collect.sbatch)
J3=$(sbatch --parsable --dependency=afterany:$C2 --export=ALL,CID=$CID,RDEPTH=1,RDIAG=true,LBL=sat_d1diag moe_dig.sbatch)
C3=$(sbatch --parsable --dependency=afterany:$J3 --export=ALL,CID=$CID,ARM=resid_regime,LBL=sat_d1diag sig_collect.sbatch)
echo "SAT_REST_SUBMITTED d1=$J1 d2=$J2 d1diag=$J3"
