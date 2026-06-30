#!/bin/bash
cd /scratch1/jc_905/harxhar-clean; mkdir -p logs
CID=xgb_all_buckets_tw1000_enetreg2_linbest_rf480_slim; BASE=ggrid
JID=$(sbatch --parsable --export=ALL,CID=$CID,BASE=$BASE carc_ggrid.sbatch)
echo "CARC ggrid dig=$JID (one array -> 6 grid outputs, EBM once/block)"
for k in auniform_w20 auniform_w30 auniform_w40 aghat_w20 aghat_w30 aghat_w40; do
  C=$(sbatch --parsable --dependency=afterany:$JID --export=ALL,CID=$CID,ARM=resid_regime,LBL=${BASE}_${k} carc_collect.sbatch)
  echo "  collect ${BASE}_${k}=$C"
done
