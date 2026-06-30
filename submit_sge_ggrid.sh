#!/bin/bash
# Run the EBM-cached ensemble GRID: ONE dig array (fits EBM once/block, emits the full uniform/ghat x {.2,.3,.4}
# grid), then one collect per grid point. ~6x fewer EBM fits than 6 separate arms -> the tunable scorer.
cd /u/scratch/j/jamesdc1/harxhar-clean
mkdir -p logs
CID=xgb_all_buckets_tw1000_enetreg2_linbest_rf480_slim
BASE=ggrid
J=$(qsub -terse -v CID=$CID,BASE=$BASE moe_ggrid.sge | cut -d. -f1)
echo "GGRID dig=$J (one array -> 6 grid outputs)"
for k in auniform_w20 auniform_w30 auniform_w40 aghat_w20 aghat_w30 aghat_w40; do
  C=$(qsub -terse -hold_jid "$J" -v CID=$CID,ARM=resid_regime,LBL=${BASE}_${k} sig_collect.sge)
  echo "  collect ${BASE}_${k} = $C"
done
