#!/bin/bash
# FINAL pre-DL pure-QLIKE dig on the best linear base (enetreg2_linbest, 0.12266):
#   Phase 1 = RETUNED HERO A (global XGB sweep, resid_subset)
#   Phase 2 = HERO B PURE POWER (d8 global + TUNED XGB regime, no EBM; resid_regime REGIME_MODEL=xgb)
# Results in logs/sig_col_*.out (COLLECT ... qlike=). Reference: deployed shaperel+EBM 0.12035.
cd /scratch1/jc_905/harxhar-clean
export PYTHONPATH="$PWD:$PWD/experiments${PYTHONPATH:+:$PYTHONPATH}"
CID=xgb_all_buckets_tw1000_enetreg2_linbest_rf480_slim
MASK=$(sbatch --parsable --export=ALL,CID=$CID jobs/slurm/sig_masks.sbatch)
echo "linbest enet_masks=$MASK"
echo "--- Phase 1: retuned Hero A (global XGB, resid_subset) ---"
for spec in "8 0.3 fa_d8c3" "10 0.3 fa_d10c3" "8 0.5 fa_d8c5" "8 0.1 fa_d8c1" "12 0.3 fa_d12c3"; do
  set -- $spec
  D=$(sbatch --parsable --dependency=afterok:$MASK --export=ALL,CID=$CID,DEPTH=$1,CS=$2,LBL=$3 heroA_dig.sbatch)
  C=$(sbatch --parsable --dependency=afterany:$D --export=ALL,CID=$CID,ARM=resid_subset,LBL=$3 jobs/slurm/sig_collect.sbatch)
  echo "  heroA $3 (d$1 cs$2) dig=$D col=$C"
done
echo "--- Phase 2: Hero B pure power (d8 global + XGB regime, resid_regime) ---"
for spec in "2 0.05 200 50 fb_r2" "3 0.05 200 30 fb_r3" "4 0.05 200 20 fb_r4" "3 0.03 400 30 fb_r3n4" "6 0.05 150 10 fb_r6"; do
  set -- $spec
  D=$(sbatch --parsable --dependency=afterok:$MASK --export=ALL,CID=$CID,RDEPTH=$1,RLR=$2,RNEST=$3,RMCW=$4,LBL=$5 jobs/slurm/heroBxgb_dig.sbatch)
  C=$(sbatch --parsable --dependency=afterany:$D --export=ALL,CID=$CID,ARM=resid_regime,LBL=$5 jobs/slurm/sig_collect.sbatch)
  echo "  heroBxgb $5 (rd$1 lr$2 n$3) dig=$D col=$C"
done
echo SUBMIT_FINALDIG_DONE