#!/bin/bash
#SBATCH --job-name=sig_prep
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=44G
#SBATCH --output=/scratch1/jc_905/harxhar-clean/logs/sig_prep_%j.out
cd /scratch1/jc_905/harxhar-clean
export TQDM_DISABLE=1
PY=/home1/jc_905/.conda/envs/harxhar/bin/python
echo "=== enetreg2_sig prep (model=xgb): enetreg2 + HIGHER-ORDER within-day path-signature block ==="
echo "    (lvl2-3 vol-arrival-time moments centroid/disp/skew + R-V Levy area + tau-R return-timing) ==="
$PY resid_amortized.py prep xgb all_buckets 1000 1 480 slim enetreg2_sig 2>&1 | grep -E 'PREP|SIG|Error|Traceback'
echo SIG_PREP_DONE
echo "=== enetreg2_sigrel prep: signature block + rolling-relative regime (stack-past-best test) ==="
$PY resid_amortized.py prep xgb all_buckets 1000 1 480 slim enetreg2_sigrel 2>&1 | grep -E 'PREP|SIG|Error|Traceback'
echo SIGREL_PREP_DONE
