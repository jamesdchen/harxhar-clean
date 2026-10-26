#!/bin/bash
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --job-name=tshapff
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --constraint=epyc-7513|epyc-7542
#SBATCH --output=/home1/jc_905/tree_shap_fomcfix/logs/%x.%A_%a.out
# Tree SHAP on the FOMC-patched design (results/ladder_2026-10-26/RULE.txt ADDENDUM 3 (4); RULE_shap.txt of
# results/tree_shap_w3000_2014start_2026-10-23 unchanged): line $SLURM_ARRAY_TASK_ID of $TASKFILE = "<est> <part> <procs>"
# commands separated by ";".  The study's code unchanged; the design: fomcfix cell B (/home1/jc_905/fomcfix_lin/runs/B,
# read only); python, shap pylib and the runner of the pre-patch study (read only).
set -euo pipefail
HB=/home1/jc_905/tree_shap_fomcfix
PY=/home1/jc_905/trees_final/venv_skx/bin/python
D=/home1/jc_905/fomcfix_lin/runs/B
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8
export OPENSSL_CONF=/home1/jc_905/envs/harxhar-r13/ssl/openssl.cnf
export PYTHONPATH=/home1/jc_905/tree_shap/pylib
export TREE_SHAP_DESIGN=$D TREE_SHAP_WORK=$HB/work TREE_SHAP_RUNNER_DIR=/home1/jc_905/tree_shap/runner
CPU=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')
echo "tshapff task ${SLURM_ARRAY_TASK_ID} on $(hostname) [$CPU] $(date)"
case "$CPU" in *"EPYC 7513"*|*"EPYC 7542"*) ;; *) echo "not an EPYC 7513/7542 node: $CPU"; exit 4 ;; esac
md5sum $D/_work/design_all_features/X.npy
TF=${TASKFILE:-$HB/res/code/tasks.txt}
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TF")
[ -n "$LINE" ] || { echo "no line ${SLURM_ARRAY_TASK_ID}"; exit 1; }
cd $HB/res/code
IFS=';' read -ra CMDS <<< "$LINE"
for c in "${CMDS[@]}"; do
  read -r EST PART PROCS <<< "$c"
  if [ "$EST" = "export_x" ]; then
    "$PY" export_x.py
    continue
  fi
  "$PY" shap_run.py --est "$EST" --part "$PART" --threads 8 --shap-threads 16 --procs "$PROCS"
done
echo "tshapff task ${SLURM_ARRAY_TASK_ID} done $(date)"
