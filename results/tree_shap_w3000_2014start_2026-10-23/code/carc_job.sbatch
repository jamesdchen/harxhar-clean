#!/bin/bash
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --job-name=tshap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --constraint=epyc-7513|epyc-7542
#SBATCH --output=/home1/jc_905/tree_shap/logs/%x.%A_%a.out
# Tree SHAP (RULE_shap.txt) on CARC: line $SLURM_ARRAY_TASK_ID of $HB/res/code/tasks.txt = "<est> <part> <procs>"
# commands separated by ";".  Everything on /home1 (python: the tree agent's /home1 venv_skx, read only; shap and its
# missing dependencies in $HB/pylib; the design: /home1/jc_905/trees_final/designs/w3000, read only, md5-checked here).
# Fit threads 8 and the thread environment of the walk (cluster/slurm/pooltree_rerun13.sbatch).
set -euo pipefail
HB=/home1/jc_905/tree_shap
PY=/home1/jc_905/trees_final/venv_skx/bin/python
D=/home1/jc_905/trees_final/designs/w3000
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8
export OPENSSL_CONF=/home1/jc_905/envs/harxhar-r13/ssl/openssl.cnf
export PYTHONPATH=$HB/pylib
export TREE_SHAP_DESIGN=$D TREE_SHAP_WORK=$HB/work TREE_SHAP_RUNNER_DIR=$HB/runner
CPU=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')
echo "tshap task ${SLURM_ARRAY_TASK_ID} on $(hostname) [$CPU] $(date)"
case "$CPU" in *"EPYC 7513"*|*"EPYC 7542"*) ;; *) echo "not an EPYC 7513/7542 node: $CPU"; exit 4 ;; esac
(cd $D/_work/design_all_features && md5sum -c --quiet MD5SUMS) && echo "design md5 ok"
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $HB/res/code/tasks.txt)
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
echo "tshap task ${SLURM_ARRAY_TASK_ID} done $(date)"
