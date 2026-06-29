#!/bin/bash
#SBATCH --job-name=explain
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=44G
#SBATCH --output=/scratch1/jc_905/harxhar-clean/logs/explain_%j.out
# Explainability audit (edge_06): the legible-coefficient trajectory (0DTE dilution test) + the learned
# MoE gate hyperplane (does it recover `hour`?), depth 1 and 2. Writes results/moe_ladder/coef_trajectory.csv.
cd /scratch1/jc_905/harxhar-clean
export TQDM_DISABLE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/home1/jc_905/.conda/envs/harxhar/bin/python
CID=xgb_all_buckets_tw1000_enetreg2_linbest_rf480_slim
echo "=== coef_trajectory (0DTE/OPEX dilution test) ==="
$PY coef_trajectory.py $CID 2>&1 | grep -E 'blocks=|mean=|WROTE|COEF_TRAJ_DONE|Error|Traceback' || true
echo "=== gate_readout depth=1 ==="
READOUT_DEPTH=1 $PY gate_readout.py $CID 2>&1 | grep -E 'block i=|gate node|w=|hour_rank|hour NOT|GATE_READOUT_DONE|Error|Traceback' || true
echo "=== gate_readout depth=2 ==="
READOUT_DEPTH=2 $PY gate_readout.py $CID 2>&1 | grep -E 'block i=|gate node|w=|hour_rank|hour NOT|GATE_READOUT_DONE|Error|Traceback' || true
echo EXPLAIN_ALLDONE
