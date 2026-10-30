#!/bin/bash
# 2026-10-30 transform ablation, LightGBM confirmation (RULE section 6) on Hoffman2: the parts that CARC's priority
# queue had scheduled for midnight.  One task line (OUT EST SET REFIT FIRST LAST PART, the format of
# results/fomcfix_2026-10-26/code/tree_task.sh) through experiments/pooled_trees_rerun_13.py walk, in the t3t container's
# venv_gb (lightgbm 4.6.0 and numpy 1.26.4, as CARC's venv_skx), via cluster/all48/py_h2.sh.  A Hoffman2 job payload
# (no sbatch, copy or move in here).
#   MODE=part (default): line SLURM_ARRAY_TASK_ID of TASKS
#   MODE=gate: V6's full-set part 2/32 into $TX/gate/V6 and its pred_adj compared bit for bit with CARC's piece at
#              $TX/gate/ref/ (copied there first); passed -> $TX/gate/passed_lgbm_h2 (exit 3 otherwise)
# env: TX (staging holding code/, runs/, gate/), DESIGN (the variant's run dir), TASKS (MODE=part)
set -euo pipefail
: "${TX:?}" "${DESIGN:?}"
MODE=${MODE:-part}
NT=${SLURM_CPUS_PER_TASK:-8}
CPU=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')
P="bash $TX/code/cluster/all48/py_h2.sh gb"
export A48_B=$TX NT
if [ "$MODE" = gate ]; then
  OUT=$TX/gate/V6
  F=lgbm_all_features_bar_2014-01-02_2024-04-30_2of32.parquet
  echo "tx_lgbm_h2 gate on $(hostname) [$CPU] $(date)"
  $P experiments/pooled_trees_rerun_13.py walk --design "$DESIGN" --out "$OUT" --window-days 3000 --est lgbm \
    --set all_features --refit bar --first-day 2014-01-02 --last-day 2024-04-30 --part 2/32 --threads "$NT"
  NT=1 $P -c "
import sys, numpy as np, pandas as pd
a = pd.read_parquet('$TX/gate/ref/$F'); b = pd.read_parquet('$OUT/pieces/$F')
same = len(a) == len(b) and a['date'].equals(b['date']) and np.array_equal(a['pred_adj'].to_numpy(), b['pred_adj'].to_numpy())
print('rows', len(a), len(b), 'bit-identical pred_adj:', same)
sys.exit(0 if same else 3)
"
  echo "$CPU" > "$TX/gate/passed_lgbm_h2"
  echo "gate passed $(date)"
  exit 0
fi
: "${TASKS:?}"
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID:?}p" "$TASKS")
read -r OUT EST SET REFIT FIRST LAST PART <<< "$LINE"
echo "tx_lgbm_h2 ${SLURM_ARRAY_TASK_ID} on $(hostname) [$CPU]: $LINE $(date)"
$P experiments/pooled_trees_rerun_13.py walk --design "$DESIGN" --out "$OUT" --window-days 3000 --est "$EST" \
  --set "$SET" --refit "$REFIT" --first-day "$FIRST" --last-day "$LAST" --part "$PART" --threads "$NT"
echo "done $(date)"
