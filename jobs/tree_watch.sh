#!/bin/bash
# tree_watch.sh — drive menu -> freeze -> treebank without supervision.
# Runs on the H2 login node (no heavy compute). Logs to logs/tree_watch.log.
set -u
export PATH=/u/local/bin:$PATH
cd /u/scratch/j/jamesdc1/harxhar-clean
PY=/u/home/j/jamesdc1/.conda/envs/hpc-pi/bin/python
LOG=logs/tree_watch.log
MIN_COMPLETE=12   # per family (need 8 unique for the freeze top-8; margin)
MAX_ATTEMPTS=4

complete_counts() {
$PY - <<'EOF'
import optuna
from optuna.storages import JournalStorage
try:
    from optuna.storages.journal import JournalFileBackend as B
except ImportError:
    from optuna.storages import JournalFileStorage as B
import os
out=[]
for fam in ("lgbm","xgb"):
    p=f"results/tree_menu_journal_{fam}.log"
    n=0
    if os.path.exists(p):
        try:
            st=optuna.load_study(study_name=f"tree_menu_dev_{fam}",storage=JournalStorage(B(p)))
            n=sum(1 for t in st.trials if t.state==optuna.trial.TrialState.COMPLETE and t.value is not None)
        except Exception:
            n=0
    out.append(str(n))
print(" ".join(out))
EOF
}

attempt=1
while [ $attempt -le $MAX_ATTEMPTS ]; do
  echo "$(date) attempt $attempt: waiting for tree_menu array to drain" >> $LOG
  while qstat -u "$USER" 2>/dev/null | grep -q 'tree_menu'; do sleep 300; done
  read L X < <(complete_counts)
  echo "$(date) attempt $attempt done: complete lgbm=$L xgb=$X" >> $LOG
  if [ "${L:-0}" -ge $MIN_COMPLETE ] && [ "${X:-0}" -ge $MIN_COMPLETE ]; then break; fi
  attempt=$((attempt+1))
  if [ $attempt -le $MAX_ATTEMPTS ]; then
    echo "$(date) insufficient trials; resubmitting tree_menu.qsub" >> $LOG
    qsub jobs/sge/tree_menu.qsub >> $LOG 2>&1
  fi
done

if [ "${L:-0}" -lt 8 ] || [ "${X:-0}" -lt 8 ]; then
  echo "$(date) ABORT: still <8 complete trials in a family after $MAX_ATTEMPTS attempts; no freeze" >> $LOG
  exit 1
fi

echo "$(date) freezing menu" >> $LOG
$PY experiments/tree_menu_dev.py --freeze \
  --journal-lgbm results/tree_menu_journal_lgbm.log \
  --journal-xgb  results/tree_menu_journal_xgb.log >> $LOG 2>&1 || {
    echo "$(date) ABORT: freeze failed" >> $LOG; exit 1; }

echo "$(date) submitting treebank" >> $LOG
bash jobs/submit/treebank_h2.sh >> $LOG 2>&1 && \
  echo "$(date) CHAIN COMPLETE: treebank submitted" >> $LOG || \
  echo "$(date) ABORT: treebank submission failed" >> $LOG
