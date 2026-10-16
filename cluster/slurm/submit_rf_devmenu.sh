#!/bin/bash
# Random-forest dev-span menu study (2026-10-06): submit on CARC from the deployment root that
# cluster/slurm/ship_rf_devmenu_carc.sh fills (/scratch1/jc_905/harxhar-rf-devmenu):
#
#   bash cluster/slurm/submit_rf_devmenu.sh canary [TAG]   gate + cost probe + 2 canary trials (one task; TAG
#                                                          names its folder and stamp, default canary)
#   bash cluster/slurm/submit_rf_devmenu.sh fleet          8 TPE workers x 50 trials (refuses without
#                                                          CANARY_OK): jobs/slurm/tree_menu.sbatch's layout
#   bash cluster/slurm/submit_rf_devmenu.sh topup N K      K more workers x N trials (a failed / timed-out
#                                                          worker's missing trials; stamp per N_K)
#   bash cluster/slurm/submit_rf_devmenu.sh seqgate        the stored entries re-scored by tree_menu_dev's own
#                                                          screen (one 8-CPU task, 8 threads a fit)
#   bash cluster/slurm/submit_rf_devmenu.sh status         journal trial states + this study's jobs
#
# Every task is pinned to one CPU class (CLASS, default epyc-7513: the class of the 16:00 campaign).
# REFUSES to submit from a shell with a conda environment active (jobs inherit the submitting
# environment; an env leak through --export=ALL killed two agents' jobs on 2026-10-06); the task
# script also drops conda / python variables and calls the harxhar interpreter by absolute path.
# A stamp (logs/submitted_rf_devmenu_<mode>.txt) refuses a second identical submission.
# Env: CLASS, CPUS (25: the 50 screen fits of a trial in two rounds), MEM (80G), CANARY_TIME (2:30:00),
# WORKER_TIME (20:00:00), CANARY_TAG (the canary folder fleet / topup require; default canary).
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs
MODE=${1:?usage: submit_rf_devmenu.sh canary|fleet|topup N K|status}
CLASS=${CLASS:-epyc-7513}
CPUS=${CPUS:-25}
MEM=${MEM:-80G}
CANARY_TAG=${CANARY_TAG:-canary}
CANARY_TIME=${CANARY_TIME:-2:30:00}
WORKER_TIME=${WORKER_TIME:-20:00:00}
WORK=results/rf_devmenu_2026-10-06/_work
SB=cluster/slurm/rf_devmenu_task.sbatch
LOG=logs/submitted_rf_devmenu.txt
if [ -n "${CONDA_PREFIX:-}" ] || [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
  echo "REFUSED: a conda environment is active (${CONDA_PREFIX:-$CONDA_DEFAULT_ENV}); submit from a clean login shell"
  exit 1
fi
for f in experiments/tree_menu_dev_rf.py experiments/tree_menu_dev.py experiments/_bootstrap.py "$SB" \
         "$WORK/dev/X.npy" "$WORK/dev/y.npy" "$WORK/dev/names.npy" "$WORK/dev/dev_manifest.json"; do
  [ -f "$f" ] || { echo "$f missing: run cluster/slurm/ship_rf_devmenu_carc.sh locally first"; exit 1; }
done
[ -f rf_devmenu_manifest.md5 ] && { md5sum -c --quiet rf_devmenu_manifest.md5 || { echo "shipped files changed since the ship; re-ship"; exit 1; }; }
COMMON=(--parsable --account=pollok_1603 --partition=main --constraint="$CLASS" --cpus-per-task="$CPUS" --mem="$MEM")

stamp_or_refuse() {
  [ -f "$1" ] && { echo "already submitted ($(cat "$1")); delete $1 by hand to resubmit"; exit 1; }
  return 0
}

case "$MODE" in
  canary)
    CT=${2:-canary}
    ST=logs/submitted_rf_devmenu_$CT.txt; stamp_or_refuse "$ST"
    J=$(sbatch "${COMMON[@]}" -J rfdm_canary --time="$CANARY_TIME" --export="ALL,STAGE=canary,CANARY_TAG=$CT" "$SB")
    echo "canary=$J tag=$CT class=$CLASS cpus=$CPUS mem=$MEM $(date)" | tee "$ST" >> "$LOG"
    echo "SUBMITTED canary $J"
    ;;
  fleet)
    [ -f "$WORK/$CANARY_TAG/CANARY_OK" ] || { echo "$CANARY_TAG/CANARY_OK missing; refusing the fleet"; exit 1; }
    ST=logs/submitted_rf_devmenu_fleet.txt; stamp_or_refuse "$ST"
    J=$(sbatch "${COMMON[@]}" -J rfdm_worker --time="$WORKER_TIME" --array=0-7 \
          --export="ALL,STAGE=worker,N_TRIALS=50,CANARY_TAG=$CANARY_TAG" "$SB")
    echo "fleet=$J workers=8 trials=50 class=$CLASS cpus=$CPUS mem=$MEM $(date)" | tee "$ST" >> "$LOG"
    echo "SUBMITTED fleet $J"
    ;;
  topup)
    N=${2:?trials per worker}; K=${3:?workers}
    [ -f "$WORK/$CANARY_TAG/CANARY_OK" ] || { echo "$CANARY_TAG/CANARY_OK missing; refusing"; exit 1; }
    ST="logs/submitted_rf_devmenu_topup_${N}_${K}.txt"; stamp_or_refuse "$ST"
    J=$(sbatch "${COMMON[@]}" -J rfdm_topup --time="$WORKER_TIME" --array=0-$((K - 1)) \
          --export="ALL,STAGE=worker,N_TRIALS=$N,CANARY_TAG=$CANARY_TAG" "$SB")
    echo "topup=$J workers=$K trials=$N class=$CLASS cpus=$CPUS mem=$MEM $(date)" | tee "$ST" >> "$LOG"
    echo "SUBMITTED topup $J"
    ;;
  seqgate)
    ST=logs/submitted_rf_devmenu_seqgate.txt; stamp_or_refuse "$ST"
    J=$(sbatch --parsable --account=pollok_1603 --partition=main --constraint="$CLASS" --cpus-per-task=8           --mem=32G -J rfdm_seqgate --time=3:00:00 --export="ALL,STAGE=seqgate" "$SB")
    echo "seqgate=$J class=$CLASS cpus=8 $(date)" | tee "$ST" >> "$LOG"
    echo "SUBMITTED seqgate $J"
    ;;
  status)
    /home1/jc_905/.conda/envs/harxhar/bin/python - "$WORK/tree_menu_journal_rf.log" <<'EOF' || true
import collections, os, sys
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
optuna.logging.set_verbosity(optuna.logging.WARNING)
p = sys.argv[1]
if not os.path.exists(p):
    print("no journal yet"); sys.exit(0)
st = optuna.load_study(study_name="tree_menu_dev_rf", storage=JournalStorage(JournalFileBackend(p)))
c = collections.Counter(t.state.name for t in st.trials)
done = [t for t in st.trials if t.state.name == "COMPLETE"]
fs = sum(t.user_attrs.get("fit_sec_sum", 0.0) for t in done) / 3600
best = min(done, key=lambda t: t.value) if done else None
print("states", dict(c), "fit core-h %.1f" % fs, "best", best.value if best else None, best.params if best else None)
EOF
    [ -f "$LOG" ] && cat "$LOG"
    squeue -u "$USER" -h -n rfdm_canary,rfdm_worker,rfdm_topup,rfdm_seqgate -o '%i %j %T %M %l %R' || true
    ;;
  *)
    echo "unknown mode $MODE"; exit 1 ;;
esac
