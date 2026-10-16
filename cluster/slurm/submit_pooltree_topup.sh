#!/bin/bash
# Pooled forest all_features (2026-10-14): start the self-chaining hourly top-up (cluster/slurm/pooltree_af_topup_chain.sbatch;
# release-only, see its header).  Installs it as $HOME/pooltree_topup.sbatch and submits the FIRST run now; each run then
# resubmits itself an hour later.  Refuses if a ptr_topup job is already queued (one chain only).
#   bash cluster/slurm/submit_pooltree_topup.sh        (on CARC, from the deployment root, no conda env)
# Stop the chain: touch $HOME/pooltree_topup.STOP   (the next run exits without releasing or resubmitting)
set -euo pipefail
cd "$(dirname "$0")/../.."
Q=$(squeue -u "$USER" -h -n ptr_topup -o %A)
[ -z "$Q" ] || { echo "a ptr_topup job is already queued ($Q); refusing"; exit 1; }
[ ! -e "$HOME/pooltree_topup.STOP" ] || { echo "$HOME/pooltree_topup.STOP exists; remove it first"; exit 1; }
install -m 644 cluster/slurm/pooltree_af_topup_chain.sbatch "$HOME/pooltree_topup.sbatch"
md5sum "$HOME/pooltree_topup.sbatch"
J=$(sbatch --parsable -o "$HOME/pooltree_topup.slurm.out" "$HOME/pooltree_topup.sbatch")
echo "$(date +%F_%T) ptr_topup chain started: $J" | tee -a logs/submitted_pooltrees.txt
