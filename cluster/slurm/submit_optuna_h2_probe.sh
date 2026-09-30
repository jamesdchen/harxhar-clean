#!/bin/bash
# I5 (2026-09-29): submit the CARC side of the design probe of the cross-cluster check (one 1-CPU
# task per line of the check's probe task file; the Hoffman2 side is cluster/submit_optuna_h2.sh).
#   cd /scratch1/jc_905/harxhar-subsection && XROOT=<root>/crosscluster/carc_<tag> WINDOW_MASK=<0|1> \
#     [FEATURES="epyc-7513 epyc-7542 xeon-4116"] bash cluster/slurm/submit_optuna_h2_probe.sh <tag>
# Task file: cluster/optuna_h2_xcheck_archprobe_<tag>.txt, else ..._probe_<tag>.txt, else ..._s2_<tag>.txt.
# FEATURES: one array per CARC node feature (sbatch --constraint), so the design can be compared
# across CARC's CPU types too; unset = one unconstrained array.
set -euo pipefail
cd "$(dirname "$0")/../.."
TAG=${1:?tag}
T=cluster/optuna_h2_xcheck_archprobe_$TAG.txt
[ -f "$T" ] || T=cluster/optuna_h2_xcheck_probe_$TAG.txt
[ -f "$T" ] || T=cluster/optuna_h2_xcheck_s2_$TAG.txt
: "${XROOT:?XROOT}" "${WINDOW_MASK:?WINDOW_MASK}"
for f in "$T" experiments/optuna_h2_design_probe.py cluster/slurm/optuna_h2_probe_carc.sbatch; do
  [ -f "$f" ] || { echo "$f missing (ship it first)"; exit 1; }
done
mkdir -p logs "$XROOT/probe"
STAMP=logs/submitted_optuna_h2_probe_$TAG.txt
[ -f "$STAMP" ] && { echo "already submitted ($(cat "$STAMP"))"; exit 1; }
IDS=""
for F in ${FEATURES:-none}; do
  C=""; [ "$F" = none ] || C="--constraint=$F"
  J=$(sbatch --parsable $C --array=1-"$(wc -l < "$T")" --export=ALL,TASKFILE=$T,XROOT=$XROOT,WINDOW_MASK=$WINDOW_MASK \
        cluster/slurm/optuna_h2_probe_carc.sbatch)
  IDS="$IDS $F:$J"
done
echo "probe $TAG ($T, mask $WINDOW_MASK):$IDS $(date)" | tee "$STAMP"
