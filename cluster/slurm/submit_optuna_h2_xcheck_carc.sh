#!/bin/bash
# I5 (2026-09-29): the CARC side of the cross-cluster check, pinned per CPU class.  The design X of the
# Optuna campaign depends on the CPU's vector level (AVX-512 vs AVX2/AVX: numpy's dispatched loops), so
# Hoffman2 output is compared with CARC output of the SAME class.  For every node feature in FEATURES
# (sbatch --constraint), this runs the check's own lines with C's task script unchanged
# (cluster/slurm/optuna_pack.sbatch: the spec call of the fleet) into
#   $X/carc_<tag>_<feature>/stage<1|2>/<arm>/chunks/c<k>/          (X = the check root below)
#   s1  cluster/optuna_h2_xcheck_s1_<tag>.txt  (a few tuning points, every trial)
#   s2  cluster/optuna_h2_xcheck_s2_<tag>.txt  (a refit block, from C's own stage-1 records of the block,
#       copied here from CROOT/stage1/<arm>/chunks/c<k>/... with STAGE1_COMPLETE naming the source + md5)
#   cd /scratch1/jc_905/harxhar-subsection && CROOT=results/linear_subsection_trees_optuna_mask \
#     WINDOW_MASK=1 FEATURES="xeon-4116 epyc-7513" bash cluster/slurm/submit_optuna_h2_xcheck_carc.sh mask
set -euo pipefail
cd "$(dirname "$0")/../.."
TAG=${1:?tag}
: "${CROOT:?CROOT}" "${WINDOW_MASK:?WINDOW_MASK}" "${FEATURES:?FEATURES}"
X=results/linear_subsection_trees_optuna/crosscluster
S1T=cluster/optuna_h2_xcheck_s1_$TAG.txt
S2T=cluster/optuna_h2_xcheck_s2_$TAG.txt
for f in "$S2T" cluster/slurm/optuna_pack.sbatch specs/causal_tune_trees_optuna.py; do  # s1 optional
  [ -f "$f" ] || { echo "$f missing"; exit 1; }
done
mkdir -p logs
STAMP=logs/submitted_optuna_h2_xcheck_carc_$TAG.txt
[ -f "$STAMP" ] && { echo "already submitted ($(cat "$STAMP"))"; exit 1; }
POOL=${POOL:-8}
OUT=""
for F in $FEATURES; do
  R=$X/carc_${TAG}_$F
  while read -r B M TW SEG C S E H; do
    SRC=$CROOT/stage1/$B/$SEG/$M/tw$TW/chunks/c$C
    [ -f "$SRC/DONE" ] || { echo "no DONE $SRC"; exit 1; }
    D=$R/stage1/$B/$SEG/$M/tw$TW
    mkdir -p "$D"
    cp -p "$SRC/causal_tune_trees/$M/$B/trials_$SEG.npz" "$D/trials_$SEG.npz"
    echo "CARC $SRC/causal_tune_trees/$M/$B/trials_$SEG.npz md5 $(md5sum < "$D/trials_$SEG.npz" | cut -c1-32)" > "$D/STAGE1_COMPLETE"
  done < "$S2T"
  J1=-
  if [ -f "$S1T" ]; then  # a stage-2-only tag has no stage-1 lines
    J1=$(sbatch --parsable -J op_xc_s1 --constraint="$F" --array=1-"$(wc -l < "$S1T")" --cpus-per-task="$POOL" \
          --mem=16G --time=2:00:00 --export=ALL,TASKFILE=$S1T,STAGE=tune,RESULTS_ROOT=$R,WINDOW_MASK=$WINDOW_MASK \
          cluster/slurm/optuna_pack.sbatch)
  fi
  J2=$(sbatch --parsable -J op_xc_s2 --constraint="$F" --array=1-"$(wc -l < "$S2T")" --cpus-per-task="$POOL" \
        --mem=16G --time=1:00:00 --export=ALL,TASKFILE=$S2T,STAGE=refit,RESULTS_ROOT=$R,WINDOW_MASK=$WINDOW_MASK \
        cluster/slurm/optuna_pack.sbatch)
  OUT="$OUT $F:s1=$J1,s2=$J2"
done
echo "xcheck carc $TAG (mask $WINDOW_MASK, pool $POOL):$OUT $(date)" | tee "$STAMP"
