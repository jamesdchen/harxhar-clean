#!/bin/bash
# Submit the 8 min-norm OLS bucket arms on H2 (author directive 2026-08-07,
# H2 port of unification_minnorm.sh): single-slot per queue policy (pe
# requests throttle scheduling), idempotent chunk resume.
set -e
cd /u/scratch/j/jamesdc1/harxhar-clean
cat > /tmp/unification_1slot.sge <<'EOF'
#!/bin/bash
#$ -N unif
#$ -cwd
#$ -l h_data=8G,h_rt=12:00:00
#$ -t 1-100
#$ -o /u/scratch/j/jamesdc1/harxhar-clean/logs/
#$ -j y
cd /u/scratch/j/jamesdc1/harxhar-clean
export PYTHONPATH="$PWD:$PWD/experiments${PYTHONPATH:+:$PYTHONPATH}"
export TQDM_DISABLE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/u/home/j/jamesdc1/.conda/envs/hpc-pi/bin/python
OUT=$(printf 'results/unification/%s/chunk_%03d.npz' "$ARM" $((SGE_TASK_ID - 1)))
[ -f "$OUT" ] && { echo "SKIP existing $OUT"; exit 0; }
$PY experiments/run_unification.py --arm "$ARM" --chunk-index "$((SGE_TASK_ID - 1))" \
    --output-dir results/unification
EOF
for arm in a_bucket_moments a_bucket_liquidity a_bucket_market_ew \
           a_bucket_market_vw a_bucket_sentiment a_bucket_implied_vol \
           a_bucket_vol_demand a_bucket_all_features; do
  /u/local/bin/qsub -N "unif_$arm" -v ARM="$arm" /tmp/unification_1slot.sge
done
