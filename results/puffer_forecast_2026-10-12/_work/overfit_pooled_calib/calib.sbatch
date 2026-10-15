#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --job-name=pf_poolcal
#SBATCH --output=logs/pf_poolcal_%j.out
set -uo pipefail
module purge >/dev/null 2>&1
module load ver/2506 gcc/14.3.0 cuda/12.9.1 >/dev/null 2>&1
export CUDA_DEVICE_MAX_CONNECTIONS=32
ROOT=$HOME/puffer_forecast
RUN=$ROOT/runs/poolcal_$SLURM_JOB_ID
mkdir -p "$RUN"; cd "$RUN"; ln -sfn "$ROOT/data" data
nvidia-smi --query-gpu=name --format=csv,noheader > env.txt
cp "$ROOT/bin/native_g5b/forecast_train" ft
for f in "$ROOT"/inputs/overfit_pooled_calib/c1.txt; do
  while read -r line; do
    id=$(echo "$line" | tr ' ' '\n' | grep '^id=' | cut -d= -f2)
    echo "$line" > one.txt
    a=$(date +%s.%N); ./ft run --jobs one.txt --out "c1_$id" --concurrent 1 --chunk 3 --pool > "c1_$id.log" 2>&1
    echo "$id,1,$(awk -v a="$a" -v b="$(date +%s.%N)" 'BEGIN{printf "%.2f", b - a}')" >> timing.csv
  done < "$f"
done
a=$(date +%s.%N); ./ft run --jobs "$ROOT/inputs/overfit_pooled_calib/c24.txt" --out c24 --concurrent 24 --chunk 3 --pool > c24.log 2>&1
echo "h256_all24,24,$(awk -v a="$a" -v b="$(date +%s.%N)" 'BEGIN{printf "%.2f", b - a}')" >> timing.csv
rm -f ft
cat timing.csv
