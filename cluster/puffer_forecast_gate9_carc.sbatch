#!/bin/bash
# 2026-10-15 puffer_forecast gate 9: early stopping on the validation QLIKE (job key es=qlike, tune jobs) --
# NEW = native_g9/ = native_g8/ plus fc_eval's criterion switch; OLD = native_g8/.  Both prebuilt (P100):
#   cd $HOME/puffer_forecast
#   B=$(sbatch --parsable --export=NONE,SRC=native_g9 puffer_forecast_build_once_carc.sbatch)
#   sbatch --export=NONE,GPU_TYPE=p100 --constraint=p100 --dependency=afterok:$B puffer_forecast_gate9_carc.sbatch
#   identity  OLD vs NEW --pool, es absent: the softplus walk's block-0 tune, 500 refits, the fc_train jobs; es=mse
#             written out = absent
#   es        the block-0 tune with es=qlike (MSE loss; QLIKE loss): best_epoch / best_val against epochs.csv's
#             val_qlike (checked locally); pool vs wave
#   guards    es=foo, es=qlike on a refit, es=qlike under --pack
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=pf_gate9
#SBATCH --output=logs/pf_gate9_%j.out
set -uo pipefail
module purge >/dev/null 2>&1
module load ver/2506 gcc/14.3.0 cuda/12.9.1 >/dev/null 2>&1
export CUDA_DEVICE_MAX_CONNECTIONS=32
ROOT=$HOME/puffer_forecast
OLD=${OLD:-native_g8}
NEW=${NEW:-native_g9}
RUN=$ROOT/runs/gate9_$SLURM_JOB_ID
I=$ROOT/inputs
G9=$I/gate9
WREF=$ROOT/runs/walk_wd0_13135336
mkdir -p "$RUN"; cd "$RUN"; ln -sfn "$ROOT/data" data
{ hostname; date; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  md5sum "$ROOT/$OLD"/*.cu* "$ROOT/$NEW"/*.cu* "$G9"/*; } > env.txt 2>&1
GPU_TYPE=${GPU_TYPE:?"GPU_TYPE is required, with the same --constraint"}
case "$GPU_TYPE" in p100) M="P100" ;; v100) M="V100" ;; *) echo "unknown GPU_TYPE $GPU_TYPE"; exit 1 ;; esac
GOT=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "$GOT" | grep -q "$M" || { echo "got a $GOT, GPU_TYPE=$GPU_TYPE required"; exit 1; }
echo "$GPU_TYPE|$GOT" | tee gpu_type.txt >> env.txt
prebuilt() {
  local b=$ROOT/bin/$1
  [ -x "$b/forecast_train" ] || { echo "no prebuilt $b/forecast_train"; exit 1; }
  grep -q "$(md5sum < "$ROOT/$1/forecast_train.cu" | cut -c1-32)" "$b/provenance.txt" || { echo "stale $b"; exit 1; }
  cp "$b/forecast_train" "$2"
  { echo "prebuilt $1:"; cat "$b/provenance.txt"; } >> env.txt
}
prebuilt "$OLD" ft_old
prebuilt "$NEW" ft_new
awk 'NR <= 500' "$WREF/block0/jobs_refit.txt" > refit500.txt
TUNE=$I/walk_softplus/tune_block0.txt
FCT=$I/overfit_gate/fctrain_jobs.txt
sed 's/$/ es=mse/' "$TUNE" > tune_off.txt
echo "name,rc,t0,t1" > rc.csv
R() { local n=$1 a; shift; a=$(date +%s); "$@" > "$n.log" 2>&1; echo "$n,$?,$a,$(date +%s)" >> rc.csv; }
for v in old new; do
  R id_${v}_tune ./ft_$v run --jobs "$TUNE" --out id_${v}_tune --concurrent 48 --chunk 5 --pool
  R id_${v}_refit ./ft_$v run --jobs refit500.txt --out id_${v}_refit --concurrent 48 --chunk 5 --pool
  R id_${v}_fct ./ft_$v run --jobs "$FCT" --out id_${v}_fct --concurrent 6 --chunk 5 --pool
done
R id_new_off_tune ./ft_new run --jobs tune_off.txt --out id_new_off_tune --concurrent 48 --chunk 5 --pool
R es_tune ./ft_new run --jobs "$G9/tune_es.txt" --out es_tune --concurrent 48 --chunk 5 --pool
R es_wave_tune ./ft_new run --jobs "$G9/tune_es.txt" --out es_wave_tune --concurrent 48 --chunk 5
R es_qlike_tune ./ft_new run --jobs "$G9/tune_qlike_es.txt" --out es_qlike_tune --concurrent 48 --chunk 5 --pool
echo "case,rc,last_line" > guards.csv
for c in bad_es bad_es_refit; do
  ./ft_new run --jobs "$G9/$c.txt" --out g_$c --pool > g_$c.log 2>&1
  echo "$c,$?,$(tail -1 g_$c.log)" >> guards.csv
done
./ft_new run --jobs "$G9/es_pack.txt" --out g_es_pack --pack 2 > g_es_pack.log 2>&1
echo "es_pack,$?,$(tail -1 g_es_pack.log)" >> guards.csv
./ft_new run --jobs "$G9/ok_small.txt" --out g_ok --pool > g_ok.log 2>&1
echo "ok_small,$?,$(tail -1 g_ok.log)" >> guards.csv
gzip -f */steps.csv 2>/dev/null
rm -f ft_old ft_new
date; echo GATE9_DONE
