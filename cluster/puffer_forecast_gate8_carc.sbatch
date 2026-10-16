#!/bin/bash
# 2026-10-15 puffer_forecast gate 8: the linear shortcut (job keys lin / lr_lin / wd_lin / lin_init) and the
# zero-initialised decoder (dec_zero) -- NEW = native_g8/ = native_g7/ plus f = link(c + s [o + v.x]), v by its
# own Adam (wd_lin an L2 penalty under Adam, pf as wd_enc's); OLD = native_g7/.  Both prebuilt (P100):
#   cd $HOME/puffer_forecast
#   B=$(sbatch --parsable --export=NONE,SRC=native_g8 puffer_forecast_build_once_carc.sbatch)
#   sbatch --export=NONE,GPU_TYPE=p100 --constraint=p100 --dependency=afterok:$B puffer_forecast_gate8_carc.sbatch
# One GPU, CUDA_DEVICE_MAX_CONNECTIONS=32:
#   identity  OLD vs NEW --pool, everything off: the softplus walk's block-0 tune (120 jobs), 500 block-0 refits,
#             the fc_train jobs; NEW pool vs wave; NEW with lin=0 dec_zero=0 written out
#   grad      gradcheck dumps (dL/dv, o, o', obs, y) for mse / huber / qlike x identity / softplus at v0 != 0
#   step      one full-batch step (lin, lin + dec_zero) and 30 full-batch steps of v alone (branch frozen)
#   ridge     the shortcut alone to convergence (3000 full-batch epochs) on the stack's block-1 window
#   reset     four same-shape lin jobs back to back in one slot vs each alone
#   guards    lin / dec_zero / lr_lin / wd_lin / lin_init misuse, --pack
#   calib     lr_lin x epochs x wd_lin, the shortcut alone (refits: distance to the ridge; tunes: best epochs)
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=pf_gate8
#SBATCH --output=logs/pf_gate8_%j.out
set -uo pipefail
module purge >/dev/null 2>&1
module load ver/2506 gcc/14.3.0 cuda/12.9.1 >/dev/null 2>&1
export CUDA_DEVICE_MAX_CONNECTIONS=32  # gate 4 (2026-10-13): 32 hardware queues for the 2 streams per model (byte-identical)
ROOT=$HOME/puffer_forecast
OLD=${OLD:-native_g7}
NEW=${NEW:-native_g8}
RUN=$ROOT/runs/gate8_$SLURM_JOB_ID
I=$ROOT/inputs
G8=$I/gate8
WREF=$ROOT/runs/walk_wd0_13135336  # the wd study's wd = 0 softplus walk (native_g6, P100)
mkdir -p "$RUN"; cd "$RUN"; ln -sfn "$ROOT/data" data
cp "$G8"/*.f32 .
{ hostname; date; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  md5sum "$ROOT/$OLD"/*.cu* "$ROOT/$NEW"/*.cu* "$G8"/* "$I"/lin_calib/*.txt "$ROOT"/data/*/enc_pf.f64; } > env.txt 2>&1
GPU_TYPE=${GPU_TYPE:?"GPU_TYPE is required, with the same --constraint"}
case "$GPU_TYPE" in p100) M="P100" ;; v100) M="V100" ;; a40) M="A40" ;; a100) M="A100" ;; l40s) M="L40S" ;;
  *) echo "unknown GPU_TYPE $GPU_TYPE"; exit 1 ;; esac
GOT=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "$GOT" | grep -q "$M" || { echo "got a $GOT, GPU_TYPE=$GPU_TYPE required"; exit 1; }
echo "$GPU_TYPE|$GOT" | tee gpu_type.txt >> env.txt
cat env.txt
prebuilt() {  # prebuilt SRC_DIR DEST: $ROOT/bin/SRC_DIR/forecast_train, checked against the current source
  local b=$ROOT/bin/$1
  [ -x "$b/forecast_train" ] || { echo "no prebuilt $b/forecast_train (puffer_forecast_build_once_carc.sbatch, SRC=$1)"; exit 1; }
  grep -q "$(md5sum < "$ROOT/$1/forecast_train.cu" | cut -c1-32)" "$b/provenance.txt" \
    || { echo "prebuilt $b is stale: $ROOT/$1/forecast_train.cu changed since it was built"; exit 1; }
  cp "$b/forecast_train" "$2"
  { echo "prebuilt $1:"; cat "$b/provenance.txt"; } >> env.txt
}
prebuilt "$OLD" ft_old
prebuilt "$NEW" ft_new
awk 'NR <= 500' "$WREF/block0/jobs_refit.txt" > refit500.txt
TUNE=$I/walk_softplus/tune_block0.txt
FCT=$I/overfit_gate/fctrain_jobs.txt
sed 's/$/ lin=0 dec_zero=0/' "$TUNE" > tune_off.txt
echo "name,rc,t0,t1" > rc.csv
R() { local n=$1 a; shift; a=$(date +%s); "$@" > "$n.log" 2>&1; echo "$n,$?,$a,$(date +%s)" >> rc.csv; }
# ---- identity
for v in old new; do
  R id_${v}_tune ./ft_$v run --jobs "$TUNE" --out id_${v}_tune --concurrent 48 --chunk 5 --pool
  R id_${v}_refit ./ft_$v run --jobs refit500.txt --out id_${v}_refit --concurrent 48 --chunk 5 --pool
  R id_${v}_fct ./ft_$v run --jobs "$FCT" --out id_${v}_fct --concurrent 6 --chunk 5 --pool
done
R id_new_wave_tune ./ft_new run --jobs "$TUNE" --out id_new_wave_tune --concurrent 48 --chunk 5
R id_new_wave_refit ./ft_new run --jobs refit500.txt --out id_new_wave_refit --concurrent 48 --chunk 5
R id_new_off_tune ./ft_new run --jobs tune_off.txt --out id_new_off_tune --concurrent 48 --chunk 5 --pool
# ---- guards
echo "case,rc,last_line" > guards.csv
for f in "$G8"/bad_*.txt; do
  b=$(basename "$f" .txt)
  ./ft_new run --jobs "$f" --out g_$b --pool > g_$b.log 2>&1
  echo "$b,$?,$(tail -1 g_$b.log)" >> guards.csv
done
for c in lin_pack dz_pack; do
  ./ft_new run --jobs "$G8/$c.txt" --out g_$c --pack 2 > g_$c.log 2>&1
  echo "$c,$?,$(tail -1 g_$c.log)" >> guards.csv
done
./ft_new run --jobs "$G8/ok_small.txt" --out g_ok --pool > g_ok.log 2>&1
echo "ok_small,$?,$(tail -1 g_ok.log)" >> guards.csv
# ---- gradient dumps, steps, ridge, pool reset
mkdir -p gradcheck
R gradcheck ./ft_new gradcheck --jobs "$G8/grad.txt" --out gradcheck
R step ./ft_new run --jobs "$G8/step.txt" --out step --concurrent 8 --chunk 5 --pool
R ridge ./ft_new run --jobs "$G8/ridge.txt" --out ridge --concurrent 8 --chunk 50 --pool
R reset_pool ./ft_new run --jobs "$G8/reset.txt" --out reset_pool --concurrent 1 --chunk 5 --pool
for i in 0 1 2 3; do
  R reset_alone_$i ./ft_new run --jobs "$G8/reset_alone_$i.txt" --out reset_alone_$i --concurrent 1 --chunk 5 --pool
done
# ---- calibration
R calib_refit ./ft_new run --jobs "$I/lin_calib/calib_refit.txt" --out calib_refit --concurrent 48 --chunk 5 --pool
R calib_tune ./ft_new run --jobs "$I/lin_calib/calib_tune.txt" --out calib_tune --concurrent 48 --chunk 5 --pool
gzip -f id_*/steps.csv step/steps.csv ridge/steps.csv reset_*/steps.csv calib_*/steps.csv 2>/dev/null
rm -f ft_old ft_new gradcheck/*.dec.f32.unused
date; echo GATE8_DONE
