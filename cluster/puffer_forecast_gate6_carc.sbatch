#!/bin/bash
# 2026-10-14 puffer_forecast gate 6: decoupled weight decay (job key wd) -- NEW = native_g6/ = native_g5b/ plus
# W <- W (1 - lr_t wd) before each Muon step (wd = 0 launches nothing).  OLD = native_g5b/.  Both prebuilt:
#   cd $HOME/puffer_forecast
#   B=$(sbatch --parsable --export=NONE,SRC=native_g6 puffer_forecast_build_once_carc.sbatch)
#   sbatch --export=NONE --dependency=afterok:$B puffer_forecast_gate6_carc.sbatch
# One GPU, CUDA_DEVICE_MAX_CONNECTIONS=32:
#   identity  OLD vs NEW --pool (wd absent = 0): Phase 2c block-0 tune, 500 block-0 refits, the fc_train jobs;
#             NEW pool vs wave
#   guards    wd < 0, wd = nan, wd > 0 under --pack exit 2
#   one step  batch >= N (one Muon step), wd 0 and > 0, + gradcheck dumps of the same step (compared locally with
#             a float64 Muon step of the decayed initial weights)
#   norms     a toy job, 40 epochs, wd 0 .. 10 (weight norms must fall with wd)
#   calib     block-0 window, both designs, 4 configurations x wd {0 .. 3}, 20 epochs: weight norms
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --job-name=pf_gate6
#SBATCH --output=logs/pf_gate6_%j.out
set -uo pipefail
module purge >/dev/null 2>&1
module load ver/2506 gcc/14.3.0 cuda/12.9.1 >/dev/null 2>&1
export CUDA_DEVICE_MAX_CONNECTIONS=32  # gate 4 (2026-10-13): 32 hardware queues for the 2 streams per model (byte-identical)
ROOT=$HOME/puffer_forecast
OLD=${OLD:-native_g5b}
NEW=${NEW:-native_g6}
RUN=$ROOT/runs/gate6_$SLURM_JOB_ID
I=$ROOT/inputs
G=$I/gate4
G6=$I/gate6
mkdir -p "$RUN"; cd "$RUN"; ln -sfn "$ROOT/data" data
{ hostname; date; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  md5sum "$ROOT/$OLD"/*.cu* "$ROOT/$NEW"/*.cu* "$G6"/* "$I"/wd_calib/calib.txt; } > env.txt 2>&1
cat env.txt
# the trainer, built ONCE per submission by puffer_forecast_build_once_carc.sbatch (gate 4, 2026-10-13), not in
# this job
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
awk 'NR <= 500' "$ROOT/runs/phase2c_13082282/block0/jobs_refit.txt" > refit500.txt
TUNE=$I/phase2c/tune_block0.txt
FCT=$I/overfit_gate/fctrain_jobs.txt
echo "name,rc" > rc.csv
R() { local n=$1; shift; "$@" > "$n.log" 2>&1; echo "$n,$?" >> rc.csv; }
# ---- identity
for v in old new; do
  R id_${v}_tune ./ft_$v run --jobs "$TUNE" --out id_${v}_tune --concurrent 48 --chunk 5 --pool
  R id_${v}_refit ./ft_$v run --jobs refit500.txt --out id_${v}_refit --concurrent 48 --chunk 5 --pool
  R id_${v}_fct ./ft_$v run --jobs "$FCT" --out id_${v}_fct --concurrent 6 --chunk 5 --pool
done
R id_new_wave_tune ./ft_new run --jobs "$TUNE" --out id_new_wave_tune --concurrent 48 --chunk 5
R id_new_wave_refit ./ft_new run --jobs refit500.txt --out id_new_wave_refit --concurrent 48 --chunk 5
# ---- guards
echo "case,rc,last_line" > guards.csv
for f in "$G6"/bad_wd_*.txt; do
  b=$(basename "$f" .txt)
  ./ft_new run --jobs "$f" --out g_$b --pool > g_$b.log 2>&1
  echo "$b,$?,$(tail -1 g_$b.log)" >> guards.csv
done
./ft_new run --jobs "$G6/wd_pack.txt" --out g_wd_pack --pack 2 > g_wd_pack.log 2>&1
echo "wd_pack,$?,$(tail -1 g_wd_pack.log)" >> guards.csv
./ft_new run --jobs "$G/ok_small.txt" --out g_ok --pool > g_ok.log 2>&1
echo "ok_small,$?,$(tail -1 g_ok.log)" >> guards.csv
# ---- one step, norms, calibration
R step ./ft_new run --jobs "$G6/step.txt" --out step --concurrent 4 --chunk 1 --pool
mkdir -p gradcheck
R gradcheck ./ft_new gradcheck --jobs "$G6/step_grad.txt" --out gradcheck
R norm ./ft_new run --jobs "$G6/norm_toy.txt" --out norm --concurrent 7 --chunk 5 --pool
R calib ./ft_new run --jobs "$I/wd_calib/calib.txt" --out calib --concurrent 48 --chunk 5 --pool
gzip -f id_*/steps.csv step/steps.csv norm/steps.csv calib/steps.csv 2>/dev/null
rm -f ft_old ft_new gradcheck/*.obs.f32 gradcheck/*.dec.f32
date; echo GATE6_DONE
