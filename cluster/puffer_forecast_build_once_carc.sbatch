#!/bin/bash
# 2026-10-13 puffer_forecast: build the native trainer ONCE per submission (CPU node), for the GPU jobs that
# depend on it (speed review: no 5-arch rebuild in every array task, and no node-/tmp nvlink flake there).
# Output: $ROOT/bin/$SRC/forecast_train + build.log + provenance.txt (source / substrate / binary md5s).
#   cd $HOME/puffer_forecast && B=$(sbatch --parsable --export=NONE,SRC=native_g4 puffer_forecast_build_once_carc.sbatch)
#   sbatch --export=NONE,SRC=native_g4 --dependency=afterok:$B --array=0-10 puffer_forecast_overfit_carc.sbatch
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --job-name=pf_build
#SBATCH --output=logs/pf_build_%j.out
set -euo pipefail
module purge >/dev/null 2>&1
module load ver/2506 gcc/14.3.0 cuda/12.9.1 >/dev/null 2>&1
SRC=${SRC:?SRC=<source dir under $HOME/puffer_forecast>}
ROOT=$HOME/puffer_forecast
PUFFER=$HOME/src/PufferLib5
COMMIT=6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2
OUT=$ROOT/bin/$SRC
mkdir -p "$OUT"
[ "$(git -C "$PUFFER" rev-parse HEAD)" = "$COMMIT" ] || { echo "PufferLib checkout is not $COMMIT"; exit 1; }
git -C "$PUFFER" diff --quiet -- src || { echo "PufferLib src/ modified"; exit 1; }
export TMPDIR=$OUT/tmp; mkdir -p "$TMPDIR"
ARCHS="-gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90"  # sm_90 + PTX: gate 5
build() {
  ( cd "$ROOT/$SRC" && nvcc -O2 --threads 4 -Wno-deprecated-gpu-targets -std=c++17 $ARCHS -DPRECISION_FLOAT \
    -I. -I"$PUFFER/src" -I"$PUFFER/vendor" -I"$CUDA_HOME/include" \
    -Xcompiler=-Wno-narrowing --diag-suppress=2361 --diag-suppress=111 --diag-suppress=128 --diag-suppress=177 \
    --diag-suppress=550 forecast_train.cu -L"$CUDA_HOME/lib64" -lcublas -lcurand -o "$OUT/forecast_train.new" )
}
a=$(date +%s)
build > "$OUT/build.log" 2>&1 || build >> "$OUT/build.log" 2>&1
BUILD_S=$(( $(date +%s) - a ))  # the per-task nvcc time this job saves (gate 4 item 4)
mv -f "$OUT/forecast_train.new" "$OUT/forecast_train"
{ hostname; date; nvcc --version | tail -1; gcc --version | head -1; git -C "$PUFFER" log -1 --format='%H %ci'
  echo "ARCHS $ARCHS"; echo "build_s $BUILD_S"
  md5sum "$ROOT/$SRC"/*.cu* "$OUT/forecast_train"; } > "$OUT/provenance.txt" 2>&1
rm -rf "$TMPDIR"
cat "$OUT/provenance.txt"
echo BUILD_DONE
