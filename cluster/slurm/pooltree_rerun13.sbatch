#!/bin/bash
# Pooled trees on the design of the pooled linear 13-bar rerun (2026-10-20): experiments/pooled_trees_rerun_13.py.
#   MODE walk    (default) line $SLURM_ARRAY_TASK_ID of $TASKFILE, "<est> <set> <refit> <first-day> <last-day> <i>/<n>"
#                -> $OUT/pieces/<est>_<set>_<refit>_<first>_<last>_<i>of<n>.parquet + .json (resumes from its checkpoint)
#   MODE design  the linear runner's design caches (HAR benchmark, all features) -> $DESIGN/_work/design_<set>/
#   MODE time    K refits of each of the 6 arms -> $OUT/timing/sec_refit.csv and the projected hours
#   MODE merge   every piece of the span $FIRST..$LAST and cadence $REFIT -> $OUT/forecasts.parquet
# Env: HARXHAR_PY (required) = the venv's python, built from cluster/requirements-rerun13.lock.txt (on the base branch
# insurance-selling-2026-09-19; the CARC env is Python 3.12.15 with exactly those pins); DESIGN (required) = the
# linear runner's --out directory holding _work/design_{baseline,all_features}/; OUT (default
# results/pooled_trees_rerun13_2026-10-20); WINDOW_DAYS (default 1000); K (time, default 3); REFIT / FIRST / LAST
# (merge; LAST empty = the last design row).
# 2026-10-20 (author, relayed): parked -- the 2,000-day window, DESIGN defaulting to OUT, and the 2020 span first.
# The trees run with the 1,000-day window over 2012-01-03 .. 2024-04-30.
# Submit (CARC Discovery, from the repo root, no conda env active):
#   P=/path/to/venv/bin/python; D=results/pooled_linear_rerun_2026-10-19_13_w1000   # the design directory
#   1. design:  sbatch --job-name=ptr13_design --cpus-per-task=2 --mem=16G --time=1:00:00 \
#                 --export=ALL,MODE=design,HARXHAR_PY=$P,DESIGN=$D,WINDOW_DAYS=1000 cluster/slurm/pooltree_rerun13.sbatch
#      then compare the sha256 of X.npy, y.npy, b_oos.npy, dates_oos.csv in $D/_work/design_{baseline,all_features}/
#      against results/pooled_trees_rerun13_2026-10-20/SUMMARY.md; if an X.npy differs, compare column hashes:
#               $P experiments/pooled_trees_rerun_13.py colsha --set all_features --design $D --out <dir>
#      (against results/pooled_trees_rerun13_2026-10-20/colsha/), or two X.npy files directly:
#               $P experiments/pooled_trees_rerun_13.py compare <X.npy> <X.npy> --names $D/_work/design_all_features/design.json
#   2. time probe (same CPUs as the array):
#               sbatch --job-name=ptr13_time --dependency=afterok:<design job> --cpus-per-task=8 --mem=8G \
#                 --export=ALL,MODE=time,HARXHAR_PY=$P,DESIGN=$D,WINDOW_DAYS=1000 cluster/slurm/pooltree_rerun13.sbatch
#   3. task file (login node; reads only dates_oos.csv and the timing table; arms cheapest first):
#               $P experiments/pooled_trees_rerun_13.py tasks --design $D --window-days 1000 \
#                 --sec-table results/pooled_trees_rerun13_2026-10-20/timing/sec_refit.csv --refits day \
#                 --span 2012-01-03:2024-04-30 --target-hours 3 --taskfile cluster/pooltree_rerun13_tasks.txt
#   4. array:   sbatch --job-name=ptr13 --array=1-$(wc -l < cluster/pooltree_rerun13_tasks.txt)%100 --cpus-per-task=8 \
#                 --mem=8G --export=ALL,MODE=walk,TASKFILE=cluster/pooltree_rerun13_tasks.txt,HARXHAR_PY=$P,DESIGN=$D,WINDOW_DAYS=1000 \
#                 cluster/slurm/pooltree_rerun13.sbatch
#      (a task that hits the time limit resumes from its checkpoint when its index is resubmitted: --array=<ids>)
#   5. merge, once every piece exists (one call for each cadence):
#               sbatch --job-name=ptr13_merge --cpus-per-task=1 --mem=8G --time=1:00:00 \
#                 --export=ALL,MODE=merge,REFIT=day,FIRST=2012-01-03,LAST=2024-04-30,HARXHAR_PY=$P,DESIGN=$D,WINDOW_DAYS=1000 \
#                 cluster/slurm/pooltree_rerun13.sbatch
#      (merge writes $OUT/forecasts.parquet; move it aside before merging the other cadence)
#SBATCH --job-name=ptr13
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x.%A_%a.out
set -eo pipefail
mkdir -p logs
PY="${HARXHAR_PY:?HARXHAR_PY must be the path to the venv python}"
# 2026-10-20 (user): parked -- a venv only; the CARC env is a conda env without bin/activate, and every call below
# runs "$PY" itself. Was: source "$(dirname "$PY")/activate"
if [ -f "$(dirname "$PY")/activate" ]; then source "$(dirname "$PY")/activate"; fi
export PATH="$(dirname "$PY"):$PATH"
set -u
NT="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONNOUSERSITE=1  # the CARC user site shadows numpy and pandas
export OMP_NUM_THREADS=$NT MKL_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT NUMBA_NUM_THREADS=$NT
export PYTHONPATH="$PWD:$PWD/experiments" TQDM_DISABLE=1 PYTHONUNBUFFERED=1
MODE="${MODE:-walk}"
OUT="${OUT:-results/pooled_trees_rerun13_2026-10-20}"
# 2026-10-20 (author): parked -- DESIGN="${DESIGN:-$OUT}" and WD="${WINDOW_DAYS:-2000}"
DESIGN="${DESIGN:?DESIGN must be the design directory, the --out of the linear runner}"
WD="${WINDOW_DAYS:-1000}"
R=experiments/pooled_trees_rerun_13.py
CPU=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')
echo "ptr13 $MODE task ${SLURM_ARRAY_TASK_ID:-} on $(hostname) [$CPU], $NT threads, $("$PY" -V)  $(date)"
case "$MODE" in
  design)
    "$PY" experiments/pooled_linear_rerun_2026_10_19_13.py design --jobs 2 --window-days "$WD" --out "$DESIGN"
    for s in baseline all_features; do
      (cd "$DESIGN/_work/design_$s" && sha256sum X.npy y.npy b_oos.npy dates_oos.csv)
    done ;;
  time)
    "$PY" $R time --design "$DESIGN" --out "$OUT" --window-days "$WD" --k "${K:-3}" --threads "$NT" ;;
  merge)
    "$PY" $R merge --design "$DESIGN" --out "$OUT" --window-days "$WD" --refit "${REFIT:?REFIT}" \
      --first-day "${FIRST:?FIRST}" ${LAST:+--last-day "$LAST"} --require-all ;;
  walk)
    LINE=$(sed -n "${SLURM_ARRAY_TASK_ID:?}p" "${TASKFILE:?TASKFILE}")
    read -r EST SET REFIT FIRST LAST PART <<< "$LINE"
    [ -n "${PART:-}" ] || { echo "malformed line ${SLURM_ARRAY_TASK_ID}: '$LINE'"; exit 1; }
    "$PY" $R walk --design "$DESIGN" --out "$OUT" --window-days "$WD" --est "$EST" --set "$SET" --refit "$REFIT" \
      --first-day "$FIRST" --last-day "$LAST" --part "$PART" --threads "$NT" ;;
  *) echo "unknown MODE '$MODE'"; exit 1 ;;
esac
echo "ptr13 $MODE done $(date)"
