#!/bin/bash
# Pull the cluster inputs of the dense-vs-sparse re-run (experiments/dvs_dedup_1530.py,
# checklist I8) into the study's work dir ($DVS_WORK; never committed), never into another
# campaign's result root.  Two sets (WHAT):
#   WHAT=t10  (default) the stored UNMASKED T10 tree runs on the de-duplicated design (rung T10
#             of the 2026-09-29 campaign: shipped configuration, refit every 10 sessions,
#             TreeSHAP at every refit; CARC root results/linear_subsection_trees_dedup/t10):
#             the six trees_bar1600.npz of live_feasible / all_features x LightGBM, XGBoost,
#             random forest -> $DVS_WORK/trees_t10/<bucket>/<model>/trees_bar1600.npz
#             (the density table's "de-dup, no mask" rows);
#   WHAT=mask this study's own MASKED T10 walk with TreeSHAP (cluster/slurm/dvs_dedup.sbatch;
#             CARC dir results/dense_vs_sparse_dedup/_work_carc, after its DONE):
#             trees_mask/<bucket>/<model>.npz and the canary / fleet logs ->
#             $DVS_WORK/trees_mask_carc/ (the density table's "de-dup + mask" rows);
#   WHAT=gate the no-mask gate (DVS_MODE=gate): block 5 of live_feasible without the mask
#             (chunks/live_feasible/<model>_nomask_all/b5.npz), gate_nomask.json and its log
#             -> $DVS_WORK/trees_mask_carc/;
#   WHAT=sacct the Slurm accounting of this study's jobs -> $DVS_WORK/trees_mask_carc/sacct.psv.
# The pull is a tar over native Windows OpenSSH; it FAILS unless every pulled file's local
# md5 equals its remote md5.  Reads only; submits nothing.
# Run from Git Bash locally:
#   DVS_WORK=<work dir> [WHAT=mask|gate|sacct] bash cluster/slurm/pull_dvs_dedup_carc.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
REMOTE=/scratch1/jc_905/harxhar-subsection
WORKDIR=${DVS_WORK:-results/dense_vs_sparse_dedup/_work}
WHAT=${WHAT:-t10}

sums() { awk '{p = $2; sub(/^\*/, "", p); print p, $1}' | sort; }

FILES=()
if [ "$WHAT" = t10 ]; then
  R=results/linear_subsection_trees_dedup/t10
  DEST="$WORKDIR/trees_t10"
  for b in live_feasible all_features; do
    for m in lgbm xgb rf; do
      FILES+=("$R/$b/bar1600/$m/tw2000/causal_tune_trees/$m/$b/trees_bar1600.npz")
    done
  done
  for f in "${FILES[@]}"; do
    d=$(dirname "$f")
    "$SSH" -o BatchMode=yes usc-discovery "test -f $REMOTE/$d/MERGED && test -f $REMOTE/$f" \
      || { echo "not merged on CARC yet: $f"; exit 1; }
  done
elif [ "$WHAT" = mask ]; then
  R=results/dense_vs_sparse_dedup/_work_carc
  DEST="$WORKDIR/trees_mask_carc"
  "$SSH" -o BatchMode=yes usc-discovery "test -f $REMOTE/$R/DONE" \
    || { echo "no $R/DONE on CARC yet"; exit 1; }
  for b in live_feasible all_features; do
    for m in lgbm xgb rf; do
      FILES+=("$R/trees_mask/$b/$m.npz")
    done
  done
  FILES+=("$R/carc_log_canary.json" "$R/carc_log_fleet.json" "$R/DONE" "$R/submitted.txt")
elif [ "$WHAT" = gate ]; then
  R=results/dense_vs_sparse_dedup/_work_carc
  DEST="$WORKDIR/trees_mask_carc"
  "$SSH" -o BatchMode=yes usc-discovery "test -f $REMOTE/$R/gate_nomask.json" \
    || { echo "no $R/gate_nomask.json on CARC yet"; exit 1; }
  for m in lgbm xgb rf; do
    FILES+=("$R/chunks/live_feasible/${m}_nomask_all/b5.npz")
  done
  FILES+=("$R/gate_nomask.json" "$R/carc_log_gate.json" "$R/submitted.txt")
elif [ "$WHAT" = sacct ]; then
  # the accounting of every job this study submitted (ids from its submitted.txt)
  R=results/dense_vs_sparse_dedup/_work_carc
  DEST="$WORKDIR/trees_mask_carc"
  mkdir -p "$DEST"
  "$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && ids=\$(grep -oE '(canary|fleet|gate) [0-9]+' $R/submitted.txt | awk '{print \$2}' | paste -sd, -) && sacct -j \$ids -X -P -o JobID,JobName,Partition,State,Elapsed,AllocCPUS,CPUTimeRAW,ElapsedRaw,Start,End" \
    | tr -d '\r' | grep -v -e 'have been reloaded' -e 'python/3' -e '^$' > "$DEST/sacct.psv"
  echo "wrote $DEST/sacct.psv ($(($(wc -l < "$DEST/sacct.psv") - 1)) jobs)"
  exit 0
else
  echo "WHAT must be t10, mask, gate or sacct, got $WHAT"; exit 1
fi
TMP="$WORKDIR/pull_${WHAT}.tmp"
RM=$("$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && md5sum ${FILES[*]}" | tr -d '\r' | sums)
rm -rf "$TMP" && mkdir -p "$TMP"
"$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && tar cf - ${FILES[*]}" | tar xf - -C "$TMP"
L=$(cd "$TMP" && md5sum "${FILES[@]}" | sums)
[ "$L" = "$RM" ] || { echo "PULL FAILED: md5 mismatch"; diff <(echo "$L") <(echo "$RM") || true; exit 1; }
for f in "${FILES[@]}"; do
  if [ "$WHAT" = t10 ]; then
    b=$(echo "$f" | cut -d/ -f4)
    m=$(echo "$f" | cut -d/ -f6)
    mkdir -p "$DEST/$b/$m"
    mv "$TMP/$f" "$DEST/$b/$m/trees_bar1600.npz"
  else
    rel=${f#"$R"/}
    rel=${rel#trees_mask/}
    rel=${rel#chunks/live_feasible/}
    mkdir -p "$DEST/$(dirname "$rel")"
    mv "$TMP/$f" "$DEST/$rel"
  fi
done
rm -rf "$TMP"
printf '%s\n' "$RM" > "$DEST/md5_remote.txt"
echo "pulled ${#FILES[@]} files into $DEST (md5 match)"
