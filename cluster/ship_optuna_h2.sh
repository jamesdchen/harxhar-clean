#!/bin/bash
# I5 (2026-09-29): fill the Hoffman2 deployment root of the Optuna cross-cluster work with EXACTLY the
# code the CARC fleet runs -- relayed from the CARC root itself (CARC -> this machine -> Hoffman2;
# a direct ssh-to-ssh pipe truncates here), then FAIL unless every relayed file's Hoffman2 md5
# equals its CARC md5 (the local working copy's md5 is printed beside them, for the record) -- plus
# this campaign's own Hoffman2 scripts (local -> Hoffman2, md5), and the data: data/*.parquet copied
# on Hoffman2 from the I1 deployment root where its md5 equals CARC's (FAIL otherwise).
# With XTAG set, also stages CARC's stage-1 chunk records the check's stage-2 lines start from:
#   CARC $CROOT/stage1/<arm>/chunks/c<chunk>/causal_tune_trees/<model>/<bucket>/trials_<seg>.npz
#   -> Hoffman2 $XROOT/stage1/<arm>/trials_<seg>.npz (+ STAGE1_COMPLETE), md5 == CARC's, where
#   XROOT = results/linear_subsection_trees_optuna/crosscluster/h2_<XTAG> (every tag's check lives there).
# The runtime (CARC's Python stack) is relayed once by the RUNTIME=1 branch and built on Hoffman2 by
# cluster/optuna_h2_runtime_setup.sh.
#
#   CROOT=results/linear_subsection_trees_optuna XTAG=nomask bash cluster/ship_optuna_h2.sh
set -euo pipefail
cd "$(dirname "$0")/.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
CARC=/scratch1/jc_905/harxhar-subsection
H2=/u/scratch/j/jamesdc1/harxhar-optuna
H2DATA=/u/scratch/j/jamesdc1/harxhar-dedup   # the I1 root (A's ship: data md5 == local)
CROOT=${CROOT:?CROOT = C results root on CARC, e.g. results/linear_subsection_trees_optuna_mask}
STAGE_DIR=${STAGE_DIR:-${TMPDIR:-/tmp}/ship_optuna_h2}
mkdir -p "$STAGE_DIR"
sums() { awk '{p = $2; sub(/^\*/, "", p); print p, $1}' | sort; }

if [ "${RUNTIME:-0}" = 1 ]; then  # once: CARC's env, user site, ./pylib_trees and glibc
  "$SSH" -o BatchMode=yes usc-discovery "cd /home1/jc_905/.conda/envs && tar czf - --exclude=harxhar/lib/python3.11/site-packages/torch --exclude=harxhar/lib/python3.11/site-packages/nvidia --exclude=harxhar/lib/python3.11/site-packages/triton --exclude='*/__pycache__' harxhar" > "$STAGE_DIR/env.tgz"
  "$SSH" -o BatchMode=yes usc-discovery "cd /home1/jc_905/.local/lib/python3.11 && tar czf - --exclude='*/__pycache__' site-packages" > "$STAGE_DIR/usersite.tgz"
  "$SSH" -o BatchMode=yes usc-discovery "cd $CARC && tar czf - pylib_trees" > "$STAGE_DIR/pylib_trees.tgz"
  "$SSH" -o BatchMode=yes usc-discovery 'cd /usr/lib64 && tar czf - $(rpm -ql glibc.x86_64 | grep -E "^/usr/lib64/[^/]+\.so" | xargs -n1 basename) libcrypt.so.1*' > "$STAGE_DIR/carc_sys.tgz"
  ( cd "$STAGE_DIR" && md5sum env.tgz usersite.tgz pylib_trees.tgz carc_sys.tgz > tgz.md5 )
  ( cd "$STAGE_DIR" && tar cf - env.tgz usersite.tgz pylib_trees.tgz carc_sys.tgz tgz.md5 ) \
    | "$SSH" -o BatchMode=yes hoffman2 "mkdir -p $H2/runtime_tgz && cd $H2/runtime_tgz && tar xf - && md5sum -c tgz.md5"
fi

# 1. the fleet's code, from the CARC root
mapfile -t CODE < <("$SSH" -o BatchMode=yes usc-discovery "cd $CARC && { find src -name '*.py' -not -path '*/__pycache__/*';
  ls specs/causal_tune_trees_optuna.py specs/causal_tune_trees_optuna_jobs.py specs/causal_tune_trees_tuned_jobs.py \
     specs/causal_tune_trees.py specs/causal_tune_linear.py experiments/reduce_trees_optuna.py \
     cluster/slurm/optuna*.sbatch cluster/slurm/submit_optuna*.sh cluster/optuna*tasks*.txt 2>/dev/null; } | sort" 2>/dev/null | tr -d '\r')
[ "${#CODE[@]}" -gt 20 ] || { echo "CARC code list too short (${#CODE[@]})"; exit 1; }
printf '%s\n' "${CODE[@]}" > "$STAGE_DIR/code.list"
"$SSH" -o BatchMode=yes usc-discovery "cd $CARC && tar czf - -T -" < "$STAGE_DIR/code.list" > "$STAGE_DIR/code.tgz" 2>/dev/null
"$SSH" -o BatchMode=yes hoffman2 "mkdir -p $H2 && cd $H2 && tar xzf -" < "$STAGE_DIR/code.tgz"
CARC_SUMS=$("$SSH" -o BatchMode=yes usc-discovery "cd $CARC && xargs -d '\n' md5sum" < "$STAGE_DIR/code.list" 2>/dev/null | sums)
H2_SUMS=$("$SSH" -o BatchMode=yes hoffman2 "cd $H2 && xargs -d '\n' md5sum" < "$STAGE_DIR/code.list" | sums)
if [ "$CARC_SUMS" != "$H2_SUMS" ]; then
  echo "SHIP FAILED: Hoffman2 md5 != CARC md5:"; diff <(printf '%s\n' "$CARC_SUMS") <(printf '%s\n' "$H2_SUMS") | head; exit 1
fi
LOCAL_SUMS=$(while read -r f; do if [ -f "$f" ]; then md5sum "$f"; else echo "absent *$f"; fi; done < "$STAGE_DIR/code.list" | sums)
N_LOCAL_SAME=$(join <(printf '%s\n' "$CARC_SUMS") <(printf '%s\n' "$LOCAL_SUMS") | awk '$2 == $3' | wc -l)
echo "code: ${#CODE[@]} files, Hoffman2 md5 == CARC md5; local working copy == CARC for $N_LOCAL_SAME"
join <(printf '%s\n' "$CARC_SUMS") <(printf '%s\n' "$LOCAL_SUMS") | awk '$2 != $3 {print "  local differs from CARC (not shipped from local): " $1}'
for f in specs/causal_tune_trees_optuna.py specs/causal_tune_trees_optuna_jobs.py specs/causal_tune_trees_tuned_jobs.py src/models/window_mask.py src/backtest/executor.py; do
  printf '%s\n' "$CARC_SUMS" | awk -v f="$f" '$1 == f {printf "  %s  %s\n", substr($2, 1, 12), $1}'
done

# 2. this campaign's Hoffman2 scripts, from the local working copy
MINE=(cluster/optuna_h2_runtime_setup.sh cluster/optuna_h2_task.sh cluster/optuna_h2_probe.sh cluster/submit_optuna_h2.sh
      experiments/optuna_h2_design_probe.py)
mapfile -t XT < <(ls cluster/optuna_h2_xcheck_*.txt cluster/optuna_h2_offload_*.txt 2>/dev/null)
MINE+=("${XT[@]}")
tar czf - "${MINE[@]}" | "$SSH" -o BatchMode=yes hoffman2 "cd $H2 && tar xzf -"
L=$(md5sum "${MINE[@]}" | sums)
H=$("$SSH" -o BatchMode=yes hoffman2 "cd $H2 && md5sum ${MINE[*]}" | sums)
[ "$L" = "$H" ] || { echo "SHIP FAILED: own scripts md5"; exit 1; }
echo "own scripts: ${#MINE[@]} files, Hoffman2 md5 == local"

# 3. data: CARC's data/*.parquet, copied on Hoffman2 from the I1 root where the md5 matches
CD=$("$SSH" -o BatchMode=yes usc-discovery "cd $CARC && md5sum data/*.parquet" 2>/dev/null | sums)
"$SSH" -o BatchMode=yes hoffman2 "mkdir -p $H2/data"
while read -r f m; do
  "$SSH" -n -o BatchMode=yes hoffman2 "cd $H2 && { [ -f $f ] && [ \"\$(md5sum < $f | cut -c1-32)\" = $m ]; } || { [ \"\$(md5sum < $H2DATA/$f | cut -c1-32)\" = $m ] && cp -p $H2DATA/$f $f; }" \
    || { echo "SHIP FAILED: no Hoffman2 copy of $f with CARC's md5 $m"; exit 1; }
done <<< "$CD"
HD=$("$SSH" -o BatchMode=yes hoffman2 "cd $H2 && md5sum data/*.parquet" | sums)
[ "$CD" = "$HD" ] || { echo "SHIP FAILED: data md5"; diff <(printf '%s\n' "$CD") <(printf '%s\n' "$HD"); exit 1; }
echo "data: $(printf '%s\n' "$CD" | wc -l) parquet files, Hoffman2 md5 == CARC md5"

# 3b. the check's CARC-side files (the design probe's twin), local -> CARC, md5
CMINE=(experiments/optuna_h2_design_probe.py cluster/slurm/optuna_h2_probe_carc.sbatch cluster/slurm/submit_optuna_h2_probe.sh)
mapfile -t CXT < <(ls cluster/optuna_h2_xcheck_s2_*.txt cluster/optuna_h2_xcheck_probe_*.txt cluster/optuna_h2_xcheck_archprobe_*.txt 2>/dev/null)
CMINE+=("${CXT[@]}")
tar czf - "${CMINE[@]}" | "$SSH" -o BatchMode=yes usc-discovery "cd $CARC && tar xzf -" 2>/dev/null
L=$(md5sum "${CMINE[@]}" | sums)
H=$("$SSH" -o BatchMode=yes usc-discovery "cd $CARC && md5sum ${CMINE[*]}" 2>/dev/null | sums)
[ "$L" = "$H" ] || { echo "SHIP FAILED: CARC-side check files md5"; exit 1; }
echo "check files on CARC: ${#CMINE[@]}, md5 == local"

# 4. the check's stage-1 records (CARC's own chunk records) for each xcheck stage-2 line
if [ -n "${XTAG:-}" ]; then
  XROOT=results/linear_subsection_trees_optuna/crosscluster/h2_$XTAG
  while read -r B M TW SEG C S E HALO; do
    SRC="$CROOT/stage1/$B/$SEG/$M/tw$TW/chunks/c$C/causal_tune_trees/$M/$B/trials_$SEG.npz"
    DST="$XROOT/stage1/$B/$SEG/$M/tw$TW"
    "$SSH" -n -o BatchMode=yes usc-discovery "cd $CARC && test -f $CROOT/stage1/$B/$SEG/$M/tw$TW/chunks/c$C/DONE && cat $SRC" \
      > "$STAGE_DIR/rec.npz" 2>/dev/null || { echo "no DONE CARC chunk records $SRC"; exit 1; }
    m1=$("$SSH" -n -o BatchMode=yes usc-discovery "cd $CARC && md5sum < $SRC" 2>/dev/null | cut -c1-32)
    "$SSH" -o BatchMode=yes hoffman2 "mkdir -p $H2/$DST && cat > $H2/$DST/trials_$SEG.npz" < "$STAGE_DIR/rec.npz"
    m2=$("$SSH" -n -o BatchMode=yes hoffman2 "md5sum < $H2/$DST/trials_$SEG.npz" | cut -c1-32)
    [ "$m1" = "$m2" ] || { echo "SHIP FAILED: records md5 $SRC"; exit 1; }
    "$SSH" -n -o BatchMode=yes hoffman2 "echo 'CARC $SRC md5 $m1' > $H2/$DST/STAGE1_COMPLETE"
    echo "staged $B $M c$C records (md5 $m1) -> $DST"
  done < "cluster/optuna_h2_xcheck_s2_$XTAG.txt"
fi
echo "ship OK -> hoffman2:$H2"
