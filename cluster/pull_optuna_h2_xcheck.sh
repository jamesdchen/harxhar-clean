#!/bin/bash
# I5 (2026-09-29): pull both sides of the cross-cluster check <tag> into
#   results/linear_subsection_trees_optuna/crosscluster/<tag>/{h2,carc}/   (md5 == source for every file)
# for experiments/optuna_h2_crosscheck.py:
#   h2    from Hoffman2 $XROOT_H2 = results/linear_subsection_trees_optuna/crosscluster/h2_<tag>:
#         stage-1 chunk records (trials npz) + run.log + H2_HOST, stage-2 per-path trees npz / results
#         csv + H2_HOST, the probe CSVs
#   carc  from CARC $CROOT (C's results root of the run checked): for every arm of the tag's stage-1
#         lines, the fleet's chunk records (chunks/c<k>/.../trials_<seg>.npz of every DONE chunk listed
#         in CHUNKS_<tag>, default c0) and, for every stage-2 line, the canary's stage 2 of that block
#         (canary_stage2/<arm>/c<k>/<path>/...), relaid under carc/stage2/<arm>/c<k>/; the probe CSVs
#         under crosscluster/carc_<tag>/probe/
#   hosts.csv  where each side ran: Hoffman2 H2_HOST files; CARC nodes from the Slurm logs' "task ... on
#         <node>" lines of the chunk (op_canary / op_s1 logs) and sinfo's node features.
# Each side is optional (SIDES="h2 carc"), so the Hoffman2 half can be pulled while CARC is unreachable;
# H2MAP="h2_mask:h2 h2_mask_avx:h2avx" pulls several Hoffman2 check roots (e.g. per CPU class).
#   CROOT=results/linear_subsection_trees_optuna_mask TAG=mask bash cluster/pull_optuna_h2_xcheck.sh
set -euo pipefail
cd "$(dirname "$0")/.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
CARC=/scratch1/jc_905/harxhar-subsection
H2=/u/scratch/j/jamesdc1/harxhar-optuna
TAG=${TAG:?TAG}
CROOT=${CROOT:?CROOT = C results root of the checked run}
SIDES=${SIDES:-h2 carc}
X=results/linear_subsection_trees_optuna/crosscluster
L=$X/$TAG
mkdir -p "$L/h2" "$L/carc"
sums() { awk '{p = $2; sub(/^\*/, "", p); print p, $1}' | sort; }

pull () {  # $1 host, $2 remote root, $3 local dir, $4 file list (paths relative to $2)
  local list=$4
  [ -s "$list" ] || { echo "  nothing listed on $1"; return 0; }
  "$SSH" -o BatchMode=yes "$1" "cd $2 && tar czf - -T -" < "$list" 2>/dev/null | tar xzf - -C "$3"
  local r l
  r=$("$SSH" -o BatchMode=yes "$1" "cd $2 && xargs -d '\n' md5sum" < "$list" 2>/dev/null | sums)
  l=$(cd "$3" && tr -d '\r' < "$OLDPWD/$list" | xargs -d '\n' md5sum | sums)
  [ "$r" = "$l" ] || { echo "PULL FAILED ($1): md5 differs"; diff <(printf '%s\n' "$r") <(printf '%s\n' "$l") | head; exit 1; }
  echo "  $1: $(wc -l < "$list") files, local md5 == remote"
}

if [[ " $SIDES " == *" h2 "* ]]; then
  # H2MAP: "<Hoffman2 check root under $X>:<local side>" pairs (default h2_<tag>:h2)
  for pair in ${H2MAP:-h2_$TAG:h2}; do
    XH=$X/${pair%%:*}; SIDE=${pair##*:}
    mkdir -p "$L/$SIDE"
    "$SSH" -o BatchMode=yes hoffman2 "cd $H2/$XH 2>/dev/null && find . -type f \( -name 'trials_*.npz' -o -name 'trees_*.npz' -o -name 'results_*.csv' -o -name 'run.log' -o -name 'H2_HOST' -o -name 'DONE' -o -name 'STAGE1_COMPLETE' -o -path './probe/*.csv' -o -path './probe_x/*.npz' \) | sed 's|^\./||' | sort"       | tr -d '\r' > "$L/.${SIDE}_list"
    pull hoffman2 "$H2/$XH" "$L/$SIDE" "$L/.${SIDE}_list"
    "$SSH" -o BatchMode=yes hoffman2 "cd $H2 && grep -h -E '^(probe|task) [0-9]+ on ' logs/op_h2_*.o* 2>/dev/null" > "$L/.h2_hosts" || true
  done
fi


if [[ " $SIDES " == *" carc "* ]]; then
  CH=${CHUNKS:-0}
  : > "$L/.carc_list"
  if [ -f "cluster/optuna_h2_xcheck_s1_$TAG.txt" ]; then
    while read -r B M TW SEG C S E H; do
      for k in $CH; do
        echo "$CROOT/stage1/$B/$SEG/$M/tw$TW/chunks/c$k/causal_tune_trees/$M/$B/trials_$SEG.npz"
      done
    done < "cluster/optuna_h2_xcheck_s1_$TAG.txt" | sort -u >> "$L/.carc_list"
  fi
  : > "$L/.carc_s2_dirs"
  if [ -f "cluster/optuna_h2_xcheck_s2_$TAG.txt" ]; then
    while read -r B M TW SEG C S E H; do
      echo "$CROOT/canary_stage2/$B/$SEG/$M/tw$TW/c$C"
    done < "cluster/optuna_h2_xcheck_s2_$TAG.txt" > "$L/.carc_s2_dirs"
  fi
  # keep only the files that exist (a missing chunk record = that chunk is not DONE on CARC yet:
  # the check reports the point as unmatched); expand the canary stage-2 dirs; the probe files
  "$SSH" -o BatchMode=yes usc-discovery "cd $CARC && while read -r f; do [ -f \"\$f\" ] && echo \"\$f\"; done; true" \
    < "$L/.carc_list" 2>/dev/null | tr -d '\r' > "$L/.carc_list.ok"
  "$SSH" -o BatchMode=yes usc-discovery "cd $CARC && while read -r d; do [ -d \"\$d\" ] && find \"\$d\" -type f \( -name 'trees_*.npz' -o -name 'results_*.csv' \); done; find $X/carc_$TAG/probe $X/carc_$TAG/probe_x -type f 2>/dev/null; true" \
    < "$L/.carc_s2_dirs" 2>/dev/null | tr -d '\r' >> "$L/.carc_list.ok"
  mkdir -p "$L/.carc_raw"
  pull usc-discovery "$CARC" "$L/.carc_raw" "$L/.carc_list.ok"
  # relay into the check layout: carc/stage1/..., carc/stage2/<arm>/c<k>/..., carc/probe(_x)/
  mkdir -p "$L/carc"
  [ -d "$L/.carc_raw/$CROOT/stage1" ] && mkdir -p "$L/carc/stage1" && cp -r "$L/.carc_raw/$CROOT/stage1/." "$L/carc/stage1/"
  [ -d "$L/.carc_raw/$CROOT/canary_stage2" ] && mkdir -p "$L/carc/stage2" && cp -r "$L/.carc_raw/$CROOT/canary_stage2/." "$L/carc/stage2/"
  for p in probe probe_x; do
    [ -d "$L/.carc_raw/$X/carc_$TAG/$p" ] && mkdir -p "$L/carc/$p" && cp -r "$L/.carc_raw/$X/carc_$TAG/$p/." "$L/carc/$p/"
  done
  # FLEET2=1: C's own stage-2 chunk of each stage-2 line (its first rows = the check block), side carc_fleet
  if [ "${FLEET2:-0}" = 1 ] && [ -s "$L/.carc_s2_dirs" ]; then
    sed "s|$CROOT/canary_stage2/\(.*\)/c\([0-9]*\)\$|$CROOT/stage2/\1/chunks/c\2|" "$L/.carc_s2_dirs" > "$L/.fleet2_dirs"
    "$SSH" -o BatchMode=yes usc-discovery "cd $CARC && while read -r d; do [ -f \"\$d/DONE\" ] && find \"\$d\" -type f \( -name 'trees_*.npz' -o -name 'DONE' \); done; true" \
      < "$L/.fleet2_dirs" 2>/dev/null | tr -d '\r' > "$L/.fleet2_list"
    mkdir -p "$L/.fleet2_raw" "$L/carc_fleet"
    pull usc-discovery "$CARC" "$L/.fleet2_raw" "$L/.fleet2_list"
    [ -d "$L/.fleet2_raw/$CROOT/stage2" ] && cp -r "$L/.fleet2_raw/$CROOT/stage2" "$L/carc_fleet/"
  fi
  # the CARC per-class check roots (cluster/slurm/submit_optuna_h2_xcheck_carc.sh):
  # CMAP="carc_<tag>_<feature>:<local side> ..."
  for pair in ${CMAP:-}; do
    XC=$X/${pair%%:*}; SIDE=${pair##*:}
    mkdir -p "$L/$SIDE"
    "$SSH" -o BatchMode=yes usc-discovery "cd $CARC/$XC 2>/dev/null && find . -type f \( -name 'trials_*.npz' -o -name 'trees_*.npz' -o -name 'results_*.csv' -o -name 'run.log' -o -name 'DONE' -o -name 'STAGE1_COMPLETE' \) | sed 's|^\./||' | sort" \
      2>/dev/null | tr -d '\r' > "$L/.${SIDE}_list"
    pull usc-discovery "$CARC/$XC" "$L/$SIDE" "$L/.${SIDE}_list"
  done
  # CARC hosts: the Slurm log lines of the canary / stage-1 / check / probe tasks, and every node's features
  "$SSH" -o BatchMode=yes usc-discovery "cd $CARC && grep -H -E '^(task|probe) [0-9]+ on ' logs/op_canary.*.out logs/op_s1.*.out logs/op_s2.*.out logs/op_xc_s*.out logs/op_h2_probe.*.out 2>/dev/null; sinfo -N -h -o 'NODE %N %f' | sort -u; true" \
    2>/dev/null | tr -d '\r' > "$L/.carc_hosts"
fi

# hosts.csv: host -> CPU (Hoffman2: the task / probe log lines; CARC: the node's Slurm features, and the
# probe log's /proc/cpuinfo model), plus which CARC Slurm task ran which chunk
/c/Users/james/miniconda3/envs/285J/python.exe - "$L" <<'EOF'
import csv, re, sys
from pathlib import Path
L = Path(sys.argv[1])
cpu, feat, ran = {}, {}, []
if (L / ".h2_hosts").is_file():
    for line in (L / ".h2_hosts").read_text().splitlines():
        m = re.match(r"^(?:task|probe) \d+ on (\S+) \[\s*([^|\]]+)", line)
        if m:
            cpu[m.group(1).split(".")[0]] = m.group(2).strip()
if (L / ".carc_hosts").is_file():
    for line in (L / ".carc_hosts").read_text().splitlines():
        if line.startswith("NODE "):
            _, n, f = line.split(" ", 2)
            feat[n] = f
            continue
        m = re.match(r"^logs/(\w+)\.(\d+)\.(\d+)\.out:task \d+ on (\S+?)(?:\.hpc\.usc\.edu)?, stage (\d) .*: (\S+) \[(\S+)\] tw(\d+) (\S+) chunk (\d+)", line)
        if m:
            jn, job, task, node, st, b, mdl, tw, seg, c = m.groups()
            ran.append((f"{job}_{task}", jn, node, f"stage{st}/{b}/{seg}/{mdl}/tw{tw}/chunks/c{c}"))
        m = re.match(r"^logs/\S+:probe \d+ on (\S+?)(?:\.hpc\.usc\.edu)? \[\s*([^\]]+)\]", line)
        if m:
            cpu[m.group(1)] = m.group(2).strip()
with open(L / "hosts.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["host", "cpu", "carc_features"])
    for h in sorted(set(cpu) | {r[2] for r in ran}):
        w.writerow([h, cpu.get(h, ""), feat.get(h, "")])
with open(L / "carc_chunk_tasks.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["slurm_task", "job_name", "node", "carc_features", "chunk"])
    for r in sorted(set(ran)):
        w.writerow([r[0], r[1], r[2], feat.get(r[2], ""), r[3]])
print(f"hosts.csv: {len(set(cpu) | {r[2] for r in ran})} hosts; carc_chunk_tasks.csv: {len(set(ran))} tasks")
EOF
echo "pull OK -> $L"
