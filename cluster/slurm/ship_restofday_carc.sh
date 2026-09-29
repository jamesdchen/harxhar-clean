#!/bin/bash
# Ship the DIRECT rest-of-day spec, its identity check and scorer, the Slurm
# scripts, the task files and the manifest to the CARC root (tar over native
# Windows OpenSSH), with the nine per-bar reference tables
# results/spxw_pnl/yhat_sub_{ridge,lasso,enet}_{all_features,baseline,
# live_feasible}.parquet landing in results/linear_subsection_restofday/ref/
# there (NOT in results/spxw_pnl).  Steps:
#   0. GENERATE cluster/restofday_manifest.md5 (md5sum format, repo-relative):
#      the shipped python, specs/causal_tune_linear.py (the spec executes its
#      machinery), every src module the shipped python can import (fixed list
#      + an AST import-closure check), every experiments module they import,
#      and the endbartime-keyed data/*.parquet (schema read only; the chain
#      and spot files are not keyed, not merged by the loader, absent on CARC).
#      Every Slurm task and the submit script refuse to run if a pinned file
#      differs on CARC.
#   1. ship (local and remote md5 printed);
#   2. compare, WITHOUT shipping, local vs remote md5 of the pinned files that
#      are not shipped (src/**, specs/causal_tune_linear.py, data/*.parquet):
#      SAME / DIFF per file.  Shipping those would touch files the running
#      per-bar tree campaign reads, so a difference is left to the human;
#   3. on the login node: bash -n the shipped scripts, ast.parse the shipped
#      python, md5sum -c the manifest, mkdir the results root and logs.
# Exits 1 at the end (after shipping and checking) if any pinned file differs
# on CARC: the canary would refuse to run.
# Run from Git Bash locally:
#   bash cluster/slurm/ship_restofday_carc.sh
# The canary/fleet time, memory and concurrency defaults live at the top of
# submit_restofday.sh (overridable by env on CARC).
set -euo pipefail
cd "$(dirname "$0")/../.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
REMOTE=/scratch1/jc_905/harxhar-subsection
PY=${PY:-/c/Users/james/miniconda3/envs/285J/python.exe}
MANIFEST=cluster/restofday_manifest.md5
REFDIR=results/linear_subsection_restofday/ref

PYF=(
  specs/causal_tune_rest_of_day.py
  experiments/check_restofday_identity.py
  experiments/score_rest_of_day.py
  experiments/ft_remaining.py
)
SH=(
  cluster/slurm/restofday_pack.sbatch
  cluster/slurm/restofday_score.sbatch
  cluster/slurm/submit_restofday.sh
)
TASKS=(
  cluster/restofday_tasks_canary.txt
  cluster/restofday_tasks_fleet.txt
)
FILES=("${PYF[@]}" "${SH[@]}" "${TASKS[@]}" "$MANIFEST")
REFS=()
REMOTE_REFS=()
for est in ridge lasso enet; do
  for b in all_features baseline live_feasible; do
    REFS+=("results/spxw_pnl/yhat_sub_${est}_${b}.parquet")
    REMOTE_REFS+=("$REFDIR/yhat_sub_${est}_${b}.parquet")
  done
done
for f in "${PYF[@]}" "${SH[@]}" "${TASKS[@]}" "${REFS[@]}" specs/causal_tune_linear.py; do
  [ -f "$f" ] || { echo "missing locally: $f"; exit 1; }
done

# 0. the manifest
# src modules the spec imports (the tree ship script's list + the linear
# machinery's), every src/features module and every src package __init__.
SRC=(
  src/backtest/executor.py
  src/backtest/multi_stage.py
  src/backtest/segmentation.py
  src/data/loading.py
  src/data/options_features.py
  src/diagnostics.py
  src/evaluation/metrics.py
  src/evaluation/diebold_mariano.py
  src/models/reclasso_har.py
  src/models/ridge.py
  src/models/rolling_least_squares.py
)
mapfile -t FEAT < <(find src/features -name '*.py' -not -path '*/__pycache__/*' | sort)
mapfile -t INIT < <(find src -name '__init__.py' -not -path '*/__pycache__/*' | sort)
# the import closure (AST, src.* and experiments/<module>.py) of the shipped
# python and the linear spec: anything the fixed list misses is added.
CLOSURE=$("$PY" - "${PYF[@]}" specs/causal_tune_linear.py <<'EOF' | tr -d '\r'
import ast
import sys
from pathlib import Path


def path_of(parts):
    if parts[0] != "src":
        q = Path("experiments", parts[0] + ".py")
        return q if q.is_file() else None
    p = Path(*parts)
    for q in (p.with_suffix(".py"), p / "__init__.py"):
        if q.is_file():
            return q
    return None


seen, todo = set(), [Path(a) for a in sys.argv[1:]]
while todo:
    f = todo.pop()
    if f in seen:
        continue
    seen.add(f)
    pkg = list(f.parent.parts)
    for n in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
        if isinstance(n, ast.Import):
            mods = [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            base = n.module or ""
            if n.level:
                base = ".".join(pkg[: len(pkg) - n.level + 1] + ([base] if base else []))
            mods = [base] + [f"{base}.{a.name}" for a in n.names]
        else:
            continue
        for m in mods:
            parts = m.split(".")
            for i in range(1, len(parts) + 1 if parts[0] == "src" else 2):
                q = path_of(parts[:i])
                if q is not None and q not in seen:
                    todo.append(q)
for f in sorted(seen):
    print(f.as_posix())
EOF
)
mapfile -t CODE < <(printf '%s\n' "${PYF[@]}" specs/causal_tune_linear.py "${SRC[@]}" "${FEAT[@]}" "${INIT[@]}" $CLOSURE | sort -u)
EXTRA=$(comm -13 <(printf '%s\n' "${PYF[@]}" specs/causal_tune_linear.py "${SRC[@]}" "${FEAT[@]}" "${INIT[@]}" | sort -u) \
                 <(printf '%s\n' $CLOSURE | sort -u))
[ -z "$EXTRA" ] || { echo "import closure adds to the fixed list (pinned, compared, not shipped):"; printf '  %s\n' $EXTRA; }
# data: the endbartime-keyed parquets only (schema read, never the data)
mapfile -t ALLDATA < <(find data -maxdepth 1 -name '*.parquet' | sort)
KEYOUT=$("$PY" -c "
import sys
import pyarrow.parquet as pq
for f in sys.argv[1:]:
    print(int('endbartime' in pq.read_schema(f).names), f)
" "${ALLDATA[@]}" | tr -d '\r')
KEYED=()
while read -r k f; do
  if [ "$k" = 1 ]; then
    KEYED+=("$f")
  else
    echo "excluded from the manifest (not endbartime-keyed: not merged by the loader, absent on CARC): $f"
  fi
done <<< "$KEYOUT"
[ "${#KEYED[@]}" -gt 0 ] || { echo "no endbartime-keyed data/*.parquet found"; exit 1; }
md5sum "${CODE[@]}" "${KEYED[@]}" | sed 's/^\([0-9a-f]\{32\}\) \*/\1  /' > "$MANIFEST"
echo "wrote $MANIFEST: ${#CODE[@]} code + ${#KEYED[@]} data files"

# 1. ship
echo
echo "local:"
md5sum "${FILES[@]}" "${REFS[@]}" | cut -c1-12,34-
tar czf - --transform="s,^results/spxw_pnl/,$REFDIR/," "${FILES[@]}" "${REFS[@]}" \
  | "$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && tar xzf - && echo remote: && md5sum ${FILES[*]} ${REMOTE_REFS[*]} | cut -c1-12,34-"

# 2. compare (not shipped): the pinned files outside the shipped set
mapfile -t DEPS < <(printf '%s\n' "${CODE[@]}" "${KEYED[@]}" | grep -vxF -f <(printf '%s\n' "${PYF[@]}"))
echo
echo "dependencies, local vs remote md5 (${#DEPS[@]} files, not shipped):"
LOCAL_SUMS=$(md5sum "${DEPS[@]}" | awk '{p = $2; sub(/^\*/, "", p); print p, $1}')
REMOTE_SUMS=$("$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && for f in ${DEPS[*]}; do if [ -f \"\$f\" ]; then md5sum \"\$f\"; else echo MISSING \"\$f\"; fi; done" \
  | awk '{p = $2; sub(/^\*/, "", p); print p, $1}')
TABLE=$(awk 'NR == FNR {r[$1] = $2; next}
     {
       if (!($1 in r)) s = "DIFF (no remote line)";
       else if (r[$1] == "MISSING") s = "DIFF (missing remote)";
       else if (r[$1] == $2) s = "SAME";
       else s = "DIFF";
       printf "  %-22s %s\n", s, $1
     }' \
  <(printf '%s\n' "$REMOTE_SUMS") <(printf '%s\n' "$LOCAL_SUMS"))
printf '%s\n' "$TABLE"
NDIFF=$(printf '%s\n' "$TABLE" | grep -c '^  DIFF' || true)
echo "  $NDIFF of ${#DEPS[@]} differ"
if [ "$NDIFF" -gt 0 ]; then
  echo
  echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
  echo "!! $NDIFF pinned dependencies differ on CARC: the canary and the fleet WILL"
  echo "!! REFUSE TO RUN (manifest check), and so will submit_restofday.sh."
  echo "!! They are NOT shipped: shipping data/ or src/ would touch files the"
  echo "!! running per-bar tree campaign reads. The human decides."
  echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
fi

# 3. remote: results root, syntax checks, manifest check
echo
RC=0
"$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && module load conda >/dev/null 2>&1 && source activate harxhar && set -e
rc=0
mkdir -p results/linear_subsection_restofday logs
for s in ${SH[*]}; do
  if bash -n \$s; then echo \"bash -n ok: \$s\"; else echo \"bash -n FAILED: \$s\"; rc=1; fi
done
for p in ${PYF[*]}; do
  if python -c \"import ast, sys; ast.parse(open(sys.argv[1]).read())\" \$p; then echo \"parses: \$p\"; else echo \"PARSE FAILED: \$p\"; rc=1; fi
done
ls -l $REFDIR || rc=1
if md5sum -c --quiet $MANIFEST; then
  echo \"MANIFEST OK (\$(wc -l < $MANIFEST) files)\"
else
  echo 'MANIFEST DIFF: the files above differ; the canary and fleet will refuse to run'
  rc=1
fi
exit \$rc" || RC=$?
echo
if [ "$RC" -ne 0 ] || [ "$NDIFF" -gt 0 ]; then
  echo "SHIPPED, BUT NOT READY: $NDIFF pinned dependencies differ on CARC (remote check rc=$RC); do not hand over the submit script yet"
  exit 1
fi
echo "SHIPPED AND READY: on CARC run"
echo "  cd $REMOTE && bash cluster/slurm/submit_restofday.sh"
