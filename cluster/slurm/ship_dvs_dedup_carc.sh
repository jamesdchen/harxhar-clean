#!/bin/bash
# Ship the C2 dense-vs-sparse re-run's cluster part (checklist I8) to the CARC root (tar over
# native Windows OpenSSH) and FAIL if any shipped file's remote md5 differs from the local
# one: the study script and the first pass's module it imports, the per-window mask helper,
# the Slurm scripts, and the two captured designs (from $DVS_WORK, into
# results/dense_vs_sparse_dedup/_work_carc/).  Then, WITHOUT shipping, compare local vs
# remote md5 of the tree spec and the src modules its model section imports (SAME / DIFF; the
# spec's model section -- parameters, make_model .. contributions -- must be SAME), and check
# on the login node that the tree stack (with shap from ./pylib_trees) imports and that the
# shipped script parses.
# Run from Git Bash locally:
#   DVS_WORK=<work dir> bash cluster/slurm/ship_dvs_dedup_carc.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
REMOTE=/scratch1/jc_905/harxhar-subsection
WORKDIR=${DVS_WORK:-results/dense_vs_sparse_dedup/_work}
RW=results/dense_vs_sparse_dedup/_work_carc
FILES=(
  experiments/dvs_dedup_1530.py
  experiments/dense_vs_sparse_1530.py
  src/models/window_mask.py
  cluster/slurm/dvs_dedup.sbatch
  cluster/slurm/submit_dvs_dedup.sh
)
CHECK=(
  specs/causal_tune_trees.py
  src/backtest/executor.py
  src/data/loading.py
  src/features/extractors/har.py
)
for f in "${FILES[@]}" "${CHECK[@]}"; do
  [ -f "$f" ] || { echo "missing locally: $f"; exit 1; }
done
for b in live_feasible all_features; do
  [ -f "$WORKDIR/design_$b.npz" ] || { echo "missing $WORKDIR/design_$b.npz (run the capture stage)"; exit 1; }
done

sums() { awk '{p = $2; sub(/^\*/, "", p); print p, $1}' | sort; }

# 1. code
L=$(md5sum "${FILES[@]}" | sums)
R=$(tar czf - "${FILES[@]}" | "$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && tar xzf - && md5sum ${FILES[*]}" | tr -d '\r' | sums)
[ "$L" = "$R" ] || { echo "SHIP FAILED (code)"; diff <(echo "$L") <(echo "$R") || true; exit 1; }
echo "shipped ${#FILES[@]} code files (md5 match)"
# 2. the designs
L=$(cd "$WORKDIR" && md5sum design_live_feasible.npz design_all_features.npz | sums)
R=$(tar czf - -C "$WORKDIR" design_live_feasible.npz design_all_features.npz \
  | "$SSH" -o BatchMode=yes usc-discovery "mkdir -p $REMOTE/$RW && cd $REMOTE/$RW && tar xzf - && md5sum design_live_feasible.npz design_all_features.npz" \
  | tr -d '\r' | sums)
[ "$L" = "$R" ] || { echo "SHIP FAILED (designs)"; diff <(echo "$L") <(echo "$R") || true; exit 1; }
echo "shipped the two designs into $RW (md5 match)"
# 3. unshipped dependencies: SAME / DIFF
L=$(md5sum "${CHECK[@]}" | sums)
R=$("$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && md5sum ${CHECK[*]}" | tr -d '\r' | sums)
join <(echo "$L") <(echo "$R") | awk '{print ($2 == $3 ? "SAME" : "DIFF"), $1}'
# the tree spec may differ as a file (another campaign's axes); its MODEL SECTION (the
# parameter block and make_model .. contributions, the only part this study executes) may not
sec() { tr -d '\r' | awk '/^LGBM_PARAMS: dict = \{/{p=1} /^PROVENANCE = \{/{p=0} /^def make_model\(\):/{q=1} /^SIDE: dict = \{\}/{q=0} p||q'; }
SEC_L=$(sec < specs/causal_tune_trees.py | md5sum | cut -c1-32)
SEC_R=$("$SSH" -o BatchMode=yes usc-discovery "cat $REMOTE/specs/causal_tune_trees.py" | sec | md5sum | cut -c1-32)
[ "$SEC_L" = "$SEC_R" ] || { echo "the tree spec's model section differs on CARC: refusing"; exit 1; }
echo "tree spec model section SAME (md5 ${SEC_L:0:12})"
# 4. the stack imports; the script parses
"$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && module load conda >/dev/null 2>&1 && source activate harxhar && PYTHONPATH=\$PWD:\$PWD/pylib_trees python -c 'import ast, lightgbm, xgboost, sklearn, shap, numpy; ast.parse(open(\"experiments/dvs_dedup_1530.py\").read()); from src.models.window_mask import window_keep; print(\"stack ok: lightgbm\", lightgbm.__version__, \"xgboost\", xgboost.__version__, \"sklearn\", sklearn.__version__, \"shap\", shap.__version__, \"numpy\", numpy.__version__)'" 2>&1 \
  | grep -v -e 'have been reloaded' -e 'python/3'
