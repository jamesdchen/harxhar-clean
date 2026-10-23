#!/bin/bash
# CARC login node, one ssh (the tarball arrives on stdin and is unpacked to /home1/jc_905/tree_shap before this runs):
#   1. sha256 of the staged files; the runner and configs are the git blobs of 3cbb85f (checked again in python)
#   2. python = the tree agent's /home1 venv_skx (read only); shap 0.49.1 and whichever of its dependencies are
#      missing installed with --no-deps into $HB/pylib (nothing installed into the shared envs)
#   3. smoke test of shap_common on the /home1 design (column table, origins, runner); the walk's origin rows
#   4. sbatch the array (tasks.txt), EPYC 7513 / 7542 only
# usage: bash carc_stage.sh <n tasks> [--hold]
set -euo pipefail
NT=$1; XF=${2:-}
HB=/home1/jc_905/tree_shap
PY=/home1/jc_905/trees_final/venv_skx/bin/python
D=/home1/jc_905/trees_final/designs/w3000
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p $HB/logs $HB/work $HB/pylib
echo "== 1. staged files"
(cd $HB && find res runner -type f | sort | xargs sha256sum)
echo "== 2. python and shap"
$PY -c "import sys, numpy, sklearn, lightgbm, xgboost, pandas, pyarrow, importlib.metadata as m; print(sys.executable, sys.version.split()[0], 'numpy', numpy.__version__, 'sklearn', sklearn.__version__, 'lightgbm', lightgbm.__version__, 'xgboost', xgboost.__version__, 'pandas', pandas.__version__, 'pyarrow', pyarrow.__version__, 'sklearnex', m.version('scikit-learn-intelex'), 'daal', m.version('daal'))"
MISSING=""
for m in shap:shap==0.49.1 numba:numba==0.68.0 llvmlite:llvmlite==0.50.0 slicer:slicer==0.0.8 \
         cloudpickle:cloudpickle==3.1.2 tqdm:tqdm==4.70.1 packaging:packaging==26.3; do
  mod=${m%%:*}; spec=${m#*:}
  PYTHONPATH=$HB/pylib $PY -c "import $mod" 2>/dev/null || MISSING="$MISSING $spec"
done
echo "missing:${MISSING:- none}"
if [ -n "$MISSING" ]; then $PY -m pip install -q --no-deps --target $HB/pylib $MISSING 2>&1 | grep -v -i notice || true; fi
PYTHONPATH=$HB/pylib $PY -c "import shap, numba, numpy; print('shap', shap.__version__, shap.__file__, '| numba', numba.__version__, '| numpy', numpy.__version__)"
echo "== 3. smoke + walk origin rows"
cd $HB/res/code
TREE_SHAP_DESIGN=$D TREE_SHAP_WORK=$HB/work TREE_SHAP_RUNNER_DIR=$HB/runner PYTHONPATH=$HB/pylib $PY -c "
import shap_common as C
t = C.column_table(); o = C.origins(); R = C.load_runner()
print('columns', len(t), 'origins', len(o), 'runner', R.RUNNER_SRC_SHA256[:12], 'configs', sorted(R.CONFIGS))"
$PY carc_fetch_origin_rows.py $HB/res/inputs/fiw_origins.csv > $HB/gate_rows.csv
grep -c '^ROW' $HB/gate_rows.csv || true
cat $HB/gate_rows.csv
echo "== 4. submit"
X=()
[ -n "$XF" ] && X=("$XF")
j=$(sbatch --parsable --array=1-$NT "${X[@]}" $HB/res/code/carc_job.sbatch)
echo "$(date +%F_%T) tshap array $j ($NT tasks)" | tee -a $HB/jobids.txt
squeue -j "$j" -o "%.18i %.9P %.8T %.10M %.6C %R" | head
