#!/bin/bash
#$ -cwd
#$ -j y
#$ -l h_rt=0:45:00,h_data=8G
# 2026-10-24 04:10: env.sh (job 15261016) installed both venvs but its sklearnex smoke test failed: daal4py imports
# distutils (removed from Python 3.12), which CARC's env provides through setuptools 84.1.0 (harxhar-r13, inherited by
# venv_skx).  This adds setuptools 84.1.0 to both venvs (req_gb.txt / req_skx.txt now pin it), re-runs env.sh's checks
# (versions, LightGBM's libgomp, the LightGBM and sklearnex smoke fits, sklearnex's dispatch message) and writes
# $B/ENV_OK and $B/h2/freeze_{gb,skx}.txt.
set -euo pipefail
. /u/local/Modules/default/init/modules.sh
module load apptainer/1.2.2
B=/u/scratch/j/jamesdc1/t3t
echo "envfix job ${JOB_ID:-} on $(hostname) [$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2)] $(date)"
export APPTAINER_CACHEDIR=$B/apcache UV_CACHE_DIR=$B/uvcache UV_PYTHON_INSTALL_DIR=$B/uvpython UV_PYTHON_PREFERENCE=only-managed
rm -f $B/ENV_OK
apptainer exec -B /u/scratch $B/deb.sif bash -c "
set -e
for v in gb skx; do
  VIRTUAL_ENV=$B/venv_\$v $B/bin/uv pip install --no-build -r $B/jobs/req_\$v.txt
  VIRTUAL_ENV=$B/venv_\$v $B/bin/uv pip freeze > $B/h2/freeze_\$v.txt
  echo \"== freeze \$v\"; cat $B/h2/freeze_\$v.txt
done
export LD_LIBRARY_PATH=$B/lib PYTHONNOUSERSITE=1
for v in gb skx; do
  $B/venv_\$v/bin/python -c 'import sys, numpy, scipy, pandas, sklearn, pyarrow, lightgbm, xgboost, numba, setuptools; print(sys.version); print({m.__name__: m.__version__ for m in (numpy, scipy, pandas, sklearn, pyarrow, lightgbm, xgboost, numba, setuptools)})'
done
$B/venv_skx/bin/python -c 'import importlib.metadata as m; print({k: m.version(k) for k in (\"scikit-learn-intelex\", \"daal\", \"tbb\", \"tcmlib\")})'
for so in \$(find $B/venv_gb/lib/python3.12/site-packages/lightgbm -name '*.so'); do echo \"== ldd \$so\"; ldd \$so | grep -E 'gomp|not found' || true; done
$B/venv_gb/bin/python -c 'import lightgbm, numpy as np; X = np.random.default_rng(0).normal(size=(500, 5)); m = lightgbm.LGBMRegressor(n_estimators=5, verbosity=-1).fit(X, X[:, 0]); print(\"lgbm smoke\", m.predict(X[:3]))'
$B/venv_skx/bin/python -c '
import logging, numpy as np
msgs = []
class H(logging.Handler):
    def emit(self, r): msgs.append(r.getMessage())
lg = logging.getLogger(\"sklearnex\"); lg.addHandler(H()); lg.setLevel(logging.INFO)
from sklearnex.ensemble import RandomForestRegressor as R
X = np.random.default_rng(0).normal(size=(500, 5))
m = R(n_estimators=5, random_state=0).fit(X, X[:, 0])
print(\"skx smoke\", m.predict(X[:3]), sorted(set(msgs)))
assert any(\"running accelerated version\" in x for x in msgs), msgs
'
"
touch $B/ENV_OK
echo "ENVFIX DONE $(date)"
