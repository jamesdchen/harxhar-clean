#!/bin/bash
#$ -cwd
#$ -j y
#$ -l h_rt=2:00:00,h_data=8G
# Hoffman2 environment of the trees' three-way comparison (RULE.txt addendum C), as the tree learning curve's
# (results/pooled_trees_lc_2014start_w1000/code/h2/env.sh): uv's CPython 3.12.3 and two venvs inside a Debian bookworm
# container (copies of /u/scratch/j/jamesdc1/tlc/{deb.sif,bin/uv,uvpython,lib/libgomp.so.1}, md5-checked; read only
# there), with CARC's pins (env_carc.txt of transfer 1):
#   venv_gb   = CARC's /scratch1/jc_905/envs/harxhar-r13 for the trees (req_gb.txt: numpy 1.26.4, pandas 2.3.3, scipy
#               1.17.1, scikit-learn 1.9.1, numba 0.68.0, pyarrow 19.0.1, lightgbm 4.6.0, xgboost 3.2.0)
#   venv_skx  = CARC's /home1/jc_905/trees_final/venv_skx (req_skx.txt: the same + scikit-learn-intelex 2026.2.0 and
#               the oneDAL packages it pulls in, pinned to CARC's versions)
# bookworm-slim has no libgomp (LightGBM's wheel links the system's): Debian bookworm's libgomp1 12.2.0-14+deb12u1 in
# $B/lib on LD_LIBRARY_PATH.  Writes $B/ENV_OK and $B/h2/freeze_{gb,skx}.txt.
set -euo pipefail
. /u/local/Modules/default/init/modules.sh
module load apptainer/1.2.2
B=/u/scratch/j/jamesdc1/t3t
T=/u/scratch/j/jamesdc1/tlc
echo "env job ${JOB_ID:-} on $(hostname) [$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2)] $(date)"
mkdir -p $B/bin $B/lib $B/apcache $B/uvcache $B/h2
[ -f $B/deb.sif ] || cp $T/deb.sif $B/deb.sif
[ -x $B/bin/uv ] || cp $T/bin/uv $B/bin/uv
[ -d $B/uvpython ] || cp -a $T/uvpython $B/uvpython
[ -f $B/lib/libgomp.so.1 ] || cp $T/lib/libgomp.so.1 $B/lib/libgomp.so.1
md5sum $B/deb.sif $T/deb.sif $B/bin/uv $T/bin/uv $B/lib/libgomp.so.1 $T/lib/libgomp.so.1
export APPTAINER_CACHEDIR=$B/apcache UV_CACHE_DIR=$B/uvcache UV_PYTHON_INSTALL_DIR=$B/uvpython UV_PYTHON_PREFERENCE=only-managed
rm -rf $B/venv_gb $B/venv_skx $B/ENV_OK
apptainer exec -B /u/scratch $B/deb.sif bash -c "
set -e
ldd --version | head -1
for v in gb skx; do
  $B/bin/uv venv -p 3.12.3 $B/venv_\$v
  VIRTUAL_ENV=$B/venv_\$v $B/bin/uv pip install --no-build -r $B/jobs/req_\$v.txt
  VIRTUAL_ENV=$B/venv_\$v $B/bin/uv pip freeze > $B/h2/freeze_\$v.txt
  echo \"== freeze \$v\"; cat $B/h2/freeze_\$v.txt
done
export LD_LIBRARY_PATH=$B/lib PYTHONNOUSERSITE=1
for v in gb skx; do
  $B/venv_\$v/bin/python -c 'import sys, numpy, scipy, pandas, sklearn, pyarrow, lightgbm, xgboost, numba; print(sys.version); print({m.__name__: m.__version__ for m in (numpy, scipy, pandas, sklearn, pyarrow, lightgbm, xgboost, numba)})'
done
$B/venv_skx/bin/python -c 'import importlib.metadata as m; print({k: m.version(k) for k in (\"scikit-learn-intelex\", \"daal\")})'
$B/venv_gb/bin/python -c 'import numpy; numpy.show_config()' 2>&1 | grep -i -E 'openblas|version|name' | head -12
for so in \$(find $B/venv_gb/lib/python3.12/site-packages/lightgbm $B/venv_gb/lib/python3.12/site-packages/xgboost -name '*.so'); do echo \"== ldd \$so\"; ldd \$so | grep -E 'gomp|not found|stdc|libc.so' || true; done
$B/venv_gb/bin/python -c 'import lightgbm, numpy as np; X = np.random.default_rng(0).normal(size=(500, 5)); m = lightgbm.LGBMRegressor(n_estimators=5, verbosity=-1).fit(X, X[:, 0]); print(\"lgbm smoke\", m.predict(X[:3]))'
$B/venv_skx/bin/python -c 'import numpy as np; from sklearnex.ensemble import RandomForestRegressor as R; X = np.random.default_rng(0).normal(size=(500, 5)); m = R(n_estimators=5, random_state=0).fit(X, X[:, 0]); print(\"skx smoke\", m.predict(X[:3]))'
"
touch $B/ENV_OK
echo "ENV DONE $(date)"
