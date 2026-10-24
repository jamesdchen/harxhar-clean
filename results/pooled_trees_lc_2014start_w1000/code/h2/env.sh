#!/bin/bash
#$ -cwd
#$ -j y
#$ -l h_rt=1:30:00,h_data=8G
# The tree walks' Python env on Hoffman2 (CentOS 7, glibc 2.17): the window sweep's approach -- uv's CPython 3.12.3
# and a venv inside a Debian bookworm container (copies of /u/scratch/j/jamesdc1/wsweep/{deb.sif,bin/uv,uvpython},
# read only there) -- with the pins of cluster/requirements-rerun13.lock.txt (the CARC env harxhar-r13 of the printed
# run: numpy 1.26.4, pandas 2.3.3, scipy 1.17.1, scikit-learn 1.9.1, numba 0.68.0, pyarrow 19.0.1, lightgbm 4.6.0,
# xgboost 3.2.0).  bookworm-slim has no libgomp (LightGBM's wheel links the system's): Debian bookworm's own
# libgomp1 12.2.0-14+deb12u1 (libgomp.so.1, shipped to $B/lib) is put on LD_LIBRARY_PATH.  Writes $B/ENV_OK.
set -euo pipefail
. /u/local/Modules/default/init/modules.sh
module load apptainer/1.2.2
B=/u/scratch/j/jamesdc1/tlc
WS=/u/scratch/j/jamesdc1/wsweep
echo "env job ${JOB_ID:-} on $(hostname) [$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2)] $(date)"
mkdir -p $B/bin $B/apcache $B/uvcache
[ -f $B/deb.sif ] || cp $WS/deb.sif $B/deb.sif
[ -x $B/bin/uv ] || cp $WS/bin/uv $B/bin/uv
[ -d $B/uvpython ] || cp -a $WS/uvpython $B/uvpython
md5sum $B/deb.sif $WS/deb.sif $B/bin/uv $WS/bin/uv
export APPTAINER_CACHEDIR=$B/apcache UV_CACHE_DIR=$B/uvcache UV_PYTHON_INSTALL_DIR=$B/uvpython UV_PYTHON_PREFERENCE=only-managed
rm -rf $B/venv $B/ENV_OK
apptainer exec -B /u/scratch $B/deb.sif bash -c "
set -e
ldd --version | head -1
$B/bin/uv venv -p 3.12.3 $B/venv
VIRTUAL_ENV=$B/venv $B/bin/uv pip install --no-build -r $B/jobs/req.txt
VIRTUAL_ENV=$B/venv $B/bin/uv pip freeze | tee $B/jobs/freeze.txt
export LD_LIBRARY_PATH=$B/lib PYTHONNOUSERSITE=1
$B/venv/bin/python -c 'import sys, platform, numpy, scipy, pandas, sklearn, pyarrow, lightgbm, xgboost, numba; print(sys.version); print({m.__name__: m.__version__ for m in (numpy, scipy, pandas, sklearn, pyarrow, lightgbm, xgboost, numba)})'
for so in \$(find $B/venv/lib/python3.12/site-packages/lightgbm $B/venv/lib/python3.12/site-packages/xgboost -name '*.so'); do echo \"== ldd \$so\"; ldd \$so | grep -E 'gomp|not found|stdc|libc.so' || true; done
$B/venv/bin/python -c 'import lightgbm, numpy as np; X = np.random.default_rng(0).normal(size=(500, 5)); m = lightgbm.LGBMRegressor(n_estimators=5, verbosity=-1).fit(X, X[:, 0]); print(\"lgbm smoke\", m.predict(X[:3]))'
"
touch $B/ENV_OK
echo "ENV DONE $(date)"
