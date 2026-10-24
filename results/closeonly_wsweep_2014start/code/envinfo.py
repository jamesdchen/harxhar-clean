"""Record the linear environment as the three-way's stage3.sh: BLAS kernel, numpy / numba versions, logspace grids."""

import numba
import numpy
import threadpoolctl

print("blas", [(d["internal_api"], d.get("architecture"), d.get("version")) for d in threadpoolctl.threadpool_info()])
print("numpy", numpy.__version__, "numba", numba.__version__)
print("logspace", repr(numpy.logspace(-6, -2, 5).tolist()), repr(numpy.logspace(-2, 3, 6).tolist()))
