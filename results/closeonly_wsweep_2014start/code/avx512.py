"""Print numpy's AVX512 dispatch targets present on this CPU (for NPY_DISABLE_CPU_FEATURES, as stage3.sh)."""

from numpy.core._multiarray_umath import __cpu_dispatch__ as d  # type: ignore[attr-defined]
from numpy.core._multiarray_umath import __cpu_features__ as f  # type: ignore[attr-defined]

print(" ".join(k for k in d if k.startswith("AVX512") and f.get(k)))
