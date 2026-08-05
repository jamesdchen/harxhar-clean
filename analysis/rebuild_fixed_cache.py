"""Rebuild the b2 feature cache with the fixed availability handling.

Writes to results/b2_mmap_fix rather than overwriting results/b2_mmap, so the
October 2023 attribution, the era table and the exogenous-versus-backbone
contrast can all be run paired against the pre-fix cache.

The fix under test is in src/backtest/executor.load_and_transform: availability
recorded before the forward fill, a bounded staleness window, and the neutral
median applied to unobserved rows.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
# root scripts moved to experiments/ on 2026-08-04
sys.path.insert(0, os.path.join(ROOT, "experiments"))

# Target directories are overridable so each prep ablation lands in its own
# cache and the builds can be differenced. Defaults to the indicator-only fix,
# which is what the executor's current defaults produce.
OUT = os.path.join(ROOT, "results", os.environ.get("OUT_TAG", "b2_mmap_indonly"))
CACHE = os.path.join(ROOT, os.environ.get("CACHE_TAG", "results_indonly"))


def main() -> None:
    os.environ.setdefault("HAR_BASE", "2")
    import run_geometry_local as R

    R.CACHE_DIR = CACHE
    os.makedirs(CACHE, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    print(f"building all_features at HAR_BASE={os.environ['HAR_BASE']} -> {CACHE}",
          flush=True)
    X, y, b, names = R.prepare_full("all_features")
    print(f"prep complete {X.shape} in {time.time() - t0:.0f}s", flush=True)
    np.save(os.path.join(OUT, "X.npy"), np.ascontiguousarray(X, dtype=np.float64))
    np.save(os.path.join(OUT, "y.npy"), np.asarray(y, dtype=np.float64))
    np.save(os.path.join(OUT, "b.npy"), np.asarray(b, dtype=np.float64))
    np.save(os.path.join(OUT, "names.npy"), np.array(names, dtype=object))
    print(f"wrote {OUT} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
