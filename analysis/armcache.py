"""Per-arm result checkpointing so multi-arm batteries survive container restarts.

Set ARMCACHE=<dir>; each arm's result vector is saved under a sanitized tag and loaded
instead of recomputed on rerun. Progress becomes monotone: a battery that loses its process
mid-run resumes at the first uncomputed arm.
"""

from __future__ import annotations

import os
import re

import numpy as np


def memo(tag: str, fn):
    d = os.environ.get("ARMCACHE")
    fp = None
    if d:
        fp = os.path.join(d, re.sub(r"[^A-Za-z0-9_.-]+", "_", tag) + ".npz")
        if os.path.exists(fp):
            return np.load(fp)["v"]
    v = fn()
    if fp:
        os.makedirs(d, exist_ok=True)
        np.savez_compressed(fp, v=np.asarray(v))
    return v
