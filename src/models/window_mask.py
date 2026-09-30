"""Per-window column mask for the per-bar models that have none of their own (trees, LSTM).

The per-bar linear arms (specs/causal_tune_linear.py) carry an IDENTIFIABILITY MASK: at each
tune, the columns of the training window that are constant, or exact byte-copies of an earlier
kept column, are removed from the fit.  Trees and the LSTM saw every column.  In the one-bar-
per-session design many columns are constant or duplicated over a window (at 16:00, 2018-06 ..
2024-04: live-feasible 64 constant + 25 copies of 232; all features 118 + 148 of 628), which a
tree still sees through column subsampling and split ties, and an LSTM through its input layer.

User decision 2026-09-29: every per-bar model class sees the same effective design.  This module
is the one rule, applied to the window a fit is trained on:

  * a column is CONSTANT on the window if every row has the same bytes as its first row
    (float64; -0.0 and 0.0 differ, NaN compares by its bit pattern);
  * a column is a COPY if its bytes over the window equal those of an earlier kept column;
  * kept = every other column, in column order (the first occurrence of a copy is kept).

It is a pure function of the window rows, so it is causal (the window is [t - W, t)), and the
same window always gives the same mask, whatever process, chunk or cluster computes it.
"""

from __future__ import annotations

import numpy as np


def window_keep(Xw: np.ndarray) -> np.ndarray:
    """Column indices a model fitted on the window ``Xw`` (rows x columns) may use."""
    X = np.ascontiguousarray(
        np.asarray(Xw, dtype=np.float64).T
    )  # one column per row, contiguous
    if X.shape[1] == 0:
        return np.arange(X.shape[0], dtype=np.int64)
    bits = X.view(np.uint64)
    const = (bits == bits[:, :1]).all(axis=1)
    keep: list[int] = []
    seen: set[bytes] = set()
    for j in range(X.shape[0]):
        if const[j]:
            continue
        key = X[j].tobytes()
        if key in seen:
            continue
        seen.add(key)
        keep.append(j)
    return np.asarray(keep, dtype=np.int64)


def scatter(
    values: np.ndarray, keep: np.ndarray, p: int, fill: float = 0.0
) -> np.ndarray:
    """Place per-kept-column values (last axis) back on all ``p`` columns; dropped columns get
    ``fill`` (0.0: a column the model never saw contributes nothing and has no importance).
    A trailing extra column (e.g. TreeSHAP's expected value) is carried over unchanged."""
    v = np.asarray(values, dtype=np.float64)
    extra = v.shape[-1] - len(keep)
    if extra not in (0, 1):
        raise ValueError(
            f"values have {v.shape[-1]} columns for {len(keep)} kept columns"
        )
    out = np.full(v.shape[:-1] + (p + extra,), fill, dtype=np.float64)
    out[..., keep] = v[..., : len(keep)]
    if extra:
        out[..., p] = v[..., -1]
    return out
