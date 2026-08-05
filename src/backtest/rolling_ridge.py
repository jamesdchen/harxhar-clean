"""Chunk-parallel rank-1 rolling-ridge backtest — the reusable fast path for L2-linear models.

Extracted from the (removed) dlinear executors so any linear-on-features model can share it: the
rank-1 sliding-window ridge readout over a design matrix (``RollingLeastSquares`` grow/roll — exact
vs a per-bar fresh Ridge, up to the solve cadence), plus embarrassingly-parallel time-chunking with
mmap workers (no multi-GB pickling). Used by ``src.models.bucket_ridge`` (ridge on a feature bucket)
and the ``rg_lru_reservoir`` readout.

The rank-1 SLIDING bookkeeping is penalty-agnostic: L1 (lasso/enet) maintains the same Gram and
correlation under rank-1 add/remove (``reclasso_har.GramState``) and updates its solution warm-started
per rank-1 data change via the Garrigues-El Ghaoui online homotopy (``reclasso_har.enet_online``) —
the L1 analog of the Sherman-Morrison solve update, so this module's chunk-parallel + no-warmup-storm
machinery transfers to it. What L1 lacks is ridge's FIXED closed-form β-update: its per-bar step
re-walks the piecewise path (data-dependent active-set pivots, variable cost) and drifts under rank-1
updates, so it needs a periodic exact refresh rather than ridge's single O(p²) inverse update.
"""

from __future__ import annotations

import os
import tempfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from src.models.rolling_least_squares import RollingLeastSquares


def roll_predict(
    D: np.ndarray,
    y: np.ndarray,
    off: int,
    W: int,
    a: int,
    b: int,
    alpha: float,
    refit: int,
    fit_intercept: bool,
    t_min: int = 0,
) -> tuple[int, np.ndarray]:
    """Rank-1 rolling ridge over global OOS bars ``[a, b)``. ``D``/``y`` are LOCAL arrays covering
    global bars ``[off, …)`` (``D[i-off]`` is bar ``i``). ONE ``init_window`` seeds the block; each
    later bar tracks the desired window ``[max(t_min, i-W), i)`` and slides it with an O(p²) ``grow``
    while still filling (no drop) or ``roll`` once full — never re-building. Solve on the global
    cadence anchor so the prediction set is independent of block boundaries. ``t_min`` is the first
    globally-valid design row (0 for a full feature matrix; >0 when leading rows are undefined)."""
    rls = RollingLeastSquares(alpha=alpha, fit_intercept=fit_intercept, reinit_every=0)
    preds = np.empty(b - a, dtype=np.float64)
    cur_lo = max(t_min, a - W)  # oldest bar currently in the window
    rls.init_window(D[cur_lo - off : a - off], y[cur_lo - off : a - off])
    rls.solve()
    preds[0] = rls.predict_one(D[a - off])
    for k in range(1, b - a):
        i = a + k
        lo = max(t_min, i - W)  # desired window [lo, i); prev was [cur_lo, i-1)
        if lo > cur_lo:  # full window slides: add bar i-1, drop bar cur_lo
            rls.roll(D[i - 1 - off], y[i - 1 - off], D[cur_lo - off], y[cur_lo - off])
        else:  # window still filling: add-only (rank-1 grow, O(p²))
            rls.grow(D[i - 1 - off], y[i - 1 - off])
        cur_lo = lo
        if i % refit == 0:
            rls.solve()
        preds[k] = rls.predict_one(D[i - off])
    return a, preds


def _predict_block_mmap(
    design_path, y_path, W, a, b, alpha, refit, fit_intercept, t_min
):
    """Chunk worker: mmap ONLY the ``[max(t_min, a-W), b)`` slice of the scratch design + targets."""
    off = max(t_min, a - W)
    D = np.ascontiguousarray(np.load(design_path, mmap_mode="r")[off:b])
    y = np.asarray(np.load(y_path, mmap_mode="r")[off:b], dtype=np.float64)
    return roll_predict(D, y, off, W, a, b, alpha, refit, fit_intercept, t_min)


def parallel_ridge(
    D: np.ndarray,
    y: np.ndarray,
    W: int,
    alpha: float,
    refit: int,
    fit_intercept: bool,
    oos_lo: int,
    oos_hi: int,
    n_jobs: int,
    t_min: int = 0,
) -> np.ndarray:
    """Serial (``n_jobs<=1``) or chunk-parallel rank-1 rolling ridge over OOS ``[oos_lo, oos_hi)``.
    Parallel persists ``D``/``y`` to scratch memmaps and fans contiguous blocks over processes, each
    mmap-loading only its slice; edges are snapped to ``refit`` multiples so the prediction set is
    n_jobs-independent. Default SERIAL (the cluster fans configs/time-chunks across jobs instead)."""
    if W < t_min:
        raise ValueError(f"train_window ({W}) must be >= t_min ({t_min})")
    if n_jobs <= 1:
        _, preds = roll_predict(
            np.ascontiguousarray(D),
            np.asarray(y, dtype=np.float64),
            0,
            W,
            oos_lo,
            oos_hi,
            alpha,
            refit,
            fit_intercept,
            t_min,
        )
        return preds

    with tempfile.TemporaryDirectory() as scratch:
        dpath = os.path.join(scratch, "_design.npy")
        ypath = os.path.join(scratch, "_y.npy")
        np.save(dpath, np.ascontiguousarray(D))
        np.save(ypath, np.asarray(y, dtype=np.float64))
        raw = np.linspace(oos_lo, oos_hi, n_jobs + 1)
        edges = [oos_lo]
        for e in raw[1:-1]:
            s = int(np.ceil(e / refit) * refit)
            if edges[-1] < s < oos_hi:
                edges.append(s)
        edges.append(oos_hi)
        blocks = [(edges[j], edges[j + 1]) for j in range(len(edges) - 1)]
        preds = np.empty(oos_hi - oos_lo, dtype=np.float64)
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futs = [
                ex.submit(
                    _predict_block_mmap,
                    dpath,
                    ypath,
                    W,
                    a,
                    b,
                    alpha,
                    refit,
                    fit_intercept,
                    t_min,
                )
                for a, b in blocks
            ]
            for f in futs:
                a, blk = f.result()
                preds[a - oos_lo : a - oos_lo + len(blk)] = blk
    return preds
