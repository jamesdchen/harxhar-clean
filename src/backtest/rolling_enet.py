"""Chunk-parallel rolling elastic-net backtest — the reusable fast path for L1-linear models.

The L1 sibling of ``rolling_ridge``: a parallelization of the ``_cadence_enet_har_unpen`` incumbent
(sklearn coordinate descent, warm-started across refits — the warm chain converges in ~2 CD sweeps per
refit, far faster than a cold solve). Refits are INDEPENDENT OPTIMIZATION PROBLEMS, so the cadence grid
fans across processes: each chunk cold-seeds its own warm chain and walks its slice. ``n_jobs=1`` is the
single global chain — bit-for-bit the incumbent; ``n_jobs>1`` agrees with it to CD tolerance (a chunk's
cold seed converges to the SAME optimum the incumbent's warm continuation sits near; only the iterate
count differs).

Per refit at ``t_r`` (window ``[t_r-W, t_r)``) the HAR block is UNPENALIZED via FWL, exactly the
incumbent convention: profile HAR out by OLS (``pinv`` — rank-deficient-safe for the collinear HAR
block), fit the L1+L2 penalty on the HAR-residualized exog, recover the HAR coefs by OLS. Prediction is
block-constant over ``[t_r, t_r+refit)``. (The Gram-form exact homotopy — ``reclasso_har.lasso_homotopy``,
whose solves are O(p³) and independent of W — is the alternative for HUGE-W / small-p regimes; at this
bucket's W=24000 / p=529 the warm CD chain is ~10x faster because the dense solution makes the cold
homotopy path walk ~500 pivots.)
"""

from __future__ import annotations

import os
import tempfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np


def refit_chain(
    D: np.ndarray,
    y: np.ndarray,
    starts: list[int],
    ends: list[int],
    W: int,
    alpha: float,
    l1_ratio: float,
    har_idx: np.ndarray,
    exog_idx: np.ndarray,
) -> list[np.ndarray]:
    """The incumbent cadence loop over a slice of refits: a cold-seeded, warm-started CD chain.

    HAR (``har_idx``) by unpenalized OLS (FWL), the exog (``exog_idx``) by ``ElasticNet(alpha,
    l1_ratio)`` on the HAR-residualized design — the exact ``_cadence_enet_har_unpen`` numerics. With
    an empty ``har_idx`` this is a plain centered elastic net over all columns. Each returned block is
    the block-constant prediction over ``[t_r, t_end)``."""
    from sklearn.linear_model import ElasticNet

    en = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        fit_intercept=False,
        warm_start=True,
        max_iter=5000,
        tol=1e-4,
    )
    p = D.shape[1]
    has_har = har_idx.size > 0
    blocks = []
    for t_r, t_end in zip(starts, ends):
        Xw = D[t_r - W : t_r]
        yw = y[t_r - W : t_r]
        xbar = Xw.mean(0)
        ybar = float(yw.mean())
        Xc = Xw - xbar
        yc = yw - ybar
        if not has_har:  # no HAR to unpenalize -> plain enet on everything
            en.fit(Xc, yc)
            beta = en.coef_.astype(np.float64, copy=True)
        else:
            H = np.ascontiguousarray(Xc[:, har_idx])
            E = np.ascontiguousarray(Xc[:, exog_idx])
            Hpinv = np.linalg.pinv(H)
            bH_y = Hpinv @ yc
            bH_E = Hpinv @ E
            en.fit(E - H @ bH_E, yc - H @ bH_y)
            bE = en.coef_
            beta = np.zeros(p, dtype=np.float64)
            beta[har_idx] = bH_y - bH_E @ bE
            beta[exog_idx] = bE
        blocks.append(np.asarray(D[t_r:t_end]) @ beta + (ybar - xbar @ beta))
    return blocks


def _refit_block_worker(
    design_path: str,
    y_path: str,
    starts: list[int],
    ends: list[int],
    W: int,
    alpha: float,
    l1_ratio: float,
    har_idx: np.ndarray,
    exog_idx: np.ndarray,
) -> tuple[list[int], list[np.ndarray]]:
    """Chunk worker: mmap the scratch design/targets and run the chunk's warm-CD refit chain."""
    D = np.load(design_path, mmap_mode="r")
    yv = np.asarray(np.load(y_path, mmap_mode="r"), dtype=np.float64)
    blocks = refit_chain(D, yv, starts, ends, W, alpha, l1_ratio, har_idx, exog_idx)
    return starts, blocks


def parallel_enet(
    D: np.ndarray,
    y: np.ndarray,
    W: int,
    alpha: float,
    l1_ratio: float,
    refit: int,
    har_idx: np.ndarray,
    oos_lo: int,
    oos_hi: int,
    n_jobs: int,
) -> np.ndarray:
    """Serial (``n_jobs<=1``) or chunk-parallel cadence-refit elastic net over OOS ``[oos_lo, oos_hi)``.

    Refits land on the grid ``range(oos_lo, oos_hi, refit)``. Serial runs ONE warm-started CD chain
    over every refit — bit-for-bit the ``_cadence_enet_har_unpen`` incumbent. Parallel splits the grid
    into ``n_jobs`` contiguous chunks, each cold-seeding its own chain: the chunk seeds converge to the
    same optima as the incumbent's warm continuation, so the prediction agrees with serial to CD
    tolerance (~1e-4 on coefs), not bit-exactly. Default SERIAL (the cluster fans configs/time-chunks
    across jobs instead)."""
    starts = list(range(oos_lo, oos_hi, refit))
    ends = [min(s + refit, oos_hi) for s in starts]
    preds = np.empty(oos_hi - oos_lo, dtype=np.float64)
    har_idx = np.asarray(har_idx, dtype=int)
    exog_idx = np.asarray(
        [j for j in range(D.shape[1]) if j not in set(har_idx.tolist())], dtype=int
    )

    def _collect(res_starts: list[int], res_blocks: list[np.ndarray]) -> None:
        for tr, blk in zip(res_starts, res_blocks):
            preds[tr - oos_lo : tr - oos_lo + len(blk)] = blk

    if n_jobs <= 1:
        blocks = refit_chain(D, y, starts, ends, W, alpha, l1_ratio, har_idx, exog_idx)
        _collect(starts, blocks)
        return preds

    with tempfile.TemporaryDirectory() as scratch:
        dpath = os.path.join(scratch, "_design.npy")
        ypath = os.path.join(scratch, "_y.npy")
        np.save(dpath, np.ascontiguousarray(D))
        np.save(ypath, np.asarray(y, dtype=np.float64))
        chunks = np.array_split(np.arange(len(starts)), n_jobs)
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futs = [
                ex.submit(
                    _refit_block_worker,
                    dpath,
                    ypath,
                    [starts[i] for i in ch],
                    [ends[i] for i in ch],
                    W,
                    alpha,
                    l1_ratio,
                    har_idx,
                    exog_idx,
                )
                for ch in chunks
                if len(ch)
            ]
            for f in futs:
                res_starts, res_blocks = f.result()
                _collect(res_starts, res_blocks)
    return preds
