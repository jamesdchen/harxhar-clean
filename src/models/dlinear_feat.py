"""Augmented DLinear on a feature bucket: ridge base ⊕ learned RV-lag filter (rank-1, splittable).

The design row for bar t is the bucket's feature vector concatenated with a free length-L filter
over the recent RV path::

    X_t = [ feature_row_t ]  ⊕  [ rv_{t-1}, rv_{t-2}, …, rv_{t-L} ]      (p = n_feats + L)

with ``rv`` = the cache's ``har_ma_1`` column (the causal shift(1) 1-bar RV, already in the same
per-slot rank space as the features). This is the ridge base AUGMENTED with the *sequence* channel
HAR throws away — "HAR with free per-lag weights" bolted onto the full state-feature model. A
single ridge penalty keeps it a closed-form least-squares fit, so the sliding walk-forward is EXACT
under rank-1 sufficient-statistic updates (``RollingLeastSquares``) — no SGD, no GPU.

Why this one is worth SPLITTING (unlike the univariate ``dlinear_ols``): all_buckets is 529 features,
so with L=240 the fit is p≈769 -> roll is O(p²)/bar and each solve O(p³); the full panel grinds for
a minute-plus serially. The OOS timeline is embarrassingly parallel, so it fans across cores/jobs:

  * LOCAL      -> ``n_jobs > 1``: each block is a separate process that mmap-loads ONLY its own
                  ``[a-W, b)`` slice of the cache and builds its own design (no multi-GB pickling).
  * CLUSTER    -> submit time-chunks as an hpc-agent job-array: each task runs ``compute`` over one
                  ``start``/``end`` OOS sub-range and writes a chunk CSV + ``*_reduce.json`` monoid
                  partial (``save_chunk_reduce``); the campaign folds the ``(count, sum)`` partials.
                  This is the scalable axis — one serial task per chunk, same idiom as the trees.

Do NOT over-split: each block re-seeds with an O(W·p²) ``init_window`` rebuild, and that overhead
grows with p; ~one block per core is the sweet spot, not dozens of tiny chunks.

NOTE rank-1 requires the L2 (ridge) penalty. An L1/enet-augmented filter has no closed-form rank-1
update — that path is the reclasso online homotopy, not this module.
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from src.models.rolling_least_squares import RollingLeastSquares

# pandas + the metrics helpers are imported lazily inside compute(); chunk-parallel workers reach
# only _predict_block_feat (numpy + the rolling solver), so their re-import stays cheap.

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

CACHE_ROOT_DEFAULT = "results/covid_imp_rank"

DEFAULT = {
    "bucket": "all_buckets",
    "context_len": 240,  # L: RV-lag filter length (0 -> plain feature ridge)
    "train_window": 24000,  # W: rolling training window (bars)
    "alpha": 1.0,  # ridge penalty
    "refit": 480,  # solve cadence (bars)
    "fit_intercept": True,
    "lag_col": "har_ma_1",  # cache column used as the causal RV path for the lag filter
}


def _cache_paths(cache_root: str, bucket: str) -> tuple[str, str, str, str]:
    b = f"{cache_root}/{bucket}"
    return f"{b}/X_imp.npy", f"{b}/y.npy", f"{b}/base.npy", f"{b}/meta.json"


def _build_design(
    cache_root: str, bucket: str, lo: int, hi: int, L: int, lag_idx: int
) -> np.ndarray:
    """Design rows for global bars ``[lo, hi)`` = ``[feature_row] ⊕ [rv_{t-1..t-L}]``. Requires
    ``lo >= L``. mmap-loads only the ``[lo-L, hi)`` slice it needs (no full-matrix read)."""
    x_path, *_ = _cache_paths(cache_root, bucket)
    X = np.load(x_path, mmap_mode="r")
    feat = np.ascontiguousarray(X[lo:hi], dtype=np.float64)  # (hi-lo, n_feats)
    r = np.asarray(X[lo - L : hi, lag_idx], dtype=np.float64)  # (hi-lo+L,)
    # swv[m] = r[m : m+L]; row k (bar t=lo+k) -> r[k : k+L] = global [t-L, t) = [rv_{t-L} … rv_{t-1}]
    lagblk = sliding_window_view(r, L)[: hi - lo]
    return np.ascontiguousarray(np.hstack([feat, lagblk]))


def _roll_predict(
    D: np.ndarray,
    y: np.ndarray,
    off: int,
    W: int,
    a: int,
    b: int,
    alpha: float,
    refit: int,
    fit_intercept: bool,
    t_min: int,
) -> tuple[int, np.ndarray]:
    """Rank-1 rolling ridge over global OOS bars ``[a, b)``. ``D``/``y`` are LOCAL arrays covering
    global bars ``[off, …)`` (``D[i-off]`` is bar ``i``). ONE exact ``init_window`` seeds the block
    at its start; each later bar tracks the desired window ``[max(t_min,i-W), i)`` and slides it with
    an O(p²) ``grow`` while it is still filling (no drop) or ``roll`` once full — never re-building.
    Solve on the global cadence anchor so the prediction set is independent of block boundaries."""
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


def _predict_block_feat(
    cache_root: str,
    bucket: str,
    W: int,
    a: int,
    b: int,
    L: int,
    lag_idx: int,
    alpha: float,
    refit: int,
    fit_intercept: bool,
) -> tuple[int, np.ndarray]:
    """Worker: predict OOS bars ``[a, b)``, mmap-loading ONLY the ``[max(L,a-W), b)`` slice it needs."""
    off = max(L, a - W)
    D = _build_design(cache_root, bucket, off, b, L, lag_idx)
    _, y_path, *_ = _cache_paths(cache_root, bucket)
    y = np.asarray(np.load(y_path, mmap_mode="r")[off:b], dtype=np.float64)
    return _roll_predict(D, y, off, W, a, b, alpha, refit, fit_intercept, L)


def run_backtest(
    cache_root: str,
    bucket: str,
    N: int,
    L: int,
    W: int,
    lag_idx: int,
    alpha: float,
    refit: int,
    fit_intercept: bool,
    oos_lo: int,
    oos_hi: int,
    n_jobs: int,
) -> np.ndarray:
    """Walk-forward augmented-ridge over OOS bars ``[oos_lo, oos_hi)``; returns preds of that length.
    ``n_jobs > 1`` splits the range into contiguous blocks, one process each (mmap slice per block)."""
    if W < L:
        raise ValueError(f"train_window ({W}) must be >= context_len ({L})")
    if n_jobs <= 1:
        _, preds = _predict_block_feat(
            cache_root,
            bucket,
            W,
            oos_lo,
            oos_hi,
            L,
            lag_idx,
            alpha,
            refit,
            fit_intercept,
        )
        return preds

    # Snap interior edges to `refit` multiples so each block's seed-solve lands on a global cadence
    # bar -> the prediction set is identical for any n_jobs. Block 0 keeps the true OOS start.
    raw = np.linspace(oos_lo, oos_hi, n_jobs + 1)
    edges = [oos_lo]
    for e in raw[1:-1]:
        snapped = int(np.ceil(e / refit) * refit)
        if edges[-1] < snapped < oos_hi:
            edges.append(snapped)
    edges.append(oos_hi)
    blocks = [(edges[j], edges[j + 1]) for j in range(len(edges) - 1)]
    logger.info(
        f"chunk-parallel: {len(blocks)} blocks over {n_jobs} procs (each mmap-loads its own slice)"
    )

    preds = np.empty(oos_hi - oos_lo, dtype=np.float64)
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futs = [
            ex.submit(
                _predict_block_feat,
                cache_root,
                bucket,
                W,
                a,
                b,
                L,
                lag_idx,
                alpha,
                refit,
                fit_intercept,
            )
            for a, b in blocks
        ]
        for f in futs:
            a, blk = f.result()
            preds[a - oos_lo : a - oos_lo + len(blk)] = blk
    return preds


def compute(args) -> None:
    from src.evaluation.metrics import (
        build_results_dataframe,
        calculate_metrics,
        save_chunk_reduce,
    )

    def _p(
        key: str,
    ):  # config value from args, else the DEFAULT (None-safe, unlike `or`)
        v = getattr(args, key, None)
        return DEFAULT[key] if v is None else v

    cache_root = getattr(args, "cache_root", None) or CACHE_ROOT_DEFAULT
    bucket = str(_p("bucket"))
    L, W = int(_p("context_len")), int(_p("train_window"))
    alpha = float(_p("alpha"))
    refit = int(_p("refit"))
    fit_intercept = bool(_p("fit_intercept"))
    lag_col = str(_p("lag_col"))
    n_jobs = int(
        getattr(args, "n_jobs", None) or 1
    )  # serial default (see dlinear_ols note)

    feats = json.load(open(_cache_paths(cache_root, bucket)[3]))["feats"]
    lag_idx = feats.index(lag_col)
    _, y_path, base_path, _ = _cache_paths(cache_root, bucket)
    y_all = np.load(y_path, mmap_mode="r")
    N = int(y_all.shape[0])

    # OOS range: default [W, N); start/end carve a sub-range so a cluster job-array task can own one
    # time-chunk (the halo bars before `start` are still loaded internally to seed the roll).
    oos_lo = max(W, int(args.start)) if getattr(args, "start", 0) else W
    oos_hi = N if getattr(args, "end", -1) in (None, -1) else min(int(args.end), N)
    if oos_hi <= oos_lo:
        raise ValueError(f"empty OOS range [{oos_lo}, {oos_hi})")

    logger.info(
        f"aug-ridge bucket={bucket} p={len(feats)}+{L} W={W} alpha={alpha} refit={refit} "
        f"OOS=[{oos_lo},{oos_hi}) n_jobs={n_jobs}"
    )
    t0 = time.time()
    preds = run_backtest(
        cache_root,
        bucket,
        N,
        L,
        W,
        lag_idx,
        alpha,
        refit,
        fit_intercept,
        oos_lo,
        oos_hi,
        n_jobs,
    )
    logger.info(f"Backtest complete in {time.time() - t0:.2f}s ({len(preds)} preds)")

    y_oos = np.asarray(y_all[oos_lo:oos_hi], dtype=np.float64)
    base_oos = np.asarray(
        np.load(base_path, mmap_mode="r")[oos_lo:oos_hi], dtype=np.float64
    )
    horizon = int(getattr(args, "horizon", 1) or 1)
    df = build_results_dataframe(
        preds, y_oos, np.arange(oos_lo, oos_hi), base_oos, horizon
    )

    out = args.output_file
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    df.to_csv(out, index=False)
    save_chunk_reduce(
        df, out
    )  # (qlike_count, qlike_sum) monoid partial for the cluster reduce
    metrics = calculate_metrics(df)
    with open(os.path.join(os.path.dirname(out) or ".", "metrics.json"), "w") as f:
        json.dump(metrics, f)
    logger.info(f"Saved {len(df)} rows -> {out}  QLIKE={metrics.get('qlike')}")
