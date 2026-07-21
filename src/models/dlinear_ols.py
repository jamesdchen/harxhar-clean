"""DLinear as windowed ridge — the exact, rank-1, parallel-CPU "learned-HAR" control.

A DLinear with a single ridge penalty COLLAPSES to a plain linear filter over the lookback:
``pred = w_s·(x − MA(x)) + w_t·MA(x) = β·x`` (the trend/seasonal decomposition is a linear
reparameterization — a no-op unless the two blocks are penalized differently). So DLinear-with-
tied-reg ≡ **ridge regression on the length-L lag window** ``[rv_{t-1}, …, rv_{t-L}]`` = "HAR
with free per-lag weights instead of fixed daily/weekly/monthly buckets".

Because the fit is least-squares (not SGD-on-QLIKE), the sliding walk-forward is EXACT under
rank-1 sufficient-statistic updates: each bar adds one lag row and drops one, an ``O(L²)``
``roll`` of ``XᵀX``/``Xᵀy`` (``src.models.rolling_least_squares.RollingLeastSquares`` — the same
incremental ridge the production Ridge base uses). Coefficients refresh at a ``refit`` cadence
(``O(L³)`` solve). No GPU, no per-window retraining; the whole panel is seconds.

Two parallelism axes, both CPU:
  * rank-1 roll + cadence solve  -> fast SERIALLY (roll is O(L²), only ~N/refit solves);
  * chunk-parallel               -> the OOS timeline is split into contiguous blocks, each
                                    seeded independently from its own ``train_window`` halo and
                                    marched in its own process (``ProcessPoolExecutor``).

Fit target = ``adj_RV`` by least squares, then Duan-smeared to raw RV (identical retransform to
the Ridge base), so its QLIKE lands directly against the HAR/ridge floor. Matches the DL-executor
result contract (``save_dl_results``): ``preds`` of length ``N_chunk − train_window`` aligned to
chunk indices ``[train_window, N_chunk)``.

NOTE the decomposition collapse above holds only for a SINGLE ridge penalty. To recover DLinear's
actual trend/seasonal split you would penalize the two blocks differently (a per-column penalty
vector), which needs a small extension to RollingLeastSquares' scalar ``alpha``; left as a
follow-up — this module is the tied-penalty (pure learned-filter) control.
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

# NOTE: `load_and_transform` / `save_dl_results` (which pull pandas + the whole data stack) are
# imported lazily inside compute(), NOT at module top level. Chunk-parallel workers re-import
# this module under Windows spawn just to reach `_predict_block`; keeping the top-level imports to
# numpy + the rolling solver makes that re-import cheap (else every worker pays a ~15s pandas load).

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT = {
    "context_len": 240,  # L: number of lags in the filter
    "train_window": 24000,  # W: rolling training window (bars)
    "alpha": 1.0,  # ridge penalty (0 -> plain OLS filter)
    "refit": 480,  # solve cadence (bars); rolls keep XᵀX current between solves
    "fit_intercept": True,
}


def _predict_block(
    X_series: np.ndarray,
    y_series: np.ndarray,
    L: int,
    W: int,
    a: int,
    b: int,
    alpha: float,
    refit: int,
    fit_intercept: bool,
) -> tuple[int, np.ndarray]:
    """Predict OOS chunk indices ``[a, b)``. Seeded independently from the ``[a-W, a)`` halo,
    so blocks run in parallel. Rank-1 ``roll`` once the nominal window ``[i-W, i)`` is fully
    valid (``i-W >= L``); a ``init_window`` rebuild seeds the block and covers the warmup where
    the earliest lag rows are undefined. Exact vs a per-bar fresh Ridge, up to the solve cadence.
    """
    lag = sliding_window_view(
        X_series, L
    )  # lag[t] = X_series[t : t+L]; lagrow(i) = lag[i-L]
    rls = RollingLeastSquares(alpha=alpha, fit_intercept=fit_intercept, reinit_every=0)
    preds = np.empty(b - a, dtype=np.float64)

    first = True
    for k, i in enumerate(range(a, b)):
        # The window is the W target bars [i-W, i); the earliest valid lag row is L, so it is at
        # full size W only once i-W > L (STRICT: at i-W == L it is still growing add-only, and a
        # symmetric roll would drop a row never added). Seed each block + the warmup by an exact
        # rebuild; slide by a rank-1 roll only in true steady state.
        if (
            first or (i - W) <= L
        ):  # seed / warmup: rebuild valid bars [max(L, i-W), i) + solve
            lo = max(L, i - W)
            rls.init_window(lag[lo - L : i - L], y_series[lo:i])
            rls.solve()
            first = False
        else:  # steady state: add bar i-1, drop bar i-1-W, O(L²)
            rls.roll(
                lag[i - 1 - L], y_series[i - 1], lag[i - 1 - W - L], y_series[i - 1 - W]
            )
            if (
                i % refit == 0
            ):  # global anchor -> solve-bars independent of block boundaries
                rls.solve()
        preds[k] = rls.predict_one(lag[i - L])
    return a, preds


def run_backtest(
    X_series: np.ndarray,
    y_series: np.ndarray,
    L: int,
    W: int,
    alpha: float,
    refit: int,
    fit_intercept: bool,
    n_jobs: int,
) -> np.ndarray:
    """Walk-forward windowed-ridge backtest over OOS indices ``[W, N)``; returns ``preds`` of
    length ``N - W``. ``n_jobs > 1`` splits ``[W, N)`` into contiguous blocks, one per process."""
    N = len(X_series)
    if W < L:
        raise ValueError(f"train_window ({W}) must be >= context_len ({L})")
    oos_lo, oos_hi = W, N
    n_oos = oos_hi - oos_lo

    if n_jobs <= 1:
        _, preds = _predict_block(
            X_series, y_series, L, W, oos_lo, oos_hi, alpha, refit, fit_intercept
        )
        return preds

    # Snap interior block edges to multiples of `refit` so each block's seed-solve lands on a
    # global cadence bar -> the set of solve-bars (and thus predictions) is identical for any
    # n_jobs. Block 0 always starts at oos_lo (the same seed the serial run uses).
    raw = np.linspace(oos_lo, oos_hi, n_jobs + 1)
    edges = [oos_lo]
    for e in raw[1:-1]:
        snapped = int(np.ceil(e / refit) * refit)
        if edges[-1] < snapped < oos_hi:
            edges.append(snapped)
    edges.append(oos_hi)
    blocks = [(edges[j], edges[j + 1]) for j in range(len(edges) - 1)]
    logger.info(
        f"chunk-parallel: {len(blocks)} blocks x ~{n_oos // max(1, len(blocks))} bars over {n_jobs} procs"
    )

    preds = np.empty(n_oos, dtype=np.float64)
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futs = [
            ex.submit(
                _predict_block,
                X_series,
                y_series,
                L,
                W,
                a,
                b,
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
    cfg = dict(DEFAULT)
    for k in ("context_len", "train_window", "alpha", "refit", "fit_intercept"):
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v
    from src.backtest.executor import load_and_transform

    L, W = int(cfg["context_len"]), int(cfg["train_window"])
    # Default SERIAL: the model runs the full panel on one core in seconds, and on the cluster
    # os.cpu_count() reports the NODE's cores (not the SLURM/SGE allocation) -> a null default
    # would oversubscribe a 1-core task. In-process parallelism is a local-laptop opt-in
    # (n_jobs>1); the cluster fans configs/time-chunks across jobs instead.
    n_jobs = int(getattr(args, "n_jobs", None) or 1)

    logger.info(f"Loading data from {args.data_path}")
    df, _ = load_and_transform(
        args.data_path,
        exog_cols=[],
        target_use_diurnal=True,
        target_winsor_window=240,
        dropna_with_exog=False,
    )
    adj_rv = df["adj_RV"].values.astype(np.float64)
    baseline = df["baseline"].values.astype(np.float64)
    dates = df["t"]

    if args.horizon > 1:
        shift = args.horizon - 1
        X_full, y_full = adj_rv[:-shift], adj_rv[shift:]
        dates = dates.iloc[:-shift].reset_index(drop=True)
        baseline = baseline[shift:]
    else:
        X_full = y_full = adj_rv

    start = args.start
    end = len(X_full) if args.end == -1 else args.end
    X_chunk = np.ascontiguousarray(X_full[start:end])
    y_chunk = np.ascontiguousarray(y_full[start:end])
    dates_chunk = dates.iloc[start:end].reset_index(drop=True)
    baselines_chunk = baseline[start:end]
    if W >= len(X_chunk):
        raise ValueError(f"train_window ({W}) >= chunk size ({len(X_chunk)})")

    logger.info(
        f"windowed-ridge L={L} W={W} alpha={cfg['alpha']} refit={cfg['refit']} n_jobs={n_jobs} rows={len(X_chunk)}"
    )
    t0 = time.time()
    preds = run_backtest(
        X_chunk,
        y_chunk,
        L,
        W,
        float(cfg["alpha"]),
        int(cfg["refit"]),
        bool(cfg["fit_intercept"]),
        n_jobs,
    )
    logger.info(f"Backtest complete in {time.time() - t0:.2f}s ({len(preds)} preds)")

    from src.backtest.dl_executor import save_dl_results

    save_dl_results(
        preds, y_chunk, dates_chunk, baselines_chunk, W, args.horizon, args.output_file
    )
    m = json.load(
        open(os.path.join(os.path.dirname(args.output_file) or ".", "metrics.json"))
    )
    logger.info(f"QLIKE={m.get('qlike')}")
