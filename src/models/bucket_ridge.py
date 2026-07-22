"""Ridge on a feature bucket (e.g. all_buckets = all_features), rank-1 + chunk-parallel.

The fast local linear-on-all_features model: a plain walk-forward ridge over the bucket's causal
feature matrix, fit by the exact rank-1 sliding-window solver and split across cores via the shared
``src.backtest.rolling_ridge`` machinery (296s -> ~42s on all_features at 8 procs). This is just OLS
/ ridge — no RV-lag filter (that "DLinear" augmentation was not more expressive than OLS), so the
model IS the linear baseline, now with the chunk-parallel speedup.

Reads ``results/covid_imp_rank/{bucket}/{X_imp,y,base}.npy``; fits ``adj_RV`` by ridge (LS + Duan,
the Ridge-base convention), writes the chunk CSV + ``*_reduce.json`` monoid partial so it also fans
across cluster job-array time-chunks (same idiom as the trees). L1 (lasso/enet) can't use this rank-1
path — that's the reclasso homotopy.
"""

from __future__ import annotations

import json
import logging
import os
import time

import numpy as np

from src.backtest.rolling_ridge import parallel_ridge

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

CACHE_ROOT_DEFAULT = "results/covid_imp_rank"

DEFAULT = {
    "bucket": "all_buckets",
    "train_window": 24000,  # W: rolling training window (bars)
    "alpha": 1.0,  # ridge penalty
    "refit": 480,  # solve cadence (bars)
    "fit_intercept": True,
    "standardize": True,  # causal train-window column standardization (one comparable alpha)
}


def compute(args) -> None:
    from src.evaluation.metrics import (
        build_results_dataframe,
        calculate_metrics,
        save_chunk_reduce,
    )

    def _p(key: str):
        v = getattr(args, key, None)
        return DEFAULT[key] if v is None else v

    cache_root = getattr(args, "cache_root", None) or CACHE_ROOT_DEFAULT
    bucket = str(_p("bucket"))
    W = int(_p("train_window"))
    alpha, refit = float(_p("alpha")), int(_p("refit"))
    fit_intercept, standardize = bool(_p("fit_intercept")), bool(_p("standardize"))
    n_jobs = int(getattr(args, "n_jobs", None) or 1)

    b = f"{cache_root}/{bucket}"
    X = np.asarray(np.load(f"{b}/X_imp.npy"), dtype=np.float64)
    y = np.asarray(np.load(f"{b}/y.npy"), dtype=np.float64)
    base = np.asarray(np.load(f"{b}/base.npy"), dtype=np.float64)
    N = X.shape[0]

    oos_hi = N if getattr(args, "end", -1) in (None, -1) else min(int(args.end), N)
    if standardize:  # causal: stats from the training region only
        mu, sd = X[:W].mean(0), X[:W].std(0)
        X = (X - mu) / np.where(sd > 0, sd, 1.0)

    logger.info(
        f"bucket-ridge {bucket} p={X.shape[1]} W={W} alpha={alpha} refit={refit} OOS=[{W},{oos_hi}) n_jobs={n_jobs}"
    )
    t0 = time.time()
    preds = parallel_ridge(
        X, y, W, alpha, refit, fit_intercept, W, oos_hi, n_jobs, t_min=0
    )
    logger.info(f"Backtest complete in {time.time() - t0:.2f}s ({len(preds)} preds)")

    y_oos, base_oos = y[W:oos_hi], base[W:oos_hi]
    horizon = int(getattr(args, "horizon", 1) or 1)
    df = build_results_dataframe(preds, y_oos, np.arange(W, oos_hi), base_oos, horizon)
    out = args.output_file
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    df.to_csv(out, index=False)
    save_chunk_reduce(df, out)
    metrics = calculate_metrics(df)
    with open(os.path.join(os.path.dirname(out) or ".", "metrics.json"), "w") as f:
        json.dump(metrics, f)
    logger.info(f"Saved {len(df)} rows -> {out}  QLIKE={metrics.get('qlike')}")
