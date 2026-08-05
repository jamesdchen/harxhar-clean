"""Elastic net on a feature bucket (all_buckets = all_features), HAR-unpenalized + chunk-parallel.

The L1 sibling of ``bucket_ridge``: a walk-forward elastic net over the bucket's causal feature matrix,
the ``_cadence_enet_har_unpen`` incumbent convention — HAR block UNPENALIZED (FWL), the rest L1+L2,
cadence-refit with block-constant prediction — sped up by the shared ``src.backtest.rolling_enet``
machinery (the incumbent's warm-started CD chain fanned across cores: each chunk cold-seeds its own
chain over a contiguous slice of the refit grid). Reads ``results/covid_imp_rank/{bucket}/{X_imp,y,
base}.npy`` (+ ``meta.json`` for the HAR mask); fits ``adj_RV`` (LS + Duan, the Ridge-base convention);
writes the chunk CSV + ``*_reduce.json`` monoid partial so it also fans across cluster job-array
time-chunks (same idiom as the trees).

Where ridge (``bucket_ridge``) gets a closed-form rank-1 inverse update, the enet's refits are simply
INDEPENDENT optimization problems, so the speedup is the embarrassingly-parallel refit grid —
``n_jobs=1`` reproduces the incumbent bit-for-bit; ``n_jobs>1`` agrees to CD tolerance.
"""

from __future__ import annotations

import json
import logging
import os
import time

import numpy as np

from src.backtest.rolling_enet import parallel_enet

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

CACHE_ROOT_DEFAULT = "results/covid_imp_rank"
HAR_FEATS = {
    "har_ma_1",
    "har_ma_5",
    "har_ma_25",
    "har_ma_125",
    "har_ma_625",
    "har_ma_3125",
}

DEFAULT = {
    "bucket": "all_buckets",
    "train_window": 24000,  # W: rolling training window (bars)
    "alpha": 0.001,  # elastic-net penalty (incumbent)
    "l1_ratio": 0.2,  # L1 mix (incumbent dense setting)
    "refit": 480,  # refit cadence (bars); prediction block-constant between refits
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
    alpha, l1_ratio = float(_p("alpha")), float(_p("l1_ratio"))
    refit = int(_p("refit"))
    standardize = bool(_p("standardize"))
    n_jobs = int(getattr(args, "n_jobs", None) or 1)

    b = f"{cache_root}/{bucket}"
    X = np.asarray(np.load(f"{b}/X_imp.npy"), dtype=np.float64)
    y = np.asarray(np.load(f"{b}/y.npy"), dtype=np.float64)
    base = np.asarray(np.load(f"{b}/base.npy"), dtype=np.float64)
    N = X.shape[0]
    feats = json.load(open(f"{b}/meta.json"))["feats"]
    har_idx = np.asarray([i for i, f in enumerate(feats) if f in HAR_FEATS], dtype=int)

    oos_hi = N if getattr(args, "end", -1) in (None, -1) else min(int(args.end), N)
    if standardize:  # causal: stats from the training region only
        mu, sd = X[:W].mean(0), X[:W].std(0)
        X = (X - mu) / np.where(sd > 0, sd, 1.0)

    logger.info(
        f"bucket-enet {bucket} p={X.shape[1]} n_har={har_idx.size} W={W} alpha={alpha} "
        f"l1={l1_ratio} refit={refit} OOS=[{W},{oos_hi}) n_jobs={n_jobs}"
    )
    t0 = time.time()
    preds = parallel_enet(X, y, W, alpha, l1_ratio, refit, har_idx, W, oos_hi, n_jobs)
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
