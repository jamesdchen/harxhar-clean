"""Causal kNN analog-residual stage on the close (h16-19) -- PSMC-shadowing-lite / regime gating.

For each close bar, find the K causal nearest neighbours (in a low-dim vol-REGIME embedding) among PAST
close bars beyond a strict embargo, average their realised Hero-A leftover residual r = y_true - pred_adj,
and add that analog forecast to pred_adj. Tests whether regime-similar history carries residual structure
the base+tree miss -- a non-local, full-history-similarity function the tree cannot reconstruct.

LEAKAGE GUARDS (analog/kNN is the #1 source of look-ahead artifacts):
  * neighbours drawn ONLY from past close bars with k' <= k - EMBARGO, EMBARGO >= HAR max lag (3125 bars)
    so neighbour residuals are fully realised + no overlapping-window bleed;
  * SHUFFLE PLACEBO: average shuffled residuals over the SAME neighbours -> breaks the state<->residual
    link, must give ~0 improvement. A gain there = artifact, not signal.
Embedding standardisation is metric-only (scales the distance, not the target); embargo controls target leak.
"""

import _bootstrap  # noqa: F401  (repo root + siblings on sys.path)

import glob
import json
import os

import numpy as np
import pandas as pd

from resid_amortized import CACHE_ROOT, PERIODS_PER_DAY
from src.evaluation.metrics import apply_duan_smearing

CELL = os.environ.get("KNN_CELL", "xgb_all_buckets_tw1000_enetreg2_rf480_slim")
LBL = os.environ.get("KNN_LBL", "resid_subset_heroA_d8")
TRAIN_WIN = 1000 * PERIODS_PER_DAY
EMBARGO = 3125  # HAR max lag (bars) -> neighbours >= ~65 trading days in the past
EMB_COLS = ["har_ma_1", "har_ma_5", "har_ma_25", "har_ma_125", "cumrv_x_close"]


def qlike(preds, y, base):
    pr, tr = apply_duan_smearing(np.asarray(preds, dtype=np.float64), y, base)
    m = (tr > 0) & (pr > 0)
    rr = tr[m] / pr[m]
    return float(np.mean(rr - np.log(rr) - 1.0))


def main() -> None:
    d = f"{CACHE_ROOT}/{CELL}"
    feats = json.load(open(f"{d}/feats.json"))
    Xs = np.load(f"{d}/Xs.npy", mmap_mode="r")

    parts = [
        pd.read_csv(f) for f in glob.glob(f"results/resid_ab/{CELL}/{LBL}/chunk_*.csv")
    ]
    A = pd.concat(parts, ignore_index=True).sort_values("k")
    k = A["k"].to_numpy().astype(np.int64)
    pred_adj = A["pred_adj"].to_numpy(np.float64)
    y_true = A["y_true"].to_numpy(np.float64)
    base = A["base"].to_numpy(np.float64)
    t = k + TRAIN_WIN

    hour = np.asarray(Xs[t, feats.index("hour")], dtype=np.float64)
    E = np.column_stack(
        [np.asarray(Xs[t, feats.index(c)], dtype=np.float64) for c in EMB_COLS] + [hour]
    )
    E = (E - E.mean(0)) / (E.std(0) + 1e-9)
    r = y_true - pred_adj
    close = (hour >= 16) & (hour <= 19)
    ci = np.where(close)[0]
    kc, Ec, rc = k[ci], np.ascontiguousarray(E[ci]), r[ci]
    nC = len(ci)
    print(
        "OOS rows=%d close bars=%d  embargo=%d  emb_dim=%d"
        % (len(k), nC, EMBARGO, E.shape[1]),
        flush=True,
    )

    def anhat(rsrc, K):
        ah = np.zeros(nC, dtype=np.float64)
        ptr = 0  # eligible = close bars [0:ptr) with kc <= kc[i]-EMBARGO (kc ascending)
        for i in range(nC):
            thr = kc[i] - EMBARGO
            while ptr < nC and kc[ptr] <= thr:
                ptr += 1
            if ptr < K:
                continue
            dd = Ec[:ptr] - Ec[i]
            dist = np.einsum("ij,ij->i", dd, dd)
            nn = np.argpartition(dist, K)[:K]
            ah[i] = rsrc[nn].mean()
        return ah

    q0 = qlike(pred_adj, y_true, base)
    q0c = qlike(pred_adj[ci], y_true[ci], base[ci])
    print("Hero A: full=%.5f  h16-19=%.5f" % (q0, q0c), flush=True)
    rng = np.random.RandomState(0)
    for K in (25, 50, 100):
        ah = anhat(rc, K)
        new = pred_adj.copy()
        new[ci] += ah
        q1 = qlike(new, y_true, base)
        q1c = qlike(new[ci], y_true[ci], base[ci])
        ahs = anhat(rc[rng.permutation(nC)], K)
        news = pred_adj.copy()
        news[ci] += ahs
        qs = qlike(news, y_true, base)
        print(
            "K=%3d full %.5f (d%+.6f) | SHUFFLE %.5f (d%+.6f) | h16-19 %.5f (d%+.6f)  ah|mean|=%.2e"
            % (K, q1, q1 - q0, qs, qs - q0, q1c, q1c - q0c, np.abs(ah[ah != 0]).mean()),
            flush=True,
        )
    print("KNN_ANALOG_DONE", flush=True)


if __name__ == "__main__":
    main()
