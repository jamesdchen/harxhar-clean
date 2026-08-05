"""Local relevance-REGRESSION test on the close leftover: does local fitting work, and does the FEATURE SET
matter? For each close bar, find the same K causal neighbours (embargoed, similarity on the HAR vol-state =
a recency-weighted history proxy), then fit a LOCAL RIDGE of the Hero-A leftover r on several feature sets and
compare -- holding neighbours fixed so only the feature set varies:
  intercept -- local mean (the naive K-mean that failed, now with large K)
  har       -- local HAR levels (the paper's 'fit a local HAR on the neighbours')  <- the assumption to test
  V         -- only the regime-VARYING features (har_ma_1/5, sumbipow, stocktwits)  <- my refinement
  all       -- HAR + V + cumrv (full local)
If har ~= V ~= all, local-fitting the stable HAR is harmless (my 'variance cost' assumption was WRONG); if
V > all/har, the refinement holds. Large K + ridge (vs the K=25 mean). Strict embargo + shuffle placebo.
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
EMBARGO = 3125
K = 1500
RIDGE = 1e-3
SIM = [
    "har_ma_1",
    "har_ma_5",
    "har_ma_25",
    "har_ma_125",
    "cumrv_x_close",
]  # similarity state (recency-history proxy)
FSETS = {
    "intercept": [],
    "har": ["har_ma_1", "har_ma_5", "har_ma_25", "har_ma_125"],
    "V": ["har_ma_1", "har_ma_5", "adj_sumbipow_ma_1", "adj_stocktwits_attention_ma_1"],
    "all": [
        "har_ma_1",
        "har_ma_5",
        "har_ma_25",
        "har_ma_125",
        "cumrv_x_close",
        "adj_sumbipow_ma_1",
        "adj_stocktwits_attention_ma_1",
    ],
}


def qlike(p, y, b):
    pr, tr = apply_duan_smearing(np.asarray(p, dtype=np.float64), y, b)
    m = (tr > 0) & (pr > 0)
    rr = tr[m] / pr[m]
    return float(np.mean(rr - np.log(rr) - 1.0))


def main() -> None:
    d = f"{CACHE_ROOT}/{CELL}"
    feats = json.load(open(f"{d}/feats.json"))
    Xs = np.load(f"{d}/Xs.npy", mmap_mode="r")
    A = pd.concat(
        [
            pd.read_csv(f)
            for f in glob.glob(f"results/resid_ab/{CELL}/{LBL}/chunk_*.csv")
        ]
    ).sort_values("k")
    k = A["k"].to_numpy().astype(np.int64)
    pred_adj, y_true, base = (
        A[c].to_numpy(np.float64) for c in ("pred_adj", "y_true", "base")
    )
    r = y_true - pred_adj
    t = k + TRAIN_WIN
    hour = np.asarray(Xs[t, feats.index("hour")], dtype=np.float64)
    close = (hour >= 16) & (hour <= 19)
    ci = np.where(close)[0]
    kc, rc = k[ci], r[ci]
    tc = t[ci]

    def grab(cols):
        return (
            np.ascontiguousarray(
                np.column_stack(
                    [np.asarray(Xs[tc, feats.index(c)], dtype=np.float64) for c in cols]
                )
            )
            if cols
            else np.zeros((len(ci), 0))
        )

    Esim = grab(SIM)
    Esim = (Esim - Esim.mean(0)) / (Esim.std(0) + 1e-9)
    Fmats = {name: grab(cols) for name, cols in FSETS.items()}
    nC = len(ci)
    print(
        "close=%d  K=%d  embargo=%d  resid_std=%.4f" % (nC, K, EMBARGO, rc.std()),
        flush=True,
    )

    preds = {name: np.zeros(nC) for name in FSETS}
    rng = np.random.RandomState(0)
    rc_shuf = rc[rng.permutation(nC)]
    preds_shuf_all = np.zeros(nC)  # 'all' fset on shuffled residuals (placebo)
    ptr = 0
    for i in range(nC):
        thr = kc[i] - EMBARGO
        while ptr < nC and kc[ptr] <= thr:
            ptr += 1
        if ptr <= K:
            continue
        dd = Esim[:ptr] - Esim[i]
        dist = np.einsum("ij,ij->i", dd, dd)
        nn = np.argpartition(dist, K)[:K]
        for name, F in Fmats.items():
            Dtr = np.column_stack([np.ones(K), F[nn]])
            G = Dtr.T @ Dtr + RIDGE * np.eye(Dtr.shape[1])
            beta = np.linalg.solve(G, Dtr.T @ rc[nn])
            preds[name][i] = np.concatenate([[1.0], F[i]]) @ beta
        Da = np.column_stack([np.ones(K), Fmats["all"][nn]])
        Ga = Da.T @ Da + RIDGE * np.eye(Da.shape[1])
        bshuf = np.linalg.solve(Ga, Da.T @ rc_shuf[nn])
        preds_shuf_all[i] = np.concatenate([[1.0], Fmats["all"][i]]) @ bshuf

    q0 = qlike(pred_adj, y_true, base)
    q0c = qlike(pred_adj[ci], y_true[ci], base[ci])
    print("Hero A: full=%.5f  h16-19=%.5f" % (q0, q0c), flush=True)
    for name in FSETS:
        new = pred_adj.copy()
        new[ci] += preds[name]
        print(
            "  %-9s full %.5f (d%+.6f)  h16-19 %.5f (d%+.6f)"
            % (
                name,
                qlike(new, y_true, base),
                qlike(new, y_true, base) - q0,
                qlike(new[ci], y_true[ci], base[ci]),
                qlike(new[ci], y_true[ci], base[ci]) - q0c,
            ),
            flush=True,
        )
    new = pred_adj.copy()
    new[ci] += preds_shuf_all
    print(
        "  %-9s full %.5f (d%+.6f)  [should be ~0]"
        % ("SHUFFLE", qlike(new, y_true, base), qlike(new, y_true, base) - q0),
        flush=True,
    )
    print("KNN_LOCAL_DONE", flush=True)


if __name__ == "__main__":
    main()
