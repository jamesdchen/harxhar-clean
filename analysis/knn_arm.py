"""§33 P1: kNN residual correction on the deliverable — the professor's estimator, upgraded
per the project's own diagnosis (ambient/state retrieval, no eigenmap, strictly causal).

Two view sets, pre-registered: (a) trailing-24-bar residual path; (b) the 20 frozen-frame
factor scores. Pool: trailing 5y, stride 3, embargoed by the label horizon. k=100, Gaussian
weights at a frozen median bandwidth (first evaluation year). Correction = weighted mean of
neighbors' residuals added to the deliverable's forecast. Gate: QLIKE DM >= +2.0.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.alpha_manifestation import TW  # noqa: E402
from analysis.alpha_panel import load_panel  # noqa: E402
from analysis.map_monitor import _frame_and_scores  # noqa: E402
from analysis.minimal_model import HOLDOUT, _hac_mean_t, _qlike_series  # noqa: E402
from analysis.straddle_horizon import _y_horizon  # noqa: E402
from analysis.synthesis import _p  # noqa: E402

K = 100
POOL_YEARS = 5
POOL_STRIDE = 3
PATH_W = 24


def _knn_correct(V: np.ndarray, r: np.ndarray, valid: np.ndarray, emb: int) -> np.ndarray:
    """Causal Gaussian-kNN prediction of r(t) from views V, chunked."""
    n = len(r)
    out = np.full(n, np.nan)
    pool_len = POOL_YEARS * 252 * 48
    start = pool_len // 2  # first predictions once a half-pool exists
    bw = None
    CH = 4000
    for c0 in range(start, n, CH):
        c1 = min(c0 + CH, n)
        p_lo = max(0, c0 - emb - pool_len)
        pool = np.arange(p_lo, c0 - emb, POOL_STRIDE)
        pool = pool[valid[pool]]
        if len(pool) < 5 * K:
            continue
        D2 = (np.sum(V[c0:c1] ** 2, 1)[:, None] - 2.0 * V[c0:c1] @ V[pool].T
              + np.sum(V[pool] ** 2, 1)[None, :])
        idx = np.argpartition(D2, K, axis=1)[:, :K]
        dk = np.take_along_axis(D2, idx, axis=1)
        if bw is None:
            bw = float(np.median(dk))  # frozen at first evaluated chunk
        w = np.exp(-dk / (2.0 * bw))
        w /= w.sum(1, keepdims=True)
        out[c0:c1] = np.sum(w * r[pool][idx], axis=1)
    return out


def main() -> None:
    p = load_panel()
    ts = pd.Series(pd.to_datetime(p.t))
    late = (ts >= HOLDOUT).to_numpy()
    G, _, _ = _frame_and_scores()  # rows 2TW:
    n_all = len(ts)

    for hname, hb in (("h=1 bar", 1), ("H=8", 8)):
        if hb == 1:
            yt, Bt = p.y, p.baseline
            z = np.load(_p("final_onestage.npz"))["yhat_bar"]
            f = np.full(n_all, np.nan)
            f[TW:] = z
        else:
            yt, Bt = _y_horizon(p, hb)
            z = np.load(_p("straddle_ladder_h8.npz"))["one-stage_679"]
            f = np.full(n_all, np.nan)
            f[TW:] = z
        r = yt - f  # deliverable residual, NaN where undefined
        valid_r = np.isfinite(r)
        rz = np.where(valid_r, r, 0.0)

        # view (a): trailing residual path of CLOSED windows only. At horizon hb, r(s) spans
        # [s, s+hb-1] and is closed by t iff s <= t-hb, so the freshest usable entry is r(t-hb):
        # Vp[t, k] = r(t - hb - k). At hb=1 this is the plain t-1 path; at hb=8 it removes the
        # 7 unrealized entries that leaked the current target's future into the view (§29's
        # overlap-leak lesson, view-side; the leaked run scored an impossible +28 DM).
        Vp = np.zeros((n_all, PATH_W))
        for k in range(PATH_W):
            sh = hb + k
            Vp[sh:, k] = rz[: n_all - sh]
        sd = Vp[valid_r].std() + 1e-12
        Vp /= sd
        # view (b): frozen-frame state
        Vs = np.zeros((n_all, G.shape[1]))
        Vs[2 * TW :] = G

        # view (c): the professor's exact spec — Laplacian eigenmap (d=6) of the state,
        # fit on the first two years of the OOS span, extended forward by kNN interpolation
        # of embedding coordinates (the Nystrom-style extension whose error the prior study
        # diagnosed). Head-to-head against the corrected views, same protocol.
        from sklearn.manifold import SpectralEmbedding
        from sklearn.neighbors import NearestNeighbors
        FIT_N = 2 * 252 * 48
        sub = np.arange(0, FIT_N, 4)  # every 4th bar of the first 2y, ~6k anchors
        anchors = G[sub]
        emb = SpectralEmbedding(n_components=6, affinity="nearest_neighbors",
                                n_neighbors=15, random_state=0).fit_transform(anchors)
        nn = NearestNeighbors(n_neighbors=10).fit(anchors)
        dist, ind = nn.kneighbors(G)
        wgt = 1.0 / (dist + 1e-9)
        wgt /= wgt.sum(1, keepdims=True)
        Ge = np.einsum("nk,nkd->nd", wgt, emb[ind])
        Ve = np.zeros((n_all, 6))
        Ve[2 * TW :] = (Ge - Ge.mean(0)) / (Ge.std(0) + 1e-12)

        for vname, V in (("residual path", Vp), ("state (frame)", Vs),
                         ("eigenmap d=6 (his spec)", Ve)):
            chat = _knn_correct(V, rz, valid_r, emb=hb + PATH_W)
            m = np.isfinite(chat) & np.isfinite(f) & np.isfinite(yt)
            q0 = np.full(n_all, np.nan)
            q1 = np.full(n_all, np.nan)
            q0[m] = _qlike_series(f[m], yt[m], Bt[m])
            q1[m] = _qlike_series(f[m] + chat[m], yt[m], Bt[m])
            d = q0 - q1
            lags = 2 * hb + 480
            g = _hac_mean_t(d[m], lags)
            print(f"{hname:8s} view={vname:14s}: QLIKE {np.nanmean(q0):.5f} -> "
                  f"{np.nanmean(q1):.5f}   DM {g:+.2f} "
                  f"(2020+ {_hac_mean_t(d[m & late], lags):+.2f})   "
                  f"gate: {'PASS' if g >= 2.0 else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
