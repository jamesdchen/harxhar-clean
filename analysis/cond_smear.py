"""§34.2: can the regime-conditional amplitude forecast improve QLIKE through the smear?

User challenge to the A3 result's "sizing only" framing. QLIKE's reconstruction is
pred_raw = (f² + E[(y−f)²]) · baseline — the second moment enters the score directly, and the
current machinery uses a CONSTANT smear. A3 says the conditional second moment is forecastable
(trailing means + regime labels, cross-half uplift +0.019 next-day). Arms, h = 1 deliverable
(same f everywhere — only the smear changes):

  (i)   global constant smear (the incumbent machinery's convention)
  (ii)  trailing-250d constant smear (causal, no conditioning — the fair baseline)
  (iii) conditional smear: daily â from an expanding regression on {trailing log-amp means,
        causal regime labels}, broadcast to bars via a trailing slot-share profile

Labels are CAUSAL: spectral embedding + k-means fit on the first 3 OOS years' days (anchors),
later days assigned by nearest anchors. Gate: (iii) vs (ii) QLIKE DM >= +2.0. Expectation:
positive but below gate — the smear term is O(2%) of pred_raw and QLIKE is second-order in it;
the prior (unconditional) smear-variant claim died at rw p = 0.181.
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
from analysis.cucuringu import _daily  # noqa: E402
from analysis.minimal_model import HOLDOUT, _hac_mean_t  # noqa: E402
from analysis.synthesis import _p  # noqa: E402


def main() -> None:
    p = load_panel()
    ts = pd.Series(pd.to_datetime(p.t))
    n = len(ts)
    f = np.full(n, np.nan)
    f[TW:] = np.load(_p("final_onestage.npz"))["yhat_bar"]
    y, B = p.y, p.baseline
    e2 = (y - f) ** 2
    late = (ts >= HOLDOUT).to_numpy()

    day = ts.dt.normalize()
    day_codes = pd.factorize(day)[0]
    slot = (ts.dt.hour * 2 + ts.dt.minute // 30).to_numpy()

    # causal regime labels (anchor convention from §34.1)
    from sklearn.manifold import SpectralEmbedding
    from sklearn.cluster import KMeans
    from sklearn.neighbors import NearestNeighbors
    G, e_res, ts_g = _frame_and_scores()
    Gd, _, tsd = _daily(G, e_res, ts_g)
    nd = len(Gd)
    n_anchor = 3 * 252
    emb = SpectralEmbedding(n_components=6, affinity="nearest_neighbors", n_neighbors=15,
                            random_state=0).fit_transform(Gd[:n_anchor])
    km = KMeans(5, n_init=10, random_state=0).fit(emb)
    nn = NearestNeighbors(n_neighbors=5).fit(Gd[:n_anchor])
    _, ind = nn.kneighbors(Gd)
    lab = np.array([np.bincount(km.labels_[r], minlength=5).argmax() for r in ind])

    # daily amplitude series aligned to Gd's days
    dgd = pd.Series(pd.to_datetime(tsd.dt.normalize()))
    a_by_day = pd.Series(e2).groupby(day.values).mean()
    a_d = a_by_day.reindex(dgd.values).to_numpy()
    la = np.log(a_d + 1e-12)
    Xm = np.column_stack([
        pd.Series(la).shift(1).to_numpy(),
        pd.Series(la).shift(1).rolling(5).mean().to_numpy(),
        pd.Series(la).shift(1).rolling(21).mean().to_numpy(),
    ])
    X = np.column_stack([Xm, np.eye(5)[np.roll(lab, 1)][:, :-1]])

    def expanding_ahat(Xf: np.ndarray) -> np.ndarray:
        out = np.full(nd, np.nan)
        for start in range(n_anchor + 63, nd, 63):
            m = np.isfinite(Xf).all(1) & np.isfinite(la)
            tr = np.flatnonzero(m & (np.arange(nd) < start))
            if len(tr) < 200:
                continue
            b = np.linalg.lstsq(np.c_[np.ones(len(tr)), Xf[tr]], la[tr], rcond=None)[0]
            seg = np.arange(start, min(start + 63, nd))
            ok = np.isfinite(Xf[seg]).all(1)
            out[seg[ok]] = np.exp(np.c_[np.ones(ok.sum()), Xf[seg][ok]] @ b)
        return out

    ahat = expanding_ahat(X)
    ahat_means = expanding_ahat(Xm)  # §34.2 ablation: no regime labels

    # broadcast to bars: daily ahat x trailing slot-share (expanding per-slot mean of e2/day-mean)
    dmap = pd.Series(dgd.values).reset_index(drop=True)
    day_to_ahat = dict(zip(dmap.values, ahat))
    ahat_bar_day = day.map(day_to_ahat).to_numpy(dtype=float)
    ahat_bar_means = day.map(dict(zip(dmap.values, ahat_means))).to_numpy(dtype=float)
    rel = pd.Series(e2) / pd.Series(e2).groupby(day_codes).transform("mean")
    slot_share = np.ones(n)
    for s in np.unique(slot):
        m = slot == s
        cs = pd.Series(np.where(m, rel, np.nan)).expanding(min_periods=100).mean().shift(48)
        slot_share[m] = cs.to_numpy()[m]
    ss = np.where(np.isfinite(slot_share), slot_share, 1.0)
    smear_cond = ahat_bar_day * ss
    smear_means = ahat_bar_means * ss

    smear_trail = pd.Series(e2).rolling(250 * 48, min_periods=5000).mean().shift(1).to_numpy()
    m0 = np.isfinite(f) & np.isfinite(y) & (B > 0)
    smear_glob = np.nanmean(e2[m0])

    qs = {}
    for name, sm in (("global", np.full(n, smear_glob)), ("trailing-250d", smear_trail),
                     ("cond-means-only", smear_means), ("conditional", smear_cond)):
        mm = m0 & np.isfinite(sm) & (sm > 0)
        pred_raw = (f**2 + sm) * B
        true_raw = y**2 * B
        q = np.full(n, np.nan)
        ok = mm & (pred_raw > 0) & (true_raw > 0)
        r = true_raw[ok] / pred_raw[ok]
        q[ok] = r - np.log(r) - 1.0
        qs[name] = q
        print(f"  smear={name:14s} QLIKE {np.nanmean(q):.5f} (2020+ {np.nanmean(q[late]):.5f})"
              f"  [{ok.sum()} bars]", flush=True)
    for a, b_ in (("global", "trailing-250d"), ("trailing-250d", "cond-means-only"),
                  ("cond-means-only", "conditional"), ("trailing-250d", "conditional")):
        d = qs[a] - qs[b_]
        md = np.isfinite(d)
        print(f"  {b_} vs {a}: DM {_hac_mean_t(d[md], 480):+.2f} "
              f"(2020+ {_hac_mean_t(d[md & late], 480):+.2f})")
    d = qs["trailing-250d"] - qs["conditional"]
    g = _hac_mean_t(d[np.isfinite(d)], 480)
    print(f"\n  gate (conditional vs trailing >= +2.0): {'PASS' if g >= 2.0 else 'FAIL'}")


if __name__ == "__main__":
    main()
