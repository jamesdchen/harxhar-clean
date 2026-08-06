"""The smear family's Romano-Wolf: every second-moment claim made after AM-10, each vs its
registered baseline, joint step-down (reuses am10_sweep.sweep). Members: leverage (vs means),
probe5 (vs means+lev), events (vs means+lev), dense-679 (vs means), regime labels (vs means),
transmission-smear/37c (vs means)."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.alpha_law import _blocks  # noqa: E402
from analysis.alpha_manifestation import TW  # noqa: E402
from analysis.alpha_panel import load_panel  # noqa: E402
from analysis.am10_sweep import sweep  # noqa: E402
from analysis.map_monitor import _frame_and_scores  # noqa: E402
from analysis.synthesis import _p  # noqa: E402
from analysis.trans_exploit import _trans_block  # noqa: E402
from analysis.wf import walk_forward_embargo_blocked  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.manifold import SpectralEmbedding  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402


def main() -> None:
    p = load_panel()
    ts = pd.Series(pd.to_datetime(p.t))
    n = len(ts)
    day = ts.dt.normalize()
    day_codes = pd.factorize(day)[0]
    f = np.full(n, np.nan)
    f[TW:] = np.load(_p("final_onestage.npz"))["yhat_bar"]
    e2 = (p.y - f) ** 2
    a_day = pd.Series(e2).groupby(day.values).mean()
    la = np.log(a_day + 1e-12)
    y = la.to_numpy()
    nd = len(y)
    r_day = pd.Series(p.X[:, p.names.index("adj_sumret_ma_1")].astype(np.float64)).groupby(day.values).sum()
    rs = (r_day - r_day.mean()) / (r_day.std() + 1e-12)
    l1s = la.shift(1)
    neg = rs.clip(upper=0)
    means = pd.DataFrame({"l1": l1s, "m5": l1s.rolling(5).mean(), "m21": l1s.rolling(21).mean()})
    lev = pd.DataFrame({"r": rs.shift(1), "absr": rs.abs().shift(1)})
    probe = pd.DataFrame({"rneg": neg.shift(1), "r2": (rs**2).shift(1),
                          "r_x_l1": rs.shift(1) * (l1s - l1s.mean()),
                          "neg5": neg.rolling(5).mean().shift(1),
                          "neg21": neg.rolling(21).mean().shift(1)})
    probe = (probe - probe.mean()) / (probe.std() + 1e-12)
    rel = pd.read_parquet("data/releases.parquet")
    rel["d"] = pd.to_datetime(rel["endbartime"]).dt.normalize()
    types = [c for c in rel.columns if c not in ("endbartime", "d")]
    days_idx = pd.DatetimeIndex(la.index)
    ev = {}
    for t in types:
        rd = np.sort(rel.loc[rel[t] > 0, "d"].unique())
        pos = np.searchsorted(rd, days_idx.values, side="right")
        last = np.where(pos > 0, rd[np.clip(pos - 1, 0, None)], np.datetime64("1990-01-01"))
        ev[t.split()[0]] = np.log1p((days_idx.values - last) / np.timedelta64(1, "D"))
    E = pd.DataFrame(ev, index=la.index).shift(1)
    # regime labels (causal anchors, §34.2 convention)
    G, e_res, ts_g = _frame_and_scores()
    from analysis.cucuringu import _daily
    Gd, _, _ = _daily(G, e_res, ts_g)
    n_anchor = 3 * 252
    emb = SpectralEmbedding(n_components=6, affinity="nearest_neighbors", n_neighbors=15,
                            random_state=0).fit_transform(Gd[:n_anchor])
    km = KMeans(5, n_init=10, random_state=0).fit(emb)
    nn = NearestNeighbors(n_neighbors=5).fit(Gd[:n_anchor])
    _, ind = nn.kneighbors(Gd)
    lab = np.array([np.bincount(km.labels_[r], minlength=5).argmax() for r in ind])
    OH = pd.DataFrame(np.eye(5)[np.roll(lab, 1)][:, :-1],
                      index=la.index[: len(lab)]).reindex(la.index).fillna(0.0)

    def ahat(Xdf):
        Xf = Xdf.to_numpy()
        out = np.full(nd, np.nan)
        for start in range(3 * 252 + 63, nd, 63):
            mm = np.isfinite(Xf).all(1) & np.isfinite(y)
            tr = np.flatnonzero(mm & (np.arange(nd) < start))
            if len(tr) < 200:
                continue
            b = np.linalg.lstsq(np.c_[np.ones(len(tr)), Xf[tr]], y[tr], rcond=None)[0]
            seg = np.arange(start, min(start + 63, nd))
            ok = np.isfinite(Xf[seg]).all(1)
            out[seg[ok]] = np.exp(np.c_[np.ones(ok.sum()), Xf[seg][ok]] @ b)
        return out

    slot = (ts.dt.hour * 2 + ts.dt.minute // 30).to_numpy()
    relb = pd.Series(e2) / pd.Series(e2).groupby(day_codes).transform("mean")
    ss = np.ones(n)
    for s in np.unique(slot):
        mm = slot == s
        cs = pd.Series(np.where(mm, relb, np.nan)).expanding(min_periods=100).mean().shift(48)
        ss[mm] = cs.to_numpy()[mm]
    ss = np.where(np.isfinite(ss), ss, 1.0)
    m0 = np.isfinite(f) & np.isfinite(p.y) & (p.baseline > 0)
    tr_raw = p.y**2 * p.baseline

    def qseries(sm):
        ok = m0 & np.isfinite(sm) & (sm > 0)
        pr = (f**2 + sm) * p.baseline
        q = np.full(n, np.nan)
        okk = ok & (pr > 0) & (tr_raw > 0)
        r = tr_raw[okk] / pr[okk]
        q[okk] = r - np.log(r) - 1.0
        return q

    def smear_of(Xdf):
        return day.map(dict(zip(la.index, ahat(Xdf)))).to_numpy(dtype=float) * ss

    q_means = qseries(smear_of(means))
    q_lev = qseries(smear_of(pd.concat([means, lev], axis=1)))
    q_probe = qseries(smear_of(pd.concat([means, lev, probe], axis=1)))
    q_events = qseries(smear_of(pd.concat([means, lev, E], axis=1)))
    q_labels = qseries(smear_of(pd.concat([means, OH], axis=1)))
    print("regression smears built", flush=True)

    # the two walk-based smears
    XH, XL, XS, P = _blocks(p)
    A = 3000.0
    X679 = np.hstack([XH * np.sqrt(A), XL, XS, P * np.sqrt(0.1)])
    z = pd.Series(np.log(np.where(np.isfinite(e2), e2, np.nan) + 1e-12)).ffill().bfill().to_numpy()
    def walk_smear(X):
        fz = walk_forward_embargo_blocked(X, z, day_codes, 250, 1, A)
        corr = pd.Series(np.exp(np.clip(z - fz, -20, 20))).expanding(min_periods=5000).mean().shift(1)
        return qseries(np.exp(fz) * corr.to_numpy())
    q_dense = walk_smear(X679)
    print("dense walk done", flush=True)
    F1 = _trans_block(G, n, lag=1, refresh_days=63)
    q_trans = walk_smear(np.hstack([X679, F1]))
    print("transmission walk done", flush=True)

    D = np.column_stack([
        q_means - q_lev,        # leverage vs means
        q_lev - q_probe,        # probe5 vs means+lev
        q_lev - q_events,       # events vs means+lev
        q_means - q_dense,      # dense vs means
        q_means - q_labels,     # labels vs means
        q_means - q_trans,      # transmission-smear vs means
    ])
    names = ["leverage", "probe5", "events", "dense679", "labels", "transmission_smear"]
    np.savez_compressed(_p("smear_family_lossdiffs.npz"), D=D, names=np.array(names))
    sweep(D, names)


if __name__ == "__main__":
    main()
