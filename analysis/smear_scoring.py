"""Shared construction of the production linear smears (means+leverage, means+leverage+probe)
so downstream arms (§34.7 LSTM) score against identical baselines."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.alpha_manifestation import TW  # noqa: E402
from analysis.alpha_panel import load_panel  # noqa: E402
from analysis.synthesis import _p  # noqa: E402


def linear_smears():
    p = load_panel()
    ts = pd.Series(pd.to_datetime(p.t))
    n = len(ts)
    f = np.full(n, np.nan)
    f[TW:] = np.load(_p("final_onestage.npz"))["yhat_bar"]
    e2 = (p.y - f) ** 2
    day = ts.dt.normalize()
    day_codes = pd.factorize(day)[0]
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
    rel = pd.Series(e2) / pd.Series(e2).groupby(day_codes).transform("mean")
    ss = np.ones(n)
    for s in np.unique(slot):
        mm = slot == s
        cs = pd.Series(np.where(mm, rel, np.nan)).expanding(min_periods=100).mean().shift(48)
        ss[mm] = cs.to_numpy()[mm]
    ss = np.where(np.isfinite(ss), ss, 1.0)
    sm_lev = day.map(dict(zip(la.index, ahat(pd.concat([means, lev], axis=1))))).to_numpy(dtype=float) * ss
    sm_probe = day.map(dict(zip(la.index, ahat(pd.concat([means, lev, probe], axis=1))))).to_numpy(dtype=float) * ss
    return sm_lev, sm_probe
