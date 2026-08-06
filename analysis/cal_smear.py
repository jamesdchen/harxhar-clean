"""§34.15: the calendar smear — the surviving fragment of the union's span? Pre-registered.

§34.10e named the union's historical span: mostly CALENDAR structure (day-of-week, the
half-day flag) plus slow realized-moment/liquidity aggregates. §34.10b showed the union's
edge died in 2022+ — but the era decomposition never separated the calendar part from the
slow aggregates. Calendar seasonality of amplitude is structural (option expiries, macro
release days, half-day sessions are not going away); if it is the surviving component, the
production smear gains six columns.

Arm: named-10 + DOW one-hots (4) + short-day flag (sessions with < 46 bars — the panel has
996 41-bar and 996 11-bar sessions), same instrument as everywhere in §34.10+ (per-refit
causal standardization, Duan, lstsq — 16 columns need no GCV). Gate: DM >= +2.0 vs named,
with the 2020+ number and an era row-set reported (the era question IS the point). Recorded
lean: small pass full-span, 2020+ genuinely uncertain — §34.10d says the variance channel's
extra content keeps dying on the modern regime, but calendar is the one candidate whose
mechanism is stationary by construction.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.alpha_manifestation import TW  # noqa: E402
from analysis.alpha_panel import load_panel  # noqa: E402
from analysis.minimal_model import HOLDOUT, _hac_mean_t  # noqa: E402
from analysis.synthesis import _p  # noqa: E402


def main() -> None:
    p = load_panel()
    ts = pd.Series(pd.to_datetime(p.t))
    n = len(ts)
    f = np.full(n, np.nan)
    f[TW:] = np.load(_p("final_onestage.npz"))["yhat_bar"]
    e2 = (p.y - f) ** 2
    day = ts.dt.normalize()
    day_codes = pd.factorize(day)[0]
    late = (ts >= HOLDOUT).to_numpy()

    a_day = pd.Series(e2).groupby(day.values).mean()
    la = np.log(a_day + 1e-12)
    y = la.to_numpy()
    nd = len(y)
    r_day = pd.Series(p.X[:, p.names.index("adj_sumret_ma_1")].astype(np.float64)).groupby(day.values).sum()
    rs = (r_day - r_day.mean()) / (r_day.std() + 1e-12)
    l1s = la.shift(1)
    neg = rs.clip(upper=0)
    named = pd.DataFrame({
        "l1": l1s, "m5": l1s.rolling(5).mean(), "m21": l1s.rolling(21).mean(),
        "r": rs.shift(1), "absr": rs.abs().shift(1),
        "rneg": neg.shift(1), "r2": (rs ** 2).shift(1),
        "r_x_l1": rs.shift(1) * (l1s - l1s.mean()),
        "neg5": neg.rolling(5).mean().shift(1), "neg21": neg.rolling(21).mean().shift(1),
    })
    didx = pd.Series(pd.to_datetime(la.index))
    dow = pd.get_dummies(didx.dt.dayofweek).reindex(columns=range(5), fill_value=0).iloc[:, :4]
    dow.columns = [f"dow{c}" for c in dow.columns]
    bars_per_day = pd.Series(e2).groupby(day.values).size()
    short = (bars_per_day.to_numpy() < 46).astype(float)
    cal = pd.concat([dow.reset_index(drop=True),
                     pd.Series(short, name="short_day").reset_index(drop=True)], axis=1)
    union_cal = pd.concat([named.reset_index(drop=True), cal], axis=1)

    mfin = np.isfinite(y)

    def ahat_of(Xraw):
        out = np.full(nd, np.nan)
        for start in range(3 * 252 + 63, nd, 63):
            tr = np.flatnonzero(mfin & np.isfinite(Xraw).all(1) & (np.arange(nd) < start))
            if len(tr) < 400:
                continue
            seg = np.arange(start, min(start + 63, nd))
            mu = np.nanmean(Xraw[tr], 0)
            sd = np.nanstd(Xraw[tr], 0) + 1e-12
            Ztr = np.clip(np.nan_to_num((Xraw[tr] - mu) / sd), -8, 8)
            Zte = np.clip(np.nan_to_num((Xraw[seg] - mu) / sd), -8, 8)
            b = np.linalg.lstsq(np.c_[np.ones(len(tr)), Ztr], y[tr], rcond=None)[0]
            fit_tr = np.c_[np.ones(len(tr)), Ztr] @ b
            fit_te = np.clip(np.c_[np.ones(len(seg)), Zte] @ b,
                             y[tr].min() - 1.0, y[tr].max() + 1.0)
            duan = np.mean(np.exp(np.clip(y[tr] - fit_tr, -10, 10)))
            out[seg] = np.exp(fit_te) * duan
        return out

    slot = (ts.dt.hour * 2 + ts.dt.minute // 30).to_numpy()
    rel = pd.Series(e2) / pd.Series(e2).groupby(day_codes).transform("mean")
    ss = np.ones(n)
    for s in np.unique(slot):
        mm = slot == s
        cs = pd.Series(np.where(mm, rel, np.nan)).expanding(min_periods=100).mean().shift(48)
        ss[mm] = cs.to_numpy()[mm]
    ss = np.where(np.isfinite(ss), ss, 1.0)
    m0 = np.isfinite(f) & np.isfinite(p.y) & (p.baseline > 0)
    tr_raw = p.y ** 2 * p.baseline

    def q_of(ahat):
        sm = day.map(dict(zip(la.index, ahat))).to_numpy(dtype=float) * ss
        ok = m0 & np.isfinite(sm) & (sm > 0)
        pr = (f ** 2 + sm) * p.baseline
        q = np.full(n, np.nan)
        okk = ok & (pr > 0) & (tr_raw > 0)
        r = tr_raw[okk] / pr[okk]
        q[okk] = r - np.log(r) - 1.0
        return q

    q_named = q_of(ahat_of(named.to_numpy(dtype=float)))
    q_cal = q_of(ahat_of(union_cal.to_numpy(dtype=float)))
    print(f"  named {np.nanmean(q_named):.5f}   named+calendar {np.nanmean(q_cal):.5f}",
          flush=True)
    d = q_named - q_cal
    md = np.isfinite(d)
    g = _hac_mean_t(d[md], 480)
    print(f"  §34.15 calendar vs named: DM {g:+.2f} (2020+ {_hac_mean_t(d[md & late], 480):+.2f})"
          f"  {'PASS' if g >= 2.0 else 'FAIL'}", flush=True)
    yr = ts.dt.year.to_numpy()
    for lo, hi in ((2008, 2011), (2012, 2015), (2016, 2019), (2020, 2021), (2022, 2026)):
        m = md & (yr >= lo) & (yr <= hi)
        if m.sum() > 1000:
            print(f"  {lo}-{hi}: {np.nanmean(d[m]) * 1e4:+.2f} (t {_hac_mean_t(d[m], 480):+.2f})",
                  flush=True)


if __name__ == "__main__":
    main()
