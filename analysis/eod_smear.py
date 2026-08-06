"""§34.14: the conditional smear at the EOD deliverable horizon — a genuine gap.

The conditional smear (means+leverage+probe, FWER-graded) exists ONLY at h = 1 bar. The
trading deliverable is the EOD/VRP forecast (remaining-session target, §29.1–29.2), whose
QLIKE composition still uses a CONSTANT smear — the second moment at the horizon that prices
straddles has never been conditioned. Arm: the production smear recipe ported to the EOD
residuals (daily log-amplitude of the EOD errors, means+leverage+probe, §34.12b instrument —
causal per-refit standardization + Duan), broadcast by a trailing slot-share of EOD e².

Gates: conditional vs trailing-constant smear, DM >= +2.0, scored (i) pooled over all bars
and (ii) on the 10:00 open slice (where the §29.2 edge lives — the slice gate is the one
that matters for the deliverable). Recorded lean: PASS pooled (the h = 1 smear passed at
+3.9 and EOD amplitude is smoother), slice uncertain (fewer observations, wider horizon).
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
    z = np.load(_p("straddle_eod.npz"))
    f = np.full(n, np.nan)
    f[TW:] = z["one-stage_679"]
    y, B, hb = z["y_eod"], z["B_eod"], z["h_bars"]
    valid = np.isfinite(f) & np.isfinite(y) & (B > 0) & (hb > 0)
    e2 = np.where(valid, (y - f) ** 2, np.nan)
    day = ts.dt.normalize()
    day_codes = pd.factorize(day)[0]
    late = (ts >= HOLDOUT).to_numpy()
    slot = (ts.dt.hour * 2 + ts.dt.minute // 30).to_numpy()
    open_slice = slot == 20  # the 10:00 decision bar (§29.2)

    a_day = pd.Series(e2).groupby(day.values).mean()
    la = np.log(a_day + 1e-12)
    yd = la.to_numpy()
    nd = len(yd)
    r_day = pd.Series(p.X[:, p.names.index("adj_sumret_ma_1")].astype(np.float64)).groupby(day.values).sum()
    rs = (r_day - r_day.mean()) / (r_day.std() + 1e-12)
    l1s = la.shift(1)
    neg = rs.clip(upper=0)
    Xdf = pd.DataFrame({
        "l1": l1s, "m5": l1s.rolling(5).mean(), "m21": l1s.rolling(21).mean(),
        "r": rs.shift(1), "absr": rs.abs().shift(1),
        "rneg": neg.shift(1), "r2": (rs ** 2).shift(1),
        "r_x_l1": rs.shift(1) * (l1s - l1s.mean()),
        "neg5": neg.rolling(5).mean().shift(1), "neg21": neg.rolling(21).mean().shift(1),
    })
    Xraw = Xdf.to_numpy(dtype=float)
    mfin = np.isfinite(yd)
    ahat = np.full(nd, np.nan)
    for start in range(3 * 252 + 63, nd, 63):
        tr = np.flatnonzero(mfin & np.isfinite(Xraw).all(1) & (np.arange(nd) < start))
        if len(tr) < 400:
            continue
        seg = np.arange(start, min(start + 63, nd))
        mu = np.nanmean(Xraw[tr], 0)
        sd = np.nanstd(Xraw[tr], 0) + 1e-12
        Ztr = np.clip(np.nan_to_num((Xraw[tr] - mu) / sd), -8, 8)
        Zte = np.clip(np.nan_to_num((Xraw[seg] - mu) / sd), -8, 8)
        b = np.linalg.lstsq(np.c_[np.ones(len(tr)), Ztr], yd[tr], rcond=None)[0]
        fit_tr = np.c_[np.ones(len(tr)), Ztr] @ b
        fit_te = np.clip(np.c_[np.ones(len(seg)), Zte] @ b, yd[tr].min() - 1.0, yd[tr].max() + 1.0)
        duan = np.mean(np.exp(np.clip(yd[tr] - fit_tr, -10, 10)))
        ahat[seg] = np.exp(fit_te) * duan

    rel = pd.Series(e2) / pd.Series(e2).groupby(day_codes).transform("mean")
    ss = np.ones(n)
    for s in np.unique(slot):
        mm = slot == s
        cs = pd.Series(np.where(mm, rel, np.nan)).expanding(min_periods=100).mean().shift(48)
        ss[mm] = cs.to_numpy()[mm]
    ss = np.where(np.isfinite(ss), ss, 1.0)
    sm_cond = day.map(dict(zip(la.index, ahat))).to_numpy(dtype=float) * ss
    sm_trail = pd.Series(e2).rolling(250 * 48, min_periods=5000).mean().shift(1).to_numpy()

    tr_raw = np.where(valid, y ** 2 * B, np.nan)

    def q_of(sm):
        ok = valid & np.isfinite(sm) & (sm > 0)
        pr = (f ** 2 + sm) * B
        q = np.full(n, np.nan)
        okk = ok & (pr > 0) & (tr_raw > 0)
        r = tr_raw[okk] / pr[okk]
        q[okk] = r - np.log(r) - 1.0
        return q

    q_tr, q_cd = q_of(sm_trail), q_of(sm_cond)
    for lbl, m in (("pooled", np.ones(n, bool)), ("10:00 slice", open_slice)):
        d = (q_tr - q_cd)
        d[~m] = np.nan
        md = np.isfinite(d)
        g = _hac_mean_t(d[md], 480 if lbl == "pooled" else 63)
        print(f"  {lbl:12s} trailing {np.nanmean(q_tr[md]):.5f} -> cond {np.nanmean(q_cd[md]):.5f}"
              f"  DM {g:+.2f} (2020+ {_hac_mean_t(d[md & late], 480 if lbl == 'pooled' else 63):+.2f})"
              f"  {'PASS' if g >= 2.0 else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
