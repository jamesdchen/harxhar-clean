"""§34.3: dense-weak applied to the second moment — 679-column ridge on next-bar log e²,
feeding the Duan smear. Gate vs the means-only conditional smear."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.alpha_law import _blocks  # noqa: E402
from analysis.alpha_manifestation import TW  # noqa: E402
from analysis.alpha_panel import load_panel  # noqa: E402
from analysis.minimal_model import HOLDOUT, _hac_mean_t  # noqa: E402
from analysis.synthesis import _p  # noqa: E402
from analysis.wf import walk_forward_embargo_blocked  # noqa: E402


def main() -> None:
    p = load_panel()
    ts = pd.Series(pd.to_datetime(p.t))
    n = len(ts)
    day_codes = pd.factorize(ts.dt.normalize())[0]
    late = (ts >= HOLDOUT).to_numpy()
    f = np.full(n, np.nan)
    f[TW:] = np.load(_p("final_onestage.npz"))["yhat_bar"]
    y, B = p.y, p.baseline
    e2 = (y - f) ** 2
    z = np.log(np.where(np.isfinite(e2), e2, np.nan) + 1e-12)

    XH, XL, XS, P = _blocks(p)
    A = 3000.0
    X679 = np.hstack([XH * np.sqrt(A), XL, XS, P * np.sqrt(0.1)])
    fz = walk_forward_embargo_blocked(X679, pd.Series(z).ffill().bfill().to_numpy(),
                                      day_codes, 250, 1, A)
    # Duan back-transform: expanding mean of exp(log-residual), causal
    lr = z - fz
    corr = pd.Series(np.exp(np.clip(lr, -20, 20))).expanding(min_periods=5000).mean().shift(1)
    smear_dense = np.exp(fz) * corr.to_numpy()

    # baseline: means-only conditional smear (§34.2 construction, condensed)
    day = ts.dt.normalize()
    a_by_day = pd.Series(e2).groupby(day.values).mean()
    la = np.log(a_by_day + 1e-12)
    Xm = pd.DataFrame({"l1": la.shift(1), "m5": la.shift(1).rolling(5).mean(),
                       "m21": la.shift(1).rolling(21).mean()}).to_numpy()
    lav = la.to_numpy()
    nd = len(lav)
    ahat = np.full(nd, np.nan)
    for start in range(3 * 252 + 63, nd, 63):
        m = np.isfinite(Xm).all(1) & np.isfinite(lav)
        tr = np.flatnonzero(m & (np.arange(nd) < start))
        if len(tr) < 200:
            continue
        b = np.linalg.lstsq(np.c_[np.ones(len(tr)), Xm[tr]], lav[tr], rcond=None)[0]
        seg = np.arange(start, min(start + 63, nd))
        ok = np.isfinite(Xm[seg]).all(1)
        ahat[seg[ok]] = np.exp(np.c_[np.ones(ok.sum()), Xm[seg][ok]] @ b)
    slot = (ts.dt.hour * 2 + ts.dt.minute // 30).to_numpy()
    rel = pd.Series(e2) / pd.Series(e2).groupby(day_codes).transform("mean")
    ss = np.ones(n)
    for s in np.unique(slot):
        m = slot == s
        cs = pd.Series(np.where(m, rel, np.nan)).expanding(min_periods=100).mean().shift(48)
        ss[m] = cs.to_numpy()[m]
    ss = np.where(np.isfinite(ss), ss, 1.0)
    smear_means = day.map(dict(zip(la.index, ahat))).to_numpy(dtype=float) * ss

    m0 = np.isfinite(f) & np.isfinite(y) & (B > 0)
    tr_raw = y**2 * B
    qs = {}
    for name, sm in (("means-only (baseline)", smear_means), ("dense-679", smear_dense)):
        ok = m0 & np.isfinite(sm) & (sm > 0)
        pr = (f**2 + sm) * B
        q = np.full(n, np.nan)
        okk = ok & (pr > 0) & (tr_raw > 0)
        r = tr_raw[okk] / pr[okk]
        q[okk] = r - np.log(r) - 1.0
        qs[name] = q
        print(f"  smear={name:22s} QLIKE {np.nanmean(q):.5f} "
              f"(2020+ {np.nanmean(q[late]):.5f})  [{okk.sum()} bars]", flush=True)
    d = qs["means-only (baseline)"] - qs["dense-679"]
    md = np.isfinite(d)
    g = _hac_mean_t(d[md], 480)
    print(f"\n  dense vs means-only: DM {g:+.2f} (2020+ {_hac_mean_t(d[md & late], 480):+.2f})")
    print(f"  gate (>= +2.0): {'PASS' if g >= 2.0 else 'FAIL'}")


if __name__ == "__main__":
    main()
