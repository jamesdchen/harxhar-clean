"""§29.3 / AM-07: per-horizon shrinkage imported from the msweep α-law, pre-registered.

α(H) = α₁·H (theory anchor: the integrated target's noise variance grows ~H), applied as
solver α = 3000·H with the backbone column scale following and the product block's relative
scale unchanged. Blocked engine, twins recomputed same-engine. No grid.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.alpha_manifestation import TW  # noqa: E402
from analysis.alpha_panel import load_panel  # noqa: E402
from analysis.minimal_model import CACHE, HOLDOUT, _hac_mean_t, _qlike_series  # noqa: E402
from analysis.nl_sparsity import base_columns  # noqa: E402
from analysis.straddle_horizon import _y_horizon  # noqa: E402
from analysis.synthesis import _p  # noqa: E402
from analysis.wf import walk_forward_embargo_blocked  # noqa: E402
from src.features.transforms.target import PERIODS_PER_DAY  # noqa: E402


def _blocks(p):
    from analysis.minimal_model import _upper
    sig = dict(np.load(_p(CACHE)))
    har_cols = np.concatenate([p.cols("har"), p.cols("calendar"), p.cols("regime")])
    lin_cols = np.concatenate([p.cols("value"), p.cols("indicator")])
    bc, _ = base_columns(p)
    XH = np.ascontiguousarray(p.X[:, har_cols], dtype=np.float64)
    XL = np.ascontiguousarray(p.X[:, lin_cols], dtype=np.float64)
    XB = np.ascontiguousarray(p.X[:, bc], dtype=np.float64)
    XS = np.load(_p("xsec_features.npz"))["F"].astype(np.float64)
    ii, jj = _upper(XB.shape[1])
    P = XB[:, ii[sig["frozen"]]] * XB[:, jj[sig["frozen"]]]
    sd = pd.DataFrame(P).rolling(250 * PERIODS_PER_DAY, min_periods=1000).std().shift(1)
    med = np.nanmedian(sd.to_numpy(), axis=1, keepdims=True)
    sdv = np.maximum(sd.to_numpy(), 0.1 * np.where(np.isfinite(med), med, 1.0))
    return XH, XL, XS, P / pd.DataFrame(sdv).bfill().to_numpy()


def main() -> None:
    p = load_panel()
    ts = pd.Series(pd.to_datetime(p.t))
    day_codes = pd.factorize(ts.dt.normalize())[0]
    late = (ts >= HOLDOUT).to_numpy()
    XH, XL, XS, P = _blocks(p)
    for hb in (8, 16):
        yh, Bh = _y_horizon(p, hb)
        lags = 2 * hb + 480
        qs = {}
        for name, a_solver in (("unretuned (3e3)", 3000.0), (f"alpha-law (3e3x{hb})", 3000.0 * hb)):
            X = np.hstack([XH * np.sqrt(a_solver / 1.0), XL, XS, P * np.sqrt(0.1)])
            pred = walk_forward_embargo_blocked(X, yh, day_codes, 250, 1, a_solver)
            m = np.isfinite(pred) & np.isfinite(yh)
            q = np.full(len(yh), np.nan)
            q[m] = _qlike_series(pred[m], yh[m], Bh[m])
            qs[name] = q
            print(f"H={hb} {name:20s} QLIKE {np.nanmean(q):.5f} "
                  f"(2020+ {np.nanmean(q[late]):.5f})", flush=True)
        k1, k2 = list(qs)
        d = qs[k1] - qs[k2]
        g = _hac_mean_t(d[np.isfinite(d)], lags)
        print(f"H={hb}: alpha-law vs unretuned QLIKE DM {g:+.2f} "
              f"(2020+ {_hac_mean_t(d[late & np.isfinite(d)], lags):+.2f})   "
              f"gate >= +2.0: {'PASS' if g >= 2.0 else 'FAIL'}\n", flush=True)


if __name__ == "__main__":
    main()
