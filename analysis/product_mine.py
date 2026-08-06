"""§35: the product lever's last unturned stones. 35a alpha-scaled count ladder; 35b
cross-time (lagged-leg) products; 35c transmission columns at h=1 (never tested there).
All h=1-bar target, blocked engine, twin recomputed same-engine, gates DM >= +2.0."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.alpha_law import _blocks  # noqa: E402
from analysis.alpha_manifestation import TW  # noqa: E402
from analysis.alpha_panel import load_panel  # noqa: E402
from analysis.map_monitor import _frame_and_scores  # noqa: E402
from analysis.minimal_model import CACHE, HOLDOUT, _hac_mean_t, _qlike_series, _upper  # noqa: E402
from analysis.nl_sparsity import _pair_ic, base_columns  # noqa: E402
from analysis.synthesis import _p  # noqa: E402
from analysis.wf import walk_forward_embargo_blocked  # noqa: E402
from src.features.transforms.target import PERIODS_PER_DAY  # noqa: E402


def _q(pred, yt, Bt):
    m = np.isfinite(pred) & np.isfinite(yt)
    q = np.full(len(yt), np.nan)
    q[m] = _qlike_series(pred[m], yt[m], Bt[m])
    return q


def _scale(P):
    sd = pd.DataFrame(P).rolling(250 * PERIODS_PER_DAY, min_periods=1000).std().shift(1)
    med = np.nanmedian(sd.to_numpy(), axis=1, keepdims=True)
    return P / pd.DataFrame(np.maximum(sd.to_numpy(),
                                       0.1 * np.where(np.isfinite(med), med, 1.0))).bfill().to_numpy()


def main() -> None:
    p = load_panel()
    ts = pd.Series(pd.to_datetime(p.t))
    n = len(ts)
    day_codes = pd.factorize(ts.dt.normalize())[0]
    late = (ts >= HOLDOUT).to_numpy()
    XH, XL, XS, P100 = _blocks(p)
    A = 3000.0
    X679 = np.hstack([XH * np.sqrt(A), XL, XS, P100 * np.sqrt(0.1)])
    f_twin = walk_forward_embargo_blocked(X679, p.y, day_codes, 250, 1, A)
    q_twin = _q(f_twin, p.y, p.baseline)
    print(f"twin (h=1, blocked): QLIKE {np.nanmean(q_twin):.5f}", flush=True)

    def verdict(name, f_arm, act=None):
        q = _q(f_arm, p.y, p.baseline)
        d = q_twin - q
        if act is not None:
            d[~act] = np.nan
        md = np.isfinite(d)
        g = _hac_mean_t(d[md], 480)
        print(f"  {name:28s} QLIKE {np.nanmean(q):.5f}  DM {g:+.2f} "
              f"(2020+ {_hac_mean_t(d[md & late], 480):+.2f})  "
              f"{'PASS' if g >= 2.0 else 'FAIL'}", flush=True)

    # ---- 35a: alpha-scaled count ladder ------------------------------------------------
    bc, _ = base_columns(p)
    XB = np.ascontiguousarray(p.X[:, bc], dtype=np.float64)
    ii, jj = _upper(XB.shape[1])
    e_full = np.load(_p("har_resid.npz"))["e"]
    Zw = XB[TW : 2 * TW]
    Zw = (Zw - Zw.mean(0)) / (Zw.std(0) + 1e-12)
    ic = np.abs(_pair_ic(Zw, e_full[:TW])[ii, jj])
    order = np.argsort(-ic)
    for k, a_prod in ((200, 6e4), (400, 1.2e5)):
        sel = order[:k]
        Pk = _scale(XB[:, ii[sel]] * XB[:, jj[sel]])
        Xk = np.hstack([XH * np.sqrt(A), XL, XS, Pk * np.sqrt(A / a_prod)])
        f = walk_forward_embargo_blocked(Xk, p.y, day_codes, 250, 1, A)
        verdict(f"35a k={k} @ {a_prod:.0e}", f)

    # ---- 35b: cross-time (lagged-leg) products -----------------------------------------
    Zl = np.zeros_like(XB)
    Zl[1:] = XB[:-1]
    Zw1 = ((Zl[TW : 2 * TW]) - Zl[TW : 2 * TW].mean(0)) / (Zl[TW : 2 * TW].std(0) + 1e-12)
    # IC of lagged_i x current_j over the full asymmetric grid, via the same three-matmul trick
    e0 = e_full[:TW]
    ec = (e0 - e0.mean()) / (e0.std() + 1e-12)
    Cij = np.abs((Zw1 * ec[:, None]).T @ Zw / TW)  # proxy IC of product via covariance of legs*e
    # exact per-pair IC is expensive at 133^2; select by the proxy, then build exactly
    flat = np.argsort(-Cij.ravel())[:100]
    li, cj = np.unravel_index(flat, Cij.shape)
    Px = _scale(Zl[:, li] * XB[:, cj])
    Xx = np.hstack([X679, Px * np.sqrt(A / 3e4)])
    f = walk_forward_embargo_blocked(Xx, p.y, day_codes, 250, 1, A)
    verdict("35b cross-time products", f)

    # ---- 35c: transmission at h=1 -------------------------------------------------------
    G20, _, _ = _frame_and_scores()
    ng = len(G20)
    TRAIL, REFRESH = 504 * PERIODS_PER_DAY, 63 * PERIODS_PER_DAY
    Ghat = np.zeros((ng, G20.shape[1]))
    for start in range(TRAIL, ng, REFRESH):
        a, b = G20[start - TRAIL : start - 1], G20[start - TRAIL + 1 : start]
        az = (a - a.mean(0)) / (a.std(0) + 1e-12)
        bz = (b - b.mean(0)) / (b.std(0) + 1e-12)
        C = (az.T @ bz) / len(az)
        D = (C - C.T) / 2.0
        Ghat[start : min(start + REFRESH, ng)] = G20[start - 1 : min(start + REFRESH, ng) - 1] @ D
    F20 = np.zeros((n, G20.shape[1]))
    F20[2 * TW :] = np.nan_to_num(Ghat)
    F20 = _scale(F20)
    F20[~np.isfinite(F20)] = 0.0
    f = walk_forward_embargo_blocked(np.hstack([X679, F20]), p.y, day_codes, 250, 1, A)
    verdict("35c transmission @ h=1", f, act=np.abs(F20).sum(1) != 0.0)


if __name__ == "__main__":
    main()
