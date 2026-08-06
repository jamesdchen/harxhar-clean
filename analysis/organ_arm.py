"""§29.4 / AM-06: the announcement organ ported onto the alpha-manifestation panel.

Construction imported verbatim from the msweep settled lever (straddle_v3.py lines 54-77):
per release type, {log1p(bars since), log1p(bars until), count in next H bars,
log1p(bars-until capped at H+1)} — all ex-ante (scheduled releases). Organ rides the LIGHT
block (their measured configuration): backbone penalty. One arm at H=8, blocked engine, twin
recomputed same-engine, gate QLIKE DM >= +2.0.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.alpha_law import _blocks  # noqa: E402
from analysis.alpha_panel import load_panel  # noqa: E402
from analysis.minimal_model import HOLDOUT, _hac_mean_t, _qlike_series  # noqa: E402
from analysis.straddle_horizon import _y_horizon  # noqa: E402
from analysis.wf import walk_forward_embargo_blocked  # noqa: E402

H = 8


def organ(ts: pd.Series, hb: int) -> np.ndarray:
    rel = pd.read_parquet("data/releases.parquet")
    rel["endbartime"] = pd.to_datetime(rel["endbartime"])
    mr = pd.DataFrame({"t": ts}).merge(rel, left_on="t", right_on="endbartime",
                                       how="left").fillna(0.0)
    types = [c for c in rel.columns if c != "endbartime"]
    flags = mr[types].to_numpy()
    n = len(ts)
    ar = np.arange(n)
    ANN = np.zeros((n, 4 * len(types)))
    for i in range(len(types)):
        f = flags[:, i]
        idx = np.flatnonzero(f)
        last = np.full(n, -float(n))
        nxt = np.full(n, float(n))
        pos = np.searchsorted(idx, ar, side="right")
        hp = pos > 0
        last[hp] = idx[np.clip(pos - 1, 0, None)][hp]
        hn = pos < len(idx)
        nxt[hn] = idx[np.clip(pos, None, len(idx) - 1)][hn]
        cum = np.concatenate([[0], np.cumsum(f)])
        cnt = np.zeros(n)
        cnt[: n - hb] = cum[hb + 1 : n + 1] - cum[1 : n - hb + 1]
        ANN[:, 4 * i] = np.log1p(ar - last)
        ANN[:, 4 * i + 1] = np.log1p(nxt - ar)
        ANN[:, 4 * i + 2] = cnt
        ANN[:, 4 * i + 3] = np.log1p(np.minimum(nxt - ar, hb + 1.0))
    return ANN


def main() -> None:
    p = load_panel()
    ts = pd.Series(pd.to_datetime(p.t))
    day_codes = pd.factorize(ts.dt.normalize())[0]
    late = (ts >= HOLDOUT).to_numpy()
    XH, XL, XS, P = _blocks(p)
    yh, Bh = _y_horizon(p, H)
    ANN = organ(ts, H)
    print(f"organ: {ANN.shape[1]} cols, {int((ANN[:, 2::4].sum()))} release-in-window "
          f"bar-events", flush=True)
    A = 3000.0
    lags = 2 * H + 480
    qs = {}
    arms = {
        "679 twin": np.hstack([XH * np.sqrt(A), XL, XS, P * np.sqrt(0.1)]),
        "679 + organ (light)": np.hstack([XH * np.sqrt(A), ANN * np.sqrt(A), XL, XS,
                                          P * np.sqrt(0.1)]),
    }
    for name, X in arms.items():
        pred = walk_forward_embargo_blocked(X, yh, day_codes, 250, 1, A)
        m = np.isfinite(pred) & np.isfinite(yh)
        q = np.full(len(yh), np.nan)
        q[m] = _qlike_series(pred[m], yh[m], Bh[m])
        qs[name] = q
        print(f"  {name:22s} QLIKE {np.nanmean(q):.5f} (2020+ {np.nanmean(q[late]):.5f})",
              flush=True)
    d = qs["679 twin"] - qs["679 + organ (light)"]
    g = _hac_mean_t(d[np.isfinite(d)], lags)
    print(f"\n  organ vs twin at H={H}: QLIKE DM {g:+.2f} "
          f"(2020+ {_hac_mean_t(d[late & np.isfinite(d)], lags):+.2f})")
    print(f"  gate (>= +2.0): {'PASS' if g >= 2.0 else 'FAIL'}")


if __name__ == "__main__":
    main()
