"""§35c entry condition: the transmission block at h=1 under PER-BAR refit (SM dual engine,
both cadences in one pass) — the registered confirmation before it enters the production stack."""

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
from analysis.minimal_model import HOLDOUT, _hac_mean_t, _qlike_series  # noqa: E402
from analysis.wf import walk_forward_embargo_dualcadence  # noqa: E402
from src.features.transforms.target import PERIODS_PER_DAY  # noqa: E402


def main() -> None:
    p = load_panel()
    ts = pd.Series(pd.to_datetime(p.t))
    n = len(ts)
    late = (ts >= HOLDOUT).to_numpy()
    XH, XL, XS, P = _blocks(p)
    A = 3000.0
    X679 = np.hstack([XH * np.sqrt(A), XL, XS, P * np.sqrt(0.1)])
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
    sd = pd.DataFrame(F20).rolling(250 * PERIODS_PER_DAY, min_periods=1000).std().shift(1)
    med = np.nanmedian(sd.to_numpy(), axis=1, keepdims=True)
    F20 = F20 / pd.DataFrame(np.maximum(sd.to_numpy(),
                                        0.1 * np.where(np.isfinite(med), med, 1.0))).bfill().to_numpy()
    F20[~np.isfinite(F20)] = 0.0
    act = np.abs(F20).sum(1) != 0.0

    qs = {}
    for name, X in (("679", X679), ("699 +transmission", np.hstack([X679, F20]))):
        pb, _ = walk_forward_embargo_dualcadence(X, p.y, TW, 1, A)
        f = np.full(n, np.nan)
        f[TW:] = pb
        m = np.isfinite(f) & np.isfinite(p.y)
        q = np.full(n, np.nan)
        q[m] = _qlike_series(f[m], p.y[m], p.baseline[m])
        qs[name] = q
        if "699" in name:
            np.savez_compressed(os.path.join(os.environ["ALPHA_PANEL_CACHE"],
                                             "final_699_perbar.npz"), yhat=f)
        print(f"  {name:18s} PER-BAR QLIKE {np.nanmean(q):.5f} "
              f"(2020+ {np.nanmean(q[late]):.5f})", flush=True)
    d = qs["679"] - qs["699 +transmission"]
    d[~act] = np.nan
    md = np.isfinite(d)
    g = _hac_mean_t(d[md], 480)
    print(f"\n  transmission @ h=1 PER-BAR: DM {g:+.2f} (2020+ {_hac_mean_t(d[md & late], 480):+.2f})")
    print(f"  entry condition (>= +2.0): {'CONFIRMED — enters the stack' if g >= 2.0 else 'NOT CONFIRMED'}")


if __name__ == "__main__":
    main()
