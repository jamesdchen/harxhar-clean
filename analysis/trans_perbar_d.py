"""§37b': the transmission block with D refreshed EVERY BAR via sliding outer-product sums —
the cadence question applied to the arrow estimator itself. Gate >= +2.0 vs the 699 twin."""

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
from analysis.trans_exploit import _scale, _trans_block  # noqa: E402
from analysis.wf import walk_forward_embargo_blocked  # noqa: E402
from src.features.transforms.target import PERIODS_PER_DAY  # noqa: E402


def main() -> None:
    p = load_panel()
    ts = pd.Series(pd.to_datetime(p.t))
    n = len(ts)
    day_codes = pd.factorize(ts.dt.normalize())[0]
    late = (ts >= HOLDOUT).to_numpy()
    XH, XL, XS, P = _blocks(p)
    A = 3000.0
    X679 = np.hstack([XH * np.sqrt(A), XL, XS, P * np.sqrt(0.1)])
    G, _, _ = _frame_and_scores()
    ng, k = G.shape
    TRAIL = 504 * PERIODS_PER_DAY

    # per-bar D via sliding sum of lag-1 outer products g(s) g(s+1)^T over the trailing window
    Ghat = np.zeros((ng, k))
    S = np.zeros((k, k))
    for s in range(TRAIL - 1):
        S += np.outer(G[s], G[s + 1])
    for t in range(TRAIL, ng):
        # window pairs (s, s+1) for s in [t-TRAIL, t-2]; predict bar t from G[t-1]
        S += np.outer(G[t - 2], G[t - 1])
        S -= np.outer(G[t - TRAIL - 1], G[t - TRAIL])
        D = (S - S.T) / (2.0 * (TRAIL - 1))
        Ghat[t] = G[t - 1] @ D
    F = np.zeros((n, k))
    F[2 * TW :] = np.nan_to_num(Ghat)
    F = _scale(F)
    act = np.abs(F).sum(1) != 0.0

    F1q = _trans_block(G, n, lag=1, refresh_days=63)  # the 699 twin's quarterly block

    def q_of(X):
        f = walk_forward_embargo_blocked(X, p.y, day_codes, 250, 1, A)
        m = np.isfinite(f) & np.isfinite(p.y)
        q = np.full(n, np.nan)
        q[m] = _qlike_series(f[m], p.y[m], p.baseline[m])
        return q

    q_twin = q_of(np.hstack([X679, F1q]))
    q_pb = q_of(np.hstack([X679, F]))
    print(f"twin 699 (quarterly D): QLIKE {np.nanmean(q_twin):.5f}", flush=True)
    print(f"per-bar D:              QLIKE {np.nanmean(q_pb):.5f}", flush=True)
    d = q_twin - q_pb
    d[~act] = np.nan
    md = np.isfinite(d)
    g = _hac_mean_t(d[md], 480)
    print(f"37b' per-bar D vs quarterly D: DM {g:+.2f} "
          f"(2020+ {_hac_mean_t(d[md & late], 480):+.2f})   "
          f"gate: {'PASS' if g >= 2.0 else 'FAIL'}")


if __name__ == "__main__":
    main()
