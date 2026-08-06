"""§38: the pool-width arm — "how do we know it's just 20?" (user challenge, pre-registered).

QPOOL = 20 is an inherited convention (map_monitor.py, §23-era), never varied: the third
unexamined convention this session's questions have exposed (products refit cadence §18,
D refresh cadence §37b'). Bracketing evidence on record: participation ratio of the
first-window spectrum ~18, raw iid MP edge keeps 22, dependence-adjusted edge keeps 5,
mid-spectrum directions gauge-degenerate (§27 C5).

Arm: q in {5, 10, 40} vs the production 20. Top-q eigenvectors of the SAME first-window
correlation (the frames nest: 5 and 10 are sub-frames of 20; 40 extends it), identical
transmission pipeline (lag-1 antisymmetric D_q, trailing-504d, quarterly refresh, floored-sd
scaling), q columns appended to the 679 design at the exog penalty, blocked engine, h = 1.

Gates: (i) per-width increment vs the 679 twin, DM >= +2.0; (ii) production moves off 20 only
if some q beats the q = 20 design head-to-head at |DM| >= 2.0. Expectations recorded before the
run: a plateau — q = 5 materially weaker (the §23 carriers PC9/PC14/PC16/PC19 live above index
5), q = 10 slightly weaker, q = 40 flat-to-worse (arrows into gauge directions are noise). A
flat 10-40 plateau is itself the informative outcome: "20" is a generous cover of a low-dim
transmission span, not a tuned knob.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.alpha_law import _blocks  # noqa: E402
from analysis.alpha_manifestation import TW  # noqa: E402
from analysis.alpha_panel import load_panel  # noqa: E402
from analysis.minimal_model import HOLDOUT, _hac_mean_t, _qlike_series  # noqa: E402
from analysis.nl_sparsity import base_columns  # noqa: E402
from analysis.trans_exploit import _trans_block  # noqa: E402
from analysis.wf import walk_forward_embargo_blocked  # noqa: E402

WIDTHS = tuple(int(w) for w in os.environ.get("WIDTHS", "5,10,20,40").split(","))


def _frame_q(p, q: int) -> np.ndarray:
    """Top-q scores on the frozen first-window frame (map_monitor._frame_and_scores, width q)."""
    bc, _ = base_columns(p)
    X = np.ascontiguousarray(p.X[TW:, bc], dtype=np.float64)
    mu0, sd0 = X[:TW].mean(0), X[:TW].std(0)
    live = sd0 > 1e-8
    sd0 = np.where(live, sd0, 1.0)
    lam_l, V_l = np.linalg.eigh(np.corrcoef(((X[:TW] - mu0) / sd0)[:, live], rowvar=False))
    order = np.argsort(lam_l)[::-1]
    V = np.zeros((X.shape[1], len(lam_l)))
    V[live] = V_l[:, order]
    W = V[:, :q] / sd0[:, None]
    G = (X[TW:] - X[TW:].mean(0)) @ W
    return (G - G.mean(0)) / (G.std(0) + 1e-12)


def main() -> None:
    p = load_panel()
    ts = pd.Series(pd.to_datetime(p.t))
    n = len(ts)
    day_codes = pd.factorize(ts.dt.normalize())[0]
    late = (ts >= HOLDOUT).to_numpy()
    XH, XL, XS, P = _blocks(p)
    A = 3000.0
    X679 = np.hstack([XH * np.sqrt(A), XL, XS, P * np.sqrt(0.1)])

    def q_of(X):
        f = walk_forward_embargo_blocked(X, p.y, day_codes, 250, 1, A)
        m = np.isfinite(f) & np.isfinite(p.y)
        q = np.full(n, np.nan)
        q[m] = _qlike_series(f[m], p.y[m], p.baseline[m])
        return q

    q_twin = q_of(X679)
    print(f"twin 679 (h=1, blocked): QLIKE {np.nanmean(q_twin):.5f}", flush=True)

    act = None
    qs = {}
    for w in WIDTHS:
        F = _trans_block(_frame_q(p, w), n, lag=1, refresh_days=63)
        if act is None:
            act = np.abs(F).sum(1) != 0.0
        qs[w] = q_of(np.hstack([X679, F]))
        d = q_twin - qs[w]
        d[~act] = np.nan
        md = np.isfinite(d)
        g = _hac_mean_t(d[md], 480)
        print(f"  q={w:2d}: QLIKE {np.nanmean(qs[w]):.5f}  vs twin DM {g:+.2f} "
              f"(2020+ {_hac_mean_t(d[md & late], 480):+.2f})  "
              f"{'PASS' if g >= 2.0 else 'FAIL'}", flush=True)

    print("\nhead-to-head vs q=20 (move off 20 only at |DM| >= 2.0):", flush=True)
    for w in WIDTHS:
        if w == 20:
            continue
        d = qs[20] - qs[w]
        d[~act] = np.nan
        md = np.isfinite(d)
        g = _hac_mean_t(d[md], 480)
        print(f"  q={w:2d} vs q=20: DM {g:+.2f} (2020+ {_hac_mean_t(d[md & late], 480):+.2f})  "
              f"{'MOVE' if abs(g) >= 2.0 and g > 0 else 'STAY 20'}", flush=True)


if __name__ == "__main__":
    main()
