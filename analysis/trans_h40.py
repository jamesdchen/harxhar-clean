"""§39a: the transmission term structure at the MEASURED width — pre-registered.

§37.1's term structure (+5.00 H=4 / +2.58 H=8 / +1.38 FAIL H=13 / +5.85 H=16) was computed at
the inherited width q = 20. §38 then measured the width at ~40 with the h = 1 gain nearly
doubling (+2.29 head-to-head, +4.59 per-bar). If width interacts with horizon, every rung is
stale — and the rung that matters most is H = 13: the EOD/VRP deliverable horizon, the one
place transmission FAILED. Arms per H in {8, 13, 16}: twin 679, +F20, +F40 (same-run, blocked
engine; §22 penalty at 8/13, the α-law penalty at 16 — each horizon's production solver).

Gates: (i) F40 vs F20 >= +2.0 — the measured width enters that horizon's stack; (ii) at
H = 13 only, F40 vs twin >= +2.0 — the revival gate (transmission would enter the EOD stack
for the first time). Recorded leans: H = 8 and 16 widen (their content is the same arrows),
H = 13 revival genuinely uncertain — the §37.1 failure tracked the open-concentration of the
EOD edge, which width does not obviously fix.
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
from analysis.pool_width import _frame_q  # noqa: E402
from analysis.straddle_horizon import _y_horizon  # noqa: E402
from analysis.trans_exploit import _trans_block  # noqa: E402
from analysis.wf import walk_forward_embargo_blocked  # noqa: E402


def main() -> None:
    p = load_panel()
    ts = pd.Series(pd.to_datetime(p.t))
    n = len(ts)
    day_codes = pd.factorize(ts.dt.normalize())[0]
    late = (ts >= HOLDOUT).to_numpy()
    XH, XL, XS, P = _blocks(p)
    A = 3000.0
    F20 = _trans_block(_frame_q(p, 20), n)
    F40 = _trans_block(_frame_q(p, 40), n)
    act = np.abs(F40).sum(1) != 0.0

    for hb in (8, 13, 16):
        yh, Bh = _y_horizon(p, hb)
        a_solver = A * 16 if hb == 16 else A
        X679 = np.hstack([XH * np.sqrt(a_solver), XL, XS, P * np.sqrt(0.1)])
        qs = {}
        for name, X in (("twin", X679), ("+F20", np.hstack([X679, F20])),
                        ("+F40", np.hstack([X679, F40]))):
            fq = walk_forward_embargo_blocked(X, yh, day_codes, 250, 1, a_solver)
            m = np.isfinite(fq) & np.isfinite(yh)
            q = np.full(n, np.nan)
            q[m] = _qlike_series(fq[m], yh[m], Bh[m])
            qs[name] = q
            print(f"H={hb:2d} {name:5s} QLIKE {np.nanmean(q):.5f}", flush=True)
        for a, b, label, gate in (("twin", "+F20", "F20 vs twin", 2.0),
                                  ("twin", "+F40", "F40 vs twin", 2.0),
                                  ("+F20", "+F40", "F40 vs F20 (width gate)", 2.0)):
            d = qs[a] - qs[b]
            d[~act] = np.nan
            md = np.isfinite(d)
            g = _hac_mean_t(d[md], 2 * hb + 480)
            print(f"  H={hb:2d} {label:26s} DM {g:+.2f} "
                  f"(2020+ {_hac_mean_t(d[md & late], 2 * hb + 480):+.2f})  "
                  f"{'PASS' if g >= gate else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
