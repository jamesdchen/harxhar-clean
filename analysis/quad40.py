"""§39b: quadratic content of the NEW arrow-carrying directions — pre-registered.

PC-quadratics from the 20-frame died long ago (casualty #6: spectral thresholds on the product
pool; pc_quadratics). But §38 just showed directions 21–40 carry real lead-lag arrows — they
are new territory the quadratic probes never touched with fresh motivation. Candidates: all
degree-2 terms involving at least one new direction (20 squares + 190 new×new + 400 new×old =
610), scores projected on the FROZEN first-window frame (first-window standardization — the
§22/§32 frozen-selection discipline), selected once by first-window |IC| against the HAR
residual, k = 100, floored-sd scaled, appended to the 719 design at the product penalty 3e4.

Gate: DM >= +2.0 vs the 719 twin (blocked engine, same-run). Recorded lean: FAIL — the
meta-law's record on quadratic cleverness is 0-for-6, and the §35a count ladder showed the
old product pool exhausted. The arm exists because the width result is exactly the kind of
fact that has been overturning such leans, and because a pass would say the new directions
carry CURVATURE, not just arrows.
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
from analysis.pool_width import _frame_q  # noqa: E402
from analysis.synthesis import _p  # noqa: E402
from analysis.trans_exploit import _scale, _trans_block  # noqa: E402
from analysis.wf import walk_forward_embargo_blocked  # noqa: E402


def main() -> None:
    p = load_panel()
    ts = pd.Series(pd.to_datetime(p.t))
    n = len(ts)
    day_codes = pd.factorize(ts.dt.normalize())[0]
    late = (ts >= HOLDOUT).to_numpy()
    XH, XL, XS, P = _blocks(p)
    A = 3000.0
    F40 = _trans_block(_frame_q(p, 40), n)
    X719 = np.hstack([XH * np.sqrt(A), XL, XS, P * np.sqrt(0.1), F40])

    # full-length scores on the frozen first-window frame (first-window standardization)
    bc, _ = base_columns(p)
    XB = np.ascontiguousarray(p.X[:, bc], dtype=np.float64)
    W0 = XB[TW : 2 * TW]
    mu0, sd0 = W0.mean(0), W0.std(0)
    live = sd0 > 1e-8
    sd0 = np.where(live, sd0, 1.0)
    lam_l, V_l = np.linalg.eigh(np.corrcoef(((W0 - mu0) / sd0)[:, live], rowvar=False))
    order = np.argsort(lam_l)[::-1]
    V = np.zeros((XB.shape[1], len(lam_l)))
    V[live] = V_l[:, order]
    S = ((XB - mu0) / sd0) @ V[:, :40]
    S = S / (S[TW : 2 * TW].std(0) + 1e-12)

    # candidate degree-2 terms involving at least one NEW direction (index 20..39)
    cand = [(i, i) for i in range(20, 40)]
    cand += [(i, j) for i in range(20, 40) for j in range(i + 1, 40)]
    cand += [(i, j) for i in range(0, 20) for j in range(20, 40)]
    e_full = np.load(_p("har_resid.npz"))["e"]
    ew = e_full[:TW]
    ez = (ew - np.nanmean(ew)) / (np.nanstd(ew) + 1e-12)
    Sw = S[TW : 2 * TW]
    ic = np.empty(len(cand))
    for k, (i, j) in enumerate(cand):
        q = Sw[:, i] * Sw[:, j]
        qz = (q - q.mean()) / (q.std() + 1e-12)
        ic[k] = np.abs(np.nanmean(qz * ez))
    top = np.argsort(-ic)[:100]
    Q = _scale(np.column_stack([S[:, i] * S[:, j] for i, j in np.array(cand)[top]]))
    print(f"selected 100/{len(cand)} new-direction quadratics "
          f"(|IC| {ic[top].min():.4f}..{ic[top].max():.4f})", flush=True)

    def q_of(X):
        f = walk_forward_embargo_blocked(X, p.y, day_codes, 250, 1, A)
        m = np.isfinite(f) & np.isfinite(p.y)
        q = np.full(n, np.nan)
        q[m] = _qlike_series(f[m], p.y[m], p.baseline[m])
        return q

    from analysis.armcache import memo
    q_twin = memo("q40_twin719", lambda: q_of(X719))
    q_arm = memo("q40_arm", lambda: q_of(np.hstack([X719, Q * np.sqrt(0.1)])))
    print(f"719 twin QLIKE {np.nanmean(q_twin):.5f}   +quad40 QLIKE {np.nanmean(q_arm):.5f}",
          flush=True)
    d = q_twin - q_arm
    md = np.isfinite(d)
    g = _hac_mean_t(d[md], 480)
    print(f"§39b new-direction quadratics: DM {g:+.2f} "
          f"(2020+ {_hac_mean_t(d[md & late], 480):+.2f})  "
          f"{'PASS' if g >= 2.0 else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
