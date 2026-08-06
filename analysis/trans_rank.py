"""§38.3: mechanics of the ~40-dim transmission span — three pre-registered probes.

§38.1 found the width curve peaks at q ≈ 40 (0.12788), shoulders at 80, and fails the twin
gate at 106. Three questions, one battery (blocked engine, h = 1, all arms same-run):

(a) SHRINKAGE — does the wider block want its own α? The 40 columns enter at the exog penalty
    3e3 by convention. α-ladder on the F40 block alone: {3e2, 3e3, 3e4} via column scaling.
    Gate: a rung moves production only at DM >= +2.0 vs the 3e3 rung. Recorded lean: 3e4 flat
    or small gain, 3e2 worse (the α-law says more weak columns want more shrinkage).

(b) SOURCE/TARGET DECOMPOSITION — where does q = 40's gain over 20 live? The q = 40 block
    factors as [T_old | T_new]: T_old = arrows from 40 sources into the OLD 20 target
    directions; T_new = arrows into the 20 NEW target directions. Arms: 679+T_old (wide
    sources alone) and 679+F20+T_new (new targets alone), each vs the 699 twin. Attribution,
    no gate. Recorded lean: wide-sources-into-old-targets carries most of it (the old targets
    are the proven carriers).

(c) ESTIMATION-NOISE DISCRIMINATOR — is the q = 106 failure genuine emptiness of the tail
    directions, or D-estimation noise (free params grow as q²: 5,565 at 106 vs 780 at 40, on
    a fixed 504d trail)? Arms: q = 106 with a 1008d trail vs q = 106 at 504d (in-run pair),
    plus q = 40 at 1008d as the control (if longer trail helps everywhere, it isn't a rescue).
    Gate: rescue = q106-1008d vs q106-504d DM >= +2.0. Recorded lean: partial rescue — both
    effects live.
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
from analysis.pool_width import _frame_q  # noqa: E402
from analysis.trans_exploit import _scale, _trans_block  # noqa: E402
from analysis.wf import walk_forward_embargo_blocked  # noqa: E402
from src.features.transforms.target import PERIODS_PER_DAY  # noqa: E402


def _ghat_raw(G, n_all, trail_days=504):
    """Unscaled transmission forecasts (for column slicing before scaling)."""
    ng = len(G)
    TRAIL, REFRESH = trail_days * PERIODS_PER_DAY, 63 * PERIODS_PER_DAY
    Ghat = np.zeros((ng, G.shape[1]))
    for start in range(TRAIL, ng, REFRESH):
        a, b = G[start - TRAIL : start - 1], G[start - TRAIL + 1 : start]
        az = (a - a.mean(0)) / (a.std(0) + 1e-12)
        bz = (b - b.mean(0)) / (b.std(0) + 1e-12)
        C = (az.T @ bz) / len(az)
        D = (C - C.T) / 2.0
        end = min(start + REFRESH, ng)
        Ghat[start:end] = G[start - 1 : end - 1] @ D
    F = np.zeros((n_all, G.shape[1]))
    F[2 * TW :] = np.nan_to_num(Ghat)
    return F


def main() -> None:
    p = load_panel()
    ts = pd.Series(pd.to_datetime(p.t))
    n = len(ts)
    day_codes = pd.factorize(ts.dt.normalize())[0]
    late = (ts >= HOLDOUT).to_numpy()
    XH, XL, XS, P = _blocks(p)
    A = 3000.0
    X679 = np.hstack([XH * np.sqrt(A), XL, XS, P * np.sqrt(0.1)])

    G40 = _frame_q(p, 40)
    G20 = G40[:, :20]
    raw40 = _ghat_raw(G40, n)
    F40 = _scale(raw40)
    F20 = _trans_block(G20, n)
    T_old = _scale(raw40[:, :20])
    T_new = _scale(raw40[:, 20:])
    G106 = _frame_q(p, 106)
    F106s = _trans_block(G106, n, trail_days=504)
    F106l = _trans_block(G106, n, trail_days=1008)
    F40l = _trans_block(G40, n, trail_days=1008)

    def q_of(X):
        walk = walk_forward_embargo_blocked(X, p.y, day_codes, 250, 1, A)
        m = np.isfinite(walk) & np.isfinite(p.y)
        q = np.full(n, np.nan)
        q[m] = _qlike_series(walk[m], p.y[m], p.baseline[m])
        return q

    arms = {
        "699 twin (F20)": (np.hstack([X679, F20]), F20),
        "F40 @3e3": (np.hstack([X679, F40]), F40),
        "F40 @3e2": (np.hstack([X679, F40 * np.sqrt(10.0)]), F40),
        "F40 @3e4": (np.hstack([X679, F40 * np.sqrt(0.1)]), F40),
        "b1 T_old (wide sources)": (np.hstack([X679, T_old]), T_old),
        "b2 F20+T_new (new targets)": (np.hstack([X679, F20, T_new]), T_new),
        "c q106 504d": (np.hstack([X679, F106s]), F106s),
        "c q106 1008d": (np.hstack([X679, F106l]), F106l),
        "c q40 1008d": (np.hstack([X679, F40l]), F40l),
    }
    qs, acts = {}, {}
    for name, (X, F) in arms.items():
        qs[name] = q_of(X)
        acts[name] = np.abs(F).sum(1) != 0.0
        print(f"  {name:28s} QLIKE {np.nanmean(qs[name]):.5f}", flush=True)

    def dm(a, b, label):
        d = qs[a] - qs[b]
        d[~(acts[a] & acts[b])] = np.nan
        md = np.isfinite(d)
        g = _hac_mean_t(d[md], 480)
        print(f"  {label:44s} DM {g:+.2f} (2020+ {_hac_mean_t(d[md & late], 480):+.2f})",
              flush=True)
        return g

    print("\n(a) alpha ladder on F40 (gate >= +2.0 to move off 3e3):", flush=True)
    dm("F40 @3e3", "F40 @3e2", "3e2 vs 3e3")
    dm("F40 @3e3", "F40 @3e4", "3e4 vs 3e3")
    print("\n(b) source/target decomposition (attribution vs the 699 twin):", flush=True)
    dm("699 twin (F20)", "F40 @3e3", "full q40 vs q20 (reference)")
    dm("699 twin (F20)", "b1 T_old (wide sources)", "wide sources, old targets")
    dm("699 twin (F20)", "b2 F20+T_new (new targets)", "new targets added to q20")
    print("\n(c) estimation-noise discriminator (rescue gate >= +2.0):", flush=True)
    dm("c q106 504d", "c q106 1008d", "q106: 1008d vs 504d trail")
    dm("F40 @3e3", "c q40 1008d", "q40 control: 1008d vs 504d")


if __name__ == "__main__":
    main()
