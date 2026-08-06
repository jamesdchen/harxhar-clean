"""AM-15 / §40.1: the CLOSING Romano-Wolf step-down — the post-§36 claim family.

Everything from §37.1 onward passed only single gates. The family (each a QLIKE
loss-differential vs its own twin, same engine within each pair; series assembled from the
session's per-arm caches, so this module is cheap):

  trans_h1_perbar   699 (679+F20) vs 679, per-bar SM engine (§35c entry, +8.54)
  width_perbar      719 (679+F40) vs 699, per-bar SM engine (§38.2, +4.59)
  trans_h4_f20      679+F20 vs 679 twin at H=4 (§37.1, +5.00)
  trans_h16_f20     679+F20 vs 679 twin at H=16 under the α-law (§37.1, +5.85)
  union_span        GCV ridge-union smear vs 10-named, both Duan (§34.10, +3.12)
  quad40            719+100 new-direction quadratics vs 719 twin (§39b, +2.39)
  calendar_smear    named+DOW+short-day vs named (§34.15, +2.65)

Same machinery as AM-10: common-time-axis circular block bootstrap, gappy-support means,
per-claim HAC studentization, step-down max-null. FAILED arms are not family members (they
claim nothing); the §38 width LADDER is represented by its production consequence
(width_perbar), not by every rung.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.alpha_manifestation import TW  # noqa: E402
from analysis.alpha_panel import load_panel  # noqa: E402
from analysis.am10_sweep import sweep  # noqa: E402
from analysis.minimal_model import _qlike_series  # noqa: E402
from analysis.synthesis import _p  # noqa: E402

ARMCACHE = os.environ.get("ARMCACHE", "")


def _arm(tag):
    return np.load(os.path.join(ARMCACHE, tag + ".npz"))["v"]


def build():
    p = load_panel()
    n = len(p.t)
    cutoff = 2 * TW + 504 * 48  # transmission-block activation row

    def q_of(f):
        m = np.isfinite(f) & np.isfinite(p.y)
        q = np.full(n, np.nan)
        q[m] = _qlike_series(f[m], p.y[m], p.baseline[m])
        return q

    ds, names = [], []

    f679 = np.full(n, np.nan)
    f679[TW:] = np.load(_p("final_onestage.npz"))["yhat_bar"]
    f699 = np.load(_p("final_699_perbar.npz"))["yhat"]
    f719 = np.load(_p("pool40_perbar.npz"))["yhat"]
    q679, q699, q719 = q_of(f679), q_of(f699), q_of(f719)

    d = q679 - q699
    d[np.nan_to_num(f679) == np.nan_to_num(f699)] = np.nan
    ds.append(d)
    names.append("trans_h1_perbar")

    d = q699 - q719
    d[np.nan_to_num(f699) == np.nan_to_num(f719)] = np.nan
    ds.append(d)
    names.append("width_perbar")

    for hb, nm in ((4, "trans_h4_f20"), (16, "trans_h16_f20")):
        d = _arm(f"h40_{hb}_twin") - _arm(f"h40_{hb}__F20")
        d[:cutoff] = np.nan
        ds.append(d)
        names.append(nm)

    d = np.load(_p("smear_union_lossdiffs.npz"))["d_union_named"]
    ds.append(d)
    names.append("union_span")

    d = _arm("q40_twin719") - _arm("q40_arm")
    d[:cutoff] = np.nan
    ds.append(d)
    names.append("quad40")

    d = np.load(_p("cal_smear_lossdiff.npz"))["d"]
    ds.append(d)
    names.append("calendar_smear")

    D = np.column_stack(ds)
    np.savez_compressed(_p("am15_lossdiffs.npz"), D=D, names=np.array(names))
    return D, names


if __name__ == "__main__":
    if os.path.exists(_p("am15_lossdiffs.npz")):
        z = np.load(_p("am15_lossdiffs.npz"), allow_pickle=True)
        D, names = z["D"], list(z["names"])
        print("loaded cached loss diffs", flush=True)
    else:
        D, names = build()
    sweep(D, names, csv_path="results/alpha_manifestation/am15_sweep.csv",
          title="AM-15 closing Romano-Wolf step-down over the post-§36 family")
