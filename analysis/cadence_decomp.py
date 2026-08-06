"""§29.5: decompose the per-bar-over-daily cadence increment at H=8 by channel.

The §21.1 caveat made testable: inside one dual-cadence pass, both coefficient vintages exist
at every bar, so hybrid predictions price the PRODUCT columns with per-bar coefficients while
everything else stays daily, and the complement. Two Shapley orderings bracket each channel's
freshness share, engine-exact (identical windows, identical solves):

  daily -> +fresh products      (product share, others stale)
  +fresh rest -> all per-bar    (product share, others fresh)

Pre-registered expectation (from §21.1 at h=1): the increment is linear-channel-driven; the
product share is small/zero. Readout only — no gate; this is attribution, not a new claim.
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
from analysis.synthesis import _p  # noqa: E402
from analysis.alpha_manifestation import TW  # noqa: E402
from analysis.wf import walk_forward_embargo_dualcadence  # noqa: E402

H = 8


def main() -> None:
    p = load_panel()
    ts = pd.Series(pd.to_datetime(p.t))
    late = (ts >= HOLDOUT).to_numpy()
    XH, XL, XS, P = _blocks(p)
    A = 3000.0
    X = np.hstack([XH * np.sqrt(A), XL, XS, P * np.sqrt(0.1)])
    n_cols = X.shape[1]
    prod_mask = np.zeros(n_cols, bool)
    prod_mask[-P.shape[1] :] = True
    yh, Bh = _y_horizon(p, H)
    pb, dy, hf, hr = walk_forward_embargo_dualcadence(X, yh, TW, H, A, fresh_mask=prod_mask)
    arms = {"all daily": dy, "fresh PRODUCTS only": hf, "fresh REST only": hr,
            "all per-bar": pb}
    qs = {}
    for name, arr in arms.items():
        f = np.full(len(yh), np.nan)
        f[TW:] = arr
        m = np.isfinite(f) & np.isfinite(yh)
        q = np.full(len(yh), np.nan)
        q[m] = _qlike_series(f[m], yh[m], Bh[m])
        qs[name] = q
        print(f"  {name:20s} QLIKE {np.nanmean(q):.5f} (2020+ {np.nanmean(q[late]):.5f})",
              flush=True)
    lags = 2 * H + 480

    def dm(a: str, b: str) -> str:
        d = qs[a] - qs[b]
        md = np.isfinite(d)
        return (f"dQLIKE {1e4*(np.nanmean(qs[a]) - np.nanmean(qs[b])):+.2f}e-4  "
                f"DM {_hac_mean_t(d[md], lags):+.2f} (2020+ {_hac_mean_t(d[md & late], lags):+.2f})")

    print(f"\n  total per-bar increment:            {dm('all daily', 'all per-bar')}")
    print(f"  PRODUCT share (others stale):       {dm('all daily', 'fresh PRODUCTS only')}")
    print(f"  PRODUCT share (others fresh):       {dm('fresh REST only', 'all per-bar')}")
    print(f"  LINEAR+ share (products stale):     {dm('all daily', 'fresh REST only')}")
    print(f"  LINEAR+ share (products fresh):     {dm('fresh PRODUCTS only', 'all per-bar')}")
    np.savez_compressed(_p("cadence_decomp_h8.npz"), perbar=pb, daily=dy,
                        hyb_prod=hf, hyb_rest=hr)
    print("cached -> cadence_decomp_h8.npz")


if __name__ == "__main__":
    main()
