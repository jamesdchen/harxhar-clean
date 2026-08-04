"""Build overnight_gap (#4) -- cumulative SIGNED return over the cross-midnight night run (20->9).

The directional sibling of the (null) overnight-VOL cumrv_x_open: cumsum of per-bar sumret over the night
window {20..23, 0..9} (segmented on night-entry, crosses the midnight wrap), causal (shift(1) within the
night run so the open bar sees only through t-1). Gated to the open (hour==9) in resid_amortized._gap_open.
Aligned to cache rows: load_raw_data('data')[3125:]. Output: results/intraday_feats/overnight_gap.npy.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (repo root + siblings on sys.path)

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.loading import load_raw_data
from src.features.extractors.har import resolve_har_lags

DATA_PATH = "data"
OFFSET: int = resolve_har_lags()[-1]
OUT_DIR = "results/intraday_feats"


def build_gap(df) -> np.ndarray:
    r = df["sumret"].astype(np.float64).to_numpy()
    hour = df["t"].dt.hour.to_numpy()
    night = (hour >= 20) | (
        hour <= 9
    )  # overnight window feeding the open (AH-end -> pre-market)
    entry = np.concatenate(
        [[night[0]], night[1:] & ~night[:-1]]
    )  # new night run at each 19->20 entry
    night_id = np.cumsum(entry.astype(int))
    cum = pd.Series(np.where(night, r, 0.0)).groupby(night_id).cumsum()
    cum_caus = (
        cum.groupby(night_id).shift(1).fillna(0.0)
    )  # causal: through t-1 within the night run
    return cum_caus.to_numpy(dtype=np.float64)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_raw_data(DATA_PATH, allow_missing=True)
    n_full = len(df)
    gap_full = build_gap(df)

    jf = n_full - 50
    df2 = df.copy()
    df2.loc[df2.index[jf], "sumret"] = 1e6
    g2 = build_gap(df2)
    eq = bool(np.array_equal(gap_full[:jf], g2[:jf]))
    fd = int(np.where(gap_full != g2)[0][0]) if not eq else -1
    print(
        "[causality] [:jf] identical=%s first_diff=%d -> %s"
        % (eq, fd, "PASS" if (eq or fd >= jf) else "FAIL"),
        flush=True,
    )

    gap = np.ascontiguousarray(gap_full[OFFSET:])
    np.save(f"{OUT_DIR}/overnight_gap.npy", gap)
    print(
        "SAVED overnight_gap.npy len=%d min=%.5f med=%.5f max=%.5f frac0=%.3f"
        % (
            len(gap),
            gap.min(),
            float(np.median(gap)),
            gap.max(),
            float((gap == 0).mean()),
        ),
        flush=True,
    )
    print("OVERNIGHT_GAP_BUILD_DONE", flush=True)


if __name__ == "__main__":
    main()
