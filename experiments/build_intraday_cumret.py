"""Build cumret -- the cumulative intraday SIGNED-return path, the directional sibling of cumrv.

cumrv accumulates abnormal vol (|magnitude|) within the day; cumret accumulates the SIGNED return
(`sumret`, per-bar, ~zero-mean) within the TRUE calendar day, causal (shift(1) WITHIN the day so bar t
sees only through t-1). NO diurnal divide (signed returns are ~zero-mean; dividing by a per-slot mean
near 0 is meaningless) -- this is the raw intraday cumulative move = "how far has price travelled since
the day's open". Gated to the close (in resid_amortized._cumret_close) it tests the close mean-reversion
of the day's directional move (the leverage / intraday-reversal effect HAR cannot see).

Aligned to the cache rows: load_raw_data('data')[3125:] (offset = HAR burn-in), same as cumrv_real.
Output: results/intraday_feats/cumret.npy (ungated, robust-scaled in resid_amortized at use time).
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (repo root + siblings on sys.path)

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.loading import load_raw_data
from src.features.extractors.har import resolve_har_lags

DATA_PATH = "data"
OFFSET: int = resolve_har_lags()[
    -1
]  # 3125 -- HAR max-lag burn-in dropped by the executor
OUT_DIR = "results/intraday_feats"


def build_cumret(df) -> np.ndarray:
    r = df["sumret"].astype(np.float64)  # per-bar signed return
    day = df["t"].dt.normalize()  # TRUE calendar day
    cum = r.groupby(
        day
    ).cumsum()  # cumulative intraday signed return (incl. current bar)
    cum_caus = (
        cum.groupby(day).shift(1).fillna(0.0)
    )  # causal: through t-1, reset each day
    return cum_caus.to_numpy(dtype=np.float64)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_raw_data(DATA_PATH, allow_missing=True)
    n_full = len(df)
    cumret_full = build_cumret(df)

    # Causality check: perturb a FUTURE sumret, recompute, earlier values must be byte-identical.
    jf = n_full - 50
    df2 = df.copy()
    df2.loc[df2.index[jf], "sumret"] = 1e6
    c2 = build_cumret(df2)
    eq = bool(np.array_equal(cumret_full[:jf], c2[:jf]))
    first_diff = int(np.where(cumret_full != c2)[0][0]) if not eq else -1
    print(
        "[causality] [:jf] identical=%s first_diff=%d (>= jf=%d) -> %s"
        % (eq, first_diff, jf, "PASS" if (eq or first_diff >= jf) else "FAIL"),
        flush=True,
    )

    cumret = np.ascontiguousarray(cumret_full[OFFSET:])
    np.save(f"{OUT_DIR}/cumret.npy", cumret)
    print(
        "SAVED cumret.npy len=%d min=%.5f med=%.5f max=%.5f frac0=%.3f"
        % (
            len(cumret),
            cumret.min(),
            float(np.median(cumret)),
            cumret.max(),
            float((cumret == 0).mean()),
        ),
        flush=True,
    )
    print("CUMRET_BUILD_DONE", flush=True)


if __name__ == "__main__":
    main()
