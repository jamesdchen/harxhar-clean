"""0DTE model-free implied variance (the VIX formula on the same-day SPXW strip)
as panel columns, at every 30-minute chain stamp.

The VIX-family regressors (vix, vvix, vix3m) are Cboe index prints: a 30-day
model-free implied variance, its 3-month sibling and the implied vol of VIX
options.  None of those can be rebuilt from the 0DTE chain (no 23-37 day
expiries, no VIX options).  What can -- and is horizon-matched to the forecast
target -- is the same calculation on the 0DTE strip: the model-free implied
variance from the stamp to the settlement.  This module writes it as three
panel columns on the ``endbartime`` grid so ``load_raw_data`` merges them like
any other feed:

  mfiv0_annvol     100 * sqrt(C / T): the strip's variance to the close,
                   annualized, in VIX units.  Deliberately NOT named with the
                   ``vix`` stem: the 30-day VIX is a clock-free level and is
                   exempt from diurnal division, but the 0DTE annualized vol
                   climbs mechanically through the day (median 35 at 10:00,
                   85 at 15:30, the closing bar's variance over half an hour),
                   so it takes the default log + diurnal division
  mfiv0_perbar_rv  C / remaining bars: implied variance per remaining
                   30-minute bar, in the target's units (sqrt + diurnal, the
                   ``rv`` stem)
  mfiv0_vvix       trailing-12-stamp std of d log(mfiv0_annvol): realized
                   vol-of-implied, the nearest thing the chain has to VVIX
                   (flat across the clock; the ``vix`` stem's exemption fits)

C is ``_mfiv_one`` of experiments/spxw_mfiv_toclose.py, 2 * sum dK/K^2 Q(K)
- (F/K0 - 1)^2, the C_t of the every-bar strip book; the gate below checks
that on every one of its stamps.  Remaining time is the chain's own
``hours_to_expiration`` (so half-days settle at 13:00); frozen stamps
(hours_to_expiration <= 0, the known half-session freezes) are dropped.

Panel clock: naive ET, bar-END labelled.  The quote at 15:30 ET sits on the
row labelled 15:30, so the design row of the 15:30-16:00 bar reaches it
through the executor's shift(1): known at 15:30, never later.  Rows after the
feature panel's last stamp are not written (the grid must not grow); the
full 2020-2025 series, holdout included, goes to the diagnostics file.

  python experiments/build_spxw_mfiv_panel.py [--parts 8]
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from experiments.spxw_mfiv_toclose import CHAIN, _mfiv_one  # noqa: E402

PANEL_END_SOURCE = os.path.join(ROOT, "data", "core_stats.parquet")
OUT_PANEL = os.path.join(ROOT, "data", "spxw_mfiv.parquet")
OUT_DIAG = os.path.join(ROOT, "results", "spxw_pnl", "mfiv_panel_diag.parquet")
EVERYBAR = os.path.join(ROOT, "results", "spxw_pnl", "everybar_mtm_trades.parquet")
ET = "America/New_York"
COLS = [
    "timestamp",
    "expiration",
    "strike",
    "cp",
    "mid",
    "underlying_price",
    "hours_to_expiration",
]
VOV_WINDOW = 12  # stamps: one session of 30-minute changes
FEATURES = ["mfiv0_annvol", "mfiv0_perbar_rv", "mfiv0_vvix"]
HOURS_PER_YEAR = 365.25 * 24.0


def _bounds(parts: int) -> list[tuple[int, object, object | None]]:
    """Split the chain's stamps into ``parts`` contiguous ranges, in the column's own dtype."""
    t = pd.read_parquet(CHAIN, columns=["timestamp"])["timestamp"]
    t = t.drop_duplicates().sort_values().to_numpy()
    cuts = np.linspace(0, len(t), parts + 1, dtype=int)
    return [
        (i, t[cuts[i]], None if i == parts - 1 else t[cuts[i + 1]])
        for i in range(parts)
    ]


def shard(args: tuple[int, object, object | None]) -> pd.DataFrame:
    part, lo, hi = args
    filters = [("timestamp", ">=", lo)]
    if hi is not None:
        filters.append(("timestamp", "<", hi))
    df = pd.read_parquet(CHAIN, columns=COLS, filters=filters)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["expiration"] = pd.to_datetime(df["expiration"])
    rows = []
    for t0, snap in df.groupby("timestamp", sort=True):
        exp = pd.Timestamp(snap["expiration"].iloc[0])
        meta = _mfiv_one(snap, pd.Timestamp(t0), exp)
        if meta is None:
            continue
        rows.append(
            {
                "t": pd.Timestamp(t0),
                "expiration": exp,
                "C": float(meta["mfiv_int"]),
                "T_1600": float(meta["T"]),
                "n_otm": int(meta["n_otm"]),
                "F": float(meta["F"]),
                "hours_to_expiration": float(snap["hours_to_expiration"].median()),
            }
        )
    print(
        f"part {part}: {df['timestamp'].nunique():,} stamps -> {len(rows):,} rows",
        flush=True,
    )
    return pd.DataFrame(rows)


def gate_against_strip_book(d: pd.DataFrame) -> None:
    """C must equal the every-bar strip book's C_t on every one of its stamps."""
    eb = pd.read_parquet(EVERYBAR, columns=["t", "C", "n_otm"])
    eb["t"] = pd.to_datetime(eb["t"], utc=True)
    j = eb.merge(d[["t", "C", "n_otm"]], on="t", how="left", suffixes=("_book", ""))
    miss = int(j["C"].isna().sum())
    gap = float(np.nanmax(np.abs(j["C"].to_numpy(float) - j["C_book"].to_numpy(float))))
    n_ne = int((j["n_otm"] != j["n_otm_book"]).sum())
    print(
        f"GATE every-bar strip book, {len(j):,} stamps: missing {miss}, "
        f"max |dC| {gap:.1e}, n_otm mismatches {n_ne}"
    )
    if miss or gap > 1e-12 or n_ne:
        raise SystemExit("GATE FAILED: the panel's C is not the strip book's C_t")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", type=int, default=8)
    a = ap.parse_args()
    with ProcessPoolExecutor(max_workers=a.parts) as ex:
        parts = list(ex.map(shard, _bounds(a.parts)))
    d = pd.concat(parts, ignore_index=True).sort_values("t").reset_index(drop=True)
    gate_against_strip_book(d)

    frozen = d["hours_to_expiration"] <= 0
    d = d[~frozen].copy()
    hte = d["hours_to_expiration"].to_numpy(float)
    n_rem = np.maximum(np.rint(hte * 2.0), 1.0)
    c = d["C"].to_numpy(float)
    c = np.where(c > 0, c, np.nan)  # a non-positive strip integral is no quote
    d["mfiv0_annvol"] = 100.0 * np.sqrt(c / (hte / HOURS_PER_YEAR))
    d["mfiv0_perbar_rv"] = c / n_rem
    d["n_rem"] = n_rem
    lv = np.log(d["mfiv0_annvol"])
    d["mfiv0_vvix"] = lv.diff().rolling(VOV_WINDOW, min_periods=VOV_WINDOW // 2).std()
    d["endbartime"] = d["t"].dt.tz_convert(ET).dt.tz_localize(None)

    full_day = np.isclose(
        d["T_1600"].to_numpy(float) * HOURS_PER_YEAR, hte, atol=1.0 / 60.0
    )
    print(
        f"stamps {len(d):,} on {d['endbartime'].dt.normalize().nunique():,} sessions "
        f"{d['endbartime'].min()} .. {d['endbartime'].max()}; frozen dropped "
        f"{int(frozen.sum())}; early-close stamps {int((~full_day).sum())}; "
        f"no-quote stamps {int(np.isnan(c).sum())}"
    )
    by_clock = d.groupby(d["endbartime"].dt.strftime("%H:%M"))[
        ["mfiv0_annvol", "mfiv0_perbar_rv", "mfiv0_vvix", "n_rem"]
    ].median()
    print(by_clock.to_string(float_format=lambda v: f"{v:.6g}"))

    os.makedirs(os.path.dirname(OUT_DIAG), exist_ok=True)
    d.to_parquet(OUT_DIAG, index=False)
    panel_end = pd.to_datetime(
        pd.read_parquet(PANEL_END_SOURCE, columns=["endbartime"])["endbartime"]
    ).max()
    p = d.loc[d["endbartime"] <= panel_end, ["endbartime", *FEATURES]].reset_index(
        drop=True
    )
    p.to_parquet(OUT_PANEL, index=False)
    print(
        f"wrote {OUT_PANEL}: {len(p):,} rows, last {p['endbartime'].max()} "
        f"(panel ends {panel_end}); diagnostics {OUT_DIAG}"
    )


if __name__ == "__main__":
    main()
