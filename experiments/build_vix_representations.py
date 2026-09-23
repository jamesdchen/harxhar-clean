"""Representations of the VIX family for the per-bar regression -> data/vix_representations.parquet.

The incumbent design carries the Cboe prints as log LEVELS (``vix``, ``vvix``,
``vix3m``: the ``vix`` stem exempts them from the diurnal division, the default
rule logs them, the HAR ladder then takes six rolling means of each).  The
target is on the sqrt-ratio scale, sqrt(RV / diurnal baseline), so a model
linear in log-VIX is not linear in what it forecasts.  This module derives the
alternative representations on the panel's ``endbartime`` grid (48 bars a day,
overnight included -- the grid of data/vix_and_voldemand.parquet and
data/core_stats.parquet); ``load_raw_data`` merges the file like any other
feed.  Each name is chosen so the stem rules of
src/features/transforms/target.py give the intended transform (checked in
``STEM_RULES`` below):

  impl30_perbar_rv   (VIX/100)^2 / (252 * 48): the 30-day implied variance per
                     30-minute bar, in the target's units.  ``rv`` stem -> sqrt;
                     no ``vix`` stem -> diurnal division: implied vol per bar
                     over the slot baseline, the representation linear in the
                     target.
  vix_slope_3m       log(VIX3M / VIX): the term slope.  Signed -> identity;
                     ``vix`` stem -> no diurnal division.
  vvix_over_vix      VVIX / VIX: vol-of-vol relative to the level.  Positive ->
                     log; ``vix`` stem -> no diurnal division.
  vix_chg_1bar       log VIX minus the last OBSERVED log VIX (the index prints
                     on ~19 of the 48 bars); vix_chg_1d / vix_chg_5d: log VIX
                     minus log VIX at the same clock 1 / 5 grid days earlier
                     (a shift within the clock, not a row count: the grid drops
                     the closed hours around weekends, so 48 rows back is not
                     the same clock on a Monday).  Signed -> identity; no
                     diurnal division.
  vix_vrp_1d/5d/22d  log(impl30_perbar_rv / trailing mean of the panel's RV
                     over 48 / 240 / 1056 rows, current row included): implied
                     relative to realized -- the trade's own structure.  Signed
                     -> identity; no diurnal division.  The executor's ladder
                     shifts every exog by one bar, so the current row is not a
                     peek.
  ivslice_over_vix   0DTE ATM-slice annualized vol (data/spxw_ivslice.parquet)
                     over VIX: the 0-day / 30-day slope.  Positive -> log; no
                     diurnal division.  2020-01 on, RTH clocks only.

Coverage follows the feeds: VIX from 2006-05, VIX3M from 2009-08, VVIX from
2012-03, all ending 2024-02-12 (the panel's last 2.5 months carry the neutral
fill, as for the incumbent).  Sanity check printed, not gated: the
log-correlation of each column at the 15:30 row with the 15:30-16:00 realized
variance, 2020-01 .. 2024-04.

  python experiments/build_vix_representations.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.transforms.target import (  # noqa: E402
    apply_semantic_transform,
    is_diurnal_excluded,
)

VIX = ROOT / "data" / "vix_and_voldemand.parquet"
CORE = ROOT / "data" / "core_stats.parquet"
IVSLICE = ROOT / "data" / "spxw_ivslice.parquet"
OUT = ROOT / "data" / "vix_representations.parquet"

BARS_PER_DAY = 48
SESSIONS_PER_YEAR = 252
CHG_DAYS = {"vix_chg_1d": 1, "vix_chg_5d": 5}  # grid days back, same clock
VRP_ROWS = {
    "vix_vrp_1d": BARS_PER_DAY,
    "vix_vrp_5d": 5 * BARS_PER_DAY,
    "vix_vrp_22d": 22 * BARS_PER_DAY,
}
FEATURES = [
    "impl30_perbar_rv",
    "vix_slope_3m",
    "vvix_over_vix",
    "vix_chg_1bar",
    "vix_chg_1d",
    "vix_chg_5d",
    "vix_vrp_1d",
    "vix_vrp_5d",
    "vix_vrp_22d",
    "ivslice_over_vix",
]
# (diurnal division applied?, semantic transform) each name must resolve to.
STEM_RULES = {
    "impl30_perbar_rv": (True, "sqrt"),
    "vix_slope_3m": (False, "identity"),
    "vvix_over_vix": (False, "log"),
    "vix_chg_1bar": (False, "identity"),
    "vix_chg_1d": (False, "identity"),
    "vix_chg_5d": (False, "identity"),
    "vix_vrp_1d": (False, "identity"),
    "vix_vrp_5d": (False, "identity"),
    "vix_vrp_22d": (False, "identity"),
    "ivslice_over_vix": (False, "log"),
}
CHECK_START, CHECK_END = "2020-01-01", "2024-04-30"


def check_stem_rules(d: pd.DataFrame) -> None:
    """Resolve each name through the library's own rules and compare with STEM_RULES."""
    probe = pd.Series([1.0, 4.0, 9.0])
    for col, (want_diurnal, want_tf) in STEM_RULES.items():
        got_diurnal = not is_diurnal_excluded(col)
        has_neg = bool((d[col].dropna() < 0).any())
        out = apply_semantic_transform(probe, col, has_neg).to_numpy()
        if np.allclose(out, np.sqrt(probe)):
            got_tf = "sqrt"
        elif np.allclose(out, np.log(probe)):
            got_tf = "log"
        elif np.allclose(out, probe):
            got_tf = "identity"
        else:
            got_tf = "other"
        ok = (got_diurnal, got_tf) == (want_diurnal, want_tf)
        print(
            f"  {col:18s} diurnal={got_diurnal!s:5s} transform={got_tf:8s} "
            f"signed={has_neg!s:5s} {'ok' if ok else 'MISMATCH'}"
        )
        assert ok, (col, got_diurnal, got_tf)


def build() -> pd.DataFrame:
    v = pd.read_parquet(VIX, columns=["endbartime", "vix", "vvix", "vix3m"])
    c = pd.read_parquet(CORE, columns=["endbartime", "sumret2"])
    s = pd.read_parquet(IVSLICE, columns=["endbartime", "ivslice_annvol"])
    for f in (v, c, s):
        f["endbartime"] = pd.to_datetime(f["endbartime"])
    d = (
        c.merge(v, on="endbartime", how="outer")
        .merge(s, on="endbartime", how="left")
        .sort_values("endbartime")
        .reset_index(drop=True)
    )
    assert len(d) == len(c) == len(v), (len(d), len(c), len(v))
    vix = d["vix"].astype(float)
    lvix = np.log(vix)

    d["impl30_perbar_rv"] = (vix / 100.0) ** 2 / (SESSIONS_PER_YEAR * BARS_PER_DAY)
    d["vix_slope_3m"] = np.log(d["vix3m"].astype(float) / vix)
    d["vvix_over_vix"] = d["vvix"].astype(float) / vix
    # last observed print strictly before this row, only where this row prints
    d["vix_chg_1bar"] = (lvix - lvix.ffill().shift(1)).where(lvix.notna())
    hhmm = d["endbartime"].dt.strftime("%H:%M")
    for col, days in CHG_DAYS.items():
        d[col] = lvix - lvix.groupby(hhmm).shift(days)
    rv = d["sumret2"].astype(float)
    for col, rows in VRP_ROWS.items():
        trail = rv.rolling(rows, min_periods=rows // 2).mean()
        d[col] = np.log(d["impl30_perbar_rv"] / trail)
    d["ivslice_over_vix"] = d["ivslice_annvol"].astype(float) / vix
    return d


def report(d: pd.DataFrame) -> None:
    print("coverage")
    for col in FEATURES:
        ok = d[col].notna()
        print(
            f"  {col:18s} valid {int(ok.sum()):7,d}  "
            f"{d.loc[ok, 'endbartime'].min()} .. {d.loc[ok, 'endbartime'].max()}"
        )
    hhmm = d["endbartime"].dt.strftime("%H:%M")
    day = d["endbartime"].dt.normalize()
    win = (d["endbartime"] >= CHECK_START) & (d["endbartime"] <= f"{CHECK_END} 23:59")
    at = d[win & (hhmm == "15:30")].set_index(day[win & (hhmm == "15:30")])
    y = d[win & (hhmm == "16:00")].set_index(day[win & (hhmm == "16:00")])["sumret2"]
    y = np.log(y.where(y > 0)).reindex(at.index)
    print(
        f"log-correlation at the 15:30 row with the 15:30-16:00 realized variance, "
        f"{CHECK_START} .. {CHECK_END} (VIX prints end 2024-02-12)"
    )
    for col in ["vix", *FEATURES]:
        x = at[col].astype(float)
        want, tf = STEM_RULES.get(col, (False, "log"))
        z = x if tf == "identity" else np.log(x.where(x > 0))
        m = np.isfinite(z) & np.isfinite(y)
        r = float(np.corrcoef(z[m], y[m])[0, 1]) if m.sum() > 2 else float("nan")
        print(f"  {col:18s} corr {r:+.3f}  on {int(m.sum())} days")


def main() -> None:
    d = build()
    print("stem rules (resolved through src/features/transforms/target.py)")
    check_stem_rules(d)
    report(d)
    out = d[["endbartime", *FEATURES]].reset_index(drop=True)
    out.to_parquet(OUT, index=False)
    print(f"wrote {OUT}: {len(out):,} rows, {len(FEATURES)} columns")


if __name__ == "__main__":
    main()
