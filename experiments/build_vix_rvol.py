"""Realized vol-of-VIX from the VIX's own path, as panel columns.

The question (2026-09-23): does the VIX bucket need VVIX and VIX3M, and can the
VIX's own path stand in for them?  On the 15:30 stamps with all three prints
(2,928 sessions 2012-2024) the two indices add +0.0064 of R^2 to the next bar's
log RV beside log VIX and the last bar's log RV; log VIX3M is 95 % the VIX level
(a mean-reversion map), log VVIX only 48 %, and the realized vol-of-VIX built
here recovers about half of the pair's increment (+0.0033).  This module writes
that proxy so the per-bar arms can test it:

  vix_volofvol_5d   rolling std of the 30-minute log change of VIX over the last
                    65 OBSERVED RTH prints (10:00 .. 16:00 stamps; ~5 sessions),
                    min_periods 40
  vix_volofvol_22d  the same over 286 observed prints (~22 sessions), min_periods 200

(The names avoid the "rv" stem -- "vix_rvol" would be square-rooted by the
library's name rules -- and carry only "vix".)

Only observed prints enter the change: a missing or delayed print never creates
a spurious 30-minute change (the free feed's Cboe prints are 15-minute delayed).
The columns sit on the grid at the RTH stamps and are NaN elsewhere (the
executor forward-fills exogs).  Names carry the ``vix`` stem -> exempt from the
diurnal division, and are positive -> the default log transform (verified below
through the library's own rules).

Gate: the +0.0033 R^2 increment of the pair beside log VIX and the last bar's
log RV on the 15:30 stamps with all three prints must reproduce.

  python experiments/build_vix_rvol.py
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
OUT = ROOT / "data" / "vix_rvol.parquet"

RTH_STAMPS = [
    f"{h:02d}:{m:02d}" for h in range(10, 17) for m in (0, 30) if (h, m) <= (16, 0)
]
# Not "vix_rvol_*": that spelling carries the "rv" stem and the library would
# take the square root; "volofvol" carries only the "vix" stem -> log, no diurnal.
WINDOWS = {
    "vix_volofvol_5d": (65, 40),
    "vix_volofvol_22d": (286, 200),
}  # prints, min_periods
FEATURES = list(WINDOWS)
STEM_RULES = {c: (False, "log") for c in FEATURES}  # (diurnal applied?, transform)
GATE_START, GATE_END = "2012-04-01", "2024-02-12"
GATE_DR2, GATE_TOL = 0.0033, 0.0010


def build() -> pd.DataFrame:
    v = pd.read_parquet(VIX, columns=["endbartime", "vix"])
    v["endbartime"] = pd.to_datetime(v["endbartime"])
    v = v.sort_values("endbartime").reset_index(drop=True)
    rth = v[
        v["endbartime"].dt.strftime("%H:%M").isin(RTH_STAMPS) & v["vix"].notna()
    ].copy()
    dlv = np.log(rth["vix"]).diff()  # consecutive OBSERVED RTH prints only
    for col, (win, minp) in WINDOWS.items():
        rth[col] = dlv.rolling(win, min_periods=minp).std()
    out = v[["endbartime"]].merge(
        rth[["endbartime", *FEATURES]], on="endbartime", how="left"
    )
    return out


def check_stem_rules(d: pd.DataFrame) -> None:
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
            f"  {col:14s} diurnal={got_diurnal!s:5s} transform={got_tf:8s} "
            f"signed={has_neg!s:5s} {'ok' if ok else 'MISMATCH'}"
        )
        assert ok, (col, got_diurnal, got_tf)


def _ols_r2(y: np.ndarray, x: np.ndarray) -> float:
    a = np.column_stack([np.ones(len(x)), x])
    b, *_ = np.linalg.lstsq(a, y, rcond=None)
    e = y - a @ b
    return float(1.0 - e.var() / y.var())


def gate(d: pd.DataFrame) -> None:
    """The pair's R^2 increment for the next bar's log RV at 15:30 reproduces the quick check."""
    v = pd.read_parquet(VIX, columns=["endbartime", "vix", "vvix", "vix3m"])
    c = pd.read_parquet(CORE, columns=["endbartime", "sumret2"])
    for f in (v, c):
        f["endbartime"] = pd.to_datetime(f["endbartime"])
    g = (
        v.merge(c, on="endbartime")
        .merge(d, on="endbartime")
        .set_index("endbartime")
        .sort_index()
    )
    g = g[(g.index >= GATE_START) & (g.index <= GATE_END)]
    hh = g.index.strftime("%H:%M")
    x = g[hh == "15:30"].copy()
    x.index = x.index.normalize()
    y = g[hh == "16:00"]["sumret2"].copy()
    y.index = y.index.normalize()
    x = x.join(y.rename("rv_next"), how="inner").dropna(
        subset=["vix", "vvix", "vix3m", "rv_next", "sumret2", *FEATURES]
    )
    x = x[(x[["vix", "rv_next", "sumret2", *FEATURES]] > 0).all(axis=1)]
    lv = np.log(x[["vix", "sumret2", *FEATURES]].to_numpy(float))
    ly = np.log(x["rv_next"].to_numpy(float))
    r2_base = _ols_r2(ly, lv[:, :2])
    r2_pair = _ols_r2(ly, lv)
    dr2 = r2_pair - r2_base
    print(
        f"GATE  15:30 stamps with all three prints {GATE_START}..{GATE_END}: n {len(x)}; next-bar log RV | "
        f"log VIX + last bar's log RV R2 {r2_base:.4f}; + realized vol-of-VIX (5d, 22d) R2 {r2_pair:.4f}, "
        f"dR2 {dr2:+.4f} (quick check {GATE_DR2:+.4f})"
    )
    assert abs(dr2 - GATE_DR2) < GATE_TOL, dr2


def report(d: pd.DataFrame) -> None:
    for col in FEATURES:
        ok = d[col].notna()
        print(
            f"  {col:14s} valid {int(ok.sum()):7d}  "
            f"{d.loc[ok, 'endbartime'].min()} .. {d.loc[ok, 'endbartime'].max()}  "
            f"median {d.loc[ok, col].median():.4f}"
        )


def main() -> None:
    d = build()
    print("stem rules through the library:")
    check_stem_rules(d)
    print("coverage:")
    report(d)
    gate(d)
    d.to_parquet(OUT, index=False)
    print(f"wrote {OUT}: {len(d):,} grid rows, {len(FEATURES)} columns")


if __name__ == "__main__":
    main()
