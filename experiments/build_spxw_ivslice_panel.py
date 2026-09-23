"""The ATM implied slice -- the deck's own object -- as panel columns at every clock.

At every clock 10:00 .. 15:30 of every session the nearest-OTM SPXW 0DTE
straddle is re-picked with the research's guards and its package midpoint is
bisected for the Black-76 total volatility over the remaining session:
proposal 43's ``build_chain`` (= live.ibkr.pricing.invert_total_vol, vectorised,
GATE V on every cell).  Three columns on the ``endbartime`` grid:

  ivslice_perbar_rv  tot^2 / remaining bars: the package's implied variance per
                     remaining 30-minute bar, in the target's units (sqrt +
                     diurnal division downstream, the ``rv`` stem)
  ivslice_annvol     100 * sqrt(tot^2 / T): annualized, VIX units.  Not named
                     with the ``vix`` stem: like the strip, it climbs
                     mechanically through the day, so it takes the default log
                     + diurnal division rather than the 30-day index's exemption
  ivslice_vvix       trailing-12-stamp std of d log(ivslice_annvol): realized
                     vol-of-implied (flat across the clock; the exemption fits)

Why this and not the strip (build_spxw_mfiv_panel.py): at 15:30 the model-free
strip is 9x the ATM slice and 85% of it is zero-bid strikes at the minimum
tick; its log-correlation with the next bar's realized variance is 0.43
against the slice's 0.92.  The slice is the quantity the 15:30 trade is
measured against (the deck's ``iv_var``, there from the vendor's per-leg
implieds); here it is re-inverted from the package midpoint, so it exists at
every clock and is never bisection-censored.

Gates and reports at 15:30 on the deck days: the package midpoint must equal
the deck's ``entry`` (exact -- the same pick on the same quotes; a difference
means the chain on disk is not the deck's); tot^2 vs the deck's ``iv_var``
(ratio and log-correlation, reported: two conventions of one quantity); the
log-correlation with the 15:30-16:00 realized variance beside the deck's.

Half sessions are excluded whole (proposal 43's rule: hours_to_expiration <= 0
at the 15:30 stamp).  The chain is read in yearly slices in ONE process; the
16M-row file is never loaded whole.  Panel clock: naive ET, bar-END labelled;
the 15:30 quote sits on the 15:30 row and reaches the 16:00 design row through
the executor's shift(1).  Rows after the feature panel's last stamp are not
written; the full series goes to the diagnostics file.

  python experiments/build_spxw_ivslice_panel.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "notebooks", ROOT / "experiments"):
    sys.path.insert(0, str(p))

P43 = ROOT / "writeup" / "intraday_proposals" / "43_causal_entry_over_time.py"
DECK = ROOT / "results" / "atm_straddle_0dte_1530" / "daily_blk2.parquet"
PANEL_END_SOURCE = ROOT / "data" / "core_stats.parquet"
OUT_PANEL = ROOT / "data" / "spxw_ivslice.parquet"
OUT_DIAG = ROOT / "results" / "spxw_pnl" / "ivslice_panel_diag.parquet"
FEATURES = ["ivslice_annvol", "ivslice_perbar_rv", "ivslice_vvix"]
VOV_WINDOW = 12
HOURS_PER_YEAR = 365.25 * 24.0
GATE_ENTRY_TOL = 1e-6


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def build(p43: ModuleType) -> pd.DataFrame:
    stamp, _ = p43.session_stamps()
    half = p43.half_sessions(stamp)
    clocks = list(p43.CLOCKS)
    sessions = pd.DatetimeIndex(stamp.index).difference(half)
    have = stamp.reindex(sessions)[clocks].notna().all(axis=1).to_numpy()
    print(
        f"sessions {len(stamp)}: half sessions excluded {len(half)}, "
        f"missing a clock stamp {int((~have).sum())}, built {int(have.sum())}"
    )
    sessions = sessions[have]
    n_rem = np.array(
        [(16 * 60 - (int(c[:2]) * 60 + int(c[3:]))) // 30 for c in clocks], float
    )
    frames = []
    for yr, dates in pd.Series(sessions, index=sessions).groupby(sessions.year):
        idx = pd.DatetimeIndex(dates)
        ch = p43.build_chain(stamp, idx)
        n, k = len(idx), len(clocks)
        frames.append(
            pd.DataFrame(
                {
                    "date": np.repeat(idx.to_numpy(), k),
                    "hhmm": np.tile(np.array(clocks), n),
                    "n_rem": np.tile(n_rem, n),
                    "S": ch["S"].ravel(),
                    "K_c": ch["K_c"].ravel(),
                    "K_p": ch["K_p"].ravel(),
                    "entry": ch["entry"].ravel(),
                    "tot": ch["tot"].ravel(),
                }
            )
        )
        print(f"  {yr}: {n} sessions, refused cells {len(ch['refused'])}", flush=True)
    d = pd.concat(frames, ignore_index=True)
    d["endbartime"] = pd.to_datetime(d["date"]) + pd.to_timedelta(
        d["hhmm"].str[:2].astype(int) * 60 + d["hhmm"].str[3:].astype(int), unit="m"
    )
    d = d.sort_values("endbartime").reset_index(drop=True)
    var_tot = d["tot"].to_numpy(float) ** 2
    d["ivslice_perbar_rv"] = var_tot / d["n_rem"].to_numpy(float)
    d["ivslice_annvol"] = 100.0 * np.sqrt(
        var_tot / (d["n_rem"].to_numpy(float) * 0.5 / HOURS_PER_YEAR)
    )
    d["ivslice_vvix"] = (
        np.log(d["ivslice_annvol"])
        .diff()
        .rolling(VOV_WINDOW, min_periods=VOV_WINDOW // 2)
        .std()
    )
    return d


def gates(d: pd.DataFrame) -> None:
    deck = pd.read_parquet(DECK).sort_index()
    di = pd.DatetimeIndex(pd.to_datetime(deck.index)).normalize()
    at = d[d["hhmm"] == "15:30"].set_index(
        pd.DatetimeIndex(d.loc[d["hhmm"] == "15:30", "date"]).normalize()
    )
    missing = di.difference(at.index)
    at = at.reindex(di)
    dev = np.abs(at["entry"].to_numpy(float) - deck["entry"].to_numpy(float))
    n_diff = int(np.sum(dev > GATE_ENTRY_TOL))
    print(
        f"GATE  15:30 package midpoint vs the deck's entry on {len(di)} deck days: "
        f"missing {len(missing)}, differing (> {GATE_ENTRY_TOL:g}) {n_diff}, max |d| {np.nanmax(dev):.2e}"
    )
    if len(missing) or n_diff:
        print(
            "      the chain on disk is not the chain the deck was built from at 15:30 -- reported, not fatal"
        )
    ratio = at["ivslice_perbar_rv"] / deck["iv_var"].to_numpy(float)
    ok = np.isfinite(ratio) & (ratio > 0)
    lc = np.corrcoef(
        np.log(at["ivslice_perbar_rv"][ok]), np.log(deck["iv_var"].to_numpy(float)[ok])
    )[0, 1]
    print(
        f"REPORT tot^2 at 15:30 vs the deck's vendor-IV iv_var: median ratio {ratio[ok].median():.3f} "
        f"(IQR {ratio[ok].quantile(0.25):.3f}-{ratio[ok].quantile(0.75):.3f}), log-corr {lc:.3f} on {int(ok.sum())} days"
    )
    rv = pd.read_parquet(PANEL_END_SOURCE, columns=["endbartime", "sumret2"])
    rv["endbartime"] = pd.to_datetime(rv["endbartime"])
    y = rv[rv["endbartime"].dt.strftime("%H:%M") == "16:00"].set_index(
        rv["endbartime"].dt.normalize()[
            rv["endbartime"].dt.strftime("%H:%M") == "16:00"
        ]
    )["sumret2"]
    y = y.reindex(di)
    for name, x in (
        ("ivslice_perbar_rv (this)", at["ivslice_perbar_rv"]),
        (
            "deck iv_var (vendor IV)",
            pd.Series(deck["iv_var"].to_numpy(float), index=di),
        ),
    ):
        m = (
            np.isfinite(x.to_numpy(float))
            & np.isfinite(y.to_numpy(float))
            & (x.to_numpy(float) > 0)
            & (y.to_numpy(float) > 0)
        )
        c = np.corrcoef(np.log(x.to_numpy(float)[m]), np.log(y.to_numpy(float)[m]))[
            0, 1
        ]
        print(
            f"REPORT log-corr with the 15:30-16:00 realized variance: {name} {c:.3f} on {int(m.sum())} days"
        )


def main() -> None:
    p43 = _load(P43, "p43_causal_entry")
    d = build(p43)
    gates(d)
    by_clock = d.groupby("hhmm")[
        ["ivslice_annvol", "ivslice_perbar_rv", "ivslice_vvix", "n_rem"]
    ].median()
    print(by_clock.to_string(float_format=lambda v: f"{v:.6g}"))
    print(
        f"stamps {len(d):,} on {d['date'].nunique():,} sessions {d['endbartime'].min()} .. {d['endbartime'].max()}; "
        f"no-quote cells {int(d['tot'].isna().sum())}"
    )
    OUT_DIAG.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(OUT_DIAG, index=False)
    panel_end = pd.to_datetime(
        pd.read_parquet(PANEL_END_SOURCE, columns=["endbartime"])["endbartime"]
    ).max()
    p = d.loc[d["endbartime"] <= panel_end, ["endbartime", *FEATURES]].reset_index(
        drop=True
    )
    p.to_parquet(OUT_PANEL, index=False)
    print(
        f"wrote {OUT_PANEL}: {len(p):,} rows, last {p['endbartime'].max()} (panel ends {panel_end}); diagnostics {OUT_DIAG}"
    )


if __name__ == "__main__":
    main()
