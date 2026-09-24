"""The panel's 30-minute rows from 1-minute bars.

THE DEFINITIONS BELOW ARE THIS PACKAGE'S, NOT THE VENDOR'S.  The repository
does not carry the builder of data/core_stats.parquet (the columns arrived
with the panel), so the moments are written down here from their names and
must be GATED against the vendor panel on the overlap the purchased ES history
provides (April 2024: the vendor's last month) before a live forecast is
trusted -- ``assert_parity`` is that gate.  Until it has run, every number
downstream of these rows is provisional.

For a 30-minute bar ending at ``tau`` with 1-minute closes ``p_0 .. p_n``
(``p_0`` the last close before the bar, ``p_1 .. p_n`` the closes of the
minutes starting in ``[tau - 30 min, tau - 1 min]``) and log returns
``r_i = ln(p_i / p_{i-1})``:

    sumret      sum r_i                       signed net return
    sumabsret   sum |r_i|
    sumret2     sum r_i^2                     realized variance (the target)
    sumret3     sum r_i^3
    sumret4     sum r_i^4
    sumpret2    sum r_i^2 1{r_i > 0}          positive semivariance (ASSUMED reading of "pret2")
    sumbipow    sum_{i>=2} |r_i| |r_{i-1}|    bipower variation, unnormalised
    sumautocov  sum_{i>=2} r_i r_{i-1}        first-order autocovariance sum
    sumvolume   sum volume_i
    numobs      n                             1-minute returns in the bar

Cboe prints at a stamp: the last 1-minute close whose minute starts at or
before ``tau - 1 min`` inside the bar (the print standing at the bar's end).
The vendor's convention is not documented either; same gate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from live.close_signal.common import (
    BAR,
    CBOE_COLS,
    CORE_COLS,
    ONE_MINUTE,
    bar_end_of,
    session_grid,
)

#: Moments that the free ^GSPC substitute cannot supply (index volume is not
#: futures volume; the minute count is not the tick count the vendor meant).
SUBSTITUTE_UNAVAILABLE: tuple[str, ...] = ("sumvolume", "numobs")
#: Consecutive 1-minute bars this far apart or more sit on either side of a
#: session break -- the CME maintenance hour (17:00-18:00 ET) or the weekend --
#: and no return is taken across them.  Overnight no-trade gaps in ES run to a
#: few minutes, never an hour.
SESSION_BREAK = pd.Timedelta(minutes=60)


def thirty_minute_moments(bars_1m: pd.DataFrame) -> pd.DataFrame:
    """CORE_COLS per 30-minute bar END from 1-minute bars indexed by bar START.

    Requires ``close`` (and ``volume``; missing volume gives NaN sumvolume).
    The first bar of the span has no ``p_0`` and is dropped, and so is the
    first minute after a SESSION_BREAK: the vendor panel starts each session
    fresh (its Sunday 18:30 bar carries 29 returns), and a return spanning the
    break put the whole gap's move into one squared term (vendor gate
    2026-09-23: the 18:30 stamps 2-8x the vendor's before this rule, in line
    with the neighbouring clocks after).  Bars with no minutes (closed periods)
    are simply absent.
    """
    if bars_1m.empty:
        return pd.DataFrame(columns=["endbartime", *CORE_COLS])
    f = bars_1m.sort_index()
    close = f["close"].astype(float)
    ret = np.log(close).diff()  # r_i uses the previous minute's close, across bar edges
    idx = pd.DatetimeIndex(f.index)
    gap = pd.Series(idx, index=idx).diff()
    ret = ret.mask((gap >= SESSION_BREAK).to_numpy())
    end = bar_end_of(idx)
    g = pd.DataFrame(
        {
            "endbartime": end,
            "r": ret.to_numpy(),
            "r_prev": ret.shift(1).to_numpy(),
            "volume": f["volume"].astype(float).to_numpy() if "volume" in f else np.nan,
        }
    )
    g = g.dropna(subset=["r"])
    # the lag-1 products need both returns inside the same bar
    same_bar = g["endbartime"].to_numpy() == g["endbartime"].shift(1).to_numpy()
    rp = np.where(same_bar, g["r_prev"].to_numpy(), np.nan)
    r = g["r"].to_numpy()
    g["absret"] = np.abs(r)
    g["ret2"] = r**2
    g["ret3"] = r**3
    g["ret4"] = r**4
    g["pret2"] = np.where(r > 0, r**2, 0.0)
    g["bipow"] = np.where(np.isfinite(rp), np.abs(r) * np.abs(rp), 0.0)
    g["autocov"] = np.where(np.isfinite(rp), r * rp, 0.0)
    agg = g.groupby("endbartime").agg(
        sumret=("r", "sum"),
        sumabsret=("absret", "sum"),
        sumret2=("ret2", "sum"),
        sumret3=("ret3", "sum"),
        sumret4=("ret4", "sum"),
        sumpret2=("pret2", "sum"),
        sumbipow=("bipow", "sum"),
        sumautocov=("autocov", "sum"),
        sumvolume=("volume", "sum"),
        numobs=("r", "size"),
    )
    agg["numobs"] = agg["numobs"].astype(float)
    return agg.reset_index()[["endbartime", *CORE_COLS]]


def cboe_stamp_values(
    prints: dict[str, pd.DataFrame], stamps: pd.DatetimeIndex
) -> pd.DataFrame:
    """The print standing at each stamp's end, per column of ``CBOE_COLS``.

    ``prints[col]`` is a 1-minute frame indexed by bar START with ``close``.
    A stamp with no print inside its bar is NaN (no forward fill here: the
    executor's own ffill does that, and only within the panel's rules).
    """
    out = pd.DataFrame({"endbartime": pd.DatetimeIndex(stamps)})
    for col in CBOE_COLS:
        fr = prints.get(col)
        vals = np.full(len(stamps), np.nan)
        if fr is not None and not fr.empty:
            s = fr["close"].astype(float).sort_index()
            starts = pd.DatetimeIndex(s.index)
            for i, tau in enumerate(stamps):
                lo, hi = tau - BAR, tau - ONE_MINUTE
                m = (starts >= lo) & (starts <= hi)
                if m.any():
                    vals[i] = float(s[m].iloc[-1])
        out[col] = vals
    return out


def panel_rows(
    es_bars_1m: pd.DataFrame,
    cboe_prints_1m: dict[str, pd.DataFrame],
    session: pd.Timestamp,
) -> pd.DataFrame:
    """The session's 48 rows: ES moments + Cboe prints on the panel grid."""
    grid = session_grid(session)
    mom = thirty_minute_moments(es_bars_1m).set_index("endbartime").reindex(grid)
    cb = cboe_stamp_values(cboe_prints_1m, grid).set_index("endbartime")
    rows = (
        pd.concat([mom, cb], axis=1)
        .reset_index()
        .rename(columns={"index": "endbartime"})
    )
    return rows


def substitute_last_bar(
    rows: pd.DataFrame, spx_bars_1m: pd.DataFrame, stamp_end: pd.Timestamp
) -> pd.DataFrame:
    """free_substitute mode: the ^GSPC moments replace the missing ES bar at ``stamp_end``.

    Only the return moments are substituted; ``SUBSTITUTE_UNAVAILABLE`` stay
    NaN for that stamp.  A MODEL CHANGE relative to the research construction;
    the run journals it and the README says what must be validated first.
    """
    sub = thirty_minute_moments(spx_bars_1m).set_index("endbartime")
    if stamp_end not in sub.index:
        raise ValueError(f"^GSPC has no complete bar ending {stamp_end}")
    out = rows.set_index("endbartime")
    for c in CORE_COLS:
        out.loc[stamp_end, c] = (
            np.nan if c in SUBSTITUTE_UNAVAILABLE else sub.loc[stamp_end, c]
        )
    return out.reset_index()


def assert_parity(
    ours: pd.DataFrame,
    vendor: pd.DataFrame,
    *,
    rel_tol: float,
    cols: tuple[str, ...] = CORE_COLS,
    min_rows: int = 100,
) -> pd.DataFrame:
    """Gate our construction against the vendor panel on the overlap.

    Both frames carry ``endbartime`` + ``cols``.  Per column the relative
    error is ``|a - b| / max(|a|, |b|)`` on rows where the vendor is finite
    and non-zero; the gate fails if the 99th percentile exceeds ``rel_tol``
    or fewer than ``min_rows`` rows overlap.  Returns the per-column report
    (rows, median, p99, max) and raises ``AssertionError`` with the worst
    rows on failure -- a failed gate is a finding, not a fill.
    """
    j = ours.set_index("endbartime")[list(cols)].join(
        vendor.set_index("endbartime")[list(cols)], how="inner", rsuffix="_vendor"
    )
    if len(j) < min_rows:
        raise AssertionError(f"parity: only {len(j)} overlapping rows (< {min_rows})")
    rep = []
    bad: list[str] = []
    for c in cols:
        a = j[c].to_numpy(float)
        b = j[f"{c}_vendor"].to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(b) & (b != 0)
        rel = np.abs(a[ok] - b[ok]) / np.maximum(np.abs(a[ok]), np.abs(b[ok]))
        p99 = float(np.percentile(rel, 99)) if ok.any() else float("nan")
        rep.append(
            {
                "column": c,
                "rows": int(ok.sum()),
                "median_rel": float(np.median(rel)) if ok.any() else float("nan"),
                "p99_rel": p99,
                "max_rel": float(rel.max()) if ok.any() else float("nan"),
            }
        )
        if ok.any() and p99 > rel_tol:
            bad.append(f"{c}: p99 rel {p99:.3g} > {rel_tol:g}")
    report = pd.DataFrame(rep)
    if bad:
        raise AssertionError(
            "parity gate failed: " + "; ".join(bad) + "\n" + report.to_string()
        )
    return report


__all__ = [
    "SUBSTITUTE_UNAVAILABLE",
    "assert_parity",
    "cboe_stamp_values",
    "panel_rows",
    "substitute_last_bar",
    "thirty_minute_moments",
]
