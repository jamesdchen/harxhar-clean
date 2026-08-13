"""Options feature ingestion: the single door for options data into the panel.

The real options feed arrives in one of two shapes:

* **date-keyed** EOD aggregates (OptionMetrics IvyDB: chain-level Greeks, the
  GEX family of ``experiments/gex.py``, volume/OI by moneyness-maturity
  bucket), one row per trade date; or
* **bar-keyed** intraday series (already on the 30-minute grid).

Bar-keyed series enter the panel exactly like every other feed: write them
as an ``endbartime``-keyed parquet in ``data/`` and ``load_raw_data`` merges
them. Date-keyed features are the dangerous case --- an EOD row for trade
date ``d`` is only *published* after that day's close, and attaching it to
bars of session ``d`` is lookahead. That attachment is centralized here so
the publication lag is enforced in code, not left to each caller (the
gex.py docstring's "the CALLER lags it one session" instruction is replaced
by a gate that fails loudly).

Go-live rule (the causal contract)
----------------------------------
Features for trade date ``d`` attach to panel bars with
``endbartime >= go_live(d)``, where ``go_live(d) = (d + 1 calendar day) +
GO_LIVE_TIME``. The default ``GO_LIVE_TIME = "00:00"`` (next midnight,
exchange time) is strictly after any EOD publication; the overnight bars
18:30--24:00 of the next session therefore carry the *prior* date's
features. If the real feed's publication SLA is later, raise the go-live
time (e.g. ``"09:30"`` = next cash open) --- never lower it without a
documented publication-time guarantee.

Downstream contract
-------------------
The output parquet is ``endbartime``-keyed with one column per channel
(``opt_`` prefix). ``load_raw_data`` outer-merges it onto the panel grid;
bars before the feed's first go-live are NaN and receive the standard
availability indicator (0) + neutral fill, so the estimator sees "feed not
live yet" exactly as it does for the late-starting VIX family. Channel
names are registered in ``src/data/loading.py`` (``OPTIONS_FEATURES``),
deliberately outside ``ALL_FEATURES`` until a panel-version bump.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

GO_LIVE_TIME = "00:00"  # next-day boundary, exchange time; see module docstring
CHANNEL_PREFIX = "opt_"
# A daily feature row is legitimate for the session after its go-live, over
# weekends, and over short holidays --- but NOT indefinitely. Without a
# staleness cutoff, merge_asof carries a dead feed's last print forward
# forever (the exact failure the panel's availability discipline documents:
# the vol_demand feed died 2023-08-31 and read as alive for eight months).
# Five calendar days covers a long weekend plus margin; beyond it the channel
# reads NaN, and the downstream availability indicator correctly reports the
# feed as not publishing.
MAX_STALENESS = pd.Timedelta(days=5)


def _go_live_index(dates: pd.DatetimeIndex, go_live_time: str) -> pd.Series:
    """Publication timestamp for each trade date: next calendar day + go-live time."""
    h, m = (int(x) for x in go_live_time.split(":"))
    live = dates.normalize() + pd.Timedelta(days=1, hours=h, minutes=m)
    return pd.Series(live.values, index=dates)


def attach_daily_to_bars(
    feat: pd.DataFrame,
    bar_times: pd.DatetimeIndex,
    go_live_time: str = GO_LIVE_TIME,
    max_staleness: pd.Timedelta = MAX_STALENESS,
) -> pd.DataFrame:
    """Attach date-keyed options features to panel bars under the go-live rule.

    Parameters
    ----------
    feat : DataFrame with a ``date`` column (or DatetimeIndex) of trade dates
        and one column per channel. Dates need not be contiguous; missing
        dates are simply never selected (holiday-safe).
    bar_times : DatetimeIndex of panel bar end-times (``endbartime``).
    go_live_time : "HH:MM" publication time on the day after the trade date.

    Returns
    -------
    DataFrame indexed like ``bar_times`` with the channel columns, plus a
    ``_live_ts`` column recording the publication timestamp of the attached
    row (for the causality gate). Bars before the first go-live are NaN.
    """
    f = feat.copy()
    if "date" in f.columns:
        f["date"] = pd.to_datetime(f["date"])
        f = f.set_index("date")
    f = f.sort_index()
    if not f.index.is_unique:
        raise ValueError("options feature frame has duplicate trade dates")
    channels = list(f.columns)
    live = _go_live_index(f.index, go_live_time)

    right = pd.DataFrame({"_live_ts": live.values, "_src_date": f.index.values})
    for ch in channels:
        right[ch] = f[ch].values
    left = pd.DataFrame({"t": pd.DatetimeIndex(bar_times).values})

    merged = pd.merge_asof(
        left.sort_values("t"),
        right.sort_values("_live_ts"),
        left_on="t",
        right_on="_live_ts",
        direction="backward",
    )
    merged = merged.set_index("t").sort_index()

    # ── Causality gate: the attached row's publication timestamp must not
    # exceed the bar's own timestamp, on every single bar. ──────────────
    ok = merged["_live_ts"].isna() | (merged["_live_ts"] <= merged.index)
    if not bool(ok.all()):
        bad = merged.loc[~ok].index[:3]
        raise AssertionError(
            f"causality violation: {int((~ok).sum())} bars carry features "
            f"published after the bar (first: {list(bad)})"
        )

    # -- Staleness cutoff: a feed that stops publishing stops existing. --
    age = merged.index.to_series().subtract(merged["_live_ts"])
    stale = merged["_live_ts"].notna() & (age > max_staleness)
    if bool(stale.any()):
        merged.loc[stale, channels] = np.nan
    return merged


def coverage_report(merged: pd.DataFrame, channels: list[str]) -> pd.DataFrame:
    """Per-channel print accounting, mirroring the panel's availability gate:
    for each channel, first/last live bar and the fraction of bars carrying a
    value inside its live window (1.0 for a healthy EOD feed mapped daily)."""
    rows = []
    for ch in channels:
        s = merged[ch]
        live = s.notna()
        rows.append(
            dict(
                channel=ch,
                first_live=live.idxmax() if live.any() else pd.NaT,
                last_live=live[::-1].idxmax() if live.any() else pd.NaT,
                live_bars=int(live.sum()),
                live_fraction=float(live.mean()),
            )
        )
    return pd.DataFrame(rows).set_index("channel")


def write_options_parquet(
    merged: pd.DataFrame,
    channels: list[str],
    out_path: str,
) -> str:
    """Write the endbartime-keyed parquet ``load_raw_data`` will merge.
    Drops the gate bookkeeping columns; asserts finite-or-NaN values."""
    out = merged.reset_index()[["t"] + list(channels)].rename(columns={"t": "endbartime"})
    vals = out[list(channels)].to_numpy(dtype="float64", na_value=np.nan)
    if np.isinf(vals).any():
        raise ValueError("options features contain +/-inf")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out.to_parquet(out_path, index=False)
    return out_path


def expected_channels() -> list[str]:
    """Channel names the options block registers, in panel order. Matches the
    validated dry-run output (data/options_features.parquet, 2026-08-12)."""
    return [
        "opt_spot",            # underlying EOD close (spot reference)
        "opt_gex_level",       # net dealer gamma exposure ($ per +1% move)
        "opt_gex_0dte",        # gamma exposure from 0-DTE contracts only
        "opt_gex_zero",        # zero-gamma flip level (index points)
        "opt_gex_flip_dist",   # (S - flip)/S: signed distance to the flip
        "opt_pin_strike",      # max |gamma*OI| strike (sign-free magnet)
        "opt_dist_to_pin",     # (S - pin)/S: signed distance to the magnet
        "opt_vanna_net",       # net vanna (calendar-timed hedging pressure)
        "opt_charm_net",       # net charm
        "opt_n_opt",           # live contracts in the chain (coverage gauge)
        "opt_total_oi",        # total open interest
        "opt_regime_long",     # 1{gex_level > 0}: long-gamma regime flag
    ]
