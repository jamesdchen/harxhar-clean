"""Expiration / rebalance calendar features (§15.3 item 6).

The row -> date map that this needs was the blocker: the study's §0 established that the panel's
242,934 rows align exactly with the cached cluster matrix, so a bar can finally be placed on a
named calendar day rather than an anonymous index.

Why these dates and not others. Realized volatility in the S&P complex has a mechanical,
non-informational component on a small set of scheduled days, and it is scheduled *flow*, not
scheduled *news* — so it is a regressor the HAR lags cannot see coming:

* **Monthly OPEX** — the third Friday. Equity and ETF options expire at that day's close; SPX
  monthlies are AM-settled off the *opening* print. Delta-hedge unwind concentrates in the opening
  and closing auctions, which is why the features below are crossed with the existing session-edge
  gates rather than left as day-level dummies.
* **Quad witching** — the third Friday of March / June / September / December, when index futures,
  index options, stock options and single-stock futures all expire together. Strictly a subset of
  OPEX, kept separate because the flow is a different order of magnitude.
* **Index rebalance** — S&P quarterly share/float updates are effective at the open following the
  third Friday of the quarter-end month, and index trackers execute in that Friday's closing
  auction. Same day as quad witching by construction; the separate flag exists so the *close* lobe
  can carry a different coefficient from the expiry itself.
* **Month / quarter end** — the last trading day, when mandate-driven rebalancing and NAV-striking
  flow lands in the closing auction.

Everything here is a deterministic function of the timestamp. Nothing is fitted, nothing is chosen
by looking at a score, and there is no look-ahead: the third Friday of a month is known years in
advance, and month-end is resolved from the *observed* trading calendar of that month, which is why
the only care needed is that ``is_month_end`` marks the last *observed* bar-day of the month rather
than the calendar's last day (a Saturday month-end would otherwise mark nothing).

The features are all bounded (binary, or a trading-day count clipped to +-``RAMP_DAYS``), so they
pass through ``rolling_robust_scale`` on the zero-IQR branch (divisor 1.0) with no scale risk, and
they are *not* HAR-averaged — they are calendar state at the bar, like ``is_open``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Horizon of the signed "trading days to the next monthly expiry" ramp. Two weeks: the hedging
# literature's window for expiry-related pinning, and short enough that the ramp is not just a
# duplicate of the month-of-year.
RAMP_DAYS: int = 10
QUARTER_MONTHS: tuple[int, ...] = (3, 6, 9, 12)

EXPIRY_FEATURES: list[str] = [
    "is_opex",
    "is_opex_week",
    "is_quad_witch",
    "is_rebalance_close",
    "is_month_end",
    "is_quarter_end",
    "days_to_opex",
]


def _third_fridays(days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The third Friday of every month spanned by ``days``.

    Computed from the calendar, not from the observed data: if the third Friday is a holiday
    (Good Friday can land on it) the expiry moves to the preceding Thursday, so the returned date
    is snapped to the last observed trading day at or before it.
    """
    months = pd.PeriodIndex(days, freq="M").unique()
    out = []
    observed = pd.DatetimeIndex(sorted(set(days)))
    for m in months:
        start = m.start_time
        # weekday: Mon=0 .. Fri=4; first Friday, then two weeks on
        first_friday = start + pd.Timedelta(days=(4 - start.dayofweek) % 7)
        third_friday = first_friday + pd.Timedelta(days=14)
        pos = observed.searchsorted(third_friday, side="right") - 1
        if pos >= 0 and observed[pos].to_period("M") == m:
            out.append(observed[pos])
    return pd.DatetimeIndex(out)


def add_expiry_features(df: pd.DataFrame) -> list[str]:
    """Add the expiration / rebalance calendar columns. Returns the new column names.

    Requires a datetime ``t``. Idempotent: re-running overwrites the same columns.
    """
    t = pd.DatetimeIndex(df["t"])
    day = t.normalize()
    observed_days = pd.DatetimeIndex(sorted(set(day)))
    opex = _third_fridays(observed_days)

    is_opex = day.isin(opex)
    # the trading week containing the expiry (Mon-Fri of that ISO week)
    opex_weeks = {(d.isocalendar()[0], d.isocalendar()[1]) for d in opex}
    iso = [(d.isocalendar()[0], d.isocalendar()[1]) for d in day]
    is_opex_week = np.fromiter((k in opex_weeks for k in iso), dtype=bool, count=len(day))
    is_quad = is_opex & np.isin(day.month, QUARTER_MONTHS)

    # Last observed trading day of each calendar month / quarter. A period's last observed day only
    # counts as a period end if a *later* period is also observed — otherwise the final day of the
    # sample is labelled a month end and a quarter end whatever date it happens to fall on, which is
    # a mislabel rather than a look-ahead but still wrong (2024-04-30 is neither).
    dser = pd.Series(observed_days, index=observed_days)
    month_end_days: set[pd.Timestamp] = set()
    quarter_end_days: set[pd.Timestamp] = set()
    for freq, sink in (("M", month_end_days), ("Q", quarter_end_days)):
        last = dser.groupby(dser.dt.to_period(freq)).max().sort_index()
        sink.update(last.iloc[:-1])

    # signed trading-day distance to the nearest monthly expiry, clipped to +-RAMP_DAYS.
    # Negative = expiry is ahead, positive = it has passed; measured in *trading* days so a
    # holiday does not silently stretch the ramp.
    pos_of = {d: i for i, d in enumerate(observed_days)}
    opex_pos = np.array([pos_of[d] for d in opex], dtype=np.int64)
    day_pos = np.fromiter((pos_of[d] for d in day), dtype=np.int64, count=len(day))
    nearest = np.searchsorted(opex_pos, day_pos)
    lo = np.clip(nearest - 1, 0, len(opex_pos) - 1)
    hi = np.clip(nearest, 0, len(opex_pos) - 1)
    dlo = day_pos - opex_pos[lo]
    dhi = day_pos - opex_pos[hi]
    signed = np.where(np.abs(dlo) <= np.abs(dhi), dlo, dhi)

    df["is_opex"] = is_opex.astype(np.int8)
    df["is_opex_week"] = is_opex_week.astype(np.int8)
    df["is_quad_witch"] = is_quad.astype(np.int8)
    # The index-tracker rebalance trade prints in the quad-witch *closing auction*, so this is the
    # one interaction built in here rather than left to the model: a day-level quad-witch dummy
    # would average the auction bar together with 47 ordinary ones.
    close_gate = (
        df["is_close"].to_numpy()
        if "is_close" in df.columns
        else ((t.hour >= 16) & (t.hour <= 19)).astype(np.int8)
    )
    df["is_rebalance_close"] = (is_quad.astype(np.int8) * close_gate).astype(np.int8)
    df["is_month_end"] = day.isin(month_end_days).astype(np.int8)
    df["is_quarter_end"] = day.isin(quarter_end_days).astype(np.int8)
    df["days_to_opex"] = np.clip(signed, -RAMP_DAYS, RAMP_DAYS).astype(np.float64)
    return list(EXPIRY_FEATURES)
