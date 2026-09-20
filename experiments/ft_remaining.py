"""At-entry (F_t-measurable) remaining-variance forecast.

Convention (verified this session): yhat panels are bar-END labelled —
the row at stamp tau carries the bar [tau-30, tau]: ``rv_raw`` is that
bar's realized variance and ``yhat``/``baseline`` its forecast, issued
at tau-30.  The one-step forecast STANDING at an entry e is therefore
the stamp e+30 row:

    f_next(e) = yhat(e+30)^2 * baseline(e+30).

The remaining-session forecast extends f_next with a CAUSAL diurnal
profile (the professor-notebook section-5b recipe):

    F_rem(e) = f_next(e) / w_next(e),

where w_next(e) is the entry bar's share of remaining-session variance
under the expanding per-clock mean of realized bar variance over PRIOR
days only (min 63 days of that clock, lagged one day); shares are over
the remaining trade clocks e..15:30 ET.  Every input is measurable at
e: the forecast is issued at e, the profile sees days <= d-1.

This replaces the ``pa_rem``/``pb_rem``/``b2_rem`` construction, which
summed per-bar forecasts issued DURING the window and is therefore not
F_t-measurable (the dh book collapses 9.11 -> 0.86 under the strictly
at-entry variant).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Trade bars of the scored session: bar starting 10:00 ET .. bar
# starting 15:30 ET (the settle bar).
TRADE_CLOCKS = [f"{h:02d}:{m:02d}" for h in range(10, 16) for m in (0, 30)]
MIN_DAYS = 63


def panel_bars(path: str) -> pd.DataFrame:
    """Per (day, clock) trade bars from a bar-END-labelled yhat panel.

    Returns columns: ``day`` (naive ET date), ``clock`` (ET HH:MM of the
    bar START), ``f_next`` (variance forecast issued at the bar start),
    ``rv`` (the bar's own realized variance).
    """
    df = pd.read_parquet(path)
    start = pd.to_datetime(df["t"], utc=True) - pd.Timedelta(minutes=30)
    et = start.dt.tz_convert("America/New_York")
    out = pd.DataFrame(
        {
            "day": et.dt.normalize().dt.tz_localize(None),
            "clock": et.dt.strftime("%H:%M"),
            "f_next": df["yhat"].to_numpy(float) ** 2 * df["baseline"].to_numpy(float),
            "rv": df["rv_raw"].to_numpy(float),
        }
    )
    out = out[out["clock"].isin(TRADE_CLOCKS)]
    return out.sort_values(["day", "clock"]).reset_index(drop=True)


def causal_slice_share(bars: pd.DataFrame, min_days: int = MIN_DAYS) -> pd.DataFrame:
    """w_next per (day, clock): the entry bar's share of remaining variance.

    Expanding per-clock mean of realized bar variance over days < d
    (``min_days`` warmup, lagged one day), normalized over the remaining
    clocks c..15:30.  NaN before warmup.
    """
    prof = bars.pivot_table(index="day", columns="clock", values="rv", aggfunc="mean")
    cols = [c for c in TRADE_CLOCKS if c in prof.columns]
    prof = prof.reindex(columns=cols).sort_index()
    prof_exp = prof.expanding(min_periods=min_days).mean().shift(1)
    rem = prof_exp[cols[::-1]].cumsum(axis=1)[cols]
    w = prof_exp / rem
    ws = w.stack().rename("w_next").reset_index()
    ws.columns = pd.Index(["day", "clock", "w_next"])
    return ws


def ft_from_bars(bars: pd.DataFrame, min_days: int = MIN_DAYS) -> pd.DataFrame:
    """F_rem(e) = f_next(e) / w_next(e) per (day, clock); NaN pre-warmup."""
    ws = causal_slice_share(bars, min_days=min_days)
    m = bars.merge(ws, on=["day", "clock"], how="left")
    with np.errstate(divide="ignore", invalid="ignore"):
        m["F_rem"] = np.where(
            m["w_next"].to_numpy(float) > 0,
            m["f_next"].to_numpy(float) / m["w_next"].to_numpy(float),
            np.nan,
        )
    return m[["day", "clock", "f_next", "rv", "w_next", "F_rem"]]


def ft_remaining(panel_path: str, min_days: int = MIN_DAYS) -> pd.DataFrame:
    """F_rem per (day, clock) straight from a yhat panel file."""
    return ft_from_bars(panel_bars(panel_path), min_days=min_days)
