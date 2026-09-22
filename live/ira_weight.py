"""The volatility-managed S&P weight from DAILY bars -- for an account the runner does not see.

Studies 67 and 68 (`writeup/intraday_proposals/67_crash_forecast.py`, `68_...`)
found that scaling an S&P position by

    weight = min(1, median forecast vol / today's forecast vol)

roughly halves its worst 20-session run at the same growth, and that a 3x
daily-reset fund held at that weight earned 10.6 %/yr against 6.7 % held flat
(1998-2024, financing ignored).  The runner reports the weight off its premium
ledger (`live/ibkr/sp_leg.py`); an IRA at another broker has no ledger.  This
script computes the same weight from the S&P 500's daily bars alone:

  variance proxy   Parkinson's range estimator (ln(High/Low))^2 / (4 ln 2) per
                   session -- a regular-hours variance from the day's range, the
                   role the panel's one-minute realized variance plays in 67/68
  forecast         `live.ibkr.sp_leg.har_variance_forecasts`: the SAME expanding
                   HAR regression on the log variance (lags 1, 5, 22 sessions),
                   fitted on sessions strictly before the one forecast
  weight           min(1, median of the earlier forecasts' vol / today's)

Validation (`--validate`, 1998-2024): the daily-bar weight against the panel's
(results/atm_straddle_intraday_holdclose/proposals/67/vm_weight_panel.parquet)
-- correlation, agreement on the days each cuts, and the 3x fund's growth and
drawdowns under each -- printed, not assumed.

The weight for session t uses bars through t-1: it is known after the close
and acted on the next session.  It is never above 1; ``--cap`` scales it (2/3
holds a 3x fund at lifecycle investing's 2:1 ceiling).  A report, not an order.

Usage
  python -m live.ira_weight                       today's weight
  python -m live.ira_weight --holding 47000 --cap 0.6667 --fund-leverage 3
  python -m live.ira_weight --validate
  python -m live.ira_weight --offline             cached bars only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from live.ibkr.sp_leg import SP_LEG_MIN_SESSIONS, har_variance_forecasts  # noqa: E402

CACHE = ROOT / "results" / "ira_weight" / "gspc_ohlc.parquet"
PANEL_WEIGHT = (
    ROOT
    / "results"
    / "atm_straddle_intraday_holdclose"
    / "proposals"
    / "67"
    / "vm_weight_panel.parquet"
)
FIRST_BAR = "1996-01-01"  # two years of range history before the 1998 comparison window
PARKINSON = 1.0 / (4.0 * np.log(2.0))
PERIODS_PER_YEAR = 252.0
DRAWDOWNS = (  # S&P 500 closing peak -> trough (study 67's dates)
    ("2000-02 dot-com", "2000-03-24", "2002-10-09"),
    ("2007-09 financial crisis", "2007-10-09", "2009-03-09"),
    ("2020 covid", "2020-02-19", "2020-03-23"),
    ("2022", "2022-01-03", "2022-10-12"),
)


def fetch_bars(offline: bool) -> pd.DataFrame:
    """^GSPC daily open/high/low/close, cached; the cache is extended when online."""
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    cached = pd.read_parquet(CACHE) if CACHE.exists() else None
    if offline:
        if cached is None:
            raise SystemExit("no cached bars at " + str(CACHE) + "; run once online")
        return cached
    import yfinance as yf

    raw = yf.download("^GSPC", start=FIRST_BAR, auto_adjust=False, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    bars = raw[["Open", "High", "Low", "Close"]].rename(columns=str.lower)
    bars.index = pd.to_datetime(bars.index).tz_localize(None).normalize()
    bars.index.name = "date"
    bars = bars[(bars["high"] > 0) & (bars["low"] > 0) & (bars["high"] >= bars["low"])]
    bars.to_parquet(CACHE)
    return bars


def parkinson_variance(bars: pd.DataFrame) -> pd.Series:
    """Per-session variance from the day's range; NaN where the range is zero."""
    v = PARKINSON * np.log(bars["high"] / bars["low"]) ** 2
    return v.where(v > 0)


def weights_from_variance(
    rv: pd.Series, min_sessions: int = SP_LEG_MIN_SESSIONS
) -> pd.DataFrame:
    """The weight for every session, and for the session after the last bar.

    Row t: the HAR forecast of session t's vol (from sessions before t), the
    median of the forecasts made for earlier sessions, and the weight.  The
    last row is dated one business day after the final bar: tomorrow's weight.
    """
    rv = rv.dropna()
    f = har_variance_forecasts(rv.to_numpy(float), min_sessions)  # len(rv) + 1
    vol = np.sqrt(np.exp(f))
    med = (
        pd.Series(vol).expanding(min_periods=min_sessions).median().shift(1).to_numpy()
    )
    w = np.where(
        np.isfinite(med) & np.isfinite(vol), np.minimum(1.0, med / vol), np.nan
    )
    dates = rv.index.append(pd.DatetimeIndex([rv.index[-1] + pd.offsets.BDay(1)]))
    out = pd.DataFrame(
        {
            "forecast_vol_ann": vol * np.sqrt(PERIODS_PER_YEAR),
            "median_vol_ann": med * np.sqrt(PERIODS_PER_YEAR),
            "weight": w,
        },
        index=dates,
    )
    out.index.name = "date"
    return out


def maxdd(r: np.ndarray) -> float:
    w = np.cumprod(1.0 + r)
    return float((w / np.maximum.accumulate(np.maximum(w, 1.0)) - 1.0).min())


def worst_run(r: np.ndarray, n: int = 20) -> float:
    lg = np.concatenate([[0.0], np.cumsum(np.log1p(r))])
    return float(np.expm1((lg[n:] - lg[:-n]).min()))


def fund_book(
    r: np.ndarray, w: np.ndarray, leverage: float, fee: float
) -> dict[str, float]:
    x = w * (leverage * r - fee / PERIODS_PER_YEAR)
    return {
        "growth": float(np.prod(1 + x) ** (PERIODS_PER_YEAR / len(x)) - 1),
        "max_drawdown": maxdd(x),
        "worst_20": worst_run(x),
    }


def validate(bars: pd.DataFrame, weights: pd.DataFrame) -> None:
    panel = pd.read_parquet(PANEL_WEIGHT)
    j = weights.join(panel, how="inner").dropna(subset=["weight", "vm_weight_panel"])
    a, b = j["weight"].to_numpy(), j["vm_weight_panel"].to_numpy()
    print(
        f"VALIDATION against the panel's weight on {len(j):,} sessions {j.index[0].date()}..{j.index[-1].date()}"
    )
    print(
        f"  correlation of the weights {np.corrcoef(a, b)[0, 1]:.3f}; of the forecast vols "
        f"{np.corrcoef(np.log(j['forecast_vol_ann']), np.log(j['har_vol_panel_ann']))[0, 1]:.3f}"
    )
    print(
        f"  mean |difference| {np.abs(a - b).mean():.3f}; below 1 on {float((a < 1).mean()):.1%} (daily bars) vs "
        f"{float((b < 1).mean()):.1%} (panel) of sessions; both below 1 on {float(((a < 1) & (b < 1)).mean()):.1%}, "
        f"one but not the other on {float(((a < 1) ^ (b < 1)).mean()):.1%}"
    )
    r = j["r"].to_numpy()
    rows = {}
    for lev, fee in ((1.0, 0.0), (3.0, 0.0091)):
        for name, w in (
            ("held", np.ones(len(r))),
            ("weight from daily bars", a),
            ("weight from the panel", b),
        ):
            rows[f"{lev:.0f}x fund, {name}"] = {
                **fund_book(r, w, lev, fee),
                **{
                    lab: float(
                        np.prod(
                            1
                            + (w * (lev * r - fee / PERIODS_PER_YEAR))[
                                (j.index > a0) & (j.index <= b0)
                            ]
                        )
                        - 1
                    )
                    for lab, a0, b0 in DRAWDOWNS
                },
            }
    pd.set_option("display.width", 250)
    print(
        "  the S&P and a 3x daily-reset fund (0.91% fee, financing ignored) under each weight:"
    )
    print(pd.DataFrame(rows).T.round(3).to_string())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--holding",
        type=float,
        default=0.0,
        help="dollars in the S&P position (or the leveraged fund) at full weight",
    )
    ap.add_argument(
        "--cap",
        type=float,
        default=1.0,
        help="maximum weight; 0.6667 holds a 3x fund at 2:1",
    )
    ap.add_argument(
        "--fund-leverage",
        type=float,
        default=1.0,
        help="the fund's leverage, to print the effective exposure",
    )
    ap.add_argument("--offline", action="store_true", help="use the cached bars only")
    ap.add_argument(
        "--validate",
        action="store_true",
        help="compare with the panel's weight, 1998-2024",
    )
    ap.add_argument(
        "--history", type=int, default=0, help="also print the last N sessions"
    )
    a = ap.parse_args(argv)
    if not (0.0 < a.cap <= 1.0):
        raise SystemExit("--cap must be in (0, 1]")
    bars = fetch_bars(a.offline)
    weights = weights_from_variance(parkinson_variance(bars))
    if a.validate:
        validate(bars, weights)
        return 0
    last_bar = bars.index[-1].date()
    row = weights.iloc[-1]
    w = float(row["weight"]) if np.isfinite(row["weight"]) else 1.0
    applied = min(a.cap, a.cap * w)
    print(
        f"last bar {last_bar}; weight for the next session ({weights.index[-1].date()}): {w:.3f}"
        + (" (warm-up: full weight)" if not np.isfinite(row["weight"]) else "")
    )
    print(
        f"  forecast vol {row['forecast_vol_ann']:.1%} a year vs its median {row['median_vol_ann']:.1%}"
        if np.isfinite(row["forecast_vol_ann"])
        else "  forecast not yet available"
    )
    print(f"  cap {a.cap:.3f} -> applied weight {applied:.3f}")
    if a.holding > 0:
        print(
            f"  hold ${applied * a.holding:,.0f} of the ${a.holding:,.0f} in the position, ${(1 - applied) * a.holding:,.0f} in cash"
            + (
                f"; effective S&P exposure {applied * a.fund_leverage:.2f}x the account"
                if a.fund_leverage != 1.0
                else ""
            )
        )
    if a.history:
        pd.set_option("display.width", 200)
        print(weights.tail(a.history + 1).round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
