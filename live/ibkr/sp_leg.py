"""The S&P-leg target of a joint account -- a REPORT, never an order.

An account that holds the S&P 500 beside the insurance book loses on both legs
in a crash, and that is when margin is called.  Studies 67 and 68 measured what
can be done about it from information the runner already has:

* the SIZE of a crash is forecastable (68-89 % of the index's worst-1 % days
  fall in the top fifth of a prior-close volatility signal), its first hit out
  of a calm market and its direction are not;
* scaling the S&P leg down when forecast volatility is high roughly halves the
  worst 20-session run of the joint book and leaves its Sharpe no lower.

Two rules, both read off the premium ledger's session-variance column and both
strictly causal (sessions before ``asof`` only):

``vol_managed``
    weight = min(1, m / s): ``s`` is today's forecast volatility from an
    expanding-OLS HAR regression of the log session variance on its previous
    value, its 5-session mean and its 22-session mean; ``m`` is the median of
    the forecasts made for earlier sessions.  Never above 1 (no leverage).
    Study 68: it meets the written criterion at the timing the runner can
    deliver.  The DEFAULT.
``brake``
    the insurance book's own deleveraging multiplier (1 / 0.5 / 0).  Study 68:
    it halves the worst 20-session run too, but a session late it no longer
    lowers the maximum drawdown in 1998-2011 (a slow bear market at moderate
    volatility), so it does not meet the criterion.  Offered, not the default.

The runner REPORTS the weight (and, given ``--sp-leg-notional``, the dollars and
the ES / MES equivalent); the operator trades the S&P leg.  Nothing here places
an order, and the runner's flat-book checks never see the S&P leg.
"""

from __future__ import annotations

from datetime import date
from math import floor, isfinite
from typing import Any

import numpy as np

from live.ibkr.premium_ledger import SESSION_RV_CLOCK, PremiumLedger

__all__ = [
    "HAR_LONG",
    "HAR_SHORT",
    "SP_LEG_MIN_SESSIONS",
    "SP_LEG_RULES",
    "har_variance_forecasts",
    "sp_leg_target",
    "vol_managed_weight",
]

SP_LEG_RULES: tuple[str, ...] = ("vol_managed", "brake")
#: The HAR regression's two trailing means, in sessions (a week and a month).
HAR_SHORT = 5
HAR_LONG = 22
#: Sessions a regression, and then a median of its forecasts, must have before
#: either is trusted: the deleveraging rule's own warm-up (proposal 50).
SP_LEG_MIN_SESSIONS = 252


def har_variance_forecasts(rv: np.ndarray, min_sessions: int) -> np.ndarray:
    """Expanding-OLS HAR forecasts of the log variance, one step ahead.

    ``rv`` is the usable (finite, positive) session variances in time order.
    Element ``t`` of the result (``t`` in ``0 .. len(rv)``) is the forecast of
    ``log rv[t]`` made from ``rv[:t]`` alone, by a regression fitted on the
    pairs strictly before ``t``; the LAST element is the forecast for the
    session after the series ends.  NaN until ``min_sessions`` pairs exist.
    """
    v = np.asarray(rv, float)
    n = int(v.size)
    out = np.full(n + 1, np.nan)
    if n < HAR_LONG:
        return out
    x = np.log(v)
    csum = np.concatenate(([0.0], np.cumsum(v)))
    xx = np.zeros((4, 4))
    xy = np.zeros(4)
    pairs = 0
    for t in range(HAR_LONG, n + 1):
        feats = np.array(
            [
                1.0,
                x[t - 1],
                np.log((csum[t] - csum[t - HAR_SHORT]) / HAR_SHORT),
                np.log((csum[t] - csum[t - HAR_LONG]) / HAR_LONG),
            ]
        )
        if pairs >= int(min_sessions):
            # least squares, not solve: a flat history (every variance the same)
            # makes the normal matrix singular, and its forecast is that level
            out[t] = float(feats @ np.linalg.lstsq(xx, xy, rcond=None)[0])
        if t < n:
            xx += np.outer(feats, feats)
            xy += feats * x[t]
            pairs += 1
    return out


def vol_managed_weight(
    ledger: PremiumLedger,
    asof: date,
    min_sessions: int = SP_LEG_MIN_SESSIONS,
    *,
    clock: str = SESSION_RV_CLOCK,
) -> tuple[float, dict[str, Any]]:
    """The volatility-managed S&P-leg weight for ``asof``, in (0, 1].

    Warm-up (too few sessions for the regression, or too few earlier forecasts
    for their median) is INACTIVE: weight 1.0, ``state="warmup"``.
    """
    series = ledger.session_rv_series(clock=clock)
    prior = series[series.index < np.datetime64(asof)].to_numpy(float)
    prior = prior[np.isfinite(prior)]
    f = har_variance_forecasts(prior, min_sessions)
    today = float(f[-1])
    earlier = f[:-1]
    earlier = earlier[np.isfinite(earlier)]
    info: dict[str, Any] = {
        "rule": "vol_managed",
        "n_sessions": int(prior.size),
        "n_forecasts": int(earlier.size),
        "forecast_var": float(np.exp(today)) if isfinite(today) else float("nan"),
    }
    if not isfinite(today) or earlier.size < int(min_sessions):
        return 1.0, {**info, "state": "warmup", "weight": 1.0}
    median_vol = float(np.median(np.sqrt(np.exp(earlier))))
    vol = float(np.sqrt(np.exp(today)))
    weight = min(1.0, median_vol / vol)
    state = "full" if weight >= 1.0 else "reduced"
    return weight, {
        **info,
        "state": state,
        "weight": weight,
        "forecast_vol": vol,
        "median_forecast_vol": median_vol,
    }


def sp_leg_target(
    ledger: PremiumLedger,
    asof: date,
    rule: str,
    *,
    notional: float = 0.0,
    spot: float = float("nan"),
    es_multiplier: float = 50.0,
    mes_multiplier: float = 5.0,
    delever_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The S&P-leg report for ``asof``: weight, and the dollars if a notional is given.

    ``notional`` is the S&P leg at FULL weight, in dollars.  With a finite
    ``spot`` the target is also stated in futures: ES for the bulk, MES for the
    remainder, both floored (the report never rounds exposure up).
    """
    if rule not in SP_LEG_RULES:
        raise ValueError(
            "sp-leg rule must be one of " + repr(SP_LEG_RULES) + ", got " + repr(rule)
        )
    if rule == "brake":
        mult, binfo = ledger.delever_multiplier(asof, **(delever_kwargs or {}))
        weight = float(mult)
        info: dict[str, Any] = {"rule": "brake", "weight": weight, **dict(binfo)}
    else:
        weight, info = vol_managed_weight(ledger, asof)
    out: dict[str, Any] = {
        **info,
        "session": asof.isoformat(),
        "notional_full": float(notional),
    }
    if notional > 0.0:
        target = weight * float(notional)
        out["target_dollars"] = target
        if isfinite(spot) and spot > 0.0:
            es = int(floor(target / (spot * es_multiplier)))
            rest = target - es * spot * es_multiplier
            out["target_es"] = es
            out["target_mes"] = int(floor(rest / (spot * mes_multiplier)))
    return out
