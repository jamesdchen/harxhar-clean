"""Dealer Gamma Exposure (GEX) regime features from an OptionMetrics IvyDB EOD SPX chain.

The tractable, prop-compatible signal (see writeup/door3 + the flow pilot): today's standing-gamma
regime is KNOWN AT THE OPEN from the PRIOR session's EOD open interest, and it conditions today's
INTRADAY path (trend vs mean-revert) -- so we trade intraday and go flat by the close, but the regime
state is free EOD data. This is NOT tick-OVI front-running (an HFT/OPRA game); it is a daily conditioning
state on intraday direction (Baltussen-Da-Lammers-Martens hedging-demand channel).

Input: an OptionMetrics extract (SPX secid 108105) with per-option EOD rows + the underlying EOD close.
Output: results/gex/gex_daily.parquet, one row/date with the regime observables. The CALLER lags it one
session (shift(1)) before merging onto intraday bars -- that shift is the causal contract (regime for day
d is built from day d-1's settled OI/surface).

Construction (all analytic; the only free choice is the dealer-sign convention, exposed as a parameter and
meant to be TESTED through the gate, never hard-assumed):
  net GEX level   = sum_i s_i * gamma_i * OI_i * M * S^2 * 0.01   ($ dealer delta re-hedged per +1% move)
  zero-gamma flip = the S where sum_i s_i * gammaBS(S;K_i,sigma_i,tau_i) * OI_i crosses 0 (regime boundary)
  pin strike      = argmax_K sum_{C,P} |gamma_K*OI_K|            (sign-FREE magnet; needs no dealer sign)
  net charm/vanna = calendar-timed hedging pressure (signed by dealer position; overall sign left to the fit)
with M = SPX multiplier (100). Greeks: IvyDB `gamma` used directly for the level when present (its smoothed
surface beats a BS gamma off noisy quotes); BS recompute from `impl_volatility` drives the flip curve.
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

MULT = 100.0  # SPX contract multiplier
MAX_DTE = 90  # standing-gamma horizon; strikes beyond add ~no gamma and bloat the pull


def _read_any(path: str) -> pd.DataFrame:
    """Read a .csv/.csv.gz/.parquet file, or concat every file matching a glob / in a directory (the WRDS
    web-query builder is chunked by year, so a folder or `data/optionm_spx/*.csv` is the natural input)."""
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*")))
    elif any(ch in path for ch in "*?["):
        files = sorted(glob.glob(path))
    else:
        files = [path]
    if not files:
        raise SystemExit(f"no files matched {path!r}")
    frames = [
        pd.read_parquet(f) if f.endswith(".parquet") else pd.read_csv(f) for f in files
    ]
    return pd.concat(frames, ignore_index=True)


def load_options(path: str) -> pd.DataFrame:
    """Normalize a WRDS OptionMetrics `Option Prices` extract to gex.py's schema: lowercases columns,
    derives `strike` from `strike_price`/1000 (IvyDB stores strikes x1000), verifies the 6 hard-required
    columns. `gamma` is OPTIONAL -- absent, gex.py computes BS gamma from `impl_volatility`."""
    o = _read_any(path)
    o.columns = [c.lower() for c in o.columns]
    if "strike" not in o.columns and "strike_price" in o.columns:
        o["strike"] = o["strike_price"].astype(float) / 1000.0
    req = ["date", "exdate", "cp_flag", "strike", "impl_volatility", "open_interest"]
    miss = [c for c in req if c not in o.columns]
    if miss:
        raise SystemExit(f"extract missing {miss}; got columns {list(o.columns)}")
    o["cp_flag"] = o["cp_flag"].astype(str).str.upper().str[0]  # 'C'/'P'
    return o


def load_spot(path: str) -> pd.Series:
    """WRDS OptionMetrics `Security Prices` (secprd) extract -> date->close Series (index level for SPX)."""
    s = _read_any(path)
    s.columns = [c.lower() for c in s.columns]
    ccol = next(
        (c for c in ("close", "close_price", "px_close") if c in s.columns), None
    )
    if ccol is None or "date" not in s.columns:
        raise SystemExit(f"spot extract needs date+close; got {list(s.columns)}")
    return pd.Series(s[ccol].astype(float).values, index=pd.to_datetime(s["date"]))


def _phi(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)


def bs_gamma(S: float, K: np.ndarray, sig: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Black-Scholes gamma per share (r=q=0; gamma is ~insensitive to the carry terms). Bounded, no clip:
    degenerate (tau<=0 or sig<=0) rows return 0 (a genuinely expired/undefined option has no gamma)."""
    ok = (tau > 0) & (sig > 0) & (K > 0)
    g = np.zeros_like(K, dtype=float)
    st = sig[ok] * np.sqrt(tau[ok])
    d1 = (np.log(S / K[ok]) + 0.5 * sig[ok] ** 2 * tau[ok]) / st
    g[ok] = _phi(d1) / (S * st)
    return g


def bs_vanna(S: float, K: np.ndarray, sig: np.ndarray, tau: np.ndarray) -> np.ndarray:
    ok = (tau > 0) & (sig > 0) & (K > 0)
    v = np.zeros_like(K, dtype=float)
    st = sig[ok] * np.sqrt(tau[ok])
    d1 = (np.log(S / K[ok]) + 0.5 * sig[ok] ** 2 * tau[ok]) / st
    d2 = d1 - st
    v[ok] = -_phi(d1) * d2 / sig[ok]
    return v


def bs_charm(S: float, K: np.ndarray, sig: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """dDelta/dtau at r=q=0 (overall sign is absorbed by the downstream fitted coefficient)."""
    ok = (tau > 0) & (sig > 0) & (K > 0)
    c = np.zeros_like(K, dtype=float)
    st = sig[ok] * np.sqrt(tau[ok])
    d1 = (np.log(S / K[ok]) + 0.5 * sig[ok] ** 2 * tau[ok]) / st
    d2 = d1 - st
    c[ok] = _phi(d1) * d2 / (2.0 * tau[ok])
    return c


def zero_gamma_level(
    S: float,
    K: np.ndarray,
    sig: np.ndarray,
    tau: np.ndarray,
    w: np.ndarray,
    span=0.1,
    n=201,
) -> float:
    """Spot at which signed net gamma crosses 0, nearest the current spot; NaN if no crossing in +-span."""
    grid = S * np.linspace(1.0 - span, 1.0 + span, n)
    G = np.array([float((w * bs_gamma(s, K, sig, tau)).sum()) for s in grid])
    sign = np.sign(G)
    cross = np.where(np.diff(sign) != 0)[0]
    if len(cross) == 0:
        return float("nan")
    j = cross[np.argmin(np.abs(grid[cross] - S))]
    g0, g1, s0, s1 = G[j], G[j + 1], grid[j], grid[j + 1]
    return float(s0 - g0 * (s1 - s0) / (g1 - g0)) if g1 != g0 else float(s0)


def gex_for_day(day: pd.DataFrame, S: float, call_sign: float = 1.0) -> dict:
    """One EOD chain -> regime observables. `day` has cols cp_flag, strike, impl_volatility, gamma,
    open_interest, tau. dealer sign s = call_sign for calls, -call_sign for puts (TEST this convention)."""
    cp = day["cp_flag"].to_numpy()
    K = day["strike"].to_numpy(float)
    sig = day["impl_volatility"].to_numpy(float)
    tau = day["tau"].to_numpy(float)
    oi = day["open_interest"].to_numpy(float)
    g_iv = day["gamma"].to_numpy(float) if "gamma" in day else bs_gamma(S, K, sig, tau)
    s = np.where(cp == "C", call_sign, -call_sign)

    dollar = s * g_iv * oi * MULT * S * S * 0.01
    w = s * oi  # weights for the BS flip curve
    # pin: sign-free magnet = strike with the most gamma*OI (calls+puts)
    pin_w = np.abs(g_iv) * oi
    pin = (
        float(pd.Series(pin_w, index=K).groupby(level=0).sum().idxmax())
        if pin_w.sum() > 0
        else float("nan")
    )
    is0 = tau <= 1.5 / 365.25  # ~next-session expiry ("0DTE" carried from prior EOD OI)
    return {
        "spot": S,
        "gex_level": float(dollar.sum()),
        "gex_0dte": float(dollar[is0].sum()),
        "zero_gamma": zero_gamma_level(S, K, sig, tau, w),
        "pin_strike": pin,
        "net_vanna": float((s * bs_vanna(S, K, sig, tau) * oi * MULT).sum()),
        "net_charm": float((s * bs_charm(S, K, sig, tau) * oi * MULT).sum()),
        "n_opt": int(len(day)),
        "total_oi": float(oi.sum()),
    }


def build_gex_daily(
    options: pd.DataFrame, spot: pd.Series, call_sign: float = 1.0
) -> pd.DataFrame:
    """options: EOD rows (date, exdate, cp_flag, strike, impl_volatility, gamma, open_interest).
    spot: date -> underlying close. Returns the daily regime table (EOD-stamped; caller lags it)."""
    o = options.copy()
    o["date"] = pd.to_datetime(o["date"])
    o["exdate"] = pd.to_datetime(o["exdate"])
    o["tau"] = (o["exdate"] - o["date"]).dt.days / 365.25
    o = o[(o["tau"] > 0) & (o["tau"] <= MAX_DTE / 365.25) & (o["open_interest"] > 0)]
    sp = pd.Series(spot.values, index=pd.to_datetime(spot.index))
    rows = []
    for d, day in o.groupby("date", sort=True):
        if d not in sp.index:
            continue
        r = gex_for_day(day, float(sp.loc[d]), call_sign)
        r["date"] = d
        rows.append(r)
    out = pd.DataFrame(rows).set_index("date").sort_index()
    out["dist_to_flip"] = (out["spot"] - out["zero_gamma"]) / out["spot"]
    out["dist_to_pin"] = (out["spot"] - out["pin_strike"]) / out["spot"]
    out["regime_long_gamma"] = np.sign(
        out["gex_level"]
    )  # +1 suppress/revert, -1 amplify/trend
    return out


def _smoke() -> None:
    """Unit checks on the primitives (no external data), each isolating one piece of the math. NOTE call
    and put gamma are EQUAL at a strike (put-call gamma parity), so net dealer gamma is driven purely by
    the call-vs-put OI IMBALANCE weighted by gamma -- an equal-OI chain is identically 0 (a real property,
    the reason the first synthetic returned 0). These tests build genuine imbalances."""
    S, sig, tau = 5000.0, 0.15, 17 / 365.25
    checks = {}

    # 1. BS gamma matches the closed form at the money
    g_atm = bs_gamma(S, np.array([S]), np.array([sig]), np.array([tau]))[0]
    d1 = 0.5 * sig * np.sqrt(tau)
    checks["bs_gamma ATM == closed form"] = (
        abs(g_atm - _phi(d1) / (S * sig * np.sqrt(tau))) < 1e-12
    )

    # 2. flip finder: dealer short a put @4900 (-) + long a call @5100 (+), equal OI -> G crosses ~mid
    K = np.array([5100.0, 4900.0])
    w = np.array([1000.0, -1000.0])  # s*OI: +long call, -short put
    flip = zero_gamma_level(S, K, np.array([sig, sig]), np.array([tau, tau]), w)
    checks["flip bracketed in (4900,5100)"] = 4900.0 < flip < 5100.0

    # 3-5. build_gex_daily on a put-OI-heavy downside chain (net short gamma) with a fat CALL pin @5000
    Ks = np.arange(4400, 5601, 25.0)
    rows = []
    for Kk in Ks:
        c_oi = 20000.0 if Kk == 5000.0 else 1000.0  # fat long-call pin at 5000
        p_oi = 8000.0 if Kk <= 4750.0 else 500.0  # heavy downside crash puts
        rows.append(
            dict(
                date="2024-01-02",
                exdate="2024-01-19",
                cp_flag="C",
                strike=Kk,
                impl_volatility=sig,
                gamma=np.nan,
                open_interest=c_oi,
            )
        )
        rows.append(
            dict(
                date="2024-01-02",
                exdate="2024-01-19",
                cp_flag="P",
                strike=Kk,
                impl_volatility=sig,
                gamma=np.nan,
                open_interest=p_oi,
            )
        )
    opt = pd.DataFrame(rows)
    opt["tau"] = (
        pd.to_datetime(opt["exdate"]) - pd.to_datetime(opt["date"])
    ).dt.days / 365.25
    opt["gamma"] = bs_gamma(
        S,
        opt["strike"].to_numpy(float),
        opt["impl_volatility"].to_numpy(float),
        opt["tau"].to_numpy(float),
    )
    spot = pd.Series({pd.Timestamp("2024-01-02"): S})
    pos = build_gex_daily(opt, spot, call_sign=1.0)
    neg = build_gex_daily(opt, spot, call_sign=-1.0)
    checks["fat call pin => net long gamma > 0"] = pos["gex_level"].iloc[0] > 0
    checks["pin at fat-OI strike 5000"] = pos["pin_strike"].iloc[0] == 5000.0
    checks["convention flip flips level sign"] = np.sign(
        neg["gex_level"].iloc[0]
    ) == -np.sign(pos["gex_level"].iloc[0])

    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(
        pos[
            [
                "gex_level",
                "zero_gamma",
                "pin_strike",
                "dist_to_flip",
                "net_vanna",
                "net_charm",
            ]
        ].to_string()
    )
    assert all(checks.values()), "smoke invariants failed"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--options", default="data/optionm_spx_options.parquet")
    ap.add_argument("--spot", default="data/optionm_spx_spot.parquet")
    ap.add_argument("--call-sign", type=float, default=1.0)
    ap.add_argument("--out", default="results/gex/gex_daily.parquet")
    a = ap.parse_args()
    if a.smoke:
        _smoke()
        return
    opt = load_options(a.options)
    spot = load_spot(a.spot)
    out = build_gex_daily(opt, spot, call_sign=a.call_sign)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    out.to_parquet(a.out)
    print(
        f"wrote {a.out}: {len(out)} days, {out.index.min().date()}..{out.index.max().date()}"
    )
    print(
        out[["gex_level", "zero_gamma", "pin_strike", "regime_long_gamma"]]
        .describe()
        .to_string()
    )


if __name__ == "__main__":
    main()
