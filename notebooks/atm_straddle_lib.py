"""Shared 0DTE ATM-package helpers for the straddle notebooks.

Protocol matches notebooks/_write_0dte_nb.py and
writeup/sections/methods_close_option.tex.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

SPX_MULTIPLIER = 100.0
WINDOW_DAYS = 250  # legacy flat MZ window (halflife=None path)
MZ_HALFLIFE_DAYS = 63  # EWMA halflife (days) for the smear's sufficient statistics
YHAT_LABEL = {
    "a0": "HAR + calendar OLS",
    "blk2": "block-diag ridge",
    "lgbm": "LightGBM",
    "xgb": "XGBoost",
    "lasso_t": "lasso (causally tuned)",
    "lasso_f": "lasso (fixed 1e-4)",
    "enet": "elastic net (causally tuned)",
}
MODEL_ORDER = ["a0", "blk2", "lgbm", "xgb", "lasso_t", "lasso_f", "enet"]
RULE_ORDER = ["always short", "long-short volatility", "unit-median VRP"]


def find_repo(start: Path | None = None) -> Path:
    start = Path.cwd() if start is None else Path(start)
    for q in [start.resolve(), *start.resolve().parents]:
        if (q / "data" / "spxw_chain.parquet").exists():
            return q
        if (q / "notebooks" / "_write_0dte_nb.py").exists():
            return q
    raise FileNotFoundError("repo root not found")


def yhat_paths(repo: Path) -> dict[str, Path]:
    root = repo / "results" / "spxw_pnl"
    return {
        "a0": root / "yhat_a0.parquet",
        "blk2": root / "yhat_blk2_fomc1.parquet",
        "lgbm": root / "yhat_tree00.parquet",
        "xgb": root / "yhat_tree16.parquet",
        "lasso_t": root / "yhat_b2lasso_tuned.parquet",
        "lasso_f": root / "yhat_b2lasso.parquet",
        "enet": root / "yhat_b3enet_tuned.parquet",
    }


def _mz_day_coefs(
    yhat, rv_raw, baseline, day_codes, n_days, need_days, halflife, fit_mask=None
):
    """Per-day causal MZ coefficients (a, b, s2) in y-space.

    halflife=None reproduces the legacy flat [d-WINDOW_DAYS, d) window
    via prefix sums; otherwise prior days are exponentially weighted
    with the given halflife and gated on a Kish effective sample size
    >= 200 (same constant as the flat path's n >= 200). Both paths use
    strictly prior days and start at day 63. fit_mask (bool per row)
    restricts which rows enter the fit — the loaders pass the scored
    session bars (10:00-15:30 ET) so off-session dynamics cannot
    pollute the calibration; coefficients still apply to every row.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.sqrt(np.maximum(rv_raw, 0.0) / np.maximum(baseline, 1e-18))
    valid_day = day_codes >= 0
    finite = np.isfinite(yhat) & np.isfinite(y) & (baseline > 0) & valid_day
    if fit_mask is not None:
        finite = finite & np.asarray(fit_mask, dtype=bool)
    x0 = np.where(finite, yhat, 0.0)
    y0 = np.where(finite, y, 0.0)
    dc = np.where(valid_day, day_codes, 0)
    stats = {
        "n": finite.astype(np.float64),
        "x": x0,
        "xx": x0 * x0,
        "y": y0,
        "xy": x0 * y0,
        "yy": y0 * y0,
    }
    daily = {
        k: np.bincount(dc, weights=np.where(valid_day, v, 0.0), minlength=n_days)
        for k, v in stats.items()
    }
    a_d = np.full(n_days, np.nan)
    b_d = np.full(n_days, np.nan)
    s2_d = np.full(n_days, np.nan)
    if halflife is None:
        pre = {k: np.concatenate([[0.0], np.cumsum(v)]) for k, v in daily.items()}
        if need_days is None:
            days = np.arange(63, n_days, dtype=np.int64)
        else:
            days = np.asarray(
                sorted(d for d in need_days if 63 <= d < n_days), dtype=np.int64
            )
        lo = np.maximum(0, days - WINDOW_DAYS)
        w = {k: p[days] - p[lo] for k, p in pre.items()}
        n = w["n"]
        denom = n * w["xx"] - w["x"] ** 2
        ok = (n >= 200) & (denom > 0)
        safe_den = np.where(ok, denom, 1.0)
        safe_n = np.where(ok, n, 1.0)
        b = np.where(ok, (n * w["xy"] - w["x"] * w["y"]) / safe_den, np.nan)
        a = np.where(ok, (w["y"] - b * w["x"]) / safe_n, np.nan)
        s2 = np.where(ok, (w["yy"] - a * w["y"] - b * w["xy"]) / safe_n, np.nan)
        a_d[days], b_d[days], s2_d[days] = a, b, s2
        return a_d, b_d, s2_d
    lam = 0.5 ** (1.0 / float(halflife))
    acc = dict.fromkeys(daily, 0.0)
    acc2_n = 0.0  # lambda^2-weighted n, for the Kish effective sample size
    for d in range(n_days):
        w_n = acc["n"]
        if d >= 63 and w_n > 0 and acc2_n > 0:
            n_eff = w_n * w_n / acc2_n
            denom = w_n * acc["xx"] - acc["x"] ** 2
            if n_eff >= 200 and denom > 0:
                b = (w_n * acc["xy"] - acc["x"] * acc["y"]) / denom
                a = (acc["y"] - b * acc["x"]) / w_n
                a_d[d] = a
                b_d[d] = b
                s2_d[d] = (acc["yy"] - a * acc["y"] - b * acc["xy"]) / w_n
        for k in acc:
            acc[k] = lam * acc[k] + daily[k][d]
        acc2_n = lam * lam * acc2_n + daily["n"][d]
    return a_d, b_d, s2_d


def second_order_raw(
    yhat,
    rv_raw,
    baseline,
    day_codes,
    n_days,
    need_days=None,
    halflife=None,
    fit_mask=None,
):
    """Causal second-moment back-transform (flat 250-day window by default)."""
    a_d, b_d, s2_d = _mz_day_coefs(
        yhat, rv_raw, baseline, day_codes, n_days, need_days, halflife, fit_mask
    )
    valid_day = day_codes >= 0
    dc = np.where(valid_day, day_codes, 0)
    te = np.isfinite(yhat) & (baseline > 0) & valid_day & np.isfinite(a_d[dc])
    m = a_d[day_codes[te]] + b_d[day_codes[te]] * yhat[te]
    f = np.full(len(yhat), np.nan)
    f[te] = (m**2 + s2_d[day_codes[te]]) * baseline[te]
    return f


def second_order_mz(
    yhat,
    rv_raw,
    baseline,
    day_codes,
    n_days,
    need_days=None,
    halflife=None,
    fit_mask=None,
):
    """Same map as second_order_raw, also returning m and s2 on each row."""
    a_d, b_d, s2_d = _mz_day_coefs(
        yhat, rv_raw, baseline, day_codes, n_days, need_days, halflife, fit_mask
    )
    valid_day = day_codes >= 0
    dc = np.where(valid_day, day_codes, 0)
    te = np.isfinite(yhat) & (baseline > 0) & valid_day & np.isfinite(a_d[dc])
    m = np.full(len(yhat), np.nan)
    s2_row = np.full(len(yhat), np.nan)
    rv = np.full(len(yhat), np.nan)
    m[te] = a_d[day_codes[te]] + b_d[day_codes[te]] * yhat[te]
    s2_row[te] = s2_d[day_codes[te]]
    rv[te] = (m[te] ** 2 + s2_row[te]) * baseline[te]
    return rv, m, s2_row


def load_yhat_1530(path: Path, need_dates=None) -> pd.DataFrame:
    # Bar-end-labelled stamps: the 15:30 book's fresh forecast — issued at
    # 15:30 for the 15:30->close bar it trades — lives on the STAMP-16:00
    # row, whose rv_raw is that bar's own realized variance. (The stamp-
    # 15:30 row is the forecast of 15:00->15:30, one bar stale.)
    df = pd.read_parquet(path).sort_values("t").reset_index(drop=True)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df["et"] = df["t"].dt.tz_convert("America/New_York")
    df["date"] = df["et"].dt.normalize().dt.tz_localize(None)
    is_row = (df["et"].dt.hour == 16) & (df["et"].dt.minute == 0)
    yhat = df["yhat"].to_numpy(float)
    base = df["baseline"].to_numpy(float)
    rv_raw = df["rv_raw"].to_numpy(float)
    day_codes, uniq = pd.factorize(df["date"], sort=True)
    need_days = None
    if need_dates is not None:
        pos = {d: k for k, d in enumerate(uniq)}
        need_days = {pos[d] for d in need_dates if d in pos}
    mins = df["et"].dt.hour * 60 + df["et"].dt.minute
    rth = ((mins >= 10 * 60 + 30) & (mins <= 16 * 60)).to_numpy()
    df["rv_hat"] = second_order_raw(
        yhat, rv_raw, base, day_codes, len(uniq), need_days=need_days, fit_mask=rth
    )
    out = (
        df.loc[is_row, ["date", "yhat", "baseline", "rv_raw", "rv_hat"]]
        .dropna(subset=["rv_hat"])
        .drop_duplicates("date")
        .set_index("date")
    )
    return out


def load_yhat_1530_mz_cached(
    tag: str, path: Path, need_dates, cache: Path
) -> pd.DataFrame:
    h = hashlib.sha1()
    st = os.stat(path)
    h.update(f"v5-mz-fresh:{st.st_size}:{st.st_mtime_ns}:flat{WINDOW_DAYS}".encode())
    for d in sorted(need_dates):
        h.update(str(d).encode())
    cp = cache / f"yhat1530mz_{tag}_{h.hexdigest()[:16]}.parquet"
    if cp.exists():
        return pd.read_parquet(cp)
    df = pd.read_parquet(path).sort_values("t").reset_index(drop=True)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df["et"] = df["t"].dt.tz_convert("America/New_York")
    df["date"] = df["et"].dt.normalize().dt.tz_localize(None)
    # Fresh row for the 15:30 book: stamp 16:00 (see load_yhat_1530).
    is_row = (df["et"].dt.hour == 16) & (df["et"].dt.minute == 0)
    yhat = df["yhat"].to_numpy(float)
    base = df["baseline"].to_numpy(float)
    rv_raw = df["rv_raw"].to_numpy(float)
    day_codes, uniq = pd.factorize(df["date"], sort=True)
    pos = {d: k for k, d in enumerate(uniq)}
    need_days = {pos[d] for d in need_dates if d in pos}
    mins = df["et"].dt.hour * 60 + df["et"].dt.minute
    rth = ((mins >= 10 * 60 + 30) & (mins <= 16 * 60)).to_numpy()
    rv, m, s2 = second_order_mz(
        yhat, rv_raw, base, day_codes, len(uniq), need_days=need_days, fit_mask=rth
    )
    df["rv_hat"] = rv
    df["m"] = m
    df["s2"] = s2
    df["yhat_vol"] = df["yhat"] * np.sqrt(np.maximum(df["baseline"], 0.0))
    df["m_vol"] = df["m"] * np.sqrt(np.maximum(df["baseline"], 0.0))
    out = (
        df.loc[
            is_row,
            [
                "date",
                "yhat",
                "baseline",
                "rv_raw",
                "rv_hat",
                "m",
                "s2",
                "yhat_vol",
                "m_vol",
            ],
        ]
        .dropna(subset=["rv_hat"])
        .drop_duplicates("date")
        .set_index("date")
    )
    for old in cache.glob(f"yhat1530mz_{tag}_*.parquet"):
        old.unlink()
    out.to_parquet(cp)
    return out


def load_yhat_1530_cached(
    tag: str, path: Path, need_dates, cache: Path
) -> pd.DataFrame:
    h = hashlib.sha1()
    st = os.stat(path)
    h.update(f"v5-vec-fresh:{st.st_size}:{st.st_mtime_ns}:flat{WINDOW_DAYS}".encode())
    for d in sorted(need_dates):
        h.update(str(d).encode())
    cp = cache / f"yhat1530_{tag}_{h.hexdigest()[:16]}.parquet"
    if cp.exists():
        return pd.read_parquet(cp)
    out = load_yhat_1530(path, need_dates)
    for old in cache.glob(f"yhat1530_{tag}_*.parquet"):
        old.unlink()
    out.to_parquet(cp)
    return out


def load_yhat_panel(path: Path) -> pd.DataFrame:
    """Every 30-min stamp. Same second-order map as the 15:30 loader."""
    df = pd.read_parquet(path).sort_values("t").reset_index(drop=True)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df["et"] = df["t"].dt.tz_convert("America/New_York")
    df["date"] = df["et"].dt.normalize().dt.tz_localize(None)
    yhat = df["yhat"].to_numpy(float)
    base = df["baseline"].to_numpy(float)
    rv_raw = df["rv_raw"].to_numpy(float)
    day_codes, uniq = pd.factorize(df["date"], sort=True)
    # Bar-end-labelled stamps: the scored trade bars 10:00-15:30 live on
    # stamps 10:30-16:00, so the session fit mask covers those stamps.
    mins = df["et"].dt.hour * 60 + df["et"].dt.minute
    rth = ((mins >= 10 * 60 + 30) & (mins <= 16 * 60)).to_numpy()
    df["rv_hat"] = second_order_raw(
        yhat, rv_raw, base, day_codes, len(uniq), need_days=None, fit_mask=rth
    )
    return df


def lagged_expanding_median(signal: pd.Series, min_periods: int = 63) -> pd.Series:
    """Expanding median of |signal|, shifted 1. Known before the current row."""
    return signal.abs().expanding(min_periods=min_periods).median().shift(1)


def causal_leverage(signal: pd.Series, cap: float = 3.0) -> pd.Series:
    med = lagged_expanding_median(signal)
    lev = (signal.abs() / med).clip(upper=cap)
    return lev.fillna(1.0)


def um_leverage_vs_lagged_scale(
    signal: pd.Series, med: pd.Series, cap: float = 3.0
) -> pd.Series:
    """|s_t| / med_t with med already lagged. Do not pass same-bar 15:30 |s| as med."""
    lev = (signal.abs() / med.astype(float)).clip(upper=cap)
    return lev.replace([np.inf, -np.inf], np.nan).fillna(1.0)


def rule_sizes(px: pd.DataFrame) -> dict[str, pd.Series]:
    lev = causal_leverage(px["signal"])
    pos = pd.Series(
        np.where(px["signal"].to_numpy(float) > 0, 1.0, -1.0), index=px.index
    )
    return {
        "always short": pd.Series(-1.0, index=px.index),
        "long-short volatility": pos,
        "unit-median VRP": pos * lev,
    }


def extra_weight_sizes(px: pd.DataFrame) -> dict[str, pd.Series]:
    s = px["signal"].astype(float)
    pos = pd.Series(np.where(s.to_numpy() > 0, 1.0, -1.0), index=px.index)
    lev = causal_leverage(s)
    abs_s = s.abs()
    rank = abs_s.expanding(min_periods=63).rank(pct=True).shift(1).fillna(0.5)
    r = px["R"].astype(float)
    vol = r.expanding(min_periods=63).std().shift(1)
    inv_vol = (vol.median() / vol).clip(upper=3.0).fillna(1.0)
    med = abs_s.expanding(min_periods=63).median().shift(1)
    dead = (abs_s >= 0.5 * med).fillna(False)
    um = pos * lev
    rp = um * r
    rp_vol = rp.expanding(min_periods=63).std().shift(1)
    vt = um * (1.0 / rp_vol).clip(upper=5.0)
    vt = vt.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    exp_var = r.expanding(min_periods=63).var().shift(1)
    kelly = (s / exp_var).clip(-3.0, 3.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return {
        "rank-|s|": pos * (1.0 + 4.0 * (rank - 0.5)).clip(0.0, 3.0),
        "inv-vol of R": pos * inv_vol,
        "dead-zone 0.5 med": pos.where(dead, 0.0),
        "vol-target unit-median": vt,
        "kelly-ish s/var(R)": kelly,
        "unit-median VRP": um,
        "long-short volatility": pos,
        "always short": pd.Series(-1.0, index=px.index),
    }


def rule_row(r: pd.Series, size: pd.Series) -> pd.Series:
    r = pd.Series(r).astype(float)
    size = pd.Series(size, index=r.index).astype(float)
    x = r.dropna()
    n = int(len(x))
    mu = float(x.mean()) if n else float("nan")
    sd = float(x.std(ddof=1)) if n >= 2 else float("nan")
    n_buy = int((size > 0).sum())
    n_sz = int(size.notna().sum())
    return pd.Series(
        {
            "n": n,
            "mean": mu,
            "std": sd,
            "min": float(x.min()) if n else float("nan"),
            "25%": float(x.quantile(0.25)) if n else float("nan"),
            "50%": float(x.median()) if n else float("nan"),
            "75%": float(x.quantile(0.75)) if n else float("nan"),
            "max": float(x.max()) if n else float("nan"),
            "skew": float(x.skew()) if n else float("nan"),
            "ex_kurt": float(x.kurt()) if n else float("nan"),
            "t_mean": mu / sd * np.sqrt(n) if (sd and sd > 0) else float("nan"),
            "Sharpe_ann": mu / sd * np.sqrt(252.0) if (sd and sd > 0) else float("nan"),
            "n_buy": n_buy,
            "pct_buy": 100.0 * n_buy / n_sz if n_sz else float("nan"),
        }
    )


def information_ratio(r_port: pd.Series, r_bench: pd.Series) -> pd.Series:
    """IR on active return R_a = R^p - R_benchmark."""
    a = r_port.astype(float).align(r_bench.astype(float), join="inner")
    active = a[0] - a[1]
    x = active.dropna()
    n = int(len(x))
    mu = float(x.mean()) if n else float("nan")
    sd = float(x.std(ddof=1)) if n >= 2 else float("nan")
    return pd.Series(
        {
            "n": n,
            "mean_active": mu,
            "te_daily": sd,
            "te_ann": sd * np.sqrt(252.0) if (sd and sd > 0) else float("nan"),
            "IR_ann": mu / sd * np.sqrt(252.0) if (sd and sd > 0) else float("nan"),
            "t_active": mu / sd * np.sqrt(n) if (sd and sd > 0) else float("nan"),
            "corr_to_bench": float(a[0].corr(a[1])) if n >= 3 else float("nan"),
        }
    )


def cboe_short_straddle_margin_points(
    S: float, K_c: float, K_p: float, premium: float
) -> float:
    """CBOE-style strategy-based margin for a short index straddle, index points."""
    otm = min(abs(float(S) - float(K_c)), abs(float(S) - float(K_p)))
    a = 0.15 * float(S) - otm + float(premium)
    b = 0.10 * float(S) + float(premium)
    return float(max(a, b, 0.0))


def crossed_premium_return(
    q: pd.Series, exit_: pd.Series, bid: pd.Series, ask: pd.Series
) -> pd.Series:
    q = q.astype(float)
    exit_ = exit_.astype(float)
    bid = bid.astype(float)
    ask = ask.astype(float)
    long_r = exit_ / ask - 1.0
    short_r = 1.0 - exit_ / bid
    return pd.Series(np.where(q >= 0, long_r, short_r), index=q.index)


def points_pnl(q: pd.Series, exit_: pd.Series, entry: pd.Series) -> pd.Series:
    return q.astype(float) * (exit_.astype(float) - entry.astype(float))


def stamp_spot(live: pd.DataFrame, keys) -> pd.Series:
    """NaN-safe copy of underlying_price at each key. Not a consensus.

    After dropna, this tape has one value per stamp. ``first`` is that
    value; all-NaN keys are absent from the index.
    """
    if isinstance(keys, str):
        keys = [keys]
    return (
        live.dropna(subset=["underlying_price"])
        .groupby(list(keys))["underlying_price"]
        .first()
    )


def attach_iv_hourly_as_30min(atm: pd.DataFrame) -> pd.DataFrame:
    out = atm.copy()
    out["iv_c"] = pd.to_numeric(out["impl_volatility_c"], errors="coerce")
    out["iv_p"] = pd.to_numeric(out["impl_volatility_p"], errors="coerce")
    out["iv_hourly"] = out[["iv_c", "iv_p"]].mean(axis=1)
    out["iv_30"] = out["iv_hourly"] / np.sqrt(2.0)
    out["iv_var"] = out["iv_30"] ** 2
    return out


def iv_var_from_conventions(
    iv_raw: pd.Series, hours_remaining: float
) -> dict[str, pd.Series]:
    iv = pd.to_numeric(iv_raw, errors="coerce")
    hrs = float(hours_remaining)
    return {
        "chris_hourly": (iv**2) * hrs,
        "annualized_om": (iv**2) * hrs / (252.0 * 6.5),
        "already_window": iv**2,
    }


def pick_nearest_otm(live: pd.DataFrame, spot: pd.Series) -> pd.DataFrame:
    c = live[live["cp"] == "C"].copy()
    p = live[live["cp"] == "P"].copy()
    c["S"] = c["expiration"].map(spot)
    p["S"] = p["expiration"].map(spot)
    c = c[np.isfinite(c["S"])]
    p = p[np.isfinite(p["S"])]
    c_otm = c[c["strike"] >= c["S"]].copy()
    c_otm["k_gap"] = c_otm["strike"].astype(float) - c_otm["S"]
    c_pick = (
        c_otm.sort_values(["expiration", "k_gap", "strike"])
        .groupby("expiration", as_index=False)
        .first()
    )
    p_otm = p[p["strike"] <= p["S"]].copy()
    p_otm["k_gap"] = p_otm["S"] - p_otm["strike"].astype(float)
    p_pick = (
        p_otm.sort_values(["expiration", "k_gap", "strike"])
        .groupby("expiration", as_index=False)
        .first()
    )
    atm = c_pick.merge(p_pick, on="expiration", suffixes=("_c", "_p"))
    atm["S"] = atm["S_c"].astype(float)
    atm["K_c"] = atm["strike_c"].astype(float)
    atm["K_p"] = atm["strike_p"].astype(float)
    atm["entry"] = atm["mid_c"].astype(float) + atm["mid_p"].astype(float)
    atm["bid_entry"] = atm["bid_c"].astype(float) + atm["bid_p"].astype(float)
    atm["ask_entry"] = atm["ask_c"].astype(float) + atm["ask_p"].astype(float)
    atm["same_strike"] = atm["K_c"] == atm["K_p"]
    return atm


def pick_wings(
    live: pd.DataFrame, body: pd.DataFrame, width: float = 25.0
) -> pd.DataFrame:
    c = live[live["cp"] == "C"].copy()
    p = live[live["cp"] == "P"].copy()
    want_c = body[["expiration", "K_c"]].copy()
    want_p = body[["expiration", "K_p"]].copy()
    c = c.merge(want_c, on="expiration", how="inner")
    p = p.merge(want_p, on="expiration", how="inner")
    c = c[c["strike"].astype(float) >= (c["K_c"] + width)]
    p = p[p["strike"].astype(float) <= (p["K_p"] - width)]
    c["k_gap"] = c["strike"].astype(float) - c["K_c"]
    p["k_gap"] = p["K_p"] - p["strike"].astype(float)
    c_w = (
        c.sort_values(["expiration", "k_gap", "strike"])
        .groupby("expiration", as_index=False)
        .first()
    )
    p_w = (
        p.sort_values(["expiration", "k_gap", "strike"])
        .groupby("expiration", as_index=False)
        .first()
    )
    wings = c_w.merge(p_w, on="expiration", suffixes=("_cw", "_pw"))
    # Keep only wing quotes so body.K_c / body.K_p / body.entry survive the merge.
    wing_cols = ["expiration", "strike_cw", "strike_pw", "mid_cw", "mid_pw"]
    extra = [c for c in ("bid_cw", "ask_cw", "bid_pw", "ask_pw") if c in wings.columns]
    out = body.merge(wings[wing_cols + extra], on="expiration", how="inner")
    out["K_c_wing"] = out["strike_cw"].astype(float)
    out["K_p_wing"] = out["strike_pw"].astype(float)
    out["mid_c_wing"] = out["mid_cw"].astype(float)
    out["mid_p_wing"] = out["mid_pw"].astype(float)
    out["entry_body"] = out["entry"].astype(float)
    out["entry_wings"] = out["mid_c_wing"] + out["mid_p_wing"]
    out["entry_ic"] = out["entry_body"] - out["entry_wings"]
    out["width"] = np.minimum(
        out["K_c_wing"] - out["K_c"], out["K_p"] - out["K_p_wing"]
    )
    return out


def settle_package(df: pd.DataFrame, s_close: pd.Series) -> pd.DataFrame:
    out = df.copy()
    exp_day = pd.to_datetime(out["expiration"])
    if getattr(exp_day.dt, "tz", None) is not None:
        exp_day = exp_day.dt.tz_convert("America/New_York").dt.tz_localize(None)
    exp_day = exp_day.dt.normalize()
    out["S_close"] = exp_day.map(s_close)
    out["pay_c"] = np.maximum(out["S_close"] - out["K_c"], 0.0)
    out["pay_p"] = np.maximum(out["K_p"] - out["S_close"], 0.0)
    out["exit"] = out["pay_c"] + out["pay_p"]
    if "K_c_wing" in out.columns:
        out["pay_c_wing"] = np.maximum(out["S_close"] - out["K_c_wing"], 0.0)
        out["pay_p_wing"] = np.maximum(out["K_p_wing"] - out["S_close"], 0.0)
        out["exit_wings"] = out["pay_c_wing"] + out["pay_p_wing"]
        out["exit_ic"] = out["exit"] - out["exit_wings"]
    return out


def fomc_and_monthend(index: pd.DatetimeIndex, repo: Path) -> pd.DataFrame:
    days = pd.DatetimeIndex(pd.to_datetime(index)).tz_localize(None).normalize()
    flags = pd.DataFrame({"is_me": False, "is_fomc": False}, index=days)
    month_end = pd.Series(days).groupby([days.year, days.month]).transform("max")
    flags["is_me"] = days.isin(pd.DatetimeIndex(month_end))
    rel = repo / "data" / "releases.parquet"
    fomc: set[pd.Timestamp] = set()
    if rel.exists():
        r = pd.read_parquet(rel)
        col = next(
            (
                c
                for c in r.columns
                if str(c).lower() in {"fomc release", "fomc", "fomc_release"}
            ),
            None,
        )
        date_col = "date" if "date" in r.columns else r.columns[0]
        if col is not None:
            dts = pd.to_datetime(r.loc[r[col].astype(bool), date_col], errors="coerce")
            dts = dts.dt.tz_localize(None).dt.normalize()
            fomc |= set(dts.dropna())
    for s in ("2020-03-03", "2023-12-13", "2024-01-31", "2024-03-20"):
        fomc.add(pd.Timestamp(s))
    flags["is_fomc"] = days.isin(list(fomc))
    flags["is_event"] = flags["is_me"] | flags["is_fomc"]
    flags.index = index
    return flags
