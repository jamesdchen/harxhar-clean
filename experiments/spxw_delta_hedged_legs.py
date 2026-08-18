"""Delta-hedged single-leg 0DTE strategy per the research spec.

Spec (Aug 2026): SPX 0DTE, 30-min periods, near-ATM calls and puts
SEPARATELY (not straddles) so mispricing can be seen leg by leg;
delta-hedged; at every period t+kD compute the price-form VRP
ln(IV_ttc / sqrt(RV_hat_ttc)) from a time-till-close IV measurement and a
time-till-close RV forecast that updates each period; sell+hedge if
sufficiently positive, buy+hedge if sufficiently negative, flat near zero.

Implementation on disk data:
  * Contracts: every 30-min bar, calls and puts with |BS delta| in
    [0.3, 0.7] (near-ATM band; documented, not tuned), both sides
    quoted (bid > 0).
  * IV_ttc: Black-Scholes implied vol from the mid, tau = hours to
    16:00 settle (r = 0 over a few hours). BS is sufficient here
    because the payoff is settled at intrinsic (model-free); BS is only
    the quoting map from price to a comparable time-till-close vol and
    the source of the hedge ratio, a second-order term.
  * RV_hat_ttc: remaining-variance forecast at the bar, reverse cumsum
    of the per-bar model forecasts (blk2; a0 for the swap test),
    causally calibrated per entry hour (expanding QLIKE multiplier,
    min 63 days, shifted), converted to the same annualized vol unit.
  * Signal: s = ln(IV_ttc / sqrt(RVhat_ttc)); dead-zone theta on a
    documented grid {0, 0.05, 0.10, 0.20}: short if s > theta, long if
    s < -theta, flat otherwise.
  * Hedge: BS delta at entry, rebalanced at each subsequent 30-min bar
    on the chain's underlying_price, to settlement. Option settled at
    intrinsic. PnL normalized by entry mid. Crossed variant: option at
    bid/ask, underlying charged 0.5 bp per rebalance (documented).
  * Controls: always-short, always-long. Swap test: identical
    machinery with a0 replacing blk2, paired daily difference t.

Outputs results/spxw_pnl/dh_legs_{summary,by_hour}.csv and the ledger
dh_legs_ledger.parquet.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "spxw_pnl")
ANN = float(np.sqrt(252.0))
BURN = 63
DELTA_LO, DELTA_HI = 0.30, 0.70
THETAS = (0.0, 0.05, 0.10, 0.20)
UNDERLYING_COST_BP = 0.5
DAILY_0DTE = pd.Timestamp("2022-05-16")
HOURS_PER_YEAR = 252.0 * 6.5


def _sh(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 3 or float(x.std()) == 0.0:
        return float("nan")
    return float(x.mean() / x.std())


def _tstat(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 3 or float(x.std(ddof=1)) == 0.0:
        return float("nan")
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(x.size)))


def bs_price_v(S, K, T, sig, call):
    """Vectorized Black-Scholes (r=0). Arrays broadcast; call is bool array."""
    S = np.asarray(S, float)
    K = np.asarray(K, float)
    T = np.asarray(T, float)
    sig = np.asarray(sig, float)
    call = np.asarray(call, bool)
    intr = np.where(call, np.maximum(S - K, 0.0), np.maximum(K - S, 0.0))
    ok = (T > 0) & (sig > 0)
    sq = np.sqrt(np.where(ok, T, 1.0))
    sg = np.where(ok, sig, 1.0)
    d1 = (np.log(S / K) + 0.5 * sg * sg * np.where(ok, T, 1.0)) / (sg * sq)
    d2 = d1 - sg * sq
    c = S * norm.cdf(d1) - K * norm.cdf(d2)
    pv = K * norm.cdf(-d2) - S * norm.cdf(-d1)
    out = np.where(call, c, pv)
    return np.where(ok, out, intr)


def bs_delta_v(S, K, T, sig, call):
    S = np.asarray(S, float)
    K = np.asarray(K, float)
    T = np.asarray(T, float)
    sig = np.asarray(sig, float)
    call = np.asarray(call, bool)
    ok = (T > 0) & (sig > 0)
    sq = np.sqrt(np.where(ok, T, 1.0))
    sg = np.where(ok, sig, 1.0)
    d1 = (np.log(S / K) + 0.5 * sg * sg * np.where(ok, T, 1.0)) / (sg * sq)
    nd1 = norm.cdf(d1)
    dl = np.where(call, nd1, nd1 - 1.0)
    expired = np.where(call, (S > K).astype(float), -(S < K).astype(float))
    return np.where(ok, dl, expired)


def bs_vega_v(S, K, T, sig):
    S = np.asarray(S, float)
    K = np.asarray(K, float)
    T = np.asarray(T, float)
    sig = np.asarray(sig, float)
    ok = (T > 0) & (sig > 0)
    sq = np.sqrt(np.where(ok, T, 1.0))
    sg = np.where(ok, sig, 1.0)
    d1 = (np.log(S / K) + 0.5 * sg * sg * np.where(ok, T, 1.0)) / (sg * sq)
    return np.where(ok, S * norm.pdf(d1) * sq, 0.0)


def bs_iv_v(price, S, K, T, call, iters: int = 60):
    """Vectorized IV: bracketed bisection (robust) then Newton polish.

    Returns NaN where price <= intrinsic or T <= 0.
    """
    price = np.asarray(price, float)
    S = np.asarray(S, float)
    K = np.asarray(K, float)
    T = np.asarray(T, float)
    call = np.asarray(call, bool)
    intr = np.where(call, np.maximum(S - K, 0.0), np.maximum(K - S, 0.0))
    valid = (T > 0) & (price > intr + 1e-9)
    lo = np.full(price.shape, 1e-4)
    hi = np.full(price.shape, 5.0)
    # 30 bisection steps -> 5/2^30 precision, ample; then 3 Newton steps
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        pm = bs_price_v(S, K, T, mid, call)
        go_up = pm < price
        lo = np.where(go_up, mid, lo)
        hi = np.where(go_up, hi, mid)
    sig = 0.5 * (lo + hi)
    for _ in range(3):
        v = bs_vega_v(S, K, T, sig)
        pm = bs_price_v(S, K, T, sig, call)
        step = np.where(v > 1e-12, (pm - price) / np.where(v > 1e-12, v, 1.0), 0.0)
        sig = np.clip(sig - step, 1e-4, 5.0)
    return np.where(valid, sig, np.nan)


def bs_delta(S: float, K: float, T: float, sig: float, call: bool) -> float:
    return float(bs_delta_v(S, K, T, sig, call))


def _settle_utc(exp_day: pd.Timestamp, sample_t: pd.Timestamp) -> pd.Timestamp:
    """16:00 ET on the expiration day, expressed in UTC (DST-aware via ET)."""
    et = pd.Timestamp(exp_day.date()).tz_localize("America/New_York") + pd.Timedelta(
        hours=16
    )
    return et.tz_convert("UTC")


def main() -> None:
    eb = pd.read_parquet(os.path.join(OUT, "everybar_mtm_trades.parquet"))
    eb = eb.sort_values(["day", "et"]).reset_index(drop=True)
    g = eb.groupby("day", sort=False)
    eb["rv_rem"] = g["rv"].transform(lambda s: s.iloc[::-1].cumsum().iloc[::-1])
    eb["b2_rem"] = g["pb"].transform(lambda s: s.iloc[::-1].cumsum().iloc[::-1])
    eb["a0_rem"] = g["pa"].transform(lambda s: s.iloc[::-1].cumsum().iloc[::-1])
    eb["hhmm"] = eb["et"].dt.strftime("%H:%M")
    eb["expiration"] = pd.to_datetime(eb["expiration"])
    for tag in ("b2_rem", "a0_rem"):
        eb[f"{tag}_cal"] = np.nan
        for _, gh in eb.groupby("hhmm"):
            gh = gh.sort_values("expiration")
            r = gh["rv_rem"].to_numpy(float) / np.maximum(
                gh[tag].to_numpy(float), 1e-18
            )
            cal = (
                pd.Series(r).expanding(min_periods=BURN).mean().shift(1).to_numpy(float)
            )
            eb.loc[gh.index, f"{tag}_cal"] = cal * gh[tag].to_numpy(float)
    eb = eb.dropna(subset=["b2_rem_cal", "a0_rem_cal"])

    ch = pd.read_parquet(
        os.path.join(ROOT, "data", "spxw_chain.parquet"),
        columns=[
            "expiration",
            "timestamp",
            "strike",
            "cp",
            "bid",
            "ask",
            "mid",
            "underlying_price",
        ],
    )
    ch = ch[(ch["bid"] > 0) & (ch["mid"] > 0) & np.isfinite(ch["underlying_price"])]
    ch = ch.rename(columns={"timestamp": "t"})
    ch["expiration"] = pd.to_datetime(ch["expiration"])
    path = ch.groupby(["expiration", "t"])["underlying_price"].first().reset_index()
    tr = pd.read_parquet(
        os.path.join(OUT, "mfiv_toclose_trades.parquet"), columns=["expiration", "S_T"]
    )
    tr["expiration"] = pd.to_datetime(tr["expiration"])
    settle = dict(zip(tr["expiration"], tr["S_T"]))

    keys = eb[["expiration", "t"]].drop_duplicates()
    ch = ch.merge(keys, on=["expiration", "t"], how="inner")
    ch = ch.merge(
        eb[["expiration", "t", "hhmm", "b2_rem_cal", "a0_rem_cal", "rv_rem"]],
        on=["expiration", "t"],
        how="inner",
    )
    ch = ch[ch["expiration"].isin(settle.keys())].copy()
    print(f"candidate contracts: {len(ch)}", flush=True)

    # tau to 16:00 ET settle, DST-aware
    t_utc = pd.to_datetime(ch["t"], utc=True).dt.tz_convert(None)
    settle_naive = pd.DatetimeIndex(
        [_settle_utc(d, d) for d in ch["expiration"]]
    ).tz_convert(None)
    hrs = (
        settle_naive.to_numpy(dtype="datetime64[ns]")
        - t_utc.to_numpy(dtype="datetime64[ns]")
    ) / np.timedelta64(1, "h")
    ch["tau"] = np.maximum(hrs.astype(float), 0.0) / HOURS_PER_YEAR
    S = ch["underlying_price"].to_numpy(float)
    K = ch["strike"].to_numpy(float)
    call = (ch["cp"] == "C").to_numpy()
    mid = ch["mid"].to_numpy(float)
    tau = ch["tau"].to_numpy(float)
    ch["iv"] = bs_iv_v(mid, S, K, tau, call)
    ch = ch[np.isfinite(ch["iv"])].copy()
    ch["delta"] = bs_delta_v(
        ch["underlying_price"].to_numpy(float),
        ch["strike"].to_numpy(float),
        ch["tau"].to_numpy(float),
        ch["iv"].to_numpy(float),
        (ch["cp"] == "C").to_numpy(),
    )
    ch = ch[ch["delta"].abs().between(DELTA_LO, DELTA_HI)].copy()
    print(f"near-ATM contracts with IV: {len(ch)}", flush=True)

    for tag in ("b2", "a0"):
        ch[f"mvol_{tag}"] = np.sqrt(
            np.maximum(ch[f"{tag}_rem_cal"].to_numpy(float), 1e-18)
            / np.maximum(ch["tau"].to_numpy(float), 1e-9)
        )
        ch[f"sig_{tag}"] = np.log(ch["iv"] / ch[f"mvol_{tag}"])

    S_T = ch["expiration"].map(settle).to_numpy(float)
    intr = np.where(
        ch["cp"] == "C",
        np.maximum(S_T - ch["strike"], 0.0),
        np.maximum(ch["strike"] - S_T, 0.0),
    )
    ch["S_T"] = S_T
    ch["intr"] = intr

    Karr = ch["strike"].to_numpy(float)
    carr = (ch["cp"] == "C").to_numpy()
    varr = ch["iv"].to_numpy(float)
    # Vectorized hedge: for each contract, at each later 30-min stamp j on its
    # day, delta_j from entry IV; pnl += -delta_j * (S_{j+1} - S_j), last step
    # to settlement S_T. Build a wide (contract x stamp-of-day) underlying grid.
    stamps = sorted(ch["t"].unique())
    day_grid = path.pivot(index="expiration", columns="t", values="underlying_price")
    day_grid = day_grid.reindex(columns=stamps)
    # per-contract row of underlying path from its own day
    grid = day_grid.loc[ch["expiration"].to_numpy()].to_numpy(float)  # (n, nstamps)
    t_idx = pd.Index(stamps).get_indexer(ch["t"].to_numpy())
    n, m = grid.shape
    col = np.arange(m)[None, :]
    active = col >= t_idx[:, None]  # stamps at/after entry
    grid = np.where(active, grid, np.nan)
    # next-price: shift left, last active gets S_T
    nxt = np.full_like(grid, np.nan)
    nxt[:, :-1] = grid[:, 1:]
    last_active = np.where(active, col, -1).max(axis=1)
    # positions with no next active stamp -> settlement
    S_T_col = ch["S_T"].to_numpy(float)
    for j in range(m):
        take = last_active == j
        nxt[take, j] = S_T_col[take]
    # NaN gaps inside a day (missing stamp): treat as no-rebalance, carry to
    # next available price
    for j in range(m - 2, -1, -1):
        gap = active[:, j] & np.isnan(nxt[:, j])
        nxt[gap, j] = nxt[gap, j + 1]
    # hours-to-settle per bar-of-day is the same every day on the ET clock
    bod_hours = np.array([int(b[:2]) + int(b[3:]) / 60.0 for b in stamps])
    hrs_row = np.maximum(16.0 - bod_hours, 0.0)
    tau_grid = np.broadcast_to(hrs_row[None, :] / HOURS_PER_YEAR, grid.shape)
    Sg = np.where(np.isnan(grid), 1.0, grid)
    dl = bs_delta_v(Sg, Karr[:, None], tau_grid, varr[:, None], carr[:, None])
    dl = np.where(active & ~np.isnan(grid), dl, 0.0)
    dS = np.where(active & ~np.isnan(grid) & ~np.isnan(nxt), nxt - grid, 0.0)
    hedge = (-dl * dS).sum(axis=1)
    prev = np.zeros_like(dl)
    prev[:, 1:] = dl[:, :-1]
    turnover = np.abs(dl - prev) * np.where(np.isnan(grid), 0.0, grid)
    hedge_cost = (turnover * UNDERLYING_COST_BP * 1e-4).sum(axis=1)
    ch["hedge"] = hedge
    ch["hedge_cost"] = hedge_cost
    ch = ch[np.isfinite(ch["hedge"])].copy()
    prem = ch["mid"].to_numpy(float)
    ch["long_dh_mid"] = (ch["intr"] - ch["mid"] + ch["hedge"]) / prem
    ch["long_dh_x"] = (ch["intr"] - ch["ask"] + ch["hedge"] - ch["hedge_cost"]) / prem
    ch["short_dh_x"] = (ch["bid"] - ch["intr"] - ch["hedge"] - ch["hedge_cost"]) / prem
    ch.to_parquet(os.path.join(OUT, "dh_legs_ledger.parquet"))

    def daily(x: np.ndarray, m: np.ndarray, idx: np.ndarray) -> np.ndarray:
        s = pd.Series(np.where(m, x, np.nan), index=idx)
        return s.groupby(level=0).mean().dropna().to_numpy()

    rows = []
    hours = []
    for era, sub in (("all", ch), ("daily_0dte", ch[ch["expiration"] >= DAILY_0DTE])):
        for leg, sl in (
            ("call", sub[sub["cp"] == "C"]),
            ("put", sub[sub["cp"] == "P"]),
            ("both", sub),
        ):
            idx = sl["expiration"].to_numpy()
            long_mid = sl["long_dh_mid"].to_numpy(float)
            lx = sl["long_dh_x"].to_numpy(float)
            sx = sl["short_dh_x"].to_numpy(float)
            for th in THETAS:
                for tag in ("b2", "a0"):
                    sig = sl[f"sig_{tag}"].to_numpy(float)
                    pos = np.where(sig > th, -1.0, np.where(sig < -th, 1.0, 0.0))
                    pnl_mid = pos * long_mid
                    pnl_x = np.where(pos > 0, lx, np.where(pos < 0, sx, np.nan))
                    traded = pos != 0
                    dm = daily(pnl_mid, traded, idx)
                    dx = daily(pnl_x, traded, idx)
                    rows.append(
                        {
                            "era": era,
                            "leg": leg,
                            "theta": th,
                            "model": tag,
                            "n_contracts": int(traded.sum()),
                            "n_days": int(dm.size),
                            "frac_traded": float(traded.mean()),
                            "frac_long": float((pos > 0).mean()),
                            "sh_daily_mid": _sh(dm) * ANN,
                            "sh_daily_crossed": _sh(dx) * ANN,
                            "mean_per_trade_mid": float(np.nanmean(pnl_mid[traded]))
                            if traded.any()
                            else float("nan"),
                            "hit_mid": float((pnl_mid[traded] > 0).mean())
                            if traded.any()
                            else float("nan"),
                        }
                    )
                sb = sl["sig_b2"].to_numpy(float)
                sa = sl["sig_a0"].to_numpy(float)
                pb = np.where(sb > th, -1.0, np.where(sb < -th, 1.0, 0.0)) * long_mid
                pa = np.where(sa > th, -1.0, np.where(sa < -th, 1.0, 0.0)) * long_mid
                dd = daily(pb - pa, np.ones(len(sl), bool), idx)
                rows.append(
                    {
                        "era": era,
                        "leg": leg,
                        "theta": th,
                        "model": "b2_minus_a0_t",
                        "n_contracts": len(sl),
                        "n_days": int(dd.size),
                        "frac_traded": float("nan"),
                        "frac_long": float("nan"),
                        "sh_daily_mid": _tstat(dd),
                        "sh_daily_crossed": float("nan"),
                        "mean_per_trade_mid": float(np.nanmean(pb - pa)),
                        "hit_mid": float("nan"),
                    }
                )
            for name, sgn in (("always_short", -1.0), ("always_long", 1.0)):
                pnl = sgn * long_mid
                dm = daily(pnl, np.ones(len(sl), bool), idx)
                rows.append(
                    {
                        "era": era,
                        "leg": leg,
                        "theta": float("nan"),
                        "model": name,
                        "n_contracts": len(sl),
                        "n_days": int(dm.size),
                        "frac_traded": 1.0,
                        "frac_long": float(sgn > 0),
                        "sh_daily_mid": _sh(dm) * ANN,
                        "sh_daily_crossed": float("nan"),
                        "mean_per_trade_mid": float(np.mean(pnl)),
                        "hit_mid": float((pnl > 0).mean()),
                    }
                )
        for hh, gh in sub.groupby("hhmm"):
            lm = gh["long_dh_mid"].to_numpy(float)
            gidx = gh["expiration"].to_numpy()
            rec = {
                "era": era,
                "entry": hh,
                "n": len(gh),
                "vrp_med": float(gh["sig_b2"].median()),
                "frac_long_b2": float((gh["sig_b2"] < -0.05).mean()),
                "sh_always_short": _sh(daily(-lm, np.ones(len(gh), bool), gidx)) * ANN,
            }
            for tag in ("b2", "a0"):
                sig = gh[f"sig_{tag}"].to_numpy(float)
                pos = np.where(sig > 0.05, -1.0, np.where(sig < -0.05, 1.0, 0.0))
                rec[f"sh_{tag}"] = _sh(daily(pos * lm, pos != 0, gidx)) * ANN
            hours.append(rec)

    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(OUT, "dh_legs_summary.csv"), index=False)
    byh = pd.DataFrame(hours)
    byh.to_csv(os.path.join(OUT, "dh_legs_by_hour.csv"), index=False)
    pd.set_option("display.width", 220)
    print(summ.to_string(index=False), flush=True)
    print(byh.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
