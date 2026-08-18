"""Trading the bumps: 0DTE SPX smile-anomaly relative value.

This is the *other* strategy in the research spec, and it is deliberately
disjoint from the RV-forecast "value" book (spxw_delta_hedged_legs.py,
spxw_atm_straddle_bars.py).  There the signal is a level statement --
implied vol against a remaining-variance forecast -- and the book is
long or short volatility as a whole.  Here nothing is forecast about
realized variance at all.  The signal is purely *relative*: at each
30-min snapshot of the 0DTE chain, fit a smooth smile across the live
strikes and ask which individual contracts sit off it.  A contract whose
implied vol is anomalously LOW versus its own neighbours on the smile
(and versus its own earlier prints today) is bought; one that is
anomalously HIGH is sold.  The bump is expected to revert; if it does
not, the position is carried to 16:00 settlement and paid intrinsic.

Everything is delta-hedged with the same rebalanced Black-Scholes hedge
loop used by the value book, so the book expresses a view on the *shape*
of the surface, not on its level.  A premium-neutral variant additionally
nets the long leg against the short leg snapshot by snapshot, which
removes the residual level exposure that survives an unbalanced count of
longs and shorts.

Construction (all constants below are documented, fixed, and were never
selected on outcomes):

  * Universe: every quoted 0DTE contract (bid > 0, mid > 0) on a day for
    which a settlement close S_T exists in mfiv_toclose_trades.parquet.
  * IV: own Black-Scholes solve from the mid, r = 0, tau = hours to
    16:00 ET settle (DST-aware) / (252 * 6.5).  The vendor
    impl_volatility column is a per-period unit and is not used.  The
    solve is a vectorized bisection on [1e-4, 5.0], 60 halvings; a quote
    at or below intrinsic, or above the sig = 5 cap, yields NaN.
  * Trading universe: |BS delta| in [0.10, 0.90] and IV finite.
  * Cross-sectional score: per (expiration, timestamp) and side, OLS of
    IV on [1, x, x^2] with x = ln(K/S), equally weighted, on that
    snapshot's strikes; residual r_i, and z_i = r_i / sigma_hat with
    sigma_hat = sqrt(RSS / (n - 3)) -- the residual std at the regression
    dof, so the flag rate does not drift with the number of live strikes.
    Fitted three ways: calls only, puts only, and pooled across both.
    Needs >= 6 strikes in the fit.
  * Temporal score: z_time = (IV_i - EWMA of the SAME strike/side's IV
    over strictly earlier stamps today) / (expanding std of those same
    earlier prints), span 5, and only where >= 3 prior prints exist.  The
    history is taken from the full finite-IV panel, not the delta-banded
    trading universe, so a strike that walks into the band still carries
    its own morning.
  * Books: for zt in {1.0, 1.5, 2.0}, long every contract with z < -zt
    and short every contract with z > +zt, per snapshot.  Held to
    settlement (intrinsic at S_T), delta-hedged at the contract's own
    entry IV, rebalanced at each subsequent 30-min stamp on the chain's
    underlying_price.  PnL is normalized by the entry mid.  Mids book:
    option at mid, hedge free.  Crossed book: long pays the ask, short
    receives the bid, and the underlying is charged 0.5 bp of notional
    per rebalance.
  * Premium-neutral variant: one number per snapshot, the mean signed
    pnl over the longs plus the mean signed pnl over the shorts -- i.e.
    the long leg minus the short leg with equal premium on each side,
    regardless of how many contracts each leg happens to hold.  Requires
    both legs non-empty.
  * Controls: always-short-all and always-long-all over the near-ATM
    band |delta| in [0.30, 0.70], the smile-fit R^2 distribution, and
    the fraction of contracts flagged.

Causality: the smile fit uses only the current snapshot, and the
temporal score only strictly earlier stamps of the same day.  Nothing
crosses a day boundary and nothing looks forward.

Outputs results/spxw_pnl/iv_bumps_{summary,by_hour,diagnostics}.csv and
the per-contract ledger iv_bumps_ledger.parquet.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "spxw_pnl")
ANN = float(np.sqrt(252.0))
HOURS_PER_YEAR = 252.0 * 6.5
DAILY_0DTE = pd.Timestamp("2022-05-16")

# --- documented constants (fixed a priori, never tuned on outcomes) ---
DELTA_LO, DELTA_HI = 0.10, 0.90  # trading universe
ATM_LO, ATM_HI = 0.30, 0.70  # control band
ZT_GRID = (1.0, 1.5, 2.0)
MIN_SMILE_N = 6  # strikes needed for the quadratic fit
SMILE_DOF = 3  # params in the quadratic smile
EWMA_SPAN = 5.0
EWMA_MIN_PRIOR = 3
IV_LO, IV_HI = 1e-4, 5.0
IV_ITERS = 60
UNDERLYING_COST_BP = 0.5
BY_HOUR_ZT = 1.5
CHUNK = 500_000


def _sh(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 3 or float(x.std()) == 0.0:
        return float("nan")
    return float(x.mean() / x.std())


def _hit(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float((x > 0).mean()) if x.size else float("nan")


def _mean(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else float("nan")


def bs_price_vec(
    S: np.ndarray, K: np.ndarray, T: np.ndarray, sig: np.ndarray, call: np.ndarray
) -> np.ndarray:
    """Black-Scholes with r = 0 (so forward = spot over a few hours)."""
    sq = sig * np.sqrt(np.maximum(T, 0.0))
    sqs = np.where(sq > 0.0, sq, 1e-12)
    d1 = (np.log(S / K) + 0.5 * sqs * sqs) / sqs
    d2 = d1 - sqs
    c = S * norm.cdf(d1) - K * norm.cdf(d2)
    return np.asarray(np.where(call, c, c - S + K), float)


def bs_delta_vec(
    S: np.ndarray, K: np.ndarray, T: np.ndarray, sig: np.ndarray, call: np.ndarray
) -> np.ndarray:
    sq = sig * np.sqrt(np.maximum(T, 0.0))
    live = np.isfinite(sq) & (sq > 0.0)
    sqs = np.where(live, sq, 1.0)
    d1 = (np.log(S / K) + 0.5 * sqs * sqs) / sqs
    dl = np.where(call, norm.cdf(d1), norm.cdf(d1) - 1.0)
    # expiry / zero-vol limit: the option is its own intrinsic indicator
    exp_dl = np.where(call, (S > K).astype(float), -(S < K).astype(float))
    return np.asarray(np.where(live, dl, exp_dl), float)


def bs_iv_vec(
    price: np.ndarray, S: np.ndarray, K: np.ndarray, T: np.ndarray, call: np.ndarray
) -> np.ndarray:
    """Vectorized bisection inverse of bs_price_vec on [IV_LO, IV_HI]."""
    n = price.size
    lo: np.ndarray = np.full(n, IV_LO)
    hi: np.ndarray = np.full(n, IV_HI)
    for _ in range(IV_ITERS):
        m = 0.5 * (lo + hi)
        f = bs_price_vec(S, K, T, m, call) - price
        up = f > 0.0
        hi = np.where(up, m, hi)
        lo = np.where(up, lo, m)
    iv = 0.5 * (lo + hi)
    intr = np.where(call, np.maximum(S - K, 0.0), np.maximum(K - S, 0.0))
    bad = (T <= 0.0) | (price <= intr + 1e-9) | (iv >= IV_HI - 1e-3)
    return np.asarray(np.where(bad, np.nan, iv), float)


def _settle_utc_naive(day: pd.Timestamp) -> pd.Timestamp:
    """16:00 ET on the expiration day as a tz-naive UTC stamp (DST-aware)."""
    et = pd.Timestamp(day).tz_localize("America/New_York") + pd.Timedelta(hours=16)
    return et.tz_convert("UTC").tz_localize(None)


def load_chain() -> tuple[pd.DataFrame, dict[str, float]]:
    tr = pd.read_parquet(
        os.path.join(OUT, "mfiv_toclose_trades.parquet"), columns=["expiration", "S_T"]
    )
    tr["expiration"] = pd.to_datetime(tr["expiration"])
    settle = dict(zip(tr["expiration"], tr["S_T"]))

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
    n_raw = len(ch)
    ch = ch[(ch["bid"] > 0) & (ch["mid"] > 0) & np.isfinite(ch["underlying_price"])]
    n_quoted = len(ch)
    ch["expiration"] = pd.to_datetime(ch["expiration"])
    ch = ch[ch["expiration"].isin(settle.keys())].copy()
    ch = ch.rename(columns={"timestamp": "t"})
    ch["t"] = pd.to_datetime(ch["t"], utc=True).dt.tz_localize(None)
    n_dup = int(ch.duplicated(["expiration", "t", "strike", "cp"]).sum())
    ch = ch.drop_duplicates(["expiration", "t", "strike", "cp"]).reset_index(drop=True)

    days = pd.DatetimeIndex(sorted(ch["expiration"].unique()))
    smap = {d: _settle_utc_naive(d) for d in days}
    st = ch["expiration"].map(smap).to_numpy().astype("datetime64[ns]")
    hrs = (st - ch["t"].to_numpy()) / np.timedelta64(1, "h")
    ch["tau"] = np.maximum(hrs.astype(float), 0.0) / HOURS_PER_YEAR
    ch["S_T"] = ch["expiration"].map(settle).astype(float)

    diag = {
        "rows_raw": float(n_raw),
        "rows_quoted": float(n_quoted),
        "rows_settled_days": float(len(ch)),
        "rows_dup_dropped": float(n_dup),
        "n_expirations": float(ch["expiration"].nunique()),
    }
    return ch, diag


def solve_iv(ch: pd.DataFrame, diag: dict[str, float]) -> pd.DataFrame:
    S = ch["underlying_price"].to_numpy(float)
    K = ch["strike"].to_numpy(float)
    T = ch["tau"].to_numpy(float)
    P = ch["mid"].to_numpy(float)
    call = (ch["cp"] == "C").to_numpy()
    intr_now = np.where(call, np.maximum(S - K, 0.0), np.maximum(K - S, 0.0))
    iv = np.full(len(ch), np.nan)
    for a in range(0, len(ch), CHUNK):
        b = min(a + CHUNK, len(ch))
        iv[a:b] = bs_iv_vec(P[a:b], S[a:b], K[a:b], T[a:b], call[a:b])
        print(f"  iv solve {b}/{len(ch)}", flush=True)
    ch["iv"] = iv
    ch["delta"] = bs_delta_vec(S, K, T, iv, call)
    diag["iv_fail_expired_tau0"] = float(np.mean(T <= 0.0))
    diag["iv_fail_at_or_below_intrinsic"] = float(np.mean(P <= intr_now + 1e-9))
    diag["iv_fail_total_frac"] = float(np.mean(~np.isfinite(iv)))
    diag["iv_finite_rows"] = float(np.isfinite(iv).sum())
    return ch[np.isfinite(ch["iv"])].copy()


def temporal_score(panel: pd.DataFrame) -> pd.DataFrame:
    """z_time from strictly earlier stamps of the same day, per strike/side."""
    out: list[pd.DataFrame] = []
    for day, g in panel.groupby("expiration", sort=True):
        w = g.pivot_table(index="t", columns=["cp", "strike"], values="iv").sort_index()
        prior = w.shift(1)
        ew = prior.ewm(span=EWMA_SPAN, min_periods=EWMA_MIN_PRIOR).mean()
        sd = prior.expanding(min_periods=EWMA_MIN_PRIOR).std()
        z = (w - ew) / sd.replace(0.0, np.nan)
        s = z.stack(["cp", "strike"], future_stack=True).rename("z_time")
        s = s.dropna().reset_index()
        s["expiration"] = day
        out.append(s)
    if not out:
        return pd.DataFrame(columns=["expiration", "t", "cp", "strike", "z_time"])
    return pd.concat(out, ignore_index=True)


def smile_scores(uni: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Quadratic-in-log-moneyness smile residual z-scores per snapshot."""
    uni = uni.reset_index(drop=True)
    uni["x"] = np.log(
        uni["strike"].to_numpy(float) / uni["underlying_price"].to_numpy(float)
    )
    z_cs = np.full(len(uni), np.nan)
    z_pool = np.full(len(uni), np.nan)
    fits: list[dict[str, object]] = []

    def _fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
        A = np.column_stack([np.ones_like(x), x, x * x])
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        res = y - A @ beta
        rss = float(res @ res)
        tss = float(((y - y.mean()) ** 2).sum())
        sig = np.sqrt(rss / max(len(x) - SMILE_DOF, 1))
        r2 = 1.0 - rss / tss if tss > 0 else np.nan
        if not np.isfinite(sig) or sig <= 0:
            return np.full(len(x), np.nan), r2
        return res / sig, r2

    for (day, t), g in uni.groupby(["expiration", "t"], sort=False):
        rows = g.index.to_numpy()
        xg = g["x"].to_numpy(float)
        yg = g["iv"].to_numpy(float)
        if len(g) >= MIN_SMILE_N:
            zz, r2 = _fit(xg, yg)
            z_pool[rows] = zz
            fits.append(
                {"expiration": day, "t": t, "side": "pooled", "n": len(g), "r2": r2}
            )
        for side in ("C", "P"):
            m = (g["cp"] == side).to_numpy()
            if int(m.sum()) < MIN_SMILE_N:
                continue
            zz, r2 = _fit(xg[m], yg[m])
            z_cs[rows[m]] = zz
            fits.append(
                {"expiration": day, "t": t, "side": side, "n": int(m.sum()), "r2": r2}
            )
    uni["z_cs"] = z_cs
    uni["z_pool"] = z_pool
    return uni, pd.DataFrame(fits)


def hedge_to_settle(uni: pd.DataFrame, path: pd.DataFrame) -> pd.DataFrame:
    """BS-delta hedge at the contract's entry IV, rebalanced every stamp."""
    hedge = np.full(len(uni), np.nan)
    cost = np.full(len(uni), np.nan)
    uni = uni.reset_index(drop=True)
    path_by_day = {d: g.sort_values("t") for d, g in path.groupby("expiration")}
    for day, g in uni.groupby("expiration", sort=False):
        p = path_by_day.get(day)
        if p is None:
            continue
        ts = p["t"].to_numpy()
        Sp = p["underlying_price"].to_numpy(float)
        M = len(Sp)
        S_T = float(g["S_T"].iloc[0])
        settle = _settle_utc_naive(day).to_datetime64()
        tau_j = (
            np.maximum(((settle - ts) / np.timedelta64(1, "h")).astype(float), 0.0)
            / HOURS_PER_YEAR
        )
        dS = np.empty(M)
        dS[: M - 1] = Sp[1:] - Sp[: M - 1]
        dS[M - 1] = S_T - Sp[M - 1]

        j0 = np.searchsorted(ts, g["t"].to_numpy())
        rows = g.index.to_numpy()
        K = g["strike"].to_numpy(float)[:, None]
        sig = g["iv"].to_numpy(float)[:, None]
        call = (g["cp"] == "C").to_numpy()[:, None]
        Sm = np.broadcast_to(Sp[None, :], (len(g), M))
        Tm = np.broadcast_to(tau_j[None, :], (len(g), M))
        dl = bs_delta_vec(
            Sm,
            np.broadcast_to(K, (len(g), M)),
            Tm,
            np.broadcast_to(sig, (len(g), M)),
            np.broadcast_to(call, (len(g), M)),
        )
        jj = np.arange(M)[None, :]
        live = jj >= j0[:, None]
        prev = np.zeros_like(dl)
        prev[:, 1:] = dl[:, :-1]
        prev = np.where(jj == j0[:, None], 0.0, prev)
        hedge[rows] = np.sum(np.where(live, -dl * dS[None, :], 0.0), axis=1)
        cost[rows] = np.sum(
            np.where(live, np.abs(dl - prev) * Sm * UNDERLYING_COST_BP * 1e-4, 0.0),
            axis=1,
        )
    uni["hedge"] = hedge
    uni["hedge_cost"] = cost
    return uni


def _daily(pnl: np.ndarray, mask: np.ndarray, day: np.ndarray) -> np.ndarray:
    s = pd.Series(np.where(mask, pnl, np.nan), index=day)
    return s.groupby(level=0).mean().dropna().to_numpy(float)


def _book_row(
    pnl_mid: np.ndarray,
    pnl_x: np.ndarray,
    pos: np.ndarray,
    day: np.ndarray,
) -> dict[str, float]:
    traded = pos != 0
    dm = _daily(pnl_mid, traded, day)
    dx = _daily(pnl_x, traded, day)
    return {
        "n_contracts": float(traded.sum()),
        "n_days": float(dm.size),
        "frac_flagged": float(traded.mean()) if traded.size else float("nan"),
        "frac_long": float((pos > 0).sum() / traded.sum())
        if traded.any()
        else float("nan"),
        "sh_daily_mid": _sh(dm) * ANN,
        "sh_daily_crossed": _sh(dx) * ANN,
        "hit_daily_mid": _hit(dm),
        "mean_per_trade_mid": _mean(pnl_mid[traded]),
        "mean_per_trade_crossed": _mean(pnl_x[traded]),
        "hit_per_trade_mid": _hit(pnl_mid[traded]),
    }


def _pn_book(
    pnl_mid: np.ndarray,
    pnl_x: np.ndarray,
    pos: np.ndarray,
    day: np.ndarray,
    snap: np.ndarray,
) -> dict[str, float]:
    """Premium-neutral: long-leg mean minus short-leg mean, per snapshot."""
    d = pd.DataFrame({"snap": snap, "day": day, "pos": pos, "m": pnl_mid, "x": pnl_x})
    d = d[d["pos"] != 0]
    if d.empty:
        return {
            "pn_n_snap": 0.0,
            "sh_pn_mid": float("nan"),
            "sh_pn_crossed": float("nan"),
            "mean_pn_mid": float("nan"),
        }
    legs = d.groupby(["snap", "day", d["pos"] > 0])[["m", "x"]].mean().unstack()
    both = legs.notna().all(axis=1)
    legs = legs[both]
    if legs.empty:
        return {
            "pn_n_snap": 0.0,
            "sh_pn_mid": float("nan"),
            "sh_pn_crossed": float("nan"),
            "mean_pn_mid": float("nan"),
        }
    pn_m = (legs[("m", True)] + legs[("m", False)]).to_numpy(float)
    pn_x = (legs[("x", True)] + legs[("x", False)]).to_numpy(float)
    dday = legs.index.get_level_values("day").to_numpy()
    ok = np.ones(len(pn_m), bool)
    return {
        "pn_n_snap": float(len(pn_m)),
        "sh_pn_mid": _sh(_daily(pn_m, ok, dday)) * ANN,
        "sh_pn_crossed": _sh(_daily(pn_x, ok, dday)) * ANN,
        "mean_pn_mid": _mean(pn_m),
    }


def main() -> None:
    pd.set_option("display.width", 240)
    ch, diag = load_chain()
    print(f"quoted rows on settled days: {len(ch)}", flush=True)
    panel = solve_iv(ch, diag)
    print(f"rows with finite BS IV: {len(panel)}", flush=True)

    path = ch.groupby(["expiration", "t"])["underlying_price"].first().reset_index()

    ztime = temporal_score(panel[["expiration", "t", "cp", "strike", "iv"]])
    print(f"contracts with a temporal score: {len(ztime)}", flush=True)

    uni = panel[panel["delta"].abs().between(DELTA_LO, DELTA_HI)].copy()
    uni = uni.sort_values(["expiration", "t", "cp", "strike"]).reset_index(drop=True)
    diag["universe_rows"] = float(len(uni))
    diag["universe_frac_of_finite_iv"] = float(len(uni) / max(len(panel), 1))
    print(f"trading universe (|delta| in band): {len(uni)}", flush=True)

    uni, fits = smile_scores(uni)
    uni = uni.merge(ztime, on=["expiration", "t", "cp", "strike"], how="left")
    uni = hedge_to_settle(uni, path)
    uni = uni[np.isfinite(uni["hedge"])].copy()

    intr = np.where(
        uni["cp"] == "C",
        np.maximum(uni["S_T"] - uni["strike"], 0.0),
        np.maximum(uni["strike"] - uni["S_T"], 0.0),
    )
    uni["intr"] = intr
    prem = uni["mid"].to_numpy(float)
    uni["long_dh_mid"] = (uni["intr"] - uni["mid"] + uni["hedge"]) / prem
    uni["long_dh_x"] = (
        uni["intr"] - uni["ask"] + uni["hedge"] - uni["hedge_cost"]
    ) / prem
    uni["short_dh_x"] = (
        uni["bid"] - uni["intr"] - uni["hedge"] - uni["hedge_cost"]
    ) / prem
    uni["et"] = (
        uni["t"]
        .dt.tz_localize("UTC")
        .dt.tz_convert("America/New_York")
        .dt.strftime("%H:%M")
    )
    uni["snap"] = uni.groupby(["expiration", "t"], sort=False).ngroup()
    uni.to_parquet(os.path.join(OUT, "iv_bumps_ledger.parquet"))

    day_all = uni["expiration"].to_numpy()
    lm = uni["long_dh_mid"].to_numpy(float)
    lx = uni["long_dh_x"].to_numpy(float)
    sx = uni["short_dh_x"].to_numpy(float)
    snap_all = uni["snap"].to_numpy()
    is_call = (uni["cp"] == "C").to_numpy()

    scores = {
        "xs_call": (uni["z_cs"].to_numpy(float), is_call),
        "xs_put": (uni["z_cs"].to_numpy(float), ~is_call),
        "xs_pooled": (uni["z_pool"].to_numpy(float), np.ones(len(uni), bool)),
        "temporal": (uni["z_time"].to_numpy(float), np.ones(len(uni), bool)),
    }
    eras = {
        "all": np.ones(len(uni), bool),
        "daily_0dte": (uni["expiration"] >= DAILY_0DTE).to_numpy(),
    }

    rows: list[dict[str, object]] = []
    for era, emask in eras.items():
        for sname, (z, umask) in scores.items():
            sel = emask & umask
            for zt in ZT_GRID:
                pos = np.where(z < -zt, 1.0, np.where(z > zt, -1.0, 0.0))
                pos = np.where(sel & np.isfinite(z), pos, 0.0)
                pnl_mid = pos * lm
                pnl_x = np.where(pos > 0, lx, np.where(pos < 0, sx, np.nan))
                rec: dict[str, object] = {"era": era, "score": sname, "zt": zt}
                rec.update(_book_row(pnl_mid, pnl_x, pos, day_all))
                rec["frac_flagged"] = float((pos != 0).sum() / max(int(sel.sum()), 1))
                rec.update(_pn_book(pnl_mid, pnl_x, pos, day_all, snap_all))
                rows.append(rec)
        atm = emask & uni["delta"].abs().between(ATM_LO, ATM_HI).to_numpy()
        for name, sgn in (("ctrl_always_short", -1.0), ("ctrl_always_long", 1.0)):
            pos = np.where(atm, sgn, 0.0)
            pnl_mid = pos * lm
            pnl_x = np.where(pos > 0, lx, np.where(pos < 0, sx, np.nan))
            rec = {"era": era, "score": name, "zt": float("nan")}
            rec.update(_book_row(pnl_mid, pnl_x, pos, day_all))
            rows.append(rec)
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(OUT, "iv_bumps_summary.csv"), index=False)

    hrows: list[dict[str, object]] = []
    zp = uni["z_pool"].to_numpy(float)
    r2map = (
        fits[fits["side"] == "pooled"].set_index(["expiration", "t"])["r2"].to_dict()
    )
    uni["r2_pool"] = [
        r2map.get((d, t), np.nan) for d, t in zip(uni["expiration"], uni["t"])
    ]
    for era, emask in eras.items():
        for hh in sorted(uni["et"].unique()):
            sel = emask & (uni["et"] == hh).to_numpy()
            pos = np.where(zp < -BY_HOUR_ZT, 1.0, np.where(zp > BY_HOUR_ZT, -1.0, 0.0))
            pos = np.where(sel & np.isfinite(zp), pos, 0.0)
            pnl_mid = pos * lm
            pnl_x = np.where(pos > 0, lx, np.where(pos < 0, sx, np.nan))
            rec = {
                "era": era,
                "entry_et": hh,
                "n_universe": float(sel.sum()),
                "zt": BY_HOUR_ZT,
            }
            rec.update(_book_row(pnl_mid, pnl_x, pos, day_all))
            rec["frac_flagged"] = float((pos != 0).sum() / max(int(sel.sum()), 1))
            rec["r2_pool_med"] = (
                float(np.nanmedian(uni["r2_pool"].to_numpy(float)[sel]))
                if sel.any()
                else float("nan")
            )
            rec.update(_pn_book(pnl_mid, pnl_x, pos, day_all, snap_all))
            hrows.append(rec)
    byh = pd.DataFrame(hrows)
    byh.to_csv(os.path.join(OUT, "iv_bumps_by_hour.csv"), index=False)

    drows: list[dict[str, object]] = [
        {"group": "data", "metric": k, "value": v} for k, v in diag.items()
    ]
    for side, g in fits.groupby("side"):
        for q in (0.10, 0.25, 0.50, 0.75, 0.90):
            drows.append(
                {
                    "group": f"smile_r2_{side}",
                    "metric": f"q{int(q * 100)}",
                    "value": float(g["r2"].quantile(q)),
                }
            )
        drows.append(
            {
                "group": f"smile_r2_{side}",
                "metric": "mean",
                "value": float(g["r2"].mean()),
            }
        )
        drows.append(
            {
                "group": f"smile_n_{side}",
                "metric": "median_strikes",
                "value": float(g["n"].median()),
            }
        )
        drows.append(
            {
                "group": f"smile_fits_{side}",
                "metric": "n_snapshots",
                "value": float(len(g)),
            }
        )
    for sname, (z, umask) in scores.items():
        drows.append(
            {
                "group": f"score_{sname}",
                "metric": "frac_defined",
                "value": float(np.isfinite(z[umask]).mean()),
            }
        )
        for zt in ZT_GRID:
            zz = z[umask]
            drows.append(
                {
                    "group": f"score_{sname}",
                    "metric": f"frac_abs_gt_{zt}",
                    "value": float(np.nanmean(np.abs(zz) > zt)),
                }
            )
    diagdf = pd.DataFrame(drows)
    diagdf.to_csv(os.path.join(OUT, "iv_bumps_diagnostics.csv"), index=False)

    print("\n=== diagnostics ===", flush=True)
    print(diagdf.to_string(index=False), flush=True)
    print("\n=== summary (books) ===", flush=True)
    print(summ.to_string(index=False), flush=True)
    print("\n=== by entry hour (xs_pooled, zt=1.5) ===", flush=True)
    print(byh.to_string(index=False), flush=True)
    print(
        "\nwrote: "
        + ", ".join(
            os.path.join(OUT, f)
            for f in (
                "iv_bumps_summary.csv",
                "iv_bumps_by_hour.csv",
                "iv_bumps_diagnostics.csv",
                "iv_bumps_ledger.parquet",
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
