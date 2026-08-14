"""Per-bar ATM straddle vs the model: the liquid, path-sensitive test.

The strip-based premium numbers (entry_hour_sweep) marked hundreds of
zero-bid wings at half-tick mids with 1/K^2 weights — a marking
artifact, not tradable premium. The honest liquid instrument at every
30-min bar is the ATM straddle: both legs live, the tightest quotes on
the board, and (late day) vega ~ sqrt(T) -> 0, so the last bars are
nearly pure realized-variance bets — exactly where blk2's one-bar edge
is largest. Unlike the strip, ATM straddle premium need not be
one-signed intraday; wherever it is two-sided the forecast finally has
a decision to change.

Per (day, bar): ATM strike = argmin |K - S_t| with both legs mid>0;
hold to 16:00 settlement, intrinsic |S_T - K|. Model value =
E|S_T - K| under the causal per-hour empirical density of standardized
remaining moves u = log(S_T/S_t)/sqrt(RV_rem), scaled by the causally
calibrated remaining-variance forecast (blk2, and a0 for the swap
test). Books per entry hour, mids and crossed:

  * always_short (control), sign(model/mid - 1) two-sided book for
    blk2 and a0, paired blk2-a0 daily difference (the exog increment),
    fraction of long days, median premium fraction.

Causality: per-hour pools/calibrations expanding (min 63 days),
shifted one day. Settlement uses the day's S_T from the toclose trades
file (same convention as the strip experiments).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "spxw_pnl")
ANN = float(np.sqrt(252.0))
BURN = 63
DAILY_0DTE = pd.Timestamp("2022-05-16")


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


def build_atm() -> pd.DataFrame:
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
    ch = ch[ch["mid"] > 0]
    c = ch[ch["cp"] == "C"]
    p = ch[ch["cp"] == "P"]
    st = c.merge(
        p,
        on=["expiration", "timestamp", "strike"],
        suffixes=("_c", "_p"),
    )
    st["S"] = st["underlying_price_c"]
    st = st[np.isfinite(st["S"])]
    st["dist"] = (st["strike"] - st["S"]).abs()
    idx = st.groupby(["expiration", "timestamp"])["dist"].idxmin().dropna()
    atm = st.loc[idx].copy()
    atm["str_mid"] = atm["mid_c"] + atm["mid_p"]
    atm["str_bid"] = atm["bid_c"] + atm["bid_p"]
    atm["str_ask"] = atm["ask_c"] + atm["ask_p"]
    return atm[
        ["expiration", "timestamp", "strike", "S", "str_mid", "str_bid", "str_ask"]
    ]


def main() -> None:
    eb = pd.read_parquet(os.path.join(OUT, "everybar_mtm_trades.parquet"))
    eb = eb.sort_values(["day", "et"]).reset_index(drop=True)
    g = eb.groupby("day", sort=False)
    eb["rv_rem"] = g["rv"].transform(lambda s: s.iloc[::-1].cumsum().iloc[::-1])
    eb["b2_rem"] = g["pb"].transform(lambda s: s.iloc[::-1].cumsum().iloc[::-1])
    eb["a0_rem"] = g["pa"].transform(lambda s: s.iloc[::-1].cumsum().iloc[::-1])
    eb["hhmm"] = eb["et"].dt.strftime("%H:%M")

    tr = pd.read_parquet(
        os.path.join(OUT, "mfiv_toclose_trades.parquet"),
        columns=["expiration", "S_T"],
    )
    tr["expiration"] = pd.to_datetime(tr["expiration"])
    eb["expiration"] = pd.to_datetime(eb["expiration"])
    eb = eb.merge(tr, on="expiration", how="inner")

    atm = build_atm()
    df = eb.merge(
        atm.rename(columns={"timestamp": "t"}),
        on=["expiration", "t"],
        how="inner",
    )
    print(f"bars with live ATM straddle + settle: {len(df)}", flush=True)

    df["intr"] = (df["S_T"] - df["strike"]).abs()
    df["umove"] = np.log(df["S_T"] / df["S"]) / np.sqrt(
        np.maximum(df["rv_rem"].to_numpy(float), 1e-18)
    )

    # per-hour causal machinery
    df["model_b2"] = np.nan
    df["model_a0"] = np.nan
    for _, gh in df.groupby("hhmm"):
        gh = gh.sort_values("expiration")
        i_loc = gh.index.to_numpy()
        rvr = gh["rv_rem"].to_numpy(float)
        u = gh["umove"].to_numpy(float)
        for tag, col in (("b2_rem", "model_b2"), ("a0_rem", "model_a0")):
            fr = gh[tag].to_numpy(float)
            cal = (
                pd.Series(rvr / np.maximum(fr, 1e-18))
                .expanding(min_periods=BURN)
                .mean()
                .shift(1)
                .to_numpy(float)
            )
            S0 = gh["S"].to_numpy(float)
            K = gh["strike"].to_numpy(float)
            vals = np.full(len(gh), np.nan)
            for j in range(BURN, len(gh)):
                if not np.isfinite(cal[j]):
                    continue
                U = u[:j]
                U = U[np.isfinite(U)]
                if U.size < BURN:
                    continue
                sv = np.sqrt(max(cal[j] * fr[j], 1e-18))
                ex = np.exp(sv * U)
                ST = S0[j] * ex / ex.mean()
                vals[j] = float(np.abs(ST - K[j]).mean())
            df.loc[i_loc, col] = vals

    ok = df[["model_b2", "model_a0", "str_mid", "intr"]].notna().all(axis=1) & (
        df["str_mid"] > 0
    )
    d = df[ok].copy()
    d["pay_long"] = (d["intr"] - d["str_mid"]) / d["str_mid"]
    for tag in ("b2", "a0"):
        d[f"edge_{tag}"] = d[f"model_{tag}"] / d["str_mid"] - 1.0
        d[f"book_{tag}"] = np.sign(d[f"edge_{tag}"]) * d["pay_long"]
    # crossed: long pays the ask; short sells the bid (needs a live bid)
    long_x = (d["intr"] - d["str_ask"]) / d["str_ask"]
    short_x = np.where(
        d["str_bid"] > 0, (d["str_bid"] - d["intr"]) / d["str_bid"], np.nan
    )
    d["book_b2_x"] = np.where(d["edge_b2"] > 0, long_x, short_x)

    rows = []
    for era, sub in (
        ("all", d),
        ("daily_0dte", d[d["expiration"] >= DAILY_0DTE]),
    ):
        for hh, gd in sub.groupby("hhmm"):
            diff = (gd["book_b2"] - gd["book_a0"]).to_numpy(float)
            rows.append(
                {
                    "era": era,
                    "entry": hh,
                    "n": len(gd),
                    "prem_frac_med": float((-gd["edge_b2"]).median()),
                    "frac_long": float((gd["edge_b2"] > 0).mean()),
                    "sh_always_short": _sh(-gd["pay_long"].to_numpy(float)) * ANN,
                    "sh_sign_b2": _sh(gd["book_b2"].to_numpy(float)) * ANN,
                    "sh_sign_a0": _sh(gd["book_a0"].to_numpy(float)) * ANN,
                    "t_b2_minus_a0": _tstat(diff),
                    "sh_sign_b2_crossed": _sh(gd["book_b2_x"].to_numpy(float)) * ANN,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "atm_straddle_bars.csv"), index=False)
    d.drop(columns=["umove"]).to_parquet(
        os.path.join(OUT, "atm_straddle_bars_daily.parquet")
    )
    print(out.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
