"""Does the cumrv x close signal pass through to IMPLIED vol (overnight dVIX)? — venue-gate test.

The measured edge is an RV forecast (straddle P&L = RV - IV monetizes it directly; that rail
needs an options venue). A VIX CFD (DNA Funded / FTMO-class CFD props) monetizes only the
RV -> IV pass-through: sell VIX at 16:00 when cumrv says close/AH vol will mean-revert below
baseline, cover at next open / next close. This script measures that pass-through on the dated
causal signal (build_cumrv_signal_dated.py) x the raw VIX bars (data/vix_and_voldemand.parquet).

If dVIX is predictable: the venue universe expands from one sketchy options firm to FTMO-class
CFD props. If not: the VIX-CFD fork dies and the options rail (Imperial + WRDS friction pricing)
is the only one standing. Costs: VIX CFD quoted spreads ~0.05-0.20 pts round-trip; financing on
overnight holds ignored (small at 1-day horizon) but flagged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SIG = "eval_sim/data/cumrv_signal_dated.parquet"
VIXP = "data/vix_and_voldemand.parquet"
SPREAD_TIERS = [0.0, 0.05, 0.10, 0.20]  # VIX pts, round-trip
CAP = 3.0  # |signal| sizing cap (matches pnl-log construction spirit)


def nw_tstat(x: np.ndarray, lags: int = 5) -> float:
    """Newey-West t-stat of mean(x) > 0."""
    x = x[np.isfinite(x)]
    n = x.size
    mu = x.mean()
    e = x - mu
    s = e @ e / n
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        s += 2 * w * (e[:-L] @ e[L:]) / n
    return float(mu / np.sqrt(s / n))


def main() -> None:
    sig = pd.read_parquet(SIG)
    sig["date"] = pd.to_datetime(sig["date"])

    v = pd.read_parquet(VIXP)[["endbartime", "vix"]].dropna()
    v["t"] = pd.to_datetime(v["endbartime"])
    v["date"] = v["t"].dt.normalize()
    tod = v["t"].dt.time
    day = v[
        (tod >= pd.Timestamp("1900-01-01 10:00").time())
        & (tod <= pd.Timestamp("1900-01-01 16:00").time())
    ]
    close = day.groupby("date")["vix"].last().rename("vix_close")
    openv = day.groupby("date")["vix"].first().rename("vix_open")
    px = pd.concat([close, openv], axis=1).sort_index()
    px["vix_next_open"] = px["vix_open"].shift(-1)
    px["vix_next_close"] = px["vix_close"].shift(-1)
    px["dvix_on"] = (
        px["vix_next_open"] - px["vix_close"]
    )  # overnight: 16:00 -> next 10:00 bar
    px["dvix_cc"] = px["vix_next_close"] - px["vix_close"]  # close -> next close

    df = sig.merge(px.reset_index(), on="date", how="inner").dropna(
        subset=["signal_1600", "dvix_on", "dvix_cc"]
    )
    print(
        f"{len(df)} joined days  {df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()}"
    )

    s = df["signal_1600"].to_numpy()
    for h in ("dvix_on", "dvix_cc"):
        d = df[h].to_numpy()
        rho = np.corrcoef(s, d)[0, 1]
        # trade: short VIX when cumrv above expectation (predicts vol mean-reversion)
        pos = -np.sign(s) * np.minimum(np.abs(s), CAP)
        gross = pos * d
        print(
            f"\n[{h}] corr(signal, dvix) = {rho:+.4f}   NW-t(gross pnl) = {nw_tstat(gross):+.2f}"
        )
        for sp in SPREAD_TIERS:
            net = gross - np.abs(pos) * sp
            sh = net.mean() / net.std() if net.std() > 0 else float("nan")
            print(
                f"  spread {sp:.2f} pts RT: mean={net.mean():+.4f} vixpts/day  "
                f"Sharpe/day={sh:+.3f}  ann={sh * np.sqrt(252):+.2f}"
            )
        # breakeven round-trip spread
        be = (
            gross.mean() / np.abs(pos).mean()
            if np.abs(pos).mean() > 0
            else float("nan")
        )
        print(f"  breakeven RT spread = {be:.3f} VIX pts")
        # regime split: post-2016 (0DTE / modern vol regime)
        late = df["date"] >= "2016-01-01"
        for lab, m in (("pre-2016", ~late), ("post-2016", late)):
            g = gross[m.to_numpy()]
            sh = g.mean() / g.std() if g.std() > 0 else float("nan")
            print(
                f"  {lab}: gross Sharpe/day={sh:+.3f}  NW-t={nw_tstat(g):+.2f}  n={g.size}"
            )


if __name__ == "__main__":
    main()
