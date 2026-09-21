"""62 - running the insurance book and the 15:30 sign(s) trade together.

Positions add, so one account holding both trades earns exactly what two
accounts earn (the book holds to settlement and never crosses a spread at 15:30,
so netting on shared strikes changes margin, not P&L).  What decides the
combined Sharpe is how much of the risk each trade carries.  A quick check with
weights fitted on the whole sample suggested a small share of risk in the 15:30
trade helps (11:00 book 3.35 -> 3.41 at 17%; 13:30 book 1.68 -> 1.86 at 33%);
fitted weights are optimistic, so here every weight is set from PAST days only.

Streams, deck days (the 15:30 signal ends 2024-04-30):
  book   short straddle sold at the entry clock, delta-hedged, held to
         settlement, index points per contract (proposal 43's tape)
  1530   the ridge's 15:30 sign(s) straddle trade, crossed (bought at the ask,
         sold at the bid), per unit of premium (the deck's sizing)
Each stream is scaled by its own trailing standard deviation (expanding over
strictly prior days, minimum 252) so that a weight is a share of risk.

Weight rules, written before running, long-only (a trade is never reversed):
  R0  the book alone, scaled the same way (the baseline every rule is paired to)
  R1  risk share of the 15:30 trade = SR2 / (SR1 + SR2), trailing Sharpes (the
      optimal split for two uncorrelated trades); a negative trailing Sharpe
      gets no risk
  R2  the two-asset tangency weights from trailing means and covariance
      (Sigma^-1 mu), negative weights set to zero, as a risk share
  R3  fixed risk shares 10%, 20%, 30%, 50% (descriptive only)
Layers: plain, and "with month-end": the book with the live month-end override
(no short that day, one 15:30 straddle bought at the ask; study 60 P5) and the
15:30 trade forced long on month-ends (study 55).

H5 (written before running): combining ADDS if R1 beats R0 with a paired
block-bootstrap interval of the Sharpe difference above zero at 11:00 or 13:30,
in either layer (four headline cells; about 0.1 pass by chance).  R2 and the
fixed shares are reported beside it.

Run:  python writeup/intraday_proposals/62_combine_book_and_1530.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "notebooks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import atm_straddle_lib as asl  # noqa: E402

DECK = ROOT / "results" / "atm_straddle_0dte_1530"
P43_DAILY = (
    ROOT
    / "results"
    / "atm_straddle_intraday_holdclose"
    / "proposals"
    / "43"
    / "a_daily_by_clock.csv"
)
P54_DAILY = DECK / "proposals" / "54" / "c_daily.csv"
OUT = ROOT / "results" / "atm_straddle_intraday_holdclose" / "proposals" / "62"
ANN = float(np.sqrt(asl.PERIODS_PER_YEAR))
WARMUP = 252
HEADLINE = ("11:00", "13:30")
FIXED_SHARES = (0.1, 0.2, 0.3, 0.5)
B, BLOCK, SEED = 2000, 21, 0
GATE_SHARPE = (1.338322, 0.869588)
GATE_TOL = 1e-6


def sharpe(x: np.ndarray) -> float:
    return float(np.mean(x) / np.std(x, ddof=1) * ANN)


def sharpe_rows(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1) / x.std(axis=1, ddof=1) * ANN


def trailing(x: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Expanding mean and sd over strictly prior days, minimum WARMUP."""
    m = x.expanding(min_periods=WARMUP).mean().shift(1)
    s = x.expanding(min_periods=WARMUP).std().shift(1)
    return m, s


def combine(book: pd.Series, t: pd.Series) -> dict[str, pd.Series]:
    mb, sb = trailing(book)
    mt, st = trailing(t)
    zb, zt = book / sb, t / st  # one unit of trailing risk each
    srb, srt = (mb / sb).clip(lower=0.0), (mt / st).clip(lower=0.0)  # long-only
    tot = srb + srt
    share_r1 = (srt / tot).where(tot > 0, 0.0)
    # tangency on the standardised streams: weights proportional to R^-1 SR
    rho = book.expanding(min_periods=WARMUP).corr(t).shift(1)
    wb = (srb - rho * srt) / (1 - rho**2)
    wt = (srt - rho * srb) / (1 - rho**2)
    wb, wt = wb.clip(lower=0.0), wt.clip(lower=0.0)
    share_r2 = (wt / (wb + wt)).where((wb + wt) > 0, 0.0)
    out = {
        "R0 book alone": zb,
        "R1 Sharpe-proportional share": (1 - share_r1) * zb + share_r1 * zt,
        "R2 tangency share": (1 - share_r2) * zb + share_r2 * zt,
    }
    for f in FIXED_SHARES:
        out[f"R3 fixed {f:.0%} share"] = (1 - f) * zb + f * zt
    out["_share_r1"] = share_r1
    out["_share_r2"] = share_r2
    out["_rho"] = rho
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    d = pd.read_parquet(DECK / "daily_blk2.parquet").sort_index()
    di = pd.DatetimeIndex(d.index)
    q = np.where(d["signal"].to_numpy(float) > 0, 1.0, -1.0)
    ask = (d["ask_c"] + d["ask_p"]).to_numpy(float)
    bid = (d["bid_c"] + d["bid_p"]).to_numpy(float)
    ex, r = d["exit"].to_numpy(float), d["R"].to_numpy(float)

    def crossed(qq: np.ndarray) -> np.ndarray:
        return np.where(qq > 0, qq * (ex / ask - 1.0), qq * (ex / bid - 1.0))

    got = (sharpe(q * r), sharpe(crossed(q)))
    assert (
        abs(got[0] - GATE_SHARPE[0]) < GATE_TOL
        and abs(got[1] - GATE_SHARPE[1]) < GATE_TOL
    ), got
    print(f"GATE  ridge sign(s) {got[0]:.6f} / {got[1]:.6f}")
    c54 = pd.read_csv(P54_DAILY, index_col=0, parse_dates=True).reindex(di)
    me = c54["month_end"].fillna(False).astype(bool).to_numpy()
    atm_ask_pts = c54["pts_ask"].to_numpy(float)
    t_plain = pd.Series(crossed(q), index=di)
    t_me = pd.Series(crossed(np.where(me, 1.0, q)), index=di)
    a = pd.read_csv(P43_DAILY, index_col=0, parse_dates=True).reindex(di)

    rows, years, shares = [], [], []
    clocks = sorted({c.split("|")[0] for c in a.columns})
    for clock in clocks:
        hold = (a[f"{clock}|hold"] * a[f"{clock}|entry"]).to_numpy(float)
        for layer, book_np, t in (
            ("plain", hold, t_plain),
            ("with month-end", np.where(me, atm_ask_pts, hold), t_me),
        ):
            book = pd.Series(book_np, index=di)
            ok = book.notna() & t.notna()
            streams = combine(book[ok], t[ok])
            scored = streams["R0 book alone"].notna()
            for k in [k for k in streams if not k.startswith("_")]:
                scored &= streams[k].notna()
            n = int(scored.sum())
            idx = asl.circular_block_bootstrap_idx(
                np.random.default_rng([SEED, n]), n, BLOCK, B
            )
            base = streams["R0 book alone"][scored].to_numpy()
            s_base = sharpe_rows(base[idx])
            for name in [k for k in streams if not k.startswith("_")]:
                v = streams[name][scored].to_numpy()
                dv = sharpe_rows(v[idx]) - s_base
                rows.append(
                    {
                        "entry": clock,
                        "layer": layer,
                        "rule": name,
                        "days": n,
                        "first_day": str(streams[name][scored].index[0].date()),
                        "Sharpe": sharpe(v),
                        "vs_book": sharpe(v) - sharpe(base),
                        "ci_lo": float(np.percentile(dv, 2.5)),
                        "ci_hi": float(np.percentile(dv, 97.5)),
                        "worst_day_risk_units": float(v.min()),
                    }
                )
                if clock in HEADLINE:
                    yy = streams[name][scored]
                    for y, g in yy.groupby(yy.index.year):
                        years.append(
                            {
                                "entry": clock,
                                "layer": layer,
                                "rule": name,
                                "year": int(y),
                                "Sharpe": sharpe(g.to_numpy()),
                            }
                        )
            if clock in HEADLINE:
                shares.append(
                    {
                        "entry": clock,
                        "layer": layer,
                        "raw_book_Sharpe_same_days": sharpe(
                            book[ok][scored].to_numpy()
                        ),
                        "raw_1530_Sharpe_same_days": sharpe(t[ok][scored].to_numpy()),
                        "share_r1_mean": float(streams["_share_r1"][scored].mean()),
                        "share_r1_last": float(streams["_share_r1"][scored].iloc[-1]),
                        "share_r2_mean": float(streams["_share_r2"][scored].mean()),
                        "rho_last": float(streams["_rho"][scored].iloc[-1]),
                    }
                )
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "a_rules_by_clock.csv", index=False)
    pd.DataFrame(years).to_csv(OUT / "b_by_year.csv", index=False)
    sh = pd.DataFrame(shares)
    sh.to_csv(OUT / "c_shares.csv", index=False)
    print(
        "\nheadline clocks: Sharpe of risk-scaled combinations, paired to the book alone"
    )
    print(tab[tab["entry"].isin(HEADLINE)].round(3).to_string(index=False))
    print("\ncausal risk shares chosen (and the raw Sharpes on the same scored days)")
    print(sh.round(3).to_string(index=False))
    print("\nR1 minus R0 by entry clock")
    print(
        tab[tab["rule"] == "R1 Sharpe-proportional share"]
        .pivot(index="entry", columns="layer", values="vs_book")
        .round(2)
        .to_string()
    )
    yt = pd.DataFrame(years)
    print("\nby year, headline clocks")
    print(
        yt[yt["rule"].isin(["R0 book alone", "R1 Sharpe-proportional share"])]
        .pivot_table(index=["entry", "layer", "rule"], columns="year", values="Sharpe")
        .round(2)
        .to_string()
    )
    cells = tab[
        tab["entry"].isin(HEADLINE)
        & (tab["rule"] == "R1 Sharpe-proportional share")
        & (tab["ci_lo"] > 0.0)
    ]
    print(
        f"\nH5  headline cells with an interval above zero: {len(cells)} of 4 -> "
        f"{'SUPPORTED' if len(cells) else 'NOT SUPPORTED'}"
    )


if __name__ == "__main__":
    main()
