"""Proposal 12 - the remaining-session oracle, held from clock t to the close.

AN EDA STUDY. Nothing here is a trade, nothing is selected and nothing is
adopted. The object described below peeks at the day's own realized variance;
it is a ceiling and a description, never a rule anyone could run.

THE OBJECT. At each clock t in 10:00..15:30:

    IV_rem_t = iv_hourly_t^2 * h_rem_t      hours from t to the 16:00 close
    RV_rem_t = sum of the session's realized bar variance from t to 16:00
    q_t      = sign(RV_rem_t - IV_rem_t)

IV_rem is a price: Black-Scholes-Merton at the vendor's hourly implied
volatility over the remaining window reproduces the quoted package midpoint
(gate 1 below). RV_rem is the same window's realized variance, which is only
known after the close - hence oracle. The position q_t is applied to the
nearest-OTM straddle picked at t and HELD to cash settlement at the official
close: one crossing at entry, no re-pick, no exit spread, and the strikes drift
off the money as the index moves.

This is NOT the 30-minute slice oracle of proposal 10's Check 0, which signs
the next bar's realized variance against a share of the remaining implied. That
object prices thirty minutes; this one prices the rest of the session.

Fills. The midpoint case enters at the quoted package midpoint. The crossed
case pays the touch once: a long buys at the ask, a short sells at the bid.
Cash settlement at the official close pays no exit spread.

Frame. The intraday notebook's cached package file joined to the
block-diagonal ridge forecast panel (bar-END labelled: the bar [t, t+30] and
its realized variance sit on the row stamped t+30), restricted to the deck's
866 days from results/atm_straddle_0dte_1530/daily_blk2.parquet. The remaining
realized variance is summed from the PANEL, not from the trade rows, so the
five day-clock cells the package guards refuse do not shorten anyone's
remainder.

Outputs: CSVs and two figures under results/atm_straddle_intraday/proposals/12/.
Every number in 12_oracle_hold_to_close.md is printed by this script.

Run:  python writeup/intraday_proposals/12_oracle_hold_to_close.py
"""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _bootstrap_repo(start: Path) -> Path:
    for q in [start.resolve(), *start.resolve().parents]:
        if (q / "notebooks" / "atm_straddle_lib.py").exists():
            return q
    raise FileNotFoundError("repo root not found from " + str(start))


sys.path.insert(0, str(_bootstrap_repo(Path(__file__)) / "notebooks"))

import atm_straddle_lib as asl  # noqa: E402

REPO = asl.find_repo(Path(__file__).resolve().parent)
CACHE_DIR = REPO / "results" / "atm_straddle_intraday" / "cache"
DECK = REPO / "results" / "atm_straddle_0dte_1530" / "daily_blk2.parquet"
VIX_FILE = REPO / "data" / "vix_and_voldemand.parquet"
OUT = REPO / "results" / "atm_straddle_intraday" / "proposals" / "12"

BAR_MINUTES = 30
PERTURB = 3.0  # the factor the peek demonstration multiplies later bars by
QS = (0.10, 0.25, 0.50, 0.75, 0.90)
REPORT_CLOCKS = ("10:00", "12:00", "15:00", "15:30")
AFTERNOON_CLOCKS = ("12:00", "13:00", "14:00", "15:00", "15:30")


# --------------------------------------------------------------------------
# small statistics
# --------------------------------------------------------------------------
def sharpe(x) -> float:
    v = pd.Series(np.asarray(x, float)).dropna()
    sd = float(v.std(ddof=1))
    if not (len(v) >= 2 and sd > 0):
        return float("nan")
    return float(v.mean() / sd * np.sqrt(asl.PERIODS_PER_YEAR))


def tstat(x) -> float:
    v = pd.Series(np.asarray(x, float)).dropna()
    sd = float(v.std(ddof=1))
    if not (len(v) >= 2 and sd > 0):
        return float("nan")
    return float(np.sqrt(len(v)) * v.mean() / sd)


def hac_t(x) -> float:
    """Autocorrelation-robust t of the mean against zero (Bartlett kernel)."""
    t, _ = asl.newey_west_t(pd.Series(np.asarray(x, float)).dropna())
    return float(t)


def max_drawdown_points(pts) -> float:
    run = pd.Series(np.asarray(pts, float)).fillna(0.0).cumsum()
    return float((run - run.cummax()).min())


def r_squared(x, y) -> float:
    a = pd.Series(np.asarray(x, float))
    b = pd.Series(np.asarray(y, float))
    ok = a.notna() & b.notna()
    if int(ok.sum()) < 3:
        return float("nan")
    c = float(np.corrcoef(a[ok], b[ok])[0, 1])
    return c * c


def phi(a, b) -> float:
    """Correlation of two sign series; NaN when either is constant."""
    x = np.asarray(a, float)
    y = np.asarray(b, float)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


# --------------------------------------------------------------------------
# frame
# --------------------------------------------------------------------------
def build_frame() -> tuple[pd.DataFrame, list[str], pd.DataFrame, pd.DataFrame]:
    """The deck's 866 days x twelve clocks, with the remaining session attached."""
    caches = sorted(CACHE_DIR.glob("trade_*.parquet"))
    if not caches:
        raise FileNotFoundError(f"no trade cache under {CACHE_DIR}")
    cache = caches[-1]
    pkg = pd.read_parquet(cache)
    pkg["t"] = pd.to_datetime(pkg["timestamp"], utc=True)

    pan = asl.load_yhat_panel_mz(asl.yhat_paths(REPO)["blk2"])
    pan["date"] = pan["et"].dt.normalize().dt.tz_localize(None)
    pan["hhmm"] = pan["et"].dt.strftime("%H:%M")

    p2 = pan.set_index("t")[["rv_raw", "in_fit"]].reset_index()
    p2["t"] = pd.to_datetime(p2["t"], utc=True) - pd.Timedelta(minutes=BAR_MINUTES)
    w = pkg.merge(p2, on="t", how="left").dropna(subset=["R", "rv_raw"])
    w["et"] = w["t"].dt.tz_convert("America/New_York")
    w["hhmm"] = w["et"].dt.strftime("%H:%M")
    w["date"] = w["et"].dt.normalize().dt.tz_localize(None)
    assert bool(w["in_fit"].all()), "a joined trade bar is outside the smear's fit mask"

    deck = pd.read_parquet(DECK)
    deck.index = pd.DatetimeIndex(deck.index).normalize()
    days = pd.DatetimeIndex(deck.index)
    w = w[w["date"].isin(days)].sort_values(["date", "t"]).reset_index(drop=True)
    clocks = sorted(w["hhmm"].unique())

    rv_rem, rv_bar = remaining_realized(pan, days, clocks)
    w = w.merge(
        rv_rem.stack().rename("rv_rem").reset_index(), on=["date", "hhmm"], how="left"
    )
    assert bool(w["rv_rem"].notna().all()), "a trade bar has no remaining realized"

    n_rem = {c: len(clocks) - i for i, c in enumerate(clocks)}
    w["h_rem"] = w["hhmm"].map(n_rem).astype(float) * (BAR_MINUTES / 60.0)
    w["iv_rem"] = w["iv_hourly"].astype(float) ** 2 * w["h_rem"]
    w["gap"] = w["rv_rem"] - w["iv_rem"]
    w["ratio"] = w["rv_rem"] / w["iv_rem"]
    w["gap_rel"] = w["gap"] / w["iv_rem"]
    w["q"] = np.where(w["gap"] > 0, 1.0, -1.0)

    # hold to cash settlement on the strikes picked at t
    w["pay"] = np.maximum(w["S_close"] - w["K_c"], 0.0) + np.maximum(
        w["K_p"] - w["S_close"], 0.0
    )
    w["R_hold"] = w["pay"] / w["entry"] - 1.0
    w["pts_long"] = w["pay"] - w["entry"]
    w["pts_short"] = w["entry"] - w["pay"]
    w["K_mid"] = 0.5 * (w["K_c"] + w["K_p"])
    w["k_gap"] = w["K_c"] - w["K_p"]
    w["offset"] = (w["S_close"] - w["K_mid"]).abs()
    # a driftless diffusion with the day's own remaining variance would move
    # the index by E|S_close - S_t| = S_t sqrt(2 RV_rem / pi)
    w["e_move"] = w["S"] * np.sqrt(2.0 * w["rv_rem"] / np.pi)
    w["term_var"] = w["entry"] - w["e_move"]
    w["term_dir"] = w["e_move"] - w["pay"]

    print(f"trade cache: {cache.name}")
    deck_st = DECK.stat()
    deck_fp = hashlib.sha1(
        np.ascontiguousarray(deck[["rv_hat", "signal", "pos", "R"]].to_numpy(float))
    ).hexdigest()[:12]
    print(
        f"deck file: {DECK.name} {deck_st.st_size} bytes, forecast fingerprint {deck_fp}"
    )
    print(f"forecast: {asl.YHAT_LABEL['blk2']}")
    print(
        f"frame: {len(w)} packages on {w['date'].nunique()} days, "
        f"{len(clocks)} clocks {clocks[0]}..{clocks[-1]}, "
        f"{days.min().date()} to {days.max().date()}"
    )
    print(
        f"day-clock cells the package guards refuse: "
        f"{len(clocks) * len(days) - len(w)} of {len(clocks) * len(days)}"
    )
    n_no_iv = int(w["iv_hourly"].isna().sum())
    print(
        f"rows with no vendor implied at t (solver-node censored): {n_no_iv}; "
        f"rows with an oracle: {len(w) - n_no_iv}"
    )
    print(
        f"strike pairs: same strike {int((w['k_gap'] == 0).sum())}, "
        f"one strike apart {int((w['k_gap'] == 5).sum())}, "
        f"wider {int((w['k_gap'] > 5).sum())}"
    )
    return w, clocks, deck, rv_bar


def remaining_realized(
    pan: pd.DataFrame, days: pd.DatetimeIndex, clocks: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """RV_rem(t) and the bar's own realized variance, both labelled by clock t.

    The panel is bar-END labelled, so the bar [c, c+30] sits on the row stamped
    c + 30 minutes. RV_rem(c) is the sum of that bar and every later one, out to
    the 16:00 stamp.
    """
    ends = [
        (pd.Timestamp("2000-01-01 " + c) + pd.Timedelta(minutes=BAR_MINUTES)).strftime(
            "%H:%M"
        )
        for c in clocks
    ]
    sub = pan[pan["date"].isin(days) & pan["hhmm"].isin(ends)]
    piv = (
        sub.pivot_table(index="date", columns="hhmm", values="rv_raw")
        .reindex(index=days, columns=ends)
        .sort_index()
    )
    assert bool(piv.notna().all().all()), "the panel is missing a session bar"
    rem = piv[ends[::-1]].cumsum(axis=1)[ends]
    ren = dict(zip(ends, clocks))
    rem = rem.rename(columns=ren)
    bar = piv.rename(columns=ren)
    rem.columns.name = "hhmm"
    bar.columns.name = "hhmm"
    rem.index.name = "date"
    bar.index.name = "date"
    return rem, bar


# --------------------------------------------------------------------------
# gates
# --------------------------------------------------------------------------
def gates(w: pd.DataFrame, clocks: list[str], deck: pd.DataFrame) -> pd.DataFrame:
    print("\n--- 0. Gates ---")
    f = w[w["iv_hourly"].notna()].copy()

    rows = []
    for c in clocks:
        g = f[f["hhmm"] == c]
        s_tot = g["iv_hourly"].to_numpy(float) * np.sqrt(g["h_rem"].to_numpy(float))
        price = np.array(
            [
                asl._bsm_package_price(s, S, kc, kp)
                for s, S, kc, kp in zip(
                    s_tot,
                    g["S"].to_numpy(float),
                    g["K_c"].to_numpy(float),
                    g["K_p"].to_numpy(float),
                )
            ]
        )
        r = price / g["entry"].to_numpy(float)
        rows.append(
            {
                "clock": c,
                "n": len(g),
                "pricing ratio p05": float(np.quantile(r, 0.05)),
                "pricing ratio median": float(np.median(r)),
                "pricing ratio p95": float(np.quantile(r, 0.95)),
            }
        )
    gate = pd.DataFrame(rows).set_index("clock")
    print("gate 1 - IV_rem is a price: BSM at the vendor hourly implied over the")
    print("remaining window against the quoted package midpoint")
    print(gate.round(4).to_string())
    assert bool((gate["pricing ratio median"].sub(1.0).abs() < 1e-3).all()), (
        "the remaining-window BSM price does not reproduce the midpoint"
    )

    last = w[w["hhmm"] == clocks[-1]].set_index("date")
    d = deck.reindex(last.index)
    diffs = {
        "K_c": float((last["K_c"] - d["K_c"]).abs().max()),
        "K_p": float((last["K_p"] - d["K_p"]).abs().max()),
        "entry": float((last["entry"] - d["entry"]).abs().max()),
        "S_close": float((last["S_close"] - d["S_close"]).abs().max()),
        "settlement payoff": float((last["pay"] - d["exit"]).abs().max()),
        "held return": float((last["R_hold"] - d["R"]).abs().max()),
        "IV_rem vs the deck iv_var": float((last["iv_rem"] - d["iv_var"]).abs().max()),
    }
    print(
        "\ngate 2 - the 15:30 row IS the deck's close trade "
        f"({len(last)} days), max absolute difference:"
    )
    for k, v in diffs.items():
        print(f"  {k}: {v:.3g}")
    assert max(diffs.values()) < 1e-5, "the 15:30 row does not reproduce the deck"

    rv_last = w[w["hhmm"] == clocks[-1]].set_index("date")["rv_rem"]
    rv_bar_last = w[w["hhmm"] == clocks[-1]].set_index("date")["rv_raw"]
    d3 = float((rv_last - rv_bar_last).abs().max())
    print(
        f"\ngate 3 - at {clocks[-1]} the remaining session IS the entry bar: "
        f"max |RV_rem - RV_bar| = {d3:.3g}"
    )
    assert d3 < 1e-18, "the last clock's remainder is not the last bar"

    sizes = asl.rule_sizes(deck)
    rule_tab = pd.DataFrame(
        {k: asl.rule_row(deck["R"] * v, v) for k, v in sizes.items()}
    ).T[["n", "mean", "t_mean", "Sharpe_ann", "pct_buy"]]
    print("\ngate 4 - the deck's own rule table on the file this run scored:")
    print(rule_tab.round(4).to_string())

    ident = float(
        (
            w["pay"]
            - np.maximum((w["S_close"] - w["K_mid"]).abs() - 0.5 * w["k_gap"], 0.0)
        )
        .abs()
        .max()
    )
    print(
        "\ngate 5 - the held package pays the terminal displacement from the strike\n"
        "centre less half the pair's width, floored at zero: "
        f"max |difference| = {ident:.3g}"
    )
    assert ident < 1e-9, "the settlement payoff is not the displacement identity"
    return gate


def peek_demonstration(w: pd.DataFrame, clocks: list[str]) -> pd.DataFrame:
    """The oracle is a peek: perturb every bar AFTER the entry bar and watch it move."""
    print("\n--- 0b. The peek, made explicit ---")
    f = w[w["iv_hourly"].notna()].copy()
    piv_bar = f.pivot_table(index="date", columns="hhmm", values="rv_raw").reindex(
        columns=clocks
    )
    piv_rem = f.pivot_table(index="date", columns="hhmm", values="rv_rem").reindex(
        columns=clocks
    )
    piv_iv = f.pivot_table(index="date", columns="hhmm", values="iv_rem").reindex(
        columns=clocks
    )
    later = piv_rem - piv_bar  # the remainder beyond the entry bar
    q0 = np.sign(piv_rem - piv_iv)
    q1 = np.sign(piv_bar + PERTURB * later - piv_iv)
    moved = (q0 != q1) & piv_rem.notna()
    rows = []
    for c in clocks:
        n = int(piv_rem[c].notna().sum())
        rows.append(
            {
                "clock": c,
                "n": n,
                "bars after the entry bar": len(clocks) - 1 - clocks.index(c),
                "sign moves when they are tripled": int(moved[c].sum()),
                "pct": 100.0 * float(moved[c].sum()) / n if n else float("nan"),
            }
        )
    tab = pd.DataFrame(rows).set_index("clock")
    print(
        f"tripling the realized variance of every bar strictly after the entry bar "
        f"({PERTURB:.0f}x):"
    )
    print(tab.round(2).to_string())
    assert int(tab.loc[clocks[-1], "sign moves when they are tripled"]) == 0, (
        "the last clock has no later bars and must not move"
    )
    print(
        "at "
        + clocks[-1]
        + " there are no later bars, so nothing moves; everywhere else the sign is "
        "made of information that does not exist at the entry clock."
    )
    return tab


# --------------------------------------------------------------------------
# 1. the gap
# --------------------------------------------------------------------------
def gap_by_clock(w: pd.DataFrame, clocks: list[str]) -> pd.DataFrame:
    print("\n--- 1. The gap: RV_rem against IV_rem ---")
    f = w[w["iv_hourly"].notna()]
    rows = []
    for c in clocks:
        g = f[f["hhmm"] == c]
        r = g["ratio"]
        row = {"clock": c, "n": len(g)}
        for q in QS:
            row[f"p{int(q * 100)}"] = float(r.quantile(q))
        row["mean"] = float(r.mean())
        row["pct short (RV_rem < IV_rem)"] = 100.0 * float((g["q"] < 0).mean())
        row["mean gap / IV_rem"] = float(g["gap_rel"].mean())
        row["t of the mean gap"] = hac_t(g["gap"])
        rows.append(row)
    tab = pd.DataFrame(rows).set_index("clock")
    print("RV_rem / IV_rem by entry clock (1.0 = the remainder came in at its price):")
    print(tab.round(4).to_string())
    return tab


# --------------------------------------------------------------------------
# 2. persistence
# --------------------------------------------------------------------------
def persistence(w: pd.DataFrame, clocks: list[str]) -> dict[str, pd.DataFrame]:
    print("\n--- 2. Does the day's sign settle early? ---")
    f = w[w["iv_hourly"].notna()]
    piv = (
        f.pivot_table(index="date", columns="hhmm", values="q")
        .reindex(columns=clocks)
        .sort_index()
    )
    full = piv.dropna()
    arr = full.to_numpy(float)
    n_days = len(full)
    never = int((np.abs(arr).sum(axis=1) == np.abs(arr.sum(axis=1))).sum())
    flips = (arr[:, 1:] != arr[:, :-1]).sum(axis=1)
    dist = pd.Series(flips).value_counts().sort_index()
    flip_tab = pd.DataFrame(
        {
            "flips per day": dist.index,
            "days": dist.to_numpy(),
            "pct of days": 100.0 * dist.to_numpy() / n_days,
        }
    ).set_index("flips per day")
    print(
        f"days with all {len(clocks)} clocks priced: {n_days}; "
        f"the sign never flips on {never} of them ({100.0 * never / n_days:.1f}%)"
    )
    print("all-short days: ", int((arr < 0).all(axis=1).sum()))
    print("all-long days:  ", int((arr > 0).all(axis=1).sum()))
    print("\nsign flips per day across the clocks:")
    print(flip_tab.round(2).to_string())
    print(
        f"mean flips {flips.mean():.3f}, median {np.median(flips):.1f}, "
        f"max {flips.max()}"
    )

    last = clocks[-1]
    rows = []
    for c in clocks:
        both = pd.DataFrame({"a": piv[c], "b": piv[last]}).dropna()
        agree = 100.0 * float((both["a"] == both["b"]).mean())
        sub = both[both["a"] < 0]
        subl = both[both["a"] > 0]
        rows.append(
            {
                "clock": c,
                "n": len(both),
                f"agrees with {last} %": agree,
                f"short at t -> short at {last} %": 100.0 * float((sub["b"] < 0).mean())
                if len(sub)
                else float("nan"),
                f"long at t -> long at {last} %": 100.0 * float((subl["b"] > 0).mean())
                if len(subl)
                else float("nan"),
            }
        )
    agree_tab = pd.DataFrame(rows).set_index("clock")
    print(f"\nagreement of the sign at t with the sign at {last}:")
    print(agree_tab.round(2).to_string())

    trans_rows = []
    tot = np.zeros((2, 2))
    for i in range(len(clocks) - 1):
        a, b = clocks[i], clocks[i + 1]
        both = pd.DataFrame({"a": piv[a], "b": piv[b]}).dropna()
        m = np.zeros((2, 2))
        for si, s in enumerate((-1.0, 1.0)):
            for ti, t in enumerate((-1.0, 1.0)):
                m[si, ti] = float(((both["a"] == s) & (both["b"] == t)).sum())
        tot += m
        rs = m.sum(axis=1)
        trans_rows.append(
            {
                "from": a,
                "to": b,
                "n": int(m.sum()),
                "short->short %": 100.0 * m[0, 0] / rs[0] if rs[0] else float("nan"),
                "short->long %": 100.0 * m[0, 1] / rs[0] if rs[0] else float("nan"),
                "long->short %": 100.0 * m[1, 0] / rs[1] if rs[1] else float("nan"),
                "long->long %": 100.0 * m[1, 1] / rs[1] if rs[1] else float("nan"),
            }
        )
    trans = pd.DataFrame(trans_rows).set_index(["from", "to"])
    print("\ntransitions between consecutive clocks:")
    print(trans.round(2).to_string())
    pooled = pd.DataFrame(
        tot, index=["from short", "from long"], columns=["to short", "to long"]
    )
    pooled_pct = 100.0 * pooled.div(pooled.sum(axis=1), axis=0)
    print("\npooled over the eleven consecutive pairs (counts, then row %):")
    print(pooled.astype(int).to_string())
    print(pooled_pct.round(2).to_string())
    return {
        "flips": flip_tab,
        "agree": agree_tab,
        "trans": trans,
        "pooled": pooled_pct,
        "pivot": piv,
    }


# --------------------------------------------------------------------------
# 3. variance against drift
# --------------------------------------------------------------------------
def variance_vs_drift(w: pd.DataFrame, clocks: list[str]) -> pd.DataFrame:
    print("\n--- 3. Variance against drift in the held straddle ---")
    f = w[w["iv_hourly"].notna()]
    rows = []
    for c in clocks:
        g = f[f["hhmm"] == c]
        q = g["q"].to_numpy(float)
        s_short = np.sign(g["pts_short"].to_numpy(float))
        short = g["pts_short"].to_numpy(float)
        v = g["term_var"].to_numpy(float)
        d = g["term_dir"].to_numpy(float)
        var_short = float(np.var(short, ddof=1))
        rows.append(
            {
                "clock": c,
                "n": len(g),
                "corr(sign gap, sign short P&L)": phi(q, s_short),
                "oracle hit rate %": 100.0
                * float((q * g["pts_long"].to_numpy(float) > 0).mean()),
                "R2 held points on the gap": r_squared(g["gap_rel"], g["pts_long"]),
                "R2 held return on the gap": r_squared(g["gap_rel"], g["R_hold"]),
                "sd variance term": float(np.std(v, ddof=1)),
                "sd directional term": float(np.std(d, ddof=1)),
                "variance share of var(short P&L)": float(
                    np.cov(v, short, ddof=1)[0, 1] / var_short
                ),
                "directional share": float(np.cov(d, short, ddof=1)[0, 1] / var_short),
                "corr(sign gap, sign variance term)": phi(q, np.sign(v)),
                "corr(|S_close - S_t|, E|move|)": float(
                    np.corrcoef(
                        (g["S_close"] - g["S"]).abs(), g["e_move"].to_numpy(float)
                    )[0, 1]
                ),
                "R2 payoff on E|move|": r_squared(g["e_move"], g["pay"]),
                "oracle Sharpe, held pair": sharpe(q * g["R_hold"]),
                "oracle Sharpe if it paid E|move|": sharpe(
                    q * (g["e_move"] - g["entry"]) / g["entry"]
                ),
            }
        )
    tab = pd.DataFrame(rows).set_index("clock")
    tab["ratio of the two oracle Sharpes"] = (
        tab["oracle Sharpe, held pair"] / tab["oracle Sharpe if it paid E|move|"]
    )
    print(
        "the gap is positive when the remainder outran its price, so the informative\n"
        "sign of corr(sign gap, sign short P&L) is NEGATIVE. The held short's P&L in\n"
        "points splits exactly as (entry - E|move|) + (E|move| - payoff), the price\n"
        "against the diffusive value of the day's own remaining realized variance,\n"
        "then that value against where the index actually landed; the two covariance\n"
        "shares sum to 1. E|move| = S_t sqrt(2 RV_rem / pi)."
    )
    print(tab.round(4).to_string())
    chk = float(
        (tab["variance share of var(short P&L)"] + tab["directional share"] - 1.0)
        .abs()
        .max()
    )
    print(f"max |variance share + directional share - 1| = {chk:.3g}")
    assert chk < 1e-9, "the covariance decomposition does not close"
    return tab


# --------------------------------------------------------------------------
# 4. the ceiling
# --------------------------------------------------------------------------
def pnl_table(
    w: pd.DataFrame, clocks: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("\n--- 4. The oracle as a ceiling, and always short beside it ---")
    f = w[w["iv_hourly"].notna()].copy()
    q = f["q"]
    f["or_mid"] = q * f["R_hold"]
    f["or_crossed"] = asl.crossed_premium_return(
        q, f["pay"], f["bid_entry"], f["ask_entry"]
    )
    f["or_pts_mid"] = q * f["pts_long"]
    f["or_pts_crossed"] = np.where(
        q > 0, f["pay"] - f["ask_entry"], f["bid_entry"] - f["pay"]
    )
    short = pd.Series(-1.0, index=f.index)
    f["as_mid"] = -f["R_hold"]
    f["as_crossed"] = asl.crossed_premium_return(
        short, f["pay"], f["bid_entry"], f["ask_entry"]
    )
    f["as_pts_mid"] = f["pts_short"]
    f["as_pts_crossed"] = f["bid_entry"] - f["pay"]
    n_bad = asl.crossed_untradeable_count(q, f["bid_entry"], f["ask_entry"])
    print(f"rows a crossed fill cannot price: {n_bad} of {len(f)}")

    rows = []
    for c in clocks:
        g = f[f["hhmm"] == c]
        row = asl.rule_row(g["or_mid"], g["q"])
        rows.append(
            {
                "clock": c,
                "n": int(row["n"]),
                "pct long": float(row["pct_buy"]),
                "oracle Sharpe mid": float(row["Sharpe_ann"]),
                "oracle t mid": float(row["t_mean"]),
                "oracle Sharpe crossed": sharpe(g["or_crossed"]),
                "oracle t crossed": tstat(g["or_crossed"]),
                "oracle pts/day mid": float(g["or_pts_mid"].mean()),
                "oracle pts/day crossed": float(g["or_pts_crossed"].mean()),
                "oracle maxDD pts mid": max_drawdown_points(g["or_pts_mid"]),
                "oracle maxDD pts crossed": max_drawdown_points(g["or_pts_crossed"]),
                "always short Sharpe mid": sharpe(g["as_mid"]),
                "always short Sharpe crossed": sharpe(g["as_crossed"]),
                "always short pts/day mid": float(g["as_pts_mid"].mean()),
                "always short pts/day crossed": float(g["as_pts_crossed"].mean()),
                "always short maxDD pts mid": max_drawdown_points(g["as_pts_mid"]),
                "median premium": float(g["entry"].median()),
                "median half-spread % of premium": 100.0
                * float(
                    (0.5 * (g["ask_entry"] - g["bid_entry"]) / g["entry"]).median()
                ),
            }
        )
    tab = pd.DataFrame(rows).set_index("clock")
    print("one trade a day, entered at clock t and held to cash settlement;")
    print("one crossing at entry, none at settlement:")
    print(tab.round(4).to_string())

    leg_rows = []
    for c in clocks:
        g = f[f["hhmm"] == c]
        ql = (g["q"] > 0).astype(float)
        qs = -(g["q"] < 0).astype(float)
        long_mid = ql * g["R_hold"]
        short_mid = qs * g["R_hold"]
        long_cr = asl.crossed_premium_return(
            ql, g["pay"], g["bid_entry"], g["ask_entry"]
        )
        short_cr = asl.crossed_premium_return(
            qs, g["pay"], g["bid_entry"], g["ask_entry"]
        )
        leg_rows.append(
            {
                "clock": c,
                "n days": len(g),
                "long days": int((g["q"] > 0).sum()),
                "short days": int((g["q"] < 0).sum()),
                "long leg Sharpe mid": sharpe(long_mid),
                "long leg Sharpe crossed": sharpe(long_cr),
                "long leg pts/day mid": float((ql * g["pts_long"]).mean()),
                "long leg pts/day crossed": float(
                    np.where(ql > 0, g["pay"] - g["ask_entry"], 0.0).mean()
                ),
                "short leg Sharpe mid": sharpe(short_mid),
                "short leg Sharpe crossed": sharpe(short_cr),
                "short leg pts/day mid": float((qs * g["pts_long"]).mean()),
                "short leg pts/day crossed": float(
                    np.where(qs < 0, g["bid_entry"] - g["pay"], 0.0).mean()
                ),
            }
        )
    legs = pd.DataFrame(leg_rows).set_index("clock")
    print("\nthe two sides of the oracle, each scored alone (flat on the other days):")
    print(legs.round(4).to_string())

    cum_or = (
        f.pivot_table(index="date", columns="hhmm", values="or_pts_mid")
        .reindex(columns=clocks)
        .fillna(0.0)
        .cumsum()
    )
    cum_as = (
        f.pivot_table(index="date", columns="hhmm", values="as_pts_mid")
        .reindex(columns=clocks)
        .fillna(0.0)
        .cumsum()
    )
    daily = pd.concat(
        [
            f.pivot_table(index="date", columns="hhmm", values="or_pts_mid").add_prefix(
                "oracle "
            ),
            f.pivot_table(index="date", columns="hhmm", values="as_pts_mid").add_prefix(
                "always short "
            ),
        ],
        axis=1,
    )
    return (
        tab,
        legs,
        pd.concat({"oracle": cum_or, "always short": cum_as}, axis=1),
        daily,
    )


# --------------------------------------------------------------------------
# 5. conditional structure
# --------------------------------------------------------------------------
def _cond_block(
    f: pd.DataFrame,
    key: pd.Series,
    name: str,
    report_clocks: tuple[str, ...] = REPORT_CLOCKS,
    order: list[str] | None = None,
    na_label: str = "unavailable",
) -> pd.DataFrame:
    g = f[f["hhmm"].isin(report_clocks)].join(
        key.astype(object).rename("_lvl"), on="date"
    )
    g["_lvl"] = g["_lvl"].where(g["_lvl"].notna(), na_label).astype(str)
    rows = []
    for lvl, sub in g.groupby("_lvl", dropna=False):
        row = {
            name: str(lvl),
            "days": int(sub["date"].nunique()),
            "bars": len(sub),
            "median RV_rem/IV_rem": float(sub["ratio"].median()),
            "pct short pooled": 100.0 * float((sub["q"] < 0).mean()),
        }
        for c in report_clocks:
            s = sub[sub["hhmm"] == c]
            row[f"pct short {c}"] = (
                100.0 * float((s["q"] < 0).mean()) if len(s) else float("nan")
            )
            row[f"median ratio {c}"] = (
                float(s["ratio"].median()) if len(s) else float("nan")
            )
        rows.append(row)
    tab = pd.DataFrame(rows).set_index(name)
    if order is not None:
        keep = [o for o in order if o in tab.index]
        keep += [i for i in tab.index if i not in keep]
        tab = tab.reindex(keep)
    return tab


def conditional(
    w: pd.DataFrame, clocks: list[str], rv_bar: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    print("\n--- 5. Conditional structure (descriptive; nothing is selected) ---")
    f = w[w["iv_hourly"].notna()].copy()
    days = pd.DatetimeIndex(sorted(w["date"].unique()))
    out: dict[str, pd.DataFrame] = {}

    dow = pd.Series(days.day_name(), index=days)
    out["dow"] = _cond_block(
        f,
        dow,
        "day of week",
        order=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
    )
    print("\nby day of week:")
    print(out["dow"].round(3).to_string())

    flags = asl.fomc_and_monthend(days, REPO, sessions=days)
    is_fomc = flags["is_fomc"].fillna(False).astype(bool)
    is_me = flags["is_me"].astype(bool)
    n_unknown = int((~flags["fomc_known"].astype(bool)).sum())
    cat = pd.Series(
        np.where(is_fomc, "fomc", np.where(is_me, "month end", "other")), index=days
    )
    out["event"] = _cond_block(f, cat, "calendar", order=["fomc", "month end", "other"])
    print(
        f"\nby calendar flag (FOMC first, then month end, else other; "
        f"{n_unknown} dates beyond the release file's horizon are read as other):"
    )
    print(out["event"].round(3).to_string())

    vix = pd.read_parquet(VIX_FILE)
    vix["endbartime"] = pd.to_datetime(vix["endbartime"])
    v10 = vix[vix["endbartime"].dt.strftime("%H:%M") == "10:00"].copy()
    v10["date"] = v10["endbartime"].dt.normalize()
    v10 = v10.set_index("date")["vix"].reindex(days)
    print(
        f"\nVIX at the 10:00 stamp (bar-end labelled, so the 09:30-10:00 bar): "
        f"{int(v10.notna().sum())} of {len(days)} days"
    )
    terc = pd.qcut(v10, 3, labels=["low VIX", "mid VIX", "high VIX"])
    edges = np.nanquantile(v10.to_numpy(float), [0.0, 1 / 3, 2 / 3, 1.0])
    print("tercile edges: " + ", ".join(f"{e:.2f}" for e in edges))
    out["vix"] = _cond_block(
        f,
        terc,
        "VIX tercile",
        order=["low VIX", "mid VIX", "high VIX"],
        na_label="no VIX",
    )
    print(out["vix"].round(3).to_string())

    yr = pd.Series(days.year.astype(str), index=days)
    out["year"] = _cond_block(f, yr, "year")
    print("\nby year:")
    print(out["year"].round(3).to_string())

    # the morning's realized against the afternoon's remaining gap
    morning_clocks = [c for c in clocks if c < "12:00"]
    rv_morning = rv_bar[morning_clocks].sum(axis=1).reindex(days)
    iv10 = f[f["hhmm"] == clocks[0]].set_index("date")["iv_rem"].reindex(days)
    h_morning = len(morning_clocks) * (BAR_MINUTES / 60.0)
    iv_morning = (
        f[f["hhmm"] == clocks[0]].set_index("date")["iv_hourly"].reindex(days) ** 2
        * h_morning
    )
    mr = pd.DataFrame(
        {
            "rv_morning": rv_morning,
            "iv_morning": iv_morning,
            "iv_rem_1000": iv10,
        }
    )
    mr["morning ratio"] = mr["rv_morning"] / mr["iv_morning"]
    print(
        f"\nmorning realized = the {len(morning_clocks)} bars "
        f"{morning_clocks[0]}-12:00 ({h_morning:.1f} h); its price at 10:00 is "
        "iv_hourly(10:00)^2 x that many hours"
    )
    rows = []
    for label, key in (
        ("morning realized", mr["rv_morning"]),
        ("morning realized / morning implied", mr["morning ratio"]),
    ):
        terc_m = pd.qcut(key, 3, labels=["low", "mid", "high"])
        blk = _cond_block(
            f,
            terc_m,
            "tercile",
            report_clocks=AFTERNOON_CLOCKS,
            order=["low", "mid", "high"],
            na_label="no morning price",
        )
        blk.insert(0, "conditioning", label)
        med = key.groupby(terc_m, observed=True).median()
        blk["median of the conditioning variable"] = [
            float(med.get(i, np.nan)) for i in blk.index
        ]
        rows.append(blk)
    out["morning"] = pd.concat(rows)
    print(
        "\nthe afternoon's remaining oracle conditional on the morning "
        "(the conditioning variable is known at 12:00; the oracle is not).\n"
        "pooled columns cover the afternoon clocks "
        f"{AFTERNOON_CLOCKS[0]}-{AFTERNOON_CLOCKS[-1]} only:"
    )
    print(out["morning"].to_string(float_format=lambda v: f"{v:.4g}"))
    return out


# --------------------------------------------------------------------------
# 6. where the strikes end up
# --------------------------------------------------------------------------
def strike_offset(w: pd.DataFrame, clocks: list[str]) -> pd.DataFrame:
    print("\n--- 6. Where the held pair sits at the close ---")
    f = w[w["iv_hourly"].notna()]
    rows = []
    for c in clocks:
        g = f[f["hhmm"] == c]
        ok = g[g["k_gap"] > 0]
        strides = ok["offset"] / ok["k_gap"]
        rows.append(
            {
                "clock": c,
                "n": len(g),
                "median premium at entry": float(g["entry"].median()),
                "median |S_close - K|": float(g["offset"].median()),
                "mean |S_close - K|": float(g["offset"].mean()),
                "p90 |S_close - K|": float(g["offset"].quantile(0.90)),
                "median in strikes": float(strides.median()),
                "pct > 1 strike": 100.0 * float((strides > 1).mean()),
                "pct > 2 strikes": 100.0 * float((strides > 2).mean()),
                "pct settling worthless": 100.0 * float((g["pay"] <= 0).mean()),
                "same-strike rows dropped from the strike columns": len(g) - len(ok),
            }
        )
    tab = pd.DataFrame(rows).set_index("clock")
    print(
        "distances are from the official close to the centre of the held pair; "
        "one strike is the pair's own width"
    )
    print(tab.round(3).to_string())
    return tab


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def figure_gap(gap: pd.DataFrame, clocks: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    x = np.arange(len(clocks))
    ax.fill_between(
        x, gap["p10"], gap["p90"], color="#4c72b0", alpha=0.18, label="10th-90th"
    )
    ax.fill_between(
        x, gap["p25"], gap["p75"], color="#4c72b0", alpha=0.35, label="25th-75th"
    )
    ax.plot(x, gap["p50"], color="#1a3f6f", lw=2.0, marker="o", ms=3.5, label="median")
    ax.plot(x, gap["mean"], color="#c44e52", lw=1.4, ls="--", label="mean")
    ax.axhline(1.0, color="0.35", lw=1.0, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(clocks, rotation=45, ha="right")
    ax.set_xlabel("entry clock (ET)")
    ax.set_ylabel("RV_rem / IV_rem")
    ax.set_title(
        "The remaining session's realized variance over its price, by clock\n"
        f"{int(gap['n'].max())} days; below 1.0 the remainder came in short of its price"
    )
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "12_gap_by_clock.png", dpi=150)
    plt.close(fig)


def figure_cum(cum: pd.DataFrame, clocks: list[str]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)
    cmap = plt.get_cmap("viridis")
    for ax, name in zip(axes, ("oracle", "always short")):
        sub = cum[name]
        for i, c in enumerate(clocks):
            ax.plot(
                sub.index,
                sub[c].to_numpy(float),
                color=cmap(i / max(1, len(clocks) - 1)),
                lw=1.2,
                label=c,
            )
        ax.axhline(0.0, color="0.4", lw=0.9)
        ax.set_title(name + ", held to settlement, midpoint entry")
        ax.set_xlabel("session")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("cumulative index points per straddle")
    axes[1].legend(frameon=False, fontsize=7, ncol=2, title="entry clock")
    fig.tight_layout()
    fig.savefig(OUT / "12_cum_points.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    print("=" * 78)
    print("12. The remaining-session oracle, held from clock t to the close")
    print("EDA. It peeks at the day's realized variance. Nothing here is adopted.")
    print("=" * 78)

    w, clocks, deck, rv_bar = build_frame()
    gate = gates(w, clocks, deck)
    peek = peek_demonstration(w, clocks)
    gap = gap_by_clock(w, clocks)
    per = persistence(w, clocks)
    vd = variance_vs_drift(w, clocks)
    pnl, legs, cum, daily = pnl_table(w, clocks)
    cond = conditional(w, clocks, rv_bar)
    off = strike_offset(w, clocks)

    figure_gap(gap, clocks)
    figure_cum(cum, clocks)

    gate.to_csv(OUT / "12_gate.csv")
    peek.to_csv(OUT / "12_peek.csv")
    gap.to_csv(OUT / "12_gap_quantiles.csv")
    per["flips"].to_csv(OUT / "12_flips.csv")
    per["agree"].to_csv(OUT / "12_persistence.csv")
    per["trans"].to_csv(OUT / "12_transitions.csv")
    per["pooled"].to_csv(OUT / "12_transitions_pooled.csv")
    vd.to_csv(OUT / "12_variance_vs_drift.csv")
    pnl.to_csv(OUT / "12_pnl_by_clock.csv")
    legs.to_csv(OUT / "12_legs_by_clock.csv")
    daily.to_csv(OUT / "12_daily_pnl_points.csv")
    cond["dow"].to_csv(OUT / "12_conditional_dow.csv")
    cond["event"].to_csv(OUT / "12_conditional_event.csv")
    cond["vix"].to_csv(OUT / "12_conditional_vix.csv")
    cond["year"].to_csv(OUT / "12_conditional_year.csv")
    cond["morning"].to_csv(OUT / "12_morning_conditional.csv")
    off.to_csv(OUT / "12_strike_offset.csv")

    print(f"\nwrote {OUT}")
    print(f"elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
