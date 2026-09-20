"""Rule-by-strategy tables for the twelve intraday entry windows, plus pooled.

The deck's standalone (writeup/rule_by_strategy_standalone.tex, built from
writeup/make_rule_by_strategy_tex.py) is one booktabs tabular with a panel per
rule -- "Short volatility (always short)" and sign(s) -- and the eight forecasts
down the rows.  This script builds the same object for every intraday entry
window 10:00, 10:30, ..., 15:30 of the intraday notebook's frame, and one for
the pooled daily sums, with ONE added column: the annualized Sharpe at the
crossed spread (entry at the touch, exit at the touch of the next stamp; the
15:30 window cash-settles and pays no exit spread; a re-pick that lands on the
same two strikes with the same sign is a hold, not a round trip, and is not
charged -- the intraday notebook's convention).

Construction (mirrors notebooks/_write_0dte_intraday_nb.py):

  * trade frame  = the latest results/atm_straddle_intraday/cache/trade_*.parquet,
    restricted to the deck's 866 expiration days;
  * forecast     = asl.load_yhat_panel_mz(...)["rv_hat"] for each of the eight
    tags in asl.MODEL_ORDER, joined on timestamp + 30 min (the panel is
    bar-end labelled);
  * implied slice = iv_hourly^2 * hours_to_close * w_slice, with w_slice the
    expanding mean of each prior day's remaining share
    RV_c / sum_{s>=c} RV_s (in-fit panel rows back to 2001, shift(1), min 63
    sessions).  At 15:30 w_slice = 1 and the slice is iv_hourly^2 / 2 exactly,
    equal to the deck's iv_var on all 866 days (both asserted);
  * censored implied: the bars whose vendor implied volatility sits on a
    solver bracket node do NOT sit flat.  They take the deck's treatment
    (iv_hourly_15_30 in §8 of notebooks/_write_0dte_nb.py): the volatility
    that reproduces the package midpoint, recovered by bisection over the
    stamp's hours_to_expiration.  With that in place the 15:30 window IS the
    deck's close trade, and gate() asserts the whole ridge row reproduces it.

Writes
  results/atm_straddle_intraday/rule_by_strategy/<hhmm>/rule_by_strategy_*.csv
  results/atm_straddle_intraday/rule_by_strategy/pooled/rule_by_strategy_*.csv
  writeup/generated_intraday/table_rule_by_strategy_intraday_<hhmm>.tex
  writeup/rule_by_strategy_intraday_<hhmm>.tex        (standalone, compiled)
  writeup/rule_by_strategy_intraday_pooled.tex        (standalone, compiled)
  writeup/rule_by_strategy_intraday_index.tex         (all thirteen, compiled)
"""

from __future__ import annotations

import concurrent.futures as cf
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pypdf import PdfReader  # noqa: E402

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(ROOT / "notebooks"))

import atm_straddle_lib as asl  # noqa: E402

DH_HOLDCLOSE = "--dh-holdclose" in sys.argv
WRITEUP = ROOT / "writeup"
GEN = WRITEUP / ("generated_intraday_dh" if DH_HOLDCLOSE else "generated_intraday")
OUT = (
    ROOT
    / "results"
    / (
        "atm_straddle_intraday_holdclose/rule_by_strategy_dh"
        if DH_HOLDCLOSE
        else "atm_straddle_intraday/rule_by_strategy"
    )
)
TRADE_CACHE = (
    ROOT
    / "results"
    / (
        "atm_straddle_intraday_holdclose/cache"
        if DH_HOLDCLOSE
        else "atm_straddle_intraday/cache"
    )
)
FIG_DIR = (
    ROOT
    / "results"
    / ("atm_straddle_intraday_holdclose" if DH_HOLDCLOSE else "atm_straddle_intraday")
)
DOC = "rule_by_strategy_dh_holdclose" if DH_HOLDCLOSE else "rule_by_strategy_intraday"
GEN_REL = GEN.name
DECK = ROOT / "results" / "atm_straddle_0dte_1530"
WORKERS = min(7, (os.cpu_count() or 4))

# Deck numbers of record for the 15:30 window (results/atm_straddle_0dte_1530/
# rule_by_strategy_*.csv).  The gate is on these, never on a loosened tolerance.
DECK_ALWAYS_SHORT_SHARPE = 0.203779128108568
DECK_SIGN_S_RIDGE_SHARPE = 1.3383216004229939

PANELS = [
    ("always_short", r"Short volatility (always short)"),
    ("sign_s", r"$\mathrm{sign}(s)$ ($\pm 1$ on the sign of $s$)"),
]

MODEL_TEX = {
    "all models": r"all models (no forecast)",
    "baseline (HAR + calendar OLS)": r"baseline (HAR + calendar OLS)",
    "block-diagonal ridge": r"block-diagonal ridge",
    "block-diagonal ridge, without the FOMC columns": (
        r"block-diagonal ridge, without the FOMC columns"
    ),
    "LightGBM": r"LightGBM$^{\dagger}$",
    "XGBoost": r"XGBoost$^{\dagger}$",
    "lasso (causally tuned)": r"lasso (causally tuned)$^{\dagger}$",
    "lasso (fixed 1e-4)": r"lasso ($\alpha=10^{-4}$)",
    "elastic net (causally tuned)": r"elastic net (causally tuned)$^{\dagger}$",
}

# (csv column, format, tex header) -- the deck's PAPER_COLS, plus the one
# addition at the end.
COLS: list[tuple[str, str, str]] = [
    ("n", "{:.0f}", r"$n$"),
    ("mean", "{:.3f}", r"mean"),
    ("std", "{:.3f}", r"std"),
    ("min", "{:.2f}", r"min"),
    ("max", "{:.2f}", r"max"),
    ("skew", "{:.2f}", r"skew"),
    ("ex_kurt", "{:.2f}", r"ex.\ kurt"),
    ("t_mean", "{:.2f}", r"$t$"),
    ("Sharpe_ann", "{:.2f}", r"Sharpe$_{\mathrm{ann}}$"),
    ("n_buy", "{:.0f}", r"$n_{\mathrm{buy}}$"),
    ("pct_buy", "{:.1f}", r"\%buy"),
    ("Sharpe_crossed", "{:.2f}", r"Sharpe$^{\times}_{\mathrm{ann}}$"),
]


def cols_for(pooled: bool) -> list[tuple[str, str, str]]:
    """Pooled page keeps Sharpe_ann; single-clock pages use Sharpe_clk."""
    out = []
    for c, fmt, h in COLS:
        if c == "Sharpe_ann":
            h = r"Sharpe$_{\mathrm{ann}}$" if pooled else r"Sharpe$_{\mathrm{clk}}$"
        elif c == "Sharpe_crossed":
            h = (
                r"Sharpe$^{\times}_{\mathrm{ann}}$"
                if pooled
                else r"Sharpe$^{\times}_{\mathrm{clk}}$"
            )
        out.append((c, fmt, h))
    return out


PREAMBLE = r"""\documentclass{article}
\usepackage[landscape,margin=0.5in]{geometry}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{graphicx}
\pagestyle{empty}
"""


def footnote() -> str:
    if DH_HOLDCLOSE:
        clk = (
            r"of that clock's series: one remaining-session short per expiration day "
            r"(enter at this clock, hold those strikes to the close, delta-hedge every "
            r"30 minutes). It answers ``if I only entered at this clock, what is my "
            r"annual Sharpe?'' It is \emph{not} the book's Sharpe$_{\mathrm{ann}}$."
        )
        pooled = (
            r"of $R^{day}_d=\sum_t R'_{d,t}$, the sum of the day's twelve overlapping "
            r"remaining-session books. That reserved name is the book's daily P\&L after "
            r"pooling the session. The twelve books overlap; they are not twelve "
            r"successive 30-minute holds."
        )
    else:
        clk = (
            r"of that clock's series: one 30-minute hold per expiration day. It answers ``if I "
            r"only entered at this clock, what is my annual Sharpe?'' It is \emph{not} the "
            r"book's Sharpe$_{\mathrm{ann}}$."
        )
        pooled = (
            r"of $R^{day}_d=\sum_t R'_{d,t}$, the sum of the day's twelve holds. That reserved "
            r"name is the book's daily P\&L after pooling the session. At 15:30 the two "
            r"coincide (one trade that day); they do not on 10:00--15:00 or on the pooled page."
        )
    return (
        r"\noindent\small One expiration day = one return; mid fill. Every $t$ here is the plain "
        r"$t=\sqrt{n}\cdot\mathrm{mean}/\mathrm{std}$, with no autocorrelation correction. "
        r"``ex.\ kurt'' is the bias-corrected sample excess kurtosis (Gaussian $=0$). "
        r"The short-volatility rule takes no forecast, so it is one row for all models. "
        r"\textbf{Two Sharpes.} Both multiply by $\sqrt{252}$ because an expiration day is "
        r"the unit of time (same year-length as the paper). They are not the same object. "
        r"On a \emph{single-clock} page, Sharpe$_{\mathrm{clk}}=(\mathrm{mean}/\mathrm{std})\times\sqrt{252}$ "
        + clk
        + r" On the \emph{pooled} page, Sharpe$_{\mathrm{ann}}=(\mathrm{mean}/\mathrm{std})\times\sqrt{252}$ "
        + pooled
        + r" Sharpe$^{\times}$ is the same convention at the crossed spread. Every other column is daily. "
        r"$n_{\mathrm{buy}}$ is the number of expiration days with position $>0$ (always-short is 0; "
        r"on the pooled page it is days the sum of the twelve positions is positive); "
        r"\%buy is $100\cdot n_{\mathrm{buy}}/n$, the notebook's column. "
        r"The other column this table adds to the deck's is Sharpe$^{\times}_{\mathrm{ann}}$, the same "
        r"rule's annualized Sharpe at the \emph{crossed spread} instead of the midpoint; every other "
        r"column is the deck's."
        "\n\n"
        r"\smallskip"
        "\n"
        r"\noindent\small Forecasts: baseline (HAR + calendar OLS); block-diagonal ridge (HAR block and exogenous block, separate penalties) "
        r"on the design carrying the FOMC calendar block, and the same ridge on the earlier design without those columns, as a diagnostic row; "
        r"LightGBM and XGBoost on the all-features design; lasso on the same design, causally tuned vs.\ fixed $\alpha=10^{-4}$; "
        r"elastic net (causally tuned). $\dagger$ marks a column still fitted on the earlier design panel; a run of those four on the "
        r"panel of record is pending. "
        r"The signal is $s_t=\widehat{RV}_t-\mathrm{IV}^{2}_{\mathrm{hr}}h_t w_t$: the forecast against the "
        r"implied variance of the same window. $w_t$ is the expanding mean of each prior day's "
        r"remaining-session share of realized variance, $\mathrm{RV}_t/\sum_{s\ge t}\mathrm{RV}_s$ "
        r"(in-fit panel history back to 2001), not the ratio of trailing clock-means. "
        r"On the bars whose vendor implied volatility is a censored "
        r"solver node the deck's treatment is used here too --- the volatility that reproduces the package "
        r"midpoint, recovered by bisection over the remaining $h_t$ hours --- so every bar carries a "
        r"signal and no bar sits flat."
    )


def load_panel_for_tag(tag: str) -> tuple[str, pd.DataFrame]:
    """Worker: one MZ panel. Module-level so ProcessPoolExecutor can pickle it."""
    return tag, asl.load_yhat_panel_mz(asl.yhat_paths(ROOT)[tag])


def load_panels_parallel() -> dict[str, pd.DataFrame]:
    t0 = time.perf_counter()
    n = min(WORKERS, len(asl.MODEL_ORDER))
    with cf.ProcessPoolExecutor(max_workers=n) as pool:
        loaded = dict(pool.map(load_panel_for_tag, asl.MODEL_ORDER))
    print(
        f"loaded {len(loaded)} forecast panels in {time.perf_counter() - t0:.1f}s "
        f"({n} workers)"
    )
    return loaded


# -------------------------------------------------------------- implied ----
def on_vendor_node(v: pd.Series) -> np.ndarray:
    """True where the vendor reported a bracket node rather than a solved vol."""
    v = pd.to_numeric(v, errors="coerce").astype(float)
    return (asl.censor_vendor_iv(v).isna() & v.notna()).to_numpy()


def iv_hourly_reinverted(work: pd.DataFrame) -> pd.Series:
    """Vendor hourly implied volatility, re-inverted on the censored bars.

    The deck's treatment (notebooks/_write_0dte_nb.py, iv_hourly_15_30 in §8):
    a bar whose call or put leg sits on a solver bracket node carries no
    volatility, so recover the one that reproduces the package MIDPOINT by
    bisection and read it back in the vendor's hourly convention.  The deck
    trades only 15:30, where the package covers the last half hour; at an
    intraday clock the package still covers the remaining session, so
    hours_remaining is that stamp's hours_to_expiration -- which the frame
    already carries as h_rem (asserted against data/spxw_chain.parquet).
    """
    iv = work["iv_hourly"].astype(float).copy()
    capped = on_vendor_node(work["impl_volatility_c"]) | on_vendor_node(
        work["impl_volatility_p"]
    )
    cap = capped & (work["entry"].to_numpy(float) > 0)
    idx = np.flatnonzero(cap)
    hrs = work["h_rem"].to_numpy(float)
    chain = pd.read_parquet(
        ROOT / "data" / "spxw_chain.parquet",
        columns=["expiration", "timestamp", "strike", "cp", "hours_to_expiration"],
    )
    chain = chain[chain["cp"] == "C"]
    got = (
        work.iloc[idx][["expiration", "timestamp", "K_c"]]
        .merge(
            chain,
            left_on=["expiration", "timestamp", "K_c"],
            right_on=["expiration", "timestamp", "strike"],
            how="left",
        )["hours_to_expiration"]
        .to_numpy(float)
    )
    assert np.allclose(got, hrs[idx]), (
        "hours_to_expiration disagrees with the clock's hours to the close"
    )
    inv = np.array(
        [
            asl.bsm_invert_package_vol(
                float(work["S"].iloc[i]),
                float(work["K_c"].iloc[i]),
                float(work["K_p"].iloc[i]),
                float(work["entry"].iloc[i]),
                hours_remaining=hrs[i],
            )
            for i in idx
        ]
    )
    rec = np.array(
        [
            asl.hourly_iv_from_total_vol(s, hours_remaining=h)
            for s, h in zip(inv, hrs[idx])
        ]
    )
    iv.iloc[idx] = rec
    print(
        f"censored vendor implied on {len(idx)} bars "
        f"({int((work['hhmm'].to_numpy()[idx] == '15:30').sum())} of them at 15:30); "
        f"re-inverted from the package midpoint on {int(np.isfinite(rec).sum())} of them"
    )
    assert bool(np.isfinite(iv.to_numpy()).all()), (
        "a bar is still without an implied volatility after the re-inversion"
    )
    return iv


def attach_long_dh(work: pd.DataFrame) -> pd.DataFrame:
    """Replace R with long-package delta-hedged t→T return (q = +1).

    Δ rebalanced every 30 minutes on vendor S; remaining vol = stamp IV × √h.
    Hedge P&L stored in hedge_long (index points) for the crossed column.
    """
    from scipy.special import erf

    def cdf(z):
        return 0.5 * (1.0 + erf(z / np.sqrt(2.0)))

    clocks = sorted(work["hhmm"].unique())
    n_rem = {c: len(clocks) - i for i, c in enumerate(clocks)}
    S_grid = work.pivot_table(index="date", columns="hhmm", values="S", aggfunc="first")
    S_grid = S_grid.reindex(columns=clocks)
    ivcol = "iv_hourly_used" if "iv_hourly_used" in work.columns else "iv_hourly"
    IV_grid = work.pivot_table(
        index="date", columns="hhmm", values=ivcol, aggfunc="first"
    )
    IV_grid = IV_grid.reindex(columns=clocks)
    h_row = np.array([n_rem[c] * 0.5 for c in clocks])
    dates = work["date"].to_numpy()
    t_idx = pd.Index(clocks).get_indexer(work["hhmm"].to_numpy())
    Sg = S_grid.loc[dates].to_numpy(float)
    IVg = IV_grid.loc[dates].to_numpy(float)
    ST = work["S_close"].to_numpy(float)
    Kc = work["K_c"].to_numpy(float)[:, None]
    Kp = work["K_p"].to_numpy(float)[:, None]
    n, m = Sg.shape
    col = np.arange(m)[None, :]
    active = col >= t_idx[:, None]
    nxt = np.full_like(Sg, np.nan)
    nxt[:, :-1] = Sg[:, 1:]
    last = np.where(active, col, -1).max(axis=1)
    for j in range(m):
        nxt[last == j, j] = ST[last == j]
    tot = np.where((IVg > 0) & active, IVg * np.sqrt(h_row[None, :]), np.nan)
    F = Sg
    s = tot
    dlt = np.zeros_like(Sg)
    pos = (s > 0) & np.isfinite(s) & (F > 0) & np.isfinite(F)
    if pos.any():
        ss, FF = s[pos], F[pos]
        kc = np.broadcast_to(Kc, Sg.shape)[pos]
        kp = np.broadcast_to(Kp, Sg.shape)[pos]
        d1c = (np.log(FF / kc) + 0.5 * ss * ss) / ss
        d1p = (np.log(FF / kp) + 0.5 * ss * ss) / ss
        dlt[pos] = cdf(d1c) + cdf(d1p) - 1.0
    dlt = np.where(active & np.isfinite(Sg), dlt, 0.0)
    dS = np.where(active & np.isfinite(Sg) & np.isfinite(nxt), nxt - Sg, 0.0)
    hedge = ((-dlt) * dS).sum(axis=1)
    entry = work["entry"].to_numpy(float)
    opt = work["exit"].to_numpy(float) - entry
    work = work.copy()
    work["hedge_long"] = hedge
    work["R"] = (opt + hedge) / entry
    print(
        f"DH t→T attached: median |Δ| {float(np.median(np.abs(dlt[active]))):.3f}; "
        f"mean hedge {float(np.nanmean(hedge)):.3f} pts"
    )
    return work


# ---------------------------------------------------------------- frame ----
def build_work() -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Trade bars on the deck's 866 days, with the slice and the eight rv_hat."""
    cache = TRADE_CACHE
    trade = max(cache.glob("trade_*.parquet"), key=lambda p: p.stat().st_mtime)
    print("trade cache:", trade.name)
    pkg = pd.read_parquet(trade)

    deck = pd.read_parquet(DECK / "daily_blk2.parquet")
    deck.index = pd.to_datetime(deck.index)
    days = pd.DatetimeIndex(deck.index)
    print("deck days", len(days), days.min().date(), "->", days.max().date())

    work = pkg[pkg["date"].isin(days)].copy()
    work["t"] = pd.to_datetime(work["timestamp"], utc=True)
    work = work.sort_values("t").reset_index(drop=True)
    clocks = sorted(work["hhmm"].unique())
    print("bars", len(work), "days", work["date"].nunique(), "clocks", clocks)

    # Causal remaining share: each prior day's RV_c / remaining-to-close sum,
    # then expanding mean. Panel in-fit history back to 2001; no warm-up.
    # Eight MZ panels load in parallel; blk2 is reused for w and for rv_hat.
    loaded = load_panels_parallel()
    pan = loaded["blk2"]
    pf = pan[pan["in_fit"].to_numpy(dtype=bool)].copy()
    clock = pf["et"] - pd.Timedelta(minutes=30)
    pf["pdate"] = clock.dt.normalize().dt.tz_localize(None)
    pf["phhmm"] = clock.dt.strftime("%H:%M")
    prof = pf.pivot_table(
        index="pdate", columns="phhmm", values="rv_raw", aggfunc="mean"
    ).sort_index()
    _pi = pd.DataFrame(index=prof.index, columns=clocks, dtype=float)
    for _i, _c in enumerate(clocks):
        _rem = prof[clocks[_i:]].sum(axis=1)
        _pi[_c] = prof[_c] / _rem.replace(0.0, np.nan)
    w_slice = _pi.expanding(min_periods=63).mean().shift(1)
    assert bool(np.isclose(w_slice["15:30"].dropna().to_numpy(), 1.0).all()), (
        "w must be 1 at 15:30"
    )
    mi = pd.MultiIndex.from_arrays([work["date"], work["hhmm"]])
    work["w_slice"] = w_slice.stack().reindex(mi).to_numpy()
    assert bool(np.isfinite(work["w_slice"]).all()), "a scored bar has no slice"
    print(
        "diurnal profile on",
        int(prof.index.size),
        "panel sessions",
        prof.index.min().date(),
        "->",
        prof.index.max().date(),
    )

    n_rem = {c: len(clocks) - i for i, c in enumerate(clocks)}
    work["h_rem"] = work["hhmm"].map(n_rem).astype(float) * 0.5
    work["iv_hourly_used"] = iv_hourly_reinverted(work).to_numpy()
    iv2 = work["iv_hourly_used"].astype(float) ** 2
    work["slice"] = iv2 * work["h_rem"] * work["w_slice"]
    at15 = work["hhmm"] == "15:30"
    chk = work.loc[at15]
    dev = float(
        (chk["slice"] / (chk["iv_hourly_used"].astype(float) ** 2 * 0.5) - 1.0)
        .abs()
        .max()
    )
    assert dev == 0.0, f"15:30 slice is not iv_hourly^2/2 (max deviation {dev})"
    print("15:30 slice equals iv_hourly^2/2 exactly on all", len(chk), "days")
    # Against the deck's own iv_var: exact on the quoted days, and on the three
    # re-inverted ones down to the float32 rounding of the package midpoint the
    # two frames store (entry 31.150001 against 31.150000, and so on).
    rel = float(
        (chk.set_index("date")["slice"] / deck["iv_var"].astype(float) - 1.0)
        .abs()
        .max()
    )
    assert rel < 1e-6, (
        f"the 15:30 slice differs from the deck's iv_var by {rel} relative"
    )
    print(
        f"15:30 slice equals the deck's iv_var on all 866 days (max rel diff {rel:.3g})"
    )

    for tag in asl.MODEL_ORDER:
        d = loaded[tag][["t", "rv_hat"]].copy()
        d["t"] = pd.to_datetime(d["t"], utc=True) - pd.Timedelta(minutes=30)
        joined = work[["t"]].merge(d, on="t", how="left")["rv_hat"].to_numpy()
        work["rv_hat_" + tag] = joined
        miss = int((~np.isfinite(work["rv_hat_" + tag])).sum())
        print(f"  {tag:9s} joined; bars with no forecast: {miss}")
    return work, clocks, deck


def positions(work: pd.DataFrame, tag: str) -> np.ndarray:
    """sign(s) over the whole frame.

    The q = 0 branch is the guard for a bar with no signal at all; with the
    censored implied re-inverted from the package midpoint and every forecast
    joined, it does not fire on this frame (build_work asserts both).
    """
    s = work["rv_hat_" + tag].to_numpy(float) - work["slice"].to_numpy(float)
    return np.where(s > 0, 1.0, np.where(np.isfinite(s), -1.0, 0.0))


# --------------------------------------------------------------- crossed ----
def crossed_points(work: pd.DataFrame, q: np.ndarray) -> pd.Series:
    """Per-bar P&L in index points at the crossed spread.

    Entry at the ask when long and at the bid when short; exit at the touch of
    the next stamp, except the 15:30 bar, which cash-settles at the official
    close and pays no exit spread.  A bar whose next stamp re-picks the SAME
    two strikes with the same sign is held through: no exit, no re-entry, no
    spread at that boundary.  A bar with no quote on the side actually used (a
    zero fill price) cannot be priced at the spread and is NaN.
    """
    ask_e = (work["ask_c"] + work["ask_p"]).to_numpy(float)
    bid_e = (work["bid_c"] + work["bid_p"]).to_numpy(float)
    ask_x = (work["ask_c_nxt"] + work["ask_p_nxt"]).to_numpy(float)
    bid_x = (work["bid_c_nxt"] + work["bid_p_nxt"]).to_numpy(float)
    is_last = work["is_last"].to_numpy(dtype=bool)
    same_k = (
        (work["K_c"].shift(-1) == work["K_c"])
        & (work["K_p"].shift(-1) == work["K_p"])
        & (work["date"].shift(-1) == work["date"])
        & ~work["is_last"]
    ).to_numpy(dtype=bool)

    q = np.asarray(q, float)
    long, short = q > 0, q < 0
    nxt_q = np.append(q[1:], 0.0)
    hold = same_k & (np.sign(nxt_q) == np.sign(q)) & (q != 0)
    held_in = np.concatenate([[False], hold[:-1]])
    entry = work["entry"].to_numpy(float)
    exit_ = work["exit"].to_numpy(float)
    entry_px = np.where(
        held_in, entry, np.where(long, ask_e, np.where(short, bid_e, entry))
    )
    exit_px = np.where(
        is_last,
        exit_,
        np.where(hold, exit_, np.where(long, bid_x, np.where(short, ask_x, exit_))),
    )
    untradeable = ~held_in & ((long & ~(ask_e > 0)) | (short & ~(bid_e > 0)))
    pts = np.where(untradeable, np.nan, q * (exit_px - entry_px))
    if DH_HOLDCLOSE and "hedge_long" in work.columns:
        pts = pts + np.asarray(q, float) * work["hedge_long"].to_numpy(float)
    return pd.Series(pts, index=work.index)


# ----------------------------------------------------------------- rows ----
def daily_series(
    work: pd.DataFrame, q: np.ndarray, mask: np.ndarray, pooled: bool
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """(return, position, crossed-spread return) per expiration day.

    One row per day: the window's own bar, or -- pooled -- the day's sum over
    the twelve windows.  The table rows and the histograms read the same series.
    """
    date = work["date"]
    days = pd.DatetimeIndex(sorted(date[mask].unique()))
    qm = np.where(mask, q, 0.0)
    rp = pd.Series(qm * work["R"].to_numpy(float), index=work.index)
    cross = pd.Series(
        crossed_points(work, qm).to_numpy() / work["entry"].to_numpy(float),
        index=work.index,
    )
    if pooled:
        return (
            rp.groupby(date).sum().reindex(days),
            pd.Series(qm, index=work.index).groupby(date).sum().reindex(days),
            cross.groupby(date).sum().reindex(days),
        )
    idx = np.flatnonzero(mask)
    return (
        pd.Series(rp.to_numpy()[idx], index=days),
        pd.Series(qm[idx], index=days),
        pd.Series(cross.to_numpy()[idx], index=days),
    )


def summary_row(
    work: pd.DataFrame, q: np.ndarray, mask: np.ndarray, pooled: bool
) -> pd.Series:
    """One rule-table row for the bars in `mask`, at the midpoint and crossed."""
    r, size, rc = daily_series(work, q, mask, pooled)
    row = asl.rule_row(r, size)
    x = rc.dropna()
    sd = float(x.std(ddof=1)) if len(x) >= 2 else float("nan")
    row["Sharpe_crossed"] = (
        float(x.mean()) / sd * np.sqrt(asl.PERIODS_PER_YEAR)
        if (sd and sd > 0)
        else float("nan")
    )
    row["n_crossed"] = float(len(x))
    return row


def window_mask(work: pd.DataFrame, key: str) -> np.ndarray:
    if key == "pooled":
        return np.ones(len(work), bool)
    return (work["hhmm"].to_numpy() == f"{key[:2]}:{key[2:]}").astype(bool)


# ------------------------------------------------------------- histogram ----
HIST_RULES = ("always short", r"sign($s$)")


def hist_payload(work: pd.DataFrame, key: str) -> dict[str, np.ndarray]:
    """The two daily return series a window's histogram draws (parent side)."""
    pooled = key == "pooled"
    mask = window_mask(work, key)
    q = {
        "always short": -np.ones(len(work)),
        r"sign($s$)": positions(work, "blk2"),
    }
    return {
        n: daily_series(work, v, mask, pooled)[0].dropna().to_numpy(float)
        for n, v in q.items()
    }


def render_hist(
    job: tuple[str, dict[str, np.ndarray]],
) -> tuple[str, str, dict[str, int]]:
    """The deck's rule_hists_blk2.png, for one entry window (worker side).

    Block-diagonal ridge, midpoint fills, one panel per rule.  40 bins across
    the pooled 1st--99th percentile window, laid on multiples of the bin width
    so an edge falls exactly on zero (the deck's construction); the range is
    widened when the series carries mass at exactly -1 -- the 15:30 leg's days
    on which the package expires worthless -- so that mass is drawn, not
    swept into the clipping bin.
    """
    key, arrays = job
    pooled = key == "pooled"
    ser = {n: pd.Series(v) for n, v in arrays.items()}
    allv = pd.concat(list(ser.values()))
    lo, hi = float(allv.quantile(0.01)), float(allv.quantile(0.99))
    at_m1 = {n: int((np.abs(s.to_numpy() + 1.0) < 1e-9).sum()) for n, s in ser.items()}
    if sum(at_m1.values()) and lo > -1.0:
        lo = -1.0
    w = (hi - lo) / 40.0
    bins = np.arange(np.floor(lo / w) - 1, np.ceil(hi / w) + 1) * w

    fig, axes = plt.subplots(1, len(ser), figsize=(9.5, 3.1), sharex=True, sharey=True)
    for ax, name in zip(np.ravel(axes), HIST_RULES):
        x = ser[name]
        nz = x[x != 0.0]
        ax.hist(nz.clip(bins[0], bins[-1]), bins=bins, color="C0", edgecolor="none")
        n_zero = int((x == 0.0).sum())
        if n_zero:
            ax.bar(
                [0.0], [n_zero], width=0.35 * w, color="C3", edgecolor="none", zorder=3
            )
            ax.annotate(
                f"{n_zero} at $R'=0$",
                xy=(0.0, n_zero),
                xytext=(4, 2),
                textcoords="offset points",
                fontsize=7,
                color="C3",
            )
        if at_m1[name]:
            ax.axvline(-1.0, color="C3", lw=0.8, ls="--", zorder=4)
            ax.annotate(
                f"{at_m1[name]} at $R'=-1$",
                xy=(-1.0, ax.get_ylim()[1]),
                xytext=(3, -10),
                textcoords="offset points",
                fontsize=7,
                color="C3",
            )
        ax.axvline(0.0, color="k", lw=0.6)
        ax.set_title(name, fontsize=8)
        ax.set_xlabel(r"$R'$")
    where = (
        "daily sum over the twelve entry windows 10:00-15:30 ET"
        if pooled
        else f"entry {key[:2]}:{key[2:]} ET, {hold_kind(key)}"
    )
    fig.suptitle(
        f"block-diagonal ridge, midpoint fills: {where}; return of the position, "
        r"1st-99th percentile window (bars: the days with $R'\neq 0$)",
        fontsize=9,
    )
    fig.tight_layout()
    dst = OUT / key / f"rule_hists_blk2_{key}.png"
    dst.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dst, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return key, str(dst.relative_to(ROOT).as_posix()), at_m1


def render_confusion(work: pd.DataFrame) -> Path:
    """12 clocks: 2x2 of sign(s) vs sign(R) of the long package (mid)."""
    clocks = sorted(work["hhmm"].unique())
    fig, axes = plt.subplots(3, 4, figsize=(10.8, 7.2))
    s = work["rv_hat_blk2"].to_numpy(float) - work["slice"].to_numpy(float)
    r = work["R"].to_numpy(float)
    hh = work["hhmm"].to_numpy()
    pred_long = np.where(np.isfinite(s), s > 0, False)
    act_up = np.isfinite(r) & (r > 0)
    for ax, c in zip(axes.ravel(), clocks):
        m = hh == c
        tp = int((m & pred_long & act_up).sum())
        fp = int((m & pred_long & ~act_up).sum())
        fn = int((m & ~pred_long & act_up).sum())
        tn = int((m & ~pred_long & ~act_up).sum())
        mat = np.array([[tp, fp], [fn, tn]], float)
        n = mat.sum()
        ax.imshow(mat, cmap="Blues", vmin=0, vmax=max(n / 2, 1))
        for (i, j), v in np.ndenumerate(mat):
            ax.text(
                j,
                i,
                f"{int(v)}\n({100 * v / n:.0f}%)" if n else "0",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if v > n / 3 else "black",
            )
        acc = (tp + tn) / n if n else float("nan")
        ax.set_title(f"{c}  acc {acc:.2f}", fontsize=9)
        ax.set_xticks([0, 1], labels=["R>0", "R≤0"], fontsize=7)
        ax.set_yticks([0, 1], labels=["s>0 buy", "s≤0 short"], fontsize=7)
    fig.suptitle(
        r"$\mathrm{sign}(s)$ vs sign of long-package $R$ (mid), block-diagonal ridge, 866 days",
        fontsize=10,
    )
    fig.tight_layout()
    dst = FIG_DIR / "confusion_by_clock.png"
    fig.savefig(dst, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return dst


def hist_caption(key: str, at_1530: bool) -> str:
    what = (
        "the daily sum over the twelve entry windows"
        if key == "pooled"
        else f"the position taken at {key[:2]}:{key[2:]} ET"
    )
    tail = (
        r"; the mass at $-1$ is the days the package expires worthless"
        if at_1530
        else ""
    )
    return (
        r"\small Distribution of the return of %s, block-diagonal ridge forecast, "
        r"one panel per rule (returns inside their 1st--99th percentile window%s)."
        % (what, tail)
    )


def build_tables(
    work: pd.DataFrame, clocks: list[str]
) -> dict[str, dict[str, pd.DataFrame]]:
    """{window key -> {rule stem -> frame indexed by model label}}."""
    hhmm = work["hhmm"].to_numpy()
    pos = {tag: positions(work, tag) for tag in asl.MODEL_ORDER}
    tables: dict[str, dict[str, pd.DataFrame]] = {}
    for key in [c.replace(":", "") for c in clocks] + ["pooled"]:
        pooled = key == "pooled"
        mask = np.ones(len(work), bool) if pooled else (hhmm == f"{key[:2]}:{key[2:]}")
        short = summary_row(work, -np.ones(len(work)), mask, pooled)
        sign_rows = {
            asl.YHAT_LABEL[tag]: summary_row(work, pos[tag], mask, pooled)
            for tag in asl.MODEL_ORDER
        }
        tables[key] = {
            "always_short": pd.DataFrame({"all models": short}).T,
            "sign_s": pd.DataFrame(sign_rows).T,
        }
    return tables


# ------------------------------------------------------------------ tex ----
def render(tabs: dict[str, pd.DataFrame], source: str, *, pooled: bool) -> list[str]:
    cols = cols_for(pooled)
    ncol = len(cols) + 1
    lines = [
        "% AUTO-GENERATED by writeup/make_rule_by_strategy_intraday_tex.py"
        " -- do not edit.",
        f"% Source: {source}",
        r"\begingroup\small\setlength{\tabcolsep}{3.6pt}",
        r"\begin{tabular}{l" + "r" * len(cols) + "}",
        r"\toprule",
        "& " + " & ".join(h for _, _, h in cols) + r" \\",
    ]
    for stem, label in PANELS:
        df = tabs[stem]
        lines.append(r"\midrule")
        lines.append(r"\multicolumn{%d}{l}{\emph{%s}} \\" % (ncol, label))
        for name, row in df.iterrows():
            cells = [MODEL_TEX.get(str(name), str(name))]
            for col, fmt, _ in cols:
                cells.append(fmt.format(float(row[col])))
            lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\endgroup", ""]
    return lines


def hold_kind(key: str) -> str:
    if DH_HOLDCLOSE:
        return "held to cash-settle"
    if key == "1530":
        return "cash-settled at the close"
    return "one-bar hold"


def window_title(key: str) -> str:
    inst = "straddle" if DH_HOLDCLOSE else "package"
    if key == "pooled":
        extra = (
            r" (overlapping remaining-session books)"
            if DH_HOLDCLOSE
            else r" (daily sums)"
        )
        return (
            rf"0DTE nearest-OTM {inst}: per-rule return summary, pooled over the twelve "
            rf"entry windows 10:00--15:30 ET{extra}, models compared on the same 866 days"
        )
    return (
        rf"0DTE nearest-OTM {inst}: per-rule return summary, entry {key[:2]}:{key[2:]} ET, "
        rf"{hold_kind(key)}, models compared on the same 866 days"
    )


def short_title(key: str) -> str:
    if key == "pooled":
        return "Rule table by strategy --- pooled over the twelve entry windows"
    return f"Rule table by strategy --- entry {key[:2]}:{key[2:]} ET, {hold_kind(key)}"


def data_note(key: str, n_days: int) -> str:
    inst = "straddle" if DH_HOLDCLOSE else "package"
    if DH_HOLDCLOSE:
        if key == "pooled":
            held = (
                r"Each row is the daily sum of the twelve overlapping remaining-session "
                r"books: enter at each clock 10:00--15:30 ET, hold those strikes to the "
                r"official close, delta-hedge every 30 minutes."
            )
        else:
            held = (
                rf"One trade a day: enter the nearest-OTM {inst} at {key[:2]}:{key[2:]} ET, "
                r"hold those strikes to the official close, and delta-hedge every 30 minutes."
            )
        crossed = (
            r"Fills are at the quoted midpoint everywhere except the last column, which is "
            r"the same rule at the crossed spread: entry at the touch (the ask when long, "
            r"the bid when short) and cash-settle at the official close (no exit spread)."
        )
        return (
            r"\noindent\small %s %d expiration days, 2020-01-03 to 2024-04-30 "
            r"(the deck's frame). %s" % (held, n_days, crossed)
        )
    if key == "pooled":
        held = (
            r"Each row is the daily sum over the twelve entry windows 10:00--15:30 ET: "
            r"one number per expiration day, then the usual statistics on that daily series."
        )
    elif key == "1530":
        held = (
            r"One trade a day: enter the nearest-OTM package at 15:30 ET and let it "
            r"cash-settle at the official close."
        )
    else:
        held = (
            r"One trade a day: enter the nearest-OTM package at %s:%s ET and exit thirty "
            r"minutes later at the next stamp, in the same two strikes."
            % (key[:2], key[2:])
        )
    return (
        r"\noindent\small %s "
        r"%d expiration days, 2020-01-03 to 2024-04-30 (the deck's frame). "
        r"Fills are at the quoted midpoint everywhere except the last column, which is the "
        r"same rule filled at the crossed spread: entry at the touch (the ask when long, the "
        r"bid when short) and exit at the touch of the next stamp; the 15:30 leg cash-settles "
        r"and pays no exit spread, and a re-pick that lands on the same two strikes with the "
        r"same sign is a hold, not a round trip, so it is not charged." % (held, n_days)
    )


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print("wrote", path.relative_to(ROOT))


LOG_BAD = ("Undefined", "not found")
AUX_SUFFIXES = (".aux", ".log", ".out")


FIG_WIDTHS = (0.68, 0.60, 0.52, 0.45, 0.38, 0.32, 0.27)


def compile_tex(job: tuple[str, str, int]) -> tuple[str, int, float, int]:
    """Write one standalone and compile it, shrinking the figure to fit.

    `template` carries the placeholder @FIGW@ for the \\includegraphics width;
    the widths are tried in order until the document fits `max_pages`, so a
    table plus a figure stays on one landscape page instead of spilling.
    pdflatex is given a distinct jobname per stem, so the fourteen compiles
    are independent and run in parallel.  The log is scanned for
    LOG_BAD (a missing graphic reports "not found" there) and the aux files
    are removed, leaving only the .tex and the .pdf.
    """
    stem, template, max_pages = job
    pages, used = -1, float("nan")
    for width in FIG_WIDTHS:
        body = template.replace("@FIGW@", f"{width:.2f}")
        with open(WRITEUP / (stem + ".tex"), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(body)
        r = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", stem + ".tex"],
            cwd=WRITEUP,
            capture_output=True,
            text=True,
        )
        pdf = WRITEUP / (stem + ".pdf")
        log = WRITEUP / (stem + ".log")
        text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
        bad = [ln for ln in text.splitlines() if any(b in ln for b in LOG_BAD)]
        assert r.returncode == 0 and pdf.exists(), f"{stem}: pdflatex rc={r.returncode}"
        assert not bad, f"{stem}: log carries {LOG_BAD}:\n" + "\n".join(bad[:5])
        pages, used = len(PdfReader(str(pdf)).pages), width
        if "@FIGW@" not in template or pages <= max_pages:
            break
    for suf in AUX_SUFFIXES:  # keep only the .tex and the .pdf beside the standalone
        (WRITEUP / (stem + suf)).unlink(missing_ok=True)
    n_fig = template.count(r"\includegraphics")
    return stem, pages, used, n_fig


def _pdf_text(path: Path) -> str:
    """Whitespace-collapsed text of a PDF."""
    return " ".join(
        " ".join(p.extract_text() or "" for p in PdfReader(str(path)).pages).split()
    )


def _row_numbers(key: str) -> list[str]:
    """The numeric cells of each data row of one generated tabular, in order."""
    src = (GEN / f"table_{DOC}_{key}.tex").read_text(encoding="utf-8")
    rows = []
    for line in src.splitlines():
        line = line.strip()
        if not line.endswith(r"\\") or line.startswith((r"\multicolumn", "&")):
            continue
        cells = [c.strip() for c in line[:-2].split("&")]
        rows.append(" ".join(cells[1:]))
    return rows


def check_bundle_matches_standalones(keys: list[str]) -> None:
    """Every window's table in the bundle is the one in its own standalone."""
    bundle = _pdf_text(WRITEUP / f"{DOC}_index.pdf")
    for key in keys:
        alone = _pdf_text(WRITEUP / f"{DOC}_{key}.pdf")
        rows = _row_numbers(key)
        assert len(rows) == 9, (key, len(rows))
        for row in rows:
            assert row in alone, f"{key}: row not rendered in its standalone: {row}"
            assert row in bundle, f"{key}: row not rendered in the bundle: {row}"
    print(
        f"bundle vs standalones: all {len(keys)} windows, "
        f"{9 * len(keys)} data rows, render identically"
    )


# ----------------------------------------------------------------- main ----
def gate(work: pd.DataFrame, deck: pd.DataFrame, tables: dict[str, Any]) -> None:
    """The 15:30 window against the deck's numbers of record.

    With the censored implied re-inverted the way the deck does it, the 15:30
    leg IS the deck's close trade -- same strikes, same entry, same forecast,
    same implied, w = 1 -- so the positions must agree on EVERY day and the
    whole ridge row must reproduce the deck's, not merely come close.
    """
    at15 = (work["hhmm"] == "15:30").to_numpy()
    c15 = work[at15].set_index("date")
    pos = positions(work, "blk2")[at15]
    j = pd.DataFrame({"pos_nb": pos}, index=c15.index).join(deck[["pos"]], how="inner")
    assert len(j) == len(deck), (len(j), len(deck))
    dis = j.index[j["pos_nb"].to_numpy() != j["pos"].to_numpy(float)]
    assert len(dis) == 0, (
        "the 15:30 sign(s) positions disagree with the deck on "
        f"{len(dis)} days: {', '.join(str(d.date()) for d in dis)}"
    )
    print(f"GATE positions: equal to the deck's on all {len(j)} days, none differing")

    if DH_HOLDCLOSE:
        print("DH t→T: skip 15:30 Sharpe-vs-deck gate (payoff is hedged)")
        return
    got = float(tables["1530"]["always_short"].loc["all models", "Sharpe_ann"])
    assert abs(got - DECK_ALWAYS_SHORT_SHARPE) < 1e-6, (got, DECK_ALWAYS_SHORT_SHARPE)
    print(
        f"GATE always short at 15:30: {got:.9f} vs the deck's "
        f"{DECK_ALWAYS_SHORT_SHARPE:.9f} -- match"
    )
    sgn = float(tables["1530"]["sign_s"].loc["block-diagonal ridge", "Sharpe_ann"])
    assert abs(sgn - DECK_SIGN_S_RIDGE_SHARPE) < 1e-6, (sgn, DECK_SIGN_S_RIDGE_SHARPE)
    print(
        f"GATE sign(s) ridge at 15:30: {sgn:.9f} vs the deck's "
        f"{DECK_SIGN_S_RIDGE_SHARPE:.9f} -- match"
    )

    # The whole 15:30 panel against the deck's own CSV, every shared column.
    cols = [c for c, _, _ in COLS if c != "Sharpe_crossed"]
    for stem, csv in (("sign_s", "sign_s"), ("always_short", "always_short")):
        ref = pd.read_csv(DECK / f"rule_by_strategy_{csv}.csv", index_col=0)
        mine = tables["1530"][stem]
        assert list(mine.index) == list(ref.index), (list(mine.index), list(ref.index))
        dev = float((mine[cols] - ref[cols]).abs().to_numpy().max())
        assert dev < 1e-6, f"the 15:30 {stem} panel differs from the deck's by {dev}"
        print(f"GATE 15:30 {stem} panel vs the deck's CSV: max |diff| {dev:.3g}")


def figure_block(key: str, at_1530: bool) -> list[str]:
    """The window's histogram, plus -- pooled only -- the two per-clock figures."""
    rel = f"../{OUT.relative_to(ROOT).as_posix()}/{key}"
    out = [
        r"\begin{center}",
        r"\includegraphics[width=@FIGW@\textwidth]{%s/rule_hists_blk2_%s.png}\par\smallskip"
        % (rel, key),
        hist_caption(key, at_1530),
        r"\end{center}",
    ]
    if key == "pooled":
        base = f"../{FIG_DIR.relative_to(ROOT).as_posix()}"
        out += [
            r"\begin{center}",
            r"\includegraphics[width=@FIGW@\textwidth]{%s/mean_by_entry_hhmm_as.png}"
            r"\par\smallskip" % base,
            r"\small Mean $R'$ and Sharpe$_{\mathrm{clk}}$ by entry clock (grouped bars). "
            r"Sharpe$_{\mathrm{clk}}$ is that clock's daily series $\times\sqrt{252}$, not "
            r"the pooled Sharpe$_{\mathrm{ann}}$. "
            + (
                r"Delta-hedged $t\to T$: hold entry $K$ to official close, "
                r"rebalance $\Delta$ every 30 minutes."
                if DH_HOLDCLOSE
                else r"Next-mid 30-minute holds 10:00--15:00; 15:30 cash-settles at the official close."
            ),
            r"\end{center}",
            r"\begin{center}",
            r"\includegraphics[width=@FIGW@\textwidth]{%s/confusion_by_clock.png}"
            r"\par\smallskip" % base,
            r"\small Confusion matrix per entry clock: $\mathrm{sign}(s)$ (buy if $s>0$) "
            r"against the sign of the long-package midpoint return. 15:30 is the paper trade.",
            r"\end{center}",
        ]
    return out


def render_mean_bars(work: pd.DataFrame) -> Path:
    """Grouped bars of mean R' and Sharpe_clk for always-short and sign(s) ridge."""
    clocks = sorted(work["hhmm"].unique())
    q_as = -np.ones(len(work))
    q_sg = positions(work, "blk2")
    rows = []
    for name, q in (("always short", q_as), ("sign(s)", q_sg)):
        rp = q * work["R"].to_numpy(float)
        for c in clocks:
            m = work["hhmm"].to_numpy() == c
            x = rp[m]
            x = x[np.isfinite(x)]
            sh = (
                float(x.mean() / x.std(ddof=1) * np.sqrt(asl.PERIODS_PER_YEAR))
                if len(x) > 1
                else np.nan
            )
            rows.append(
                {"hhmm": c, "rule": name, "mean": float(np.mean(x)), "Sharpe_clk": sh}
            )
    stab = pd.DataFrame(rows)
    hh = clocks
    x = np.arange(len(hh))
    w = 0.35
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.4), sharex=True)
    for i, rule in enumerate(("always short", "sign(s)")):
        sub = stab[stab["rule"] == rule].set_index("hhmm").reindex(hh)
        axes[0].bar(x + (i - 0.5) * w, sub["mean"].to_numpy(float), w, label=rule)
        axes[1].bar(x + (i - 0.5) * w, sub["Sharpe_clk"].to_numpy(float), w, label=rule)
    for ax, ylab in ((axes[0], "mean $R'$"), (axes[1], r"Sharpe$_{\mathrm{clk}}$")):
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylabel(ylab)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper left")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(hh, rotation=45, ha="right")
    title = (
        r"delta-hedged $t\to T$"
        if DH_HOLDCLOSE
        else "next-mid 30-min holds; 15:30 cash-settles"
    )
    axes[0].set_title(title)
    fig.tight_layout()
    dst = FIG_DIR / "mean_by_entry_hhmm_as.png"
    fig.savefig(dst, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return dst


def main() -> None:
    GEN.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    work, clocks, deck = build_work()
    if DH_HOLDCLOSE:
        work = attach_long_dh(work)
        render_mean_bars(work)
    tables = build_tables(work, clocks)
    gate(work, deck, tables)
    keys = [c.replace(":", "") for c in clocks] + ["pooled"]

    for key in keys:
        for stem, _ in PANELS:
            dst = OUT / key / f"rule_by_strategy_{stem}.csv"
            dst.parent.mkdir(parents=True, exist_ok=True)
            tables[key][stem].to_csv(dst)
        src = (
            f"results/atm_straddle_intraday/rule_by_strategy/{key}/"
            f"rule_by_strategy_*.csv"
        )
        write(
            GEN / f"table_{DOC}_{key}.tex",
            "\n".join(render(tables[key], src, pooled=(key == "pooled"))),
        )

    # --- the thirteen histograms, in parallel (Agg, one figure per worker) ---
    jobs = [(key, hist_payload(work, key)) for key in keys]
    t0 = time.perf_counter()
    with cf.ProcessPoolExecutor(max_workers=WORKERS) as pool:
        figs = dict((k, (p, m)) for k, p, m in pool.map(render_hist, jobs))
    for key in keys:
        print(f"  {figs[key][0]}  mass at -1: {figs[key][1]}")
        assert (ROOT / figs[key][0]).exists(), figs[key][0]
    render_confusion(work)
    for extra in ("mean_by_entry_hhmm_as.png", "confusion_by_clock.png"):
        p = FIG_DIR / extra
        assert p.exists(), f"the pooled section's figure is missing: {p}"
    print(f"13 histograms rendered in {time.perf_counter() - t0:.1f}s")

    # --- the fourteen documents, written as templates and compiled in parallel ---
    compile_jobs = []
    for key in keys:
        n_days = int(tables[key]["always_short"].loc["all models", "n"])
        body = "\n".join(
            [
                f"% Standalone render of {GEN_REL}/table_{DOC}_{key}.tex",
                f"% Build: pdflatex -interaction=nonstopmode {DOC}_{key}.tex",
                PREAMBLE,
                r"\begin{document}",
                r"\begin{center}",
                r"\textbf{%s}\\[1.5ex]" % window_title(key),
                r"\input{%s/table_%s_%s}" % (GEN_REL, DOC, key),
                r"\end{center}",
                r"\vspace{1ex}",
                *figure_block(key, key == "1530"),
                r"\vspace{1ex}",
                data_note(key, n_days),
                r"\vspace{1ex}",
                footnote(),
                r"\end{document}",
                "",
            ]
        )
        compile_jobs.append((f"{DOC}_{key}", body, 2 if key == "pooled" else 1))

    # The bundle: the same thirteen tabulars and figures, in order, one section
    # each with its own data note; the shared footnote once, at the end.
    idx = [
        "% AUTO-GENERATED by writeup/make_rule_by_strategy_intraday_tex.py.",
        f"% Build: pdflatex -interaction=nonstopmode {DOC}_index.tex",
        PREAMBLE,
        r"\begin{document}",
        r"\begin{center}\textbf{\large 0DTE nearest-OTM "
        + ("straddle" if DH_HOLDCLOSE else "package")
        + r": rule table by strategy, every intraday entry window"
        + (r" --- delta-hedged, held to cash-settle}" if DH_HOLDCLOSE else r"}")
        + r"\end{center}",
    ]
    for key in keys:
        n_days = int(tables[key]["always_short"].loc["all models", "n"])
        idx += [
            r"\section*{%s}" % short_title(key),
            r"\begin{center}",
            r"\input{%s/table_%s_%s}" % (GEN_REL, DOC, key),
            r"\end{center}",
            *figure_block(key, key == "1530"),
            data_note(key, n_days),
            r"\clearpage",
        ]
    idx += [r"\section*{Notes}", footnote(), r"\end{document}", ""]
    compile_jobs.append((f"{DOC}_index", "\n".join(idx), 99))

    t0 = time.perf_counter()
    with cf.ProcessPoolExecutor(max_workers=WORKERS) as pool:
        built = list(pool.map(compile_tex, compile_jobs))
    for stem, pages, width, n_fig in built:
        print(
            f"  {stem}: {pages} page(s), {n_fig} figure(s) at "
            f"{width:.2f}\\textwidth, log clean, aux removed"
        )
    for stem, pages, _, _ in built[:-1]:
        cap = 2 if stem.endswith("pooled") else 1
        assert pages <= cap, f"{stem} spilled to {pages} pages (cap {cap})"
    print(f"14 documents compiled in {time.perf_counter() - t0:.1f}s")

    pages = built[-1][1]
    print(f"BUNDLE {DOC}_index.pdf: {pages} pages")
    check_bundle_matches_standalones(keys)

    print()
    print("sign(s) Sharpe by forecast -- 15:30 and pooled, mid and crossed")
    rep = pd.DataFrame(
        {
            "15:30 mid": tables["1530"]["sign_s"]["Sharpe_ann"],
            "15:30 crossed": tables["1530"]["sign_s"]["Sharpe_crossed"],
            "pooled mid": tables["pooled"]["sign_s"]["Sharpe_ann"],
            "pooled crossed": tables["pooled"]["sign_s"]["Sharpe_crossed"],
        }
    )
    print(rep.to_string(float_format=lambda x: f"{x:+.4f}"))
    print()
    print("always short, Sharpe mid / crossed")
    ash = pd.DataFrame(
        {
            "15:30": tables["1530"]["always_short"].loc["all models"],
            "pooled": tables["pooled"]["always_short"].loc["all models"],
        }
    ).T[["n", "mean", "std", "t_mean", "Sharpe_ann", "Sharpe_crossed"]]
    print(ash.to_string(float_format=lambda x: f"{x:+.4f}"))


if __name__ == "__main__":
    main()
