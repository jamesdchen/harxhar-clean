"""Study 73 -- Vilkov (SSRN 4641356, "0DTE Trading Rules") checked against our deck.

Three comparisons on the 866 deck days (results/atm_straddle_0dte_1530/daily_<tag>.parquet),
the second deck (blk2) beside the first where it helps:

  (i)  Vilkov's implied variance at 15:30 -- the VIX formula on the mids of the LISTED strikes
       with K/S in [0.98, 1.02] (he Akima-interpolates to a 0.001 grid; we trade listed strikes,
       so no interpolation here; r = 0; F from the strike minimising |C - P|; K0 = largest strike
       <= F; dK one-sided at the window edges; Q(K0) = average of call and put) -- against
       (a) the deck's ATM slice iv_var (the re-inverted package variance) and (b) the full-chain
       strip C of experiments/spxw_mfiv_toclose.py::_mfiv_one.  Pre-registered expectation: within
       a few percent of the slice, far below the strip; report the zero-bid share of the window.
  (ii) His Panel-C regression (15:00 -> close straddle PnL on IV and RV, R^2 0.20-0.31, IV t
       -2.7 / -3.8, RV t +2.5 / +5.2) at 15:30 on our deck: R and the %-of-spot PnL on the implied
       slice and the realized last-bar variance, OLS with Newey-West (3 lags) t-stats; IV-only,
       RV-only, both, and the ex-ante version with rv_hat in place of realized RV.
 (iii) The sign(s) book in his units (daily PnL in % of spot at mid and crossed): mean/day, vol,
       Sharpe (sqrt(252), his convention), ES1%, worst day, worst 5 consecutive days, max drawdown,
       skew, loss probability of the long and short legs, turnover proxy (|premium| + half-spread
       in bp of spot), |mean| / ES1%; the same for always-short, always-long and the long leg alone;
       his straddle-basket numbers beside ours, with the caveats stated in the table.

Gates: R = exit / entry - 1 and signal = rv_hat - iv_var on every deck day; the chain's 15:30 legs
at (K_c, K_p) reproduce the deck's leg mids and spot; the strip C recomputed here equals the
MFIV panel diagnostics' C at 15:30; realized last-bar variance matched on every day.

ONE chain read, filtered to the 866 15:30 stamps (pyarrow filter); never the whole chain.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "experiments", ROOT / "notebooks"):
    sys.path.insert(0, str(p))

OUT = ROOT / "results" / "atm_straddle_intraday_holdclose" / "proposals" / "73"
DECK = ROOT / "results" / "atm_straddle_0dte_1530"
CHAIN = ROOT / "data" / "spxw_chain.parquet"
CORE = ROOT / "data" / "core_stats.parquet"
MFIV_DIAG = ROOT / "results" / "spxw_pnl" / "mfiv_panel_diag.parquet"
MFIV_SRC = ROOT / "experiments" / "spxw_mfiv_toclose.py"
TAGS = ("sub_live_ridge", "blk2")
WINDOW = (0.98, 1.02)  # Vilkov's moneyness window K/S (paper section 2)
NW_LAGS = 3  # his Newey-West lag count
ANN = float(
    np.sqrt(252.0)
)  # his annualisation; the repo's trades-per-year is 200.3/yr (x0.89)
ES_LEVEL = 0.01
WORST_RUN = 5
GATE_TOL = 1e-6
# Vilkov, implementable-cost table, equal-weight strangle/straddle basket, 10:00 entry, 953 days
# 2016-2024, corrected costs (KNOWN-ISSUES Aug 2026), long side, % of spot.
VILKOV = {
    "mean_pct_spot": -0.0215,
    "Sharpe": -0.97,
    "ES1_pct_spot": 1.61,
    "worst_day": -2.67,
    "worst_5d": -7.77,
    "max_dd": 25.9,
    "loss_prob_long": 69.9,
}


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ data --
def load_deck(tag: str) -> pd.DataFrame:
    d = pd.read_parquet(DECK / f"daily_{tag}.parquet").sort_index()
    d.index = pd.DatetimeIndex(pd.to_datetime(d.index)).normalize()
    r = d["exit"].to_numpy(float) / d["entry"].to_numpy(float) - 1.0
    assert np.nanmax(np.abs(r - d["R"].to_numpy(float))) < GATE_TOL
    s = d["rv_hat"].to_numpy(float) - d["iv_var"].to_numpy(float)
    assert np.nanmax(np.abs(s - d["signal"].to_numpy(float))) < GATE_TOL
    return d


def realized_last_bar(dates: pd.DatetimeIndex) -> pd.Series:
    rv = pd.read_parquet(CORE, columns=["endbartime", "sumret2"])
    t = pd.to_datetime(rv["endbartime"])
    last = rv[t.dt.strftime("%H:%M") == "16:00"]
    out = pd.Series(last["sumret2"].to_numpy(float), index=t[last.index].dt.normalize())
    out = out.reindex(dates)
    assert out.notna().all(), "realized last-bar variance missing on a deck day"
    return out


def chain_1530(stamps: pd.Series) -> pd.DataFrame:
    ts = sorted(pd.to_datetime(stamps, utc=True).drop_duplicates())
    ch = pq.read_table(
        CHAIN,
        columns=[
            "timestamp",
            "expiration",
            "strike",
            "cp",
            "bid",
            "ask",
            "mid",
            "underlying_price",
        ],
        filters=[("timestamp", "in", [pd.Timestamp(t).to_pydatetime() for t in ts])],
    ).to_pandas()
    ch["timestamp"] = pd.to_datetime(ch["timestamp"], utc=True)
    ch["cp"] = ch["cp"].astype(str).str.upper().str[0]
    for c in ("strike", "bid", "ask", "mid", "underlying_price"):
        ch[c] = ch[c].astype(float)
    return ch


# ------------------------------------------------------------- Vilkov IV --
def vilkov_iv(snap: pd.DataFrame) -> dict[str, float]:
    """VIX formula on the listed strikes inside K/S in WINDOW, r = 0, no interpolation.

    Returns the window integral (all positive mids, his convention), the same with the
    zero-bid strikes dropped, the share of the integral carried by zero-bid strikes, F, K0
    and the strike count.  NaN when the window holds fewer than two usable strikes.
    """
    nan = {
        k: float("nan") for k in ("iv", "iv_nozero", "zero_bid_share", "F", "K0", "n_K")
    }
    live = snap[np.isfinite(snap["mid"]) & (snap["mid"] > 0)]
    spot = live["underlying_price"].dropna()
    if spot.empty:
        return nan
    S = float(spot.iloc[-1])
    w = live[(live["strike"] / S >= WINDOW[0]) & (live["strike"] / S <= WINDOW[1])]
    piv = w.pivot_table(
        index="strike", columns="cp", values=["mid", "bid"], aggfunc="last"
    )
    if ("mid", "C") not in piv.columns or ("mid", "P") not in piv.columns:
        return nan
    both = piv.dropna(subset=[("mid", "C"), ("mid", "P")])
    if both.empty:
        return nan
    gap = (both[("mid", "C")] - both[("mid", "P")]).abs()
    k_star = float(gap.idxmin())
    F = k_star + float(both.loc[k_star, ("mid", "C")] - both.loc[k_star, ("mid", "P")])
    strikes = piv.index.to_numpy(float)
    below = strikes[strikes <= F]
    if below.size == 0:
        return nan
    K0 = float(below.max())
    rows = []
    for k in strikes:
        if k < K0:
            q, b = piv.loc[k, ("mid", "P")], piv.loc[k, ("bid", "P")]
        elif k > K0:
            q, b = piv.loc[k, ("mid", "C")], piv.loc[k, ("bid", "C")]
        else:
            qc, qp = piv.loc[k, ("mid", "C")], piv.loc[k, ("mid", "P")]
            bc, bp = piv.loc[k, ("bid", "C")], piv.loc[k, ("bid", "P")]
            if np.isfinite(qc) and np.isfinite(qp):
                q, b = 0.5 * (qc + qp), min(bc, bp)
            elif np.isfinite(qc):
                q, b = qc, bc
            else:
                q, b = qp, bp
        if np.isfinite(q) and q > 0:
            rows.append((float(k), float(q), bool(b <= 0)))
    if len(rows) < 2:
        return nan
    K = np.array([r[0] for r in rows])
    Q = np.array([r[1] for r in rows])
    zero = np.array([r[2] for r in rows])
    dK = np.empty_like(K)
    dK[0] = K[1] - K[0]
    dK[-1] = K[-1] - K[-2]
    if K.size > 2:
        dK[1:-1] = 0.5 * (K[2:] - K[:-2])
    contrib = 2.0 * dK / K**2 * Q
    fwd = (F / K0 - 1.0) ** 2
    total = float(contrib.sum())
    return {
        "iv": total - fwd,
        "iv_nozero": float(contrib[~zero].sum()) - fwd,
        "zero_bid_share": float(contrib[zero].sum() / total)
        if total > 0
        else float("nan"),
        "F": F,
        "K0": K0,
        "n_K": float(K.size),
    }


# ------------------------------------------------------------------ OLS --
def ols_hac(
    y: np.ndarray, X: np.ndarray, lags: int
) -> tuple[np.ndarray, np.ndarray, float, float, int]:
    """OLS with an intercept; Newey-West (Bartlett) covariance with ``lags`` lags.

    Returns (beta, t, R^2, adjusted R^2, n); beta[0] and t[0] are the intercept.
    """
    X1 = np.column_stack([np.ones(len(y)), X])
    n, k = X1.shape
    xtx_inv = np.linalg.inv(X1.T @ X1)
    b = xtx_inv @ X1.T @ y
    e = y - X1 @ b
    xe = X1 * e[:, None]
    s = xe.T @ xe
    for lag in range(1, lags + 1):
        g = xe[lag:].T @ xe[:-lag]
        s += (1.0 - lag / (lags + 1.0)) * (g + g.T)
    v = xtx_inv @ s @ xtx_inv
    r2 = float(1.0 - e.var() / y.var())
    return (
        b,
        b / np.sqrt(np.diag(v)),
        r2,
        float(1.0 - (1.0 - r2) * (n - 1) / (n - k)),
        n,
    )


# ----------------------------------------------------------------- book --
def book_stats(
    pnl: np.ndarray, pos: np.ndarray, turnover_bp: np.ndarray
) -> dict[str, float]:
    """Vilkov-style statistics of a daily PnL series in % of spot (his table units)."""
    pnl = np.asarray(pnl, float)
    n = pnl.size
    k = max(1, int(np.ceil(ES_LEVEL * n)))
    worst = np.sort(pnl)[:k]
    cum = np.cumsum(pnl)
    dd = np.maximum.accumulate(cum) - cum
    run5 = np.convolve(pnl, np.ones(WORST_RUN), mode="valid") if n >= WORST_RUN else pnl
    longs, shorts = pos > 0, pos < 0
    mean, sd = float(pnl.mean()), float(pnl.std(ddof=1))
    sk = float(((pnl - mean) ** 3).mean() / sd**3) if sd > 0 else float("nan")
    active = pos != 0
    return {
        "days": int(n),
        "active_days": int(active.sum()),
        "mean_pct_spot": mean,
        "vol_pct_spot": sd,
        "Sharpe": mean / sd * ANN if sd > 0 else float("nan"),
        "ES1_pct_spot": float(-worst.mean()),
        "worst_day": float(pnl.min()),
        "worst_5d": float(run5.min()),
        "max_dd": float(dd.max()),
        "skew": sk,
        "loss_prob_long": float(100.0 * (pnl[longs] < 0).mean())
        if longs.any()
        else float("nan"),
        "loss_prob_short": float(100.0 * (pnl[shorts] < 0).mean())
        if shorts.any()
        else float("nan"),
        "turnover_bp": float(turnover_bp[active].mean()) if active.any() else 0.0,
        "abs_mean_over_ES1": abs(mean) / float(-worst.mean())
        if worst.mean() < 0
        else float("nan"),
    }


def main() -> None:  # noqa: PLR0915
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    mfiv = _load(MFIV_SRC, "p_mfiv_toclose")
    deck = load_deck(TAGS[0])
    deck2 = load_deck(TAGS[1])
    assert deck.index.equals(deck2.index)
    idx = deck.index
    rv_next = realized_last_bar(idx)

    # ---------------------------------------------------- (i) the three IVs --
    ch = chain_1530(deck["timestamp_c"])
    stamp_of = pd.Series(
        pd.to_datetime(deck["timestamp_c"], utc=True).to_numpy(), index=idx
    )
    rows = []
    for d, ts in stamp_of.items():
        snap = ch[ch["timestamp"] == ts]
        v = vilkov_iv(snap)
        meta = mfiv._mfiv_one(snap, pd.Timestamp(ts), pd.Timestamp(d))
        c_strip = float(meta["mfiv_int"]) if meta is not None else float("nan")
        # alignment gate: the deck's legs and spot are on this snapshot
        kc, kp = float(deck.loc[d, "K_c"]), float(deck.loc[d, "K_p"])
        mc = snap.loc[(snap["cp"] == "C") & (snap["strike"] == kc), "mid"]
        mp = snap.loc[(snap["cp"] == "P") & (snap["strike"] == kp), "mid"]
        s_chain = snap["underlying_price"].dropna()
        rows.append(
            {
                "date": d,
                "iv_window": v["iv"],
                "iv_window_nozero": v["iv_nozero"],
                "zero_bid_share": v["zero_bid_share"],
                "n_K_window": v["n_K"],
                "F_window": v["F"],
                "strip_C": c_strip,
                "slice_iv_var": float(deck.loc[d, "iv_var"]),
                "rv_next": float(rv_next.loc[d]),
                "leg_gap": max(
                    abs(float(mc.iloc[-1]) - float(deck.loc[d, "mid_c"]))
                    if len(mc)
                    else np.inf,
                    abs(float(mp.iloc[-1]) - float(deck.loc[d, "mid_p"]))
                    if len(mp)
                    else np.inf,
                ),
                "spot_gap": abs(float(s_chain.iloc[-1]) - float(deck.loc[d, "S"]))
                if len(s_chain)
                else np.inf,
            }
        )
    a = pd.DataFrame(rows).set_index("date")
    assert float(a["leg_gap"].max()) < GATE_TOL, float(a["leg_gap"].max())
    assert float(a["spot_gap"].max()) < GATE_TOL, float(a["spot_gap"].max())
    diag = pd.read_parquet(MFIV_DIAG, columns=["t", "C"])
    diag["t"] = pd.to_datetime(diag["t"], utc=True)
    c_diag = diag.set_index("t")["C"].reindex(stamp_of.to_numpy())
    dev_c = float(
        np.nanmax(np.abs(c_diag.to_numpy(float) - a["strip_C"].to_numpy(float)))
    )
    assert dev_c < 1e-12, dev_c
    print(
        f"GATE  chain 15:30 legs = deck mids and spot on {len(a)} days (max gap {a['leg_gap'].max():.1e}); "
        f"strip C recomputed = MFIV panel diagnostics on every stamp ({dev_c:.1e}); "
        f"R and signal reproduce on both decks"
    )
    a.to_csv(OUT / "a_iv_measures_daily.csv")

    ok = np.isfinite(
        a[["iv_window", "iv_window_nozero", "strip_C", "slice_iv_var", "rv_next"]]
    ).all(axis=1)
    ok &= (
        a[["iv_window", "iv_window_nozero", "strip_C", "slice_iv_var", "rv_next"]] > 0
    ).all(axis=1)
    aa = a[ok]
    print(
        f"\n(i)  Vilkov's +-2% window IV vs the ATM slice and the strip at 15:30, {int(ok.sum())} of {len(a)} deck days"
    )
    ratios = pd.DataFrame(
        {
            "window / slice": aa["iv_window"] / aa["slice_iv_var"],
            "window (zero-bid dropped) / slice": aa["iv_window_nozero"]
            / aa["slice_iv_var"],
            "strip / slice": aa["strip_C"] / aa["slice_iv_var"],
            "window / strip": aa["iv_window"] / aa["strip_C"],
        }
    )
    summ = ratios.describe(percentiles=[0.25, 0.5, 0.75]).T[["25%", "50%", "75%"]]
    summ.columns = ["q25", "median", "q75"]
    summ["zero_bid_share_of_window_median"] = np.nan
    summ.loc["window / slice", "zero_bid_share_of_window_median"] = float(
        aa["zero_bid_share"].median()
    )
    print(summ.to_string(float_format=lambda v: f"{v:.3f}"))
    print(
        f"  strikes in the window: median {aa['n_K_window'].median():.0f}; zero-bid share of the window integral: "
        f"median {aa['zero_bid_share'].median():.3f}, IQR {aa['zero_bid_share'].quantile(0.25):.3f}-{aa['zero_bid_share'].quantile(0.75):.3f}"
    )
    logs = np.log(
        aa[["iv_window", "iv_window_nozero", "strip_C", "slice_iv_var", "rv_next"]]
    )
    corr = logs.corr()
    corr.to_csv(OUT / "a_log_correlations.csv")
    summ.to_csv(OUT / "a_ratio_summary.csv")
    print("  log-correlations:")
    print(corr.to_string(float_format=lambda v: f"{v:.3f}"))

    # ------------------------------------------------ (ii) Panel-C at 15:30 --
    print(
        "\n(ii) Vilkov's Panel-C at 15:30 on our deck: straddle PnL on the implied slice and realized variance, NW(3) t-stats"
    )
    reg_rows = []
    for tag, dk in (("sub_live_ridge", deck), ("blk2", deck2)):
        y_r = dk["R"].to_numpy(float)
        y_spot = (
            (dk["exit"].to_numpy(float) - dk["entry"].to_numpy(float))
            / dk["S"].to_numpy(float)
            * 100.0
        )
        iv = dk["iv_var"].to_numpy(float)
        rv = rv_next.to_numpy(float)
        rvh = dk["rv_hat"].to_numpy(float)
        z = lambda v: (v - v.mean()) / v.std(ddof=1)  # noqa: E731
        s_gap = rvh - iv  # the deck's signal: forecast minus implied
        # (spec name, design, regressor names); the last two are the sign(s) construct
        # itself, linearly and as the traded dummy
        specs: list[tuple[str, np.ndarray, list[str]]] = [
            ("IV only", np.column_stack([iv]), ["IV"]),
            ("RV only", np.column_stack([rv]), ["RV"]),
            ("IV + RV (his Panel C, ex post)", np.column_stack([iv, rv]), ["IV", "RV"]),
            ("IV + rv_hat (ex ante)", np.column_stack([iv, rvh]), ["IV", "rv_hat"]),
            ("IV + RV, standardised", np.column_stack([z(iv), z(rv)]), ["IV", "RV"]),
            (
                "IV + rv_hat, standardised",
                np.column_stack([z(iv), z(rvh)]),
                ["IV", "rv_hat"],
            ),
            ("s = rv_hat - IV, standardised", np.column_stack([z(s_gap)]), ["s"]),
            (
                "1{s > 0} (the traded sign)",
                np.column_stack([(s_gap > 0).astype(float)]),
                ["s"],
            ),
        ]
        for yname, y in (("R (return on premium)", y_r), ("PnL % of spot", y_spot)):
            for sname, X, names in specs:
                beta, tstat, _, adj_r2, n_obs = ols_hac(y, X, NW_LAGS)
                row = {
                    "deck": tag,
                    "y": yname,
                    "spec": sname,
                    "n": n_obs,
                    "adj_R2": adj_r2,
                }
                for j, nm in enumerate(names, start=1):
                    row[f"b_{nm}"] = float(beta[j])
                    row[f"t_{nm}"] = float(tstat[j])
                reg_rows.append(row)
    b = pd.DataFrame(reg_rows)
    b.to_csv(OUT / "b_panel_c_1530.csv", index=False)
    show = b[b["deck"] == "sub_live_ridge"][
        [
            "y",
            "spec",
            "n",
            "adj_R2",
            "b_IV",
            "t_IV",
            "b_RV",
            "t_RV",
            "b_rv_hat",
            "t_rv_hat",
            "b_s",
            "t_s",
        ]
    ]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(
        "  Vilkov 15:00 -> close (pooled combos, combo FE, date-clustered): IV t -2.68 / -3.79, RV t +2.47 / +5.24, adj R^2 0.201 / 0.313; at 10:00 R^2 0.056"
    )

    # ------------------------------------------------- (iii) book, his units --
    print(
        "\n(iii) The 15:30 books in Vilkov's units: daily PnL in % of spot, Sharpe with sqrt(252)"
    )
    S = deck["S"].to_numpy(float)
    entry, ex = deck["entry"].to_numpy(float), deck["exit"].to_numpy(float)
    ask = (deck["ask_c"] + deck["ask_p"]).to_numpy(float)
    bid = (deck["bid_c"] + deck["bid_p"]).to_numpy(float)
    half = 0.5 * (ask - bid)
    turnover_bp = (entry + half) / S * 1e4
    long_mid, long_x = (ex - entry) / S * 100.0, (ex - ask) / S * 100.0
    short_mid, short_x = (entry - ex) / S * 100.0, (bid - ex) / S * 100.0
    q = np.where(deck["signal"].to_numpy(float) > 0, 1.0, -1.0)
    books = {
        "sign(s), mid": (np.where(q > 0, long_mid, short_mid), q),
        "sign(s), crossed": (np.where(q > 0, long_x, short_x), q),
        "always short, mid": (short_mid, -np.ones_like(q)),
        "always short, crossed": (short_x, -np.ones_like(q)),
        "always long, mid": (long_mid, np.ones_like(q)),
        "always long, crossed": (long_x, np.ones_like(q)),
        "long leg alone, mid": (
            np.where(q > 0, long_mid, 0.0),
            np.where(q > 0, 1.0, 0.0),
        ),
        "long leg alone, crossed": (
            np.where(q > 0, long_x, 0.0),
            np.where(q > 0, 1.0, 0.0),
        ),
    }
    c_rows = []
    for name, (pnl, pos) in books.items():
        c_rows.append({"book": name, **book_stats(pnl, pos, turnover_bp)})
    c_rows.append(
        {
            "book": "Vilkov: long strangle/straddle basket, 10:00, net of B/A + 0.5 bp, 953 days 2016-2024",
            "days": 953,
            **{k: v for k, v in VILKOV.items()},
        }
    )
    c = pd.DataFrame(c_rows)
    c.to_csv(OUT / "c_books_vilkov_units.csv", index=False)
    cols = [
        "book",
        "days",
        "active_days",
        "mean_pct_spot",
        "vol_pct_spot",
        "Sharpe",
        "ES1_pct_spot",
        "worst_day",
        "worst_5d",
        "max_dd",
        "skew",
        "loss_prob_long",
        "loss_prob_short",
        "turnover_bp",
        "abs_mean_over_ES1",
    ]
    print(c[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(
        "  caveats: his 10:00 entry vs our 15:30; his synthetic exactly-ATM straddle vs our nearest-OTM listed strikes; "
        "his 953 days 2016-2024 (Mon/Wed/Fri expiries before 2022) vs our 866 days 2020-2024; his basket is equal-weight "
        "over strangle widths, long side, entry-only costs; ours is one straddle a day, position by sign(s), crossed = "
        "long at the ask / short at the bid, settlement free. Sharpe here uses sqrt(252) as he does; the repo's "
        "convention (200.3 trades/yr) scales it by 0.89."
    )

    # ----------------------------------------------------------- verdict --
    s0 = c.set_index("book")
    ss = s0.loc["sign(s), crossed"]
    al = s0.loc["always long, mid"]
    r_both = b[
        (b["deck"] == "sub_live_ridge")
        & (b["y"] == "PnL % of spot")
        & (b["spec"] == "IV + RV (his Panel C, ex post)")
    ].iloc[0]
    r_ante = b[
        (b["deck"] == "sub_live_ridge")
        & (b["y"] == "PnL % of spot")
        & (b["spec"] == "IV + rv_hat (ex ante)")
    ].iloc[0]
    print(
        "\nVERDICT  (i) his +-2% window IV is "
        f"{summ.loc['window / slice', 'median']:.2f}x the ATM slice (IQR {summ.loc['window / slice', 'q25']:.2f}-{summ.loc['window / slice', 'q75']:.2f}) "
        f"and {summ.loc['window / strip', 'median']:.2f}x the full strip; zero-bid strikes carry {aa['zero_bid_share'].median():.0%} of the window "
        f"(log-corr with the next bar: window {corr.loc['iv_window', 'rv_next']:.2f}, slice {corr.loc['slice_iv_var', 'rv_next']:.2f}, strip {corr.loc['strip_C', 'rv_next']:.2f}); "
        f"(ii) Panel C at 15:30 on our deck: IV t {r_both['t_IV']:+.2f}, RV t {r_both['t_RV']:+.2f}, adj R^2 {r_both['adj_R2']:.3f} (his 0.20-0.31 at 15:00), "
        f"ex ante with rv_hat: IV t {r_ante['t_IV']:+.2f}, rv_hat t {r_ante['t_rv_hat']:+.2f}, adj R^2 {r_ante['adj_R2']:.3f}; "
        f"(iii) sign(s) crossed in his units: mean {ss['mean_pct_spot']:+.4f}%/day, Sharpe {ss['Sharpe']:.2f}, ES1% {ss['ES1_pct_spot']:.2f}, "
        f"worst day {ss['worst_day']:.2f}, worst 5d {ss['worst_5d']:.2f}, max DD {ss['max_dd']:.1f}, |mean|/ES1% {ss['abs_mean_over_ES1']:.3f} "
        f"(his basket: -0.0215, -0.97, 1.61, -2.67, -7.77, 25.9, 0.013); always-long loss probability {al['loss_prob_long']:.1f}% (his 69.9%)."
    )


if __name__ == "__main__":
    main()
