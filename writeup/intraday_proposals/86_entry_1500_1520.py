"""Study 86 -- earlier decision AND entry: study 82's grid extended to 15:00 .. 15:20.

QUESTION.  Study 82 moved the decision + entry time tau of the close card over
15:25 .. 15:50 (XSP 0DTE book, quotes 15:20-16:16) and found no tau
distinguishable from 15:30.  With XSP quotes for 15:00-15:21 now on disk
(data/archive/xsp_opra/cbbo1m-1500-1521_<day>.parquet, 876 days, same schema,
definitions shared with the cbbo1m_<day>.parquet files), does an EARLIER entry
tau in {15:00, 15:05, 15:10, 15:15, 15:20} do better?  At tau: forecast the ES
variance from tau to 16:00 with information through tau, buy the nearest-OTM
0DTE strangle at tau at its ask iff ask <= P*(forecast)
(live.ibkr.pricing.package_price), hold to the SPX close / 10.  Month-ends
(live.ibkr.calendar_guard.is_last_session_of_month) are excluded from the rule
and reported separately as the unconditional long, as in study 82.

SAME CODE PATH.  Everything is study 82's (imported from its file, not copied):
ES minutes, the minute matrices, tau_features (T0 = the 30-minute design moved
to tau, T1 = T0 + the 1-minute FINE block), the session-by-session OLS refit on
strictly prior sessions with smearing, the common session list, QLIKE, the
block bootstrap (20 sessions, seed 82), study 79's book helpers.  The two
functions that loop over study 82's module-level TAUS (the per-day book and the
card rule) are re-stated here with tau as an argument, line for line.  The VIX
regressor is the 15:00 stamp at every tau: the print standing just BEFORE 15:00
(vendor convention; the Yahoo hourly bar ENDING 15:00, live/close_signal/
ingest.py), so it is F_tau-measurable at tau = 15:00 in the research clock (the
live Cboe feed lags ~15 minutes -- study 85; not modelled here).

BOOKS.  tau in 15:00 .. 15:20 from the new cbbo1m-1500-1521 files, tau in
15:25 .. 15:50 from the cbbo1m files (study 82's books).  The day set is the
one study 82 uses: every tau and both models present (now 11 taus).

GATES (the run stops on a failed hard gate):
  G1  study 82's G1 (1-min ES -> 30-min vs panel_free), G2 (baseline vs deck),
      G3 (T0/T1 at 15:30 == S0/S4) re-run; the new taus' features finite on
      study 82's common sessions (the session list must not move).
  GB  the new window's 15:20 book vs the old window's 15:20 book (both files
      hold 15:20): pair, bid, ask, parity spot, per day; record level, every
      symbol both files hold at 15:20:00 must carry the same BBO and sizes
      (hard); and this script's books at 15:25 .. 15:50 == study 82's
      books_tau.parquet (hard).  The two pulls subscribe different strike
      bands (live/close_signal/pull_databento_xsp.py: +-2 % around the chain's
      15:30 spot; the old files were pulled before the 2026-09-24 UTC fix, when
      that "15:30" spot was the midday one), so the symbol SETS differ and a
      book can differ where one band misses a strike the other holds.
  GR  REPRODUCTION: the card rule run with study 82's taus on study 82's books
      must equal its summary_tau.csv, month_end_tau.csv, bootstrap_tau.csv and
      qlike_tau.csv exactly; the extended run's 15:25 .. 15:50 rows likewise
      when its day set equals study 82's.
  GL  study 82's leak test (x sqrt(50) on every 1-minute return at or after
      the decision minute; the minute before must move the features) at all
      11 taus.
  GC  the new books' clock: the parity-spot move tau -> 15:30 against the ES
      move over the same minutes shifted -2 .. +2; zero shift must fit best
      (study 82's G5 criterion, median |gap| < 1 bp).

MULTIPLE TESTING.  tau - 15:30 per-day P&L: block-bootstrap one-sided p
(H1: tau beats 15:30; (#{boot mean <= 0} + 1) / (N + 1)) and Newey-West (lag 5)
normal p; Holm over the 10 taus != 15:30 within each (model, price) family.
Month-ends: one-sided paired-t p (df = n - 1), Holm over the 10 taus.

SPX (SPXW, spread ~4x tighter) at 15:00 .. 15:20 is PENDING the SPXW
15:00-15:21 pull: SPXW rows here are study 82's 15:25 .. 15:50 month-ends only.

OUTPUTS (results/atm_straddle_intraday_holdclose/proposals/86/):
  gate.json, gate_book_1520.csv, gate_records_1520.csv, leak_test.csv,
  gate_reproduction_82.csv (incl. forecasts_tau / per_day_pnl_tau parquet)
  forecasts_tau.parquet, qlike_tau.csv, books_tau.parquet
  per_day_pnl_tau.parquet/.csv, summary_tau.csv, bootstrap_tau.csv,
  month_end_tau.csv, spx_close_yf.csv (copy of study 82's cache), summary.txt

    C:/Users/james/miniconda3/envs/285J/python.exe writeup/intraday_proposals/86_entry_1500_1520.py
    (about 2 minutes; the option books run in a process pool, --workers 4)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from typing import Any
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sst

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from live.ibkr.calendar_guard import is_last_session_of_month  # noqa: E402
from live.ibkr.pricing import package_price  # noqa: E402

S82_SRC = REPO / "writeup" / "intraday_proposals" / "82_decision_granularity.py"


def _load_s82():
    spec = importlib.util.spec_from_file_location("s82_decision_granularity", S82_SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


s82 = _load_s82()

OUT = REPO / "results" / "atm_straddle_intraday_holdclose" / "proposals" / "86"
OUT82 = REPO / "results" / "atm_straddle_intraday_holdclose" / "proposals" / "82"
XSP_DIR = s82.XSP_DIR
SPXW_DIR = s82.SPXW_DIR
NEW_PREFIX = "cbbo1m-1500-1521_"
TAUS_NEW = ("15:00", "15:05", "15:10", "15:15", "15:20")
TAUS82 = tuple(s82.TAUS)
TAUS = TAUS_NEW + TAUS82
GATE_TAU = "15:20"
REF_TAU = s82.REF_TAU
RTH0, RTH1 = s82.RTH0, s82.RTH1
BASE, FINE = s82.BASE, s82.FINE
BOOK_COLS = ["S_book", "kc", "kp", "bid", "ask", "mid", "ask_size", "n_strikes_quoted"]

_LOG: list[str] = []


def log(msg: str) -> None:
    _LOG.append(msg)
    print(msg, flush=True)


def minute(tau: str) -> int:
    hh, mn = (int(x) for x in tau.split(":"))
    return hh * 60 + mn


# --------------------------------------------------------------------------- books
def books_for_day(args: tuple[str, str, str, str, str, tuple[str, ...]]) -> list[dict]:
    """study 82's books_for_day with the taus (and the source window) as arguments."""
    qpath, dpath, day_s, venue, src, taus = args
    s79 = s82._s79()
    day = pd.Timestamp(day_s)
    defs = pd.read_parquet(dpath)
    if defs.empty:
        return []
    strikes = np.sort(defs["strike_price"].astype(float).unique())
    q = s79.load_quotes(Path(qpath))
    rows = []
    for tau in taus:
        stamp = pd.Timestamp(f"{day.date()} {tau}:00")
        b = s79.book_at(q, day, f"{tau}:00")
        if b.empty:
            continue
        bt = b.index.get_level_values(0)
        s = s79.parity_spot(b)
        kc, kp = s79.nearest_otm(strikes, s) if np.isfinite(s) else (np.nan, np.nan)
        bc, ac, bsc, asc = s79.quote(b, kc, "C")
        bp, ap, bsp, asp = s79.quote(b, kp, "P")
        bid, ask = bc + bp, ac + ap
        t_book = q.loc[q["t"] <= stamp, "t"].max()
        rows.append(
            {
                "day": day,
                "venue": venue,
                "tau": tau,
                "book_lag_s": float((stamp - t_book).total_seconds()),
                "S_book": s,
                "kc": kc,
                "kp": kp,
                "bid": bid,
                "ask": ask,
                "mid": (bid + ask) / 2,
                "ask_size": min(asc, asp),
                "n_strikes_quoted": int(len(np.unique(bt))),
                "src": src,
            }
        )
    return rows


def records_1520(args: tuple[str, str, str]) -> dict:
    """Both windows' raw cbbo-1m records at 15:20:00 of one day, symbol by symbol."""
    old_p, new_p, day_s = args
    ts = pd.Timestamp(f"{day_s} {GATE_TAU}", tz=s82.ET).tz_convert("UTC")
    cols = ["symbol", "bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"]
    o = pd.read_parquet(old_p, columns=["ts_recv", *cols])
    n = pd.read_parquet(new_p, columns=["ts_recv", *cols])
    o = o[pd.to_datetime(o["ts_recv"], utc=True) == ts].set_index("symbol")[cols[1:]]
    n = n[pd.to_datetime(n["ts_recv"], utc=True) == ts].set_index("symbol")[cols[1:]]
    both = o.index.intersection(n.index)
    a_, b_ = o.loc[both].to_numpy(float), n.loc[both].to_numpy(float)
    same = (a_ == b_) | (np.isnan(a_) & np.isnan(b_))
    return {
        "day": day_s,
        "symbols_old": int(len(o)),
        "symbols_new": int(len(n)),
        "symbols_both": int(len(both)),
        "symbols_both_quote_differs": int((~same.all(1)).sum()),
        "symbols_file_old": int(
            pd.read_parquet(old_p, columns=["symbol"])["symbol"].nunique()
        ),
        "symbols_file_new": int(
            pd.read_parquet(new_p, columns=["symbol"])["symbol"].nunique()
        ),
    }


def spx_closes() -> pd.Series:
    """Study 82's ^GSPC cache, read only (copied into this study's folder)."""
    c = pd.read_csv(OUT82 / "spx_close_yf.csv", index_col=0, parse_dates=True)["close"]
    c.to_frame().to_csv(OUT / "spx_close_yf.csv")
    return c


# --------------------------------------------------------------------------- stats
def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (n - i) * p[j]))
        adj[j] = run
    return adj


def t_all(x: pd.Series) -> float:
    x = x.to_numpy(float)
    sd = x.std(ddof=1)
    return float(x.mean() / (sd / np.sqrt(len(x)))) if sd > 0 else np.nan


# --------------------------------------------------------------------------- card rule
def card_rule(
    ftab: pd.DataFrame,
    books_in: pd.DataFrame,
    closes: pd.Series,
    taus: tuple[str, ...],
    venues: tuple[str, ...],
) -> dict[str, pd.DataFrame]:
    """study 82's (b) trading block with taus / month-end venues as arguments."""
    books = books_in.copy()
    books["close"] = closes.reindex(books["day"]).to_numpy()
    sc = np.where(books["venue"] == "XSP", s82.XSP_SCALE, 1.0)
    books["payoff"] = np.maximum(books["close"] * sc - books["kc"], 0) + np.maximum(
        books["kp"] - books["close"] * sc, 0
    )
    books["R_ask"] = books["payoff"] / books["ask"] - 1
    books["R_mid"] = books["payoff"] / books["mid"] - 1
    books["rel_spread"] = (books["ask"] - books["bid"]) / books["mid"]
    books["month_end"] = [is_last_session_of_month(d.date()) for d in books["day"]]
    xb = books[books.venue == "XSP"]
    pnl = ftab.merge(xb, on=["day", "tau"], how="inner").dropna(
        subset=["f", "ask", "payoff"]
    )
    pnl = pnl[pnl["ask"] > 0]
    pnl["P_star"] = [
        package_price(np.sqrt(f), s, kc, kp)
        for f, s, kc, kp in zip(pnl["f"], pnl["S_book"], pnl["kc"], pnl["kp"])
    ]
    pnl["trade_ask"] = pnl["ask"] <= pnl["P_star"]
    pnl["trade_mid"] = pnl["mid"] <= pnl["P_star"]
    pnl["pnl_ask"] = np.where(pnl["trade_ask"], pnl["R_ask"], 0.0)
    pnl["pnl_mid"] = np.where(pnl["trade_mid"], pnl["R_mid"], 0.0)
    cnt = pnl.groupby("day").size()
    full = cnt.index[cnt == 2 * len(taus)]
    dropped_days = int(len(cnt) - len(full))
    pnl = pnl[pnl.day.isin(full)]
    nm = pnl[~pnl.month_end]
    sb = []
    for (mdl, tau), g in nm.groupby(["model", "tau"]):
        sa = s82.stats(g.loc[g.trade_ask, "R_ask"])
        sm = s82.stats(g.loc[g.trade_mid, "R_mid"])
        sb.append(
            {
                "model": mdl,
                "tau": tau,
                "days": int(len(g)),
                "traded_ask": sa["n"],
                "mean_R_ask": sa["mean_R"],
                "t_ask": sa["t"],
                "hit_ask": sa["hit"],
                "pnl_per_day_ask": float(g.pnl_ask.mean()),
                "traded_mid": sm["n"],
                "mean_R_mid": sm["mean_R"],
                "t_mid": sm["t"],
                "pnl_per_day_mid": float(g.pnl_mid.mean()),
                "uncond_R_ask": float(g.R_ask.mean()),
                "rel_spread_median": float(g.rel_spread.median()),
                # added in study 86: t of the per-day P&L over ALL days (zeros included)
                "t_pnl_day_ask": t_all(g.pnl_ask),
                "t_pnl_day_mid": t_all(g.pnl_mid),
            }
        )
    sb = pd.DataFrame(sb)
    bb = []
    for mdl in ("T0", "T1"):
        base = nm[(nm.model == mdl) & (nm.tau == REF_TAU)].set_index("day").sort_index()
        for tau in taus:
            if tau == REF_TAU:
                continue
            o = nm[(nm.model == mdl) & (nm.tau == tau)].set_index("day").loc[base.index]
            for col in ("pnl_ask", "pnl_mid"):
                d = (o[col] - base[col]).to_numpy(float)
                lo, hi, p_neg = s82.boot_ci(d)
                k_le0 = p_neg * s82.N_BOOT  # boot means < 0 (ties at 0 have measure ~0)
                t_nw = s82.nw_t(d)
                bb.append(
                    {
                        "model": mdl,
                        "tau_minus_1530": tau,
                        "stat": col,
                        "days": int(len(d)),
                        "point": float(d.mean()),
                        "ci_lo": lo,
                        "ci_hi": hi,
                        "p_boot_1s": float((k_le0 + 1) / (s82.N_BOOT + 1)),
                        "t_nw5": t_nw,
                        "p_nw_1s": float(sst.norm.sf(t_nw)),
                    }
                )
    bb = pd.DataFrame(bb)
    if len(bb):
        for c in ("p_boot_1s", "p_nw_1s"):
            bb[c.replace("p_", "holm_")] = np.nan
            for _, idx in bb.groupby(["model", "stat"]).groups.items():
                bb.loc[idx, c.replace("p_", "holm_")] = holm(bb.loc[idx, c].to_numpy())
    me_rows = []
    for venue in venues:
        g = books[(books.venue == venue) & books.month_end].dropna(subset=["R_ask"])
        g = g[g.ask > 0]
        cnt_ = g.groupby("day").size()
        g = g[g.day.isin(cnt_.index[cnt_ == len(taus)])]
        wR = g.pivot(index="day", columns="tau", values="R_ask")
        for tau, h in g.groupby("tau"):
            s_ = s82.stats(h["R_ask"])
            dd = (wR[tau] - wR[REF_TAU]).dropna()
            tt = (
                float(dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd))))
                if tau != REF_TAU
                else np.nan
            )
            me_rows.append(
                {
                    "venue": venue,
                    "tau": tau,
                    "month_ends": s_["n"],
                    "mean_R_ask": s_["mean_R"],
                    "t_ask": s_["t"],
                    "hit_ask": s_["hit"],
                    "mean_R_mid": float(h["R_mid"].mean()),
                    "rel_spread_median": float(h["rel_spread"].median()),
                    "minus_1530_R_ask": float(dd.mean()),
                    "minus_1530_paired_t": tt,
                    "p_paired_1s": float(sst.t.sf(tt, len(dd) - 1))
                    if tau != REF_TAU
                    else np.nan,
                }
            )
    me_t = pd.DataFrame(me_rows)
    if len(me_t):
        me_t["holm_paired_1s"] = np.nan
        for _, idx in me_t[me_t.tau != REF_TAU].groupby("venue").groups.items():
            me_t.loc[idx, "holm_paired_1s"] = holm(
                me_t.loc[idx, "p_paired_1s"].to_numpy()
            )
    return {
        "pnl": pnl,
        "sb": sb,
        "bb": bb,
        "me": me_t,
        "n_days_nm": pd.DataFrame({"n": [nm.day.nunique()], "dropped": [dropped_days]}),
    }


def ref82(name: str) -> pd.DataFrame:
    """Study 82's CSV, parsed round-trip exact (the default C parser is off in the last bit)."""
    return pd.read_csv(OUT82 / name, float_precision="round_trip")


def compare(mine: pd.DataFrame, ref: pd.DataFrame, keys: list[str], name: str) -> dict:
    """Exact comparison of every shared numeric column on the shared keys."""
    m = mine.copy()
    r = ref.copy()
    for k in keys:
        m[k] = m[k].astype(str)
        r[k] = r[k].astype(str)
    j = r.merge(m, on=keys, how="left", suffixes=("_82", "_86"), indicator=True)
    missing = int((j["_merge"] != "both").sum())
    cols = [c for c in ref.columns if c not in keys and c in mine.columns]
    worst = 0.0
    worst_col = ""
    for c in cols:
        a = pd.to_numeric(j[f"{c}_82"], errors="coerce").to_numpy(float)
        b = pd.to_numeric(j[f"{c}_86"], errors="coerce").to_numpy(float)
        nan_ok = np.array_equal(np.isnan(a), np.isnan(b))
        fin = np.isfinite(a) & np.isfinite(b)
        d = float(np.max(np.abs(a[fin] - b[fin]))) if fin.any() else 0.0
        if not nan_ok:
            d = np.inf
        if d > worst:
            worst, worst_col = d, c
    return {
        "table": name,
        "rows_ref": int(len(ref)),
        "rows_missing": missing,
        "columns": len(cols),
        "max_abs_diff": worst,
        "worst_column": worst_col,
        "exact": bool(missing == 0 and worst == 0.0),
    }


# --------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    T0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    gate: dict[str, Any] = {}

    def stop(tag: str) -> int:
        log(f"{tag} FAILED")
        (OUT / "gate.json").write_text(json.dumps(gate, indent=1, default=str))
        (OUT / "summary.txt").write_text("\n".join(_LOG), encoding="utf-8")
        return 2

    # ---- study 82's data path
    df = s82.es_minutes()
    log(
        f"ES 1-minute: {len(df)} bars {df['t'].min()} .. {df['t'].max()} (naive ET bar start)"
    )
    mom = s82.moments_30(df)
    rep, g1 = s82.gate_parity(mom)
    gate["G1_es_30min_vs_panel_free"] = g1
    log("G1 (study 82) " + json.dumps(g1))
    if not g1["passed"]:
        return stop("G1")
    mm = s82.minute_matrices(df)
    del df
    vix = s82.vix_1500()
    log(f"VIX at 15:00: {vix.attrs['seam']}; {len(vix)} days")
    S, _ = s82.s_features(mom, mm, vix)
    L = s82.long_table(mom, vix)
    del mom
    L = L[L.notna().all(1)]
    L["f_L0"] = s82.ols_forecasts(
        L[BASE].to_numpy(), np.log(L["y"].to_numpy()), s82.LONG_MIN, window=s82.LONG_WIN
    )
    L["f_naive22"] = np.exp(L["lslot22"])
    S["lf_L0"] = np.log(L["f_L0"].reindex(S.index))
    tabs = {
        tau: s82.tau_features(mm["R"], mm["PK"], mm["days"], vix, minute(tau))
        for tau in TAUS
    }
    # study 82's common sessions, exactly (its 15:30 table + its six taus)
    need = sorted(set(sum(s82.S_MODELS.values(), []))) + ["lf_L0"]
    ok_rows = S[need + ["y"]].notna().all(1) & np.isfinite(S[need].to_numpy()).all(1)
    common = S.index[ok_rows]
    for tau in TAUS82:
        t = tabs[tau]
        ok = t[BASE + FINE + ["y"]].notna().all(1) & np.isfinite(
            t[BASE + FINE].to_numpy()
        ).all(1)
        common = common.intersection(t.index[ok])
    bad_new = {}
    for tau in TAUS_NEW:
        t = tabs[tau].loc[common]
        ok = t[BASE + FINE + ["y"]].notna().all(1) & np.isfinite(
            t[BASE + FINE].to_numpy()
        ).all(1)
        bad_new[tau] = int((~ok).sum())
    gate["G1_common_sessions"] = {
        "sessions": int(len(common)),
        "first": str(common.min().date()),
        "last": str(common.max().date()),
        "new_tau_nonfinite_on_common": bad_new,
        "passed": bool(sum(bad_new.values()) == 0),
    }
    log("G1 common sessions " + json.dumps(gate["G1_common_sessions"]))
    if not gate["G1_common_sessions"]["passed"]:
        return stop("G1 common sessions")
    S = S.loc[common].copy()

    # ---- G2 (study 82): the long baseline vs the deck
    deck = pd.read_parquet(s82.DECK)
    deck.index = pd.DatetimeIndex(pd.to_datetime(deck.index)).normalize()
    j = deck[["rv_hat"]].join(L[["y", "f_L0", "f_naive22"]], how="inner").dropna()
    q_deck = float(s82.qlike(j["y"].to_numpy(), j["rv_hat"].to_numpy()).mean())
    q_l0 = float(s82.qlike(j["y"].to_numpy(), j["f_L0"].to_numpy()).mean())
    q_nv = float(s82.qlike(j["y"].to_numpy(), j["f_naive22"].to_numpy()).mean())
    g2 = {
        "days": int(len(j)),
        "qlike_deck": q_deck,
        "qlike_L0": q_l0,
        "qlike_naive22": q_nv,
    }
    g2["passed"] = bool(
        s82.DECK_QLIKE_BAND[0] <= q_l0 / q_deck <= s82.DECK_QLIKE_BAND[1]
        and q_l0 < q_nv
    )
    gate["G2_baseline_vs_deck"] = g2
    log("G2 (study 82) " + json.dumps(g2))
    if not g2["passed"]:
        return stop("G2")

    # ---- forecasts at every tau (study 82's refit, study 82's sessions)
    fT = {}
    for tau in TAUS:
        t = tabs[tau].loc[common]
        lyt = np.log(t["y"].to_numpy())
        fT[(tau, "T0")] = s82.ols_forecasts(t[BASE].to_numpy(), lyt, s82.MIN_TRAIN)
        fT[(tau, "T1")] = s82.ols_forecasts(
            t[BASE + FINE].to_numpy(), lyt, s82.MIN_TRAIN
        )
    ly = np.log(S["y"].to_numpy())
    f_s0 = s82.ols_forecasts(S[s82.S_MODELS["S0_base30"]].to_numpy(), ly, s82.MIN_TRAIN)
    f_s4 = s82.ols_forecasts(S[s82.S_MODELS["S4_+fine"]].to_numpy(), ly, s82.MIN_TRAIN)
    g3 = {
        "max_rel_T0_vs_S0": float(np.nanmax(np.abs(fT[(REF_TAU, "T0")] / f_s0 - 1))),
        "max_rel_T1_vs_S4": float(np.nanmax(np.abs(fT[(REF_TAU, "T1")] / f_s4 - 1))),
    }
    g3["passed"] = bool(max(g3.values()) <= 1e-9)
    gate["G3_tau1530_is_a"] = g3
    log("G3 (study 82) " + json.dumps(g3))
    if not g3["passed"]:
        return stop("G3")

    # ---- GL: study 82's leak test at all 11 taus
    k = len(mm["days"]) // 2
    lt_rows = []
    for tau in TAUS:
        m = minute(tau)
        ref_t = tabs[tau]
        for case, cols_lo, expect_same in (
            ("after_decision", m, True),
            ("minute_before_decision", m - 1, False),
        ):
            Rp = mm["R"].copy()
            PKp = mm["PK"].copy()
            if expect_same:
                Rp[k, cols_lo - RTH0 :] *= np.sqrt(s82.LEAK_FACTOR)
                PKp[k, cols_lo - RTH0 :] *= s82.LEAK_FACTOR
            else:
                Rp[k, cols_lo - RTH0] *= np.sqrt(s82.LEAK_FACTOR)
                PKp[k, cols_lo - RTH0] *= s82.LEAK_FACTOR
            pt = s82.tau_features(Rp, PKp, mm["days"], vix, m)
            del Rp, PKp
            feats = BASE + FINE
            a_ = ref_t.iloc[: k + 1][feats].to_numpy()
            b_ = pt.iloc[: k + 1][feats].to_numpy()
            fin = np.isfinite(a_) & np.isfinite(b_)
            nan_same = bool((np.isfinite(a_) == np.isfinite(b_)).all())
            dev = float(
                (np.abs(a_[fin] - b_[fin]) / np.maximum(np.abs(a_[fin]), 1e-300)).max()
            )
            dev_day = float(
                np.nanmax(np.abs(a_[-1] - b_[-1]) / np.maximum(np.abs(a_[-1]), 1e-300))
            )
            y_dev = float(abs(pt["y"].iloc[k] / ref_t["y"].iloc[k] - 1))
            lt_rows.append(
                {
                    "tau": tau,
                    "case": case,
                    "day": str(mm["days"][k].date()),
                    "max_rel_dev_features_through_day": dev,
                    "max_rel_dev_features_on_day": dev_day,
                    "rel_dev_target_on_day": y_dev,
                    "pass": (
                        nan_same and dev <= s82.LEAK_TOL and y_dev > s82.MOVE_TOL
                        if expect_same
                        else dev_day > s82.MOVE_TOL
                    ),
                }
            )
    lt = pd.DataFrame(lt_rows)
    lt.to_csv(OUT / "leak_test.csv", index=False)
    lt82 = ref82("leak_test.csv")
    gate["GL_leak_test"] = {
        "pass": bool(lt["pass"].all()),
        "rows_82_reproduced": compare(lt, lt82, ["tau", "case", "day"], "leak_test"),
    }
    log("GL leak test:\n" + lt.to_string(index=False))
    if not lt["pass"].all():
        return stop("GL")

    ftab = []
    for tau in TAUS:
        t = tabs[tau].loc[common]
        for mdl in ("T0", "T1"):
            ftab.append(
                pd.DataFrame(
                    {
                        "day": common,
                        "tau": tau,
                        "model": mdl,
                        "y": t["y"].to_numpy(),
                        "f": fT[(tau, mdl)],
                        "me": t["me"].to_numpy() > 0,
                    }
                )
            )
    ftab = pd.concat(ftab, ignore_index=True)
    ftab.to_parquet(OUT / "forecasts_tau.parquet", index=False)
    del tabs
    qb = []
    for tau in TAUS:
        a0 = ftab[(ftab.tau == tau) & (ftab.model == "T0")].dropna()
        a1 = ftab[(ftab.tau == tau) & (ftab.model == "T1")].dropna()
        q0 = s82.qlike(a0["y"].to_numpy(), a0["f"].to_numpy())
        q1 = s82.qlike(a1["y"].to_numpy(), a1["f"].to_numpy())
        lo, hi, _ = s82.boot_ci(q1 - q0)
        qb.append(
            {
                "tau": tau,
                "horizon_min": RTH1 - minute(tau),
                "oos_days": int(len(a0)),
                "qlike_T0": float(q0.mean()),
                "qlike_T1": float(q1.mean()),
                "dqlike_pct": float((q1 - q0).mean() / q0.mean()),
                "dm_t_nw5": s82.nw_t(q1 - q0),
                "boot_ci_lo": lo,
                "boot_ci_hi": hi,
            }
        )
    qb = pd.DataFrame(qb)
    qb.to_csv(OUT / "qlike_tau.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    log("\nQLIKE of the tau->16:00 RV forecast, T0 vs T1 (+FINE):")
    log(qb.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---- option books: old window (study 82's taus + 15:20), new window (15:00 .. 15:20)
    jobs = []
    for venue, d_ in (("XSP", XSP_DIR), ("SPXW", SPXW_DIR)):
        for qp in sorted(d_.glob("cbbo1m_*.parquet")):
            ds = qp.stem.split("_")[1]
            dp = d_ / f"defs_{ds}.parquet"
            if dp.exists():
                taus_old = ((GATE_TAU,) if venue == "XSP" else ()) + TAUS82
                jobs.append((str(qp), str(dp), ds, venue, "w1520", taus_old))
    n_old = len(jobs)
    for qp in sorted(XSP_DIR.glob(f"{NEW_PREFIX}*.parquet")):
        ds = qp.stem[len(NEW_PREFIX) :]
        dp = XSP_DIR / f"defs_{ds}.parquet"
        if dp.exists():
            jobs.append((str(qp), str(dp), ds, "XSP", "w1500", TAUS_NEW))
    t_b = time.time()
    rec_jobs = []
    for qp in sorted(XSP_DIR.glob("cbbo1m_*.parquet")):
        ds = qp.stem.split("_")[1]
        npth = XSP_DIR / f"{NEW_PREFIX}{ds}.parquet"
        if npth.exists():
            rec_jobs.append((str(qp), str(npth), ds))
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        res = list(ex.map(books_for_day, jobs, chunksize=8))
        rec = pd.DataFrame(list(ex.map(records_1520, rec_jobs, chunksize=8)))
    rec.to_csv(OUT / "gate_records_1520.csv", index=False)
    allb = pd.DataFrame([r for rr in res for r in rr])
    del res
    log(
        f"books: {n_old} old-window + {len(jobs) - n_old} new-window day files -> "
        f"{len(allb)} (day, tau) rows in {time.time() - t_b:.0f}s; book lag > 0 s on "
        f"{int((allb.book_lag_s > 0).sum())} rows"
    )

    # ---- GB: 15:20 in both windows; the old window's books == study 82's
    o20 = allb[(allb.src == "w1520") & (allb.tau == GATE_TAU)].set_index("day")
    n20 = allb[(allb.src == "w1500") & (allb.tau == GATE_TAU)].set_index("day")
    both = o20.index.intersection(n20.index)
    cmpc = ["S_book", "kc", "kp", "bid", "ask", "mid", "ask_size", "n_strikes_quoted"]
    gb_rows = pd.DataFrame(index=both)
    for c in cmpc:
        x, y = o20.loc[both, c].astype(float), n20.loc[both, c].astype(float)
        gb_rows[f"{c}_old"] = x
        gb_rows[f"{c}_new"] = y
        gb_rows[f"{c}_same"] = (x == y) | (x.isna() & y.isna())
    gb_rows["all_same"] = gb_rows[[f"{c}_same" for c in cmpc]].all(1)
    gb_rows.index.name = "day"
    gb_rows.to_csv(OUT / "gate_book_1520.csv", float_format="%.10g")
    pair_same = gb_rows["kc_same"] & gb_rows["kp_same"]
    ask_diff = (gb_rows["ask_new"] - gb_rows["ask_old"]).abs()
    b82 = pd.read_parquet(OUT82 / "books_tau.parquet")
    mine82 = allb[(allb.src == "w1520") & allb.tau.isin(TAUS82)].drop(columns="src")
    rep82 = compare(mine82, b82, ["day", "venue", "tau"], "books_tau_82")
    gb: dict[str, Any] = {
        "records_1520": {
            "days": int(len(rec)),
            "symbols_both_total": int(rec["symbols_both"].sum()),
            "symbols_both_quote_differs_total": int(
                rec["symbols_both_quote_differs"].sum()
            ),
            "days_symbol_set_differs": int(
                (
                    (rec["symbols_both"] != rec["symbols_old"])
                    | (rec["symbols_both"] != rec["symbols_new"])
                ).sum()
            ),
            "days_file_symbol_count_differs": int(
                (rec["symbols_file_old"] != rec["symbols_file_new"]).sum()
            ),
        },
        "days_old_1520": int(len(o20)),
        "days_new_1520": int(len(n20)),
        "days_both": int(len(both)),
        "days_only_new": [str(d.date()) for d in n20.index.difference(o20.index)],
        "days_only_old": [str(d.date()) for d in o20.index.difference(n20.index)],
        "days_all_fields_identical": int(gb_rows["all_same"].sum()),
        "days_same_pair": int(pair_same.sum()),
        "days_same_bid_ask": int((gb_rows["bid_same"] & gb_rows["ask_same"]).sum()),
        "max_abs_ask_diff_same_pair": float(ask_diff[pair_same].max()),
        "max_rel_spot_diff": float(
            (gb_rows["S_book_new"] / gb_rows["S_book_old"] - 1).abs().max()
        ),
        "fields_differing_count": {c: int((~gb_rows[f"{c}_same"]).sum()) for c in cmpc},
        "old_window_books_vs_study82": rep82,
    }
    bad = ~gb_rows[["S_book_same", "kc_same", "kp_same", "bid_same", "ask_same"]].all(1)
    gb["book_differs_days"] = (
        gb_rows.loc[
            bad,
            [
                "S_book_old",
                "S_book_new",
                "kc_old",
                "kc_new",
                "kp_old",
                "kp_new",
                "ask_old",
                "ask_new",
            ],
        ]
        .reset_index()
        .astype(str)
        .to_dict("records")
    )
    gb["passed"] = bool(
        rep82["exact"]
        and gb["records_1520"]["symbols_both_quote_differs_total"] == 0
        and pair_same.mean() > 0.95
    )
    gate["GB_books"] = gb
    log("GB " + json.dumps(gb, default=str))
    if not gb["passed"]:
        return stop("GB")
    books = pd.concat(
        [
            allb[allb.src == "w1500"],
            allb[(allb.src == "w1520") & allb.tau.isin(TAUS82)],
        ],
        ignore_index=True,
    )
    books.to_parquet(OUT / "books_tau.parquet", index=False)
    del allb
    closes = spx_closes()

    # ---- GC: the new books' clock (study 82's G5 criterion, tau < 15:30)
    xs = books[books.venue == "XSP"].pivot(index="day", columns="tau", values="S_book")
    Rm = pd.DataFrame(
        np.nan_to_num(mm["R"]), index=mm["days"], columns=np.arange(RTH0, RTH1)
    )
    gc_rows: list[dict[str, Any]] = []
    for tau in TAUS_NEW:
        m = minute(tau)
        lo_, hi_ = sorted((930, m))
        bk_move = np.log(xs[tau] / xs[REF_TAU])
        med: dict[int, float] = {}
        z0: pd.DataFrame | None = None
        for sh in (-2, -1, 0, 1, 2):
            es_s = Rm.loc[:, lo_ + sh : hi_ - 1 + sh].sum(1) * (1 if m > 930 else -1)
            z = pd.concat([es_s, bk_move], axis=1, join="inner").dropna()
            z.columns = ["es", "book"]
            med[sh] = float((z.es - z.book).abs().median() * 1e4)
            if sh == 0:
                z0 = z
        assert z0 is not None  # the shift-0 frame
        best = min(med, key=lambda k_: med[k_])
        gc_rows.append(
            {
                "tau": tau,
                "days": len(z0),
                "pearson": float(z0.corr().iloc[0, 1]),
                "spearman": float(z0.corr("spearman").iloc[0, 1]),
                "median_abs_diff_bp_by_shift": med,
                "best_shift": best,
            }
        )
    gate["GC_new_book_clock"] = {
        "rows": gc_rows,
        "passed": bool(
            all(r["best_shift"] == 0 for r in gc_rows)
            and max(r["median_abs_diff_bp_by_shift"][0] for r in gc_rows) < 1.0
        ),
    }
    log("GC " + json.dumps(gate["GC_new_book_clock"], default=str))
    if not gate["GC_new_book_clock"]["passed"]:
        return stop("GC")
    del Rm, mm

    # ---- GR: reproduction of study 82 (its taus, its books, its venues)
    rr = card_rule(
        ftab[ftab.tau.isin(TAUS82)],
        books[books.tau.isin(TAUS82)],
        closes,
        TAUS82,
        ("XSP", "SPXW"),
    )
    gr = [
        compare(rr["sb"], ref82("summary_tau.csv"), ["model", "tau"], "summary_tau"),
        compare(
            rr["me"], ref82("month_end_tau.csv"), ["venue", "tau"], "month_end_tau"
        ),
        compare(
            rr["bb"],
            ref82("bootstrap_tau.csv"),
            ["model", "tau_minus_1530", "stat"],
            "bootstrap_tau",
        ),
        compare(qb, ref82("qlike_tau.csv"), ["tau"], "qlike_tau"),
        compare(
            ftab[ftab.tau.isin(TAUS82)],
            pd.read_parquet(OUT82 / "forecasts_tau.parquet"),
            ["day", "tau", "model"],
            "forecasts_tau_parquet",
        ),
        compare(
            rr["pnl"],
            pd.read_parquet(OUT82 / "per_day_pnl_tau.parquet"),
            ["day", "tau", "model"],
            "per_day_pnl_tau_parquet",
        ),
    ]
    # ---- the extended run: 11 taus, XSP
    ex_ = card_rule(ftab, books, closes, TAUS, ("XSP",))
    gr += [
        compare(
            ex_["sb"], ref82("summary_tau.csv"), ["model", "tau"], "ext_summary_tau"
        ),
        compare(
            ex_["me"],
            ref82("month_end_tau.csv").query("venue == 'XSP'"),
            ["venue", "tau"],
            "ext_month_end_tau_XSP",
        ),
        compare(
            ex_["bb"],
            ref82("bootstrap_tau.csv"),
            ["model", "tau_minus_1530", "stat"],
            "ext_bootstrap_tau",
        ),
    ]
    grd = pd.DataFrame(gr)
    grd.to_csv(OUT / "gate_reproduction_82.csv", index=False)
    gate["GR_reproduction_82"] = {
        "tables": gr,
        "reproduction_days_nonmonthend": int(rr["n_days_nm"]["n"].iloc[0]),
        "reproduction_days_dropped_missing_a_tau": int(
            rr["n_days_nm"]["dropped"].iloc[0]
        ),
        "extended_days_nonmonthend": int(ex_["n_days_nm"]["n"].iloc[0]),
        "extended_days_dropped_missing_a_tau": int(ex_["n_days_nm"]["dropped"].iloc[0]),
        "passed": bool(grd.loc[~grd.table.str.startswith("ext_"), "exact"].all()),
    }
    log("\nGR reproduction of study 82:\n" + grd.to_string(index=False))
    log(
        json.dumps(
            {k: v for k, v in gate["GR_reproduction_82"].items() if k != "tables"}
        )
    )
    if not gate["GR_reproduction_82"]["passed"]:
        return stop("GR")

    pnl, sb, bb, me_t = ex_["pnl"], ex_["sb"], ex_["bb"], ex_["me"]
    pnl.to_parquet(OUT / "per_day_pnl_tau.parquet", index=False)
    pnl.to_csv(OUT / "per_day_pnl_tau.csv", index=False, float_format="%.6g")
    sb = sb.merge(qb[["tau", "qlike_T0", "qlike_T1"]], on="tau", how="left")
    sb.to_csv(OUT / "summary_tau.csv", index=False)
    bb.to_csv(OUT / "bootstrap_tau.csv", index=False)
    me_t.to_csv(OUT / "month_end_tau.csv", index=False)
    nm = pnl[~pnl.month_end]
    log(
        f"\ncard rule on XSP, non-month-end, {nm.day.nunique()} sessions "
        f"({nm.day.min().date()} .. {nm.day.max().date()}), same days at every tau:"
    )
    log(
        sb[
            [
                "model",
                "tau",
                "days",
                "traded_ask",
                "pnl_per_day_ask",
                "t_pnl_day_ask",
                "t_ask",
                "traded_mid",
                "pnl_per_day_mid",
                "t_pnl_day_mid",
                "t_mid",
                "rel_spread_median",
            ]
        ].to_string(index=False, float_format=lambda v: f"{v:.4f}")
    )
    log(
        "\nper-day P&L difference tau - 15:30 (block bootstrap 20 sessions; Holm over 10 taus):"
    )
    log(bb.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    log(
        "\nmonth-ends: unconditional long by tau on XSP (same month-ends at every tau):"
    )
    log(me_t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    best = bb.sort_values("holm_boot_1s").head(4)
    log(
        "\nsmallest Holm-adjusted one-sided p (tau beats 15:30):\n"
        + best[
            [
                "model",
                "tau_minus_1530",
                "stat",
                "point",
                "p_boot_1s",
                "holm_boot_1s",
                "p_nw_1s",
                "holm_nw_1s",
            ]
        ].to_string(index=False, float_format=lambda v: f"{v:.4f}")
    )
    gate["holm"] = {
        "min_holm_boot_1s": float(bb["holm_boot_1s"].min()),
        "min_holm_nw_1s": float(bb["holm_nw_1s"].min()),
        "any_tau_beats_1530_holm_5pct": bool(
            (bb["holm_boot_1s"] < 0.05).any() or (bb["holm_nw_1s"] < 0.05).any()
        ),
        "month_end_min_holm_paired_1s": float(me_t["holm_paired_1s"].min()),
    }
    log("holm " + json.dumps(gate["holm"]))
    gate["spxw_new_taus"] = "PENDING: no SPXW 15:00-15:21 quotes on disk"
    gate["wall_s"] = round(time.time() - T0, 1)
    (OUT / "gate.json").write_text(json.dumps(gate, indent=1, default=str))
    log(f"\nwall clock {gate['wall_s']} s")
    (OUT / "summary.txt").write_text("\n".join(_LOG), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
