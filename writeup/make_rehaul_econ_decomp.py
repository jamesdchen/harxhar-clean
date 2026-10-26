"""Where the straddle strategy's profits come from (2026-10-25): a decomposition of the sign strategy of Section 7
(writeup/rehaul/07_economics.tex) for the pooled intraday forecasts of the final rerun.

DATA (reused, not re-implemented)
  The straddle frame, the forecasts and the trade are those of writeup/make_rehaul_economics.py (imported as `me`):
  me.load() (the quote book results/atm_straddle_0dte_1530/daily_a0.parquet joined to the master table's 866 option
  days), me.add_pooled_linear / me.add_pooled_trees (the six full-set pooled forecasts of the final rerun, read at the
  16:00 bar through writeup/pooled_linear_13.py and writeup/pooled_trees_13.py), the HAR-benchmark pooled linear rows
  (poolpt13_<est>_baseline, added by me.add_pooled_linear) and the HAR-benchmark pooled trees (me.tree_16, traded as
  me.pooled_rank_series trades them; gated against it).  The closing-interval HAR forecast is the master table's
  sub_ols_baseline row (me.OLS).  Position q_t = sign(sigma2_t - IV_t^2), ties short; mid return q_t R_t; bid/ask
  return master_table_gui.rows._crossed_return (a buy pays the ask, a sale receives the bid, at entry only).  Every
  statistic is on the 614 post-warmup option days t > 252 (me.WARMUP), the sample of the paper's Sharpe ratios.
  Run with REHAUL_FINAL_RUN=1 (writeup/final_run.py); the script stops if the switch is off or an arm is pending.
  Market data: the 15:30 underlying print S_t, the strikes K_C, K_P and the close S^close_t of the quote book; the
  S&P 500 daily closes of the scored frame's settlement source (results/atm_straddle_0dte_1530/cache/
  gspc_close_*.parquet, asserted equal to S^close_t on all 866 option days), which give the close-to-close return on
  every trading day; the VIX daily close (^VIX in results/atm_straddle_0dte_1530/cross_asset_daily_closes.parquet).
  Event calendar: data/releases.parquet (CPI, employment report = NFP, PPI, GDP, retail sales, trade balance, FOMC
  statement; a day is tagged when any of its bars carries the flag).  The file is stale: its FOMC flags end on
  1 November 2023 and its macro-release flags between October 2023 and January 2024.  FOMC statement days after its
  last flag are taken from live/close_signal/state/fomc_statement_dates.csv (maintained from the Federal Reserve's
  calendar), so the FOMC tag covers the whole sample; the macro-release tags do not cover days after the earliest
  last flag of the six release series (\\rhedRelEndText), and those days carry a dagger in the best/worst-days table.
  Monthly SPX expiration: the third Friday of the month (the previous trading day when it is a holiday); triple
  witching: the March, June, September and December ones; month end: the last trading day of the month.

GATE (before anything new; the run stops on a mismatch)
  \\rhrShEnetPoolFull = 1.94 and every other \\rhrSh(X)<Est>Pool(Full|Har) (writeup/generated/rehaul_results_numbers.tex),
  \\rheShortSh, \\rheShortShX, \\rheWorthlessDays and \\rhePoolWorthlessShareMed / Min / Max
  (writeup/generated/rehaul_econ_numbers.tex) are recomputed and compared with the printed macros.

STATISTICS
  1. Long vs short days per forecast: counts, mean return (mid, bid/ask), hit rate (q_t = sign(RV_t - IV_t^2)), share
     of days with a positive return, share of the total return.  Identity: r_t = -R_t + 2 R_t 1{q_t = +1}, so the gain
     over the unconditional short straddle, Delta_t = 2 R_t 1{q_t = +1}, accrues on long days only.
  2. Quintiles of the forecast gap log(sigma2_t / IV_t^2) over the 614 days (each forecast its own breakpoints;
     "Mean" panels average the six forecasts' quintile statistics, and their return shares are ratios of sums).
  3. Calibration: mean sigma2 / mean RV; mean(RV / sigma2) (the multiplicative correction of the Jensen bias of
     05_methods.tex, the scale that minimizes mean QLIKE).  Diagnostic correction: c_t = mean over the earlier option
     days t' < t (all 866 days, warmup included) of RV / sigma2; q^c_t = sign(c_t sigma2_t - IV_t^2).  The level
     scale sum RV / sum sigma2 over t' < t is the alternative (macros only).
  4. Each full-set forecast's 10 best and 10 worst days by its mid return; their union over the six forecasts with
     tags; the change in Delta SR (vs the short straddle and vs the HAR twin) when those days are dropped from both
     series (D6 Sharpe ratio, hac_inference.d6_sharpe, the sample's span held fixed).
  5. Worthless expiries (R_t = -1: the close settles in [K_P, K_C]): observed count against the count implied by a
     normal half-hour log move with the implied volatility IV_t, by the empirical distribution of the earlier days'
     moves standardized by IV (t' < t), and by the earlier days' raw moves; distance of the close to the nearest
     5-point strike against a uniform (Kolmogorov-Smirnov, asymptotic 5 % critical value 1.358 / sqrt(n)).
  6. Risk regressions of the daily return (mid, bid/ask) on (a) the short-straddle return, (b) the S&P 500 15:30-16:00
     log return and its absolute value (basis points), (c) the S&P 500 close-to-close log return (%) and the change in
     the VIX close (points), (d) all jointly.  OLS; HAC standard errors with the Bartlett kernel at the paper's
     bandwidth S = ceil(1.3 T^(1/2)) = 33 (autocovariances divided by T, as hac_inference.bartlett_lrv); two-sided
     fixed-b p-values of each coefficient's t from hac_inference.fixedb_p at b = S / T (the Kiefer-Vogelsang 2005
     limit of a t-test of one coefficient in a regression with an intercept is that of the mean).  The intercept-only
     regression reproduces hac_inference.llsw_test (asserted).

OUTPUTS
  writeup/generated/rehaul_econ_decomp_numbers.tex     \\rhed... macros (macro-only)
  writeup/generated/rehaul_econ_decomp_longshort.tex   tab:rh_econ_longshort
  writeup/generated/rehaul_econ_decomp_quintiles.tex   tab:rh_econ_quintiles
  writeup/generated/rehaul_econ_decomp_bestdays.tex    tab:rh_econ_bestdays
  writeup/generated/rehaul_econ_decomp_riskreg.tex     tab:rh_econ_riskreg
  (REHAUL_GEN_DIR redirects them, as for the other generators.)

Run (Git Bash, repository root):  REHAUL_FINAL_RUN=1 python writeup/make_rehaul_econ_decomp.py
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
WRITEUP = ROOT / "writeup"
for _p in (ROOT, ROOT / "notebooks", WRITEUP):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import final_run as fr  # noqa: E402

if not fr.FINAL:
    raise SystemExit("make_rehaul_econ_decomp.py reads the final rerun: set REHAUL_FINAL_RUN=1")

import hac_inference as hi  # noqa: E402
import make_rehaul_economics as me  # noqa: E402
from master_table_gui import rows as gui  # noqa: E402

GEN = fr.GEN  # outputs (REHAUL_GEN_DIR redirects them)
PAPER_GEN = WRITEUP / "generated"  # the printed macros the gate and the name-clash check read
HEADER = "% AUTO-GENERATED by writeup/make_rehaul_econ_decomp.py -- do not edit; re-run the script.\n"
GUARD = "\\ifdefined\\rhedNDays\\else\\input{generated/rehaul_econ_decomp_numbers}\\fi\n"

EST = (
    ("ridge", "Ridge", "Ridge"),
    ("lasso", "Lasso", "Lasso"),
    ("enet", "Elastic net", "Enet"),
    ("lgbm", "LightGBM", "Lgbm"),
    ("xgb", "XGBoost", "Xgb"),
    ("rf", "Random forest", "Rf"),
)
LINEAR = ("enet", "lasso", "ridge")
HEAD_TEX = {  # column heads of the per-forecast tables (two-word names on two lines)
    "ridge": "Ridge",
    "lasso": "Lasso",
    "enet": "\\makecell[r]{Elastic\\\\net}",
    "lgbm": "LightGBM",
    "xgb": "XGBoost",
    "rf": "\\makecell[r]{Random\\\\forest}",
}
N_TOP = 10  # best / worst days per forecast
SPAN_OF_DAYS = None  # set in main(): (first, last) post-warmup day, the D6 span held fixed when days are dropped
RELEASES = {  # data/releases.parquet column -> tag
    "fomc release": "FOMC",
    "cpi release": "CPI",
    "employment release": "NFP",
    "ppi release": "PPI",
    "gdp release": "GDP",
    "retail release": "RS",
    "trade release": "TB",
}
MACRO_RELEASES = ("CPI", "NFP", "PPI", "GDP", "RS", "TB")


def key_of(est: str, full: bool) -> str:
    pset = "all_features" if full else "baseline"
    return f"poolpt13_{est}_{pset}" if est in LINEAR else f"pooltree_{est}_{pset}"


# ============================================================================= formatting
def fnum(x: float, dp: int = 2, plus: bool = False) -> str:
    """A number for a macro or a text cell: \\ensuremath{-} for a minus (as rehaul_econ_numbers.tex)."""
    v = round(float(x), dp) + 0.0
    s = f"{abs(v):,.{dp}f}".replace(",", "{,}")
    return (r"\ensuremath{-}" if v < 0 else (r"\ensuremath{+}" if plus and v > 0 else "")) + s


def mnum(x: float, dp: int = 2) -> str:
    """A number for a math-mode table cell."""
    v = round(float(x), dp) + 0.0
    s = f"{abs(v):,.{dp}f}".replace(",", "{,}")
    return ("-" if v < 0 else "") + s


def tcell(x: float, dp: int = 1) -> str:
    return "--" if x is None or not np.isfinite(x) else f"${mnum(x, dp)}$"


def pdig(p: float) -> str:
    """A fixed-b p for prose: me.p_digits, or "<0.001" (as a math-mode relation) below 0.001."""
    return me.p_digits(p) if p >= 0.001 else r"\ensuremath{<}0.001"


def nice_date(d: pd.Timestamp) -> str:
    return f"{d.day}~{d.strftime('%B')}~{d.year}"


def short_date(d: pd.Timestamp) -> str:
    return d.strftime("%Y-%m-%d")


MACROS: dict[str, tuple[str, str]] = {}


def mac(name: str, val: str, note: str) -> None:
    assert re.fullmatch(r"rhed[A-Za-z]+", name), name
    assert name not in MACROS, name
    MACROS[name] = (val, note)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    print("wrote", fr.rel(path))


def read_macros(path: Path) -> dict[str, str]:
    """name -> value of every providecommand / renewcommand / newcommand of a generated macro file (the last
    definition of a name wins; values are read with balanced braces)."""
    text = path.read_text(encoding="utf-8")
    out: dict[str, str] = {}
    for m in re.finditer(r"\\(?:provide|new|renew)command\{\\([A-Za-z]+)\}\{", text):
        i, depth = m.end(), 1
        while depth:
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        out[m.group(1)] = text[m.end() : i - 1]
    return out


# ============================================================================= data
def build() -> dict:
    """Everything on the 866 option days; the post-warmup slice is taken by the statistics."""
    mt, model_set, series, fam = me.load()
    me.add_pooled_linear(series, fam)
    trees = me.add_pooled_trees(series, fam)
    if len(trees) != 3 or fr.pool_n() != fr.pool_n(False):
        raise SystemExit(f"a pooled arm is pending ({fr.pending_text()}): this decomposition needs all twelve")
    days = series[me.OLS].index
    daily = pd.read_parquet(me.RES / "master_table_daily.parquet", columns=["key", "day", "iv_var", "rv_unclipped"])
    daily["day"] = pd.to_datetime(daily["day"]).dt.normalize()
    ref = daily[daily["key"] == me.OLS].set_index("day").sort_index()
    ref.index = ref.index.astype(days.dtype)
    ref = ref.reindex(days)
    iv, rv = ref["iv_var"].to_numpy(float), ref["rv_unclipped"].to_numpy(float)
    assert np.isfinite(iv).all() and np.isfinite(rv).all() and (iv > 0).all()

    qb = pd.read_parquet(gui._QUOTE_BOOK, columns=["S", "K_c", "K_p", "S_close", "entry", "exit", "R", "iv_var"])
    qb.index = pd.to_datetime(qb.index).normalize().astype("datetime64[ns]")
    qb = qb.reindex(days)
    b = series[me.OLS]
    R, X = b["R"].to_numpy(float), b["ex"].to_numpy(float)
    bid, ask = b["bid"].to_numpy(float), b["ask"].to_numpy(float)
    assert np.array_equal(qb["R"].to_numpy(float), R) and np.array_equal(qb["exit"].to_numpy(float), X)
    assert np.allclose(qb["iv_var"].to_numpy(float), iv, rtol=1e-12, atol=0.0)
    assert np.allclose(X / qb["entry"].to_numpy(float) - 1.0, R, rtol=0, atol=1e-6)

    # the forecasts: plain = gap + IV^2 (me's rows store the gap s = plain - iv_var), and the HAR-benchmark trees
    plain: dict[str, np.ndarray] = {}
    for e, _, _ in EST:
        for full in (True, False):
            k = key_of(e, full)
            if k in series:
                plain[k] = series[k]["s"].to_numpy(float) + iv
    plain[me.OLS] = series[me.OLS]["s"].to_numpy(float) + iv
    for e in ("lgbm", "xgb", "rf"):
        plain[key_of(e, False)] = me.tree_16(e, "baseline", days, rv)
    assert len(plain) == 13, sorted(plain)
    q = {k: np.where(p > iv, 1.0, -1.0) for k, p in plain.items()}
    mid = {k: q[k] * R for k in plain}
    ba = {k: gui._crossed_return(q[k], X, bid, ask) for k in plain}
    # gates: the rows me trades (position, mid, bid/ask) and me.pooled_rank_series' HAR-benchmark trees
    for k in plain:
        if k in series:
            g = series[k]
            assert np.array_equal(g["q"].to_numpy(float), q[k]), k
            assert np.allclose(g["mid"].to_numpy(float), mid[k], rtol=0, atol=1e-12), k
            assert np.allclose(g["x"].to_numpy(float), ba[k], rtol=0, atol=1e-12), k
    pser = me.pooled_rank_series(series)
    for e in ("lgbm", "xgb", "rf"):
        k = key_of(e, False)
        assert np.allclose(pser[k]["mid"].to_numpy(float), mid[k], rtol=0, atol=1e-12), k
    mid["short"], ba["short"] = series[me.SHORT]["mid"].to_numpy(float), series[me.SHORT]["x"].to_numpy(float)
    q["short"] = -np.ones(len(days))

    # market data
    gcf = sorted((ROOT / "results" / "atm_straddle_0dte_1530" / "cache").glob("gspc_close_*.parquet"))
    assert len(gcf) == 1, gcf
    gc = pq.read_table(gcf[0]).to_pandas()["close"].astype(float)
    gc.index = pd.to_datetime(gc.index).normalize().astype("datetime64[ns]")
    S, Sc = qb["S"].to_numpy(float), qb["S_close"].to_numpy(float)
    assert np.array_equal(gc.reindex(days).to_numpy(float), Sc), "S^close_t is the cached S&P 500 close"
    rcc = np.log(gc).diff()
    ca = pq.read_table(ROOT / "results" / "atm_straddle_0dte_1530" / "cross_asset_daily_closes.parquet").to_pandas()
    vix = ca[ca["ticker"] == "^VIX"].set_index("date")["close"].astype(float)
    vix.index = pd.to_datetime(vix.index).normalize().astype("datetime64[ns]")
    vix = vix.reindex(gc.index)
    dvix = vix.diff()
    tdays = pd.DatetimeIndex(gc.index)

    # calendar
    rel = pq.read_table(ROOT / "data" / "releases.parquet").to_pandas()
    rel["d"] = pd.to_datetime(rel["endbartime"]).dt.normalize().astype("datetime64[ns]")
    ev: dict[str, pd.DatetimeIndex] = {}
    last: dict[str, pd.Timestamp] = {}
    for col, tag in RELEASES.items():
        ev[tag] = pd.DatetimeIndex(rel.loc[rel[col] > 0, "d"].unique()).sort_values()
        last[tag] = ev[tag].max()
    fomc_live = pd.read_csv(ROOT / "live" / "close_signal" / "state" / "fomc_statement_dates.csv", comment="#")
    live = pd.DatetimeIndex(pd.to_datetime(fomc_live.iloc[:, 0])).astype("datetime64[ns]")
    supp = live[live > last["FOMC"]]
    ev["FOMC"] = ev["FOMC"].union(supp)

    def third_friday(y: int, m: int) -> pd.Timestamp:
        d = pd.Timestamp(year=y, month=m, day=15)
        while d.weekday() != 4:
            d += pd.Timedelta(days=1)
        while d not in tdays:
            d -= pd.Timedelta(days=1)
        return d

    yrs = range(days.min().year, days.max().year + 1)
    ev["OpEx"] = pd.DatetimeIndex([third_friday(y, m) for y in yrs for m in range(1, 13)])
    ev["TW"] = pd.DatetimeIndex([third_friday(y, m) for y in yrs for m in (3, 6, 9, 12)])
    ev["ME"] = pd.DatetimeIndex(pd.Series(tdays).groupby([tdays.year, tdays.month]).max().to_numpy())
    rel_end = min(last[t] for t in MACRO_RELEASES)
    return {
        "series": series,
        "days": days,
        "iv": iv,
        "rv": rv,
        "R": R,
        "X": X,
        "bid": bid,
        "ask": ask,
        "S": S,
        "Kc": qb["K_c"].to_numpy(float),
        "Kp": qb["K_p"].to_numpy(float),
        "Sc": Sc,
        "plain": plain,
        "q": q,
        "mid": mid,
        "ba": ba,
        "r30": np.log(Sc / S),
        "rcc": rcc.reindex(days).to_numpy(float),
        "dvix": dvix.reindex(days).to_numpy(float),
        "ev": ev,
        "last": last,
        "rel_end": rel_end,
        "fomc_supp": supp,
        "pser": pser,
    }


# ============================================================================= gate
def gate(d: dict) -> None:
    """The printed values this decomposition builds on, recomputed (the run stops on a mismatch)."""
    rr = read_macros(PAPER_GEN / "rehaul_results_numbers.tex")
    re_ = read_macros(PAPER_GEN / "rehaul_econ_numbers.tex")
    W = me.WARMUP
    bad = []
    for e, _, cap in EST:
        for full, tag in ((True, "Full"), (False, "Har")):
            k = key_of(e, full)
            for col, nm in (("mid", f"rhrSh{cap}Pool{tag}"), ("ba", f"rhrShX{cap}Pool{tag}")):
                got = fnum(me.sharpe_days(d[col][k][W:]), 2)
                if got != rr[nm]:
                    bad.append((nm, got, rr[nm]))
    for col, nm in (("mid", "rheShortSh"), ("ba", "rheShortShX")):
        got = fnum(me.sharpe_days(d[col]["short"][W:]), 2)
        if got != re_[nm]:
            bad.append((nm, got, re_[nm]))
    Rp = d["R"][W:]
    worth = Rp == -1.0
    if str(int(worth.sum())) != re_["rheWorthlessDays"]:
        bad.append(("rheWorthlessDays", str(int(worth.sum())), re_["rheWorthlessDays"]))
    shares = []
    for k in d["pser"]:
        r = d["mid"][k][W:]
        pos = r > 0
        shares.append(100.0 * r[pos & worth].sum() / r[pos].sum())
    for nm, v in (
        ("rhePoolWorthlessShareMed", np.median(shares)),
        ("rhePoolWorthlessShareMin", min(shares)),
        ("rhePoolWorthlessShareMax", max(shares)),
    ):
        if f"{v:.0f}" != re_[nm]:
            bad.append((nm, f"{v:.0f}", re_[nm]))
    if bad:
        raise SystemExit(f"GATE FAILED (recomputed, printed): {bad}")
    print(
        "GATE ok: rhrShEnetPoolFull",
        rr["rhrShEnetPoolFull"],
        "rheShortSh",
        re_["rheShortSh"],
        "rheWorthlessDays",
        re_["rheWorthlessDays"],
        "rhePoolWorthlessShareMed",
        re_["rhePoolWorthlessShareMed"],
        "(and the 24 rhrSh(X)<Est>Pool(Full|Har), rheShortShX, the worthless-share range)",
    )


# ============================================================================= statistics
def sr(x: np.ndarray, dates: pd.DatetimeIndex) -> float:
    """The paper's D6 Sharpe ratio on the days given, the post-warmup span held fixed."""
    return float(hi.d6_sharpe(np.asarray(x, float), dates=dates, span=SPAN_OF_DAYS)["sharpe"])


def hac_ols(y: np.ndarray, X: np.ndarray) -> dict:
    """OLS of y on [1, X]; Bartlett HAC at the LLSW bandwidth; fixed-b two-sided p of each t."""
    y = np.asarray(y, float)
    T = len(y)
    Xa = np.asarray(X, float)
    Xa = Xa[:, None] if Xa.ndim == 1 else Xa
    Z = np.column_stack([np.ones(T), Xa]) if Xa.shape[1] else np.ones((T, 1))
    S = hi.llsw_bandwidth(T)
    ZtZi = np.linalg.inv(Z.T @ Z)
    beta = ZtZi @ Z.T @ y
    u = y - Z @ beta
    g = Z * u[:, None]
    om = g.T @ g / T
    for j in range(1, S):
        w = 1.0 - j / S
        G = g[j:].T @ g[:-j] / T
        om += w * (G + G.T)
    V = T * ZtZi @ om @ ZtZi
    se = np.sqrt(np.diag(V))
    t = beta / se
    p = np.array([hi.fixedb_p(float(tt), S / T) for tt in t])
    r2 = 1.0 - float(u @ u) / float((y - y.mean()) @ (y - y.mean()))
    return {"b": beta, "t": t, "p": p, "r2": r2, "S": S}


def long_short(d: dict, k: str) -> dict:
    W = me.WARMUP
    q, Rp, rvp, ivp = d["q"][k][W:], d["R"][W:], d["rv"][W:], d["iv"][W:]
    m, x = d["mid"][k][W:], d["ba"][k][W:]
    L = q > 0
    out: dict = {"n_long": int(L.sum()), "n": len(q)}
    for side, msk in (("L", L), ("S", ~L)):
        out[f"{side}_n"] = int(msk.sum())
        out[f"{side}_mid"] = 100.0 * float(m[msk].mean()) if msk.any() else float("nan")
        out[f"{side}_ba"] = 100.0 * float(x[msk].mean()) if msk.any() else float("nan")
        out[f"{side}_hit"] = 100.0 * float((np.sign(rvp - ivp)[msk] == q[msk]).mean()) if msk.any() else float("nan")
        out[f"{side}_pos"] = 100.0 * float((m[msk] > 0).mean()) if msk.any() else float("nan")
        out[f"{side}_sum"] = float(m[msk].sum())
        out[f"{side}_sumx"] = float(x[msk].sum())
        out[f"{side}_R"] = 100.0 * float(Rp[msk].mean()) if msk.any() else float("nan")
    out["sum"], out["sumx"] = float(m.sum()), float(x.sum())
    out["L_share"] = 100.0 * out["L_sum"] / out["sum"]
    out["S_share"] = 100.0 * out["S_sum"] / out["sum"]
    out["L_sharex"] = 100.0 * out["L_sumx"] / out["sumx"]
    out["mean"] = 100.0 * float(m.mean())
    out["meanx"] = 100.0 * float(x.mean())
    out["short_part"] = 100.0 * float((-Rp).mean())
    gain = m - d["mid"]["short"][W:]
    assert np.allclose(gain, 2.0 * Rp * L, rtol=0, atol=1e-12), k  # Delta_t = 2 R_t 1{long}
    out["gain"] = 100.0 * float(gain.mean())
    out["gain_t"] = me.nw_t(gain)
    out["gain_p"] = me.hac_p(out["gain_t"])
    out["hit"] = 100.0 * float((np.sign(rvp - ivp) == q).mean())
    out["pos"] = 100.0 * float((m > 0).mean())
    out["sh"] = me.sharpe_days(m)
    out["shx"] = me.sharpe_days(x)
    return out


def calib(d: dict, k: str) -> dict:
    W = me.WARMUP
    p, rv, iv, R = d["plain"][k], d["rv"], d["iv"], d["R"]
    pw = slice(W, None)
    out: dict = {
        "level": float(p[pw].mean() / rv[pw].mean()),
        "scale": float((rv[pw] / p[pw]).mean()),
        "med_fc": float(np.median(p[pw] / iv[pw])),
    }
    q0 = d["q"][k][pw]
    for nm, c in (
        ("bc", pd.Series(rv / p).expanding().mean().shift(1).to_numpy()),
        ("lev", (pd.Series(rv).expanding().sum().shift(1) / pd.Series(p).expanding().sum().shift(1)).to_numpy()),
    ):
        cc = c[pw]
        assert np.isfinite(cc).all()
        qc = np.where(cc * p[pw] > iv[pw], 1.0, -1.0)
        flip = qc != q0
        rc = qc * R[pw]
        xc = gui._crossed_return(qc, d["X"][pw], d["bid"][pw], d["ask"][pw])
        out[f"{nm}_cmin"], out[f"{nm}_cmax"] = float(cc.min()), float(cc.max())
        out[f"{nm}_flip"] = int(flip.sum())
        out[f"{nm}_tolong"] = int((flip & (qc > 0)).sum())
        out[f"{nm}_long"] = int((qc > 0).sum())
        out[f"{nm}_sh"] = me.sharpe_days(rc)
        out[f"{nm}_shx"] = me.sharpe_days(xc)
        out[f"{nm}_flipR"] = 100.0 * float(R[pw][flip].mean()) if flip.any() else float("nan")
    return out


def quintiles(d: dict, k: str) -> pd.DataFrame:
    W = me.WARMUP
    pw = slice(W, None)
    p, iv, rv, R = d["plain"][k][pw], d["iv"][pw], d["rv"][pw], d["R"][pw]
    q, m, x = d["q"][k][pw], d["mid"][k][pw], d["ba"][k][pw]
    g = np.log(p / iv)
    qi = pd.qcut(g, 5, labels=False)
    df = pd.DataFrame(
        {"qi": qi, "g": g, "rr": rv / iv, "R": R, "long": q > 0, "m": m, "x": x, "hit": np.sign(rv - iv) == q}
    )
    out = df.groupby("qi").agg(
        n=("m", "size"),
        fc=("g", lambda v: float(np.exp(v.mean()))),
        long=("long", "mean"),
        rr=("rr", "mean"),
        R=("R", "mean"),
        m=("m", "mean"),
        x=("x", "mean"),
        hit=("hit", "mean"),
        sum=("m", "sum"),
    )
    out.loc["all"] = [
        len(df),
        float(np.exp(df["g"].mean())),
        df["long"].mean(),
        df["rr"].mean(),
        df["R"].mean(),
        df["m"].mean(),
        df["x"].mean(),
        df["hit"].mean(),
        df["m"].sum(),
    ]
    assert out.loc[[0, 1, 2, 3, 4], "n"].sum() == len(df)
    return out


def drop_days(a: np.ndarray, b: np.ndarray, dates: pd.DatetimeIndex, drop: np.ndarray) -> float:
    keep = ~drop
    return sr(a[keep], dates[keep]) - sr(b[keep], dates[keep])


# ============================================================================= tables
def table_longshort(d: dict, ls: dict, cal: dict) -> str:
    cols = [key_of(e, True) for e, _, _ in EST]
    har = [key_of(e, False) for e, _, _ in EST]

    def mean_of(keys: list[str], f) -> float:
        return float(np.mean([f(k) for k in keys]))

    def share(keys: list[str], side: str, col: str = "") -> float:
        return 100.0 * sum(ls[k][f"{side}_sum{col}"] for k in keys) / sum(ls[k][f"sum{col}"] for k in keys)

    def row(label: str, f, dp: int = 1, share_side: str | None = None) -> str:
        cells = [tcell(f(k), dp) for k in cols]
        if share_side:
            cells += [tcell(share(cols, share_side), dp), tcell(share(har, share_side), dp)]
        else:
            cells += [tcell(mean_of(cols, f), dp), tcell(mean_of(har, f), dp)]
        return f"{label} & " + " & ".join(cells) + " \\\\\n"

    body = me.panel_head(r"Panel A: Long days ($q_t = +1$)", 9, first=True)
    body += row("Days", lambda k: ls[k]["L_n"], 0)
    body += row(r"Mean return, mid (\%)", lambda k: ls[k]["L_mid"])
    body += row(r"Mean return, bid/ask (\%)", lambda k: ls[k]["L_ba"])
    body += row(r"$\RV_t > \mathrm{IV}_t^2$ (\%)", lambda k: ls[k]["L_hit"])
    body += row(r"Positive return (\%)", lambda k: ls[k]["L_pos"])
    body += row(r"Share of return, mid (\%)", lambda k: ls[k]["L_share"], share_side="L")
    body += me.panel_head(r"Panel B: Short days ($q_t = -1$)", 9)
    body += row("Days", lambda k: ls[k]["S_n"], 0)
    body += row(r"Mean return, mid (\%)", lambda k: ls[k]["S_mid"])
    body += row(r"Mean return, bid/ask (\%)", lambda k: ls[k]["S_ba"])
    body += row(r"$\RV_t < \mathrm{IV}_t^2$ (\%)", lambda k: ls[k]["S_hit"])
    body += row(r"Positive return (\%)", lambda k: ls[k]["S_pos"])
    body += row(r"Share of return, mid (\%)", lambda k: ls[k]["S_share"], share_side="S")
    body += me.panel_head(r"Panel C: All \rheSzDays{} post-warmup days", 9)
    body += row(r"Mean return, mid (\%)", lambda k: ls[k]["mean"])
    body += row(r"\quad Short straddle, $-\bar R$ (\%)", lambda k: ls[k]["short_part"])
    top = [me.cell(ls[k]["gain"], 1, me.n_stars_p(ls[k]["gain_p"])) for k in cols]
    top += [me.cell(mean_of(cols, lambda k: ls[k]["gain"]), 1, 0), me.cell(mean_of(har, lambda k: ls[k]["gain"]), 1, 0)]
    under = [me.tstat(ls[k]["gain_t"]) for k in cols] + ["", ""]
    body += me.rows2(r"\quad Gain, $\bar\Delta$ (\%)", top, under)
    body += row(r"Hit rate (\%)", lambda k: ls[k]["hit"])
    body += row(r"Positive return (\%)", lambda k: ls[k]["pos"])
    body += me.panel_head(r"Panel D: Calibration; a bias-corrected forecast (diagnostic)", 9)
    body += row(r"Mean $\hat\sigma^2_t$ / mean $\RV_t$", lambda k: cal[k]["level"], 2)
    body += row(r"Mean of $\RV_t/\hat\sigma^2_t$", lambda k: cal[k]["scale"], 2)
    body += row(r"Long days, corrected", lambda k: cal[k]["bc_long"], 0)
    body += row(r"Positions changed", lambda k: cal[k]["bc_flip"], 0)
    body += row(r"$\bar R$ on changed days (\%)", lambda k: cal[k]["bc_flipR"])
    body += row(r"$\SR$, mid: uncorrected", lambda k: ls[k]["sh"], 2)
    body += row(r"$\SR$, mid: corrected", lambda k: cal[k]["bc_sh"], 2)
    body += row(r"$\SR$, bid/ask: uncorrected", lambda k: ls[k]["shx"], 2)
    body += row(r"$\SR$, bid/ask: corrected", lambda k: cal[k]["bc_shx"], 2)
    heads = " & ".join(HEAD_TEX[e] for e, _, _ in EST)
    return (
        HEADER
        + "% Table tab:rh_econ_longshort: long and short days of the sign strategy, the gain over the short straddle, and\n"
        + "% the calibration of the forecasts (writeup/make_rehaul_econ_decomp.py, statistics 1 and 3).\n"
        + GUARD
        + rf"""\begin{{table}}[tbp]
\centering
\small
\setlength{{\tabcolsep}}{{2.5pt}}
\begin{{threeparttable}}
\caption{{Long and short days of the sign strategy}}
\label{{tab:rh_econ_longshort}}
\begin{{tabular}}{{@{{}}lrrrrrrrr@{{}}}}
\toprule
 & \multicolumn{{6}}{{c}}{{Full predictor set}} & \multicolumn{{2}}{{c}}{{Mean}} \\
\cmidrule(lr){{2-7}}\cmidrule(l){{8-9}}
 & {heads} & Full & HAR \\
\midrule
{body}\bottomrule
\end{{tabular}}
\begin{{tablenotes}}[flushleft]
\footnotesize
\item \textit{{Notes:}} Sign strategy $q_t = \sign(\hat\sigma^2_t - \mathrm{{IV}}_t^2)$ of each pooled intraday forecast
on the full predictor set, on the \rheSzDays{{}} post-warmup option days (Section~\ref{{sec:rh_econ_strategy}}); returns
in \% of the premium. Mean: the mean over the six full-set forecasts and over their six HAR-benchmark counterparts (the
three penalized linear models on the HAR benchmark take the same position on every day); in the share rows, the summed
returns of the six. $\RV_t > \mathrm{{IV}}_t^2$ and $\RV_t < \mathrm{{IV}}_t^2$: the share of the long (short) days on
which realized variance exceeds (falls short of) implied variance, which happens on \rhedUncRVAboveIV\% of all days;
Positive return: the share of days with $r_t > 0$ at mid-quotes; Share of return: the part of the summed mid-quote
return earned on those days; Hit rate: the share of days on which $q_t$ has the sign of $\RV_t - \mathrm{{IV}}_t^2$
(\rhedUncShortHit\% for the unconditional short straddle). Panel C: because
$r_t = -R_t + 2R_t\mathbf{{1}}\{{q_t = +1\}}$, the mean return is that of the unconditional short straddle plus the gain
$\bar\Delta$ over it, earned on the long days only; in parentheses, the $t$-statistic of $\bar\Delta$ (Newey--West,
bandwidth $S = \rheHACS$, fixed-$b$ critical values; Section~\ref{{sec:rh_methods_tests}}). Panel D: the corrected
forecast is $c_t\hat\sigma^2_t$, with $c_t$ the mean of $\RV_{{t'}}/\hat\sigma^2_{{t'}}$ over the earlier option days
$t' < t$; a diagnostic, not a strategy of the paper. $\SR$ is the paper's Sharpe ratio
(Section~\ref{{sec:rh_econ_measures}}).
$^{{*}}$, $^{{**}}$ and $^{{***}}$ denote significance at the 10\%, 5\% and 1\% levels.
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table}}
"""
    )


def table_quintiles(qt: dict) -> str:
    def panel(keys: list[str]) -> str:
        out = ""
        for i, lab in ((0, "1 (lowest)"), (1, "2"), (2, "3"), (3, "4"), (4, "5 (highest)"), ("all", "All")):
            fr_ = [qt[k] for k in keys]
            tot = sum(f.loc["all", "sum"] for f in fr_)
            cells = [
                tcell(np.mean([f.loc[i, "fc"] for f in fr_]), 2),
                tcell(100.0 * np.mean([f.loc[i, "long"] for f in fr_]), 1),
                tcell(np.mean([f.loc[i, "rr"] for f in fr_]), 2),
                tcell(100.0 * np.mean([f.loc[i, "R"] for f in fr_]), 1),
                tcell(100.0 * np.mean([f.loc[i, "m"] for f in fr_]), 1),
                tcell(100.0 * np.mean([f.loc[i, "x"] for f in fr_]), 1),
                tcell(100.0 * np.mean([f.loc[i, "hit"] for f in fr_]), 1),
                tcell(100.0 * sum(f.loc[i, "sum"] for f in fr_) / tot, 1),
            ]
            pre = "\\addlinespace[2pt]\n" if i == "all" else ""
            out += pre + f"{lab} & " + " & ".join(cells) + " \\\\\n"
        return out

    full = [key_of(e, True) for e, _, _ in EST]
    har = [key_of(e, False) for e, _, _ in EST]
    body = me.panel_head("Panel A: Full predictor set (mean of the six forecasts)", 9, first=True)
    body += panel(full)
    body += me.panel_head("Panel B: HAR benchmark (mean of the six forecasts)", 9)
    body += panel(har)
    return (
        HEADER
        + "% Table tab:rh_econ_quintiles: the sign strategy by quintile of the forecast variance gap\n"
        + "% (writeup/make_rehaul_econ_decomp.py, statistic 2).\n"
        + GUARD
        + rf"""\begin{{table}}[tbp]
\centering
\small
\setlength{{\tabcolsep}}{{4pt}}
\begin{{threeparttable}}
\caption{{Straddle returns by quintile of the forecast variance gap}}
\label{{tab:rh_econ_quintiles}}
\begin{{tabular}}{{@{{}}lrrrrrrrr@{{}}}}
\toprule
 & & & & Straddle & \multicolumn{{2}}{{c}}{{Strategy}} & & Share of \\
\cmidrule(lr){{6-7}}
Quintile & $\hat\sigma^2_t/\mathrm{{IV}}_t^2$ & Long (\%) & $\RV_t/\mathrm{{IV}}_t^2$ & $R_t$ (\%) & Mid (\%) & Bid/ask (\%) & Hit (\%) & return (\%) \\
\midrule
{body}\bottomrule
\end{{tabular}}
\begin{{tablenotes}}[flushleft]
\footnotesize
\item \textit{{Notes:}} For each pooled intraday forecast, the \rheSzDays{{}} post-warmup option days are sorted into
quintiles of $\log(\hat\sigma^2_t/\mathrm{{IV}}_t^2)$, the forecast variance gap that signs the straddle position
(Section~\ref{{sec:rh_econ_strategy}}); breakpoints from the whole sample, about 123 days a quintile. Each entry is the mean
over the six forecasts of a panel of the quintile's statistic: $\hat\sigma^2_t/\mathrm{{IV}}_t^2$, the geometric mean of
the forecast ratio; Long, the share of days with $q_t = +1$; $\RV_t/\mathrm{{IV}}_t^2$, the mean ratio of realized to
implied variance; $R_t$, the mean return of a long straddle at mid-quotes; Strategy, the mean return of the sign strategy
at mid-quotes and at quoted bid/ask prices; Hit, the share of days on which $q_t$ has the sign of
$\RV_t - \mathrm{{IV}}_t^2$; Share of return, the quintile's part of the summed mid-quote returns of the six
forecasts. Returns in \% of the premium.
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table}}
"""
    )


def tags_of(d: dict, day: pd.Timestamp) -> str:
    ev = d["ev"]
    t = [nm for nm in ("FOMC", *MACRO_RELEASES) if day in ev[nm]]
    if day in ev["TW"]:
        t.append("TW")
    elif day in ev["OpEx"]:
        t.append("OpEx")
    if day in ev["ME"]:
        t.append("ME")
    s = ", ".join(t) if t else "--"
    if day > d["rel_end"]:
        s += r"$^{\dagger}$"
    return s


def table_bestdays(d: dict, bw: dict, contrib: dict) -> str:
    W = me.WARMUP
    pwd = d["days"][W:]
    full = [key_of(e, True) for e, _, _ in EST]
    Q = np.array([d["q"][k][W:] for k in full])
    R, X, bid, ask = d["R"][W:], d["X"][W:], d["bid"][W:], d["ask"][W:]
    rr = d["rv"][W:] / d["iv"][W:]

    def day_rows(cnt: dict, members: dict) -> str:
        out = ""
        for day in sorted(cnt):
            i = pwd.get_loc(day)
            n_long = int((Q[:, i] > 0).sum())
            pos = {float(d["q"][k][W:][i]) for k in members[day]}  # the position of the forecasts that count the day
            assert len(pos) == 1, (day, members[day])
            lng = pos.pop() > 0
            r_mid = R[i] if lng else -R[i]
            r_ba = X[i] / ask[i] - 1.0 if lng else 1.0 - X[i] / bid[i]
            cells = [
                short_date(day),
                str(cnt[day]),
                f"{n_long}",
                tcell(100.0 * r_mid, 0),
                tcell(100.0 * r_ba, 0),
                tcell(rr[i], 2),
                tcell(100.0 * d["r30"][W:][i], 2),
                tcell(100.0 * d["rcc"][W:][i], 2),
                tags_of(d, day),
            ]
            out += " & ".join(cells) + " \\\\\n"
        return out

    body = me.panel_head("Panel A: Best days", 9, first=True) + day_rows(bw["best_cnt"], bw["best_of"])
    body += me.panel_head("Panel B: Worst days", 9) + day_rows(bw["worst_cnt"], bw["worst_of"])
    body2 = ""
    for e, nm, _ in EST:
        c = contrib[key_of(e, True)]
        cells = [
            tcell(c["vs_short_all"], 2),
            tcell(c["vs_short_exb"], 2),
            tcell(c["vs_short_exbw"], 2),
            tcell(c["best_share"], 0),
            tcell(c["vs_har_all"], 2),
            tcell(c["vs_har_exb"], 2),
            tcell(c["vs_har_exbw"], 2),
        ]
        body2 += f"{nm} & " + " & ".join(cells) + " \\\\\n"
    return (
        HEADER
        + "% Table tab:rh_econ_bestdays: the ten best and ten worst days of each full-set pooled forecast, their union\n"
        + "% over the six, and what they contribute to Delta SR (writeup/make_rehaul_econ_decomp.py, statistic 4).\n"
        + GUARD
        + rf"""\begin{{table}}[tbp]
\centering
\small
\setlength{{\tabcolsep}}{{3.5pt}}
\begin{{threeparttable}}
\caption{{The best and worst days of the sign strategy}}
\label{{tab:rh_econ_bestdays}}
\begin{{tabular}}{{@{{}}lrrrrrrrl@{{}}}}
\toprule
 & In & Long & \multicolumn{{2}}{{c}}{{Return (\%)}} & & \multicolumn{{2}}{{c}}{{S\&P~500 (\%)}} & \\
\cmidrule(lr){{4-5}}\cmidrule(lr){{7-8}}
Date & ten & (of 6) & Mid & Bid/ask & $\RV_t/\mathrm{{IV}}_t^2$ & 15:30--16:00 & Daily & Events \\
\midrule
{body}\bottomrule
\end{{tabular}}

\medskip
\begin{{tabular}}{{@{{}}lrrrrrrr@{{}}}}
\toprule
 & \multicolumn{{4}}{{c}}{{$\Delta\SR$ against the short straddle}} & \multicolumn{{3}}{{c}}{{$\Delta\SR$ against the HAR twin}} \\
\cmidrule(lr){{2-5}}\cmidrule(l){{6-8}}
Panel C & All days & No best & No best, & Best-day & All days & No best & No best, \\
 & & & no worst & share (\%) & & & no worst \\
\midrule
{body2}\bottomrule
\end{{tabular}}
\begin{{tablenotes}}[flushleft]
\footnotesize
\item \textit{{Notes:}} Panels A and B: the union over the six pooled intraday forecasts on the full predictor set of
each forecast's \rhedNTop{{}} best and \rhedNTop{{}} worst post-warmup days by its mid-quote return. In ten: the number of
the six forecasts with the day among their ten; Long: the number of the six that hold a long straddle. Return: of the
position of the forecasts that count the day among their ten (a long straddle bought at the ask, a short one sold at
the bid, in the bid/ask column). $\RV_t/\mathrm{{IV}}_t^2$: realized
over implied variance of the closing half hour. S\&P~500: the log return from the 15:30 underlying print $S_t$ to the
close $S^{{\mathrm{{close}}}}_t$, and from the previous trading day's close. Events: FOMC statement, CPI, employment report
(NFP), PPI, GDP, retail sales (RS) and trade balance (TB) release days; monthly SPX expiration (OpEx; TW, the quarterly
triple witching); month end (ME). $^{{\dagger}}$: after \rhedRelEndText, when the release calendar ends (the FOMC
calendar is complete). Panel C: for each forecast, the Sharpe ratio gain $\Delta\SR$ (mid-quotes) over the unconditional
short straddle and over the same model class on the HAR benchmark, on all \rheSzDays{{}} post-warmup days, without the
forecast's ten best days, and without its ten best and ten worst days (dropped from both series); Best-day share: the
part of the summed return gain over the short straddle earned on the ten best days.
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table}}
"""
    )


REG_SPECS = (
    ("m", "Mean return", ()),
    ("a", "Short-straddle return", ("short",)),
    ("b", "S\\&P~500 half-hour return", ("r30", "abs30")),
    ("c", "S\\&P~500 daily return and VIX", ("rcc", "dvix")),
    ("d", "Joint", ("short", "r30", "abs30", "rcc", "dvix")),
)
REG_LABEL = {
    "short": r"$\beta$, short straddle",
    "r30": r"$\beta$, $r^{30}_t$ (bp)",
    "abs30": r"$\beta$, $|r^{30}_t|$ (bp)",
    "rcc": r"$\beta$, $r^{\mathrm{cc}}_t$ (\%)",
    "dvix": r"$\beta$, $\Delta\mathrm{VIX}_t$",
}


def regressors(d: dict) -> dict:
    """The regressors on the 614 days. The short-straddle return is a traded return and enters as it is; the four
    non-traded series are demeaned, so the intercept is the mean return net of the short-straddle exposure only."""
    W = me.WARMUP
    nontraded = {
        "r30": 1e4 * d["r30"][W:],
        "abs30": 1e4 * np.abs(d["r30"][W:]),
        "rcc": 100.0 * d["rcc"][W:],
        "dvix": d["dvix"][W:],
    }
    out = {"short": 100.0 * d["mid"]["short"][W:]}
    out.update({k: v - v.mean() for k, v in nontraded.items()})
    return out


def run_regs(d: dict, ys: dict) -> dict:
    """ys: name -> (mid, bid/ask) daily returns in % of premium on the 614 days. Returns name -> col -> spec -> fit."""
    Xs = regressors(d)
    out: dict = {}
    for nm, (ym, yx) in ys.items():
        out[nm] = {}
        for col, y in (("mid", ym), ("ba", yx)):
            out[nm][col] = {}
            for sp, _, regs in REG_SPECS:
                if nm == "short" and "short" in regs:
                    continue
                X = np.column_stack([Xs[r] for r in regs]) if regs else np.zeros((len(y), 0))
                out[nm][col][sp] = hac_ols(y, X)
    return out


def table_riskreg(regs: dict) -> str:
    cols = [key_of(e, True) for e, _, _ in EST] + ["avg", "short"]

    def coef_row(col: str, sp: str, j: int, lab: str) -> str:
        top, under = [], []
        for c in cols:
            f = regs[c][col].get(sp)
            if f is None:
                top.append("--$^{\\phantom{***}}$")
                under.append("")
                continue
            top.append(me.cell(f["b"][j], 1 if j == 0 else 2, me.n_stars_p(f["p"][j])))
            under.append(me.tstat(f["t"][j]))
        return me.rows2(lab, top, under)

    def r2_row(col: str, sp: str) -> str:
        r2 = [tcell(regs[c][col][sp]["r2"], 2) if sp in regs[c][col] else "--" for c in cols]
        return r"$R^2$ & " + " & ".join(r2) + " \\\\[3pt]\n"

    body = me.panel_head("Panel A: Returns at mid-quotes", 9, first=True)
    body += coef_row("mid", "m", 0, r"Mean return (\%)")
    for sp, title, rg in REG_SPECS[1:]:
        body += me.sub_head(f"({sp}) {title}", 9)
        if "short" in rg:
            body += coef_row("mid", sp, 0, r"$\alpha$ (\%)")
        for j, nm in enumerate(rg, start=1):
            body += coef_row("mid", sp, j, REG_LABEL[nm])
        body += r2_row("mid", sp)
    body += me.panel_head("Panel B: Returns at quoted bid/ask prices", 9)
    body += coef_row("ba", "m", 0, r"Mean return (\%)")
    body += coef_row("ba", "a", 0, r"$\alpha$ (\%), (a)")
    body += coef_row("ba", "d", 0, r"$\alpha$ (\%), (d)")
    heads = " & ".join(HEAD_TEX[e] for e, _, _ in EST)
    return (
        HEADER
        + "% Table tab:rh_econ_riskreg: risk regressions of the sign strategy's daily return\n"
        + "% (writeup/make_rehaul_econ_decomp.py, statistic 6).\n"
        + GUARD
        + rf"""\begin{{table}}[tbp]
\centering
\footnotesize
\setlength{{\tabcolsep}}{{2.5pt}}
\begin{{threeparttable}}
\caption{{Risk exposures of the sign strategy}}
\label{{tab:rh_econ_riskreg}}
\begin{{tabular}}{{@{{}}lrrrrrrrr@{{}}}}
\toprule
 & \multicolumn{{6}}{{c}}{{Full predictor set}} & & \\
\cmidrule(lr){{2-7}}
 & {heads} & Mean & \makecell[r]{{Short\\straddle}} \\
\midrule
{body}\bottomrule
\end{{tabular}}
\begin{{tablenotes}}[flushleft]
\footnotesize
\item \textit{{Notes:}} OLS regressions of the daily return of the sign strategy (in \% of the premium) on the
\rheSzDays{{}} post-warmup option days. Columns: the six pooled intraday forecasts on the full predictor set, their
equal-weight mean return (Mean), and the unconditional short straddle. Regressors: (a) the return of the unconditional
short straddle at mid-quotes, $-R_t$ (\%); (b) the S\&P~500 log return from the 15:30 underlying print to the close,
$r^{{30}}_t = \log(S^{{\mathrm{{close}}}}_t/S_t)$, and its absolute value, in basis points; (c) the S\&P~500
close-to-close log return $r^{{\mathrm{{cc}}}}_t$ (\%) and the daily change in the VIX close (points); (d) all of them.
The four non-traded regressors of (b)--(d) are demeaned, so the intercept $\alpha$ is the mean return net of the
exposure to the short straddle, a traded return; in (b) and (c) it is the mean return. In parentheses, $t$-statistics with
Newey--West standard errors (Bartlett kernel, bandwidth $S = \rheHACS$); stars from fixed-$b$ critical values
(Section~\ref{{sec:rh_methods_tests}}). $^{{*}}$, $^{{**}}$ and $^{{***}}$ denote significance at the 10\%, 5\% and 1\% levels.
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table}}
"""
    )


# ============================================================================= main
def main() -> None:
    global SPAN_OF_DAYS
    d = build()
    gate(d)
    W = me.WARMUP
    days = d["days"]
    pwd = days[W:]
    assert len(pwd) == me.N_PW == 614
    SPAN_OF_DAYS = (pwd[0], pwd[-1])
    for k in d["mid"]:
        if k != "short":
            assert abs(sr(d["mid"][k][W:], pwd) - me.sharpe_days(d["mid"][k][W:])) < 1e-12, k
    full = [key_of(e, True) for e, _, _ in EST]
    har = [key_of(e, False) for e, _, _ in EST]
    lin_har_q = [d["q"][key_of(e, False)] for e in LINEAR]
    har_lin_same = all(np.array_equal(lin_har_q[0], x) for x in lin_har_q[1:])
    assert har_lin_same, "the table note says the three penalized linear HAR forecasts take the same positions"
    Rp, rvp, ivp = d["R"][W:], d["rv"][W:], d["iv"][W:]

    mac("rhedNDays", str(len(pwd)), "post-warmup option days (= rheSzDays)")
    mac("rhedNFull", str(len(full)), "pooled intraday forecasts on the full predictor set")
    mac("rhedNTop", str(N_TOP), "best / worst days per forecast")

    # ---------------------------------------------------------------- 1. long vs short
    ls = {k: long_short(d, k) for k in [*full, *har, me.OLS, "short"]}
    print("\n1. LONG / SHORT (mid, % of premium)")
    for k, v in ls.items():
        print(
            f"  {k:30s} long {v['L_n']:3d}  meanL {v['L_mid']:+6.1f} meanS {v['S_mid']:+6.1f}  hitL {v['L_hit']:5.1f} "
            f"hitS {v['S_hit']:5.1f} hit {v['hit']:5.1f} pos {v['pos']:5.1f}  shareL {v['L_share']:6.1f}  "
            f"gain {v['gain']:+5.1f} (t {v['gain_t']:+.2f})  SR {v['sh']:.2f} / {v['shx']:.2f}"
        )

    def rng(keys: list[str], f, nm: str, note: str, dp: int = 1, plus: bool = False) -> None:
        vals = [f(k) for k in keys]
        mac(f"{nm}Min", fnum(min(vals), dp, plus), f"lowest {note}")
        mac(f"{nm}Max", fnum(max(vals), dp, plus), f"highest {note}")

    fs = "across the six full-set pooled forecasts, 614 days"
    hs = "across the six HAR-benchmark pooled forecasts, 614 days"
    mac("rhedUncRVAboveIV", fnum(100.0 * (rvp > ivp).mean(), 1), "% of the 614 days with RV_t > IV_t^2")
    mac("rhedUncRVIVMean", fnum(float((rvp / ivp).mean()), 2), "mean RV_t / IV_t^2, 614 days")
    mac("rhedUncRVIVMed", fnum(float(np.median(rvp / ivp)), 2), "median RV_t / IV_t^2, 614 days")
    mac(
        "rhedUncShortMean",
        fnum(ls["short"]["mean"], 1),
        "mean mid return of the unconditional short straddle, % of premium",
    )
    mac(
        "rhedUncShortHit",
        fnum(ls["short"]["hit"], 1),
        "% of days on which q = -1 matches sign(RV - IV^2) (always short)",
    )
    rng(full, lambda k: ls[k]["L_n"], "rhedLongN", f"number of long days {fs}", 0)
    rng(full, lambda k: 100.0 * ls[k]["L_n"] / ls[k]["n"], "rhedLongPct", f"% long days {fs}", 0)
    rng(har, lambda k: ls[k]["L_n"], "rhedHarLongN", f"number of long days {hs}", 0)
    rng(full, lambda k: ls[k]["L_mid"], "rhedLongMid", f"mean mid return on long days, % {fs}", 0)
    rng(full, lambda k: ls[k]["L_ba"], "rhedLongBa", f"mean bid/ask return on long days, % {fs}", 0)
    rng(full, lambda k: ls[k]["S_mid"], "rhedShortMid", f"mean mid return on short days, % {fs}", 0)
    rng(full, lambda k: ls[k]["S_ba"], "rhedShortBa", f"mean bid/ask return on short days, % {fs}", 0)
    rng(har, lambda k: ls[k]["L_mid"], "rhedHarLongMid", f"mean mid return on long days, % {hs}", 0)
    rng(har, lambda k: ls[k]["S_mid"], "rhedHarShortMid", f"mean mid return on short days, % {hs}", 0)
    rng(full, lambda k: ls[k]["L_hit"], "rhedLongHit", f"% of long days with RV > IV^2 {fs}", 0)
    rng(har, lambda k: ls[k]["L_hit"], "rhedHarLongHit", f"% of long days with RV > IV^2 {hs}", 0)
    rng(full, lambda k: ls[k]["L_pos"], "rhedLongPos", f"% of long days with a positive return {fs}", 0)
    rng(full, lambda k: ls[k]["hit"], "rhedHit", f"% of days on which q matches sign(RV - IV^2) {fs}", 1)
    rng(full, lambda k: ls[k]["pos"], "rhedPos", f"% of days with a positive mid return {fs}", 1)
    rng(full, lambda k: ls[k]["L_share"], "rhedLongShare", f"% of the summed mid return earned on long days {fs}", 0)
    rng(
        full,
        lambda k: ls[k]["L_sharex"],
        "rhedLongShareX",
        f"% of the summed bid/ask return earned on long days {fs}",
        0,
    )
    rng(full, lambda k: ls[k]["mean"], "rhedMean", f"mean mid return, % {fs}", 1)
    rng(full, lambda k: ls[k]["gain"], "rhedGain", f"mean gain over the short straddle (2 R 1(long)), % {fs}", 1)
    rng(
        full,
        lambda k: 100.0 * ls[k]["gain"] / ls[k]["mean"],
        "rhedGainShare",
        f"gain over short / mean return, % {fs}",
        0,
    )
    rng(har, lambda k: ls[k]["gain"], "rhedHarGain", f"mean gain over the short straddle, % {hs}", 1)
    mac("rhedOlsLongN", str(ls[me.OLS]["L_n"]), "long days of the closing-interval HAR forecast (sub_ols_baseline)")
    mac("rhedOlsLongMid", fnum(ls[me.OLS]["L_mid"], 0), "... its mean mid return on long days, %")
    mac("rhedOlsShortMid", fnum(ls[me.OLS]["S_mid"], 0), "... its mean mid return on short days, %")
    mac("rhedOlsLongHit", fnum(ls[me.OLS]["L_hit"], 0), "... % of its long days with RV > IV^2")
    mac("rhedOlsSh", fnum(ls[me.OLS]["sh"], 2), "... its Sharpe ratio, mid (= rhrOlsSharpe)")

    # ---------------------------------------------------------------- 3. calibration
    cal = {k: calib(d, k) for k in [*full, *har, me.OLS]}
    print("\n3. CALIBRATION / BIAS CORRECTION (diagnostic)")
    for k, v in cal.items():
        print(
            f"  {k:30s} mean fc/mean RV {v['level']:.3f} mean RV/fc {v['scale']:.3f} med fc/IV {v['med_fc']:.3f} | "
            f"c {v['bc_cmin']:.3f}-{v['bc_cmax']:.3f} flips {v['bc_flip']} (to long {v['bc_tolong']}) "
            f"SR {ls[k]['sh']:.2f}->{v['bc_sh']:.2f} bid/ask {ls[k]['shx']:.2f}->{v['bc_shx']:.2f} meanR flipped "
            f"{v['bc_flipR']:+.1f} | level c {v['lev_cmin']:.3f}-{v['lev_cmax']:.3f} flips {v['lev_flip']} "
            f"(to long {v['lev_tolong']}) SR {v['lev_sh']:.2f}"
        )
    rng(full, lambda k: cal[k]["level"], "rhedCalLevel", f"mean forecast / mean RV {fs}", 2)
    rng(full, lambda k: cal[k]["scale"], "rhedCalScale", f"mean of RV / forecast (QLIKE-optimal scale) {fs}", 2)
    rng(har, lambda k: cal[k]["scale"], "rhedHarCalScale", f"mean of RV / forecast {hs}", 2)
    rng(full, lambda k: cal[k]["med_fc"], "rhedMedFcIV", f"median forecast / IV^2 {fs}", 2)
    rng(full, lambda k: cal[k]["bc_cmin"], "rhedBcScaleLo", f"lowest causal scale c_t over the 614 days {fs}", 2)
    rng(full, lambda k: cal[k]["bc_cmax"], "rhedBcScaleHi", f"highest causal scale c_t over the 614 days {fs}", 2)
    rng(full, lambda k: cal[k]["bc_flip"], "rhedBcFlip", f"positions changed by the bias correction {fs}", 0)
    assert all(cal[k]["bc_flip"] == cal[k]["bc_tolong"] for k in full), "every correction flip is short -> long"
    rng(full, lambda k: cal[k]["bc_long"], "rhedBcLong", f"long days under the bias correction {fs}", 0)
    rng(full, lambda k: cal[k]["bc_flipR"], "rhedBcFlipR", f"mean mid straddle return R on the changed days, % {fs}", 0)
    rng(full, lambda k: cal[k]["bc_sh"], "rhedBcSh", f"Sharpe ratio mid under the bias correction (diagnostic) {fs}", 2)
    rng(full, lambda k: cal[k]["bc_shx"], "rhedBcShX", f"Sharpe ratio bid/ask under the bias correction {fs}", 2)
    rng(
        full,
        lambda k: cal[k]["lev_flip"],
        "rhedLevFlip",
        f"positions changed by the level correction (sum RV / sum fc) {fs}",
        0,
    )
    rng(full, lambda k: cal[k]["lev_sh"], "rhedLevSh", f"Sharpe ratio mid under the level correction {fs}", 2)
    rng(har, lambda k: cal[k]["bc_sh"], "rhedHarBcSh", f"Sharpe ratio mid under the bias correction {hs}", 2)
    mac(
        "rhedBcShUpN",
        str(sum(cal[k]["bc_sh"] > ls[k]["sh"] for k in full)),
        "full-set forecasts whose mid SR rises under the bias correction",
    )
    mac(
        "rhedBcShDownN",
        str(sum(cal[k]["bc_sh"] < ls[k]["sh"] for k in full)),
        "full-set forecasts whose mid SR falls under the bias correction",
    )
    mac(
        "rhedHarBcShUpN",
        str(sum(cal[k]["bc_sh"] > ls[k]["sh"] for k in har)),
        "HAR-benchmark forecasts whose mid SR rises under the bias correction",
    )
    r30p, ivp_ = d["r30"][W:], d["iv"][W:]
    mac(
        "rhedMoveSqIV",
        fnum(float((r30p**2 / ivp_).mean()), 2),
        "mean of the squared S&P 500 half-hour log return over IV_t^2, 614 days",
    )
    mac(
        "rhedMoveSqRV",
        fnum(float((r30p**2).mean() / rvp.mean()), 2),
        "mean squared S&P 500 half-hour log return over mean RV_t, 614 days",
    )
    mac("rhedOlsCalScale", fnum(cal[me.OLS]["scale"], 2), "mean of RV / forecast, closing-interval HAR forecast")
    mac(
        "rhedOlsBcSh", fnum(cal[me.OLS]["bc_sh"], 2), "Sharpe ratio mid under the bias correction, closing-interval HAR"
    )
    for e, _, cap in EST:
        k = key_of(e, True)
        mac(f"rhedBcFlip{cap}", str(cal[k]["bc_flip"]), f"positions changed by the bias correction, {k}")
        mac(f"rhedBcSh{cap}", fnum(cal[k]["bc_sh"], 2), f"Sharpe ratio mid under the bias correction, {k}")

    # ---------------------------------------------------------------- 2. quintiles
    qt = {k: quintiles(d, k) for k in [*full, *har, me.OLS]}
    print("\n2. QUINTILES (mean RV/IV^2 by quintile; mean R %; strategy mean %; share of return %)")
    for k, f in qt.items():
        print(
            f"  {k:30s} RV/IV {np.round(f['rr'].to_numpy()[:5], 3)} R {np.round(100 * f['R'].to_numpy()[:5], 1)} "
            f"long {np.round(100 * f['long'].to_numpy()[:5], 0)} share {np.round(100 * f['sum'].to_numpy()[:5] / f.loc['all', 'sum'], 0)}"
        )

    def mono(k: str) -> bool:
        v = qt[k]["rr"].to_numpy()[:5]
        return bool(np.all(np.diff(v) > 0))

    def r_pattern(k: str) -> bool:
        v = qt[k]["R"].to_numpy()[:5]
        return bool((v[:4] < 0).all() and v[4] > 0)

    mac(
        "rhedQMonoN",
        str(sum(mono(k) for k in full)),
        "full-set forecasts whose mean RV/IV^2 rises across all five quintiles",
    )
    mac(
        "rhedHarQMonoN",
        str(sum(mono(k) for k in har)),
        "HAR-benchmark forecasts whose mean RV/IV^2 rises across all five quintiles",
    )
    mac(
        "rhedQRPatternN",
        str(sum(r_pattern(k) for k in full)),
        "full-set forecasts with mean R < 0 in quintiles 1-4 and > 0 in 5",
    )
    mac("rhedHarQRPatternN", str(sum(r_pattern(k) for k in har)), "HAR-benchmark forecasts with that pattern")

    def qmean(keys: list[str], i, col: str, scale: float = 1.0) -> float:
        return float(np.mean([scale * qt[k].loc[i, col] for k in keys]))

    for tag, keys in (("", full), ("Har", har)):
        mac(
            f"rhed{tag}QOneRVIV",
            fnum(qmean(keys, 0, "rr"), 2),
            f"mean RV/IV^2 in the lowest quintile, mean of the six {tag or 'full-set'}",
        )
        mac(
            f"rhed{tag}QFiveRVIV",
            fnum(qmean(keys, 4, "rr"), 2),
            f"mean RV/IV^2 in the highest quintile, mean of the six {tag or 'full-set'}",
        )
        mac(
            f"rhed{tag}QFiveR",
            fnum(qmean(keys, 4, "R", 100.0), 0),
            f"mean straddle return R in the highest quintile, %, mean of the six {tag or 'full-set'}",
        )
        mac(
            f"rhed{tag}QFiveFc",
            fnum(qmean(keys, 4, "fc"), 2),
            f"geometric-mean forecast / IV^2 in the highest quintile, mean of the six {tag or 'full-set'}",
        )
        mac(f"rhed{tag}QFourFc", fnum(qmean(keys, 3, "fc"), 2), "... in the fourth quintile")
        mac(
            f"rhed{tag}QFiveLong",
            fnum(qmean(keys, 4, "long", 100.0), 0),
            f"% long days in the highest quintile, mean of the six {tag or 'full-set'}",
        )
        mac(
            f"rhed{tag}QFiveHit",
            fnum(qmean(keys, 4, "hit", 100.0), 0),
            f"hit rate in the highest quintile, %, mean of the six {tag or 'full-set'}",
        )
        mac(
            f"rhed{tag}QOneHit",
            fnum(qmean(keys, 0, "hit", 100.0), 0),
            f"hit rate in the lowest quintile, %, mean of the six {tag or 'full-set'}",
        )
        tot = sum(qt[k].loc["all", "sum"] for k in keys)
        mac(
            f"rhed{tag}QFiveShare",
            fnum(100.0 * sum(qt[k].loc[4, "sum"] for k in keys) / tot, 0),
            f"% of the summed mid return earned in the highest quintile, six {tag or 'full-set'} forecasts",
        )
        lowR = [qmean(keys, i, "R", 100.0) for i in range(4)]
        mac(
            f"rhed{tag}QLowRMin",
            fnum(min(lowR), 0),
            "most negative mean straddle return R in quintiles 1-4, %, mean of the six",
        )
        mac(
            f"rhed{tag}QLowRMax",
            fnum(max(lowR), 0),
            "least negative mean straddle return R in quintiles 1-4, %, mean of the six",
        )
    rng(
        full,
        lambda k: 100.0 * qt[k].loc[4, "sum"] / qt[k].loc["all", "sum"],
        "rhedQFiveShareRng",
        f"% of the summed mid return in quintile 5 {fs}",
        0,
    )

    # ---------------------------------------------------------------- 4. best / worst days
    best_cnt: dict[pd.Timestamp, int] = {}
    worst_cnt: dict[pd.Timestamp, int] = {}
    best_of: dict[pd.Timestamp, list[str]] = {}
    worst_of: dict[pd.Timestamp, list[str]] = {}
    contrib: dict[str, dict] = {}
    print("\n4. BEST / WORST DAYS per full-set forecast (mid return %)")
    for e, _, _ in EST:
        k, kh = key_of(e, True), key_of(e, False)
        r = d["mid"][k][W:]
        o = np.argsort(r, kind="stable")
        wi, bi = o[:N_TOP], o[-N_TOP:]
        for i in bi:
            best_cnt[pwd[i]] = best_cnt.get(pwd[i], 0) + 1
            best_of.setdefault(pwd[i], []).append(k)
        for i in wi:
            worst_cnt[pwd[i]] = worst_cnt.get(pwd[i], 0) + 1
            worst_of.setdefault(pwd[i], []).append(k)
        print(f"  {k}: best {[(short_date(pwd[i]), round(100 * r[i])) for i in bi[::-1]]}")
        print(f"  {' ' * len(k)}  worst {[(short_date(pwd[i]), round(100 * r[i])) for i in wi]}")
        drop_b = np.zeros(len(pwd), bool)
        drop_b[bi] = True
        drop_bw = drop_b.copy()
        drop_bw[wi] = True
        sh, hm = d["mid"]["short"][W:], d["mid"][kh][W:]
        gain = r - sh
        contrib[k] = {
            "vs_short_all": sr(r, pwd) - sr(sh, pwd),
            "vs_short_exb": drop_days(r, sh, pwd, drop_b),
            "vs_short_exbw": drop_days(r, sh, pwd, drop_bw),
            "best_share": 100.0 * float(gain[bi].sum() / gain.sum()),
            "vs_har_all": sr(r, pwd) - sr(hm, pwd),
            "vs_har_exb": drop_days(r, hm, pwd, drop_b),
            "vs_har_exbw": drop_days(r, hm, pwd, drop_bw),
            "har_best_share": 100.0 * float((r - hm)[bi].sum() / (r - hm).sum()),
            "har_bw_share": 100.0 * float((r - hm)[drop_bw].sum() / (r - hm).sum()),
        }
        print("   ", {a: round(b, 2) for a, b in contrib[k].items()})
    n6 = len(full)
    mac("rhedBestUnion", str(len(best_cnt)), "days in the union of the six full-set forecasts' ten best days")
    mac("rhedWorstUnion", str(len(worst_cnt)), "days in the union of the six full-set forecasts' ten worst days")
    mac("rhedBestAllSix", str(sum(v == n6 for v in best_cnt.values())), "days among the ten best of all six forecasts")
    mac(
        "rhedWorstAllSix",
        str(sum(v == n6 for v in worst_cnt.values())),
        "days among the ten worst of all six forecasts",
    )
    top_day = max(best_cnt, key=lambda dd: d["R"][W:][pwd.get_loc(dd)])
    ti = pwd.get_loc(top_day)
    mac("rhedTopDate", nice_date(top_day), "the day of the largest straddle return among the best days")
    mac("rhedTopR", fnum(100.0 * Rp[ti], 0), "... the straddle's mid return that day, % of premium")
    mac("rhedTopRVIV", fnum(rvp[ti] / ivp[ti], 2), "... RV / IV^2 that day")
    mac("rhedTopMove", fnum(100.0 * d["r30"][W:][ti], 2), "... S&P 500 15:30-16:00 log return that day, %")
    all_r = [d["mid"][k][W:] for k in full]
    mac(
        "rhedTopShareMin",
        fnum(min(100.0 * float(r[ti] / r.sum()) for r in all_r), 0),
        "lowest % of a full-set forecast's summed mid return earned on that one day",
    )
    mac(
        "rhedTopShareMax",
        fnum(max(100.0 * float(r[ti] / r.sum()) for r in all_r), 0),
        "highest % of a full-set forecast's summed mid return earned on that one day",
    )
    ev = d["ev"]

    def n_tagged(cnt: dict, nm: str) -> int:
        return sum(dd in ev[nm] for dd in cnt)

    mac("rhedBestFomcN", str(n_tagged(best_cnt, "FOMC")), "best-day union: FOMC statement days")
    mac("rhedWorstFomcN", str(n_tagged(worst_cnt, "FOMC")), "worst-day union: FOMC statement days")
    mac("rhedWorstMeN", str(n_tagged(worst_cnt, "ME")), "worst-day union: month ends")
    mac("rhedBestMeN", str(n_tagged(best_cnt, "ME")), "best-day union: month ends")
    mac("rhedBestOpexN", str(n_tagged(best_cnt, "OpEx")), "best-day union: monthly expirations")
    mac("rhedWorstOpexN", str(n_tagged(worst_cnt, "OpEx")), "worst-day union: monthly expirations")
    mac(
        "rhedBestNoTagN",
        str(sum(tags_of(d, dd).startswith("--") for dd in best_cnt)),
        "best-day union: days without any event tag (calendar as covered)",
    )
    mac(
        "rhedBestAfterRelN",
        str(sum(dd > d["rel_end"] for dd in best_cnt)),
        "best-day union: days after the release calendar ends (dagger)",
    )
    mac(
        "rhedWorstAfterRelN",
        str(sum(dd > d["rel_end"] for dd in worst_cnt)),
        "worst-day union: days after the release calendar ends",
    )
    rng(
        full,
        lambda k: contrib[k]["vs_short_all"],
        "rhedDShortAll",
        f"Delta SR mid vs the short straddle, all days {fs}",
        2,
    )
    rng(
        full,
        lambda k: contrib[k]["vs_short_exb"],
        "rhedDShortExBest",
        f"Delta SR mid vs the short straddle without the forecast's ten best days {fs}",
        2,
    )
    rng(
        full,
        lambda k: contrib[k]["vs_short_exbw"],
        "rhedDShortExBoth",
        f"... without its ten best and ten worst days {fs}",
        2,
    )
    rng(
        full,
        lambda k: contrib[k]["best_share"],
        "rhedBestShare",
        f"% of the summed gain over the short straddle earned on the ten best days {fs}",
        0,
    )
    rng(full, lambda k: contrib[k]["vs_har_all"], "rhedDHarAll", f"Delta SR mid vs the HAR twin, all days {fs}", 2)
    rng(
        full,
        lambda k: contrib[k]["vs_har_exb"],
        "rhedDHarExBest",
        f"Delta SR mid vs the HAR twin without the forecast's ten best days {fs}",
        2,
    )
    rng(
        full,
        lambda k: contrib[k]["vs_har_exbw"],
        "rhedDHarExBoth",
        f"... without its ten best and ten worst days {fs}",
        2,
    )
    rng(
        full,
        lambda k: contrib[k]["har_best_share"],
        "rhedHarBestShare",
        f"% of the summed gain over the HAR twin earned on the ten best days {fs}",
        0,
    )
    mac(
        "rhedDShortExBestNeg",
        str(sum(contrib[k]["vs_short_exb"] < 0 for k in full)),
        "full-set forecasts whose Delta SR vs short is negative without the ten best days",
    )
    mac(
        "rhedDHarExBestNeg",
        str(sum(contrib[k]["vs_har_exb"] < 0 for k in full)),
        "full-set forecasts whose Delta SR vs the HAR twin is negative without the ten best days",
    )

    # event-day statistics (the FOMC calendar is complete; the release tags cover days up to rel_end only)
    Q = np.array([d["q"][k][W:] for k in full])
    M = np.array([d["mid"][k][W:] for k in full])
    QH = np.array([d["q"][k][W:] for k in har])
    print("\n4b. EVENT DAYS (614 days; straddle R mid %, full-set long share %, full-set strategy mean %, HAR long %)")
    for nm, mname in (("FOMC", "Fomc"), ("CPI", "Cpi"), ("NFP", "Nfp"), ("OpEx", "Opex"), ("TW", "Tw"), ("ME", "Me")):
        covered = np.asarray(pwd <= d["last"][nm]) if nm in MACRO_RELEASES else np.ones(len(pwd), bool)
        on = pd.Index(pwd).isin(ev[nm]) & covered
        n = int(on.sum())
        rbar = 100.0 * float(Rp[on].mean())
        rsd = 100.0 * float(Rp[on].std(ddof=1))
        print(
            f"  {nm:5s} n {n:3d} R {rbar:+6.1f} (iid t {rbar / (rsd / math.sqrt(n)):+.2f}) long {100 * (Q[:, on] > 0).mean():5.1f} "
            f"strat {100 * M[:, on].mean():+6.1f} | other covered days: R {100 * Rp[covered & ~on].mean():+6.1f} "
            f"strat {100 * M[:, covered & ~on].mean():+6.1f} | HAR long {100 * (QH[:, on] > 0).mean():5.1f}"
        )
        mac(f"rhed{mname}N", str(n), f"{nm} days among the 614 post-warmup days")
        mac(f"rhed{mname}R", fnum(rbar, 0), f"mean mid straddle return R on {nm} days, %")
        mac(f"rhed{mname}RT", fnum(rbar / (rsd / math.sqrt(n)), 2), "... its t (i.i.d.; the days are weeks apart)")
        mac(
            f"rhed{mname}Long",
            fnum(100.0 * float((Q[:, on] > 0).mean()), 0),
            f"% of {nm} days long, mean of the six full-set forecasts",
        )
        mac(
            f"rhed{mname}Strat",
            fnum(100.0 * float(M[:, on].mean()), 0),
            f"mean mid strategy return on {nm} days, %, mean of the six full-set",
        )
        mac(
            f"rhed{mname}OtherR",
            fnum(100.0 * float(Rp[covered & ~on].mean()), 0),
            "mean straddle R on the other (covered) days, %",
        )
        mac(
            f"rhed{mname}OtherStrat",
            fnum(100.0 * float(M[:, covered & ~on].mean()), 0),
            "mean strategy return on the other (covered) days, %",
        )
        mac(f"rhed{mname}RVIV", fnum(float((rvp / ivp)[on].mean()), 2), f"mean RV_t / IV_t^2 on {nm} days")
        mac(
            f"rhed{mname}MoveSqIV",
            fnum(float((d["r30"][W:] ** 2 / ivp)[on].mean()), 2),
            f"mean squared S&P 500 half-hour log return / IV_t^2 on {nm} days",
        )
        mac(
            f"rhed{mname}Share",
            fnum(100.0 * float(np.mean([M[j, on].sum() / M[j].sum() for j in range(len(full))])), 0),
            f"% of the summed mid return earned on {nm} days, mean of the six full-set forecasts",
        )
    mac(
        "rhedRelEndText",
        nice_date(d["rel_end"]),
        "earliest last flag of the six macro-release series in data/releases.parquet",
    )
    mac("rhedRelFomcEndText", nice_date(d["last"]["FOMC"]), "last FOMC flag in data/releases.parquet")
    mac(
        "rhedFomcSuppN",
        str(int(pd.Index(pwd).isin(d["fomc_supp"]).sum())),
        "FOMC statement days of the sample taken from live/close_signal/state/fomc_statement_dates.csv",
    )
    mac("rhedRelAfterN", str(int((pwd > d["rel_end"]).sum())), "post-warmup days after the release calendar ends")

    # ---------------------------------------------------------------- 5. worthless expiries
    S, Kc, Kp, Sc = d["S"], d["Kc"], d["Kp"], d["Sc"]
    worth = Rp == -1.0
    inside = (Sc[W:] >= Kp[W:]) & (Sc[W:] <= Kc[W:])
    assert np.array_equal(worth, inside), "R_t = -1 exactly when the close settles in [K_P, K_C]"
    gap = (Kc - Kp)[W:]

    def ncdf(x: np.ndarray) -> np.ndarray:
        return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(x, float) / math.sqrt(2.0)))

    sig = np.sqrt(ivp)
    lo, hi_ = np.log(Kp[W:] / S[W:]), np.log(Kc[W:] / S[W:])
    p_iv = ncdf(hi_ / sig) - ncdf(lo / sig)
    z_all = d["r30"] / np.sqrt(d["iv"])
    p_emp = np.array(
        [
            float(((z_all[:t] >= lo[t - W] / sig[t - W]) & (z_all[:t] <= hi_[t - W] / sig[t - W])).mean())
            for t in range(W, len(days))
        ]
    )
    p_raw = np.array(
        [float(((d["r30"][:t] >= lo[t - W]) & (d["r30"][:t] <= hi_[t - W])).mean()) for t in range(W, len(days))]
    )
    zstat = float((worth.sum() - p_iv.sum()) / math.sqrt(float((p_iv * (1 - p_iv)).sum())))
    frac = np.mod(Sc[W:], 5.0)
    dist = np.minimum(frac, 5.0 - frac)
    u = np.sort(frac / 5.0)
    n = len(u)
    ks = float(max(np.max(np.arange(1, n + 1) / n - u), np.max(u - np.arange(n) / n)))
    print(
        f"\n5. WORTHLESS: observed {int(worth.sum())}; expected normal(IV) {p_iv.sum():.1f} (z {zstat:+.2f}); empirical "
        f"past z {p_emp.sum():.1f}; past raw moves {p_raw.sum():.1f}; strike gaps {pd.Series(gap).value_counts().to_dict()}; "
        f"dist to nearest strike mean {dist.mean():.3f} (uniform 1.25), KS D {ks:.4f} (5% cv {1.358 / math.sqrt(n):.4f}); "
        f"sd z {z_all[W:].std():.3f} kurt {pd.Series(z_all[W:]).kurt():.2f}; median 1-sd implied move "
        f"{np.median(S[W:] * sig):.1f} points"
    )
    mac("rhedWorthN", str(int(worth.sum())), "post-warmup days with R_t = -1 (= rheWorthlessDays)")
    mac("rhedWorthPct", fnum(100.0 * worth.mean(), 1), "... % of the 614 days")
    mac(
        "rhedWorthExpIV",
        fnum(p_iv.sum(), 0),
        "expected R = -1 days under a normal half-hour log move with volatility IV_t",
    )
    mac("rhedWorthExpIVPct", fnum(100.0 * p_iv.mean(), 1), "... as % of the 614 days")
    mac("rhedWorthZ", fnum(zstat, 2), "(observed - expected) / sd of the count under that normal (independent days)")
    mac(
        "rhedWorthExpEmp",
        fnum(p_emp.sum(), 0),
        "expected R = -1 days under the empirical distribution of the earlier days' moves standardized by IV",
    )
    mac("rhedWorthExpRaw", fnum(p_raw.sum(), 0), "expected R = -1 days under the earlier days' raw half-hour moves")
    mac("rhedGapFiveN", str(int((gap == 5.0).sum())), "post-warmup days with K_C - K_P = 5 points")
    mac("rhedGapOtherN", str(int((gap != 5.0).sum())), "post-warmup days with another strike gap")
    mac("rhedMoveSdPts", fnum(float(np.median(S[W:] * sig)), 1), "median one-sd implied half-hour move, index points")
    mac(
        "rhedStrikeDist",
        fnum(float(dist.mean()), 2),
        "mean distance of the close to the nearest 5-point strike, points",
    )
    mac("rhedStrikeDistUnif", fnum(1.25, 2), "... its mean under a uniform close")
    mac(
        "rhedKsD",
        fnum(ks, 3),
        "Kolmogorov-Smirnov D of the close's position within its 5-point strike interval against a uniform",
    )
    mac("rhedKsCV", fnum(1.358 / math.sqrt(n), 3), "... asymptotic 5% critical value")
    mac("rhedZSd", fnum(float(z_all[W:].std()), 2), "sd of the half-hour log move standardized by IV_t, 614 days")
    mac("rhedZKurt", fnum(float(pd.Series(z_all[W:]).kurt()), 1), "excess kurtosis of that standardized move")
    rng(
        full,
        lambda k: int((worth & (d["q"][k][W:] < 0)).sum()),
        "rhedWorthShort",
        f"R = -1 days on which the forecast is short {fs}",
        0,
    )
    # the classic pinning setting: monthly SPX expirations (the large open interest expires at the close)
    opx = pd.Index(pwd).isin(d["ev"]["OpEx"])
    mac("rhedWorthOpexN", str(int(worth[opx].sum())), "R = -1 days among the monthly-expiration days")
    mac("rhedWorthOpexExp", fnum(float(p_iv[opx].sum()), 1), "... expected under the normal with IV_t")
    mac("rhedWorthOtherPct", fnum(100.0 * float(worth[~opx].mean()), 1), "% of the other days with R = -1")
    mac("rhedWorthOtherExpPct", fnum(100.0 * float(p_iv[~opx].mean()), 1), "... expected under the normal with IV_t, %")
    z_opx = float((worth[opx].sum() - p_iv[opx].sum()) / math.sqrt(float((p_iv[opx] * (1 - p_iv[opx])).sum())))
    mac("rhedWorthOpexZ", fnum(z_opx, 2), "(observed - expected) / sd of the count on the monthly-expiration days")
    print(
        f"  OpEx: worthless {int(worth[opx].sum())} of {int(opx.sum())}, expected {p_iv[opx].sum():.1f}; other days "
        f"{100 * worth[~opx].mean():.1f}% vs {100 * p_iv[~opx].mean():.1f}%"
    )

    # ---------------------------------------------------------------- 6. risk regressions
    ys = {k: (100.0 * d["mid"][k][W:], 100.0 * d["ba"][k][W:]) for k in [*full, *har, me.OLS, "short"]}
    ys["avg"] = (
        np.mean([ys[k][0] for k in full], axis=0),
        np.mean([ys[k][1] for k in full], axis=0),
    )
    ys["avghar"] = (
        np.mean([ys[k][0] for k in har], axis=0),
        np.mean([ys[k][1] for k in har], axis=0),
    )
    # check: the intercept-only regression is the paper's test of a mean
    for k in full:
        f0 = hac_ols(ys[k][0], np.zeros((len(pwd), 0)))
        assert abs(float(f0["t"][0]) - hi.llsw_test(ys[k][0])["t"]) < 1e-9, k
    regs = run_regs(d, ys)
    print("\n6. RISK REGRESSIONS (alpha % per day (t, fixed-b p); betas)")
    for nm in [*full, "avg", *har, "avghar", me.OLS, "short"]:
        for col in ("mid", "ba"):
            line = f"  {nm:30s} {col:3s}"
            for sp, _, rg in REG_SPECS:
                f = regs[nm][col].get(sp)
                if f is None:
                    continue
                line += (
                    f" | {sp}: a {f['b'][0]:+5.1f} ({f['t'][0]:+.2f}, {f['p'][0]:.3f}) "
                    + " ".join(f"{r}={b:+.3f}({t:+.1f})" for r, b, t in zip(rg, f["b"][1:], f["t"][1:]))
                    + f" R2 {f['r2']:.2f}"
                )
            print(line)

    def reg_rng(spec: str, j: int, nm: str, note: str, col: str = "mid", dp: int = 1, keys: list[str] = full) -> None:
        rng(keys, lambda k: float(regs[k][col][spec]["b"][j]), nm, f"{note} {fs if keys is full else hs}", dp)

    reg_rng("a", 0, "rhedRegAlphaA", "alpha, (a) on the short straddle, mid, % per day")
    reg_rng("a", 0, "rhedRegAlphaAX", "alpha, (a) on the short straddle, bid/ask, % per day", col="ba")
    reg_rng("a", 1, "rhedRegBetaShort", "beta on the short-straddle return, (a), mid", dp=2)
    rng(full, lambda k: float(regs[k]["mid"]["a"]["p"][0]), "rhedRegAlphaAP", f"fixed-b p of alpha, (a), mid {fs}", 3)
    rng(
        full,
        lambda k: float(regs[k]["ba"]["a"]["p"][0]),
        "rhedRegAlphaAXP",
        f"fixed-b p of alpha, (a), bid/ask {fs}",
        3,
    )
    rng(
        full,
        lambda k: float(regs[k]["mid"]["a"]["t"][1]),
        "rhedRegBetaShortT",
        f"t of the beta on the short straddle, (a), mid {fs}",
        2,
    )
    rng(full, lambda k: float(regs[k]["mid"]["a"]["r2"]), "rhedRegRsqA", f"R^2 of (a), mid {fs}", 2)
    reg_rng("b", 2, "rhedRegBetaAbs", "beta on |r30| (bp), (b), mid, % of premium per bp", dp=2)
    rng(
        full,
        lambda k: float(regs[k]["mid"]["b"]["t"][2]),
        "rhedRegBetaAbsT",
        f"t of the beta on |r30|, (b), mid {fs}",
        2,
    )
    reg_rng("b", 1, "rhedRegBetaMove", "beta on r30 (bp), (b), mid", dp=2)
    reg_rng("m", 0, "rhedRegMean", "mean return, mid, % per day")
    reg_rng("m", 0, "rhedRegMeanX", "mean return, bid/ask, % per day", col="ba")
    rng(full, lambda k: float(regs[k]["mid"]["c"]["r2"]), "rhedRegRsqC", f"R^2 of (c), mid {fs}", 2)
    rng(
        full,
        lambda k: float(np.max(regs[k]["mid"]["c"]["p"][1:])),
        "rhedRegCPMax",
        f"larger fixed-b p of the two betas of (c), mid {fs}",
        2,
    )
    rng(
        full,
        lambda k: float(np.min(regs[k]["mid"]["c"]["p"][1:])),
        "rhedRegCPMin",
        f"smaller fixed-b p of the two betas of (c), mid {fs}",
        2,
    )
    reg_rng("d", 0, "rhedRegAlphaD", "alpha, (d) joint, mid, % per day")
    reg_rng("d", 0, "rhedRegAlphaDX", "alpha, (d) joint, bid/ask, % per day", col="ba")
    rng(full, lambda k: float(regs[k]["mid"]["d"]["p"][0]), "rhedRegAlphaDP", f"fixed-b p of alpha, (d), mid {fs}", 3)
    rng(
        full,
        lambda k: float(regs[k]["ba"]["d"]["p"][0]),
        "rhedRegAlphaDXP",
        f"fixed-b p of alpha, (d), bid/ask {fs}",
        3,
    )
    for col, tag in (("mid", ""), ("ba", "X")):
        for sp in ("m", "a", "d"):
            f = regs["avg"][col][sp]
            mac(
                f"rhedRegAvgAlpha{sp.upper()}{tag}",
                fnum(float(f["b"][0]), 1),
                f"alpha of the mean of the six full-set returns, ({sp}), {col}, % per day",
            )
            mac(f"rhedRegAvgAlpha{sp.upper()}{tag}T", fnum(float(f["t"][0]), 2), "... its t")
            mac(f"rhedRegAvgAlpha{sp.upper()}{tag}P", pdig(float(f["p"][0])), "... its fixed-b p")
    fa = regs["avg"]["mid"]
    mac(
        "rhedRegAvgBetaShort",
        fnum(float(fa["a"]["b"][1]), 2),
        "beta of the mean full-set return on the short straddle, (a), mid",
    )
    mac("rhedRegAvgBetaShortT", fnum(float(fa["a"]["t"][1]), 2), "... its t")
    mac("rhedRegAvgBetaShortP", pdig(float(fa["a"]["p"][1])), "... its fixed-b p")
    mac("rhedRegAvgRsqA", fnum(float(fa["a"]["r2"]), 2), "R^2 of (a) for the mean full-set return")
    mac(
        "rhedRegAvgBetaAbs", fnum(float(fa["b"]["b"][2]), 2), "beta of the mean full-set return on |r30| (bp), (b), mid"
    )
    mac("rhedRegAvgBetaAbsT", fnum(float(fa["b"]["t"][2]), 2), "... its t")
    fs_ = regs["short"]["mid"]
    mac("rhedRegShortBetaAbs", fnum(float(fs_["b"]["b"][2]), 2), "beta of the short straddle on |r30| (bp), (b), mid")
    mac("rhedRegShortRsqB", fnum(float(fs_["b"]["r2"]), 2), "R^2 of (b) for the short straddle")
    mac(
        "rhedRegAbsRatio",
        fnum(100.0 * float(fa["b"]["b"][2]) / float(fs_["b"]["b"][2]), 0),
        "beta on |r30| of the mean full-set return as % of the short straddle's",
    )
    mac(
        "rhedRegAvgBetaAbsJ",
        fnum(float(fa["d"]["b"][3]), 2),
        "beta of the mean full-set return on |r30|, (d) joint, mid",
    )
    mac("rhedRegAvgBetaAbsJT", fnum(float(fa["d"]["t"][3]), 2), "... its t")
    rng(har, lambda k: float(regs[k]["mid"]["a"]["b"][0]), "rhedHarRegAlphaA", f"alpha, (a), mid, % per day {hs}", 1)
    rng(har, lambda k: float(regs[k]["mid"]["a"]["p"][0]), "rhedHarRegAlphaAP", f"fixed-b p of alpha, (a), mid {hs}", 3)
    mac(
        "rhedOlsRegAlphaA",
        fnum(float(regs[me.OLS]["mid"]["a"]["b"][0]), 1),
        "alpha, (a), mid, closing-interval HAR forecast, % per day",
    )
    mac("rhedOlsRegAlphaAP", pdig(float(regs[me.OLS]["mid"]["a"]["p"][0])), "... its fixed-b p")
    # alpha of (a) without each forecast's ten best days (the heavy right tail)
    Xs = regressors(d)
    exb: dict[str, dict] = {}
    for e, _, _ in EST:
        k = key_of(e, True)
        keep = np.ones(len(pwd), bool)
        keep[np.argsort(d["mid"][k][W:], kind="stable")[-N_TOP:]] = False
        exb[k] = hac_ols(ys[k][0][keep], Xs["short"][keep])
        print(
            f"  (a) without the ten best days {k}: alpha {exb[k]['b'][0]:+.1f} (t {exb[k]['t'][0]:+.2f}, p {exb[k]['p'][0]:.3f})"
        )
    rng(
        full,
        lambda k: float(exb[k]["b"][0]),
        "rhedRegAlphaAExBest",
        f"alpha, (a), mid, without the forecast's ten best days, % per day {fs}",
        1,
    )
    rng(full, lambda k: float(exb[k]["p"][0]), "rhedRegAlphaAExBestP", f"... its fixed-b p {fs}", 2)

    # ---------------------------------------------------------------- outputs
    existing: set[str] = set()
    for f in PAPER_GEN.glob("*.tex"):
        if f.name == "rehaul_econ_decomp_numbers.tex":
            continue
        existing |= set(
            re.findall(
                r"\\(?:provide|new|renew)command\{\\([A-Za-z]+)\}", f.read_text(encoding="utf-8", errors="ignore")
            )
        )
    clash = sorted(set(MACROS) & existing)
    assert not clash, f"macro names already defined in writeup/generated: {clash}"
    nums = HEADER + "% \\rhed... macros: where the straddle strategy's profits come from (Section 7 decomposition).\n"
    nums += "".join(
        f"\\providecommand{{\\{nm}}}{{}}\\renewcommand{{\\{nm}}}{{{v}}}  % {note}\n" for nm, (v, note) in MACROS.items()
    )
    bw = {"best_cnt": best_cnt, "worst_cnt": worst_cnt, "best_of": best_of, "worst_of": worst_of}
    write(GEN / "rehaul_econ_decomp_numbers.tex", nums)
    write(GEN / "rehaul_econ_decomp_longshort.tex", table_longshort(d, ls, cal))
    write(GEN / "rehaul_econ_decomp_quintiles.tex", table_quintiles(qt))
    write(GEN / "rehaul_econ_decomp_bestdays.tex", table_bestdays(d, bw, contrib))
    write(GEN / "rehaul_econ_decomp_riskreg.tex", table_riskreg(regs))
    print(f"{len(MACROS)} macros")


if __name__ == "__main__":
    main()
