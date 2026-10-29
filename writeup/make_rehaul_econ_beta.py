"""Market exposure of the 0DTE straddle strategy and its optimal allocation with the market (appendix, 2026-10-26).

PRE-REGISTRATION
  The specification below mirrors results/econ_beta_2026-10-26/RULE.txt, created 2026-10-26T02:24:25Z (2026-10-25
  19:24 PDT; its header says "02:30Z", approximate) at HEAD e235754, before any statistic on strategy-vs-market data was
  computed (the first such run started 02:31Z).  RULE.txt is not changed after
  results are seen; a deviation goes into a dated ADDENDUM at its end, with the reason.

1. SAMPLE
  The 614 post-warmup option days of Section 7 (me.pw_days(): option days t > 252, 2021-08-11 .. 2024-04-30).

2. STRATEGIES (14 series; nothing of the strategy is re-implemented)
  The six pooled full-set forecasts (ridge, lasso, elastic net, LightGBM, XGBoost, random forest), their six HAR-set
  twins (the three linear ones coincide: four distinct), the closing-interval OLS HAR forecast (CI-HAR, master-table
  key sub_ols_baseline) and the unconditional short straddle, built exactly as writeup/make_rehaul_econ_main.py builds
  them (me.load, me.gate_against_master, me.add_pooled_linear, me.add_pooled_trees, em.add_har_trees).  Daily return
  r^s_t = q_t R_t at mid-quotes and master_table_gui.rows._crossed_return at quoted bid/ask prices, as fractions of the
  premium, treated as excess returns (Section 7 omits the risk-free return over the 30-minute holding period).
  GATE: every strategy's Sharpe ratio (the paper's: D6 until 2026-10-26, pooled since D19), mid and bid/ask, reproduces \\rhemSh<tag> / \\rhemShX<tag>
  (writeup/generated/rehaul_econ_main_numbers.tex).

3. MARKET
  Primary: r^m_t = C_t / C_{t-1} - 1 - rf_t, the S&P 500 (^GSPC) close-to-close PRICE return of option day t (C_{t-1}
  the previous NYSE trading day's close), from the cached closes results/atm_straddle_0dte_1530/cache/
  gspc_close_*.parquet (asserted equal to the quote book's settlement close S^close_t on all 866 option days, as
  make_rehaul_econ_decomp.py does).  No dividends: the repository has no SPY or total-return series
  (data/spy_and_sentiment.parquet carries stocktwits columns only; data/spxw_spot.parquet's 16:00 print misses 124 of
  858 trading days in 2021-01 .. 2024-05 and is not the official close).
  Risk-free: FRED DTB3 (3-month T-bill, discount basis, % a year), data/fred_dtb3.csv, downloaded 2026-10-26T02:21Z
  from https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3 (sha256 asserted); gaps (bond-market holidays) carried
  forward; rf_t = DTB3_{t-1} / 100 * d_t / 360, d_t the calendar days from the previous trading day to t.
  Secondary window: r^30_t = S^close_t / S_t - 1, the index's 15:30-16:00 return (no risk-free deduction); exposure and
  CAPM alpha only.  Descriptive: the market's standard Sharpe ratio on all trading days of the span.

4. EXPOSURE (each strategy; mid and bid/ask; each market window; returns in %)
  Correlation; OLS beta on [1, r^m] with the HAC t (Bartlett, bandwidth S = ceil(1.3 T^(1/2)) = 33 at T = 614) and
  two-sided fixed-b p (make_rehaul_econ_decomp.hac_ols, the regression form of the paper's nw_t / hac_p);
  Treynor-Mazuy r^s = a + b r^m + gamma (r^m)^2 + e (same t / p), also on the strategy's long days and short days
  separately (t only: the expectation gamma > 0 on long days, < 0 on short days is stated, not tested); mean return on
  market-down days (r^m < 0) and on the worst-decile market days (r^m <= its 10 % sample quantile) against the other
  days, the difference's t / p from the slope on [1, indicator].
  Decision: "exposed to market beta" iff the beta's fixed-b p < 0.05; counts over the six full-set forecasts.

5. ALPHA AND SPANNING (primary window; mid and bid/ask)
  CAPM alpha (HAC t, fixed-b p; % per day, and x 252 annualized); sigma_eps^2 = var(r^s) - beta^2 var(r^m) (divisor
  T - 1); appraisal ratio AR = alpha / sigma_eps x sqrt(252); SR_m, SR_s = mean / sd x sqrt(252) (the STANDARD ratio);
  SR_max = sqrt(SR_m^2 + AR^2), the in-sample tangency (short sales allowed), asserted equal to the Sharpe ratio of
  w ~ Sigma^-1 mu; Delta = SR_max - |SR_m|.  One-asset GRS test of alpha = 0 (W = T alpha^2 / s_eps^2 /
  (1 + mean(r^m)^2 / var_MLE(r^m)), s_eps^2 = SSR / (T - 2), F(1, T - 2); W = the squared i.i.d. OLS t, asserted).
  Bootstrap of Delta: hac_inference._boot_index (block 10, 4,999 resamples, seed D6_SEED) on the pairs
  (r^m_t, r^s_t), Delta* re-estimated in each resample; 95 % percentile interval; p = share of |Delta* - Delta| >=
  |Delta| (hi.d6_diff_test's convention; Delta >= 0 by construction, so this p is descriptive and the GRS and HAC tests
  of alpha are the tests of record).  Tangency risk terms: each leg's Euler share of the tangency portfolio's
  volatility, RC_i = w_i (Sigma w)_i / (w' Sigma w).  D6 of the market and of the fixed-weight tangency: descriptive.

6. REAL-TIME ALLOCATION (mid and bid/ask)
  Burn-in: the first 126 post-warmup days; evaluation: days 127 .. 614 (488), the same for the market alone and both
  rules.  Dollar weights w_m + w_s = 1, w >= 0, set from days strictly before t; r^p_t = w_m,t r^m_t + w_s,t r^s_t;
  volatility weights v_i = w_i sigma_i, theta = v_s / (v_m + v_s).  Mean-variance: expanding window; theta_t = the
  point of {0, 0.001, ..., 1} maximizing the estimated Sharpe ratio among those at which neither leg's Euler share of
  the estimated portfolio volatility exceeds 90 % (ties: the smallest theta).  Inverse volatility: w_i ~ 1 / sigma_i
  from the trailing 63 post-warmup days.  Standard and D6 Sharpe ratios against the market alone on the same days;
  D6 difference by hi.d6_diff_test, standard difference by the same paired resample indices
  (hi._boot_index(488, 10, 4999, D6_SEED)), the weights held at their real-time values.
  Decision: a combination "beats the market" iff its Sharpe ratio difference is positive with bootstrap p < 0.05.

7. ROBUSTNESS
  (a) bid/ask: 4-6 on the bid/ask returns (estimation and evaluation).  (b) Without the union of the six full-set
  forecasts' ten best days by mid return (make_rehaul_econ_decomp.py statistic 4; gated against \\rhedBestUnion and
  \\rhedTopDate), removed from every series, 4-6 recomputed (real-time rules rerun, burn-in 126; D6 with the full span
  fixed); a stress test biased against the strategies.  (c) 4 and the CAPM alpha against r^30.

8. REPORTING
  Every Sharpe-ratio macro comment says "standard" (sqrt(252), pooled daily moments) or "D6".
  2026-10-26 (author decision D19): the paper's Sharpe ratio is now the standard pooled ratio annualized by
  sqrt(q), q = T / sum_y c_y of the sample's own days (hac_inference.paper_q: 225.5 on the 614 post-warmup days); every
  "STANDARD" ratio here (SR_m, SR_s, AR, SR_max, the real-time ratios) uses that q in place of 252 (a common scale
  factor sqrt(q / 252) within a sample: no p-value, GRS statistic, real-time weight or decision moves), alpha is
  annualized by q, and the strategy's STANDARD ratio is gated equal to the paper's \\rhemSh / \\rhemShX. The
  real-time combination's Sharpe-ratio difference against the market is tested by Ledoit and Wolf (2008)
  (rhebRt<Rule>P..., the paper's test of a Sharpe-ratio difference); the paired block bootstrap of the difference
  (RULE.txt section 6) is kept as a check (stats.csv, stats mv_pb / iv_pb, and the console). D6
  macros stay (robustness).  The table shows the six
  full-set forecasts, CI-HAR and the short straddle; the HAR-set twins are in the macros.  Figure: cumulative excess
  return over the 488 evaluation days of the market alone and of the real-time combinations with the full-set
  LightGBM strategy at mid, each scaled ex post to the market's standard deviation over those days (display only).

REUSE
  writeup/make_rehaul_economics.py (me): load, gate_against_master, add_pooled_linear, add_pooled_trees, sharpe /
  sharpe_days (D6), pw_days, WARMUP, the table cell helpers and the figure rcParams.  writeup/make_rehaul_econ_main.py
  (em): key, tag, add_har_trees, _macros / _norm / f2 (the gate), num / pval / write.  writeup/
  make_rehaul_econ_decomp.py (dc): hac_ols (HAC OLS with fixed-b p), N_TOP, read_macros, nice_date, HEAD_TEX.
  writeup/hac_inference.py (hi): d6_sharpe, d6_diff_test, _boot_index, LW_BLOCK, LW_BOOT, D6_SEED.

AFTER THE RESULTS (RULE.txt ADDENDUM 2026-10-26; descriptive only, no registered statistic changed)
  The real-time mean-variance rule's median strategy share of volatility, the share of days at the cap and the median
  dollar weights of both rules; the CAPM R^2; the ranges of the bootstrap intervals.  DRAFT CLAIMS (draft_claims) asserts
  the words of writeup/drafts/2026-10-26_econ_beta.tex that no macro carries; check_macros_used asserts that every
  \\rheb name the table, the figure float and the draft use is defined.

OUTPUTS (new files)
  writeup/generated/rehaul_econ_beta_numbers.tex   \\rheb... macros (macro-only)
  writeup/generated/rehaul_econ_beta_table.tex     Table tab:rh_econ_beta
  writeup/generated/rehaul_econ_beta_cum.pdf       the cumulative excess returns (figure)
  writeup/generated/rehaul_econ_beta_fig.tex       its float, fig:rh_econ_beta_cum
  results/econ_beta_2026-10-26/stats.csv           every statistic, one row per strategy x price x sample

Run (Git Bash, repository root):  /c/Users/james/miniconda3/python.exe writeup/make_rehaul_econ_beta.py [--preview DIR]
REHAUL_FINAL_RUN is set to 1 here (the paper's pooled forecasts are the final rerun).
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
import sys
from pathlib import Path

os.environ["REHAUL_FINAL_RUN"] = "1"  # before final_run is imported (through make_rehaul_econ_main)

WRITEUP = Path(__file__).resolve().parent
ROOT = WRITEUP.parent
if str(WRITEUP) not in sys.path:
    sys.path.insert(0, str(WRITEUP))

import make_rehaul_econ_main as em  # noqa: E402  (sets the switch; key, tag, add_har_trees, the gate helpers)
import final_run as fr  # noqa: E402
import hac_inference as hi  # noqa: E402
import make_rehaul_econ_decomp as dc  # noqa: E402  (hac_ols, the best-day rule)
import make_rehaul_economics as me  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from master_table_gui import rows as gui  # noqa: E402

GEN = fr.GEN
PAPER_GEN = WRITEUP / "generated"
OUT = ROOT / "results" / "econ_beta_2026-10-26"
HEADER = "% AUTO-GENERATED by writeup/make_rehaul_econ_beta.py -- do not edit; re-run the script.\n"
NUMBERS = GEN / "rehaul_econ_beta_numbers.tex"
TABLE = GEN / "rehaul_econ_beta_table.tex"
FIG_PDF = GEN / "rehaul_econ_beta_cum.pdf"
FIG_TEX = GEN / "rehaul_econ_beta_fig.tex"
PREVIEW: Path | None = None

DTB3 = ROOT / "data" / "fred_dtb3.csv"
DTB3_SHA256 = "3437358abeac96f2417ec10d173e985b8aa831d09dee78debdafa52f016f21c8"  # RULE.txt, section 3
ANN = 252  # 2026-10-26 (D19): no longer used (the STANDARD ratio is annualized by sqrt(q), q_of)
BURN, TRAIL, CAP = 126, 63, 0.90  # RULE.txt, section 6
GRID = np.linspace(0.0, 1.0, 1001)
WORST_Q = 0.10

EST, SETS, LINEAR, TREES = em.EST, em.SETS, em.LINEAR, em.TREES
CI, SHORT = em.CI, em.SHORT
FULL = [em.key(e, "all_features") for e in EST]
HAR = [em.key(e, "baseline") for e in EST]
HAR4 = [em.key("ridge", "baseline")] + [em.key(e, "baseline") for e in TREES]
ROWS = FULL + HAR + [CI, SHORT]
# 2026-10-26 (integration plan D2/D3): the CI-HAR column leaves the table (closing-interval models: final appendix
# section); its statistics stay in the macros. Was: TAB_ROWS = FULL + [CI, SHORT]
TAB_ROWS = FULL + [SHORT]
PRICES = (("mid", ""), ("x", "X"))
WORDS = em.WORDS


# ============================================================================================ small helpers
# 2026-10-26 (author decision D19): parked -- the sqrt(252) annualization of the STANDARD ratio. Was:
# def sr252(x) -> float:
#     """The standard annualized Sharpe ratio: sqrt(252) mean / sd (divisor n - 1) of the pooled daily returns."""
#     x = np.asarray(x, float)
#     return float(x.mean() / x.std(ddof=1) * math.sqrt(ANN))
# def sr252_rows(a: np.ndarray) -> np.ndarray:
#     return a.mean(axis=1) / a.std(axis=1, ddof=1) * math.sqrt(ANN)
def q_of(dates, span=None) -> float:
    """D19: the annualization of the paper's Sharpe ratio on these days: q = T / sum_y c_y (hac_inference.paper_q)."""
    return float(hi.paper_q(pd.DatetimeIndex(dates), span=span)["q"])


def srq(x, q: float) -> float:
    """The STANDARD (= the paper's, D19) Sharpe ratio: sqrt(q) mean / sd (divisor n - 1) of the pooled daily returns."""
    x = np.asarray(x, float)
    return float(x.mean() / x.std(ddof=1) * math.sqrt(q))


def srq_rows(a: np.ndarray, q: float) -> np.ndarray:
    return a.mean(axis=1) / a.std(axis=1, ddof=1) * math.sqrt(q)


def d6(x, dates, span) -> float:
    """The paper's D6 Sharpe ratio (hac_inference.d6_sharpe) on the days given, with the span given."""
    return float(hi.d6_sharpe(np.asarray(x, float), dates=dates, span=span)["sharpe"])


def hac_ols_t(y: np.ndarray, X: np.ndarray) -> dict:
    """dc.hac_ols without its fixed-b p: for the long-day / short-day subsamples (non-contiguous days, a b of their
    own), whose t is reported without a test (RULE.txt, section 4)."""
    saved = hi.fixedb_p
    hi.fixedb_p = lambda t, b: float("nan")  # type: ignore[assignment]
    try:
        return dc.hac_ols(y, X)
    finally:
        hi.fixedb_p = saved  # type: ignore[assignment]


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction of the incomplete beta function (modified Lentz; Press et al., Numerical Recipes 6.4)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 1000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-15:
            return h
    raise RuntimeError("betacf did not converge")


def betainc_reg(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b) (scipy.stats is broken in this environment: numpy ABI)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbt) * _betacf(b, a, 1.0 - x) / b


def f_sf(w: float, d1: int, d2: int) -> float:
    """P(F(d1, d2) > w) = I_{d2 / (d2 + d1 w)}(d2 / 2, d1 / 2)."""
    return betainc_reg(d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * w))


def _check_f_sf() -> None:
    # F(1, nu) at w = t^2 is the two-sided t p-value: against a numerical integral of the t density, and the normal
    # limit (5 % at 1.959964^2 for nu -> infinity)
    nu, t = 612, 2.0
    z = np.linspace(t, 60.0, 600_001)
    dens = np.exp(
        math.lgamma((nu + 1) / 2)
        - math.lgamma(nu / 2)
        - 0.5 * math.log(nu * math.pi)
        - (nu + 1) / 2 * np.log1p(z**2 / nu)
    )
    tail = float(np.sum((dens[1:] + dens[:-1]) / 2 * np.diff(z)))
    assert abs(f_sf(t * t, 1, nu) - 2 * tail) < 1e-9, (f_sf(t * t, 1, nu), 2 * tail)
    assert abs(f_sf(1.959963984540054**2, 1, 10**9) - 0.05) < 1e-6


# ============================================================================================ data
def load_strategies() -> tuple[dict, pd.DatetimeIndex, pd.DatetimeIndex]:
    """The 14 strategies on the 866 option days, built as make_rehaul_econ_main.main builds them."""
    assert fr.FINAL and not fr.PENDING_TREES and fr.pool_n() == fr.POOL_N_ALL == 12, fr.tag()
    print(fr.tag())
    mt, _model_set, series, fam = me.load()
    me.gate_against_master(mt, series, [SHORT, CI])
    me.add_pooled_linear(series, fam)
    me.add_pooled_trees(series, fam)
    em.add_har_trees(series)
    days = series[CI].index
    for k in ROWS:
        assert series[k].index.equals(days), k
    return series, days, me.pw_days()


def gate_strategies(series: dict) -> None:
    """GATE (RULE.txt, section 2): each strategy's Sharpe ratio (the paper's; D19: pooled), mid and bid/ask, as printed in
    rehaul_econ_main_numbers.tex."""
    mm = em._macros(PAPER_GEN / "rehaul_econ_main_numbers.tex")
    bad = []
    for k in ROWS:
        t = em.tag(k)
        for col, nm in (("mid", f"rhemSh{t}"), ("x", f"rhemShX{t}")):
            got = em.f2(me.sharpe(series[k][col].to_numpy(float)))
            if got != em._norm(mm[nm]):
                bad.append((nm, got, mm[nm]))
    assert not bad, f"GATE FAILED: {bad}"
    print(
        f"  gate: Sharpe ratios (D19 pooled) of the {len(ROWS)} strategies (mid, bid/ask) = \\rhemSh<tag> / \\rhemShX<tag>"
    )


def load_market(days: pd.DatetimeIndex, pw: pd.DatetimeIndex) -> dict:
    """The S&P 500 close-to-close excess return on the post-warmup option days, the 15:30-16:00 index return, and the
    excess return on every trading day of the span (RULE.txt, section 3)."""
    gcf = sorted((ROOT / "results" / "atm_straddle_0dte_1530" / "cache").glob("gspc_close_*.parquet"))
    assert len(gcf) == 1, gcf
    gc = pq.read_table(gcf[0]).to_pandas()["close"].astype(float)
    gc.index = pd.to_datetime(gc.index).normalize().astype("datetime64[ns]")
    assert gc.index.is_monotonic_increasing and gc.index.is_unique and np.isfinite(gc.to_numpy()).all()
    qb = pd.read_parquet(gui._QUOTE_BOOK, columns=["S", "S_close"])
    qb.index = pd.to_datetime(qb.index).normalize().astype("datetime64[ns]")
    qb = qb.reindex(days)
    assert np.array_equal(gc.reindex(days).to_numpy(float), qb["S_close"].to_numpy(float)), "S^close_t = ^GSPC close"
    # a trading day of the cache between two option days would be a trading day; holidays are absent (checked: the
    # 32 weekday NYSE holidays of 2021-01 .. 2024-05 are exactly the weekdays the cache lacks)
    raw = DTB3.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == DTB3_SHA256, "data/fred_dtb3.csv is not the registered download"
    fred = pd.read_csv(DTB3, na_values=["."])
    fred["observation_date"] = pd.to_datetime(fred["observation_date"]).astype("datetime64[ns]")
    rate = fred.set_index("observation_date")["DTB3"].astype(float)
    tdays = gc.index
    rate_td = rate.reindex(rate.index.union(tdays)).ffill().reindex(tdays)
    prev = pd.Series(tdays[:-1], index=tdays[1:])
    ret = gc / gc.shift(1) - 1.0
    gap = pd.Series((tdays[1:] - tdays[:-1]).days, index=tdays[1:]).astype(float)
    rf = rate_td.shift(1) / 100.0 * gap.reindex(tdays) / 360.0  # DTB3 of the previous trading day, actual/360
    assert np.isfinite(rf.reindex(pw).to_numpy()).all() and np.isfinite(ret.reindex(pw).to_numpy()).all()
    assert (prev.reindex(pw) < pw).all()
    span_td = tdays[(tdays >= pw[0]) & (tdays <= pw[-1])]
    qp = qb.reindex(pw)
    return {
        "rm": (ret - rf).reindex(pw).to_numpy(float),
        "rm_raw": ret.reindex(pw).to_numpy(float),
        "rf": rf.reindex(pw).to_numpy(float),
        "r30": (qp["S_close"] / qp["S"] - 1.0).to_numpy(float),
        "rm_all": (ret - rf).reindex(span_td).to_numpy(float),
        "q_all": q_of(span_td),  # 2026-10-26 (D19): q of every trading day of the span (about 252)
        "n_all": len(span_td),
        "rate_mean": float(rate_td.reindex(span_td).mean()),
    }


def best_union(series: dict, pw: pd.DatetimeIndex) -> tuple[np.ndarray, pd.Timestamp]:
    """The union of the six full-set forecasts' ten best days by mid return (make_rehaul_econ_decomp.py statistic 4),
    gated against \\rhedBestUnion and \\rhedTopDate."""
    W = me.WARMUP
    drop = np.zeros(len(pw), bool)
    for k in FULL:
        r = series[k]["mid"].to_numpy(float)[W:]
        drop[np.argsort(r, kind="stable")[-dc.N_TOP :]] = True
    R = series[CI]["R"].to_numpy(float)[W:]
    top = pw[np.flatnonzero(drop)[np.argmax(R[drop])]]
    dm = dc.read_macros(PAPER_GEN / "rehaul_econ_decomp_numbers.tex")
    assert str(int(drop.sum())) == dm["rhedBestUnion"], (int(drop.sum()), dm["rhedBestUnion"])
    assert dc.nice_date(top) == dm["rhedTopDate"], (dc.nice_date(top), dm["rhedTopDate"])
    print(
        f"  gate: best-day union {int(drop.sum())} days = \\rhedBestUnion; top day {dc.nice_date(top)} = \\rhedTopDate"
    )
    return drop, top


# ============================================================================================ statistics
def exposure(rs: np.ndarray, rm: np.ndarray, q: np.ndarray | None, full: bool = True) -> dict:
    """Section 4 (returns in %): correlation, CAPM beta and alpha, Treynor-Mazuy, down-day and worst-decile means."""
    y, x = 100.0 * rs, 100.0 * rm
    c = dc.hac_ols(y, x)
    out = {
        "T": len(y),
        "corr": float(np.corrcoef(y, x)[0, 1]),
        "alpha": float(c["b"][0]),
        "alpha_t": float(c["t"][0]),
        "alpha_p": float(c["p"][0]),
        "beta": float(c["b"][1]),
        "beta_t": float(c["t"][1]),
        "beta_p": float(c["p"][1]),
        "r2": float(c["r2"]),
    }
    if not full:
        return out
    assert q is not None
    tm = dc.hac_ols(y, np.column_stack([x, x * x]))
    out.update(
        tm_b=float(tm["b"][1]),
        tm_bt=float(tm["t"][1]),
        tm_g=float(tm["b"][2]),
        tm_gt=float(tm["t"][2]),
        tm_gp=float(tm["p"][2]),
    )
    for side in ("long", "short"):
        m = np.asarray(q > 0 if side == "long" else q < 0, dtype=bool)
        out[f"n_{side}"] = int(m.sum())
        if m.sum() >= 30:
            r = hac_ols_t(y[m], np.column_stack([x[m], x[m] ** 2]))
            out[f"tm_{side}_g"], out[f"tm_{side}_gt"] = float(r["b"][2]), float(r["t"][2])
            out[f"tm_{side}_b"], out[f"tm_{side}_bt"] = float(r["b"][1]), float(r["t"][1])
        else:
            for s_ in ("g", "gt", "b", "bt"):
                out[f"tm_{side}_{s_}"] = float("nan")
    down = x < 0
    cut = float(np.quantile(x, WORST_Q))
    worst = x <= cut
    for nm, msk in (("down", down), ("worst", worst)):
        r = dc.hac_ols(y, msk.astype(float))
        out[f"{nm}_n"] = int(msk.sum())
        out[f"{nm}_mean"] = float(y[msk].mean())
        out[f"{nm}_rest"] = float(y[~msk].mean())
        out[f"{nm}_diff"] = float(r["b"][1])
        out[f"{nm}_t"] = float(r["t"][1])
        out[f"{nm}_p"] = float(r["p"][1])
        assert abs(out[f"{nm}_diff"] - (out[f"{nm}_mean"] - out[f"{nm}_rest"])) < 1e-9
    out["worst_cut"] = cut
    return out


def spanning(rs: np.ndarray, rm: np.ndarray, dates: pd.DatetimeIndex, span: tuple) -> dict:
    """Section 5: CAPM alpha, appraisal ratio, maximum Sharpe ratio (standard), GRS, bootstrap, tangency risk shares."""
    T = len(rs)
    X = np.vstack([rm, rs])
    mu = X.mean(axis=1)
    S = np.cov(X, ddof=1)
    beta = S[0, 1] / S[0, 0]
    alpha = mu[1] - beta * mu[0]
    s2e = S[1, 1] - beta**2 * S[0, 0]
    # 2026-10-26 (D19): sqrt(q) of these days (span fixed) in place of sqrt(ANN = 252); alpha x q in place of x 252
    q = q_of(dates, span)
    sr_m = mu[0] / math.sqrt(S[0, 0]) * math.sqrt(q)
    sr_s = mu[1] / math.sqrt(S[1, 1]) * math.sqrt(q)
    ar = alpha / math.sqrt(s2e) * math.sqrt(q)
    sr_max = math.hypot(sr_m, ar)
    w = np.linalg.solve(S, mu)
    rp = w[0] * rm + w[1] * rs
    assert abs(srq(rp, q) - sr_max) < 1e-9, (srq(rp, q), sr_max)
    assert abs(sr_s - hi.paper_sharpe(rs, dates=dates, span=span)["sharpe"]) < 1e-9, sr_s
    rc = w * (S @ w) / float(w @ S @ w)
    assert abs(rc.sum() - 1.0) < 1e-12 and np.allclose(rc, w * mu / float(w @ mu), rtol=0, atol=1e-10)
    # one-asset GRS = the squared i.i.d. OLS t of alpha
    e = rs - alpha - beta * rm
    s2 = float(e @ e) / (T - 2)
    W = T * alpha**2 / s2 / (1.0 + mu[0] ** 2 / rm.var(ddof=0))
    se_iid = math.sqrt(s2 * (1.0 / T + mu[0] ** 2 / float((rm - mu[0]) @ (rm - mu[0]))))
    assert abs(W - (alpha / se_iid) ** 2) < 1e-8 * max(1.0, W), (W, (alpha / se_iid) ** 2)
    p_grs = f_sf(W, 1, T - 2)
    # paired circular block bootstrap of Delta = SR_max - |SR_m|, re-estimated in every resample
    idx = hi._boot_index(T, hi.LW_BLOCK, hi.LW_BOOT, hi.D6_SEED)
    m_, s_ = rm[idx], rs[idx]
    mm, ms = m_.mean(axis=1), s_.mean(axis=1)
    vm, vs = m_.var(axis=1, ddof=1), s_.var(axis=1, ddof=1)
    cms = ((m_ - mm[:, None]) * (s_ - ms[:, None])).sum(axis=1) / (T - 1)
    b_ = cms / vm
    a_ = ms - b_ * mm
    srm_ = mm / np.sqrt(vm) * math.sqrt(q)
    ar_ = a_ / np.sqrt(vs - b_**2 * vm) * math.sqrt(q)
    dstar = np.hypot(srm_, ar_) - np.abs(srm_)
    delta = sr_max - abs(sr_m)
    lo, hi_ = np.quantile(dstar, [0.025, 0.975])
    return {
        "sr_m": sr_m,
        "sr_s": sr_s,
        "ar": ar,
        "sr_max": sr_max,
        "dsr": delta,
        "dsr_lo": float(lo),
        "dsr_hi": float(hi_),
        "dsr_p": float(np.mean(np.abs(dstar - delta) >= abs(delta))),
        "grs": float(W),
        "grs_p": float(p_grs),
        "share_s": float(rc[1]),
        "w_sign": float(np.sign(w[1]) * np.sign(w[0])),
        "d6_m": d6(rm, dates, span),
        "d6_s": d6(rs, dates, span),
        "d6_tan": d6(rp, dates, span),
        "alpha_ann": 100.0 * q * alpha,  # 2026-10-26 (D19): was 100.0 * ANN * alpha
        "q": q,
        "sig_eps": math.sqrt(s2e),
    }


def mv_theta(m: np.ndarray, s: np.ndarray) -> tuple[float, float, float, float]:
    """The mean-variance rule (RULE.txt, section 6) on the window (m, s): theta (the strategy's volatility weight) that
    maximizes the estimated Sharpe ratio among the grid points where neither leg's Euler share exceeds CAP; returns
    (theta, the strategy's Euler share, sd_m, sd_s)."""
    sm, ss = m.std(ddof=1), s.std(ddof=1)
    srm, srs = m.mean() / sm, s.mean() / ss
    rho = float(np.corrcoef(m, s)[0, 1])
    g = GRID
    V = (1 - g) ** 2 + g**2 + 2 * g * (1 - g) * rho
    rc_s = g * (g + (1 - g) * rho) / V
    ok = (rc_s <= CAP) & (1.0 - rc_s <= CAP)
    assert ok.any()
    sr = ((1 - g) * srm + g * srs) / np.sqrt(V)
    i = int(np.argmax(np.where(ok, sr, -np.inf)))
    return float(g[i]), float(rc_s[i]), float(sm), float(ss)


def realtime(rs: np.ndarray, rm: np.ndarray, dates: pd.DatetimeIndex) -> dict:
    """Section 6: the real-time mean-variance and inverse-volatility combinations against the market alone."""
    T = len(rs)
    w_mv = np.full((T, 2), np.nan)
    w_iv = np.full((T, 2), np.nan)
    share = np.full(T, np.nan)
    for t in range(BURN, T):
        th, rc_s, sm, ss = mv_theta(rm[:t], rs[:t])
        v = np.array([(1.0 - th) / sm, th / ss])
        w_mv[t] = v / v.sum()
        share[t] = rc_s
        v = np.array([1.0 / rm[t - TRAIL : t].std(ddof=1), 1.0 / rs[t - TRAIL : t].std(ddof=1)])
        w_iv[t] = v / v.sum()
    ev = slice(BURN, T)
    R2 = np.column_stack([rm, rs])[ev]
    r_mv = (w_mv[ev] * R2).sum(axis=1)
    r_iv = (w_iv[ev] * R2).sum(axis=1)
    r_m = rm[ev]
    d_ev = dates[BURN:]
    span = (d_ev[0], d_ev[-1])
    Te = len(r_m)
    idx = hi._boot_index(Te, hi.LW_BLOCK, hi.LW_BOOT, hi.D6_SEED)
    q = q_of(d_ev, span)  # 2026-10-26 (D19): sqrt(q) of the evaluation days in place of sqrt(252)
    sm_star = srq_rows(r_m[idx], q)
    out: dict = {
        "n": Te,
        "first": d_ev[0],
        "q": q,
        "sr_m": srq(r_m, q),
        "d6_m": d6(r_m, d_ev, span),
        "share_med": float(np.median(share[ev])),
        "share_min": float(np.min(share[ev])),
        # the cap binds at either end (the strategy's share at CAP or at 1 - CAP), to the grid's resolution
        "cap_bind": float(np.mean((share[ev] >= CAP - 2e-3) | (share[ev] <= 1.0 - CAP + 2e-3))),
        "ws_mv_med": float(np.median(w_mv[ev, 1])),
        "ws_iv_med": float(np.median(w_iv[ev, 1])),
        "r_m": r_m,
        "r_mv": r_mv,
        "r_iv": r_iv,
        "dates": d_ev,
    }
    for nm, rp in (("mv", r_mv), ("iv", r_iv)):
        s_p = srq(rp, q)
        delta = s_p - out["sr_m"]
        dstar = srq_rows(rp[idx], q) - sm_star
        r6 = hi.d6_diff_test(rp, r_m, d_ev, span=span)
        # 2026-10-26 (D19): the paper's test of the difference, Ledoit-Wolf (nm_p); the paired block bootstrap of
        # RULE.txt section 6 is the check nm_pb. Was: f"{nm}_p": float(np.mean(np.abs(dstar - delta) >= abs(delta)))
        lw = hi.paper_sharpe_test(rp, r_m, d_ev, span=span, check=False)
        assert abs(lw["delta"] - delta) < 1e-9, (lw["delta"], delta)
        out.update(
            {
                f"{nm}_sr": s_p,
                f"{nm}_dsr": delta,
                f"{nm}_p": float(lw["p_lw"]),
                f"{nm}_pb": float(np.mean(np.abs(dstar - delta) >= abs(delta))),
                f"{nm}_d6": r6["sr_a"],
                f"{nm}_d6_dsr": r6["delta"],
                f"{nm}_d6_p": r6["p"],
            }
        )
        assert abs(r6["sr_b"] - out["d6_m"]) < 1e-10
    return out


def run_all(series: dict, mk: dict, pw: pd.DatetimeIndex, drop: np.ndarray) -> dict:
    W = me.WARMUP
    full_span = (pw[0], pw[-1])
    keep = ~drop
    R: dict = {}
    for k in ROWS:
        q = series[k]["q"].to_numpy(float)[W:]
        for col, _ in PRICES:
            rs = series[k][col].to_numpy(float)[W:]
            assert len(rs) == len(pw) == me.N_PW and np.isfinite(rs).all(), k
            r = {"exp": exposure(rs, mk["rm"], q), "span": spanning(rs, mk["rm"], pw, full_span)}
            r["rt"] = realtime(rs, mk["rm"], pw)
            r["exp30"] = exposure(rs, mk["r30"], q)
            r["ex_exp"] = exposure(rs[keep], mk["rm"][keep], None, full=False)
            r["ex_span"] = spanning(rs[keep], mk["rm"][keep], pw[keep], full_span)
            r["ex_rt"] = realtime(rs[keep], mk["rm"][keep], pw[keep])
            R[(k, col)] = r
        print(f"  {k}: done")
    return R


# ============================================================================================ LaTeX
num, pval = em.num, em.pval
MACROS: dict[str, tuple[str, str]] = {}


def mac(name: str, val: str, note: str) -> None:
    assert re.fullmatch(r"rheb[A-Za-z]+", name), name
    assert name not in MACROS, name
    MACROS[name] = (val, note)


def numbers(R: dict, mk: dict, drop: np.ndarray, top: pd.Timestamp, pw: pd.DatetimeIndex) -> str:
    r0 = R[(FULL[0], "mid")]
    sp0, rt0, ex0, exrt0 = r0["span"], r0["rt"], r0["ex_span"], r0["ex_rt"]
    # ---- sample and market
    mac("rhebNDays", str(len(pw)), "post-warmup option days of the exposure and spanning statistics")
    mac("rhebNBurn", str(BURN), "burn-in days of the real-time rules (expanding window)")
    mac("rhebNTrail", str(TRAIL), "trailing days of the inverse-volatility rule")
    mac("rhebNRt", str(rt0["n"]), "evaluation days of the real-time rules (post-warmup days after the burn-in)")
    mac("rhebRtFirst", dc.nice_date(rt0["first"]), "first evaluation day of the real-time rules")
    mac("rhebCap", f"{100 * CAP:.0f}", "cap: largest share (%) of the portfolio volatility either leg may carry")
    mac("rhebNHarDistinct", str(len(HAR4)), "distinct HAR-set forecasts (the three linear ones coincide)")
    mac("rhebNHarDistinctWord", WORDS[len(HAR4)], "... in words")
    mac("rhebNBest", str(int(drop.sum())), "days removed in the robustness check (= \\rhedBestUnion)")
    mac("rhebNEx", str(int((~drop).sum())), "post-warmup days without them")
    mac("rhebNExRt", str(exrt0["n"]), "real-time evaluation days without them")
    mac("rhebTopDate", dc.nice_date(top), "the largest of them (= \\rhedTopDate)")
    mac("rhebBootBlock", str(hi.LW_BLOCK), "block length of the paired circular block bootstrap")
    mac("rhebBootN", f"{hi.LW_BOOT:,}".replace(",", "{,}"), "bootstrap resamples")
    mac("rhebMktMean", num(100 * float(mk["rm"].mean()), 3), "market excess return, mean, % per day, option days")
    mac("rhebMktSd", num(100 * float(mk["rm"].std(ddof=1)), 2), "... standard deviation, % per day")
    mac("rhebRfMean", num(100 * float(mk["rf"].mean()), 4), "risk-free return (DTB3, actual/360), mean, % per day")
    mac("rhebRateMean", num(mk["rate_mean"], 2), "DTB3, mean over the trading days of the span, % a year")
    mac("rhebSRm", num(sp0["sr_m"]), "market: Sharpe ratio, STANDARD (D19: sqrt(q), q = 225.5), 614 option days")
    mac("rhebSRmDSix", num(sp0["d6_m"]), "market: Sharpe ratio, D6, 614 option days")
    mac(
        "rhebSRmAll",
        num(srq(mk["rm_all"], mk["q_all"])),
        "market: Sharpe ratio, STANDARD (D19: sqrt(q) of these days), every trading day of the span",
    )
    mac("rhebQAll", f"{mk['q_all']:.1f}", "D19: q of every trading day of the span")
    mac("rhebQRt", f"{rt0['q']:.1f}", "D19: q of the real-time evaluation days")
    mac("rhebNAll", str(mk["n_all"]), "... trading days of the span 2021-08-11 .. 2024-04-30")
    mac("rhebSRmRt", num(rt0["sr_m"]), "market: Sharpe ratio, STANDARD, real-time evaluation days")
    mac("rhebSRmRtDSix", num(rt0["d6_m"]), "market: Sharpe ratio, D6, real-time evaluation days")
    mac("rhebSRmEx", num(ex0["sr_m"]), "market: Sharpe ratio, STANDARD, without the best-day union")
    mac("rhebSRmExRt", num(exrt0["sr_m"]), "market: Sharpe ratio, STANDARD, real-time days without the union")
    e0 = r0["exp"]
    mac("rhebWorstCut", num(e0["worst_cut"], 2), "worst-decile threshold: 10% quantile of the market excess return, %")
    mac("rhebNWorst", str(e0["worst_n"]), "worst-decile market days")
    mac("rhebNDown", str(e0["down_n"]), "market-down days (excess return < 0)")
    mac("rhebWorstCutThirty", num(r0["exp30"]["worst_cut"], 2), "... 10% quantile of the 15:30-16:00 return, %")

    # ---- per strategy
    for k in ROWS:
        t = em.tag(k)
        for col, P in PRICES:
            r = R[(k, col)]
            e, s, rt, ee, es, ert = r["exp"], r["span"], r["rt"], r["ex_exp"], r["ex_span"], r["ex_rt"]
            pr = "mid" if col == "mid" else "bid/ask"
            mac(f"rhebCorr{P}{t}", num(e["corr"]), f"{k}, {pr}: correlation with the market excess return")
            mac(f"rhebBeta{P}{t}", num(e["beta"]), "... CAPM beta (% of premium per % of market)")
            mac(f"rhebBetaT{P}{t}", num(e["beta_t"]), "... its HAC t (S = 33)")
            mac(f"rhebBetaP{P}{t}", pval(e["beta_p"]), "... its fixed-b p")
            mac(f"rhebRsq{P}{t}", num(e["r2"]), "... CAPM R^2")
            mac(f"rhebAlpha{P}{t}", num(e["alpha"], 1), "... CAPM alpha, % of premium per day")
            mac(f"rhebAlphaT{P}{t}", num(e["alpha_t"]), "... its HAC t")
            mac(f"rhebAlphaP{P}{t}", pval(e["alpha_p"]), "... its fixed-b p")
            mac(f"rhebAlphaAnn{P}{t}", num(s["alpha_ann"], 0), "... alpha x q (D19; was x 252), % of premium a year")
            mac(f"rhebAR{P}{t}", num(s["ar"]), "... appraisal ratio alpha / sigma_eps, annualized sqrt(q) (D19)")
            mac(f"rhebSRs{P}{t}", num(s["sr_s"]), "... strategy Sharpe ratio, STANDARD = the paper's (= \\rhemSh / X)")
            mac(f"rhebSRsDSix{P}{t}", num(s["d6_s"]), "... strategy Sharpe ratio, D6 (= \\rhemShDSix)")
            mac(f"rhebSRmax{P}{t}", num(s["sr_max"]), "... max Sharpe ratio of market + strategy, STANDARD")
            mac(f"rhebDSRmax{P}{t}", num(s["dsr"]), "... its gain over the market alone, STANDARD")
            mac(f"rhebDSRmaxLo{P}{t}", num(s["dsr_lo"]), "... bootstrap 95% interval, lower end")
            mac(f"rhebDSRmaxHi{P}{t}", num(s["dsr_hi"]), "... upper end")
            mac(f"rhebDSRmaxP{P}{t}", pval(s["dsr_p"]), "... bootstrap p (descriptive; Delta >= 0)")
            mac(f"rhebGrs{P}{t}", num(s["grs"]), "... one-asset GRS statistic, F(1, T - 2)")
            mac(f"rhebGrsP{P}{t}", pval(s["grs_p"]), "... its p (i.i.d. errors)")
            mac(f"rhebTanShare{P}{t}", num(100 * s["share_s"], 0), "... tangency: strategy's share of volatility, %")
            mac(f"rhebTanDSix{P}{t}", num(s["d6_tan"]), "... tangency (fixed in-sample weights): Sharpe ratio, D6")
            for nm, lab in (("mv", "Mv"), ("iv", "Iv")):
                rule = "mean-variance" if nm == "mv" else "inverse-volatility"
                mac(f"rhebRt{lab}{P}{t}", num(rt[f"{nm}_sr"]), f"... real-time {rule}: Sharpe ratio, STANDARD")
                mac(f"rhebRt{lab}D{P}{t}", num(rt[f"{nm}_dsr"]), "... minus the market's, STANDARD")
                mac(f"rhebRt{lab}P{P}{t}", pval(rt[f"{nm}_p"]), "... Ledoit-Wolf p of the difference (D19)")
                mac(f"rhebRt{lab}DSix{P}{t}", num(rt[f"{nm}_d6"]), f"... real-time {rule}: Sharpe ratio, D6")
                mac(f"rhebRt{lab}DSixD{P}{t}", num(rt[f"{nm}_d6_dsr"]), "... minus the market's, D6")
                mac(f"rhebRt{lab}DSixP{P}{t}", pval(rt[f"{nm}_d6_p"]), "... bootstrap p of the D6 difference")
                mac(f"rhebExRt{lab}{P}{t}", num(ert[f"{nm}_sr"]), f"... w/o best days, real-time {rule}: SR, STANDARD")
                mac(f"rhebExRt{lab}D{P}{t}", num(ert[f"{nm}_dsr"]), "... minus the market's, STANDARD")
                mac(f"rhebExRt{lab}P{P}{t}", pval(ert[f"{nm}_p"]), "... its Ledoit-Wolf p (D19)")
            mac(f"rhebRtMvShare{P}{t}", num(100 * rt["share_med"], 0), "... real-time MV: median strategy vol share, %")
            mac(f"rhebRtMvCapDays{P}{t}", num(100 * rt["cap_bind"], 0), "... % of days the 90% cap binds")
            mac(
                f"rhebRtMvWs{P}{t}", num(100 * rt["ws_mv_med"], 1), "... real-time MV: median strategy dollar weight, %"
            )
            mac(
                f"rhebRtIvWs{P}{t}", num(100 * rt["ws_iv_med"], 1), "... real-time IV: median strategy dollar weight, %"
            )
            mac(f"rhebExBeta{P}{t}", num(ee["beta"]), "... without the best-day union: CAPM beta")
            mac(f"rhebExBetaT{P}{t}", num(ee["beta_t"]), "... its HAC t")
            mac(f"rhebExAlpha{P}{t}", num(ee["alpha"], 1), "... without the best-day union: alpha, % per day")
            mac(f"rhebExAlphaT{P}{t}", num(ee["alpha_t"]), "... its HAC t")
            mac(f"rhebExAlphaP{P}{t}", pval(ee["alpha_p"]), "... its fixed-b p")
            mac(f"rhebExAR{P}{t}", num(es["ar"]), "... appraisal ratio, sqrt(q) (D19)")
            mac(f"rhebExSRmax{P}{t}", num(es["sr_max"]), "... max Sharpe ratio, STANDARD")
            mac(f"rhebExDSRmaxP{P}{t}", pval(es["dsr_p"]), "... bootstrap p of its gain")
            mac(f"rhebExGrsP{P}{t}", pval(es["grs_p"]), "... GRS p")
            if col == "x":
                continue
            # mid only: Treynor-Mazuy, down days, the 15:30-16:00 window
            mac(f"rhebTmGamma{t}", num(e["tm_g"]), f"{k}, mid: Treynor-Mazuy gamma on (r^m)^2 (% per %^2)")
            mac(f"rhebTmGammaT{t}", num(e["tm_gt"]), "... its HAC t")
            mac(f"rhebTmGammaP{t}", pval(e["tm_gp"]), "... its fixed-b p")
            mac(f"rhebTmBeta{t}", num(e["tm_b"]), "... Treynor-Mazuy beta on r^m")
            mac(f"rhebTmBetaT{t}", num(e["tm_bt"]), "... its HAC t")
            for side, S_ in (("long", "Long"), ("short", "Short")):
                mac(f"rhebN{S_}{t}", str(e[f"n_{side}"]), f"... {side} days")
                mac(f"rhebTmGamma{S_}{t}", num(e[f"tm_{side}_g"]), f"... TM gamma on its {side} days only")
                mac(f"rhebTmGamma{S_}T{t}", num(e[f"tm_{side}_gt"]), "... its HAC t (no test: RULE section 4)")
            for nm, S_ in (("down", "Down"), ("worst", "Worst")):
                what = "market-down days" if nm == "down" else "worst-decile market days"
                mac(f"rheb{S_}Mean{t}", num(e[f"{nm}_mean"], 1), f"... mean return on {what}, % of premium")
                mac(f"rheb{S_}Rest{t}", num(e[f"{nm}_rest"], 1), "... on the other days")
                mac(f"rheb{S_}DiffT{t}", num(e[f"{nm}_t"]), "... HAC t of the difference")
                mac(f"rheb{S_}DiffP{t}", pval(e[f"{nm}_p"]), "... its fixed-b p")
            e3 = r["exp30"]
            mac(f"rhebThirtyCorr{t}", num(e3["corr"]), "... correlation with the 15:30-16:00 index return")
            mac(f"rhebThirtyBeta{t}", num(e3["beta"]), "... beta on the 15:30-16:00 index return")
            mac(f"rhebThirtyBetaT{t}", num(e3["beta_t"]), "... its HAC t")
            mac(f"rhebThirtyBetaP{t}", pval(e3["beta_p"]), "... its fixed-b p")
            mac(f"rhebThirtyAlpha{t}", num(e3["alpha"], 1), "... alpha against it, % per day")
            mac(f"rhebThirtyAlphaT{t}", num(e3["alpha_t"]), "... its HAC t")
            mac(f"rhebThirtyTmGamma{t}", num(e3["tm_g"]), "... TM gamma on the squared 15:30-16:00 return")
            mac(f"rhebThirtyTmGammaT{t}", num(e3["tm_gt"]), "... its HAC t")
            mac(f"rhebThirtyTmGammaP{t}", pval(e3["tm_gp"]), "... its fixed-b p")
            for side, S_ in (("long", "Long"), ("short", "Short")):
                mac(f"rhebThirtyTmGamma{S_}{t}", num(e3[f"tm_{side}_g"]), f"... ... on its {side} days only")
                mac(f"rhebThirtyTmGamma{S_}T{t}", num(e3[f"tm_{side}_gt"]), "... its HAC t (no test)")
            for nm, S_ in (("down", "Down"), ("worst", "Worst")):
                mac(f"rhebThirty{S_}Mean{t}", num(e3[f"{nm}_mean"], 1), f"... mean on 15:30-16:00 {nm} days, %")
                mac(f"rhebThirty{S_}DiffT{t}", num(e3[f"{nm}_t"]), "... HAC t of the difference with the other days")

    # ---- ranges and counts over the six full-set forecasts and the four distinct HAR-set forecasts
    def rng(stem: str, ks: list[str], f, dp: int, note: str) -> None:
        """Min and max over ks; dp = -1 formats a p-value (pval: "<0.001" below 0.001)."""
        v = [f(k) for k in ks]
        fmt = pval if dp < 0 else (lambda x: num(x, dp))
        mac(f"{stem}Min", fmt(min(v)), note + ": min")
        mac(f"{stem}Max", fmt(max(v)), "... max")

    def cnt(name: str, n: int, note: str) -> None:
        mac(name, str(n), note)
        mac(name + "Word", WORDS[n], "... in words")

    for col, P in PRICES:
        pr = "mid" if col == "mid" else "bid/ask"
        for grp, ks, gl in (("Full", FULL, "six full-set"), ("Har", HAR4, "four distinct HAR-set")):
            g = f"{grp}{P}"

            def G(k: str, part: str, field: str, c: str = col) -> float:
                return R[(k, c)][part][field]

            rng(f"rhebBeta{g}", ks, lambda k: G(k, "exp", "beta"), 2, f"CAPM beta, {pr}, {gl}")
            rng(f"rhebBetaT{g}", ks, lambda k: G(k, "exp", "beta_t"), 2, f"... its HAC t, {pr}, {gl}")
            rng(f"rhebBetaP{g}", ks, lambda k: G(k, "exp", "beta_p"), -1, f"... its fixed-b p, {pr}, {gl}")
            rng(f"rhebCorr{g}", ks, lambda k: G(k, "exp", "corr"), 2, f"correlation, {pr}, {gl}")
            rng(f"rhebRsq{g}", ks, lambda k: G(k, "exp", "r2"), 2, f"CAPM R^2, {pr}, {gl}")
            rng(f"rhebAlpha{g}", ks, lambda k: G(k, "exp", "alpha"), 1, f"alpha % per day, {pr}, {gl}")
            rng(f"rhebAlphaT{g}", ks, lambda k: G(k, "exp", "alpha_t"), 2, f"alpha HAC t, {pr}, {gl}")
            rng(f"rhebAlphaP{g}", ks, lambda k: G(k, "exp", "alpha_p"), -1, f"alpha fixed-b p, {pr}, {gl}")
            rng(f"rhebAR{g}", ks, lambda k: G(k, "span", "ar"), 2, f"appraisal ratio (sqrt q, D19), {pr}, {gl}")
            rng(f"rhebSRmax{g}", ks, lambda k: G(k, "span", "sr_max"), 2, f"max SR STANDARD, {pr}, {gl}")
            rng(f"rhebDSRmax{g}", ks, lambda k: G(k, "span", "dsr"), 2, f"max SR gain STANDARD, {pr}, {gl}")
            rng(f"rhebDSRmaxP{g}", ks, lambda k: G(k, "span", "dsr_p"), -1, f"its bootstrap p, {pr}, {gl}")
            rng(f"rhebDSRmaxLo{g}", ks, lambda k: G(k, "span", "dsr_lo"), 2, f"its 95% interval, lower end, {gl}")
            rng(f"rhebDSRmaxHi{g}", ks, lambda k: G(k, "span", "dsr_hi"), 2, f"its 95% interval, upper end, {gl}")
            rng(f"rhebGrsP{g}", ks, lambda k: G(k, "span", "grs_p"), -1, f"GRS p, {pr}, {gl}")
            rng(f"rhebTanShare{g}", ks, lambda k: 100 * G(k, "span", "share_s"), 0, f"tangency vol share %, {pr}, {gl}")
            rng(f"rhebSRs{g}", ks, lambda k: G(k, "span", "sr_s"), 2, f"strategy SR STANDARD, {pr}, {gl}")
            for nm, lab in (("mv", "Mv"), ("iv", "Iv")):
                rng(
                    f"rhebRt{lab}{g}",
                    ks,
                    lambda k: G(k, "rt", f"{nm}_sr"),
                    2,
                    f"real-time {lab} SR STANDARD, {pr}, {gl}",
                )
                rng(f"rhebRt{lab}D{g}", ks, lambda k: G(k, "rt", f"{nm}_dsr"), 2, f"... minus market, STANDARD, {gl}")
                rng(f"rhebRt{lab}P{g}", ks, lambda k: G(k, "rt", f"{nm}_p"), -1, f"... Ledoit-Wolf p (D19), {gl}")
                rng(
                    f"rhebRt{lab}DSix{g}", ks, lambda k: G(k, "rt", f"{nm}_d6"), 2, f"real-time {lab} SR D6, {pr}, {gl}"
                )
                rng(f"rhebRt{lab}DSixD{g}", ks, lambda k: G(k, "rt", f"{nm}_d6_dsr"), 2, f"... minus market, D6, {gl}")
                rng(f"rhebRt{lab}DSixP{g}", ks, lambda k: G(k, "rt", f"{nm}_d6_p"), -1, f"... bootstrap p, D6, {gl}")
                rng(
                    f"rhebExRt{lab}{g}",
                    ks,
                    lambda k: G(k, "ex_rt", f"{nm}_sr"),
                    2,
                    f"w/o best days: RT {lab} SR STANDARD",
                )
                rng(f"rhebExRt{lab}P{g}", ks, lambda k: G(k, "ex_rt", f"{nm}_p"), -1, "... its Ledoit-Wolf p (D19)")
            rng(f"rhebRtMvShare{g}", ks, lambda k: 100 * G(k, "rt", "share_med"), 0, f"RT MV median vol share %, {gl}")
            rng(
                f"rhebRtMvCapDays{g}",
                ks,
                lambda k: 100 * G(k, "rt", "cap_bind"),
                0,
                f"RT MV: % of days at the cap, {gl}",
            )
            rng(f"rhebRtMvWs{g}", ks, lambda k: 100 * G(k, "rt", "ws_mv_med"), 1, f"RT MV median dollar weight %, {gl}")
            rng(f"rhebRtIvWs{g}", ks, lambda k: 100 * G(k, "rt", "ws_iv_med"), 1, f"RT IV median dollar weight %, {gl}")
            rng(f"rhebExBeta{g}", ks, lambda k: G(k, "ex_exp", "beta"), 2, f"w/o best days: beta, {pr}, {gl}")
            rng(f"rhebExBetaT{g}", ks, lambda k: G(k, "ex_exp", "beta_t"), 2, "... its HAC t")
            rng(f"rhebExAlpha{g}", ks, lambda k: G(k, "ex_exp", "alpha"), 1, f"w/o best days: alpha %, {pr}, {gl}")
            rng(f"rhebExAlphaT{g}", ks, lambda k: G(k, "ex_exp", "alpha_t"), 2, "... its HAC t")
            rng(f"rhebExAlphaP{g}", ks, lambda k: G(k, "ex_exp", "alpha_p"), -1, "... its fixed-b p")
            rng(f"rhebExAR{g}", ks, lambda k: G(k, "ex_span", "ar"), 2, "... appraisal ratio (sqrt q, D19)")
            rng(f"rhebExSRmax{g}", ks, lambda k: G(k, "ex_span", "sr_max"), 2, "... max SR STANDARD")
            rng(f"rhebExGrsP{g}", ks, lambda k: G(k, "ex_span", "grs_p"), -1, "... GRS p")
            cnt(f"rhebBetaSig{g}", sum(G(k, "exp", "beta_p") < 0.05 for k in ks), f"{gl}: beta sig. 5%, {pr}")
            cnt(f"rhebAlphaSig{g}", sum(G(k, "exp", "alpha_p") < 0.05 for k in ks), f"{gl}: alpha sig. 5% (HAC), {pr}")
            cnt(f"rhebGrsSig{g}", sum(G(k, "span", "grs_p") < 0.05 for k in ks), f"{gl}: GRS sig. 5%, {pr}")
            cnt(f"rhebBootSig{g}", sum(G(k, "span", "dsr_p") < 0.05 for k in ks), f"{gl}: max-SR bootstrap p < 0.05")
            for nm, lab in (("mv", "Mv"), ("iv", "Iv")):
                cnt(
                    f"rhebRt{lab}Beat{g}",
                    sum(G(k, "rt", f"{nm}_dsr") > 0 for k in ks),
                    f"{gl}: RT {lab} SR > mkt (STD)",
                )
                cnt(
                    f"rhebRt{lab}Sig{g}",
                    sum(G(k, "rt", f"{nm}_dsr") > 0 and G(k, "rt", f"{nm}_p") < 0.05 for k in ks),
                    "... and Ledoit-Wolf p < 0.05 (D19; was the bootstrap p)",
                )
                cnt(
                    f"rhebRt{lab}DSixSig{g}",
                    sum(G(k, "rt", f"{nm}_d6_dsr") > 0 and G(k, "rt", f"{nm}_d6_p") < 0.05 for k in ks),
                    "... D6 difference > 0 with bootstrap p < 0.05",
                )
                cnt(
                    f"rhebExRt{lab}Sig{g}",
                    sum(G(k, "ex_rt", f"{nm}_dsr") > 0 and G(k, "ex_rt", f"{nm}_p") < 0.05 for k in ks),
                    "... w/o best days: STANDARD difference > 0 with p < 0.05",
                )
            cnt(
                f"rhebExAlphaSig{g}", sum(G(k, "ex_exp", "alpha_p") < 0.05 for k in ks), "... w/o best days: alpha sig."
            )
            cnt(f"rhebExGrsSig{g}", sum(G(k, "ex_span", "grs_p") < 0.05 for k in ks), "... w/o best days: GRS sig.")
            if col == "mid":
                rng(f"rhebTmGamma{grp}", ks, lambda k: G(k, "exp", "tm_g"), 2, f"TM gamma, mid, {gl}")
                rng(f"rhebTmGammaT{grp}", ks, lambda k: G(k, "exp", "tm_gt"), 2, "... its HAC t")
                rng(f"rhebTmGammaLong{grp}", ks, lambda k: G(k, "exp", "tm_long_g"), 2, f"TM gamma, long days, {gl}")
                rng(f"rhebTmGammaLongT{grp}", ks, lambda k: G(k, "exp", "tm_long_gt"), 2, "... its HAC t")
                rng(f"rhebTmGammaShort{grp}", ks, lambda k: G(k, "exp", "tm_short_g"), 2, f"TM gamma, short days, {gl}")
                rng(f"rhebTmGammaShortT{grp}", ks, lambda k: G(k, "exp", "tm_short_gt"), 2, "... its HAC t")
                rng(f"rhebThirtyBeta{grp}", ks, lambda k: G(k, "exp30", "beta"), 2, f"beta on r30, mid, {gl}")
                rng(f"rhebThirtyBetaT{grp}", ks, lambda k: G(k, "exp30", "beta_t"), 2, "... its HAC t")
                rng(f"rhebThirtyTmGamma{grp}", ks, lambda k: G(k, "exp30", "tm_g"), 2, f"TM gamma on r30^2, {gl}")
                rng(f"rhebThirtyTmGammaT{grp}", ks, lambda k: G(k, "exp30", "tm_gt"), 2, "... its HAC t")
                rng(f"rhebThirtyTmGammaLong{grp}", ks, lambda k: G(k, "exp30", "tm_long_g"), 2, f"... long days, {gl}")
                rng(f"rhebThirtyTmGammaLongT{grp}", ks, lambda k: G(k, "exp30", "tm_long_gt"), 2, "... its HAC t")
                rng(
                    f"rhebThirtyTmGammaShort{grp}",
                    ks,
                    lambda k: G(k, "exp30", "tm_short_g"),
                    2,
                    f"... short days, {gl}",
                )
                rng(f"rhebThirtyTmGammaShortT{grp}", ks, lambda k: G(k, "exp30", "tm_short_gt"), 2, "... its HAC t")
                rng(f"rhebDownMean{grp}", ks, lambda k: G(k, "exp", "down_mean"), 1, f"mean on down days %, {gl}")
                rng(f"rhebDownRest{grp}", ks, lambda k: G(k, "exp", "down_rest"), 1, "... on the other days")
                rng(f"rhebDownDiffT{grp}", ks, lambda k: G(k, "exp", "down_t"), 2, "... HAC t of the difference")
                rng(
                    f"rhebWorstMean{grp}",
                    ks,
                    lambda k: G(k, "exp", "worst_mean"),
                    1,
                    f"mean on worst-decile days, {gl}",
                )
                rng(f"rhebWorstRest{grp}", ks, lambda k: G(k, "exp", "worst_rest"), 1, "... on the other days")
                rng(f"rhebWorstDiffT{grp}", ks, lambda k: G(k, "exp", "worst_t"), 2, "... HAC t of the difference")
                cnt(f"rhebTmSig{grp}", sum(G(k, "exp", "tm_gp") < 0.05 for k in ks), f"{gl}: TM gamma sig. 5%")
                cnt(f"rhebThirtyBetaSig{grp}", sum(G(k, "exp30", "beta_p") < 0.05 for k in ks), f"{gl}: r30 beta sig.")
                cnt(f"rhebDownSig{grp}", sum(G(k, "exp", "down_p") < 0.05 for k in ks), f"{gl}: down-day diff sig.")
                cnt(f"rhebWorstSig{grp}", sum(G(k, "exp", "worst_p") < 0.05 for k in ks), f"{gl}: worst-decile sig.")
    existing: set[str] = set()
    for f in PAPER_GEN.glob("*.tex"):
        if f.name == NUMBERS.name:
            continue
        existing |= set(re.findall(r"\\(?:provide|new|renew)command\{\\([A-Za-z]+)\}", f.read_text("utf-8", "ignore")))
    clash = sorted(set(MACROS) & existing)
    assert not clash, f"macro names already defined in writeup/generated: {clash}"
    lines = [
        HEADER.rstrip("\n"),
        "% \\rheb... macros: market exposure of the sign strategy and its allocation with the market (appendix,",
        "% tab:rh_econ_beta). Pre-registration: results/econ_beta_2026-10-26/RULE.txt. Row stems <Est><Set> (Est = Ridge,",
        "% Lasso, Enet, Lgbm, Xgb, Rf; Set = Full, Har), CiHar, Short; X = quoted bid/ask prices, else mid-quotes; Ex =",
        "% without the union of the full-set forecasts' ten best days. Market: S&P 500 close-to-close price return minus",
        "% the DTB3 T-bill return (no dividends); Thirty = the index's 15:30-16:00 return. STANDARD Sharpe ratio =",
        "% sqrt(q) mean / sd of the pooled daily returns, q = T / sum_y c_y of the sample's days (2026-10-26, D19: the",
        "% paper's Sharpe ratio; was sqrt(252)); D6 = the equal-year mean of SR_y (hac_inference.d6_sharpe; robustness).",
    ]
    lines += [f"\\providecommand{{\\{n}}}{{}}\\renewcommand{{\\{n}}}{{{v}}}  % {c}" for n, (v, c) in MACROS.items()]
    return "\n".join(lines) + "\n"


def table_tex(R: dict) -> str:
    # 2026-10-26 (D2/D3): parked -- the "Closing HAR" column. Was: ncol = 9, heads + "\\makecell[r]{Closing\\\\HAR}"
    ncol = 8
    heads = [dc.HEAD_TEX[e] for e in EST] + ["\\makecell[r]{Short\\\\straddle}"]
    star = "--$^{\\phantom{***}}$"

    def V(k: str, col: str, part: str, field: str) -> float:
        return R[(k, col)][part][field]

    def est_row(label: str, col: str, part: str, f: str, ft: str | None, fp: str | None, dp: int = 2, sc=1.0) -> str:
        top, under = [], []
        for k in TAB_ROWS:
            x = V(k, col, part, f)
            if not np.isfinite(x):
                top.append(star)
                under.append("")
                continue
            stars = me.n_stars_p(V(k, col, part, fp)) if fp else 0
            top.append(me.cell(sc * x, dp, stars))
            under.append(me.tstat(V(k, col, part, ft)) if ft else "")
        return me.rows2(label, top, under)

    def plain_row(label: str, col: str, part: str, f: str, dp: int = 2, sc=1.0) -> str:
        return (
            f"{label} & "
            + " & ".join(me.cell(sc * V(k, col, part, f), dp, 0) for k in TAB_ROWS)  # phantom stars: aligned
            + " \\\\\n"
        )

    def pb(p: float) -> str:
        """me.pbrack, with "<0.001" below 0.001."""
        return "$[<0.001]^{\\phantom{***}}$" if p < 0.001 else me.pbrack(p)

    def sr_row(label: str, col: str, part: str, f: str, fp: str) -> str:
        top = [me.cell(V(k, col, part, f), 2, 0) for k in TAB_ROWS]
        under = [pb(V(k, col, part, fp)) for k in TAB_ROWS]
        return me.rows2(label, top, under)

    b = ""
    b += me.panel_head("Panel A: Exposure to the market, mid-quotes", ncol, first=True)
    b += est_row("$\\beta$", "mid", "exp", "beta", "beta_t", "beta_p")
    b += est_row("$\\gamma$, $(r^{m}_t)^2$", "mid", "exp", "tm_g", "tm_gt", "tm_gp")
    b += est_row("Worst decile (\\%)", "mid", "exp", "worst_mean", "worst_t", "worst_p", 1)
    b += est_row("$\\gamma$, $(r^{30}_t)^2$", "mid", "exp30", "tm_g", "tm_gt", "tm_gp", 0)
    b += me.panel_head("Panel B: Alpha and the maximum Sharpe ratio, mid-quotes", ncol)
    b += est_row("$\\alpha$ (\\% a day)", "mid", "exp", "alpha", "alpha_t", "alpha_p", 1)
    b += sr_row("Maximum $\\SR$", "mid", "span", "sr_max", "dsr_p")
    b += (
        "GRS $p$ & "
        + " & ".join(
            "$<0.001^{\\phantom{***}}$"
            if V(k, "mid", "span", "grs_p") < 0.001
            else f"${V(k, 'mid', 'span', 'grs_p'):.3f}^{{\\phantom{{***}}}}$"
            for k in TAB_ROWS
        )
        + " \\\\\n"
    )
    b += plain_row("Risk share (\\%)", "mid", "span", "share_s", 0, 100.0)
    b += me.panel_head("Panel C: Real-time allocation, mid-quotes, $\\SR$", ncol)
    b += sr_row("Mean--variance", "mid", "rt", "mv_sr", "mv_p")
    b += sr_row("Inverse volatility", "mid", "rt", "iv_sr", "iv_p")
    b += me.panel_head("Panel D: Quoted bid/ask prices", ncol)
    b += est_row("$\\alpha$ (\\% a day)", "x", "exp", "alpha", "alpha_t", "alpha_p", 1)
    b += plain_row("Maximum $\\SR$", "x", "span", "sr_max")
    b += sr_row("Mean--variance $\\SR$", "x", "rt", "mv_sr", "mv_p")
    b += sr_row("Inverse vol.\\ $\\SR$", "x", "rt", "iv_sr", "iv_p")
    b += me.panel_head("Panel E: Without the \\rhebNBest{} best days, mid-quotes", ncol)
    b += est_row("$\\alpha$ (\\% a day)", "mid", "ex_exp", "alpha", "alpha_t", "alpha_p", 1)
    b += plain_row("Maximum $\\SR$", "mid", "ex_span", "sr_max")
    b += sr_row("Mean--variance $\\SR$", "mid", "ex_rt", "mv_sr", "mv_p")
    b += sr_row("Inverse vol.\\ $\\SR$", "mid", "ex_rt", "iv_sr", "iv_p")
    return (
        HEADER
        + "% Table tab:rh_econ_beta: market exposure of the sign strategy, its CAPM alpha, the maximum Sharpe ratio of\n"
        "% the market and the strategy, and the real-time allocations. Macros: \\rheb... (rehaul_econ_beta_numbers.tex).\n"
        "\\ifdefined\\rhebSRm\\else\\input{generated/rehaul_econ_beta_numbers}\\fi\n"
        "\\begin{table}[tbp]\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{2.5pt}\n"
        "\\begin{threeparttable}\n"
        "\\caption{Market exposure and allocation with the market}\n"
        "\\label{tab:rh_econ_beta}\n"
        "\\begin{tabular}{@{}lrrrrrrr@{}}\n\\toprule\n"
        # 2026-10-28 (author: prune "pooled"). Was: " & \\multicolumn{6}{c}{Pooled intraday, full predictor set} ..."
        " & \\multicolumn{6}{c}{Intraday, full predictor set} & \\\\\n\\cmidrule(lr){2-7}\n"
        " & " + " & ".join(heads) + " \\\\\n\\midrule\n" + b + "\\bottomrule\n\\end{tabular}\n"
        "\\begin{tablenotes}[flushleft]\n\\footnotesize\n"
        "\\item \\textit{Notes:} Daily return of the sign strategy (\\% of the premium, an excess return), "
        "\\rhebNDays{} post-warmup days. $r^{m}_t$: S\\&P~500 close-to-close price return minus the three-month "
        "Treasury bill; $r^{30}_t$: the index's 15:30--16:00 ET return. $\\beta$, $\\gamma$: slopes on $r^{m}_t$ and "
        "its square \\citep{TreynorMazuy1966}; worst decile: mean return on the tenth of the days with the lowest "
        "$r^{m}_t$ (stars and $t$: its difference from the other days). $t$-statistics in parentheses (Newey--West, "
        "$S = \\rheHACS$, fixed-$b$). " + me.STAR_NOTE + " Maximum $\\SR$: the in-sample tangency portfolio of the "
        "market ($\\SR_m = \\rhebSRm$) and the strategy, with the block-bootstrap $p$-value of its gain in brackets; "
        "GRS $p$: the test of $\\alpha = 0$ \\citep{GibbonsRossShanken1989}; risk share: the strategy's share of the "
        "tangency portfolio's volatility. Mean--variance and inverse volatility: weights from earlier days only, on "
        "the \\rhebNRt{} days from \\rhebRtFirst{} (market $\\rhebSRmRt$; without the best days of "
        "Table~\\ref{tab:rh_econ_bestdays}, $\\rhebSRmEx$ in sample and $\\rhebSRmExRt$ in real time); no short "
        "positions, neither leg above \\rhebCap\\% of the volatility; inverse volatility over the trailing "
        "\\rhebNTrail{} days; in brackets, the $p$-value of the \\citet{LedoitWolf2008} test of its gain over the "
        "market. Every $\\SR$ here is the paper's: $\\sqrt{q}$ times the mean over the standard deviation of the daily "
        "returns, $q$ the days per calendar year of the sample (\\rheAnnDays{} on the \\rhebNDays{} days, "
        "\\rhebQRt{} on the \\rhebNRt{} real-time days).\n"
        "\\end{tablenotes}\n\\end{threeparttable}\n\\end{table}\n"
    )


# ============================================================================================ figure
def figure(R: dict) -> None:
    rt = R[(em.key("lgbm", "all_features"), "mid")]["rt"]
    d, rm = rt["dates"], rt["r_m"]
    sd = rm.std(ddof=1)
    fig, ax = plt.subplots(figsize=(me.FIG_W, 2.8), layout="constrained")
    ax.axhline(0, color="#bdbdbd", lw=0.8, zorder=0)
    for r, lab, c, ls in (
        (rm, "Market alone", me.FAM_COLOR["short"], (0, (4, 2))),
        (rt["r_iv"], "Market and LightGBM strategy, inverse volatility", me.FAM_COLOR["linear"], "-"),
        (rt["r_mv"], "Market and LightGBM strategy, mean--variance", me.FAM_COLOR["pooled"], "-"),
    ):
        ax.plot(d, 100 * np.cumsum(r * sd / r.std(ddof=1)), color=c, lw=1.3, ls=ls, label=lab.replace("--", "–"))
    ax.set_ylabel("Cumulative excess return (%)")
    me._style(ax)
    ax.legend(loc="upper left", frameon=False)
    w, _ = fig.get_size_inches()
    assert abs(w - me.FIG_W) < 1e-9
    fig.savefig(FIG_PDF)
    if PREVIEW is not None:
        PREVIEW.mkdir(parents=True, exist_ok=True)
        fig.savefig(PREVIEW / "rehaul_econ_beta_cum.png", dpi=200)
    plt.close(fig)
    print("wrote", fr.rel(FIG_PDF))


def fig_float_tex() -> str:
    return (
        HEADER + "\\ifdefined\\rhebSRm\\else\\input{generated/rehaul_econ_beta_numbers}\\fi\n"
        "\\begin{figure}[tbp]\n\\centering\n"
        "\\includegraphics[width=\\linewidth]{generated/rehaul_econ_beta_cum.pdf}\n"
        # 2026-10-28 (author: prune "pooled"). Was: "strategy of the pooled intraday LightGBM forecast on the full ..."
        "\\caption{Cumulative excess return of the market alone and of its real-time combinations with the sign "
        "strategy of the intraday LightGBM forecast on the full predictor set, at mid-quotes, on the "
        "\\rhebNRt{} days from \\rhebRtFirst{} (Table~\\ref{tab:rh_econ_beta}, Panel C). Each series is the sum of "
        "its daily excess returns, scaled to the standard deviation of the market's daily excess return over these "
        "days, so the lines differ by their Sharpe ratios: \\rhebSRmRt{} for the market, \\rhebRtIvLgbmFull{} with "
        "inverse-volatility weights and \\rhebRtMvLgbmFull{} with mean--variance weights (annualized by "
        "$\\sqrt{q}$, $q = \\rhebQRt$).}\n"
        "\\label{fig:rh_econ_beta_cum}\n\\end{figure}\n"
    )


# ============================================================================================ draft claims
DRAFT = WRITEUP / "drafts" / "2026-10-26_econ_beta.tex"


def draft_claims(R: dict, top: pd.Timestamp) -> None:
    """DRAFT CLAIMS: the words of writeup/drafts/2026-10-26_econ_beta.tex that no macro carries ("none", "every", the
    signs and orderings stated in words); a rerun whose data no longer bear one out stops here, naming the sentence."""

    def v(k: str, part: str, f: str, col: str = "mid") -> float:
        return R[(k, col)][part][f]

    # 2026-10-26 (D2/D3): the CI-HAR leaves the main-text claims (its GRS claim is reported only, below). Was:
    # rows4 = FULL + HAR4 + [CI, SHORT], "no beta is significant (full set, HAR set, CI-HAR, short; ...)"
    rows4 = FULL + HAR4 + [SHORT]
    claims = [
        (
            all(v(k, "exp", "beta_p", c) >= 0.05 for k in rows4 for c in ("mid", "x")),
            "no beta is significant (full set, HAR set, short; mid and bid/ask)",
        ),
        (all(v(k, "exp", "down_p") >= 0.05 for k in FULL), "none of the six down-day differences is significant"),
        (all(v(k, "exp", "worst_mean") < 0 for k in FULL), "every full-set strategy loses on worst-decile days"),
        (all(v(k, "exp", "tm_gp") >= 0.05 for k in FULL), "no close-to-close Treynor-Mazuy gamma is significant"),
        (
            all(v(k, "exp30", "tm_long_g") > 0 and v(k, "exp30", "tm_short_g") < 0 for k in FULL),
            "on the 15:30-16:00 window every full-set gamma is positive on long days and negative on short days",
        ),
        (all(v(k, "exp30", "tm_g") < 0 for k in FULL), "over all days the full-set strategies are short (r30)^2"),
        (all(v(k, "exp30", "beta_p") >= 0.05 for k in FULL), "no full-set slope on r30 is significant"),
        (all(v(k, "exp", "alpha_p") >= 0.05 for k in HAR4), "no HAR-set alpha is significant"),
        (all(v(k, "exp", "alpha_p") < 0.05 for k in FULL), "every full-set alpha is significant (mid)"),
        (v(CI, "span", "grs_p") < 0.05 <= v(SHORT, "span", "grs_p"), "GRS rejects for CI-HAR, not for the short"),
        (
            all(v(k, "span", "dsr_p", c) < v(k, "span", "grs_p", c) for k in rows4 for c in ("mid", "x")),
            "the bootstrap p of the max-SR gain lies below the GRS p",
        ),
        (all(v(k, "span", "grs_p") >= 0.05 for k in HAR4), "the GRS p is above 5 % for every HAR-set forecast"),
        (all(v(k, "span", "share_s") >= 0.98 for k in FULL), "the tangency is almost all strategy (share >= 98 %)"),
        (top < R[(FULL[0], "mid")]["rt"]["first"], "the largest best day falls in the burn-in"),
        (
            all(v(k, "rt", "share_min") >= CAP - 2e-3 for k in FULL),
            "the MV rule holds every full-set strategy at the upper cap on every evaluation day (mid)",
        ),
        (
            not any(v(k, "rt", "mv_dsr") > 0 and v(k, "rt", "mv_p") < 0.05 for k in FULL),
            "no real-time MV combination is significant at mid",
        ),
        (
            not any(
                v(k, "rt", f"{nm}_dsr", "x") > 0 and v(k, "rt", f"{nm}_p", "x") < 0.05
                for k in FULL
                for nm in ("mv", "iv")
            ),
            "no real-time combination is significant at bid/ask",
        ),
        (
            not any(
                v(k, "ex_rt", f"{nm}_dsr") > 0 and v(k, "ex_rt", f"{nm}_p") < 0.05 for k in FULL for nm in ("mv", "iv")
            ),
            "no real-time combination is significant without the best days",
        ),
        (all(v(k, "ex_exp", "beta_p") >= 0.05 for k in FULL), "without the best days the betas stay insignificant"),
        (
            v(SHORT, "ex_exp", "alpha") > max(v(SHORT, "exp", "alpha"), *[v(k, "ex_exp", "alpha") for k in FULL]),
            "without the best days the short straddle's alpha rises above the full-set strategies'",
        ),
    ]
    # 2026-10-26 (D19): the real-time claims above use the Ledoit-Wolf p (nm_p); the registered bootstrap p (nm_pb) is
    # printed as a check
    for nm in ("mv", "iv"):
        for part in ("rt", "ex_rt"):
            for c in ("mid", "x"):
                n_lw = sum(v(k, part, f"{nm}_dsr", c) > 0 and v(k, part, f"{nm}_p", c) < 0.05 for k in FULL)
                n_bt = sum(v(k, part, f"{nm}_dsr", c) > 0 and v(k, part, f"{nm}_pb", c) < 0.05 for k in FULL)
                print(f"  D19 check: {part} {nm} {c}: significant gains, Ledoit-Wolf {n_lw} / bootstrap {n_bt} (of 6)")
    reported = {"GRS rejects for CI-HAR, not for the short"}  # 2026-10-26 (D2/D3): reported, not asserted
    bad = [c for ok, c in claims if not ok and c not in reported]
    for ok, c in claims:
        print(f"  draft claim {'holds' if ok else 'FAILS'}{' (reported only)' if c in reported else ''}: {c}")
    assert not bad, f"DRAFT CLAIMS no longer hold: {bad}"


def check_macros_used() -> None:
    """Every \\rheb... name in the table, the figure float and the draft is defined in the numbers file."""
    used: set[str] = set()
    for f in (TABLE, FIG_TEX, DRAFT):
        if f.exists():
            used |= set(re.findall(r"\\(rheb[A-Za-z]+)", f.read_text(encoding="utf-8")))
    missing = sorted(used - set(MACROS))
    assert not missing, f"undefined \\rheb macros: {missing}"
    print(f"  {len(used)} \\rheb macros used by the table, the figure float and the draft: all defined")


# ============================================================================================ CSV
def stats_csv(R: dict) -> pd.DataFrame:
    rows = []
    for (k, col), r in R.items():
        for part in ("exp", "span", "rt", "exp30", "ex_exp", "ex_span", "ex_rt"):
            for f, v in r[part].items():
                if isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool):
                    rows.append({"key": k, "price": col, "part": part, "stat": f, "value": float(v)})
    return pd.DataFrame(rows)


# ============================================================================================ main
def main() -> None:
    _check_f_sf()
    series, days, pw = load_strategies()
    gate_strategies(series)
    mk = load_market(days, pw)
    drop, top = best_union(series, pw)
    print(
        f"market: {len(pw)} option days, mean excess {100 * mk['rm'].mean():.4f} %/day, sd {100 * mk['rm'].std(ddof=1):.3f}"
        f" %, rf mean {100 * mk['rf'].mean():.5f} %/day; all {mk['n_all']} trading days of the span"
    )
    R = run_all(series, mk, pw, drop)
    draft_claims(R, top)
    em.write(NUMBERS, numbers(R, mk, drop, top, pw))
    em.write(TABLE, table_tex(R))
    figure(R)
    em.write(FIG_TEX, fig_float_tex())
    check_macros_used()
    OUT.mkdir(parents=True, exist_ok=True)
    df = stats_csv(R)
    df.to_csv(OUT / "stats.csv", index=False, float_format="%.10g")
    print("wrote", fr.rel(OUT / "stats.csv"), len(df), "rows")

    # console summary
    print(
        f"\nmarket SR std {R[(FULL[0], 'mid')]['span']['sr_m']:.3f}  D6 {R[(FULL[0], 'mid')]['span']['d6_m']:.3f}  "
        f"all-days std {srq(mk['rm_all'], mk['q_all']):.3f}  RT-days std {R[(FULL[0], 'mid')]['rt']['sr_m']:.3f}"
    )
    hdr = (
        f"{'row':30s} {'pr':3s} {'beta':>6s} {'t':>6s} {'corr':>6s} {'gam':>7s} {'t':>6s} {'alpha':>6s} {'t':>6s} "
        f"{'AR':>5s} {'SRs':>5s} {'SRmax':>6s} {'pB':>6s} {'pGRS':>6s} {'shr':>4s} {'MV':>5s} {'p':>6s} {'IV':>5s} "
        f"{'p':>6s} | {'exA':>6s} {'t':>6s} {'exMax':>6s} {'exMV':>6s} {'p':>6s}"
    )
    print(hdr)
    for k in ROWS:
        for col, _ in PRICES:
            r = R[(k, col)]
            e, s, rt, ee, es, ert = r["exp"], r["span"], r["rt"], r["ex_exp"], r["ex_span"], r["ex_rt"]
            print(
                f"{k:30s} {col:3s} {e['beta']:6.2f} {e['beta_t']:6.2f} {e['corr']:6.3f} {e['tm_g']:7.2f} "
                f"{e['tm_gt']:6.2f} {e['alpha']:6.2f} {e['alpha_t']:6.2f} {s['ar']:5.2f} {s['sr_s']:5.2f} "
                f"{s['sr_max']:6.2f} {s['dsr_p']:6.3f} {s['grs_p']:6.3f} {100 * s['share_s']:4.0f} {rt['mv_sr']:5.2f} "
                f"{rt['mv_p']:6.3f} {rt['iv_sr']:5.2f} {rt['iv_p']:6.3f} | {ee['alpha']:6.2f} {ee['alpha_t']:6.2f} "
                f"{es['sr_max']:6.2f} {ert['mv_sr']:6.2f} {ert['mv_p']:6.3f}"
            )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preview", type=Path, default=None, help="directory for a PNG preview of the figure")
    PREVIEW = ap.parse_args().preview
    main()
