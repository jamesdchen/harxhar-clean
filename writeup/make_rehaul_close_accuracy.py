"""Closing-half-hour accuracy on both evaluation spans, and calibration diagnostics of the 12 pooled forecasts.

2026-10-25.  Two pieces the paper lacks: (1) the closing-half-hour (16:00 bar) accuracy table of the pooled forecasts
on both evaluation spans (Table 1's 2014-2024 span and T_oos 2020-2024, whose numbers Section 6 gives in the text
only), and (2) the calibration diagnostics promised by note E33 (05_methods.tex), used to explain why the full
predictor set lowers QLIKE but raises the MSE and cuts R2_OOS.

FORECASTS  the 12 pooled forecasts of the final rerun (writeup/final_run.py, REHAUL_FINAL_RUN=1: ridge, lasso, elastic
           net, LightGBM, XGBoost, random forest x {HAR benchmark, full set}) and the pooled OLS (Table 1's reference
           row), read with Table 1's own loaders and gates (make_rehaul_oos_rerun.forecasts / tree_forecasts /
           scored_set: the 33,527 bars of the 2,579 days 2014-01-02 .. 2024-04-30, the same bars in every arm).  The
           16:00 bar of each day is the 15:30 forecast of the closing half hour (pooled_linear_13.at16 /
           pooled_trees_13.at16, asserted equal).
           The closing-interval HAR benchmark (sub_ols_baseline) and the models trained on the closing half hour only
           (Appendix sec:rh_econ_last30, Table tab:rh_oos_last30: subpt_<est>_<set>, subtree_daily_<set>_<est>) through
           experiments/rolling_eval_2026_10_15.scored on its common span (T_oos, 1,082 days).  Their first forecast is
           25 June 2018 (a 2,000-day window from July 2010), so they have no forecast on 2014-2019 and the gap to the
           closing-interval HAR benchmark exists on T_oos only.
SPANS      Span 2014-01-02 .. 2024-04-30 (2,579 days); Pre 2014-01-02 .. 2019-12-31 (1,496); Oos (T_oos)
           2020-01-03 .. 2024-04-30 (1,082) -- Table 1's subperiods (final_run.SUB_A / SUB_B; 2 January 2020 is in
           neither, as there).  Oos is asserted equal to rolling.common_span().
SCORING    as Table 1 and Section 6: sigma2 = plain (pred_adj^2 B), truth the core_stats print (asserted);
           QLIKE = mean of r - ln r - 1, r = RV / sigma2; MSE and MAE in variance units; R2_OOS (Campbell-Thompson) with
           the expanding historical mean of the same bar from 5 January 1998 (make_rehaul_oos_rerun.slot_benchmark; at
           16:00 it equals rolling.ct_benchmark, asserted).  Delta% = 100 (Lbar_A / Lbar_B - 1).
TESTS      src.evaluation.diebold_mariano.dm_test (HLN, Newey-West at its automatic lag: 8 on 2,579 days, 7 on 1,496,
           6 on 1,082) on the daily loss: the 16:00 loss of each day, or the daily average over the 13 bars (Table 1).
CALIBRATION  per pooled forecast, on every bar and at 16:00, Pre and Oos (Span in the CSV):
           mean of r = RV / sigma2; ratio of means RVbar / sigma2bar (note E33); under-prediction rate share(RV > sigma2);
           MAE; Mincer-Zarnowitz RV = a + b sigma2 and ln RV = a + b ln sigma2 by OLS with Newey-West (Bartlett, 1/T)
           standard errors -- lag = the DM lag of the sample's day count at 16:00, 13 x that lag on every bar (the
           same calendar reach); t-tests of a = 0 and b = 1 and the Wald test of (a, b) = (0, 1).
           Two exact decompositions:
             Theil:  MSE = (RVbar - sigma2bar)^2 + (1 - b)^2 Var(sigma2) + (1 - R2_MZ) Var(RV)   (bias, slope, disturbance;
                     population moments, b and R2_MZ of the levels MZ regression);
             QLIKE:  QLIKE = [rbar - 1 - ln rbar] + [ln rbar - mean ln r]   (level: the part a constant rescaling of the
                     forecast removes, zero iff rbar = 1; dispersion: invariant to that rescaling).
EXTREMES   every bar, Oos: the share of the full-minus-HAR squared-error difference carried by the 5 days with the
           largest daily difference; the share of each forecast's squared error from over-predictions; the MSE change
           without 24 February .. 30 April 2020 (make_rehaul_results.CRASH); the largest linear full-set forecast.
DECILES    mean RV against mean forecast by forecast decile (elastic net and LightGBM, HAR and full set), every bar and
           16:00, Pre and Oos: the figure, and deciles.csv.
GATES      (stop on any mismatch) Table 1 (rehaul_results_oos.tex cells, results/oos_final_2014start/table1_w3000/
           metrics.csv and subperiods.csv, the \\rhorDm...B macros); the closing-half-hour numbers of Section 6 on T_oos
           (\\rhrQ<M>Pool<S>, \\rhrAf<M>Pool{Pct,Dm,Pval}, \\rhrVsBm<M>Pool<S>{Pct,Dm,Pval}, \\rhrOlsQ); the 16:00 numbers of
           Table tab:rh_robust_w3500 at W = 3,000 (\\rhrbQ...CloseMain, \\rhrbPctF / \\rhrbDMF ...CloseMain, its
           metrics.csv); Table tab:rh_oos_last30 (cells) and \\rhrAf<M>Close{Pct,Dm,Pval}.

WRITES     writeup/generated/rehaul_close_accuracy_numbers.tex   \\rhca... macros (macro-only)
           writeup/generated/rehaul_close_accuracy_table.tex     Table tab:rh_close_accuracy
           writeup/generated/rehaul_calibration_table.tex        Table tab:rh_calibration
           writeup/generated/rehaul_calibration.pdf              the decile calibration figure
           writeup/generated/rehaul_calibration_fig.tex          its float (fig:rh_calibration)
           results/rehaul_close_accuracy/                        accuracy16.csv, accuracy_bar.csv, calibration.csv,
                                                                 deciles.csv, extremes.csv, yearly16.csv, gates.csv
           (final_run.GEN and final_run.res_dir: REHAUL_GEN_DIR / REHAUL_RES_DIR redirect a test run.)

Run:  REHAUL_FINAL_RUN=1 python writeup/make_rehaul_close_accuracy.py   (from the repository root; the switch is set
      to 1 when absent and must not be off)
"""

from __future__ import annotations

import math
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("REHAUL_FINAL_RUN", "1")

ROOT = Path(__file__).resolve().parent  # writeup/
REPO = ROOT.parent
for _p in (REPO, REPO / "notebooks", REPO / "experiments", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import final_run as fr  # noqa: E402
import make_rehaul_oos_rerun as oos  # noqa: E402  (Table 1's loaders, scored set, benchmark, loss)
import pooled_linear_13 as pl13  # noqa: E402  (the 16:00 rows Section 6 reads)
import pooled_trees_13 as pt13  # noqa: E402
import rolling_eval_2026_10_15 as rolling  # noqa: E402  (T_oos and the closing-interval forecasts)
from src.evaluation.diebold_mariano import _nw_lag, dm_test  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator  # noqa: E402

assert fr.FINAL, "REHAUL_FINAL_RUN is off: this generator reads the final pooled rerun only"
assert not fr.PENDING_TREES, f"a pooled tree is pending ({fr.PENDING_TREES}): no table without its rows"

GEN = fr.GEN  # outputs
SRC_GEN = ROOT / "generated"  # the printed numbers the gates reproduce (never redirected)
OUT_NUM = GEN / "rehaul_close_accuracy_numbers.tex"
OUT_ACC = GEN / "rehaul_close_accuracy_table.tex"
OUT_CAL = GEN / "rehaul_calibration_table.tex"
OUT_FIG = GEN / "rehaul_calibration.pdf"
OUT_FIGTEX = GEN / "rehaul_calibration_fig.tex"
RES = fr.res_dir(REPO / "results" / "rehaul_close_accuracy")
HEAD = "% AUTO-GENERATED by writeup/make_rehaul_close_accuracy.py -- do not edit; re-run the script."
T1_CSV = REPO / "results" / "oos_final_2014start" / f"table1_w{fr.WINDOW_DAYS}"
RB_CSV = REPO / "results" / "rehaul_robust_window_final" / "metrics.csv"

MSE_EXP = 10  # MSE printed x 10^MSE_EXP (Tables 1 and 2, \rhoosMseExp)
UNIT_EXP = 6  # MAE and the levels intercept printed x 10^UNIT_EXP
SIG = 0.05
LEVELS = (0.10, 0.05, 0.01)  # star levels (make_rehaul_results.LEVELS)
CRASH = (pd.Timestamp("2020-02-24"), pd.Timestamp("2020-04-30"))  # make_rehaul_results.CRASH
TOP_DAYS = 5
N_DEC = 10
SPANS = {  # tag -> first, last, days
    "Span": (fr.FIRST_DAY, fr.LAST_DAY, 2579),
    "Pre": (fr.SUB_A[0], fr.SUB_A[1], 1496),
    "Oos": (fr.SUB_B[0], fr.SUB_B[1], 1082),
}
EST = ("ridge", "lasso", "enet", "lgbm", "xgb", "rf")
LIN, TREE = EST[:3], EST[3:]
SETS = ("baseline", "all_features")
NAME = {
    "ols": "OLS",
    "ridge": "Ridge",
    "lasso": "Lasso",
    "enet": "Elastic net",
    "lgbm": "LightGBM",
    "xgb": "XGBoost",
    "rf": "Random forest",
}
MTOK = {"ols": "Ols", "ridge": "Ridge", "lasso": "Lasso", "enet": "Enet", "lgbm": "Lgbm", "xgb": "Xgb", "rf": "Rf"}
STOK = {"baseline": "Har", "all_features": "Full"}
SETNAME = {"baseline": "HAR", "all_features": "Full"}
ARMS = [("ols", "baseline")] + [(e, s) for e in EST for s in SETS]
POOL12 = [(e, s) for e in EST for s in SETS]
CLOSE_KEY = {("ols", "baseline"): rolling.OLS} | {
    (e, s): (f"subpt_{e}_{s}" if e in LIN else f"subtree_daily_{s}_{e}") for e in EST for s in SETS
}
WORD = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
COL_EN, COL_GB = "#2a78d6", "#eb6834"  # dataviz reference palette, slots 1-2 (validated: CVD dE 24.7, contrast >= 3:1)


def key(e: str, s: str) -> str:
    return f"{e}_{s}"


# --------------------------------------------------------------------------- formatting (make_rehaul_results helpers)


def n_stars(p: float) -> int:
    return int(sum(p < a for a in LEVELS)) if np.isfinite(p) else 0


def _fmt(x: float, dp: int) -> str:
    v = round(float(x), dp)
    if v == 0:
        v = 0.0
    return f"{'-' if v < 0 else ''}{abs(v):.{dp}f}"


def cell(x: float | None, dp: int = 2, stars: int = 0, starred: bool = True) -> str:
    """An estimate in math mode, padded to three star slots when the column is starred (make_rehaul_results.cell)."""
    if x is None or not np.isfinite(x):
        return "--$^{\\phantom{***}}$" if starred else "--"
    body = _fmt(x, dp)
    if not starred:
        return f"${body}$"
    pad = "*" * (3 - stars)
    sup = "*" * stars + (f"\\phantom{{{pad}}}" if pad else "")
    return f"${body}^{{{sup}}}$"


def stat_col(t: float | None, dp: int = 2) -> str:
    if t is None or not np.isfinite(t):
        return "--"
    return f"$({_fmt(t, dp)})$"


def mnum(x: float, dp: int) -> str:
    v = round(float(x), dp)
    if v == 0:
        v = 0.0
    return ("\\ensuremath{-}" if v < 0 else "") + f"{abs(v):.{dp}f}"


def mpval(p: float) -> str:
    """The whole p statement (make_rehaul_results.mpval)."""
    if p < 0.001:
        return "\\ensuremath{p < 0.001}"
    s = f"{p:.2f}" if p >= 0.10 else f"{p:.3f}"
    if s in ("0.10", "0.100", "0.050", "0.010"):
        s = f"{p:.4f}"
    return f"\\ensuremath{{p = {s}}}"


def pct_dp(x: float) -> int:
    """Decimals of a printed % change: 1, or 2 when one decimal would print a nonzero change as 0.0."""
    return 2 if (x != 0 and abs(x) < 0.05) else 1


def pct_cell(x: float, p: float) -> str:
    """A starred % change; a nonzero change that rounds to zero keeps its sign ($-0.00$)."""
    c = cell(x, pct_dp(x), n_stars(p))
    if x < 0 and round(x, pct_dp(x)) == 0:
        c = c.replace("$0.00", "$-0.00", 1)
    return c


def thousands(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def long_date(t: pd.Timestamp) -> str:
    return f"{t.day} {t.strftime('%B')} {t.year}"


# --------------------------------------------------------------------------- statistics


def qlike(y: np.ndarray, f: np.ndarray) -> np.ndarray:
    r = y / f
    return r - np.log(r) - 1.0


def ols_hac(yv: np.ndarray, xv: np.ndarray, lag: int) -> dict:
    """OLS of y on [1, x]; Newey-West (Bartlett, divisor T) covariance; t-tests a = 0 and b = 1, Wald (a, b) = (0, 1)."""
    T = len(yv)
    X = np.column_stack([np.ones(T), xv])
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    u = yv - X @ beta
    g = X * u[:, None]
    S = g.T @ g / T
    for j in range(1, lag + 1):
        G = g[j:].T @ g[:-j] / T
        S += (1.0 - j / (lag + 1.0)) * (G + G.T)
    Qi = np.linalg.inv(X.T @ X / T)
    V = Qi @ S @ Qi / T
    se = np.sqrt(np.diag(V))
    d = beta - np.array([0.0, 1.0])
    W = float(d @ np.linalg.solve(V, d))
    r2 = 1.0 - float(u @ u) / float(((yv - yv.mean()) ** 2).sum())
    ta, tb = beta[0] / se[0], (beta[1] - 1.0) / se[1]
    return {
        "a": float(beta[0]),
        "b": float(beta[1]),
        "se_a": float(se[0]),
        "se_b": float(se[1]),
        "p_a": math.erfc(abs(ta) / math.sqrt(2.0)),
        "p_b": math.erfc(abs(tb) / math.sqrt(2.0)),
        "wald": W,
        "p_wald": math.exp(-W / 2.0),  # chi-square, 2 degrees of freedom
        "r2": r2,
    }


def calib(y: np.ndarray, f: np.ndarray, lag: int) -> dict:
    """The calibration diagnostics of one forecast on one sample (module docstring, CALIBRATION)."""
    r = y / f
    e = y - f
    mse = float((e**2).mean())
    rbar = float(r.mean())
    lr = float(np.log(r).mean())
    q = float(qlike(y, f).mean())
    vf, vy = float(f.var()), float(y.var())
    b_th = float(((f - f.mean()) * (y - y.mean())).mean() / vf)
    r2_th = float(np.corrcoef(y, f)[0, 1] ** 2)
    bias2 = float((y.mean() - f.mean()) ** 2)
    slope = (1.0 - b_th) ** 2 * vf
    dist = (1.0 - r2_th) * vy
    assert np.isclose(bias2 + slope + dist, mse, rtol=1e-8), (bias2, slope, dist, mse)
    q_lev = rbar - 1.0 - math.log(rbar)
    q_disp = math.log(rbar) - lr
    assert np.isclose(q_lev + q_disp, q, rtol=1e-10), (q_lev, q_disp, q)
    lev = ols_hac(y, f, lag)
    lg = ols_hac(np.log(y), np.log(f), lag)
    assert np.isclose(lev["b"], b_th, rtol=1e-8) and np.isclose(lev["r2"], r2_th, rtol=1e-8)
    out = {
        "n": len(y),
        "hac_lag": lag,
        "qlike": q,
        "mse": mse,
        "mae": float(np.abs(e).mean()),
        "mean_r": rbar,
        "ratio_means": float(y.mean() / f.mean()),
        "under": float((y > f).mean()),
        "mean_log_r": lr,
        "q_level": q_lev,
        "q_disp": q_disp,
        "theil_bias": bias2,
        "theil_slope": slope,
        "theil_dist": dist,
        "sh_bias": 100.0 * bias2 / mse,
        "sh_slope": 100.0 * slope / mse,
        "sh_dist": 100.0 * dist / mse,
        "over_se_share": 100.0 * float((e[f > y] ** 2).sum()) / float((e**2).sum()),
    }
    out.update({f"mz_{k}": v for k, v in lev.items()})
    out.update({f"mzl_{k}": v for k, v in lg.items()})
    return out


# --------------------------------------------------------------------------- data


def load() -> dict:
    """The pooled forecasts on Table 1's scored bars (its loaders and gates), their 16:00 rows, and the closing-interval
    forecasts on T_oos."""
    cfg = oos.SPANS["2014"]
    rv = oos.core()
    days = oos.evaluation_days(rv, cfg["first"], cfg["last"], cfg["n_days"])
    fc = oos.forecasts("13", cfg["r13"])
    ft = oos.tree_forecasts(cfg)
    idx = oos.scored_set(fc, rv, days)
    assert oos.scored_set({**fc, **ft}, rv, days).equals(idx), "the trees miss a bar the linear rows score"
    fa = {**fc, **ft}
    assert set(fa) == set(ARMS), sorted(fa)
    P = pd.DataFrame({key(*k): fa[k]["plain"].reindex(idx).to_numpy(float) for k in ARMS}, index=idx)
    y = pd.Series(rv.reindex(idx).to_numpy(float), index=idx)
    assert np.isfinite(P.to_numpy()).all() and (P.to_numpy() > 0).all() and (y > 0).all()
    assert len(idx) == 13 * len(days), (len(idx), len(days))  # every evaluation day has its 13 bars
    print(f"pooled: {len(idx):,} bars on {len(days):,} days; slot benchmark ...", flush=True)
    bench = pd.Series(oos.slot_benchmark(rv, idx).to_numpy(float), index=idx)
    # 16:00 rows, by day
    i16 = idx[(idx.hour == 16) & (idx.minute == 0)]
    d16 = pd.DatetimeIndex(i16.normalize())
    assert d16.equals(days), "a day without its 16:00 bar"
    P16 = pd.DataFrame(P.loc[i16].to_numpy(), index=d16, columns=P.columns)
    y16 = pd.Series(y.loc[i16].to_numpy(), index=d16)
    b16 = pd.Series(bench.loc[i16].to_numpy(), index=d16)
    # T_oos: the common rolling span of Section 6 (asserted = the Oos cut of the 2014 span)
    doos = rolling.common_span()
    f0, l0, n0 = SPANS["Oos"]
    assert doos.equals(days[(days >= f0) & (days <= l0)]) and len(doos) == n0, (len(doos), doos[0], doos[-1])
    hist = rolling.history()
    assert np.array_equal(hist.reindex(days).to_numpy(), y16.to_numpy()), "core_stats 16:00 RV != rolling.history"
    ct = rolling.ct_benchmark(hist, days)
    assert np.allclose(ct.to_numpy(), b16.to_numpy(), rtol=1e-10, atol=0), "16:00 benchmark != rolling.ct_benchmark"
    # the 16:00 rows are the rows Section 6 reads (pooled_linear_13.at16 / pooled_trees_13.at16)
    for e, s in ARMS:
        a = (pl13 if e in pl13.EST or e == "ols" else pt13).at16(e, s)
        assert np.array_equal(a["plain"].reindex(days).to_numpy(), P16[key(e, s)].to_numpy()), (e, s)
        assert np.array_equal(a["rv"].reindex(days).to_numpy(), y16.to_numpy()), (e, s)
    # closing-interval forecasts on T_oos
    C = {}
    for k, ck in CLOSE_KEY.items():
        g = rolling.scored(ck)
        assert doos.isin(g.index).all(), ck
        C[key(*k)] = g["plain"].reindex(doos).to_numpy(float)
        assert np.array_equal(g["rv"].reindex(doos).to_numpy(), y16.reindex(doos).to_numpy()), ck
    C16 = pd.DataFrame(C, index=doos)
    assert np.isfinite(C16.to_numpy()).all() and (C16.to_numpy() > 0).all()
    first_close = min(rolling.scored(ck).index.min() for ck in CLOSE_KEY.values())
    return dict(
        rv=rv,
        days=days,
        idx=idx,
        P=P,
        y=y,
        bench=bench,
        P16=P16,
        y16=y16,
        b16=b16,
        doos=doos,
        C16=C16,
        first_close=first_close,
    )


def span_days(days: pd.DatetimeIndex, tag: str) -> pd.DatetimeIndex:
    f, la, n = SPANS[tag]
    d = days[(days >= f) & (days <= la)]
    assert len(d) == n, (tag, len(d))
    return d


# --------------------------------------------------------------------------- gates


def read_macros(path: Path) -> dict[str, str]:
    """name -> value of every \\providecommand / \\renewcommand / \\newcommand in a numbers file (balanced braces)."""
    txt = path.read_text(encoding="utf-8")
    out: dict[str, str] = {}
    for m in re.finditer(r"\\(?:renew|provide|new)command\{\\([A-Za-z]+)\}", txt):
        i = m.end()
        if txt[i] != "{":
            continue
        depth, j = 0, i
        while True:
            c = txt[j]
            depth += c == "{"
            depth -= c == "}"
            j += 1
            if depth == 0:
                break
        val = txt[i + 1 : j - 1]
        if val or m.group(1) not in out:
            out[m.group(1)] = val
    return out


def table_rows(path: Path) -> dict[str, list[str]]:
    """First cell -> the other cells of every body row of a generated table."""
    out = {}
    body = path.read_text(encoding="utf-8").split("\\midrule", 1)[1].split("\\bottomrule", 1)[0]
    for line in body.splitlines():
        if " & " in line and not line.lstrip().startswith("\\multicolumn"):
            cells = [c.strip() for c in line.split("\\\\")[0].split(" & ")]
            out[cells[0]] = cells[1:]
    return out


GATES: list[dict] = []


def gate(name: str, item: str, got: object, want: object, ok: bool | None = None) -> None:
    ok = (got == want) if ok is None else ok
    GATES.append(dict(gate=name, item=item, got=str(got), want=str(want), ok=bool(ok)))


def run_gates(D: dict, A16: pd.DataFrame, ABar: pd.DataFrame) -> None:
    """Reproduce the printed values (module docstring, GATES); stop on any mismatch."""
    mor = read_macros(SRC_GEN / "rehaul_oos_rerun_numbers.tex")
    mrr = read_macros(SRC_GEN / "rehaul_results_numbers.tex")
    mrb = read_macros(SRC_GEN / "rehaul_robust_w3500_numbers.tex")
    t1 = table_rows(SRC_GEN / "rehaul_results_oos.tex")
    t2 = table_rows(SRC_GEN / "rehaul_results_oos_last30.tex")
    # --- Table 1: every bar, 2014-2024
    met = pd.read_csv(T1_CSV / "metrics.csv")
    sub = pd.read_csv(T1_CSV / "subperiods.csv")
    for e, s in ARMS:
        r = ABar[(ABar.model == e) & (ABar.set == s) & (ABar.span == "Span")].iloc[0]
        m = met[(met.model == e) & (met.set == s)].iloc[0]
        for c in ("qlike", "mse", "r2_oos_pct"):
            gate("T1 metrics.csv", f"{e} {s} {c}", r[c], m[c], bool(np.isclose(r[c], m[c], rtol=1e-9)))
        row = t1[NAME[e]]
        cells = row[0:3] if s == "baseline" else row[3:6]
        want = [oos.fmt(r.mse * 10.0**MSE_EXP, 2), f"{r.qlike:.4f}", oos.fmt(r.r2_oos_pct, 1)]
        gate("T1 printed", f"{e} {s}", want, cells)
        for tag in ("A", "B"):
            sp = "Pre" if tag == "A" else "Oos"
            rs = ABar[(ABar.model == e) & (ABar.set == s) & (ABar.span == sp)].iloc[0]
            ms = sub[(sub.model == e) & (sub.set == s) & (sub.tag == tag) & sub.against.isna()].iloc[0]
            for c in ("qlike", "mse", "r2_oos_pct"):
                gate("T1 subperiods.csv", f"{e} {s} {tag} {c}", rs[c], ms[c], bool(np.isclose(rs[c], ms[c], rtol=1e-9)))
    for e in EST:
        r = ABar[(ABar.model == e) & (ABar.set == "all_features") & (ABar.span == "Span")].iloc[0]
        gate("T1 DM", e, oos.fmt(r.dm, 2), mor[f"rhorDm{MTOK[e]}B"])
        gate("T1 DM printed", e, f"{oos.fmt(r.dm, 2)} ({oos.fmt_p(r.p)})", t1[NAME[e]][6])
    # --- Section 6, closing half hour on T_oos
    gate("rhrOlsQ", "bm", f"{A16[(A16.scheme == 'close') & (A16.model == 'ols')].iloc[0].qlike:.4f}", mrr["rhrOlsQ"])
    for e in EST:
        for s in SETS:
            r = A16[(A16.scheme == "pool") & (A16.model == e) & (A16.set == s) & (A16.span == "Oos")].iloc[0]
            gate("rhrQ", f"{e} {s}", f"{r.qlike:.4f}", mrr[f"rhrQ{MTOK[e]}Pool{STOK[s]}"])
            gate("rhrVsBm Pct", f"{e} {s}", mnum(r.gap, 1), mrr[f"rhrVsBm{MTOK[e]}Pool{STOK[s]}Pct"])
            gate("rhrVsBm Dm", f"{e} {s}", mnum(r.gap_dm, 2), mrr[f"rhrVsBm{MTOK[e]}Pool{STOK[s]}Dm"])
            gate("rhrVsBm Pval", f"{e} {s}", mpval(r.gap_p), mrr[f"rhrVsBm{MTOK[e]}Pool{STOK[s]}Pval"])
        r = A16[(A16.scheme == "pool") & (A16.model == e) & (A16.set == "all_features") & (A16.span == "Oos")].iloc[0]
        gate("rhrAf Pct", e, mnum(r.pct, 1), mrr[f"rhrAf{MTOK[e]}PoolPct"])
        gate("rhrAf Dm", e, mnum(r.dm, 2), mrr[f"rhrAf{MTOK[e]}PoolDm"])
        gate("rhrAf Pval", e, mpval(r.p), mrr[f"rhrAf{MTOK[e]}PoolPval"])
        # --- the closing-interval models (Table 2, \rhrAf<M>Close...)
        rc = A16[(A16.scheme == "close") & (A16.model == e) & (A16.set == "all_features")].iloc[0]
        gate("rhrAf Close Pct", e, mnum(rc.pct, 1), mrr[f"rhrAf{MTOK[e]}ClosePct"])
        gate("rhrAf Close Dm", e, mnum(rc.dm, 2), mrr[f"rhrAf{MTOK[e]}CloseDm"])
        # the printed p of rhrAfXgbClosePval carries 4 decimals (a 10 % cut); compare numerically
        pw = float(re.findall(r"[0-9.]+", mrr[f"rhrAf{MTOK[e]}ClosePval"])[-1])
        gate("rhrAf Close Pval", e, rc.p, pw, bool(abs(rc.p - pw) < 0.5 * 10 ** -(len(str(pw).split(".")[1]))))
    t2name = {"ols": "OLS (benchmark forecast)"} | {e: NAME[e] for e in EST}
    for e, s in ARMS:
        rc = A16[(A16.scheme == "close") & (A16.model == e) & (A16.set == s)].iloc[0]
        row = t2[t2name[e]]
        cells = row[0:3] if s == "baseline" else row[3:6]
        want = [oos.fmt(rc.mse * 10.0**MSE_EXP, 2), f"{rc.qlike:.4f}", oos.fmt(rc.r2_oos_pct, 1)]
        gate("T2 printed", f"{e} {s}", want, cells)
    # --- Table tab:rh_robust_w3500, 16:00 at W = 3,000 (2014-2024)
    rb = pd.read_csv(RB_CSV)
    rb = rb[rb.W == fr.WINDOW_DAYS]
    for e, s in ARMS:
        r = A16[(A16.scheme == "pool") & (A16.model == e) & (A16.set == s) & (A16.span == "Span")].iloc[0]
        m = rb[(rb.model == e) & (rb.set == s)].iloc[0]
        gate(
            "robust metrics.csv", f"{e} {s}", r.qlike, m.qlike_1600, bool(np.isclose(r.qlike, m.qlike_1600, rtol=1e-10))
        )
        gate("rhrbQ", f"{e} {s}", f"{r.qlike:.4f}", mrb[f"rhrbQ{MTOK[e]}{STOK[s]}CloseMain"])
    for e in EST:
        r = A16[(A16.scheme == "pool") & (A16.model == e) & (A16.set == "all_features") & (A16.span == "Span")].iloc[0]
        gate("rhrbPctF", e, mnum(r.pct, 1), mrb[f"rhrbPctF{MTOK[e]}CloseMain"])
        gate("rhrbDMF", e, mnum(r.dm, 2), mrb[f"rhrbDMF{MTOK[e]}CloseMain"])
    g = pd.DataFrame(GATES)
    RES.mkdir(parents=True, exist_ok=True)
    g.to_csv(RES / "gates.csv", index=False)
    bad = g[~g.ok]
    print(f"gates: {len(g)} checks, {len(bad)} failed")
    if len(bad):
        print(bad.to_string())
        raise SystemExit("GATE FAILED: the printed values are not reproduced; nothing written")


# --------------------------------------------------------------------------- accuracy


def accuracy16(D: dict) -> pd.DataFrame:
    """Every 16:00 number of the accuracy table: pooled forecasts on Span / Pre / Oos, the closing-interval forecasts on
    Oos; Full vs HAR within each class and each forecast vs the closing-interval HAR benchmark (Oos)."""
    rows = []
    days, P16, y16, b16, C16 = D["days"], D["P16"], D["y16"], D["b16"], D["C16"]
    for scheme in ("pool", "close"):
        for tag in SPANS:
            if scheme == "close" and tag != "Oos":
                continue
            d = span_days(days, tag)
            y, bn = y16.reindex(d).to_numpy(), b16.reindex(d).to_numpy()
            den = float(((y - bn) ** 2).sum())
            F = P16.reindex(d) if scheme == "pool" else C16.reindex(d)
            arms = ARMS if scheme == "pool" else list(CLOSE_KEY)
            L = {k: qlike(y, F[key(*k)].to_numpy()) for k in arms}
            SE = {k: (y - F[key(*k)].to_numpy()) ** 2 for k in arms}
            qb = qlike(y, C16[key("ols", "baseline")].reindex(d).to_numpy()) if tag == "Oos" else None
            for k in arms:
                e, s = k
                r = dict(
                    scheme=scheme,
                    span=tag,
                    model=e,
                    set=s,
                    n_days=len(d),
                    qlike=float(L[k].mean()),
                    mse=float(SE[k].mean()),
                    mae=float(np.sqrt(SE[k]).mean()),
                    r2_oos_pct=float(100.0 * (1.0 - SE[k].sum() / den)),
                )
                if s == "all_features":
                    h = (e, "baseline")
                    t = dm_test(L[k], L[h])
                    tm = dm_test(SE[k], SE[h])
                    r.update(
                        pct=float(100.0 * (L[k].mean() / L[h].mean() - 1.0)),
                        dm=float(t["dm"]),
                        p=float(t["p"]),
                        hac_lag=int(t["hac_lag"]),
                        mse_pct=float(100.0 * (SE[k].mean() / SE[h].mean() - 1.0)),
                        mse_dm=float(tm["dm"]),
                        mse_p=float(tm["p"]),
                    )
                if qb is not None and not (scheme == "close" and e == "ols"):
                    t = dm_test(L[k], qb)
                    r.update(
                        gap=float(100.0 * (L[k].mean() / qb.mean() - 1.0)), gap_dm=float(t["dm"]), gap_p=float(t["p"])
                    )
                rows.append(r)
    return pd.DataFrame(rows)


def accuracy_bar(D: dict) -> pd.DataFrame:
    """Every-bar QLIKE / MSE / MAE / R2_OOS of each pooled arm on Span / Pre / Oos (Table 1 and its subperiods), with
    the Full vs HAR tests on the daily average QLIKE and squared error (Table 1's test)."""
    rows = []
    idx, P, y, bench = D["idx"], D["P"], D["y"], D["bench"]
    dd = pd.DatetimeIndex(idx.normalize())
    for tag in SPANS:
        f0, l0, _ = SPANS[tag]
        m = (dd >= f0) & (dd <= l0)
        sidx = idx[m]
        yy, bb = y.to_numpy()[m], bench.to_numpy()[m]
        den = float(((yy - bb) ** 2).sum())
        L = {k: qlike(yy, P[key(*k)].to_numpy()[m]) for k in ARMS}
        SE = {k: (yy - P[key(*k)].to_numpy()[m]) ** 2 for k in ARMS}
        for k in ARMS:
            e, s = k
            r = dict(
                span=tag,
                model=e,
                set=s,
                n_bars=int(m.sum()),
                n_days=int(pd.DatetimeIndex(sidx.normalize()).nunique()),
                qlike=float(L[k].mean()),
                mse=float(SE[k].mean()),
                mae=float(np.sqrt(SE[k]).mean()),
                r2_oos_pct=float(100.0 * (1.0 - SE[k].sum() / den)),
            )
            if s == "all_features":
                h = (e, "baseline")
                t = dm_test(oos.daily(L[k], sidx), oos.daily(L[h], sidx))
                tm = dm_test(oos.daily(SE[k], sidx), oos.daily(SE[h], sidx))
                r.update(
                    pct=float(100.0 * (L[k].mean() / L[h].mean() - 1.0)),
                    dm=float(t["dm"]),
                    p=float(t["p"]),
                    hac_lag=int(t["hac_lag"]),
                    mse_pct=float(100.0 * (SE[k].mean() / SE[h].mean() - 1.0)),
                    mse_dm=float(tm["dm"]),
                    mse_p=float(tm["p"]),
                )
            rows.append(r)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- calibration, extremes, deciles


def calibration(D: dict) -> pd.DataFrame:
    rows = []
    idx, P, y = D["idx"], D["P"], D["y"]
    dd = pd.DatetimeIndex(idx.normalize())
    for tag in SPANS:
        f0, l0, n0 = SPANS[tag]
        lag16 = _nw_lag(n0)
        mb = (dd >= f0) & (dd <= l0)
        d16 = span_days(D["days"], tag)
        for sample in ("Bar", "Close"):
            if sample == "Bar":
                yy, FF, lag = y.to_numpy()[mb], P.to_numpy()[mb], 13 * lag16
            else:
                yy, FF, lag = D["y16"].reindex(d16).to_numpy(), D["P16"].reindex(d16).to_numpy(), lag16
            for j, k in enumerate(ARMS):
                assert P.columns[j] == key(*k)
                c = calib(yy, FF[:, j], lag)
                rows.append(dict(sample=sample, span=tag, model=k[0], set=k[1]) | c)
    return pd.DataFrame(rows)


def extremes(D: dict) -> tuple[pd.DataFrame, dict]:
    """Every bar and 16:00 on T_oos (and the Span): where the full set's extra squared error sits."""
    rows, top_days = [], {}
    idx, P, y = D["idx"], D["P"], D["y"]
    dd = pd.DatetimeIndex(idx.normalize())
    for tag in ("Oos", "Span"):
        f0, l0, _ = SPANS[tag]
        m = (dd >= f0) & (dd <= l0)
        ex = m & ~((dd >= CRASH[0]) & (dd <= CRASH[1]))
        sidx, yy = idx[m], y.to_numpy()[m]
        d16 = span_days(D["days"], tag)
        d16x = d16[~((d16 >= CRASH[0]) & (d16 <= CRASH[1]))]
        y16 = D["y16"]
        for e in EST:
            fh, ff = P[key(e, "baseline")].to_numpy(), P[key(e, "all_features")].to_numpy()
            seh, sef = (y.to_numpy() - fh) ** 2, (y.to_numpy() - ff) ** 2
            dse = pd.Series(sef[m] - seh[m], index=pd.DatetimeIndex(sidx.normalize())).groupby(level=0).sum()
            top = dse.sort_values(ascending=False)
            tx = dm_test(oos.daily(sef[ex], idx[ex]), oos.daily(seh[ex], idx[ex]))
            txa = dm_test(oos.daily(np.sqrt(sef[ex]), idx[ex]), oos.daily(np.sqrt(seh[ex]), idx[ex]))
            h16, f16 = D["P16"][key(e, "baseline")], D["P16"][key(e, "all_features")]
            s16h, s16f = (y16 - h16) ** 2, (y16 - f16) ** 2
            tx16 = dm_test(s16f.reindex(d16x).to_numpy(), s16h.reindex(d16x).to_numpy())
            jmax = int(np.argmax(ff[m]))
            rows.append(
                dict(
                    span=tag,
                    model=e,
                    dse_total=float(dse.sum()),
                    top_share_pct=float(100.0 * top.iloc[:TOP_DAYS].sum() / dse.sum()),
                    top_first=str(top.index[:TOP_DAYS].min().date()),
                    top_last=str(top.index[:TOP_DAYS].max().date()),
                    top_days=" ".join(str(t.date()) for t in top.index[:TOP_DAYS]),
                    bar_mse_pct_excrash=float(100.0 * (sef[ex].mean() / seh[ex].mean() - 1.0)),
                    bar_mse_dm_excrash=float(tx["dm"]),
                    bar_mse_p_excrash=float(tx["p"]),
                    close_mse_pct_excrash=float(100.0 * (s16f.reindex(d16x).mean() / s16h.reindex(d16x).mean() - 1.0)),
                    close_mse_dm_excrash=float(tx16["dm"]),
                    close_mse_p_excrash=float(tx16["p"]),
                    bar_mae_pct=float(100.0 * (np.sqrt(sef[m]).mean() / np.sqrt(seh[m]).mean() - 1.0)),
                    bar_mae_pct_excrash=float(100.0 * (np.sqrt(sef[ex]).mean() / np.sqrt(seh[ex]).mean() - 1.0)),
                    bar_mae_dm_excrash=float(txa["dm"]),
                    bar_mae_p_excrash=float(txa["p"]),
                    max_fc_full=float(ff[m][jmax]),
                    max_fc_time=str(sidx[jmax]),
                    max_fc_rv=float(yy[jmax]),
                    max_fc_har=float(fh[m][jmax]),
                    max_rv=float(yy.max()),
                )
            )
            top_days[(tag, e)] = list(top.index[:TOP_DAYS])
    return pd.DataFrame(rows), top_days


def yearly16(D: dict) -> pd.DataFrame:
    """16:00 QLIKE by calendar year (2020 from 3 January), Full vs HAR, with the level / dispersion split."""
    rows = []
    days = D["days"]
    for yr in sorted(set(days.year)):
        d = days[(days.year == yr) & ~((days > SPANS["Pre"][1]) & (days < SPANS["Oos"][0]))]
        y = D["y16"].reindex(d).to_numpy()
        for e in EST:
            fh, ff = (
                D["P16"][key(e, "baseline")].reindex(d).to_numpy(),
                D["P16"][key(e, "all_features")].reindex(d).to_numpy(),
            )
            ch, cf = calib(y, fh, _nw_lag(len(d))), calib(y, ff, _nw_lag(len(d)))
            rows.append(
                dict(
                    year=yr,
                    n_days=len(d),
                    model=e,
                    q_har=ch["qlike"],
                    q_full=cf["qlike"],
                    pct=100.0 * (cf["qlike"] / ch["qlike"] - 1.0),
                    mean_r_har=ch["mean_r"],
                    mean_r_full=cf["mean_r"],
                    under_har=ch["under"],
                    under_full=cf["under"],
                    med_log_full_over_har=float(np.median(np.log(ff / fh))),
                )
            )
    return pd.DataFrame(rows)


def deciles(D: dict) -> pd.DataFrame:
    rows = []
    idx, P, y = D["idx"], D["P"], D["y"]
    dd = pd.DatetimeIndex(idx.normalize())
    for tag in ("Pre", "Oos"):
        f0, l0, _ = SPANS[tag]
        mb = (dd >= f0) & (dd <= l0)
        d16 = span_days(D["days"], tag)
        for sample in ("Bar", "Close"):
            for e in EST:
                for s in SETS:
                    if sample == "Bar":
                        yy, ff = y.to_numpy()[mb], P[key(e, s)].to_numpy()[mb]
                    else:
                        yy, ff = D["y16"].reindex(d16).to_numpy(), D["P16"][key(e, s)].reindex(d16).to_numpy()
                    q = pd.qcut(pd.Series(ff).rank(method="first"), N_DEC, labels=False).to_numpy()
                    for k in range(N_DEC):
                        m = q == k
                        rows.append(
                            dict(
                                sample=sample,
                                span=tag,
                                model=e,
                                set=s,
                                decile=k + 1,
                                n=int(m.sum()),
                                mean_fc=float(ff[m].mean()),
                                mean_rv=float(yy[m].mean()),
                                ratio=float(yy[m].mean() / ff[m].mean()),
                                se_share_pct=float(100.0 * ((yy[m] - ff[m]) ** 2).sum() / ((yy - ff) ** 2).sum()),
                            )
                        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- outputs: macros


class Macros:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.names: set[str] = set()

    def add(self, name: str, value: str, comment: str = "") -> None:
        assert re.fullmatch(r"rhca[A-Za-z]+", name), name
        assert name not in self.names, name
        self.names.add(name)
        self.lines.append(
            f"\\providecommand{{\\{name}}}{{}}\\renewcommand{{\\{name}}}{{{value}}}"
            + (f"  % {comment}" if comment else "")
        )

    def section(self, text: str) -> None:
        self.lines.append(f"% --- {text}")


def write_macros(
    D: dict,
    A16: pd.DataFrame,
    ABar: pd.DataFrame,
    CAL: pd.DataFrame,
    EXT: pd.DataFrame,
    YR: pd.DataFrame,
    DEC: pd.DataFrame,
) -> dict:
    M = Macros()
    V: dict = {}  # values the tables and the draft need
    M.section("spans and constants")
    for tag, (f0, l0, n0) in SPANS.items():
        M.add(f"rhcaNdays{tag}", thousands(n0), f"trading days, {f0.date()} .. {l0.date()}")
        M.add(f"rhcaStart{tag}", long_date(f0), "first day")
        M.add(f"rhcaEnd{tag}", long_date(l0), "last day")
        M.add(f"rhcaYears{tag}", f"{f0.year}--{l0.year}", "years")
        M.add(f"rhcaDMLag{tag}", str(_nw_lag(n0)), "Newey-West lag of the daily DM tests (dm_test automatic)")
        M.add(f"rhcaHacLagBar{tag}", str(13 * _nw_lag(n0)), "Newey-West lag of the every-bar MZ regressions, bars")
    M.add("rhcaNbars", thousands(len(D["idx"])), "scored bars of Table 1, 2014-2024")
    M.add("rhcaCloseFirst", long_date(D["first_close"]), "first forecast of the closing-interval models")
    M.add("rhcaMseExp", str(MSE_EXP), "MSE printed x 10^this")
    M.add("rhcaUnitExp", str(UNIT_EXP), "MAE and the levels intercept printed x 10^this")
    M.add("rhcaCrashStart", long_date(CRASH[0]), "first day of the excluded window (make_rehaul_results.CRASH)")
    M.add("rhcaCrashEnd", long_date(CRASH[1]), "last day of the excluded window")
    M.add("rhcaTopDays", str(TOP_DAYS), "days in the extreme-day share")

    M.section(
        "closing half hour (16:00 bar): pooled forecasts, Span / Pre / Oos; Q QLIKE, Mse (x 10^10), Mae (x 10^6), "
        "Rsq R2_OOS %, Pct / Dm / P QLIKE full vs HAR, MsePct / MseDm / MseP, Gap / GapDm / GapP vs the "
        "closing-interval HAR benchmark (Oos only)"
    )
    for _, r in A16.iterrows():
        tag = r.span
        mt = ("Ci" if r.scheme == "close" else "") + MTOK[r.model]
        if r.scheme == "close" and r.model == "ols":
            mt = "Bm"
        st = "" if mt == "Bm" else STOK[r.set]
        base = f"{mt}{st}Close{tag}"
        M.add(f"rhcaQ{base}", f"{r.qlike:.4f}")
        M.add(f"rhcaMse{base}", mnum(r.mse * 10**MSE_EXP, 2))
        M.add(f"rhcaMae{base}", mnum(r.mae * 10**UNIT_EXP, 2))
        M.add(f"rhcaRsq{base}", mnum(r.r2_oos_pct, 1))
        if r.set == "all_features":
            b2 = f"{mt}Close{tag}"
            M.add(f"rhcaPct{b2}", mnum(r.pct, 1), "QLIKE %, full vs HAR of the same class")
            M.add(f"rhcaDm{b2}", mnum(r.dm, 2))
            M.add(f"rhcaP{b2}", mpval(r.p))
            M.add(f"rhcaMsePct{b2}", mnum(r.mse_pct, 1), "MSE %, full vs HAR")
            M.add(f"rhcaMseDm{b2}", mnum(r.mse_dm, 2))
            M.add(f"rhcaMseP{b2}", mpval(r.mse_p))
        if np.isfinite(r.get("gap", np.nan)):
            M.add(f"rhcaGap{base}", mnum(r.gap, pct_dp(r.gap)), "QLIKE % vs the closing-interval HAR benchmark")
            M.add(f"rhcaGapDm{base}", mnum(r.gap_dm, 2))
            M.add(f"rhcaGapP{base}", mpval(r.gap_p))

    M.section("every bar: pooled forecasts, Span / Pre / Oos (Table 1 and its subperiods)")
    for _, r in ABar.iterrows():
        base = f"{MTOK[r.model]}{STOK[r.set]}Bar{r.span}"
        M.add(f"rhcaQ{base}", f"{r.qlike:.4f}")
        M.add(f"rhcaMse{base}", mnum(r.mse * 10**MSE_EXP, 2))
        M.add(f"rhcaMae{base}", mnum(r.mae * 10**UNIT_EXP, 2))
        M.add(f"rhcaRsq{base}", mnum(r.r2_oos_pct, 1))
        if r.set == "all_features":
            b2 = f"{MTOK[r.model]}Bar{r.span}"
            M.add(f"rhcaPct{b2}", mnum(r.pct, 1))
            M.add(f"rhcaDm{b2}", mnum(r.dm, 2))
            M.add(f"rhcaP{b2}", mpval(r.p))
            M.add(f"rhcaMsePct{b2}", mnum(r.mse_pct, 1))
            M.add(f"rhcaMseDm{b2}", mnum(r.mse_dm, 2))
            M.add(f"rhcaMseP{b2}", mpval(r.mse_p))

    M.section(
        "calibration, Pre / Oos, every bar and 16:00: Mr mean of RV/sigma2, Rm ratio of means, Under %, MzA (x 10^6), "
        "MzB, MzBse, MzRsq (levels MZ), MzlA, MzlB, MzlBse, MzlRsq (logs MZ), ShBias / ShSlope / ShDist (Theil "
        "shares of the MSE, %), Ql / Qd (QLIKE level / dispersion terms), OverSe (% of SE from over-predictions)"
    )
    # (the pooled OLS is left out: its forecasts are those of the HAR-set linear models, rhcaHarLinGap)
    for _, r in CAL[CAL.span.isin(["Pre", "Oos"]) & (CAL.model != "ols")].iterrows():
        base = f"{MTOK[r.model]}{STOK[r.set]}{r['sample']}{r.span}"
        M.add(f"rhcaMr{base}", f"{r.mean_r:.3f}")
        M.add(f"rhcaRm{base}", f"{r.ratio_means:.3f}")
        M.add(f"rhcaUnder{base}", mnum(100 * r.under, 1))
        M.add(f"rhcaMzA{base}", mnum(r.mz_a * 10**UNIT_EXP, 2))
        M.add(f"rhcaMzB{base}", mnum(r.mz_b, 2))
        M.add(f"rhcaMzBse{base}", mnum(r.mz_se_b, 2))
        M.add(f"rhcaMzRsq{base}", mnum(r.mz_r2, 2))
        M.add(f"rhcaMzlA{base}", mnum(r.mzl_a, 2))
        M.add(f"rhcaMzlB{base}", mnum(r.mzl_b, 2))
        M.add(f"rhcaMzlBse{base}", mnum(r.mzl_se_b, 2))
        M.add(f"rhcaMzlRsq{base}", mnum(r.mzl_r2, 2))
        M.add(f"rhcaShBias{base}", mnum(r.sh_bias, 1))
        M.add(f"rhcaShSlope{base}", mnum(r.sh_slope, 1))
        M.add(f"rhcaShDist{base}", mnum(r.sh_dist, 1))
        M.add(f"rhcaQl{base}", f"{r.q_level:.4f}")
        M.add(f"rhcaQd{base}", f"{r.q_disp:.4f}")
        M.add(f"rhcaOverSe{base}", mnum(r.over_se_share, 1))

    # ---------------------------------------------------------------- aggregates for the prose
    M.section("aggregates for the prose")

    def a16(e: str, s: str, tag: str, scheme: str = "pool") -> pd.Series:
        return A16[(A16.scheme == scheme) & (A16.model == e) & (A16.set == s) & (A16.span == tag)].iloc[0]

    def cal(e: str, s: str, sample: str, tag: str) -> pd.Series:
        return CAL[(CAL.model == e) & (CAL.set == s) & (CAL["sample"] == sample) & (CAL.span == tag)].iloc[0]

    def rng(name: str, vals: list[float], dp: int, comment: str) -> None:
        lo, hi = mnum(min(vals), dp), mnum(max(vals), dp)
        M.add(f"{name}Min", lo, comment + ": min")
        M.add(f"{name}Max", hi, "... max")
        # the range as text: one value when both ends print alike, "lo--hi", or "lo to hi" when an end is negative
        txt = lo if lo == hi else (f"{lo} to {hi}" if min(vals) < 0 and round(min(vals), dp) != 0 else f"{lo}--{hi}")
        M.add(f"{name}Rng", txt, "... the range as text")

    # closing half hour: the full set vs HAR, per span, counts
    for tag in SPANS:
        fa = [a16(e, "all_features", tag) for e in EST]
        low = [r for r in fa if r.pct < 0]
        M.add(
            f"rhcaNLowerClose{tag}", str(len(low)), f"classes (of 6) with the lower 16:00 QLIKE on the full set, {tag}"
        )
        M.add(f"rhcaNLowerSigClose{tag}", str(sum(r.p < SIG for r in low)), "... and DM p < 0.05")
        hi = [r for r in fa if r.pct > 0]
        M.add(f"rhcaNHigherClose{tag}", str(len(hi)), "classes with the higher 16:00 QLIKE on the full set")
        M.add(f"rhcaNHigherSigClose{tag}", str(sum(r.p < SIG for r in hi)), "... and DM p < 0.05")
        M.add(f"rhcaNLowerCloseWord{tag}", WORD[len(low)], "as a word")
        lin = [a16(e, "all_features", tag) for e in LIN]
        M.add(
            f"rhcaNLinHigherSigClose{tag}",
            str(sum(r.pct > 0 and r.p < SIG for r in lin)),
            "linear classes (of 3): full-set 16:00 QLIKE significantly higher",
        )
        M.add(f"rhcaNLinLowerClose{tag}", str(sum(r.pct < 0 for r in lin)), "linear classes with a lower 16:00 QLIKE")
        tr = [a16(e, "all_features", tag) for e in TREE]
        M.add(
            f"rhcaNTreeLowerSigClose{tag}",
            str(sum(r.pct < 0 and r.p < SIG for r in tr)),
            "tree classes (of 3): full-set 16:00 QLIKE significantly lower",
        )
        rng(f"rhcaLinPctClose{tag}", [r.pct for r in lin], 1, f"linear 16:00 QLIKE %, full vs HAR, {tag}")
        rng(f"rhcaTreePctClose{tag}", [r.pct for r in tr], 1, f"tree 16:00 QLIKE %, full vs HAR, {tag}")
        rng(f"rhcaLinMsePctClose{tag}", [r.mse_pct for r in lin], 1, f"linear 16:00 MSE %, full vs HAR, {tag}")
        rng(f"rhcaTreeMsePctClose{tag}", [r.mse_pct for r in tr], 1, f"tree 16:00 MSE %, full vs HAR, {tag}")
        M.add(f"rhcaNMseHigherClose{tag}", str(sum(r.mse_pct > 0 for r in fa)), "classes with a higher 16:00 MSE, full")
        M.add(
            f"rhcaNMseHigherSigClose{tag}", str(sum(r.mse_pct > 0 and r.mse_p < SIG for r in fa)), "... significantly"
        )
    # every bar: MSE direction, per span
    for tag in SPANS:
        fa = ABar[(ABar.set == "all_features") & (ABar.span == tag) & ABar.model.isin(EST)]
        M.add(
            f"rhcaNMseHigherBar{tag}",
            str(int((fa.mse_pct > 0).sum())),
            f"classes with a higher every-bar MSE, full, {tag}",
        )
        M.add(f"rhcaNMseHigherSigBar{tag}", str(int(((fa.mse_pct > 0) & (fa.mse_p < SIG)).sum())), "... significantly")
        M.add(f"rhcaNMseLowerBar{tag}", str(int((fa.mse_pct < 0).sum())), "classes with a lower every-bar MSE, full")
        M.add(f"rhcaNMseLowerSigBar{tag}", str(int(((fa.mse_pct < 0) & (fa.mse_p < SIG)).sum())), "... significantly")
        lin, tr = fa[fa.model.isin(LIN)], fa[fa.model.isin(TREE)]
        rng(f"rhcaLinMsePctBar{tag}", list(lin.mse_pct), 1, f"linear every-bar MSE %, full vs HAR, {tag}")
        rng(f"rhcaTreeMsePctBar{tag}", list(tr.mse_pct), 1, f"tree every-bar MSE %, full vs HAR, {tag}")
    # the gap to the closing-interval HAR benchmark on T_oos: closing-interval models
    ci = A16[(A16.scheme == "close") & (A16.model != "ols")]
    M.add(
        "rhcaNCiLowerBm",
        str(int((ci.gap < 0).sum())),
        f"closing-interval models (of {len(ci)}) below the benchmark's QLIKE",
    )
    M.add("rhcaNCiLowerSigBm", str(int(((ci.gap < 0) & (ci.gap_p < SIG)).sum())), "... significantly")
    M.add("rhcaNCi", str(len(ci)), "closing-interval models other than the benchmark")
    ci_har = ci[(ci.set == "baseline") & ci.model.isin(LIN)]
    bound = 0.01
    assert (ci_har.gap.abs() < bound).all(), ci_har.gap
    M.add(
        "rhcaCiHarGapBound",
        f"{bound:.2f}",
        "|QLIKE %| of the closing-interval HAR-set linear models vs the benchmark is below",
    )
    # calibration aggregates (Oos and Pre, every bar and 16:00)
    for sample in ("Bar", "Close"):
        for tag in ("Pre", "Oos"):
            sfx = f"{sample}{tag}"
            har = [cal(e, "baseline", sample, tag) for e in EST]
            full = [cal(e, "all_features", sample, tag) for e in EST]
            linf, trf = full[:3], full[3:]
            linh, trh = har[:3], har[3:]
            rng(f"rhcaLinFullMzB{sfx}", [r.mz_b for r in linf], 2, f"levels MZ slope, linear full, {sfx}")
            rng(f"rhcaTreeFullMzB{sfx}", [r.mz_b for r in trf], 2, f"levels MZ slope, tree full, {sfx}")
            rng(f"rhcaTreeHarMzB{sfx}", [r.mz_b for r in trh], 2, f"levels MZ slope, tree HAR, {sfx}")
            M.add(f"rhcaLinHarMzB{sfx}", mnum(linh[0].mz_b, 2), f"levels MZ slope, linear HAR (= OLS), {sfx}")
            rng(f"rhcaLinFullMzlB{sfx}", [r.mzl_b for r in linf], 2, f"logs MZ slope, linear full, {sfx}")
            rng(f"rhcaTreeFullMzlB{sfx}", [r.mzl_b for r in trf], 2, f"logs MZ slope, tree full, {sfx}")
            rng(f"rhcaAllHarMzlB{sfx}", [r.mzl_b for r in har], 2, f"logs MZ slope, HAR forecasts, {sfx}")
            rng(f"rhcaLinFullMzlRsq{sfx}", [r.mzl_r2 for r in linf], 2, f"logs MZ R2, linear full, {sfx}")
            rng(f"rhcaTreeFullMzlRsq{sfx}", [r.mzl_r2 for r in trf], 2, f"logs MZ R2, tree full, {sfx}")
            rng(f"rhcaAllHarMzlRsq{sfx}", [r.mzl_r2 for r in har], 2, f"logs MZ R2, HAR forecasts, {sfx}")
            rng(f"rhcaLinFullMzRsq{sfx}", [r.mz_r2 for r in linf], 2, f"levels MZ R2, linear full, {sfx}")
            rng(f"rhcaAllHarMzRsq{sfx}", [r.mz_r2 for r in har], 2, f"levels MZ R2, HAR forecasts, {sfx}")
            rng(f"rhcaLinFullShSlope{sfx}", [r.sh_slope for r in linf], 1, f"Theil slope share %, linear full, {sfx}")
            rng(f"rhcaTreeFullShSlope{sfx}", [r.sh_slope for r in trf], 1, f"Theil slope share %, tree full, {sfx}")
            rng(f"rhcaAllHarShSlope{sfx}", [r.sh_slope for r in har], 1, f"Theil slope share %, HAR forecasts, {sfx}")
            rng(f"rhcaAllShBias{sfx}", [r.sh_bias for r in har + full], 1, f"Theil bias share %, all 12, {sfx}")
            rng(f"rhcaAllRm{sfx}", [r.ratio_means for r in har + full], 2, f"ratio of means, all 12, {sfx}")
            rng(f"rhcaFullRm{sfx}", [r.ratio_means for r in full], 2, f"ratio of means, full set, {sfx}")
            rng(f"rhcaHarRm{sfx}", [r.ratio_means for r in har], 2, f"ratio of means, HAR, {sfx}")
            rng(f"rhcaAllMr{sfx}", [r.mean_r for r in har + full], 2, f"mean of RV/sigma2, all 12, {sfx}")
            rng(f"rhcaAllUnder{sfx}", [100 * r.under for r in har + full], 1, f"under-prediction %, all 12, {sfx}")
            rng(f"rhcaLinFullUnder{sfx}", [100 * r.under for r in linf], 1, f"under-prediction %, linear full, {sfx}")
            M.add(f"rhcaLinHarUnder{sfx}", mnum(100 * linh[0].under, 1), f"under-prediction %, linear HAR, {sfx}")
            rng(
                f"rhcaLinFullOverSe{sfx}",
                [r.over_se_share for r in linf],
                1,
                f"% of SE from over-predictions, linear full, {sfx}",
            )
            rng(
                f"rhcaTreeFullOverSe{sfx}",
                [r.over_se_share for r in trf],
                1,
                f"% of SE from over-predictions, tree full, {sfx}",
            )
            rng(
                f"rhcaAllHarOverSe{sfx}",
                [r.over_se_share for r in har],
                1,
                f"% of SE from over-predictions, HAR, {sfx}",
            )
            # MAE direction
            n_mae = sum(f.mae < h.mae for f, h in zip(full, har))
            M.add(f"rhcaNMaeLower{sfx}", str(n_mae), f"classes (of 6) with the lower MAE on the full set, {sfx}")
            # QLIKE decomposition of the full-vs-HAR change: share of the level term (linear models)
            dq = [f.qlike - h.qlike for f, h in zip(linf, linh)]
            dl = [f.q_level - h.q_level for f, h in zip(linf, linh)]
            if all(x > 0 for x in dq):
                rng(
                    f"rhcaLinQlShare{sfx}",
                    [100 * a / b for a, b in zip(dl, dq)],
                    0,
                    f"% of the linear full-set QLIKE increase from the level term, {sfx}",
                )
            rng(f"rhcaLinFullMr{sfx}", [r.mean_r for r in linf], 3, f"mean of RV/sigma2, linear full, {sfx}")
            M.add(f"rhcaLinHarMr{sfx}", f"{linh[0].mean_r:.3f}", f"mean of RV/sigma2, linear HAR, {sfx}")
            rng(f"rhcaLinFullQl{sfx}", [r.q_level for r in linf], 4, f"QLIKE level term, linear full, {sfx}")
            M.add(f"rhcaLinHarQl{sfx}", f"{linh[0].q_level:.4f}", f"QLIKE level term, linear HAR, {sfx}")
            rng(f"rhcaLinFullQd{sfx}", [r.q_disp for r in linf], 4, f"QLIKE dispersion term, linear full, {sfx}")
            M.add(f"rhcaLinHarQd{sfx}", f"{linh[0].q_disp:.4f}", f"QLIKE dispersion term, linear HAR, {sfx}")
            rng(f"rhcaTreeFullQl{sfx}", [r.q_level for r in trf], 4, f"QLIKE level term, tree full, {sfx}")
            rng(f"rhcaTreeHarQl{sfx}", [r.q_level for r in trh], 4, f"QLIKE level term, tree HAR, {sfx}")
            rng(f"rhcaTreeFullQd{sfx}", [r.q_disp for r in trf], 4, f"QLIKE dispersion term, tree full, {sfx}")
            rng(f"rhcaTreeHarQd{sfx}", [r.q_disp for r in trh], 4, f"QLIKE dispersion term, tree HAR, {sfx}")
            rng(f"rhcaTreeFullUnder{sfx}", [100 * r.under for r in trf], 1, f"under-prediction %, tree full, {sfx}")
            rng(f"rhcaTreeHarUnder{sfx}", [100 * r.under for r in trh], 1, f"under-prediction %, tree HAR, {sfx}")
    # extremes, every bar, T_oos
    ex = EXT[EXT.span == "Oos"].set_index("model")
    rng(
        "rhcaLinTopShareBarOos",
        list(ex.loc[list(LIN), "top_share_pct"]),
        0,
        f"% of the full-minus-HAR squared-error sum (every bar, Oos) on the {TOP_DAYS} days of largest difference, linear",
    )
    rng("rhcaTreeTopShareBarOos", list(ex.loc[list(TREE), "top_share_pct"]), 0, "... tree ensembles")
    first = min(pd.Timestamp(x) for x in ex.top_first)
    last = max(pd.Timestamp(x) for x in ex.top_last)
    M.add("rhcaTopFirstBarOos", long_date(first), "earliest of those days, any class")
    M.add("rhcaTopLastBarOos", long_date(last), "latest of those days, any class")
    V["top_window"] = (first, last)
    rng(
        "rhcaLinMsePctBarOosExCrash",
        list(ex.loc[list(LIN), "bar_mse_pct_excrash"]),
        1,
        "linear every-bar MSE %, full vs HAR, Oos without the crash window",
    )
    rng("rhcaTreeMsePctBarOosExCrash", list(ex.loc[list(TREE), "bar_mse_pct_excrash"]), 1, "... tree ensembles")
    M.add(
        "rhcaNMseLowerBarOosExCrash",
        str(int((ex.bar_mse_pct_excrash < 0).sum())),
        "classes (of 6) with the lower every-bar MSE on the full set, Oos without the crash window",
    )
    M.add(
        "rhcaNMseLowerSigBarOosExCrash",
        str(int(((ex.bar_mse_pct_excrash < 0) & (ex.bar_mse_p_excrash < SIG)).sum())),
        "... significantly (DM on the daily mean squared error)",
    )
    rng(
        "rhcaLinMsePctCloseOosExCrash",
        list(ex.loc[list(LIN), "close_mse_pct_excrash"]),
        1,
        "linear 16:00 MSE %, full vs HAR, Oos without the crash window",
    )
    M.add(
        "rhcaNMseLowerCloseOosExCrash",
        str(int((ex.close_mse_pct_excrash < 0).sum())),
        "classes with the lower 16:00 MSE on the full set, Oos without the crash window",
    )
    M.add(
        "rhcaNMseHigherSigCloseOosExCrash",
        str(int(((ex.close_mse_pct_excrash > 0) & (ex.close_mse_p_excrash < SIG)).sum())),
        "... classes with a significantly higher 16:00 MSE there",
    )
    rng(
        "rhcaTreeMsePctCloseOosExCrash",
        list(ex.loc[list(TREE), "close_mse_pct_excrash"]),
        1,
        "tree 16:00 MSE %, full vs HAR, Oos without the crash window",
    )
    rng("rhcaAllMaePctBarOos", list(ex.bar_mae_pct), 1, "every-bar MAE %, full vs HAR, Oos, all six classes")
    rng("rhcaAllMaePctBarOosExCrash", list(ex.bar_mae_pct_excrash), 1, "... without the crash window")
    M.add(
        "rhcaNMaeLowerBarOosExCrash",
        str(int((ex.bar_mae_pct_excrash < 0).sum())),
        "classes with the lower every-bar MAE on the full set, Oos without the crash window",
    )
    M.add(
        "rhcaNMaeLowerSigBarOosExCrash",
        str(int(((ex.bar_mae_pct_excrash < 0) & (ex.bar_mae_p_excrash < SIG)).sum())),
        "... significantly",
    )
    for e in EST:
        r = ex.loc[e]
        M.add(f"rhcaTopShare{MTOK[e]}BarOos", mnum(r.top_share_pct, 0), "top-day share, %")
        M.add(f"rhcaMsePct{MTOK[e]}BarOosExCrash", mnum(r.bar_mse_pct_excrash, 1), "every-bar MSE %, ex crash")
        M.add(f"rhcaMseDm{MTOK[e]}BarOosExCrash", mnum(r.bar_mse_dm_excrash, 2))
        M.add(f"rhcaMseP{MTOK[e]}BarOosExCrash", mpval(r.bar_mse_p_excrash))
    for e in LIN:
        r = ex.loc[e]
        M.add(
            f"rhcaMaxFc{MTOK[e]}",
            mnum(r.max_fc_full * 10**UNIT_EXP, 0),
            "largest full-set forecast on a bar, Oos, x 10^6",
        )
        M.add(f"rhcaMaxFcRv{MTOK[e]}", mnum(r.max_fc_rv * 10**UNIT_EXP, 0), "the realized variance of that bar, x 10^6")
        M.add(f"rhcaMaxFcHar{MTOK[e]}", mnum(r.max_fc_har * 10**UNIT_EXP, 0), "the HAR forecast of that bar, x 10^6")
        M.add(
            f"rhcaMaxFcRatio{MTOK[e]}", mnum(r.max_fc_full / r.max_fc_rv, 1), "forecast / realized variance on that bar"
        )
    t = pd.Timestamp(ex.loc["enet", "max_fc_time"])
    assert all(pd.Timestamp(ex.loc[e, "max_fc_time"]) == t for e in LIN), "the linear maxima fall on different bars"
    M.add("rhcaMaxFcDate", long_date(t), "the bar of the largest linear full-set forecast (all three)")
    M.add("rhcaMaxFcClock", t.strftime("%H:%M"), "its bar end, ET")
    M.add("rhcaMaxRvOos", mnum(float(ex.loc["enet", "max_rv"]) * 10**UNIT_EXP, 0), "largest bar RV on Oos, x 10^6")
    # yearly 16:00 (linear full vs HAR)
    for e in EST:
        g = YR[(YR.model == e) & (YR.year >= SPANS["Oos"][0].year)]
        M.add(
            f"rhcaYrsUp{MTOK[e]}CloseOos",
            str(int((g.pct > 0).sum())),
            f"calendar years of Oos (of {len(g)}) with a higher 16:00 QLIKE on the full set",
        )
        gp = YR[(YR.model == e) & (YR.year <= SPANS["Pre"][1].year)]
        M.add(
            f"rhcaYrsDown{MTOK[e]}ClosePre",
            str(int((gp.pct < 0).sum())),
            f"calendar years of Pre (of {len(gp)}) with a lower 16:00 QLIKE",
        )
        w = g.loc[g.pct.idxmax()]
        M.add(f"rhcaWorstYr{MTOK[e]}CloseOos", str(int(w.year)), "the Oos year of the largest full-set increase")
        M.add(f"rhcaWorstYrPct{MTOK[e]}CloseOos", mnum(w.pct, 1), "... its QLIKE %")
    M.add("rhcaNYrsOos", str(int(YR[YR.year >= SPANS["Oos"][0].year].year.nunique())), "calendar years of Oos")
    M.add("rhcaNYrsPre", str(int(YR[YR.year <= SPANS["Pre"][1].year].year.nunique())), "calendar years of Pre")
    # median log ratio of the forecasts, 16:00, Oos and Pre
    for tag in ("Pre", "Oos"):
        d = span_days(D["days"], tag)
        v = [
            100
            * float(
                np.median(np.log(D["P16"][key(e, "all_features")].reindex(d) / D["P16"][key(e, "baseline")].reindex(d)))
            )
            for e in LIN
        ]
        rng(f"rhcaLinMedLogRatioClose{tag}", v, 1, f"median 100 ln(full / HAR forecast), linear, 16:00, {tag}")
        V[f"medlog_{tag}"] = v
        vt = [
            100
            * float(
                np.median(np.log(D["P16"][key(e, "all_features")].reindex(d) / D["P16"][key(e, "baseline")].reindex(d)))
            )
            for e in TREE
        ]
        rng(f"rhcaTreeMedLogRatioClose{tag}", vt, 1, f"median 100 ln(full / HAR forecast), trees, 16:00, {tag}")
        # draft claim: at 16:00 the full-set forecasts of every class lie below the HAR forecasts (median)
        assert max(v + vt) < 0, (tag, v, vt)
    # deciles: the top decile of the linear full set, every bar, Oos
    for sample in ("Bar", "Close"):
        for tag in ("Pre", "Oos"):
            for e in ("enet", "lgbm"):
                for s in SETS:
                    r = DEC[
                        (DEC["sample"] == sample)
                        & (DEC.span == tag)
                        & (DEC.model == e)
                        & (DEC.set == s)
                        & (DEC.decile == N_DEC)
                    ].iloc[0]
                    base = f"{MTOK[e]}{STOK[s]}{sample}{tag}"
                    M.add(f"rhcaTopDecRatio{base}", f"{r.ratio:.2f}", "top forecast decile: mean RV / mean forecast")
                    M.add(f"rhcaTopDecSe{base}", mnum(r.se_share_pct, 0), "top forecast decile: % of the squared error")
                    r1 = DEC[
                        (DEC["sample"] == sample)
                        & (DEC.span == tag)
                        & (DEC.model == e)
                        & (DEC.set == s)
                        & (DEC.decile == 1)
                    ].iloc[0]
                    M.add(
                        f"rhcaBotDecRatio{base}", f"{r1.ratio:.2f}", "bottom forecast decile: mean RV / mean forecast"
                    )
    # the HAR linear rows equal the OLS row (frozen penalties numerically zero): largest relative forecast gap
    gap = max(float(np.max(np.abs(D["P"][key(e, "baseline")] / D["P"][key("ols", "baseline")] - 1.0))) for e in LIN)
    assert 0 < gap < 1e-3, gap
    ex10 = int(math.floor(math.log10(gap)))
    M.add(
        "rhcaHarLinGap",
        f"{gap / 10**ex10:.0f} \\times 10^{{{ex10}}}",
        "max relative gap, HAR linear vs OLS forecasts (math mode)",
    )
    V["har_lin_gap"] = gap
    txt = "\n".join(
        [
            HEAD,
            "% \\rhca... macros: closing-half-hour accuracy on both spans (tab:rh_close_accuracy) and the calibration",
            "% diagnostics of the 12 pooled forecasts (tab:rh_calibration, fig:rh_calibration). Names: <Stat><Model><Set>",
            "% <Sample><Span>; Model Ridge Lasso Enet Lgbm Xgb Rf Ols (pooled), Ci<Model> (trained on the closing half",
            "% hour), Bm (closing-interval HAR benchmark); Set Har Full; Sample Close (16:00 bar) Bar (every bar);",
            "% Span Span (2014-2024) Pre (2014-2019) Oos (T_oos, 2020-2024).",
            *M.lines,
        ]
    )
    OUT_NUM.write_text(txt + "\n", encoding="utf-8")
    V["names"] = M.names
    return V


# --------------------------------------------------------------------------- outputs: tables


def write_accuracy_table(A16: pd.DataFrame) -> None:
    def row_cells(r: pd.Series, gap_ok: bool) -> list[str]:
        out = [f"${r.qlike:.4f}$", cell(r.mse * 10**MSE_EXP, 2, starred=False), cell(r.r2_oos_pct, 1, starred=False)]
        if r.set == "all_features":
            out += [cell(r.pct, 1, n_stars(r.p)), stat_col(r.dm)]
        else:
            out += ["", ""]
        if gap_ok and np.isfinite(r.get("gap", np.nan)):
            out += [pct_cell(r.gap, r.gap_p), stat_col(r.gap_dm)]
        else:
            out += ["--", ""]
        return out

    body: list[str] = []

    def block(scheme: str, tag: str, ests: tuple[str, ...]) -> None:
        for e in ests:
            for s in SETS:
                r = A16[(A16.scheme == scheme) & (A16.span == tag) & (A16.model == e) & (A16.set == s)].iloc[0]
                body.append(
                    " & ".join([NAME[e] if s == "baseline" else "", SETNAME[s]] + row_cells(r, tag == "Oos")) + " \\\\"
                )

    def head(title: str, first: bool = False) -> None:
        body.extend(
            ([] if first else ["\\addlinespace[4pt]"])
            + [f"\\multicolumn{{9}}{{@{{}}l}}{{\\textit{{{title}}}}} \\\\", "\\addlinespace[2pt]"]
        )

    head("Panel A: \\rhcaStartSpan{} to \\rhcaEndSpan{} ($\\rhcaNdaysSpan$ days), pooled intraday models", first=True)
    block("pool", "Span", EST)
    head(
        "Panel B: $\\mathcal T_{\\mathrm{oos}}$, \\rhcaStartOos{} to \\rhcaEndOos{} ($\\rhcaNdaysOos$ days), pooled intraday models"
    )
    block("pool", "Oos", EST)
    head("Panel B, continued: models trained on the closing half hour (Appendix~\\ref{sec:rh_econ_last30})")
    r = A16[(A16.scheme == "close") & (A16.model == "ols")].iloc[0]
    body.append(" & ".join(["OLS$^{a}$", "HAR"] + row_cells(r, True)) + " \\\\")
    block("close", "Oos", EST)
    lines = [
        HEAD,
        "% tab:rh_close_accuracy: the forecast of the closing half hour (16:00 bar) of the 12 pooled forecasts on Table 1's",
        "% 2014-2024 span and on T_oos, and of the closing-interval models on T_oos; DM on the daily 16:00 loss.",
        "\\ifdefined\\rhcaNdaysSpan\\else\\input{generated/rehaul_close_accuracy_numbers}\\fi",
        "\\begin{table}[tbp]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{threeparttable}",
        "\\caption{Forecast accuracy for the closing half hour on both evaluation spans}",
        "\\label{tab:rh_close_accuracy}",
        "\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}llrrrrrrr@{}}",
        "\\toprule",
        " & & & & & \\multicolumn{2}{c}{Full vs HAR} & \\multicolumn{2}{c}{vs closing HAR} \\\\",
        "\\cmidrule(lr){6-7}\\cmidrule(l){8-9}",
        "Model & Set & QLIKE & MSE & $R^2_{\\mathrm{OOS}}$ (\\%) & $\\Delta\\%$ & $\\DM$ & $\\Delta\\%$ & $\\DM$ \\\\",
        "\\midrule",
        *body,
        "\\bottomrule",
        "\\end{tabular*}",
        "\\begin{tablenotes}[flushleft]",
        "\\footnotesize",
        (
            "\\item \\textit{Notes:} The 15:30 forecast of the closing half hour (the 16:00 bar) of the pooled models of "
            "Table~\\ref{tab:rh_oos}, scored against the actual realized variance $\\RV_t$ of the closing half hour, one "
            "forecast a day. Panel~A: the $\\rhcaNdaysSpan$ evaluation days of Table~\\ref{tab:rh_oos}; Panel~B: "
            "$\\mathcal T_{\\mathrm{oos}}$, the days of Section~\\ref{sec:rh_results_predictors}, with the models trained on "
            "the closing half hour only (Table~\\ref{tab:rh_oos_last30}). HAR: HAR benchmark; Full: full predictor set; "
            "on the HAR benchmark the pooled ridge, lasso and elastic net reproduce the OLS forecast "
            "(Section~\\ref{sec:rh_exp_setup}). MSE in units of $10^{-\\rhcaMseExp}$. $R^2_{\\mathrm{OOS}}$ against the "
            "historical mean of the closing-half-hour realized variance from \\rhoosBenchStart{} \\citep{CampbellThompson2008}. "
            "Full vs HAR: $\\Delta\\% = 100(\\bar L^{\\mathrm{Full}}/\\bar L^{\\mathrm{HAR}} - 1)$ for the mean QLIKE of the "
            "same model; vs closing HAR: the same change against the closing-interval HAR benchmark ($^{a}$, the OLS "
            "on the HAR benchmark estimated on the closing half hour, Appendix~\\ref{app:rh_closing_bench}), which has "
            "no forecast before \\rhcaCloseFirst{} (--: not available on Panel~A's span); the closing-interval ridge, "
            "lasso and elastic net on the HAR benchmark are within \\rhcaCiHarGapBound\\% of it. $\\DM$: "
            "Diebold--Mariano statistic of the "
            "daily QLIKE differential (Harvey--Leybourne--Newbold, $q_{\\mathrm{NW}} = \\rhcaDMLagSpan$ in Panel~A and "
            "\\rhcaDMLagOos{} in Panel~B); negative: the first forecast has the lower loss. $^{*}$, $^{**}$, $^{***}$: "
            "significance at the 10\\%, 5\\% and 1\\% levels. The penalized linear models' full-set changes differ between "
            "the panels because the two halves of Panel~A's span disagree: on \\rhcaYearsPre{} the full set lowers their "
            "closing-half-hour QLIKE ($\\Delta\\% = \\rhcaPctRidgeClosePre$, \\rhcaPctLassoClosePre{} and "
            "\\rhcaPctEnetClosePre{} for ridge, lasso and elastic net; $\\DM = \\rhcaDmRidgeClosePre$, "
            "\\rhcaDmLassoClosePre{} and \\rhcaDmEnetClosePre), on $\\mathcal T_{\\mathrm{oos}}$ it raises it, and over "
            "the whole span the two offset."
        ),
        "\\end{tablenotes}",
        "\\end{threeparttable}",
        "\\end{table}",
    ]
    OUT_ACC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_calibration_table(CAL: pd.DataFrame) -> None:
    rows_def = [("baseline", "lin")] + [("baseline", e) for e in TREE] + [("all_features", e) for e in EST]

    def cells(r: pd.Series) -> list[str]:
        return [
            f"${r.mean_r:.2f}$",
            f"${r.ratio_means:.2f}$",
            cell(100 * r.under, 1, starred=False),
            cell(r.mae * 10**UNIT_EXP, 2, starred=False),
            cell(r.mz_a * 10**UNIT_EXP, 2, n_stars(r.mz_p_a)),
            cell(r.mz_b, 2, n_stars(r.mz_p_b)),
            stat_col(r.mz_se_b),
            cell(r.mz_r2, 2, starred=False),
            cell(r.mzl_a, 2, n_stars(r.mzl_p_a)),
            cell(r.mzl_b, 2, n_stars(r.mzl_p_b)),
            stat_col(r.mzl_se_b),
            cell(r.mzl_r2, 2, starred=False),
        ]

    body: list[str] = []
    panels = [
        ("Bar", "Pre", "Panel A: every bar, \\rhcaYearsPre"),
        ("Bar", "Oos", "Panel B: every bar, \\rhcaYearsOos{} ($\\mathcal T_{\\mathrm{oos}}$)"),
        ("Close", "Pre", "Panel C: closing half hour (16:00 bar), \\rhcaYearsPre"),
        ("Close", "Oos", "Panel D: closing half hour (16:00 bar), \\rhcaYearsOos{} ($\\mathcal T_{\\mathrm{oos}}$)"),
    ]
    for j, (sample, tag, title) in enumerate(panels):
        body += ([] if j == 0 else ["\\addlinespace[4pt]"]) + [
            f"\\multicolumn{{14}}{{@{{}}l}}{{\\textit{{{title}}}}} \\\\",
            "\\addlinespace[2pt]",
        ]
        sub = CAL[(CAL["sample"] == sample) & (CAL.span == tag)]
        for s, e in rows_def:
            if e == "lin":
                rr = [sub[(sub.model == m) & (sub.set == s)].iloc[0] for m in ("ols",) + LIN]
                cc = [cells(x) for x in rr]
                assert all(c == cc[0] for c in cc), ("the HAR linear rows differ from the OLS row", sample, tag)
                label, c = "Linear$^{a}$", cc[0]
            else:
                label, c = NAME[e], cells(sub[(sub.model == e) & (sub.set == s)].iloc[0])
            first = (s == "baseline" and e == "lin") or (s == "all_features" and e == EST[0])
            body.append(" & ".join([SETNAME[s] if first else "", label] + c) + " \\\\")
    lines = [
        HEAD,
        "% tab:rh_calibration: calibration diagnostics of the 12 pooled forecasts, every bar and the 16:00 bar,",
        "% 2014-2019 and 2020-2024; Mincer-Zarnowitz regressions with Newey-West standard errors.",
        "\\ifdefined\\rhcaNdaysSpan\\else\\input{generated/rehaul_close_accuracy_numbers}\\fi",
        "\\begin{table}[tbp]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.5pt}",
        "\\begin{threeparttable}",
        "\\caption{Calibration of the pooled intraday forecasts}",
        "\\label{tab:rh_calibration}",
        "\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}llrrrrrrrrrrrr@{}}",
        "\\toprule",
        " & & & & & & \\multicolumn{4}{c}{Levels: $\\RV = a + b\\,\\hat\\sigma^2$} & "
        "\\multicolumn{4}{c}{Logs: $\\ln\\RV = a + b\\ln\\hat\\sigma^2$} \\\\",
        "\\cmidrule(lr){7-10}\\cmidrule(l){11-14}",
        "Set & Model & $\\overline{r}$ & $\\overline{\\RV}/\\overline{\\hat\\sigma^2}$ & Under (\\%) & MAE & $a$ & $b$ & "
        "(s.e.) & $R^2$ & $a$ & $b$ & (s.e.) & $R^2$ \\\\",
        "\\midrule",
        *body,
        "\\bottomrule",
        "\\end{tabular*}",
        "\\begin{tablenotes}[flushleft]",
        "\\footnotesize",
        (
            "\\item \\textit{Notes:} The forecasts of Table~\\ref{tab:rh_oos} (HAR: HAR benchmark; Full: full predictor "
            "set), scored against the actual realized variance on every bar (Panels~A and B, the bars of "
            "Table~\\ref{tab:rh_oos}) and on the 16:00 bar (Panels~C and D, one forecast a day); \\rhcaYearsPre{} runs from "
            "\\rhcaStartPre{} ($\\rhcaNdaysPre$ days) and \\rhcaYearsOos{} is $\\mathcal T_{\\mathrm{oos}}$ "
            "($\\rhcaNdaysOos$ days). $\\overline{r}$: the mean of $r_t = \\RV_t/\\hat\\sigma^2_t$; "
            "$\\overline{\\RV}/\\overline{\\hat\\sigma^2}$: the ratio of the means; Under: the share of forecasts below the "
            "realized variance; MAE and the levels intercept $a$ in units of $10^{-\\rhcaUnitExp}$. Levels and Logs: "
            "Mincer--Zarnowitz regressions by OLS, Newey--West standard errors (Bartlett kernel; $\\rhcaHacLagBarPre$ and "
            "$\\rhcaHacLagBarOos$ bars in Panels~A and B, $\\rhcaDMLagPre$ and $\\rhcaDMLagOos$ days in Panels~C and D); "
            "stars test $a = 0$ and $b = 1$: $^{*}$, $^{**}$, $^{***}$ at the 10\\%, 5\\% and 1\\% levels. "
            "$^{a}$~OLS, ridge, lasso and elastic net, whose forecasts on the HAR benchmark coincide "
            "(largest relative difference $\\rhcaHarLinGap$)."
        ),
        "\\end{tablenotes}",
        "\\end{threeparttable}",
        "\\end{table}",
    ]
    OUT_CAL.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- outputs: figure


def write_figure(DEC: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#52514e",
            "axes.linewidth": 0.6,
            "xtick.color": "#52514e",
            "ytick.color": "#52514e",
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.6))
    titles = {
        ("Bar", "Pre"): "Every bar, 2014--2019",
        ("Bar", "Oos"): "Every bar, 2020--2024",
        ("Close", "Pre"): "Closing half hour, 2014--2019",
        ("Close", "Oos"): "Closing half hour, 2020--2024",
    }
    sty = {
        ("enet", "baseline"): dict(color=COL_EN, ls="--", marker="o", mfc="white"),
        ("enet", "all_features"): dict(color=COL_EN, ls="-", marker="o", mfc=COL_EN),
        ("lgbm", "baseline"): dict(color=COL_GB, ls="--", marker="s", mfc="white"),
        ("lgbm", "all_features"): dict(color=COL_GB, ls="-", marker="s", mfc=COL_GB),
    }
    sub = DEC[DEC.model.isin(["enet", "lgbm"])]
    x_lo, x_hi = 10**UNIT_EXP * sub.mean_fc.min() / 1.3, 10**UNIT_EXP * sub.mean_fc.max() * 1.3
    r_lo, r_hi = min(sub.ratio.min(), 1.0) / 1.05, max(sub.ratio.max(), 1.0) * 1.05
    yt = [t for t in (0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6) if r_lo <= t <= r_hi]
    for i, sample in enumerate(("Bar", "Close")):
        for j, tag in enumerate(("Pre", "Oos")):
            ax = axes[i, j]
            ax.axhline(1.0, color="#9a9893", lw=0.8, ls=":", zorder=1)
            for (e, s), st in sty.items():
                g = DEC[(DEC["sample"] == sample) & (DEC.span == tag) & (DEC.model == e) & (DEC.set == s)].sort_values(
                    "decile"
                )
                xx = 10**UNIT_EXP * g.mean_fc
                # the line and the markers as two artists, so no PDF viewer draws the markers with the line's dashes
                ax.plot(xx, g.ratio, color=st["color"], ls=st["ls"], lw=1.2, zorder=3)
                ax.plot(
                    xx,
                    g.ratio,
                    ls="none",
                    marker=st["marker"],
                    ms=3.8,
                    mfc=st["mfc"],
                    mec=st["color"],
                    mew=0.9,
                    zorder=4,
                )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(r_lo, r_hi)
            ax.yaxis.set_major_locator(FixedLocator(yt))
            ax.yaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in yt]))
            ax.set_title(titles[(sample, tag)], loc="left", color="#0b0b0b")
            ax.grid(True, which="major", color="#e4e3df", lw=0.5)
            ax.set_axisbelow(True)
            if i == 1:
                ax.set_xlabel(r"mean forecast in the decile ($10^{-6}$, log scale)")
            if j == 0:
                ax.set_ylabel("mean RV / mean forecast")

    def handle(color: str, ls: str, marker: str, mfc: str) -> tuple[Line2D, Line2D]:
        """A legend entry as a line and a marker, two artists (as in the panels)."""
        return (
            Line2D([], [], color=color, ls=ls, lw=1.2),
            Line2D([], [], ls="none", marker=marker, ms=3.8, mfc=mfc, mec=color, mew=0.9),
        )

    handles = [
        handle(COL_EN, "--", "o", "white"),
        handle(COL_EN, "-", "o", COL_EN),
        handle(COL_GB, "--", "s", "white"),
        handle(COL_GB, "-", "s", COL_GB),
        Line2D([], [], color="#9a9893", ls=":", lw=0.8),
    ]
    labels = ["Elastic net, HAR", "Elastic net, Full", "LightGBM, HAR", "LightGBM, Full", "calibrated (ratio 1)"]
    fig.legend(
        handles=handles,
        labels=labels,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)},
        loc="upper center",
        ncol=5,
        frameon=False,
        fontsize=7,
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT_FIG)
    fig.savefig(RES / "rehaul_calibration.png", dpi=160)
    plt.close(fig)
    lines = [
        HEAD,
        "% fig:rh_calibration: decile calibration of the elastic net and LightGBM, HAR benchmark and full predictor set.",
        "\\ifdefined\\rhcaNdaysSpan\\else\\input{generated/rehaul_close_accuracy_numbers}\\fi",
        "\\begin{figure}[tbp]",
        "\\centering",
        "\\includegraphics[width=\\linewidth]{generated/rehaul_calibration.pdf}",
        (
            "\\caption{Calibration by forecast decile. Within each decile of the forecast, the mean realized variance "
            "relative to the mean forecast (vertical axis; 1: calibrated, above 1: the forecasts are too low), plotted "
            "at the decile's mean forecast (horizontal axis), for the pooled elastic net and LightGBM on the HAR "
            "benchmark (open markers, dashed) and on the full predictor set (filled markers, solid); log scales. Top: "
            "every bar of Table~\\ref{tab:rh_oos}; bottom: the 16:00 bar (the forecast of the closing half hour). Left: "
            "\\rhcaYearsPre{} ($\\rhcaNdaysPre$ days); right: \\rhcaYearsOos{} ($\\mathcal T_{\\mathrm{oos}}$, "
            "$\\rhcaNdaysOos$ days). On every bar in \\rhcaYearsOos, the ratio of the full-set elastic net falls from "
            "\\rhcaBotDecRatioEnetFullBarOos{} in the bottom decile to \\rhcaTopDecRatioEnetFullBarOos{} in the top "
            "decile (HAR benchmark: \\rhcaBotDecRatioEnetHarBarOos{} and \\rhcaTopDecRatioEnetHarBarOos), and the top "
            "decile carries \\rhcaTopDecSeEnetFullBarOos\\% of its squared error.}"
        ),
        "\\label{fig:rh_calibration}",
        "\\end{figure}",
    ]
    OUT_FIGTEX.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- main


def check_macro_names(names: set[str]) -> None:
    """No \\rhca name is defined in another generated file, and none of ours collides with an existing macro."""
    for f in sorted(set(GEN.glob("*.tex")) | set(SRC_GEN.glob("*.tex"))):
        if f.name in {OUT_NUM.name, OUT_ACC.name, OUT_CAL.name, OUT_FIGTEX.name}:
            continue
        txt = f.read_text(encoding="utf-8", errors="replace")
        assert "command{\\rhca" not in txt, f
        for m in re.finditer(r"command\{\\([A-Za-z]+)\}", txt):
            assert m.group(1) not in names, (f, m.group(1))


def used_macros_defined(names: set[str]) -> None:
    """Every \\rhca macro the floats use is defined."""
    for f in (OUT_ACC, OUT_CAL, OUT_FIGTEX):
        for m in re.finditer(r"\\(rhca[A-Za-z]+)", f.read_text(encoding="utf-8")):
            assert m.group(1) in names, (f.name, m.group(1))


def claims(
    A16: pd.DataFrame, ABar: pd.DataFrame, CAL: pd.DataFrame, EXT: pd.DataFrame, DEC: pd.DataFrame, YR: pd.DataFrame
) -> None:
    """The statements of the draft (writeup/drafts/2026-10-25_close_accuracy.tex) that name classes or directions
    rather than print a number: each is asserted here, so a rerun that breaks one stops before writing."""
    Z = 1.959963984540054
    f16 = A16[(A16.scheme == "pool") & (A16.set == "all_features")].set_index(["span", "model"])

    def lower(tag: str) -> set[str]:
        return {e for e in EST if f16.loc[(tag, e), "pct"] < 0}

    def sig(tag: str) -> set[str]:
        return {e for e in EST if f16.loc[(tag, e), "p"] < SIG}

    # the closing half hour, full set vs HAR, per span
    assert lower("Span") == {"enet", "lgbm", "xgb", "rf"} and sig("Span") == {"xgb"}, (lower("Span"), sig("Span"))
    assert lower("Oos") == set(TREE) and sig("Oos") == {"ridge", "lasso"}, (lower("Oos"), sig("Oos"))
    assert lower("Pre") == set(EST), lower("Pre")
    # every-bar MSE: higher with the full set in every class on Span and Oos, lower on Pre
    fb = ABar[(ABar.set == "all_features") & ABar.model.isin(EST)].set_index(["span", "model"])
    for tag, sign in (("Span", 1), ("Oos", 1), ("Pre", -1)):
        assert all(sign * fb.loc[(tag, e), "mse_pct"] > 0 for e in EST), tag
    # without the crash window, the full set has the lower every-bar MSE and MAE in every class, significantly
    ex = EXT[EXT.span == "Oos"].set_index("model")
    assert ((ex.bar_mse_pct_excrash < 0) & (ex.bar_mse_p_excrash < SIG)).all()
    assert ((ex.bar_mae_pct_excrash < 0) & (ex.bar_mae_p_excrash < SIG)).all()
    # the top days and the largest linear forecast fall inside the crash window
    for days in EXT.top_days:
        assert all(CRASH[0] <= pd.Timestamp(d) <= CRASH[1] for d in days.split()), days
    for t in ex.loc[list(LIN), "max_fc_time"]:
        assert CRASH[0] <= pd.Timestamp(t).normalize() <= CRASH[1], t
    c = CAL.set_index(["sample", "span", "model", "set"])
    # levels MZ slopes: full-set slopes significantly below one on every bar of Oos, the HAR linear slope is not
    for e in EST:
        r = c.loc[("Bar", "Oos", e, "all_features")]
        assert r.mz_b + Z * r.mz_se_b < 1.0, (e, r.mz_b, r.mz_se_b)
    r = c.loc[("Bar", "Oos", "ols", "baseline")]
    assert abs(r.mz_b - 1.0) < Z * r.mz_se_b, (r.mz_b, r.mz_se_b)
    # ... and on Pre the linear full-set slopes are not significantly different from one
    for e in LIN:
        r = c.loc[("Bar", "Pre", e, "all_features")]
        assert abs(r.mz_b - 1.0) < Z * r.mz_se_b, (e, r.mz_b, r.mz_se_b)
    # logs MZ R2: the full set above the HAR benchmark of the same class, every bar, both periods
    for tag in ("Pre", "Oos"):
        for e in EST:
            assert c.loc[("Bar", tag, e, "all_features")].mzl_r2 > c.loc[("Bar", tag, e, "baseline")].mzl_r2, (tag, e)
    # no level bias: the Theil bias term below 1 % of every pooled forecast's MSE
    assert (CAL.sh_bias < 1.0).all(), CAL.sh_bias.max()
    # the closing half hour on Oos: the linear full-set forecasts sit below the HAR forecasts, under-predict more
    # often, and the level term carries most of their QLIKE increase
    for e in LIN:
        f, h = c.loc[("Close", "Oos", e, "all_features")], c.loc[("Close", "Oos", e, "baseline")]
        assert f.under > h.under and f.mean_r > h.mean_r and f.qlike > h.qlike
        assert (f.q_level - h.q_level) / (f.qlike - h.qlike) > 0.5, e
    # the deciles: the full-set elastic net's top decile over-predicts on every bar of Oos, its bottom decile
    # under-predicts more than the HAR benchmark's
    dd = DEC[(DEC["sample"] == "Bar") & (DEC.span == "Oos") & (DEC.model == "enet")].set_index(["set", "decile"])
    assert dd.loc[("all_features", N_DEC), "ratio"] < 1.0 < dd.loc[("baseline", N_DEC), "ratio"]
    assert dd.loc[("all_features", 1), "ratio"] > dd.loc[("baseline", 1), "ratio"] > 1.0
    # 16:00 MSE: higher with the full set on the Span in no class significantly; on Pre lower in every class, on every
    # bar and at 16:00, and so is the MAE
    assert not ((f16.loc["Span", "mse_pct"] > 0) & (f16.loc["Span", "mse_p"] < SIG)).any()
    assert (f16.loc["Pre", "mse_pct"] < 0).all()
    for sample in ("Bar", "Close"):
        for e in EST:
            assert c.loc[(sample, "Pre", e, "all_features")].mae < c.loc[(sample, "Pre", e, "baseline")].mae, (
                sample,
                e,
            )
    # every pooled forecast has the higher 16:00 QLIKE than the closing-interval HAR benchmark on Oos; the lowest
    # QLIKE of the Oos panel is the closing-interval lasso on the full set
    po = A16[(A16.span == "Oos") & (A16.scheme == "pool") & (A16.model != "ols")]
    assert (po.gap > 0).all(), po.gap.min()
    oo = A16[A16.span == "Oos"]
    lo = oo.loc[oo.qlike.idxmin()]
    assert (lo.scheme, lo.model, lo.set) == ("close", "lasso", "all_features"), lo
    # the calendar years of Oos: the three linear models are worse in the same number of years, most in the same year
    yo = YR[(YR.year >= SPANS["Oos"][0].year) & YR.model.isin(LIN)]
    assert yo.groupby("model").pct.apply(lambda s: int((s > 0).sum())).nunique() == 1
    assert yo.loc[yo.groupby("model").pct.idxmax()].year.nunique() == 1
    # the closing half hour on Oos: the trees' level term rises too, but their dispersion term falls by more; the
    # linear models' dispersion term rises
    for e in TREE:
        f, h = c.loc[("Close", "Oos", e, "all_features")], c.loc[("Close", "Oos", e, "baseline")]
        assert f.q_level > h.q_level and (h.q_disp - f.q_disp) > (f.q_level - h.q_level), e
    for e in LIN:
        assert c.loc[("Close", "Oos", e, "all_features")].q_disp > c.loc[("Close", "Oos", e, "baseline")].q_disp, e
    print("claims: all hold")


def main() -> None:
    print(fr.tag())
    D = load()
    A16 = accuracy16(D)
    ABar = accuracy_bar(D)
    run_gates(D, A16, ABar)
    CAL = calibration(D)
    EXT, _ = extremes(D)
    YR = yearly16(D)
    DEC = deciles(D)
    claims(A16, ABar, CAL, EXT, DEC, YR)
    RES.mkdir(parents=True, exist_ok=True)
    A16.to_csv(RES / "accuracy16.csv", index=False)
    ABar.to_csv(RES / "accuracy_bar.csv", index=False)
    CAL.to_csv(RES / "calibration.csv", index=False)
    EXT.to_csv(RES / "extremes.csv", index=False)
    YR.to_csv(RES / "yearly16.csv", index=False)
    DEC.to_csv(RES / "deciles.csv", index=False)
    V = write_macros(D, A16, ABar, CAL, EXT, YR, DEC)
    check_macro_names(V["names"])
    write_accuracy_table(A16)
    write_calibration_table(CAL)
    write_figure(DEC)
    used_macros_defined(V["names"])
    print(
        f"wrote {fr.rel(OUT_NUM)} ({len(V['names'])} macros), {fr.rel(OUT_ACC)}, {fr.rel(OUT_CAL)}, "
        f"{fr.rel(OUT_FIG)}, {fr.rel(OUT_FIGTEX)}, {fr.rel(RES)}/"
    )


if __name__ == "__main__":
    main()
