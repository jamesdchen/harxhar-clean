"""Interval-shift test and compounding on the 12 pooled intraday forecasts of the final run (2026-10-23; notes E26,
N15b, N18 and E27 of writeup/rehaul/07_economics.tex and 10_appendix.tex).

WHAT
  1. Interval-shift test (E26 / N15b / N18).  The method of experiments/close_shift_test_headline.py (main() and
     write_numbers(), 2026-10-06 / 2026-10-15), applied to the pooled intraday forecasts of the final run instead of
     the earlier draft's nine 48-bar forecasts.  For each forecast and each offset j in -12 .. +6 the sign strategy
     reads the forecast of another regular-hours bar, placed on the traded bar's scale,
         RV_hat(j) = yhat_j^2 * B_16:00,     q_t = sign(RV_hat(j) - IV_t^2),
     with yhat_j the adjusted-scale forecast (pred_adj) of bar 16:00 + j half hours of day t for j <= 0 (j = -12 is
     the 10:00 bar, j = 0 the traded 15:30-16:00 bar, whose placed forecast is the plain forecast the economics uses,
     gated), and of the next session's 10:00 + (j - 1) half hours bar for j > 0 (the next session's 10:00 .. 12:30
     bars), and B_16:00 the run's diurnal baseline of the traded bar on day t.  The pooled models have exactly the 13
     regular-hours bars (10:00 .. 16:00 bar ends) that the original test shifted through (SESSION_BARS there), so the
     offsets are the original ones; nothing new is chosen.  Every forecast and offset is scored on the same days:
     the post-warmup option days t > WARMUP on which every shifted forecast exists (the last option day has no next
     session).  Per j: the median Sharpe ratio across the forecasts, its range, how many fall below their own j = 0
     value, the median change against j = 0 and the median share of days whose position agrees with j = 0.  The
     swing statistics are write_numbers()'s: the smallest and largest median over every j (j = 0 included), half that
     range, and the smallest and largest median change against j = 0 (j != 0); here also over j <= 0 alone (the
     offsets that read no later information; j > 0 reads day t + 1, a placebo).
     What changes against the original (the task's two fixes):
       Sharpe ratio  the paper's (D6, writeup/hac_inference.d6_sharpe, Section 7.1): the equal-weight mean over
                     calendar years of SR_y = sqrt(A_y) mean_y / sd_y, A_y = N_y / c_y, on exactly the days scored;
                     the original used sqrt(252) mean / sd on all option days.
       days          the post-warmup option days t > WARMUP (the sample of every printed Sharpe ratio), not all
                     866 option days.
     Sampling band (\\rhpsNoiseBand): the original's 1.96 sqrt(252 / n) is the 95 % half-width of sqrt(252) times
     the daily Sharpe ratio under i.i.d. returns at SR = 0 (Lo 2002: Var(mean / sd) = 1 / n).  The same band of the
     D6 ratio on the same assumptions: SR_y = sqrt(A_y) mean_y / sd_y has variance A_y / N_y = 1 / c_y, the years
     are independent, so SR = mean_y SR_y has variance sum_y (1 / c_y) / K^2 (K years) and the band is
     1.96 sqrt(sum_y A_y / N_y) / K.  It reduces to 1.96 sqrt(252 / n) when every A_y = 252 and K = 1.
  2. Compounding (E27).  The method of experiments/close_compounding_headline.py (main(), wealth_stats, 2026-10-21):
     a fixed fraction F_FIXED = 0.03 of wealth as straddle premium every day, V = prod_t (1 + f r_t),
     G = 252 mean_t log(1 + f r_t), maximum drawdown max_t (1 - V_t / max_{t' <= t} V_t'), on the post-warmup days,
     at mid-quotes (r_t = q_t R_t) and at quoted bid/ask prices; per forecast, then the median, range and count above
     one unit of wealth across the forecasts, and the largest drawdown.  Only the forecasts change: the 12 pooled
     intraday forecasts (note E25) instead of the master table's model set.

FORECASTS  (paths and constants from writeup/final_run.py; the run files are read directly, not through the pooled
  loaders pooled_linear_13.py / pooled_trees_13.py, whose logic is copied where needed, cited at each copy)
  linear  final_run.LINEAR_RUN  ridge, lasso, elastic net on the HAR benchmark (baseline) and on all features
          (the min-norm OLS row of the run is not one of the 12 and is not read)
  trees   final_run.TREES_RUN   LightGBM, XGBoost (model keys lgbm, xgb), the same two predictor sets
  forest  final_run.RF_RUN      the random forest on oneDAL (model key final_run.RF_KEY), the same two sets
  An arm whose forecasts.parquet is missing is PENDING: its cells print \\rhPending, and so does every statistic over
  the set (a median over part of the 12 is not the set's median).

MARKET  (as writeup/make_rehaul_economics.py load() / add_pooled_trees and experiments/master_table_close.trade_days)
  the 866 option days and the straddle (R, exit, bid, ask) of the quote book (master_table_gui.rows._QUOTE_BOOK,
  results/atm_straddle_0dte_1530/daily_a0.parquet), the implied variance iv_var and realized variance rv_unclipped of
  the OLS row (sub_ols_baseline) of results/close_master_table/master_table_daily.parquet; gated equal to the deck the
  shift test read (results/atm_straddle_0dte_1530/daily_blk2.parquet).  q = sign(forecast - iv_var); mid return
  q R; bid/ask return q (exit / ask - 1) when long and q (exit / bid - 1) when short.

SELF-CHECKS  (run first, on the inputs the printed numbers were computed from; the script stops on a failure)
  A  shift test: this module's code with the original's sample and Sharpe ratio (sqrt(252), the 866 option days) on
     the earlier draft's nine forecasts (results/spxw_pnl/yhat_*.parquet) reproduces
     results/close_shift_test/shift_test_by_k.csv (every j, every column, 1e-9) and the six macros of
     writeup/generated/close_shift_test_numbers.tex (as committed at PRINTED_REF) character for character.
  B  compounding: wealth statistics of the master table's model set (paper_model_set.csv, master_table_daily.parquet
     returns) on the 614 post-warmup days reproduce results/close_compounding/fixedfrac_modelset_summary.csv (1e-12)
     and the four macros of close_compounding_numbers.tex (PRINTED_REF).
  C  the pooled trade and the D6 ratio: this module's j = 0 trade of the printed pooled runs (13 bars, W = 1,000,
     results/pooled_linear_rerun_2026-10-20_13_2012_w1000, results/pooled_trees_rerun13_2026-10-20/w1000) on the
     614 post-warmup days reproduces the 24 printed Sharpe ratios \\rhrSh... / \\rhrShX...Pool{Har,Full} of
     writeup/generated/rehaul_results_numbers.tex (PRINTED_REF) to their two printed decimals.

OUTPUTS
  results/rehaul_pooled_extras_final_w<W>/  arms.csv, shift_by_arm.csv, shift_by_k.csv, compounding_by_arm.csv,
      compounding_summary.csv, selfcheck_*.csv, gates.csv; shift_by_{arm,k}_jle0_614days.csv: an ALTERNATIVE for the
      owner, not the original's method (the offsets j <= 0 alone on all 614 post-warmup days, so j = 0 is the
      printed Sharpe ratio; the common days of the original drop 30 April 2024, R = +3.8, which moves every D6
      ratio by 0.14-0.20); macros \\rhpsPw...
  writeup/generated/rehaul_pooled_extras_numbers.tex  macros \\rhps... (shift test), \\rhpc... (compounding)
  writeup/generated/rehaul_pooled_extras_table.tex    a per-forecast table (optional; the text needs only the macros)
  The two .tex files go to final_run.GEN at the main window (W = 3,000) and to the results directory otherwise.

RUN  (repository root)  python writeup/make_rehaul_pooled_extras.py
"""

from __future__ import annotations

import io
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "writeup", ROOT / "notebooks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import hac_inference as hi  # noqa: E402  (the paper's D6 Sharpe ratio)
from master_table_gui import rows as gui  # noqa: E402  (the quote book and the sizing warmup)


def _import_final_run():
    """final_run is being edited by another agent; retry a transient import failure for a minute."""
    for attempt in range(7):
        try:
            import final_run

            return final_run
        except Exception as e:  # noqa: BLE001
            if attempt == 6:
                raise
            print(f"import final_run failed ({e!r}); retrying in 10 s")
            sys.modules.pop("final_run", None)
            time.sleep(10)
    raise RuntimeError("unreachable")


fr = _import_final_run()

# ---------------------------------------------------------------------------------------------- constants
WARMUP = (
    gui._SIZING_WARMUP
)  # 252: the days t > WARMUP are the sample of every printed Sharpe ratio
assert WARMUP == 252, WARMUP
N_OPT = 866  # option days, 2020-01-03 .. 2024-04-30
N_PW = N_OPT - WARMUP  # 614 post-warmup days
F_FIXED = 0.03  # experiments/close_compounding_headline.py F_FIXED (\rhappCompF = 3)
PPY = 252.0  # close_compounding_headline.PPY = atm_straddle_lib.PERIODS_PER_YEAR: G's factor (not a Sharpe ratio)
GATE_REL = 1e-9
# experiments/close_shift_test_headline.py: SESSION_BARS, SAME_SHIFTS, NEXT_SHIFTS, ALL_SHIFTS, bar_of (copied)
SESSION_BARS = [f"{h:02d}:{m:02d}" for h in range(10, 17) for m in (0, 30)][:13]
SAME_SHIFTS = list(range(-12, 1))  # -12 = the 10:00 bar ... 0 = the 16:00 bar (traded)
NEXT_SHIFTS = list(range(1, 7))  # +1 .. +6 = the next session's 10:00 .. 12:30 bars
ALL_SHIFTS = SAME_SHIFTS + NEXT_SHIFTS
BAND_Z = 1.96  # close_shift_test_headline.write_numbers: 1.96 sqrt(252 / n)

DECK = (
    ROOT / "results" / "atm_straddle_0dte_1530" / "daily_blk2.parquet"
)  # score_linear_subsection.DECK
MT_DIR = ROOT / "results" / "close_master_table"
OLS_KEY = "sub_ols_baseline"  # make_rehaul_economics.OLS: the row whose iv_var / rv_unclipped every pooled row reads

EST_NAME = {
    "ridge": "Ridge",
    "lasso": "Lasso",
    "enet": "Elastic net",
    "lgbm": "LightGBM",
    "xgb": "XGBoost",
    "rf": "Random forest",
}
EST_MACRO = {
    "ridge": "Ridge",
    "lasso": "Lasso",
    "enet": "Enet",
    "lgbm": "Lgbm",
    "xgb": "Xgb",
    "rf": "Rf",
}
SET_NAME = {"baseline": "HAR", "all_features": "Full"}
SET_MACRO = {
    "baseline": "Har",
    "all_features": "Full",
}  # the \rhrSh...Pool{Har,Full} names
SETS = ("baseline", "all_features")
LIN_EST = ("ridge", "lasso", "enet")
TREE_EST = ("lgbm", "xgb", "rf")

WINDOW = fr.WINDOW_DAYS
OUT = ROOT / "results" / f"rehaul_pooled_extras_final_w{WINDOW}"
GEN = fr.GEN if WINDOW == 3000 else OUT
NUMBERS_TEX = GEN / "rehaul_pooled_extras_numbers.tex"
TABLE_TEX = GEN / "rehaul_pooled_extras_table.tex"
PENDING = r"\rhPending"
PROVIDE_PENDING = r"\providecommand{\rhPending}{\textit{pending}}"

# the printed runs of self-check C (pooled_linear_13.RUN, pooled_trees_13.RUN before the final switch)
PRINTED_LINEAR = (
    ROOT
    / "results"
    / "pooled_linear_rerun_2026-10-20_13_2012_w1000"
    / "forecasts.parquet"
)
PRINTED_TREES = (
    ROOT / "results" / "pooled_trees_rerun13_2026-10-20" / "w1000" / "forecasts.parquet"
)

GATES: list[dict] = []


def gate(name: str, value: float, bound: float, n: int) -> None:
    GATES.append(
        {
            "gate": name,
            "value": float(value),
            "bound": bound,
            "n": n,
            "ok": bool(value <= bound),
        }
    )
    if not value <= bound:
        raise SystemExit(f"GATE FAILED: {name}: {value} > {bound} (n = {n})")


# The last commit whose generated files hold the printed (pre-final-run) numbers the self-checks reproduce; HEAD
# carries the final-run values since 7206771, so the reference is pinned here.
PRINTED_REF = "300a8d2"


def committed(rel: str) -> str:
    """A tracked file as committed at PRINTED_REF (the printed reference the self-checks reproduce)."""
    return subprocess.check_output(
        ["git", "show", f"{PRINTED_REF}:{rel}"], cwd=ROOT
    ).decode("utf-8")


def macro_values(tex: str) -> dict[str, str]:
    """name -> body of every one-line \\newcommand / \\providecommand{}\\renewcommand definition in a numbers file."""
    out = {}
    for m in re.finditer(
        r"\\(?:newcommand|renewcommand|providecommand)\{\\([A-Za-z]+)\}\{(.*?)\}(?:\s|%|$)",
        tex,
    ):
        out[m.group(1)] = m.group(2)
    return out


# ---------------------------------------------------------------------------------------------- Sharpe ratios
def sr_d6(x, dates) -> float:
    """The paper's Sharpe ratio (D6) on exactly the days given (make_rehaul_economics.sharpe_days)."""
    return float(
        hi.d6_sharpe(np.asarray(x, float), dates=pd.DatetimeIndex(dates))["sharpe"]
    )


def sr_fixed(x, dates) -> float:  # noqa: ARG001
    """master_table_close.sharpe: sqrt(252) mean / sd (ddof 1): the original shift test's ratio (self-check A)."""
    x = np.asarray(x, float)
    return float(x.mean() / x.std(ddof=1) * np.sqrt(252.0))


def band_d6(dates) -> float:
    """95 % half-width of the D6 ratio under i.i.d. returns at SR = 0: 1.96 sqrt(sum_y A_y / N_y) / K (docstring)."""
    yy = hi.d6_years(pd.DatetimeIndex(dates))
    k = len(yy["A"])
    return float(BAND_Z * np.sqrt(sum(yy["A"][y] / yy["N"][y] for y in yy["A"])) / k)


def band_fixed(dates) -> float:
    """close_shift_test_headline.write_numbers: 1.96 sqrt(252 / n)."""
    return float(BAND_Z * np.sqrt(252.0) / np.sqrt(len(dates)))


# ---------------------------------------------------------------------------------------------- market
def load_market() -> pd.DataFrame:
    """The 866 option days: R, exit, bid, ask (quote book), iv_var and rv_unclipped (the OLS row), gated against the
    deck the shift test read."""
    qb = pd.read_parquet(
        gui._QUOTE_BOOK, columns=["R", "exit", "bid_c", "bid_p", "ask_c", "ask_p"]
    )
    qb.index = pd.DatetimeIndex(pd.to_datetime(qb.index)).normalize().as_unit("ns")
    qb = qb.sort_index()
    deck = pd.read_parquet(
        DECK, columns=["R", "exit", "bid_c", "bid_p", "ask_c", "ask_p", "iv_var"]
    ).sort_index()
    deck.index = pd.DatetimeIndex(pd.to_datetime(deck.index)).normalize().as_unit("ns")
    d = pd.read_parquet(
        MT_DIR / "master_table_daily.parquet",
        columns=["key", "day", "iv_var", "rv_unclipped"],
    )
    o = d[d["key"] == OLS_KEY].copy()
    o.index = pd.DatetimeIndex(pd.to_datetime(o["day"])).normalize().as_unit("ns")
    o = o.sort_index()
    days = deck.index
    assert len(days) == N_OPT and qb.index.equals(days) and o.index.equals(days), (
        len(days),
        len(qb),
        len(o),
    )
    for c in ("R", "exit", "bid_c", "bid_p", "ask_c", "ask_p"):
        gate(
            f"quote book {c} = deck {c}",
            float(np.abs(qb[c].to_numpy(float) - deck[c].to_numpy(float)).max()),
            0,
            N_OPT,
        )
    gate(
        "OLS row iv_var = deck iv_var",
        float(
            np.abs(o["iv_var"].to_numpy(float) - deck["iv_var"].to_numpy(float)).max()
        ),
        0,
        N_OPT,
    )
    m = pd.DataFrame(
        {
            "R": qb["R"].to_numpy(float),
            "ex": qb["exit"].to_numpy(float),
            "bid": (qb["bid_c"] + qb["bid_p"]).to_numpy(float),
            "ask": (qb["ask_c"] + qb["ask_p"]).to_numpy(float),
            "iv": o["iv_var"].to_numpy(float),
            "rv": o["rv_unclipped"].to_numpy(float),
        },
        index=days,
    )
    assert np.isfinite(m.to_numpy()).all()
    return m


def trade(pred: pd.Series, mk: pd.DataFrame) -> pd.DataFrame:
    """master_table_close.trade_days, line for line (the days of mk the forecast covers): q, mid, x (bid/ask)."""
    f = pred.reindex(mk.index)
    ok = f.notna().to_numpy()
    dk = mk[ok]
    q = np.where(f.to_numpy(float)[ok] > dk["iv"].to_numpy(float), 1.0, -1.0)
    ex, r = dk["ex"].to_numpy(float), dk["R"].to_numpy(float)
    mid = q * r
    x = q * np.where(
        q > 0,
        ex / dk["ask"].to_numpy(float) - 1.0,
        ex / dk["bid"].to_numpy(float) - 1.0,
    )
    return pd.DataFrame({"q": q, "mid": mid, "x": x}, index=mk.index[ok])


# ---------------------------------------------------------------------------------------------- forecasts
def read_run(
    path: Path, models: dict[str, str], n_rows: int | None
) -> dict[tuple[str, str], pd.DataFrame]:
    """The arms (model -> name, both predictor sets) of one pooled forecasts.parquet, each a frame of its regular-
    hours bars (day, hhmm, pred_adj, B, plain, rv).  Gates (pooled_linear_13._frame / pooled_trees_13._frame): finite,
    plain = pred_adj^2 B to 1e-12, the 13 bars 10:00 .. 16:00, n_rows rows an arm, no duplicated bar."""
    f = pd.read_parquet(
        path,
        columns=[
            "model",
            "predictor_set",
            "date",
            "pred_adj",
            "B",
            "plain",
            "rv_actual",
        ],
    )
    f = f[f["model"].isin(list(models))]
    v = f[["pred_adj", "B", "plain", "rv_actual"]].to_numpy(float)
    assert np.isfinite(v).all() and (f["plain"] > 0).all() and (f["B"] > 0).all(), path
    rel = float(np.max(np.abs(f["plain"] / (f["pred_adj"] ** 2 * f["B"]) - 1.0)))
    gate(f"plain = pred_adj^2 B ({path.parent.name})", rel, 1e-12, len(f))
    t = pd.to_datetime(f["date"])
    if t.dt.tz is not None:
        t = t.dt.tz_convert("America/New_York").dt.tz_localize(None)
    f = f.assign(
        day=t.dt.normalize().astype("datetime64[ns]"), hhmm=t.dt.strftime("%H:%M")
    )
    assert sorted(f["hhmm"].unique()) == SESSION_BARS, sorted(f["hhmm"].unique())
    out = {}
    for (m, s), g in f.groupby(["model", "predictor_set"]):
        if s not in SETS:
            continue
        if n_rows is not None:
            assert len(g) == n_rows, (path, m, s, len(g), n_rows)
        assert not g.duplicated(["day", "hhmm"]).any(), (path, m, s)
        out[(models[m], s)] = g[
            ["day", "hhmm", "pred_adj", "B", "plain", "rv_actual"]
        ].rename(columns={"rv_actual": "rv"})
    assert set(out) == {(e, s) for e in models.values() for s in SETS}, (
        path,
        sorted(out),
    )
    return out


def standin_guard(run: Path) -> None:
    """final_run.guard_standin on the run's own record (run_info.json or forecasts.json)."""
    for name in ("run_info.json", "forecasts.json"):
        p = run / name
        if p.is_file():
            fr.guard_standin(json.loads(p.read_text()), run)


def final_arms() -> tuple[dict[tuple[str, str], pd.DataFrame], list[dict]]:
    """The 12 pooled forecasts of the final run; an arm whose run has no forecasts.parquet is pending."""
    spec = (
        (fr.LINEAR_RUN, {e: e for e in LIN_EST}, fr.N_LINEAR_ROWS, LIN_EST),
        (fr.TREES_RUN, {"lgbm": "lgbm", "xgb": "xgb"}, fr.N_BARS, ("lgbm", "xgb")),
        (fr.RF_RUN, {fr.RF_KEY: "rf"}, fr.N_BARS, ("rf",)),
    )
    arms: dict[tuple[str, str], pd.DataFrame] = {}
    status = []
    for run, models, n_rows, ests in spec:
        fc = run / "forecasts.parquet"
        ok = fc.is_file()
        if ok:
            standin_guard(run)
            arms.update(read_run(fc, models, n_rows))
        for e in ests:
            for s in SETS:
                status.append(
                    {
                        "est": e,
                        "set": s,
                        "status": "finished" if ok else "pending",
                        "forecasts": fr.rel(fc),
                    }
                )
    return arms, status


def shifted(
    panel: pd.DataFrame, mk: pd.DataFrame, plain_ref: pd.Series | None, name: str
) -> dict[int, pd.Series]:
    """close_shift_test_headline.main(): every offset's forecast on the option days, placed as yhat_j^2 B_16:00.
    Gates: j = 0 equals the plain 16:00 forecast (the run's own column, or plain_ref when given); the 16:00 bar's
    realized variance equals the OLS row's rv_unclipped (make_rehaul_economics.tree_16 / lin_16), when stored."""
    days = mk.index
    sessions = pd.DatetimeIndex(sorted(panel["day"].unique()))
    nxt = pd.Series(
        sessions[1:], index=sessions[:-1]
    )  # session -> the next session in the panel
    by_bar = {b: panel[panel["hhmm"] == b].set_index("day") for b in SESSION_BARS}
    b16 = by_bar["16:00"]["B"].reindex(days)
    out: dict[int, pd.Series] = {}
    for j in ALL_SHIFTS:
        same = j <= 0
        bar = SESSION_BARS[12 + j] if same else SESSION_BARS[j - 1]
        key_days = days if same else pd.DatetimeIndex(nxt.reindex(days).to_numpy())
        yk = by_bar[bar]["pred_adj"].reindex(key_days)
        yk.index = days
        out[j] = yk**2 * b16
    ref = (
        by_bar["16:00"]["plain"].reindex(days)
        if plain_ref is None
        else plain_ref.reindex(days)
    )
    gate(
        f"j = 0 placed = the plain 16:00 forecast ({name})",
        float((out[0] / ref - 1).abs().max()),
        GATE_REL,
        len(days),
    )
    if "rv" in panel:
        rv16 = by_bar["16:00"]["rv"].reindex(days).to_numpy(float)
        gate(
            f"16:00 realized variance = the OLS row's rv_unclipped ({name})",
            float(np.max(np.abs(rv16 / mk["rv"].to_numpy(float) - 1))),
            1e-12,
            len(days),
        )
    return out


# ---------------------------------------------------------------------------------------------- the shift test
def shift_test(
    F: dict[str, dict[int, pd.Series]],
    mk: pd.DataFrame,
    eval_days: pd.DatetimeIndex,
    sr,
    band,
):
    """close_shift_test_headline.main() from 'every forecast at every shift' to the per-j table, with the Sharpe
    ratio sr(x, dates) and the band band(dates).  Returns (long, by_k, common days)."""
    have = np.ones(len(eval_days), bool)
    for Fk in F.values():
        have &= pd.DataFrame(Fk).reindex(eval_days).notna().all(axis=1).to_numpy()
    common = eval_days[have]
    mkc = mk.loc[common]
    rows = []
    for key, Fk in F.items():
        t0 = trade(Fk[0], mk.loc[eval_days])
        assert len(t0) == len(eval_days), key
        t0c = trade(Fk[0], mkc)
        q0 = t0c["q"].to_numpy()
        for j, series in Fk.items():
            t = trade(series, mkc)
            assert t.index.equals(t0c.index)
            rows.append(
                {
                    "key": key,
                    "k": j,
                    "session": "same" if j <= 0 else "next",
                    "clock": SESSION_BARS[12 + j] if j <= 0 else SESSION_BARS[j - 1],
                    "n_days": len(t),
                    "Sharpe_mid": sr(t["mid"], common),
                    "Sharpe_crossed": sr(t["x"], common),
                    "pct_buy": float(100 * (t["q"] > 0).mean()),
                    "agree_with_bar0_pct": float(
                        100 * (t["q"].to_numpy() == q0).mean()
                    ),
                    "dSharpe_mid_vs_bar0": sr(t["mid"], common)
                    - sr(t0c["mid"], common),
                    "dSharpe_crossed_vs_bar0": sr(t["x"], common)
                    - sr(t0c["x"], common),
                    "Sharpe_mid_eval_days_at_k0": sr(t0["mid"], eval_days),
                    "Sharpe_crossed_eval_days_at_k0": sr(t0["x"], eval_days),
                }
            )
    long = pd.DataFrame(rows)
    by_k = []
    for j in sorted(long["k"].unique()):
        x = long[long["k"] == j]
        by_k.append(
            {
                "row": f"k={j:+d}",
                "k": j,
                "session": "same" if j <= 0 else "next",
                "clock": SESSION_BARS[12 + j] if j <= 0 else SESSION_BARS[j - 1],
                "n_forecasts": len(x),
                "n_days": len(common),
                "Sharpe_mid_median": float(x["Sharpe_mid"].median()),
                "Sharpe_mid_min": float(x["Sharpe_mid"].min()),
                "Sharpe_mid_max": float(x["Sharpe_mid"].max()),
                "Sharpe_crossed_median": float(x["Sharpe_crossed"].median()),
                "n_mid_below_k0": int((x["dSharpe_mid_vs_bar0"] < 0).sum()),
                "n_crossed_below_k0": int((x["dSharpe_crossed_vs_bar0"] < 0).sum()),
                "dSharpe_mid_median": float(x["dSharpe_mid_vs_bar0"].median()),
                "agree_median_pct": float(x["agree_with_bar0_pct"].median()),
                "pct_buy_median": float(x["pct_buy"].median()),
            }
        )
    # the rows that read no forecast, on the same days (realized variance of the traded bar; always short)
    t = trade(mk["rv"], mkc)
    by_k.append(
        {
            "row": "realized (unclipped)",
            "session": "traded bar",
            "clock": "16:00",
            "n_forecasts": 0,
            "n_days": len(t),
            "Sharpe_mid_median": sr(t["mid"], common),
            "Sharpe_crossed_median": sr(t["x"], common),
            "pct_buy_median": float(100 * (t["q"] > 0).mean()),
        }
    )
    r_mid = -mkc["R"].to_numpy(float)
    r_x = 1.0 - mkc["ex"].to_numpy(float) / mkc["bid"].to_numpy(float)
    by_k.append(
        {
            "row": "always short",
            "session": "",
            "clock": "",
            "n_forecasts": 0,
            "n_days": len(common),
            "Sharpe_mid_median": sr(r_mid, common),
            "Sharpe_crossed_median": sr(r_x, common),
            "pct_buy_median": 0.0,
        }
    )
    tab = pd.DataFrame(by_k).set_index("row")
    tab["noise_band"] = band(common)
    return long, tab, common


def swing(tab: pd.DataFrame) -> dict[str, float]:
    """close_shift_test_headline.write_numbers(): the swing of the median Sharpe ratio across the offsets (every j,
    j = 0 included), half that range, the median change against j = 0 (j != 0), the band; and the same over j <= 0."""
    sh = tab[tab["n_forecasts"] > 0]
    assert 0 in set(sh["k"].astype(int)), "no j = 0 row"
    n_days = sh["n_days"].astype(int).unique()
    assert len(n_days) == 1, n_days
    lo, hi_ = float(sh["Sharpe_mid_median"].min()), float(sh["Sharpe_mid_median"].max())
    off = sh[sh["k"].astype(int) != 0]["dSharpe_mid_median"]
    same = sh[sh["k"].astype(int) <= 0]
    s_lo, s_hi = (
        float(same["Sharpe_mid_median"].min()),
        float(same["Sharpe_mid_median"].max()),
    )
    s_off = same[same["k"].astype(int) != 0]["dSharpe_mid_median"]
    return {
        "lo": lo,
        "hi": hi_,
        "half": (hi_ - lo) / 2,
        "dlo": float(off.min()),
        "dhi": float(off.max()),
        "band": float(sh["noise_band"].iloc[0]),
        "n_days": int(n_days[0]),
        "n_fc": int(sh["n_forecasts"].max()),
        "zero": float(sh.loc["k=+0", "Sharpe_mid_median"]),
        "same_lo": s_lo,
        "same_hi": s_hi,
        "same_half": (s_hi - s_lo) / 2,
        "same_dlo": float(s_off.min()),
        "same_dhi": float(s_off.max()),
        # close_shift_test_headline.main(): shiftStaleNBelowMin / Max (forecasts below their own j = 0, j < 0)
        "stale_nbelow_min": int(
            same[same["k"].astype(int) < 0]["n_mid_below_k0"].min()
        ),
        "stale_nbelow_max": int(
            same[same["k"].astype(int) < 0]["n_mid_below_k0"].max()
        ),
    }


def sgn(v: float, plus: bool = False, dp: int = 2) -> str:
    """close_shift_test_headline.write_numbers().sgn (dp decimals; a minus sign in math mode)."""
    s = f"{v:+.{dp}f}" if plus else f"{v:.{dp}f}"
    return s.replace("-", r"\ensuremath{-}")


# ---------------------------------------------------------------------------------------------- compounding
def wealth_stats(f: float, r: np.ndarray) -> dict[str, float]:
    """experiments/close_compounding_headline.wealth_stats, line for line."""
    factors = 1.0 + f * np.asarray(r, float)
    assert (factors > 0).all(), "a wealth factor hit zero - ruin"
    w = np.cumprod(factors)
    return {
        "f": f,
        "g_ann": PPY * float(np.mean(np.log(factors))),
        "terminal": float(w[-1]),
        "maxDD_frac": float((w / np.maximum.accumulate(w) - 1.0).min()),
        "worst_day_factor": float(factors.min()),
        "ruin_bound_f": 1.0 / abs(float(np.min(r))),
        "n": len(r),
    }


def comp_summary(x: pd.DataFrame) -> dict[str, float]:
    """close_compounding_headline.main(): one group's summary row (one fill)."""
    return {
        "n": len(x),
        "n_above_1": int((x["terminal"] > 1).sum()),
        "W_median": float(x["terminal"].median()),
        "W_min": float(x["terminal"].min()),
        "W_max": float(x["terminal"].max()),
        "g_median": float(x["g_ann"].median()),
        "g_min": float(x["g_ann"].min()),
        "g_max": float(x["g_ann"].max()),
        "maxDD_median_pct": float(100 * x["maxDD_frac"].abs().median()),
        "maxDD_max_pct": float(100 * x["maxDD_frac"].abs().max()),
        "worst_day_min_factor": float(x["worst_day_factor"].min()),
        "ruin_bound_min": float(x["ruin_bound_f"].min()),
        "n_ruin_bound_above_f": int((x["ruin_bound_f"] > F_FIXED).sum()),
    }


# ---------------------------------------------------------------------------------------------- self-checks
def selfcheck_shift(mk: pd.DataFrame) -> None:
    """A: the original shift test (the earlier draft's nine forecasts, sqrt(252), all 866 option days)."""
    import atm_straddle_lib as asl

    stored = pd.read_csv(
        ROOT / "results" / "close_shift_test" / "shift_test_by_k.csv", index_col=0
    )
    keys = sorted(
        pd.read_csv(ROOT / "results" / "close_shift_test" / "shift_test_modelset.csv")[
            "key"
        ].unique()
    )
    paths = {
        **asl.yhat_paths(ROOT),
        "b2lasso": ROOT / "results" / "spxw_pnl" / "yhat_b2lasso.parquet",
    }
    mt = pd.read_csv(MT_DIR / "master_table.csv").set_index("key")
    D = pd.read_parquet(
        MT_DIR / "master_table_daily.parquet", columns=["key", "day", "plain"]
    )
    D["day"] = pd.to_datetime(D["day"]).dt.normalize().astype("datetime64[ns]")
    F = {}
    for k in keys:
        d = pd.read_parquet(paths[k], columns=["t", "yhat", "baseline"])
        t = pd.to_datetime(d["t"])
        et = (
            t.dt.tz_convert("America/New_York").dt.tz_localize(None)
            if t.dt.tz is not None
            else t
        )
        p = pd.DataFrame(
            {"pred_adj": d["yhat"].to_numpy(float), "B": d["baseline"].to_numpy(float)},
            index=pd.DatetimeIndex(et.to_numpy()).as_unit("ns"),
        )
        p = p[np.isfinite(p["pred_adj"])].sort_index()
        p["hhmm"] = p.index.strftime("%H:%M")
        p = p[p["hhmm"].isin(SESSION_BARS)]
        p["day"] = p.index.normalize()
        plain_ref = D[D["key"] == k].set_index("day")["plain"]
        F[k] = shifted(p.reset_index(drop=True), mk, plain_ref, f"self-check A, {k}")
        t0 = trade(F[k][0], mk)
        gate(
            f"self-check A: j = 0 Sharpe mid on 866 days = master_table.csv ({k})",
            abs(sr_fixed(t0["mid"], mk.index) / float(mt.loc[k, "Sharpe_mid"]) - 1),
            GATE_REL,
            len(t0),
        )
    _, tab, common = shift_test(F, mk, mk.index, sr_fixed, band_fixed)
    cols = [
        "n_forecasts",
        "n_days",
        "Sharpe_mid_median",
        "Sharpe_mid_min",
        "Sharpe_mid_max",
        "Sharpe_crossed_median",
        "n_mid_below_k0",
        "n_crossed_below_k0",
        "dSharpe_mid_median",
        "agree_median_pct",
        "pct_buy_median",
    ]
    rows = [r for r in stored.index if r.startswith("k=")] + [
        "realized (unclipped)",
        "always short",
    ]
    a = tab.loc[rows, cols].to_numpy(float)
    b = stored.loc[rows, cols].to_numpy(float)
    both = np.isnan(a) & np.isnan(b)
    diff = float(
        np.max(np.where(both, 0.0, np.abs(a - b) / np.maximum(1.0, np.abs(b))))
    )
    tab.loc[rows, cols].to_csv(OUT / "selfcheck_A_shift_by_k.csv")
    gate(
        "self-check A: shift_test_by_k.csv reproduced (every j, every column)",
        diff,
        GATE_REL,
        int(a.size),
    )
    s = swing(tab)
    mine = {
        "shiftSwingMedLo": sgn(s["lo"]),
        "shiftSwingMedHi": sgn(s["hi"]),
        "shiftSwingHalf": sgn(s["half"]),
        "shiftSwingDeltaLo": sgn(s["dlo"], True),
        "shiftSwingDeltaHi": sgn(s["dhi"], True),
        "shiftNoiseBand": sgn(s["band"]),
    }
    printed = macro_values(committed("writeup/generated/close_shift_test_numbers.tex"))
    bad = {k: (v, printed.get(k)) for k, v in mine.items() if printed.get(k) != v}
    gate(
        f"self-check A: the six printed shift macros reproduced ({len(common)} days)",
        float(len(bad)),
        0,
        len(mine),
    )
    print(
        "self-check A (shift test, original input):",
        ", ".join(f"\\{k} = {v}" for k, v in mine.items()),
    )


def selfcheck_compounding() -> None:
    """B: the printed compounding numbers (the master table's model set, the 614 post-warmup days).

    2026-10-23 (user, E32): the printed numbers are those of the model set before note E32 -- the forecasts of
    paper_model_set.csv plus the closing-interval tree rungs that refit every 10 sessions, which left it for their
    every-session twins (each ten-day key whose twin is in the set; the master table's tables, read_by_master, of
    results/e32_cadence_gate/twins.csv) -- and they are checked against fixedfrac_modelset_summary.csv as committed at
    PRINTED_REF (results/close_compounding/ now holds the post-E32 set's).  Was: keys = paper_model_set.csv only, and
    the summary read from the working tree.
    """
    keys = list(pd.read_csv(MT_DIR / "paper_model_set.csv")["key"].astype(str))
    tw = pd.read_csv(ROOT / "results" / "e32_cadence_gate" / "twins.csv")
    tw = tw[tw["read_by_master"].astype(bool) & tw["key1"].isin(keys)]
    ten_day = sorted(set(tw["key10"]))
    assert ten_day and not set(ten_day) & set(keys), ten_day
    keys = keys + ten_day
    D = pd.read_parquet(
        MT_DIR / "master_table_daily.parquet",
        columns=["key", "day", "ret_mid", "ret_crossed"],
    )
    D = D[D["key"].isin(keys)].sort_values(["key", "day"])
    rows = []
    for k, x in D.groupby("key"):
        assert len(x) == N_OPT, (k, len(x))
        for fill in ("mid", "crossed"):
            rows.append(
                {
                    "key": k,
                    "fill": fill,
                    **wealth_stats(F_FIXED, x[f"ret_{fill}"].to_numpy(float)[WARMUP:]),
                }
            )
    fa = pd.DataFrame(rows)
    # 2026-10-23 (user, E32): the summary as committed at PRINTED_REF (the pre-E32 model set's). Was:
    # stored = pd.read_csv(ROOT / "results" / "close_compounding" / "fixedfrac_modelset_summary.csv")
    stored = pd.read_csv(
        io.StringIO(
            committed("results/close_compounding/fixedfrac_modelset_summary.csv")
        )
    )
    stored = stored[stored["group"] == "model set"].set_index("fill")
    worst = 0.0
    sm = {}
    for fill in ("mid", "crossed"):
        sm[fill] = comp_summary(fa[fa["fill"] == fill])
        for c, v in sm[fill].items():
            worst = max(
                worst,
                abs(v - float(stored.loc[fill, c]))
                / max(1.0, abs(float(stored.loc[fill, c]))),
            )
    pd.DataFrame(sm).T.to_csv(OUT / "selfcheck_B_compounding_summary.csv")
    gate(
        "self-check B: fixedfrac_modelset_summary.csv (model set) reproduced",
        worst,
        1e-12,
        len(keys),
    )
    mine = {
        "cmpWMedMid": f"{sm['mid']['W_median']:.1f}",
        "cmpWMedX": f"{sm['crossed']['W_median']:.1f}",
        "cmpDDMaxMid": f"{sm['mid']['maxDD_max_pct']:.0f}",
        "cmpDDMaxX": f"{sm['crossed']['maxDD_max_pct']:.0f}",
    }
    printed = macro_values(committed("writeup/generated/close_compounding_numbers.tex"))
    bad = {k: (v, printed.get(k)) for k, v in mine.items() if printed.get(k) != v}
    gate(
        "self-check B: the four printed compounding macros reproduced",
        float(len(bad)),
        0,
        len(mine),
    )
    print(
        "self-check B (compounding, original input):",
        ", ".join(f"\\{k} = {v}" for k, v in mine.items()),
    )


def selfcheck_pooled_d6(mk: pd.DataFrame) -> None:
    """C: the j = 0 trade of the printed pooled runs and the D6 ratio reproduce the printed pooled Sharpe ratios."""
    arms = read_run(PRINTED_LINEAR, {e: e for e in LIN_EST}, None)
    arms.update(read_run(PRINTED_TREES, {e: e for e in TREE_EST}, None))
    printed = macro_values(committed("writeup/generated/rehaul_results_numbers.tex"))
    pw = mk.index[WARMUP:]
    rows = []
    for (e, s), p in arms.items():
        F0 = shifted(p, mk, None, f"self-check C, {e} {s}")[0]
        t = trade(F0, mk)
        assert len(t) == N_OPT, (e, s, len(t))
        for col, tag in (("mid", "Sh"), ("x", "ShX")):
            name = f"rhr{tag}{EST_MACRO[e]}Pool{SET_MACRO[s]}"
            v = sr_d6(t[col].to_numpy(float)[WARMUP:], pw)
            rows.append(
                {
                    "macro": name,
                    "printed": printed.get(name),
                    "recomputed": v,
                    "match": printed.get(name) == f"{v:.2f}",
                }
            )
    sc = pd.DataFrame(rows)
    sc.to_csv(OUT / "selfcheck_C_pooled_d6.csv", index=False)
    gate(
        "self-check C: the printed pooled Sharpe ratios (D6, 614 days) reproduced",
        float((~sc["match"]).sum()),
        0,
        len(sc),
    )
    print(
        f"self-check C (pooled trade + D6, printed W = 1,000 runs): {int(sc['match'].sum())} of {len(sc)} printed ratios"
    )


# ---------------------------------------------------------------------------------------------- outputs
def arm_key(e: str, s: str) -> str:
    return f"{e}_{s}"


def write_tex(
    st: pd.DataFrame,
    sw: dict | None,
    csum: dict | None,
    cb: pd.DataFrame,
    long: pd.DataFrame | None,
    aw: dict,
    sw_alt: dict | None = None,
):
    n_all = len(st)
    n_fin = int((st["status"] == "finished").sum())
    full = n_fin == n_all

    def agg(v: str) -> str:
        return v if full else PENDING

    head = [
        "% AUTO-GENERATED by writeup/make_rehaul_pooled_extras.py -- do not edit; re-run the script.",
        f"% Interval-shift test (\\rhps...) and compounding (\\rhpc...) on the {n_all} pooled intraday forecasts of the "
        f"final run, W = {WINDOW:,} days",
        f"% (results/rehaul_pooled_extras_final_w{WINDOW}/). A statistic over the set prints \\rhPending until every "
        "arm has forecasts.",
        PROVIDE_PENDING,
    ]
    M: list[tuple[str, str, str]] = [
        ("rhpsNfcAll", str(n_all), "pooled intraday forecasts in the set (note E25)"),
        (
            "rhpsNfc",
            str(n_fin),
            "of them with final-run forecasts (shifted and compounded)",
        ),
        ("rhpsNPending", str(n_all - n_fin), "of them pending"),
        (
            "rhpsJMin",
            sgn(float(min(ALL_SHIFTS)), dp=0),
            "smallest offset j (the 10:00 bar of day t)",
        ),
        (
            "rhpsJMax",
            f"{max(ALL_SHIFTS)}",
            "largest offset j (the 12:30 bar of day t+1)",
        ),
    ]
    if sw is not None:
        M += [
            (
                "rhpsNdays",
                agg(str(sw["n_days"])),
                "post-warmup option days on which every shifted forecast exists",
            ),
            (
                "rhpsZeroMed",
                agg(sgn(sw["zero"])),
                "median SR (D6, mid) across the forecasts at j = 0, on those days",
            ),
            (
                "rhpsMedLo",
                agg(sgn(sw["lo"])),
                "smallest median SR (D6, mid) over the offsets j (j = 0 included)",
            ),
            (
                "rhpsMedHi",
                agg(sgn(sw["hi"])),
                "largest median SR (D6, mid) over the offsets j",
            ),
            ("rhpsHalf", agg(sgn(sw["half"])), "half that range: the swing, +-"),
            (
                "rhpsDeltaLo",
                agg(sgn(sw["dlo"], True)),
                "smallest median change of SR against j = 0 (j != 0)",
            ),
            (
                "rhpsDeltaHi",
                agg(sgn(sw["dhi"], True)),
                "largest median change of SR against j = 0 (j != 0)",
            ),
            (
                "rhpsSameMedLo",
                agg(sgn(sw["same_lo"])),
                "smallest median SR over j <= 0 (no information after 15:30 of day t)",
            ),
            ("rhpsSameMedHi", agg(sgn(sw["same_hi"])), "largest median SR over j <= 0"),
            ("rhpsSameHalf", agg(sgn(sw["same_half"])), "half that range"),
            (
                "rhpsSameDeltaLo",
                agg(sgn(sw["same_dlo"], True)),
                "smallest median change against j = 0, j < 0",
            ),
            (
                "rhpsSameDeltaHi",
                agg(sgn(sw["same_dhi"], True)),
                "largest median change against j = 0, j < 0",
            ),
            (
                "rhpsStaleNBelowMin",
                agg(str(sw["stale_nbelow_min"])),
                "fewest forecasts below their own j = 0 SR at an offset j < 0",
            ),
            (
                "rhpsStaleNBelowMax",
                agg(str(sw["stale_nbelow_max"])),
                "most forecasts below their own j = 0 SR at an offset j < 0",
            ),
            (
                "rhpsNoiseBand",
                agg(sgn(sw["band"])),
                "1.96 sqrt(sum_y A_y/N_y)/K: 95 % half-width of the D6 SR under iid returns at SR = 0, those days",
            ),
        ]
    if (
        sw_alt is not None
    ):  # ALTERNATIVE (owner decision): j <= 0 only, on all 614 post-warmup days
        M += [
            (
                "rhpsPwNdays",
                agg(str(sw_alt["n_days"])),
                "ALT: j <= 0 only, on all post-warmup days",
            ),
            (
                "rhpsPwZeroMed",
                agg(sgn(sw_alt["zero"])),
                "ALT: median SR at j = 0 (= median of the printed SRs)",
            ),
            (
                "rhpsPwMedLo",
                agg(sgn(sw_alt["lo"])),
                "ALT: smallest median SR over j <= 0",
            ),
            (
                "rhpsPwMedHi",
                agg(sgn(sw_alt["hi"])),
                "ALT: largest median SR over j <= 0",
            ),
            ("rhpsPwHalf", agg(sgn(sw_alt["half"])), "ALT: half that range"),
            (
                "rhpsPwDeltaLo",
                agg(sgn(sw_alt["dlo"], True)),
                "ALT: smallest median change against j = 0, j < 0",
            ),
            (
                "rhpsPwDeltaHi",
                agg(sgn(sw_alt["dhi"], True)),
                "ALT: largest median change against j = 0, j < 0",
            ),
            (
                "rhpsPwNoiseBand",
                agg(sgn(sw_alt["band"])),
                "ALT: the D6 iid band on those days",
            ),
        ]
    if csum is not None:
        m_, x_ = csum["mid"], csum["crossed"]
        M += [
            ("rhpcNdays", str(N_PW), "post-warmup option days compounded (t > 252)"),
            (
                "rhpcNPosMid",
                agg(str(m_["n_above_1"])),
                "forecasts ending above one unit of wealth, mid-quotes",
            ),
            (
                "rhpcNPosX",
                agg(str(x_["n_above_1"])),
                "the same at quoted bid/ask prices",
            ),
            (
                "rhpcWMedMid",
                agg(f"{m_['W_median']:.1f}"),
                "median terminal value V across the forecasts, mid-quotes (f = 3 %)",
            ),
            (
                "rhpcWMedX",
                agg(f"{x_['W_median']:.1f}"),
                "the same at quoted bid/ask prices",
            ),
            ("rhpcWLoMid", agg(f"{m_['W_min']:.1f}"), "smallest V, mid-quotes"),
            ("rhpcWHiMid", agg(f"{m_['W_max']:.1f}"), "largest V, mid-quotes"),
            ("rhpcWLoX", agg(f"{x_['W_min']:.1f}"), "smallest V, bid/ask"),
            ("rhpcWHiX", agg(f"{x_['W_max']:.1f}"), "largest V, bid/ask"),
            (
                "rhpcGMedMid",
                agg(sgn(m_["g_median"], True)),
                "median annualized log growth G (factor 252), mid-quotes",
            ),
            (
                "rhpcGMedX",
                agg(sgn(x_["g_median"], True)),
                "the same at quoted bid/ask prices",
            ),
            (
                "rhpcDDMedMid",
                agg(f"{m_['maxDD_median_pct']:.0f}"),
                "median maximum drawdown, %, mid-quotes",
            ),
            (
                "rhpcDDMaxMid",
                agg(f"{m_['maxDD_max_pct']:.0f}"),
                "largest maximum drawdown across the forecasts, %, mid-quotes",
            ),
            (
                "rhpcDDMedX",
                agg(f"{x_['maxDD_median_pct']:.0f}"),
                "median maximum drawdown, %, bid/ask",
            ),
            (
                "rhpcDDMaxX",
                agg(f"{x_['maxDD_max_pct']:.0f}"),
                "largest maximum drawdown, %, bid/ask",
            ),
            (
                "rhpcAW",
                f"{aw['mid']['terminal']:.1f}",
                "always short: V on the same days, mid-quotes",
            ),
            ("rhpcAWX", f"{aw['crossed']['terminal']:.1f}", "always short: V, bid/ask"),
            (
                "rhpcADDMid",
                f"{100 * abs(aw['mid']['maxDD_frac']):.0f}",
                "always short: maximum drawdown, %, mid-quotes",
            ),
            (
                "rhpcADDX",
                f"{100 * abs(aw['crossed']['maxDD_frac']):.0f}",
                "always short: maximum drawdown, %, bid/ask",
            ),
        ]
    L = head + [rf"\newcommand{{\{k}}}{{{v}}}  % {w}" for k, v, w in M]
    GEN.mkdir(parents=True, exist_ok=True)
    NUMBERS_TEX.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")

    # ---- the per-forecast table (one row an arm; a pending arm's cells print \rhPending)
    T = [
        "% AUTO-GENERATED by writeup/make_rehaul_pooled_extras.py -- do not edit; re-run the script.",
        "% Per-forecast interval-shift test and compounding of the pooled intraday forecasts (final run).",
        # \rhPending comes from the numbers file loaded below: a second \providecommand here would stop the Overleaf
        # flattener (writeup/overleaf_sync/flatten.py), which expands number macros and cannot replace a defined one.
        r"\ifdefined\rhpsNfcAll\else\input{generated/rehaul_pooled_extras_numbers}\fi",
        r"\begin{table}[tbp]",
        r"\centering",
        r"\small",
        r"\begin{threeparttable}",
        r"\caption{Interval-shift test and compounded wealth of the pooled intraday forecasts}",
        r"\label{tab:rh_pooled_shift_compounding}",
        r"\begin{tabular}{@{}llrrrrrrr@{}}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{Interval-shift test: $\SR$} & \multicolumn{4}{c}{Compounding} \\",
        r"\cmidrule(lr){3-5}\cmidrule(l){6-9}",
        r" & & & \multicolumn{2}{c}{Over $j$} & \multicolumn{2}{c}{$V$} & \multicolumn{2}{c}{Max.\ drawdown (\%)} \\",
        r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(l){8-9}",
        r"Model & Predictors & $j=0$ & Min & Max & Mid & Bid/ask & Mid & Bid/ask \\",
        r"\midrule",
    ]
    for i, r in st.iterrows():
        e, s = r["est"], r["set"]
        lab = EST_NAME[e] if s == "baseline" else ""
        if r["status"] != "finished" or long is None:
            cells = [PENDING] * 7
        else:
            x = long[long["key"] == arm_key(e, s)]
            z = float(x.loc[x["k"] == 0, "Sharpe_mid"].iloc[0])
            cm = cb[(cb["key"] == arm_key(e, s)) & (cb["fill"] == "mid")].iloc[0]
            cx = cb[(cb["key"] == arm_key(e, s)) & (cb["fill"] == "crossed")].iloc[0]
            cells = [
                f"${z:.2f}$",
                f"${x['Sharpe_mid'].min():.2f}$",
                f"${x['Sharpe_mid'].max():.2f}$",
                f"${cm['terminal']:.2f}$",
                f"${cx['terminal']:.2f}$",
                f"${100 * abs(cm['maxDD_frac']):.0f}$",
                f"${100 * abs(cx['maxDD_frac']):.0f}$",
            ]
        sep = r" \\[2pt]" if (s == "all_features" and i < len(st) - 1) else r" \\"
        T.append(f"{lab} & {SET_NAME[s]} & " + " & ".join(cells) + sep)
    T += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]",
        r"\footnotesize",
        r"\item \textit{Notes:} Interval-shift test: the sign strategy with the forecast of bar $c+j$ in place of that"
        r" of the closing interval $c$, placed on the closing interval's diurnal adjustment, "
        r"$q_t = \sign\bigl(\hat y_{t,c+j}^{\,2}\, s^2_{t,c} - \mathrm{IV}_t^2\bigr)$, for $j = \rhpsJMin, \dots, "
        r"\rhpsJMax$ ($j<0$: earlier intervals of day $t$; $j>0$: the first six intervals of day $t+1$). $\SR$ is the"
        r" paper's Sharpe ratio at mid-quotes (Section~\ref{sec:rh_econ_measures}) on the \rhpsNdays{} post-warmup"
        r" days on which every shifted forecast exists (the last post-warmup day has no next session), so the $j=0$"
        r" column differs from the Sharpe ratios on all \rheSzDays{} post-warmup days; Min and Max run over $j$,"
        r" $j = 0$ included. Compounding: a"
        r" fixed $f = \rhappCompF\%$ of wealth per day over the \rhpcNdays{} post-warmup days, terminal value $V$ and"
        r" maximum drawdown at mid-quotes and at quoted bid/ask prices.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
    ]
    TABLE_TEX.write_text("\n".join(T) + "\n", encoding="utf-8", newline="\n")
    return M


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"window W = {WINDOW:,}; outputs {fr.rel(OUT)}, tex {fr.rel(GEN)}")
    mk = load_market()
    pw = mk.index[WARMUP:]
    assert len(pw) == N_PW

    # ---- self-checks on the inputs the printed numbers came from
    selfcheck_shift(mk)
    selfcheck_compounding()
    selfcheck_pooled_d6(mk)

    # ---- the final run's pooled forecasts
    arms, status = final_arms()
    st = pd.DataFrame(status)
    st.to_csv(OUT / "arms.csv", index=False)
    print(st.to_string(index=False))
    F = {}
    for (e, s), p in arms.items():
        F[arm_key(e, s)] = shifted(p, mk, None, f"final {e} {s}")
    order = [
        arm_key(r["est"], r["set"])
        for _, r in st.iterrows()
        if r["status"] == "finished"
    ]
    F = {k: F[k] for k in order}

    sw = csum = sw_alt = None
    long = None
    cb = pd.DataFrame()
    aw: dict[str, dict] = {}
    if F:
        long, tab, common = shift_test(F, mk, pw, sr_d6, band_d6)
        long.to_csv(OUT / "shift_by_arm.csv", index=False)
        tab.to_csv(OUT / "shift_by_k.csv")
        sw = swing(tab)
        print(tab.round(3).to_string())
        print(
            f"shift test: {len(F)} forecasts, {len(common)} common post-warmup days ({common[0].date()} .. {common[-1].date()})"
        )
        # ALTERNATIVE for the owner (not the original's method): the offsets j <= 0 alone, which read no day t + 1 and
        # so exist on all 614 post-warmup days; j = 0 is then the paper's printed Sharpe ratio (the common days above
        # drop 30 April 2024, which has no next session, and that one day moves every D6 ratio by about 0.15-0.2).
        F_same = {k: {j: v for j, v in Fk.items() if j <= 0} for k, Fk in F.items()}
        long_a, tab_a, common_a = shift_test(F_same, mk, pw, sr_d6, band_d6)
        assert len(common_a) == N_PW, len(common_a)
        long_a.to_csv(OUT / "shift_by_arm_jle0_614days.csv", index=False)
        tab_a.to_csv(OUT / "shift_by_k_jle0_614days.csv")
        sw_alt = swing(tab_a)
        print(tab_a.round(3).to_string())

        # ---- compounding on the post-warmup days (positions read only the 16:00 forecast and the implied variance)
        rows = []
        for key, Fk in F.items():
            t = trade(Fk[0], mk)
            assert len(t) == N_OPT, key
            for fill, col in (("mid", "mid"), ("crossed", "x")):
                rows.append(
                    {
                        "key": key,
                        "fill": fill,
                        **wealth_stats(F_FIXED, t[col].to_numpy(float)[WARMUP:]),
                    }
                )
        r_short = {
            "mid": -mk["R"].to_numpy(float),
            "crossed": 1.0 - mk["ex"].to_numpy(float) / mk["bid"].to_numpy(float),
        }
        for fill, r in r_short.items():
            aw[fill] = wealth_stats(F_FIXED, r[WARMUP:])
            rows.append({"key": "always short", "fill": fill, **aw[fill]})
        cb = pd.DataFrame(rows)
        cb.to_csv(OUT / "compounding_by_arm.csv", index=False)
        # always short reads no forecast: it equals the committed 614-day row of the model-set compounding
        fa = pd.read_csv(
            ROOT / "results" / "close_compounding" / "fixedfrac_modelset.csv"
        )
        fa = fa[fa["key"] == "always short"].set_index("fill")
        gate(
            "always short V (614 days) = results/close_compounding/fixedfrac_modelset.csv",
            max(
                abs(aw[f]["terminal"] / float(fa.loc[f, "terminal"]) - 1)
                for f in ("mid", "crossed")
            ),
            1e-12,
            N_PW,
        )
        fs = cb[cb["key"] != "always short"]
        csum = {
            fill: comp_summary(fs[fs["fill"] == fill]) for fill in ("mid", "crossed")
        }
        pd.DataFrame(csum).T.to_csv(OUT / "compounding_summary.csv")
        print(pd.DataFrame(csum).T.round(3).to_string())

    M = write_tex(st, sw, csum, cb, long, aw, sw_alt)
    pd.DataFrame(GATES).to_csv(OUT / "gates.csv", index=False)
    print(f"gates: {sum(g['ok'] for g in GATES)} of {len(GATES)} pass")
    for k, v, w in M:
        print(f"  \\{k} = {v}   ({w})")
    print("wrote", fr.rel(NUMBERS_TEX), "and", fr.rel(TABLE_TEX))


if __name__ == "__main__":
    main()
