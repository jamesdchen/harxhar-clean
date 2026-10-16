"""Out-of-sample accuracy table of the rehauled paper (tab:rh_oos, writeup/rehaul/06_results_floats.tex).

User request (2026-10-15): "there needs to be a very clean table in section 6 on the out of sample performance. it
needs to be the statistical values from the master table with 2 large columns of pooled vs last 30 min results and MSE,
qlike, and OOS R2 as subcolumns. it needs to be grouped by model with each data bucket a separate row. I expect the
table to be a full page"

LAYOUT
  Columns: Model / predictor set | Last 30 minutes (closing-interval models): MSE, QLIKE, R2_OOS | Pooled intraday
  models: MSE, QLIKE, R2_OOS.  Rows: Panel A linear models (OLS, Ridge, Lasso, Elastic net), Panel B tree ensembles
  (LightGBM, XGBoost, Random forest); under each model one row per predictor set, nested HAR benchmark (baseline)
  < parsimonious set (lean_vol_vix) < retail-feed set (live_feasible) < full predictor set (all_features); the OLS has
  the HAR benchmark only.  No stars and no bold: the tests are in tab:rh_axis1 / tab:rh_axis2 / tab:rh_rank.

SPECIFICATIONS OF RECORD (= tab:rh_axis1 / tab:rh_axis2, writeup/make_rehaul_results.py FAMILY_ROWS)
  closing-interval linear  subpt_<est>_<set> (penalty pre-tuned on the pre-sample; experiments/stack_pretune_yhat.py),
                           sub_ols_baseline for the OLS;
  pooled linear            poolpt_<est>_<set> (their pooled twins, as T2), a0 for the OLS (the earlier draft's pooled
                           OLS on HAR + calendar, as tab:rh_rank: note a);
  closing-interval trees   subtree_daily_<set>_<est> (fixed hyperparameters, re-estimated daily, as T1);
  pooled trees             results/pooled_trees_2026-10-07/yhat_pooltree_<est>_<set>.parquet, 16:00 rows, plain =
                           yhat^2 x baseline (make_rehaul_results.plain_16), on baseline and lean_vol_vix; on the full
                           set the master table's lgbm / xgb rows (the earlier draft's pooled trees: note a) and the
                           forest's INTERIM1600 table (preliminary: dagger).
  Missing (no forecast of the specification of record exists): every live_feasible linear and pooled forecast (the
  pre-tuned linear arms were stacked for baseline / lean_vol_vix / all_features only, stack_pretune_yhat.BUCKETS; no
  pooled walk ran on live_feasible) and the OLS on every set but the HAR benchmark.  They print as "--".
  The block-penalized ridge (blk2, blk2_inc) is left out: it exists only as the earlier draft's pooled intraday
  forecast on its full predictor set, with no closing-interval twin and no other predictor set (T1 and T2 leave it out
  too; it enters only the ranking, tab:rh_rank).

SCORING (every number against the ACTUAL realized variance of the closing half hour, user decision #10)
  # days   the 866 out-of-sample trading days 2020-01-03 .. 2024-04-30 of the master table (\\rhrDays), the same for
  #        every cell (asserted);  -- 2026-10-15 (user): parked -- the option-data days; accuracy now on the rolling span
  days   the COMMON ROLLING SPAN (2026-10-15, user: "Out of sample period doesn't make sense for forecasts because we're
         performing rolling backtests. The forecasting results need to be calculated to reflect such (that the out of
         sample forecasts are done on a rolling basis after the burn in training window)"): every forecast is made by a
         model re-estimated on its trailing window, so every forecast on disk is out of sample; each cell is scored on
         the days on which EVERY forecast of the table exists -- experiments/rolling_eval_2026_10_15.py common_span():
         2020-01-03 .. 2024-04-30, 1,082 trading days, NYSE early closes dropped, no gap (asserted).  The start is set
         by the pooled pre-tuned linear forecasts (their walk starts on 2020-01-03) and is also the first day after
         the pre-sample on which the linear penalties were selected (validation folds to 2020-01-02).  The 866 days
         of the master table (\\rhrDays) are the option-data subsample of the straddle economics only; on them every
         master key still reproduces master_table.csv qlike_raw (asserted);
  truth  the 16:00 ET sumret2 of data/core_stats.parquet (rolling_eval history(); = rv_unclipped of
         master_table_daily.parquet = the production table's rv_raw on the 866 days, asserted);
  MSE    mean of (RV_t - sigma2_t)^2, printed x 10^MSE_EXP;
  QLIKE  mean of r - log r - 1, r = RV_t / sigma2_t (asserted equal to master_table.csv qlike_raw for every master key);
  R2_OOS Campbell and Thompson (2008): 1 - sum (RV_t - sigma2_t)^2 / sum (RV_t - RVbar_{t-1})^2, RVbar_{t-1} the
         historical mean of the closing-half-hour RV over every trading day before t (expanding, from the first 16:00
         print of data/core_stats.parquet, 1998-01-05; early closes and non-positive prints dropped).  The benchmark is
         causal by construction (cumulative sums shifted one day) and re-checked by brute force on every day.  The GUI's
         R2 (writeup/master_table_gui/rows.py _loss_metrics) divides by sum (RV_t - mean_oos RV)^2 instead, the
         out-of-sample mean, which looks ahead; it is printed to the console for comparison only.

WRITES
  writeup/generated/rehaul_results_oos.tex   the table (tab:rh_oos);
  writeup/generated/rehaul_oos_numbers.tex   macro-only \\rhoos... numbers (loaded once by the paper's preamble): the
                                             rolling span (start, end, days, early closes dropped, days outside the
                                             option subsample), the R2 benchmark's start, the DM lag on the span, and
                                             the 19 March 2020 effect on the closing-interval trees.

Run:  python writeup/make_rehaul_oos_table.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent  # writeup/
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "notebooks"))
sys.path.insert(0, str(REPO / "experiments"))
import atm_straddle_lib as asl  # noqa: E402
import rolling_eval_2026_10_15 as rolling  # noqa: E402  (2026-10-15: the common rolling span)
from src.evaluation.diebold_mariano import _nw_lag  # noqa: E402

GEN = ROOT / "generated"
MT = REPO / "results" / "close_master_table"
POOLTREE = REPO / "results" / "pooled_trees_2026-10-07"
CORE = REPO / "data" / "core_stats.parquet"
OUT_TABLE = GEN / "rehaul_results_oos.tex"
OUT_NUMBERS = GEN / "rehaul_oos_numbers.tex"

OLS = "sub_ols_baseline"
NDAYS = 866  # = \rhrDays: the option-data subsample (the master table's days; asserted against rehaul_results_numbers.tex)
FIRST_DAY, LAST_DAY = pd.Timestamp("2020-01-03"), pd.Timestamp("2024-04-30")
SPAN_DAYS = 1082  # the common rolling span 2020-01-03 .. 2024-04-30 (asserted; experiments/rolling_eval_2026_10_15.py)
BIG_DAY = pd.Timestamp(
    "2020-03-19"
)  # the closing-interval trees' largest squared errors on the span (asserted)
MSE_EXP = 10  # MSE printed x 10^MSE_EXP
HEAD = "% AUTO-GENERATED by writeup/make_rehaul_oos_table.py -- do not edit."

# predictor sets, in nesting order (asserted nested below): code name -> row label
# 2026-10-15 (user): parked -- "park the parsimonious and retail feed rows and rename the full predictor set as all
# features"
# SETS = [
#     ("baseline", "HAR benchmark"),
#     ("lean_vol_vix", "Parsimonious set"),
#     ("live_feasible", "Retail-feed set"),
#     ("all_features", "Full predictor set"),
# ]
SETS = [
    ("baseline", "HAR benchmark"),
    ("all_features", "All features"),
]
LIN, TREE = "Panel A: Linear models", "Panel B: Tree ensembles"
EST_NAME = {
    "ridge": "Ridge",
    "lasso": "Lasso",
    "enet": "Elastic net",
    "lgbm": "LightGBM",
    "xgb": "XGBoost",
    "rf": "Random forest",
}
EARLIER = {
    "a0",
    "lgbm",
    "xgb",
}  # the earlier draft's pooled intraday specifications (note a)
PRELIM = {
    "pooltree_rf_all_features"
}  # preliminary (dagger): 16:00 rows of a walk still running


def model_rows() -> list[tuple[str, str, dict[str, str | None], dict[str, str | None]]]:
    """(model, panel, closing-interval key per set, pooled intraday key per set); None = no forecast."""
    rows: list[tuple[str, str, dict[str, str | None], dict[str, str | None]]] = [
        ("OLS", LIN, {"baseline": OLS}, {"baseline": "a0"})
    ]
    for est in ("ridge", "lasso", "enet"):
        close: dict[str, str | None] = {b: f"subpt_{est}_{b}" for b, _ in SETS}
        pool: dict[str, str | None] = {b: f"poolpt_{est}_{b}" for b, _ in SETS}
        close["live_feasible"] = pool["live_feasible"] = None
        rows.append((EST_NAME[est], LIN, close, pool))
    for est in ("lgbm", "xgb", "rf"):
        close = {b: f"subtree_daily_{b}_{est}" for b, _ in SETS}
        pool = {
            "baseline": f"pooltree_{est}_baseline",
            "lean_vol_vix": f"pooltree_{est}_lean_vol_vix",
            "live_feasible": None,
            "all_features": est
            if est in ("lgbm", "xgb")
            else f"pooltree_{est}_all_features",
        }
        rows.append((EST_NAME[est], TREE, close, pool))
    return rows


# --------------------------------------------------------------------------- data


def _day(values) -> pd.DatetimeIndex:
    t = pd.to_datetime(pd.Series(values))
    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_convert("America/New_York").dt.tz_localize(None)
    return pd.DatetimeIndex(t.dt.normalize().to_numpy())


def plain_16(path: Path) -> pd.Series:
    """16:00 ET rows of a pooled yhat table: f^2 x B (= make_rehaul_results.plain_16, rows.py plain_16)."""
    y = pd.read_parquet(path, columns=["t", "yhat", "baseline"])
    t = pd.to_datetime(y["t"], utc=True).dt.tz_convert("America/New_York")
    at16 = ((t.dt.hour == 16) & (t.dt.minute == 0)).to_numpy()
    v = y["yhat"].to_numpy(float)[at16] ** 2 * y["baseline"].to_numpy(float)[at16]
    s = pd.Series(v, index=_day(t[at16].dt.tz_localize(None)))
    assert not s.index.duplicated().any(), path
    return s


def rv_history() -> pd.Series:
    """The closing-half-hour RV of every trading day with a print: the 16:00 ET sumret2 of data/core_stats.parquet
    (naive-ET stamps), early closes and non-positive prints dropped, by day."""
    c = pd.read_parquet(CORE, columns=["endbartime", "sumret2"])
    et = pd.to_datetime(c["endbartime"])
    assert et.dt.tz is None, "core_stats stamps are naive ET"
    m = ((et.dt.hour == 16) & (et.dt.minute == 0)).to_numpy()
    s = pd.Series(c.loc[m, "sumret2"].to_numpy(float), index=_day(et[m])).sort_index()
    assert not s.index.duplicated().any()
    early = pd.DatetimeIndex(pd.to_datetime(list(asl.EARLY_CLOSE_DATES)))
    s = s[np.isfinite(s.to_numpy()) & (s.to_numpy() > 0) & ~s.index.isin(early)]
    return s


def load_cells() -> tuple[
    dict[str, pd.DataFrame], pd.DataFrame, pd.DatetimeIndex, pd.DatetimeIndex
]:
    """Daily (plain, rv) frames of every key the table uses on the common rolling span, the master table, the span and
    the master table's 866 option-data days.

    2026-10-15 (user: accuracy over every rolling forecast): the frames come from experiments/rolling_eval_2026_10_15.py
    (scored(): the table's 16:00 rows, plain = yhat^2 B, the actual RV, early closes dropped), on its common_span().
    The 866-day load of record is parked below this function; on those days every master key still reproduces the
    master table's plain, truth, daily QLIKE and qlike_raw (asserted here), and every pooltree key make_rehaul_results'
    plain_16 rule."""
    m = pd.read_csv(MT / "master_table.csv").set_index("key")
    d = pd.read_parquet(
        MT / "master_table_daily.parquet",
        columns=["key", "day", "plain", "rv_unclipped", "qlike_raw"],
    )
    d["day"] = _day(d["day"])
    # keys = {k for _, _, close, pool in model_rows() for k in [*close.values(), *pool.values()] if k is not None}
    # 2026-10-15: parked -- also loaded the parked sets' pooled trees (lean_vol_vix), which no cell prints
    keys = {
        k
        for _, _, close, pool in model_rows()
        for b, _ in SETS
        for k in (close.get(b), pool.get(b))
        if k is not None
    }
    assert keys == {k.key for k in rolling.table_keys()}, sorted(
        keys ^ {k.key for k in rolling.table_keys()}
    )
    master_keys = {k for k in keys if not k.startswith("pooltree_")}
    missing = master_keys - set(m.index)
    assert not missing, f"keys not in master_table.csv: {sorted(missing)}"
    by = {
        k: f.set_index("day").sort_index()
        for k, f in d[d["key"].isin(master_keys | {OLS})].groupby("key")
    }
    assert set(by) == master_keys | {OLS}, sorted(master_keys - set(by))
    mt_days = pd.DatetimeIndex(by[OLS].index)
    assert (
        len(mt_days) == NDAYS and mt_days[0] == FIRST_DAY and mt_days[-1] == LAST_DAY
    ), (
        len(mt_days),
        mt_days[0],
        mt_days[-1],
    )
    days = rolling.common_span()
    assert len(days) == SPAN_DAYS and days[0] == FIRST_DAY and days[-1] == LAST_DAY, (
        len(days),
        days[0],
        days[-1],
    )
    assert mt_days.isin(days).all(), "an option-data day outside the rolling span"
    out: dict[str, pd.DataFrame] = {}
    for k in sorted(keys):
        f = rolling.scored(k)
        assert days.isin(f.index).all(), (
            f"{k}: a forecast is missing on a day of the span"
        )
        on = f.reindex(mt_days)
        if k in master_keys:
            g = by[k]
            # the 866 option-data days reproduce the master table: days, forecast, truth, daily QLIKE, qlike_raw
            assert int(m.loc[k, "n_days"]) == NDAYS, (k, m.loc[k, "n_days"])
            assert len(g) == NDAYS and g.index.equals(mt_days), k
            assert np.allclose(on["plain"], g["plain"], rtol=1e-10, atol=0), (
                f"{k}: plain"
            )
            assert np.array_equal(on["rv"].to_numpy(), g["rv_unclipped"].to_numpy()), (
                f"{k}: truth"
            )
            r = (on["rv"] / on["plain"]).to_numpy(float)
            assert np.allclose(
                g["qlike_raw"], r - np.log(r) - 1.0, rtol=1e-10, atol=1e-14
            ), f"{k}: daily QLIKE"
            assert np.isclose(
                np.mean(r - np.log(r) - 1.0), m.loc[k, "qlike_raw"], rtol=1e-12, atol=0
            ), f"{k}: QLIKE"
        else:  # pooltree_*: the GUI's / make_rehaul_results' plain_16 rule on the table's own 16:00 rows
            est, b = k.split("_")[1], k.split("_", 2)[2]
            fn = (
                "yhat_pooltree_rf_all_features_INTERIM1600.parquet"
                if k == "pooltree_rf_all_features"
                else f"yhat_pooltree_{est}_{b}.parquet"
            )
            p = plain_16(POOLTREE / fn).reindex(days)
            assert np.allclose(f["plain"].reindex(days), p, rtol=1e-12, atol=0), (
                f"{k}: plain_16"
            )
        out[k] = f.reindex(days)[["plain", "rv"]]
    return out, m, days, mt_days


# 2026-10-15 (user): parked -- the 866-day load of record (accuracy on the option-data days only).  Was:
# def load_cells() -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DatetimeIndex]:
#     """Daily (plain, rv) frames of every key the table uses, on the OLS row's 866 days."""
#     ... the master keys' plain / rv_unclipped of master_table_daily.parquet on the OLS row's 866 days (asserted:
#     866 days 2020-01-03 .. 2024-04-30, the OLS row's truth, daily QLIKE, qlike_raw), the pooltree keys' plain_16 on
#     the same days; return out, m, days


# --------------------------------------------------------------------------- statistics


def ct_benchmark(hist: pd.Series, days: pd.DatetimeIndex) -> pd.Series:
    """RVbar_{t-1}: the mean of hist over every day strictly before t (expanding, causal), on the scored days."""
    csum = hist.cumsum().shift(1)
    n = pd.Series(
        np.arange(len(hist), dtype=float), index=hist.index
    )  # days strictly before
    prior = (csum / n).replace([np.inf], np.nan)
    # a scored day is a history day (asserted by the caller), so the value at t uses days < t only
    b = prior.reindex(days)
    # brute-force causality check on every scored day
    hv, hi = hist.to_numpy(float), hist.index
    for t, v in b.items():
        before = hv[hi < t]
        assert before.size > 0 and np.isclose(v, before.mean(), rtol=1e-12, atol=0), t
        assert hi[hi < t].max() < t
    return b


def metrics(f: pd.DataFrame, bench: pd.Series) -> dict[str, float]:
    e2 = (f["rv"] - f["plain"]) ** 2
    r = (f["rv"] / f["plain"]).to_numpy(float)
    rv = f["rv"]
    return {
        "mse": float(e2.mean()),
        "qlike": float(np.mean(r - np.log(r) - 1.0)),
        "r2": float(100.0 * (1.0 - e2.sum() / ((rv - bench) ** 2).sum())),
        "r2_gui": float(100.0 * (1.0 - e2.sum() / ((rv - rv.mean()) ** 2).sum())),
    }


# --------------------------------------------------------------------------- formatting


def _fmt(x: float, dp: int) -> str:
    v = round(float(x), dp)
    if v == 0:
        v = 0.0  # no negative zero
    return f"{'$-$' if v < 0 else ''}{abs(v):.{dp}f}"


def cells(x: dict[str, float] | None) -> list[str]:
    if x is None:
        return ["--", "--", "--"]
    return [_fmt(x["mse"] * 10.0**MSE_EXP, 2), f"{x['qlike']:.4f}", _fmt(x["r2"], 1)]


SIDE_SPEC = {  # 2026-10-15 (user): Table 1 split -- pooled intraday in Experiments, last 30 minutes in Section 6.4
    "pool": dict(
        out="rehaul_results_oos.tex",
        label="tab:rh_oos",
        caption="Forecast accuracy of the pooled intraday models",
        head="Pooled intraday",
    ),
    "close": dict(
        out="rehaul_results_oos_last30.tex",
        label="tab:rh_oos_last30",
        caption="Forecast accuracy when training on the last 30 minutes",
        head="Last 30 minutes",
    ),
}


def table_side(rows: list[dict], side: str) -> str:
    """One scheme's columns of the former two-scheme table (user, 2026-10-15: 'split it')."""
    spec = SIDE_SPEC[side]
    body: list[str] = []
    panel = None
    for i, r in enumerate(rows):
        if r["panel"] != panel:
            if panel is not None:
                body.append("\\addlinespace[6pt]")
            body.append(
                f"\\multicolumn{{4}}{{@{{}}l}}{{\\textit{{{r['panel']}}}}} \\\\"
            )
            body.append("\\addlinespace[3pt]")
            panel = r["panel"]
        elif i > 0:
            body.append("\\addlinespace[4pt]")
        name = r["model"]
        if side == "close" and name == "OLS":
            name = "OLS (benchmark forecast)"
        body.append(f"{name} \\\\")
        for b, label in r["sets"]:
            c = r[side][b]
            mark = ""
            if side == "pool" and c is not None and c["key"] in EARLIER:
                mark += "\\tnote{a}"
            if side == "pool" and c is not None and c["key"] in PRELIM:
                mark += "\\tnote{$\\dagger$}"
            body.append(
                " & ".join([f"\\hspace{{1em}}{label}{mark}", *cells(c and c["m"])])
                + " \\\\"
            )
    common = (
        "\\item \\textit{Notes:} Rolling forecasts from \\rhoosSpanStart{} to \\rhoosSpanEnd{}, every forecast made "
        "with a model estimated on its trailing window only, so every forecast is out of sample; each is scored "
        "against the actual realized variance of the closing half hour, $\\RV_t$, on the $T_{\\mathrm{oos}} = "
        "\\rhoosDays$ trading days on which every forecast of both training schemes exists, early closes dropped "
        "(Section~\\ref{sec:rh_results_eval}). "
        "The span starts with the pooled intraday linear forecasts, on the first trading day after the pre-sample on "
        "which the penalties were selected (\\cmPtValFirst{} to \\cmPtValLast). "
        "MSE is the mean squared error $T_{\\mathrm{oos}}^{-1}\\sum_t (\\RV_t - \\hat\\sigma^2_t)^2$, multiplied by "
        "$10^{\\rhoosMseExp}$, and QLIKE the mean QLIKE loss. "
        "$R^2_{\\mathrm{OOS}} = 1 - \\sum_t (\\RV_t - \\hat\\sigma^2_t)^2 / \\sum_t (\\RV_t - \\overline{\\RV}_{t-1})^2$, "
        "in percent, where the benchmark $\\overline{\\RV}_{t-1}$ is the historical mean of the closing-half-hour "
        "realized variance over all trading days from \\rhoosBenchStart{} to day $t-1$ \\citep{CampbellThompson2008}. "
        "All features is the full predictor set of Section~\\ref{sec:rh_predictors}. "
    )
    if side == "pool":
        notes = common + (
            "The same days for the models trained on the last 30 minutes are in "
            "Table~\\ref{tab:rh_oos_last30} (Section~\\ref{sec:rh_econ_last30})."
        )
        marks = [
            "\\item[a] The earlier draft's pooled intraday specification.",
            "\\item[$\\dagger$] Preliminary: the closing-interval forecasts of the pooled intraday random forest on "
            "all features, from an exercise still in progress.",
        ]
    else:
        notes = common + (
            "Each model is estimated on the last 30 minutes of each trading day only (one observation per day); the "
            "OLS on the HAR benchmark is the benchmark forecast. Trees use fixed hyperparameters and are re-estimated "
            "daily (Section~\\ref{sec:rh_exp_schemes}). The pooled intraday models on the same days are in "
            "Table~\\ref{tab:rh_oos}."
        )
        marks = []
    return "\n".join(
        [
            HEAD,
            f"% {spec['label']}: MSE, QLIKE and Campbell-Thompson R2_OOS of every model x predictor set, "
            f"{spec['head']} scheme only (Table 1 split, user 2026-10-15).",
            "\\begin{table}[htbp]",
            "\\centering",
            "\\small",
            "\\begin{threeparttable}",
            f"\\caption{{{spec['caption']}}}",
            f"\\label{{{spec['label']}}}",
            # full text width, so the notes are not a tall narrow column (2026-10-15)
            "\\begin{tabular*}{0.85\\linewidth}{@{\\extracolsep{\\fill}}lrrr@{}}",
            "\\toprule",
            f" & \\multicolumn{{3}}{{c}}{{{spec['head']}}} \\\\",
            "\\cmidrule(l){2-4}",
            "Model and predictor set & MSE & QLIKE & $R^2_{\\mathrm{OOS}}$ (\\%) \\\\",
            "\\midrule",
            *body,
            "\\bottomrule",
            "\\end{tabular*}",
            "\\begin{tablenotes}[flushleft]",
            "\\footnotesize",
            notes,
            *marks,
            "\\end{tablenotes}",
            "\\end{threeparttable}",
            "\\end{table}",
            "",
        ]
    )


# 2026-10-15 (user): parked -- the two-scheme table, split into table_side("pool") and table_side("close");
# kept for reference, no longer written.
def table(rows: list[dict]) -> str:
    body: list[str] = []
    panel = None
    for i, r in enumerate(rows):
        if r["panel"] != panel:
            if panel is not None:
                body.append("\\addlinespace[6pt]")
            body.append(
                f"\\multicolumn{{7}}{{@{{}}l}}{{\\textit{{{r['panel']}}}}} \\\\"
            )
            body.append("\\addlinespace[3pt]")
            panel = r["panel"]
        elif i > 0:
            body.append("\\addlinespace[4pt]")
        body.append(f"{r['model']} \\\\")
        for b, label in r["sets"]:
            c, p = r["close"][b], r["pool"][b]
            mark = ""
            if p is not None and p["key"] in EARLIER:
                mark += "\\tnote{a}"
            if p is not None and p["key"] in PRELIM:
                mark += "\\tnote{$\\dagger$}"
            line = [
                f"\\hspace{{1em}}{label}{mark}",
                *cells(c and c["m"]),
                *cells(p and p["m"]),
            ]
            body.append(" & ".join(line) + " \\\\")
    # 2026-10-15 (user): parked -- "out-of-sample period" and the option-data days; accuracy over every rolling
    # forecast. Was: "\\item \\textit{Notes:} The out-of-sample period runs from \\rhrSampleStart{} to \\rhrSampleEnd{}
    # ($T_{\\mathrm{oos}} = \\rhrDays$ trading days); every forecast $\\hat\\sigma^2_t$ is scored against the actual
    # realized variance of the closing half hour, $\\RV_t$, on the same days (Section~\\ref{sec:rh_results_eval}). MSE is
    # the mean squared error $T_{\\mathrm{oos}}^{-1}\\sum_t (\\RV_t - \\hat\\sigma^2_t)^2$, multiplied by
    # $10^{\\rhoosMseExp}$, and QLIKE the mean QLIKE loss. $R^2_{\\mathrm{OOS}} = ...$, in percent, where the benchmark
    # $\\overline{\\RV}_{t-1}$ is the historical mean of the closing-half-hour realized variance over all trading days
    # from \\rhoosBenchStart{} to day $t-1$ \\citep{CampbellThompson2008}."
    notes = (
        "\\item \\textit{Notes:} Rolling forecasts from \\rhoosSpanStart{} to \\rhoosSpanEnd{}, every forecast made "
        "with a model estimated on its trailing window only, so every forecast is out of sample; each is scored "
        "against the actual realized variance of the closing half hour, $\\RV_t$, on the $T_{\\mathrm{oos}} = "
        "\\rhoosDays$ trading days on which every forecast in the table exists, early closes dropped "
        "(Section~\\ref{sec:rh_results_eval}). "
        "The span starts with the pooled intraday linear forecasts, on the first trading day after the pre-sample on "
        "which the closing-interval penalties were selected (\\cmPtValFirst{} to \\cmPtValLast); the straddle "
        "economics use its option-data subsample of \\rhrDays{} days (Section~\\ref{sec:rh_econ}). "
        "MSE is the mean squared error $T_{\\mathrm{oos}}^{-1}\\sum_t (\\RV_t - \\hat\\sigma^2_t)^2$, multiplied by "
        "$10^{\\rhoosMseExp}$, and QLIKE the mean QLIKE loss. "
        "$R^2_{\\mathrm{OOS}} = 1 - \\sum_t (\\RV_t - \\hat\\sigma^2_t)^2 / \\sum_t (\\RV_t - \\overline{\\RV}_{t-1})^2$, "
        "in percent, where the benchmark $\\overline{\\RV}_{t-1}$ is the historical mean of the closing-half-hour "
        "realized variance over all trading days from \\rhoosBenchStart{} to day $t-1$ \\citep{CampbellThompson2008}. "
        # 2026-10-15 (user): parked -- the parsimonious and retail-feed rows are parked, so is their definition:
        # "The parsimonious set adds to the HAR benchmark the ES return, realized absolute variation, realized "
        # "quarticity and realized autocovariance, ES volume and the VIX; the retail-feed set adds the "
        # "\\rhoosNSeriesLive{} series available at 15:30 from retail data feeds (the seven ES return moments, ES "
        # "volume, the number of one-minute returns, the VIX, VVIX and VIX3M, and the FOMC calendar; "
        # "Section~\\ref{sec:rh_predictors}). "
        "All features is the full predictor set of Section~\\ref{sec:rh_predictors}. "
        "Closing-interval trees use fixed hyperparameters and are re-estimated daily "
        # 2026-10-15: "-- marks a forecast not estimated" parked -- no such cell after the parsimonious and
        # retail-feed rows were parked.
        "(Section~\\ref{sec:rh_exp_schemes})."
        # 2026-10-15 (user): parked -- "park all tables that I haven't mentioned in this chat"; the significance
        # tables are parked. Was: "(Section~\\ref{sec:rh_exp_schemes}); significance tests are in "
        # "Tables~\\ref{tab:rh_axis1} and~\\ref{tab:rh_axis2}."
    )
    return "\n".join(
        [
            HEAD,
            "% tab:rh_oos: out-of-sample MSE, QLIKE and Campbell-Thompson R2_OOS of every model x predictor set, closing-",
            "% interval (last 30 minutes) and pooled intraday.  Keys per cell: the generator's model_rows(); the",
            "% MSE scale and the R2 benchmark's start are \\rhoos... macros of rehaul_oos_numbers.tex.",
            "\\begin{table}[p]",
            "\\centering",
            "\\small",
            "\\begin{threeparttable}",
            "\\caption{Out-of-sample forecast accuracy}",
            "\\label{tab:rh_oos}",
            "\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}lrrrrrr@{}}",
            "\\toprule",
            " & \\multicolumn{3}{c}{Last 30 minutes} & \\multicolumn{3}{c}{Pooled intraday} \\\\",
            "\\cmidrule(lr){2-4}\\cmidrule(l){5-7}",
            "Model and predictor set & MSE & QLIKE & $R^2_{\\mathrm{OOS}}$ (\\%) & MSE & QLIKE & "
            "$R^2_{\\mathrm{OOS}}$ (\\%) \\\\",
            "\\midrule",
            *body,
            "\\bottomrule",
            "\\end{tabular*}",
            "\\begin{tablenotes}[flushleft]",
            "\\footnotesize",
            notes,
            "\\item[a] The earlier draft's pooled intraday specification.",
            "\\item[$\\dagger$] Preliminary: the closing-interval forecasts of the pooled intraday random forest on "
            "all features, from an exercise still in progress.",
            "\\end{tablenotes}",
            "\\end{threeparttable}",
            "\\end{table}",
            "",
        ]
    )


def date_words(t: pd.Timestamp) -> str:
    return f"{t.day} {t.strftime('%B')} {t.year}"


def thousands(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def numbers(
    hist: pd.Series,
    days: pd.DatetimeIndex,
    n_cells: int,
    n_missing: int,
    n_live: int,
    mt_days: pd.DatetimeIndex,
    big: dict[str, float],
) -> str:
    n_before = int((hist.index < days[0]).sum())
    n_early = int(((rolling.EARLY >= days[0]) & (rolling.EARLY <= days[-1])).sum())
    lines = [
        "% AUTO-GENERATED by writeup/make_rehaul_oos_table.py -- do not edit. Macro-only (loaded once by the preamble",
        "% of paper_rehaul.tex); used by tab:rh_oos (generated/rehaul_results_oos.tex) and the section text.",
        f"\\newcommand{{\\rhoosMseExp}}{{{MSE_EXP}}}  % MSE printed x 10^this",
        f"\\newcommand{{\\rhoosBenchStart}}{{{date_words(hist.index[0])}}}  % first day of the R2_OOS benchmark's history",
        # f"\\newcommand{{\\rhoosBenchDaysBefore}}{{{n_before:,}}}".replace(",", "{,}")  # 2026-10-15: parked --
        # + "  % history days before the first out-of-sample day",  # the comment ("out-of-sample day")
        f"\\newcommand{{\\rhoosBenchDaysBefore}}{{{thousands(n_before)}}}"
        + "  % history days before the first day of the rolling span",
        f"\\newcommand{{\\rhoosNSeriesLive}}{{{n_live}}}  % series the retail-feed set adds to the HAR benchmark",
        f"\\newcommand{{\\rhoosNCells}}{{{n_cells}}}  % filled cells (model x predictor set x scheme)",
        f"\\newcommand{{\\rhoosNMissing}}{{{n_missing}}}  % cells marked --",
        # 2026-10-15 (user: accuracy over every rolling forecast): the common rolling span of tab:rh_oos
        f"\\newcommand{{\\rhoosSpanStart}}{{{date_words(days[0])}}}  % first day of the common rolling span",
        f"\\newcommand{{\\rhoosSpanEnd}}{{{date_words(days[-1])}}}  % last day of the common rolling span",
        f"\\newcommand{{\\rhoosDays}}{{{thousands(len(days))}}}"
        + "  % T_oos: trading days of the span, every forecast of tab:rh_oos exists (early closes dropped)",
        f"\\newcommand{{\\rhoosEarlyCloses}}{{{n_early}}}  % NYSE early-close days inside the span, dropped",
        f"\\newcommand{{\\rhoosNonOptDays}}{{{len(days) - len(mt_days)}}}"
        + "  % days of the span outside the option-data subsample of the economics (\\rhrDays days)",
        f"\\newcommand{{\\rhoosDMLag}}{{{_nw_lag(len(days))}}}  % dm_test's Newey-West lag on the span",
        f"\\newcommand{{\\rhoosBigDay}}{{{date_words(BIG_DAY)}}}"
        + "  % the day outside the option-data subsample that dominates the closing-interval trees' squared error",
        f"\\newcommand{{\\rhoosBigDaySEMin}}{{{big['se_min']:.0f}}}"
        + "  % its share of the closing-interval trees' total squared error on the span, %: min over the six",
        f"\\newcommand{{\\rhoosBigDaySEMax}}{{{big['se_max']:.0f}}}  % ... max",
        f"\\newcommand{{\\rhoosBigDayQLMax}}{{{big['ql_max']:.1f}}}"
        + "  % its share of their total QLIKE loss, %: max over the six",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- main


def main() -> None:
    sys.path.insert(0, str(REPO))
    from src.data.loading import SUBGROUPS

    # the predictor sets nest: HAR benchmark < parsimonious < retail-feed < full
    sizes = [len(SUBGROUPS[b]) for b, _ in SETS]
    for (a, _), (b, _) in zip(SETS, SETS[1:]):
        assert set(SUBGROUPS[a]) < set(SUBGROUPS[b]), (a, b)
    assert sizes[0] == 0
    # \rhrDays is the day count used here
    rn = (GEN / "rehaul_results_numbers.tex").read_text(encoding="utf-8")
    assert (
        f"\\providecommand{{\\rhrDays}}{{{NDAYS}}}" in rn
        or f"\\newcommand{{\\rhrDays}}{{{NDAYS}}}" in rn
    )
    # no \rhoos macro defined in any other generated file
    # 2026-10-15: parked -- "used" was tested; make_rehaul_results' notes now cite the span macros. Was:
    # for f in GEN.glob("*.tex"):
    #     if f != OUT_NUMBERS:
    #         assert ("\\rhoos" not in f.read_text(encoding="utf-8", errors="replace") or f == OUT_TABLE), f
    for f in GEN.glob("*.tex"):
        if f != OUT_NUMBERS:
            txt = f.read_text(encoding="utf-8", errors="replace")
            # no \newcommand / \providecommand / \renewcommand of an \rhoos macro
            assert "command{\\rhoos" not in txt, f

    data, m, days, mt_days = load_cells()
    hist = rv_history()
    # the benchmark's history is the scoring truth's own series, and holds every scored day
    assert hist.equals(rolling.history()), (
        "the R2 history differs from the rolling span's truth"
    )
    assert days.isin(hist.index).all(), "a scored day has no history print"
    truth = data[OLS]["rv"]
    assert np.array_equal(hist.reindex(days).to_numpy(), truth.to_numpy()), (
        "core_stats 16:00 RV != the rolling span's truth"
    )
    bench = ct_benchmark(hist, days)

    # 2026-10-15: 19 March 2020 (a Thursday without a same-day SPXW expiration in 2020, so outside the option-data
    # subsample) carries a large share of the closing-interval trees' squared error on the span, and little QLIKE
    assert BIG_DAY in days and BIG_DAY not in mt_days
    se_sh, ql_sh = [], []
    for b, _ in SETS:
        for est in ("lgbm", "xgb", "rf"):
            g = data[f"subtree_daily_{b}_{est}"]
            e2 = (g["rv"] - g["plain"]) ** 2
            r_ = (g["rv"] / g["plain"]).to_numpy(float)
            ql = pd.Series(r_ - np.log(r_) - 1.0, index=g.index)
            se_sh.append(100.0 * float(e2[BIG_DAY] / e2.sum()))
            ql_sh.append(100.0 * float(ql[BIG_DAY] / ql.sum()))
            if (
                b == "all_features"
            ):  # the largest squared error of the trees on all features
                assert e2.idxmax() == BIG_DAY, (b, est, e2.idxmax())
    big = {"se_min": min(se_sh), "se_max": max(se_sh), "ql_max": max(ql_sh)}
    assert big["se_min"] >= 25.0 and big["ql_max"] < 1.0, big
    print(
        f"{BIG_DAY.date()}: closing-interval trees' squared-error share {se_sh}, QLIKE share {ql_sh}"
    )

    rows, n_cells, n_missing, gui = [], 0, 0, []
    for name, panel, close, pool in model_rows():
        sets = [(b, lab) for b, lab in SETS if name != "OLS" or b == "baseline"]
        r: dict = {"model": name, "panel": panel, "sets": sets, "close": {}, "pool": {}}
        for b, _ in sets:
            for side, keys in (("close", close), ("pool", pool)):
                k = keys.get(b)
                if k is None:
                    r[side][b] = None
                    n_missing += 1
                    continue
                x = metrics(data[k], bench)
                # 2026-10-15: parked -- the QLIKE printed was master_table qlike_raw (866 days; now asserted on those
                # days in load_cells, and the printed QLIKE is on the rolling span: = rolling_eval metrics.csv below)
                # if k in m.index:
                #     assert np.isclose(x["qlike"], m.loc[k, "qlike_raw"], rtol=1e-12, atol=0), k
                r[side][b] = {"key": k, "m": x}
                n_cells += 1
                gui.append((name, b, side, k, x))
        rows.append(r)
    assert n_cells + n_missing == 2 * (1 + 6 * len(SETS))
    # 2026-10-15: every cell equals the rolling evaluation's common-span metrics (results/rolling_eval_2026-10-15)
    rm = pd.read_csv(REPO / "results" / "rolling_eval_2026-10-15" / "metrics.csv")
    rm = rm[rm["span"] == "common"].set_index("key")
    for _, _, _, k, x in gui:
        assert int(rm.loc[k, "n"]) == len(days), k
        for a, b in (("mse", "mse"), ("qlike", "qlike"), ("r2", "r2_oos_pct")):
            assert np.isclose(x[a], rm.loc[k, b], rtol=1e-10, atol=0), (k, a)

    GEN.mkdir(exist_ok=True)
    # OUT_TABLE.write_text(table(rows), encoding="utf-8", newline="\n")  # 2026-10-15 (user): parked -- split
    for side, spec in SIDE_SPEC.items():
        (GEN / spec["out"]).write_text(
            table_side(rows, side), encoding="utf-8", newline="\n"
        )
    OUT_NUMBERS.write_text(
        # numbers(hist, days, n_cells, n_missing, len(SUBGROUPS["live_feasible"])),  # 2026-10-15: + span, big day
        numbers(
            hist,
            days,
            n_cells,
            n_missing,
            len(SUBGROUPS["live_feasible"]),
            mt_days,
            big,
        ),
        encoding="utf-8",
        newline="\n",
    )

    print(
        f"history: {hist.index[0].date()} .. {hist.index[-1].date()}, {len(hist)} days "
        f"({int((hist.index < days[0]).sum())} before {days[0].date()}); benchmark on day 1 = {bench.iloc[0]:.4g}, "
        f"on the last day = {bench.iloc[-1]:.4g}; out-of-sample mean RV = {truth.mean():.4g}"
    )
    print(f"cells: {n_cells} filled, {n_missing} missing ('--')")
    print(
        f"{'model':14s} {'set':14s} {'scheme':6s} {'key':38s} {'MSEe10':>8s} {'QLIKE':>8s} {'R2 CT %':>8s} {'R2 GUI %':>8s}"
    )
    for name, b, side, k, x in gui:
        print(
            f"{name:14s} {b:14s} {side:6s} {k:38s} {x['mse'] * 10**MSE_EXP:8.3f} {x['qlike']:8.4f} "
            f"{x['r2']:8.2f} {x['r2_gui']:8.2f}"
        )
    print(f"wrote {OUT_TABLE.relative_to(REPO)} and {OUT_NUMBERS.relative_to(REPO)}")


if __name__ == "__main__":
    main()
