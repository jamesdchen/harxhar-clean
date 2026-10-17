"""Tables and figures of the rehauled paper's Results section (writeup/rehaul/06_results_floats.tex).

Outline: writeup/PAPER_REHAUL_OUTLINE_2026-10-15.md, section 6 (T1, T2, T3; F1, F2, F3, F5).

SCORING (one convention for every number written here)
  forecast   plain back-transform  sigma2_t = f_t^2 B_t, no recalibration;
  # target     the winsorised 15:30-16:00 RV (``target`` of master_table_daily.parquet);  -- 2026-10-15 (user): parked -- scored against the trimmed target; scores now use actual RV (data audit #10)
  truth      the ACTUAL (unwinsorised) 15:30-16:00 RV (``rv_unclipped`` of master_table_daily.parquet; 2026-10-15,
             data audit #10: the models still fit the winsorised target, ``target``, but every score here is
             against the actual RV -- QLIKE, the squared error, DM);
  # days       the 866 trade days 2020-01-03 .. 2024-04-30, early closes excluded;  -- 2026-10-15 (user): parked --
  #            accuracy is scored over every rolling forecast, not over the option-data days only (next line)
  days       ACCURACY (QLIKE, squared error, DM, ranks, the figures' loss series): the common rolling span of
             experiments/rolling_eval_2026_10_15.py (common_span(): 2020-01-03 .. 2024-04-30, the 1,082 trading days on
             which every forecast of tab:rh_oos exists, early closes dropped; every forecast comes from a model
             re-estimated on its trailing window).  User (2026-10-15): "Out of sample period doesn't make sense for
             forecasts because we're performing rolling backtests. The forecasting results need to be calculated to
             reflect such (that the out of sample forecasts are done on a rolling basis after the burn in training
             window)".  TRADES (Sharpe, HAC t): the 866 days of the option-data subsample (a same-day SPXW
             expiration), early closes excluded, as before.  --span mt866 re-runs every accuracy number on the 866
             days and asserts the values of record (the reproduction gates; it writes under
             results/rolling_eval_2026-10-15/mt866_check/, never into generated/ or figures/);
  loss       daily QLIKE  L_t = r - log r - 1,  r = RV_t / sigma2_t;  squared error in variance space;
  DM         src.evaluation.diebold_mariano.dm_test on the daily QLIKE difference (a minus b): Bartlett
             Newey-West at dm_test's automatic lag (6 on 866 days and on 1,082), HLN factor, two-sided normal p;
             DM < 0 favours a;
  trade      sign(s) straddle at the midpoint: q_t = +1 if sigma2_t > IV variance else -1, return q_t R_t;
             Sharpe = mean / sd * sqrt(252); the change is tested by notebooks/atm_straddle_lib.newey_west_t
             on the daily return difference (lag floor(1.5 n^(1/3)) = 14);
  95 % band  of a % change: pct +- 1.96 |pct / DM| (the interval the DM test inverts).

SOURCES
  results/close_master_table/master_table.csv, master_table_daily.parquet, paper_model_set.csv  (the model set);
  results/pooled_trees_2026-10-07/yhat_pooltree_<est>_baseline.parquet, ..._rf_all_features_INTERIM1600.parquet
      (pooled trees not in the master table: 16:00 rows, plain = yhat^2 x baseline, as the GUI's plain_16);
      the pooled LightGBM / XGBoost on all features are the master table's ``lgbm`` / ``xgb`` rows;
  results/puffer_forecast_2026-10-12/qlike_study/score_rinitq/forecasts.csv  (the per-bar ridge-init MinGRU,
      QLIKE-tuned, 5-seed mean), scored exactly as writeup/master_table_gui/rows.py _nn_daily: row -> date by the
      feature study's inputs.json, B from the close-study targets.parquet, plain = f^2 B, target / IV variance
      of the OLS row, R from the quote book results/atm_straddle_0dte_1530/daily_a0.parquet;
  results/puffer_forecast_2026-10-12/pooled_rinit_study/runs/pwalk_rinitq_composite/block*/refit/preds.csv  (the
      pooled ridge-init MinGRU, QLIKE-tuned, refit monthly on its trailing 24,000 bars): seed mean per panel row, the
      16:00 rows of _work/pooled_walk/rows.csv, plain = f^2 x the panel's 16:00 baseline -- rows.py's pooled path and
      experiments/puffer_forecast/pooled_score.py view 2; gated against pooled_rinit_study/at1600_scores.csv and
      at1600_contrasts.csv.

WRITES
  writeup/generated/rehaul_results_numbers.tex     \\rhr... macros used by the captions;
  writeup/generated/rehaul_results_t1_axis1.tex    T1  HAR + calendar vs all features, within each family;
  writeup/generated/rehaul_results_t2_pooled.tex   T2  pooled minus per-bar;
  writeup/generated/rehaul_results_t3_ranking.tex  T3  headline forecasts ranked by QLIKE, vs the per-bar OLS;
  writeup/generated/rehaul_results_t3_full.tex     T3 (full) every forecast, a longtable;
  writeup/figures/rehaul_results_f1_forest.pdf     F1  forest plot of T1;
  writeup/figures/rehaul_results_f2_dumbbell.pdf   F2  pooled vs per-bar;
  writeup/figures/rehaul_results_f3_cumulative.pdf F3  cumulative daily QLIKE difference vs the OLS;
  writeup/figures/rehaul_results_f5_se_share.pdf   F5  squared-error share by day.
  (.png previews beside the figures.)

Known values are asserted (the outline's numbers): the run stops if the data no longer reproduce them.

TABLE LEGIBILITY (2026-10-15)
  The tables follow writeup/NOTATION_GUIDE_2026-10-15.md sections 6b and 8 and TABLE_FIXES_FROM_FLOATS_2026-10-15.md
  (the conventions of make_rehaul_economics.py): short captions; Notes of about five sentences below, referring to
  Methods; stars for the 10/5/1 % levels (two-sided normal p) on the estimate, with its test statistic in parentheses
  in its own column, one row per forecast (under the estimate, T1 was taller than a page); no p columns; Panel A
  closing-interval models, Panel B intraday models (T2: Panel A HAR benchmark, Panel B full predictor set);
  model names in words; r columns whose unused star slots are phantoms, so decimals align.
  Every number a note cites is a \\rhr... macro of rehaul_results_numbers.tex, which is macro-only and loaded once by
  the paper's preamble; the table files \\input nothing.  The numbers file also carries per-model macros against the
  HAR benchmark (its header lists the families), so the prose can drop its typed values.  T3 (full) is meant for the
  appendix (its \\input moves from 06_results_floats.tex to 10_appendix_floats.tex).  Replaced code is parked in
  comments marked "2026-10-15 (table legibility)".

NN PARKED (2026-10-15, user: "park all of the NN stuff for now")
  The per-bar and pooled MinGRU leave every table and figure: T1 / T2 rows (FAMILY_ROWS -> NN_ROW), the four MinGRU
  forecasts of the ranking (T3, T3 full: ranks recompute among the 116 others), the MinGRU line of F3 (and F5) and the
  MinGRU clauses of the notes.  Their forecasts are still loaded and gated against the studies' values, and their
  \\rhr...Nn... macros (CumNn*, NnAllQ, NnAf*, NNPooledSeeds, Q/VsBm/Af/PoolVsClose Nn*) stay defined with unchanged
  values but unused by the floats, so prose that still cites them compiles.  Parked code is marked
  "2026-10-15 (user): parked -- NN content parked for now".

Run:  python writeup/make_rehaul_results.py [--span rolling | mt866]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent  # writeup/
REPO = ROOT.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "notebooks"))
sys.path.insert(0, str(REPO / "experiments"))
from src.evaluation.diebold_mariano import dm_test  # noqa: E402
import atm_straddle_lib as asl  # noqa: E402
import rolling_eval_2026_10_15 as rolling  # noqa: E402  (2026-10-15: the common rolling span of every accuracy number)

# 2026-10-15 (user, restructure): the GUI's sizing scorer, as make_rehaul_economics.py calls it (pooled sizing medians)
from master_table_gui import rows as gui  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

GEN = ROOT / "generated"
FIG = ROOT / "figures"
MT = REPO / "results" / "close_master_table"
POOLTREE = REPO / "results" / "pooled_trees_2026-10-07"
NN_FC = (
    REPO
    / "results"
    / "puffer_forecast_2026-10-12"
    / "qlike_study"
    / "score_rinitq"
    / "forecasts.csv"
)
NN_INPUTS = (
    REPO / "results" / "feature_mcs_2026-10-08" / "_work" / "inputs" / "inputs.json"
)
NN_TARGETS = (
    REPO / "results" / "close_studies_2026-10-03" / "forecasts" / "targets.parquet"
)
NN_POOLED = REPO / "results" / "puffer_forecast_2026-10-12" / "pooled_rinit_study"
NN_POOLED_RUN = NN_POOLED / "runs" / "pwalk_rinitq_composite"
NN_POOLED_ROWS = (
    REPO
    / "results"
    / "puffer_forecast_2026-10-12"
    / "_work"
    / "pooled_walk"
    / "rows.csv"
)
NN_POOLED_SEEDS: list[int] = []  # filled by nn_pooled_frames
QUOTES = REPO / "results" / "atm_straddle_0dte_1530" / "daily_a0.parquet"
GUI_DAILY = REPO / "results" / "close_master_table_gui" / "master_table_daily.parquet"

OLS = "sub_ols_baseline"
NDAYS = 866  # the option-data subsample: trades and Sharpe ratios (and every accuracy number under --span mt866)
# 2026-10-15 (user): accuracy over every rolling forecast -- SPAN "rolling" scores QLIKE / squared error / DM / ranks on
# rolling.common_span() (ACC holds each forecast's frame there, keyed by id() of its trade frame in `by`); "mt866"
# scores them on the 866 days, as before, and asserts the values of record
SPAN = "rolling"
ACC: dict[int, pd.DataFrame] = {}
NN_PLAIN: dict[
    str, pd.Series
] = {}  # the MinGRU forecasts on every day they exist (filled by nn_frames / nn_pooled)
ANN = 252.0**0.5
SIG = 0.05
LEVELS = (0.10, 0.05, 0.01)  # the star levels (2026-10-15, table legibility)
DM_LAG = 6  # dm_test's automatic Newey-West lag on 866 days and on the 1,082-day rolling span (asserted in pair)
HAC_LAG = 14  # newey_west_t's lag floor(1.5 n^(1/3)) on 866 days (asserted in pair)
Z95 = 1.959963984540054
CRASH = (
    pd.Timestamp("2020-02-24"),
    pd.Timestamp("2020-04-30"),
)  # the outline's exclusion window
TOP_DAYS = 10  # the tail check: days of largest gain over the short straddle dropped (= make_rehaul_economics.TOP_DAYS)

# --------------------------------------------------------------------------- data


def _day(values) -> np.ndarray:
    t = pd.to_datetime(pd.Series(values))
    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_convert("America/New_York").dt.tz_localize(None)
    return t.dt.normalize().to_numpy()


def load_master() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    m = pd.read_csv(MT / "master_table.csv").set_index("key")
    s = pd.read_csv(MT / "paper_model_set.csv")
    d = pd.read_parquet(
        MT / "master_table_daily.parquet",
        columns=[
            "key",
            "day",
            "plain",
            "target",
            "rv_unclipped",
            "iv_var",
            "qlike_raw",
            "q",
            "ret_mid",
            "ret_crossed",
        ],
    )
    d["day"] = _day(d["day"])
    by = {k: f.set_index("day").sort_index() for k, f in d.groupby("key")}
    return m, s, by


def frame(plain: pd.Series, ref: pd.DataFrame, quotes: pd.DataFrame) -> pd.DataFrame:
    """Daily frame of a forecast not in the master table, on the OLS row's days (GUI construction)."""
    j = (
        ref[["target", "rv_unclipped", "iv_var"]]
        .join(plain.rename("plain"), how="inner")
        .join(quotes, how="inner")
    )
    if len(j) != NDAYS or not np.isfinite(j["plain"]).all():
        raise ValueError("a forecast is missing on a trade day")
    # r = (j["target"] / j["plain"]).to_numpy(float)  # 2026-10-15 (user): parked -- scored against the trimmed target; scores now use actual RV (data audit #10)
    r = (j["rv_unclipped"] / j["plain"]).to_numpy(float)
    q = np.where(j["plain"] > j["iv_var"], 1.0, -1.0)
    return pd.DataFrame(
        {
            "plain": j["plain"],
            "target": j["target"],
            "rv_unclipped": j["rv_unclipped"],
            "iv_var": j["iv_var"],
            "qlike_raw": r - np.log(r) - 1.0,
            "q": q,
            "ret_mid": q * j["R"].to_numpy(float),
            "ret_crossed": crossed(q, j),
        },
        index=j.index,
    )


def crossed(q: np.ndarray, j: pd.DataFrame) -> np.ndarray:
    """Return crossing the spread: buy at the ask, sell at the bid (writeup/master_table_gui/rows.py _crossed_return)."""
    ex = j["exit"].to_numpy(float)
    bid = (j["bid_c"] + j["bid_p"]).to_numpy(float)
    ask = (j["ask_c"] + j["ask_p"]).to_numpy(float)
    return np.where(
        q > 0, q * (ex / ask - 1.0), np.where(q < 0, q * (ex / bid - 1.0), 0.0)
    )


def plain_16(path: Path) -> pd.Series:
    """16:00 ET rows of a yhat table: f^2 x B (writeup/master_table_gui/rows.py plain_16)."""
    y = pd.read_parquet(path, columns=["t", "yhat", "baseline"])
    t = pd.to_datetime(y["t"], utc=True).dt.tz_convert("America/New_York")
    at16 = ((t.dt.hour == 16) & (t.dt.minute == 0)).to_numpy()
    v = y["yhat"].to_numpy(float)[at16] ** 2 * y["baseline"].to_numpy(float)[at16]
    return pd.Series(v, index=_day(t[at16].dt.tz_localize(None)))


def nn_frames(ref: pd.DataFrame, quotes: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The per-bar ridge-init MinGRU, QLIKE-tuned: writeup/master_table_gui/rows.py _nn_daily, plain f^2 B."""
    dates = pd.Series(json.loads(NN_INPUTS.read_text())["dates"])
    tg = pd.read_parquet(NN_TARGETS, columns=["date", "baseline"])
    b_day = pd.Series(tg["baseline"].to_numpy(float), index=_day(tg["date"]))
    fc = pd.read_csv(NN_FC)
    assert (fc["count"] == 5).all(), "a MinGRU forecast is not the mean of 5 seeds"
    out = {}
    for b in ("baseline", "all_features"):
        e = fc[fc["design"] == b]
        f = pd.Series(
            e["mean"].to_numpy(float), index=_day(dates.iloc[e["row"].to_numpy()])
        )
        assert not f.index.duplicated().any()
        out[f"mingru_rinitq_{b}"] = frame(f**2 * b_day.reindex(f.index), ref, quotes)
        NN_PLAIN[f"mingru_rinitq_{b}"] = f**2 * b_day.reindex(
            f.index
        )  # every day (2026-10-15: rolling span)
    return out


def nn_pooled_frames(
    ref: pd.DataFrame, quotes: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """The pooled ridge-init MinGRU, QLIKE-tuned, at the panel's 16:00 bar (rows.py's pooled path, pooled_score view 2):
    f = the seed mean of every block's refit forecasts, plain = f^2 x the panel's own 16:00 baseline."""
    panel = pd.read_csv(NN_POOLED_ROWS, usecols=["row", "t_et", "baseline"]).set_index(
        "row"
    )
    t = pd.to_datetime(panel["t_et"])
    at16 = panel[((t.dt.hour == 16) & (t.dt.minute == 0)).to_numpy()]
    at16 = at16.assign(day=_day(pd.to_datetime(at16["t_et"])))
    blocks = sorted(NN_POOLED_RUN.glob("block*/refit/preds.csv"))
    assert len(blocks) == 5, blocks
    pr = pd.concat(
        [pd.read_csv(f, usecols=["id", "kind", "row", "f"]) for f in blocks],
        ignore_index=True,
    )
    pr = pr[pr["kind"] == "fc"].copy()
    pr["design"] = pr["id"].str.extract(r"^prefit_(baseline|all_features)_r")[0]
    out = {}
    for b in ("baseline", "all_features"):
        m = pr[pr["design"] == b].groupby("row")["f"].agg(["mean", "count"])
        assert (m["count"] == m["count"].iloc[0]).all(), (
            "seed counts differ across rows"
        )
        NN_POOLED_SEEDS.append(int(m["count"].iloc[0]))
        g = at16.join(m, how="inner")
        assert not g["day"].duplicated().any(), "two 16:00 bars on one day"
        plain = pd.Series(
            g["mean"].to_numpy(float) ** 2 * g["baseline"].to_numpy(float),
            index=g["day"],
        )
        out[f"mingru_pooled_rinitq_{b}"] = frame(plain, ref, quotes)
        NN_PLAIN[f"mingru_pooled_rinitq_{b}"] = (
            plain  # every 16:00 row (2026-10-15: rolling span)
        )
    assert len(set(NN_POOLED_SEEDS)) == 1
    return out


def load_all() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    m, s, by = load_master()
    ref = by[OLS]
    quotes = pd.read_parquet(
        QUOTES, columns=["R", "exit", "bid_c", "bid_p", "ask_c", "ask_p"]
    )
    quotes.index = pd.to_datetime(quotes.index).normalize()
    # gate: the master table's mid / crossed returns are the quote book's, so the added rows' returns are comparable
    jr = ref.join(quotes, how="inner")
    assert len(jr) == NDAYS and np.allclose(
        jr["ret_mid"], jr["q"] * jr["R"], rtol=1e-12, atol=1e-15
    ), "quote-book R"
    assert np.allclose(
        jr["ret_crossed"], crossed(jr["q"].to_numpy(), jr), rtol=1e-10, atol=1e-14
    ), "crossed"
    # gate: the master table's QLIKE is the plain f^2 B QLIKE of its own daily columns (against the actual RV)
    # r = (ref["target"] / ref["plain"]).to_numpy(float)  # 2026-10-15 (user): parked -- scored against the trimmed target; scores now use actual RV (data audit #10)
    r = (ref["rv_unclipped"] / ref["plain"]).to_numpy(float)
    assert np.allclose(ref["qlike_raw"], r - np.log(r) - 1.0, rtol=1e-10)
    if GUI_DAILY.is_file():  # the GUI's master copy carries the same OLS target and RV (the NN scoring reads it there)
        g = pd.read_parquet(GUI_DAILY, columns=["key", "day", "target", "rv_unclipped"])
        g = g[g["key"] == OLS].assign(day=lambda x: _day(x["day"])).set_index("day")
        for c in ("target", "rv_unclipped"):
            assert np.allclose(g[c].reindex(ref.index), ref[c], rtol=1e-12), (
                f"GUI vs master OLS {c}"
            )
    by.update(nn_frames(ref, quotes))
    by.update(nn_pooled_frames(ref, quotes))
    for est in ("lgbm", "xgb", "rf"):
        by[f"pooltree_{est}_baseline"] = frame(
            plain_16(POOLTREE / f"yhat_pooltree_{est}_baseline.parquet"), ref, quotes
        )
    by["pooltree_rf_all_features"] = frame(
        plain_16(POOLTREE / "yhat_pooltree_rf_all_features_INTERIM1600.parquet"),
        ref,
        quotes,
    )
    by["pooltree_lgbm_all_features"] = by["lgbm"]
    by["pooltree_xgb_all_features"] = by["xgb"]
    if SPAN == "rolling":
        build_acc(s, by)
    return m, s, by


ACC_ALIAS = {
    "pooltree_lgbm_all_features": "lgbm",
    "pooltree_xgb_all_features": "xgb",
}  # the master table's keys


def build_acc(s: pd.DataFrame, by: dict[str, pd.DataFrame]) -> None:
    """2026-10-15 (user: accuracy over every rolling forecast): the accuracy frame (plain, actual RV, daily QLIKE) of
    every forecast this script scores, on the common rolling span rolling.common_span(), keyed by id() of its trade
    frame (aliased keys share one frame).  Gate: on the 866 option-data days it carries the trade frame's forecast
    and truth."""
    days = rolling.common_span()
    rv = rolling.history().reindex(days)
    assert np.isfinite(rv.to_numpy()).all() and (rv > 0).all()
    keys = (
        set(s["key"])
        | set(NN_PLAIN)
        | {k for _, _, pb, pool in FAMILY_ROWS + [NN_ROW] for k in (*pb, *(pool or ()))}
    )
    keys |= {f"pooltree_{e}_baseline" for e in ("lgbm", "xgb", "rf")} | {
        "pooltree_rf_all_features"
    }
    for k in sorted(keys):
        f = by[ACC_ALIAS.get(k, k)]
        if id(f) in ACC:
            continue
        plain = (
            NN_PLAIN[k]
            if k in NN_PLAIN
            else rolling.scored(ACC_ALIAS.get(k, k))["plain"]
        )
        p = plain.reindex(days)
        assert np.isfinite(p.to_numpy()).all() and (p > 0).all(), (
            k,
            int(p.isna().sum()),
        )
        r = (rv / p).to_numpy(float)
        a = pd.DataFrame(
            {"plain": p, "rv_unclipped": rv, "qlike_raw": r - np.log(r) - 1.0},
            index=days,
        )
        assert f.index.isin(days).all(), k
        rel = float(np.max(np.abs(a["plain"].reindex(f.index) / f["plain"] - 1.0)))
        assert rel <= 1e-10, (k, rel)
        assert np.array_equal(
            a["rv_unclipped"].reindex(f.index).to_numpy(), f["rv_unclipped"].to_numpy()
        ), k
        ACC[id(f)] = a
    print(
        f"accuracy frames: {len(ACC)} forecasts on the common rolling span {days[0].date()} .. {days[-1].date()} "
        f"({len(days)} days; trades on the {NDAYS} option-data days)"
    )


def acc_of(f: pd.DataFrame) -> pd.DataFrame:
    """The frame every accuracy number is computed on: the rolling-span frame (SPAN rolling) or the trade frame."""
    return ACC[id(f)] if SPAN == "rolling" else f


def n_acc() -> int:
    return len(rolling.common_span()) if SPAN == "rolling" else NDAYS


# --------------------------------------------------------------------------- statistics


def sharpe(x: pd.Series) -> float:
    return float(x.mean() / x.std(ddof=1) * ANN)


def level(f: pd.DataFrame) -> dict:
    # e = f["target"] - f["plain"]  # 2026-10-15 (user): parked -- scored against the trimmed target; scores now use actual RV (data audit #10)
    # e = f["rv_unclipped"] - f["plain"]  # 2026-10-15 (user): parked -- accuracy on the rolling span (acc_of)
    a = acc_of(f)
    e = a["rv_unclipped"] - a["plain"]
    return {
        "qlike": float(a["qlike_raw"].mean()),
        "mse": float((e**2).mean()),
        "sharpe": sharpe(f["ret_mid"]),
        "sharpe_x": sharpe(f["ret_crossed"]),
    }


def pair(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """a minus b on the common days: QLIKE % change, DM, p; squared-error % change and DM; Sharpe change, HAC t.
    2026-10-15: the accuracy statistics on acc_of() (the common rolling span), the trade statistics on the 866 days."""
    # j = a[["qlike_raw", "ret_mid", "rv_unclipped", "plain"]].join(  # 2026-10-15 (user): parked -- one frame for
    #     b[["qlike_raw", "ret_mid", "plain"]], rsuffix="_b", how="inner"  # accuracy and trades; split below
    # )
    # assert len(j) == NDAYS, len(j)
    j = acc_of(a)[["qlike_raw", "rv_unclipped", "plain"]].join(
        acc_of(b)[["qlike_raw", "plain"]], rsuffix="_b", how="inner"
    )
    assert len(j) == n_acc(), len(j)
    jt = a[["ret_mid"]].join(b[["ret_mid"]], rsuffix="_b", how="inner")
    assert len(jt) == NDAYS, len(jt)
    dm = dm_test(j["qlike_raw"].to_numpy(), j["qlike_raw_b"].to_numpy())
    assert dm["hac_lag"] == DM_LAG
    # se_a, se_b = (j["target"] - j["plain"]) ** 2, (j["target"] - j["plain_b"]) ** 2  # 2026-10-15 (user): parked -- scored against the trimmed target; scores now use actual RV (data audit #10)
    se_a = (j["rv_unclipped"] - j["plain"]) ** 2
    se_b = (j["rv_unclipped"] - j["plain_b"]) ** 2
    dms = dm_test(se_a.to_numpy(), se_b.to_numpy())
    # t, lag = asl.newey_west_t(j["ret_mid"] - j["ret_mid_b"])  # 2026-10-15: trades on jt (parked: one frame)
    t, lag = asl.newey_west_t(jt["ret_mid"] - jt["ret_mid_b"])
    assert lag == HAC_LAG
    pct = 100.0 * (j["qlike_raw"].mean() / j["qlike_raw_b"].mean() - 1.0)
    return {
        "qa": float(j["qlike_raw"].mean()),
        "qb": float(j["qlike_raw_b"].mean()),
        "pct": float(pct),
        "dm": float(dm["dm"]),
        "p": float(dm["p"]),
        "hw": float(Z95 * abs(pct / dm["dm"])) if dm["dm"] != 0 else float("nan"),
        "mse_pct": float(100.0 * (se_a.mean() / se_b.mean() - 1.0)),
        "dm_mse": float(dms["dm"]),
        "p_mse": float(dms["p"]),
        # "dsh": sharpe(j["ret_mid"]) - sharpe(j["ret_mid_b"]),  # 2026-10-15: trades on jt
        "dsh": sharpe(jt["ret_mid"]) - sharpe(jt["ret_mid_b"]),
        "t": float(t),
        "p_t": float(math.erfc(abs(t) / math.sqrt(2.0))),
    }


# --------------------------------------------------------------------------- model definitions

LIN, TREE, NN, POOL = "linear", "tree", "nn", "pooled"
EST_NAME = {
    "ridge": "Ridge",
    "lasso": "Lasso",
    "enet": "Elastic net",
    "lgbm": "LightGBM",
    "xgb": "XGBoost",
    "rf": "Random forest",
}
NN_NAME = "MinGRU"
# T1 / T2 / F1 / F2 rows, family order linear -> trees -> NN; (name, family, per-bar keys, pooled keys) per input set
FAMILY_ROWS = [
    (
        "Ridge",
        LIN,
        ("subpt_ridge_baseline", "subpt_ridge_all_features"),
        ("poolpt_ridge_baseline", "poolpt_ridge_all_features"),
    ),
    (
        "Lasso",
        LIN,
        ("subpt_lasso_baseline", "subpt_lasso_all_features"),
        ("poolpt_lasso_baseline", "poolpt_lasso_all_features"),
    ),
    (
        "Elastic net",
        LIN,
        ("subpt_enet_baseline", "subpt_enet_all_features"),
        ("poolpt_enet_baseline", "poolpt_enet_all_features"),
    ),
    (
        "LightGBM",
        TREE,
        ("subtree_daily_baseline_lgbm", "subtree_daily_all_features_lgbm"),
        ("pooltree_lgbm_baseline", "pooltree_lgbm_all_features"),
    ),
    (
        "XGBoost",
        TREE,
        ("subtree_daily_baseline_xgb", "subtree_daily_all_features_xgb"),
        ("pooltree_xgb_baseline", "pooltree_xgb_all_features"),
    ),
    (
        "Random forest",
        TREE,
        ("subtree_daily_baseline_rf", "subtree_daily_all_features_rf"),
        ("pooltree_rf_baseline", "pooltree_rf_all_features"),
    ),
    # 2026-10-15 (user): parked -- NN content parked for now (the MinGRU row is NN_ROW below; no table or figure shows it)
    # (
    #     NN_NAME,
    #     NN,
    #     ("mingru_rinitq_baseline", "mingru_rinitq_all_features"),
    #     ("mingru_pooled_rinitq_baseline", "mingru_pooled_rinitq_all_features"),
    # ),
]
# 2026-10-15 (user): NN content parked for now -- the MinGRU row, out of FAMILY_ROWS: main() still pairs it, for the value
# gates and the \rhr...Nn... macros (kept defined, unused by the floats), but drops it from every table and figure.
NN_ROW = (
    NN_NAME,
    NN,
    ("mingru_rinitq_baseline", "mingru_rinitq_all_features"),
    ("mingru_pooled_rinitq_baseline", "mingru_pooled_rinitq_all_features"),
)
INTERIM = {
    "pooltree_rf_all_features"
}  # the pooled forest on all features: 16:00 rows of a walk still running


# 2026-10-15 (table legibility): parked -- code labels (per-bar, HAR + cal., rand. search (SE), refit 10, earlier panel); replaced by describe_key below
# def rung_label(key: str, m: pd.DataFrame) -> tuple[str, str, str, str]:
#     """(forecast name, family, fit, input set) of a key, for the ranking tables."""
#     sets = {"baseline": "HAR + cal.", "all_features": "All"}
#     if key.startswith(("mingru_rinitq_", "mingru_pooled_rinitq_")):
#         return (
#             "MinGRU (ridge-init, QLIKE-tuned)",
#             NN,
#             "pooled" if "_pooled_" in key else "per-bar",
#             sets[key.split("rinitq_")[1]],
#         )
#     if key.startswith("pooltree_"):
#         est, b = key.split("_")[1], key.split("_", 2)[2]
#         return (
#             EST_NAME[est],
#             TREE,
#             "pooled",
#             sets[b],
#         )  # the interim forest carries a dagger in the tables
#     paper = {
#         "a0": ("OLS, earlier panel", LIN, "pooled", "HAR + cal."),
#         "blk2": ("Block-diagonal ridge, earlier panel", LIN, "pooled", "All"),
#         "blk2_inc": (
#             "Block-diag.\\ ridge, no FOMC, earlier panel",
#             LIN,
#             "pooled",
#             "All",
#         ),
#         "b2lasso": ("Lasso (fixed penalty), earlier panel", LIN, "pooled", "All"),
#         "lasso_t": ("Lasso (causally tuned), earlier panel", LIN, "pooled", "All"),
#         "lasso_f": ("Lasso (fixed penalty), earlier panel", LIN, "pooled", "All"),
#         "enet": ("Elastic net (causally tuned), earlier panel", LIN, "pooled", "All"),
#     }
#     if key in paper:
#         return paper[key]
#     if key in ("lgbm", "xgb"):  # the earlier paper's pooled trees on all features
#         return EST_NAME[key], TREE, "pooled", "All"
#     if key == OLS:
#         return "OLS", LIN, "per-bar", "HAR + cal."
#     b = "baseline" if "_baseline" in key else "all_features"
#     for pre, fit in (("subpt_", "per-bar"), ("poolpt_", "pooled")):
#         if key.startswith(pre):
#             return EST_NAME[key[len(pre) :].split("_")[0]], LIN, fit, sets[b]
#     assert key.startswith("subtree_"), key
#     est = (
#         key.rsplit("_", 1)[1]
#         if key.split("_")[-1] in ("lgbm", "xgb", "rf")
#         else key.split("_")[1]
#     )
#     rest = key[len("subtree_") :].replace(f"_{b}", "").replace(f"{est}", "").strip("_")
#     rest = rest.replace("__", "_")
#     if rest == "":
#         how = "fixed, refit 10"
#     elif rest == "daily":
#         how = "fixed, daily refit"
#     elif rest in ("tuned", "tunedq", "tuned_daily", "tunedq_daily"):
#         loss = "QLIKE" if rest.startswith("tunedq") else "SE"
#         how = f"rand.\\ search ({loss}), " + (
#             "daily refit" if rest.endswith("daily") else "refit 10"
#         )
#     else:
#         tok = rest.split("_")
#         loss = "QLIKE" if tok[0] == "optunaq" else "SE"
#         tp = tok[1][2:]
#         k = tok[2][1:] if len(tok) > 2 else "50"
#         how = f"Optuna ({loss}), tune {tp}, top {k}"
#     return f"{EST_NAME[est]}, {how}", TREE, "per-bar", sets[b]

# 2026-10-15 (table legibility): the guide's names for a forecast key (NOTATION_GUIDE section 8: model names in words,
# panels instead of a Fit column, no code labels or arm names), replacing rung_label above.
SET_NAME = {"baseline": "HAR", "all_features": "Full"}
# The earlier draft's pooled intraday linear models, Panel C of T3 (full): (model, specification, predictor set).
# notebooks/atm_straddle_lib.py:35-40: blk2 and lasso_f are fitted on the FOMC panel; blk2_inc, lasso_t and enet on the
# earlier panel, which carries no FOMC predictors; experiments/master_table_close.py:532-539: b2lasso is lasso_f's
# fixed penalty on that earlier panel.
EARLIER = {
    "a0": ("OLS", "--", "HAR"),
    "blk2": ("Ridge", "Block penalty", "Full"),
    "blk2_inc": ("Ridge", "Block penalty, no FOMC", "Full"),
    "lasso_f": ("Lasso", "Fixed penalty", "Full"),
    "b2lasso": ("Lasso", "Fixed penalty, no FOMC", "Full"),
    "lasso_t": ("Lasso", "Re-tuned penalty, no FOMC", "Full"),
    "enet": ("Elastic net", "Re-tuned penalty, no FOMC", "Full"),
}
EARLIER_TREES = (
    "lgbm",
    "xgb",
)  # the earlier draft's pooled trees on its full predictor set (earlier panel)


def tree_spec(rest: str) -> str:
    """Specification of a closing-interval tree from its key's tuning tokens: fixed hyperparameters or random search,
    with the re-estimation frequency ("every 10 d" or "daily"); Optuna, with the re-tuning frequency and the number of
    trials searched. The selection loss is MSE or QLIKE (Methods, sec:rh_methods_tuning)."""
    if rest == "":
        return "Fixed, every 10 d"
    if rest == "daily":
        return "Fixed, daily"
    if rest in ("tuned", "tunedq", "tuned_daily", "tunedq_daily"):
        loss = "QLIKE" if rest.startswith("tunedq") else "MSE"
        return f"Random search ({loss}), " + (
            "daily" if rest.endswith("daily") else "every 10 d"
        )
    tok = rest.split("_")
    assert tok[0] in ("optuna", "optunaq") and tok[1].startswith("tp"), rest
    loss = "QLIKE" if tok[0] == "optunaq" else "MSE"
    tp = tok[1][2:]
    k = tok[2][1:] if len(tok) > 2 else "50"
    return f"Optuna ({loss}), {'daily' if tp == '1' else tp + ' d'}, {k} trials"


def describe_key(key: str) -> dict[str, str]:
    """model (in words), spec ("--" if none), family, fit (per-bar / pooled: internal), panel (A closing-interval,
    B intraday, C the earlier draft's intraday linear models) and inputs (HAR / Full) of a key."""
    if key.startswith(("mingru_rinitq_", "mingru_pooled_rinitq_")):
        pooled = "_pooled_" in key
        return dict(
            model=NN_NAME,
            spec="--",
            family=NN,
            fit="pooled" if pooled else "per-bar",
            panel="B" if pooled else "A",
            inputs=SET_NAME[key.split("rinitq_")[1]],
        )
    if key.startswith("pooltree_"):
        est, b = key.split("_")[1], key.split("_", 2)[2]
        return dict(
            model=EST_NAME[est],
            spec="--",
            family=TREE,
            fit="pooled",
            panel="B",
            inputs=SET_NAME[b],
        )
    if key in EARLIER:
        mdl, spec, inputs = EARLIER[key]
        return dict(
            model=mdl, spec=spec, family=LIN, fit="pooled", panel="C", inputs=inputs
        )
    if key in EARLIER_TREES:
        return dict(
            model=EST_NAME[key],
            spec="Earlier draft, no FOMC",
            family=TREE,
            fit="pooled",
            panel="B",
            inputs="Full",
        )
    if key == OLS:
        return dict(
            model="OLS", spec="--", family=LIN, fit="per-bar", panel="A", inputs="HAR"
        )
    b = "baseline" if "_baseline" in key else "all_features"
    for pre, fit in (("subpt_", "per-bar"), ("poolpt_", "pooled")):
        if key.startswith(pre):
            return dict(
                model=EST_NAME[key[len(pre) :].split("_")[0]],
                spec="--",
                family=LIN,
                fit=fit,
                panel="A" if fit == "per-bar" else "B",
                inputs=SET_NAME[b],
            )
    assert key.startswith("subtree_"), key
    est = (
        key.rsplit("_", 1)[1]
        if key.split("_")[-1] in ("lgbm", "xgb", "rf")
        else key.split("_")[1]
    )
    rest = key[len("subtree_") :].replace(f"_{b}", "").replace(f"{est}", "").strip("_")
    rest = rest.replace("__", "_")
    return dict(
        model=EST_NAME[est],
        spec=tree_spec(rest),
        family=TREE,
        fit="per-bar",
        panel="A",
        inputs=SET_NAME[b],
    )


# --------------------------------------------------------------------------- formatting


# 2026-10-15 (table legibility): parked -- bold significance and p columns; replaced by the star cells below
# def num(x: float, nd: int, bold: bool = False, sign: bool = True) -> str:
#     s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
#     if sign and float(s) == 0.0:
#         s = f"{0.0:.{nd}f}"
#     return f"$\\bm{{{s}}}$" if bold else f"${s}$"
#
#
# def pval(p: float, bold: bool | None = None) -> str:
#     bold = p < SIG if bold is None else bold
#     s = "{<}0.001" if p < 0.001 else f"{p:.3f}"  # {<}: no relation spacing
#     if s == "0.050":  # keep the 5 % cut visible: 0.0497 is bold, 0.0503 is not
#         s = f"{p:.4f}"
#     return f"$\\bm{{{s}}}$" if bold else f"${s}$"

# 2026-10-15 (table legibility): finance-style cells, as make_rehaul_economics.py (cell, panel_head): stars from
# the two-sided normal p of the test, on the estimate; the statistic in parentheses in its own column (stat_col),
# one row per forecast, so the tables stay within a page; in a starred column
# every cell carries a three-star superscript, the unused stars as a phantom, so plain r columns align the decimals.


def n_stars(p: float) -> int:
    """1, 2 or 3 stars when the two-sided normal p is below 10, 5 or 1 %."""
    return int(sum(p < a for a in LEVELS)) if np.isfinite(p) else 0


def _fmt(x: float, dp: int) -> str:
    v = round(float(x), dp)
    if v == 0:
        v = 0.0  # no negative zero
    return f"{'-' if v < 0 else ''}{abs(v):.{dp}f}"


def cell(x: float | None, dp: int = 2, stars: int = 0, starred: bool = True) -> str:
    """An estimate in math mode, padded to three star slots when the column is starred."""
    if x is None or not np.isfinite(x):
        return "--$^{\\phantom{***}}$" if starred else "--"
    body = _fmt(x, dp)
    if not starred:
        return f"${body}$"
    pad = "*" * (3 - stars)
    sup = "*" * stars + (f"\\phantom{{{pad}}}" if pad else "")
    return f"${body}^{{{sup}}}$"


def stat_col(t: float | None, dp: int = 2) -> str:
    """A test statistic in parentheses in its own column (no star slot)."""
    if t is None or not np.isfinite(t):
        return "--"
    return f"$({_fmt(t, dp)})$"


def panel_head(title: str, ncol: int, first: bool = False) -> str:
    return (
        ("" if first else "\\addlinespace[4pt]\n")
        + f"\\multicolumn{{{ncol}}}{{@{{}}l}}{{\\textit{{{title}}}}} \\\\\n"
        + "\\addlinespace[2pt]\n"
    )


# macro values for the prose (rehaul_results_numbers.tex): signed, with a text-safe minus and no plus sign
def mnum(x: float, dp: int) -> str:
    v = round(float(x), dp)
    if v == 0:
        v = 0.0
    return ("\\ensuremath{-}" if v < 0 else "") + f"{abs(v):.{dp}f}"


def mpval(p: float) -> str:
    """The whole p statement, '$p = 0.18$' or '$p < 0.001$': 2 decimals from 0.10, else 3; 4 when a cut would blur."""
    if p < 0.001:
        return "\\ensuremath{p < 0.001}"
    s = f"{p:.2f}" if p >= 0.10 else f"{p:.3f}"
    if s in ("0.10", "0.100", "0.050", "0.010"):  # keep the 10 / 5 / 1 % cuts visible
        s = f"{p:.4f}"
    return f"\\ensuremath{{p = {s}}}"


def mstars(p: float) -> str:
    k = n_stars(p)
    return f"\\ensuremath{{^{{{'*' * k}}}}}" if k else ""


def q4(x: float) -> str:
    return f"{x:.4f}"


def tex_escape(s: str) -> str:
    return s.replace("&", "\\&").replace("%", "\\%")


# --------------------------------------------------------------------------- tables

# 2026-10-15 (table legibility): parked -- the old tables (method-text captions, p columns, bold significance, code terms, notes of 10+ sentences); replaced by the finance-style tables below
# NOTE_CONV = (
#     "QLIKE is the mean over the {days} trade days (2020-01-03 to 2024-04-30, early closes excluded) of the daily loss "
#     "$L_t = RV_t/\\hat\\sigma^2_t - \\log(RV_t/\\hat\\sigma^2_t) - 1$, with the plain forecast "
#     # "$\\hat\\sigma^2_t = \\hat y_t^2 B_t$ (no recalibration) and $RV_t$ the winsorised 15:30--16:00 realised variance."  # 2026-10-15 (user): parked -- scored against the trimmed target; scores now use actual RV (data audit #10)
#     "$\\hat\\sigma^2_t = \\hat y_t^2 B_t$ (no recalibration) and $RV_t$ the actual (not winsorised) 15:30--16:00 "
#     "realised variance; the models are fitted on the winsorised target."
# )
# NOTE_DM = (
#     "DM is the Diebold--Mariano statistic of the daily QLIKE difference {what}, with a Bartlett Newey--West variance at "
#     "lag 6 and the Harvey--Leybourne--Newbold correction; $p$ is two-sided (normal). "
# )
#
#
# def table_t1(rows: list[dict]) -> str:
#     out = [
#         "% AUTO-GENERATED by writeup/make_rehaul_results.py -- do not edit.",
#         "\\begin{table}[htbp]",
#         "\\centering",
#         "\\small",
#         "\\caption{Axis 1: HAR + calendar against all features, within each family. QLIKE of the two input sets, the "
#         "change (all features minus HAR + calendar), its Diebold--Mariano test and the change in the trade's Sharpe ratio.}",
#         "\\label{tab:rh_axis1}",
#         "\\begin{threeparttable}",
#         "\\begin{tabular}{@{}l rr rrr rr@{}}",
#         "\\toprule",
#         " & \\multicolumn{2}{c}{QLIKE} & \\multicolumn{3}{c}{All vs HAR + cal.} & \\multicolumn{2}{c}{Sharpe change} \\\\",
#         "\\cmidrule(lr){2-3}\\cmidrule(lr){4-6}\\cmidrule(l){7-8}",
#         "Model & HAR + cal. & All & $\\Delta$ (\\%) & DM & $p$ & $\\Delta$SR & $t$ \\\\",
#         "\\midrule",
#     ]
#     for fit, title in (
#         ("per-bar", "Per-bar (one model per bar)"),
#         ("pooled", "Pooled (one model for all 48 bars)"),
#     ):
#         out.append(f"\\multicolumn{{8}}{{@{{}}l}}{{\\textit{{{title}}}}} \\\\")
#         for r in [r for r in rows if r["fit"] == fit]:
#             name = r["name"] + (r.get("mark") or "")
#             if r.get("pending"):
#                 out.append(
#                     f"\\quad {name} & \\multicolumn{{7}}{{l}}{{\\textit{{pending (walk running)}}}} \\\\"
#                 )
#                 continue
#             sig, sig_t = r["p"] < SIG, r["p_t"] < SIG
#             out.append(
#                 f"\\quad {name} & {q4(r['qb'])} & {q4(r['qa'])} & {num(r['pct'], 1)} & {num(r['dm'], 2, sig)} & "
#                 f"{pval(r['p'])} & {num(r['dsh'], 2)} & {num(r['t'], 2, sig_t)} \\\\"
#             )
#         if fit == "per-bar":
#             out.append("\\addlinespace")
#     out += [
#         "\\bottomrule",
#         "\\end{tabular}",
#         "\\begin{tablenotes}[flushleft]\\footnotesize",
#         "\\item \\textit{Notes.} "
#         + NOTE_CONV.format(days="\\rhrDays{}")
#         + " $\\Delta = 100(\\overline{L}^{\\,\\text{all}}/\\overline{L}^{\\,\\text{HAR+cal.}} - 1)$. "
#         + NOTE_DM.format(what="(all features minus HAR + calendar, same model)")
#         + "Negative $\\Delta$ and DM favour all features. $\\Delta$SR is the change in the annualised Sharpe ratio of "
#         "the midpoint straddle trade ($\\sign$ of forecast minus implied variance); $t$ is the Newey--West $t$ (lag 14) "
#         "of the daily return difference. Bold: $p < 0.05$ (DM, or the $t$ against the normal). "
#         "Per-bar linear models are pre-tuned; the per-bar trees use fixed settings refit every session; the MinGRU "
#         "is the ridge-initialised network trained and selected on QLIKE (seed mean: 5 per-bar, \\rhrNNPooledSeeds{} "
#         "pooled). The pooled MinGRU is refit monthly, each refit on its trailing 24,000 bars; the pooled trees are "
#         "refit at every bar. The pooled LightGBM and XGBoost on all features are the earlier paper's pooled trees. "
#         "$^\\dagger$Interim: the pooled forest's 16:00 forecasts on all features; its full every-bar walk is still "
#         "running. Across all \\rhrAfPairs{} input-set pairs of the model set, all features lower QLIKE in "
#         "\\rhrAfLower{} (\\rhrAfLowerSig{} with $p < 0.05$) and raise variance-space MSE in \\rhrAfMseHigher{} "
#         "(\\rhrAfMseHigherSig{} with $p < 0.05$).",
#         "\\end{tablenotes}",
#         "\\end{threeparttable}",
#         "\\end{table}",
#         "",
#     ]
#     return "\n".join(out)
#
#
# def table_t2(rows: list[dict]) -> str:
#     out = [
#         "% AUTO-GENERATED by writeup/make_rehaul_results.py -- do not edit.",
#         "\\begin{table}[htbp]",
#         "\\centering",
#         "\\small",
#         "\\caption{Axis 2: pooled against per-bar, by family and input set. QLIKE change (pooled minus per-bar), its "
#         "Diebold--Mariano test and the $t$ of the trade's return difference.}",
#         "\\label{tab:rh_axis2}",
#         "\\begin{threeparttable}",
#         "\\begin{tabular}{@{}l rrrr rrrr@{}}",
#         "\\toprule",
#         " & \\multicolumn{4}{c}{HAR + calendar} & \\multicolumn{4}{c}{All features} \\\\",
#         "\\cmidrule(lr){2-5}\\cmidrule(l){6-9}",
#         "Model & $\\Delta$ (\\%) & DM & $p$ & $t$ & $\\Delta$ (\\%) & DM & $p$ & $t$ \\\\",
#         "\\midrule",
#     ]
#     groups = ((LIN, "Linear"), (TREE, "Trees"), (NN, "Neural network"))
#     for fam, title in groups:
#         out.append(f"\\multicolumn{{9}}{{@{{}}l}}{{\\textit{{{title}}}}} \\\\")
#         for r in [r for r in rows if r["family"] == fam]:
#             cells = []
#             for b in ("baseline", "all_features"):
#                 x = r[b]
#                 cells += [
#                     num(x["pct"], 1),
#                     num(x["dm"], 2, x["p"] < SIG),
#                     pval(x["p"]),
#                     num(x["t"], 2, x["p_t"] < SIG),
#                 ]
#             out.append(
#                 f"\\quad {r['name']}{r.get('mark', '')} & "
#                 + " & ".join(cells)
#                 + " \\\\"
#             )
#         if fam != NN:
#             out.append("\\addlinespace")
#     out += [
#         "\\bottomrule",
#         "\\end{tabular}",
#         "\\begin{tablenotes}[flushleft]\\footnotesize",
#         "\\item \\textit{Notes.} "
#         + NOTE_CONV.format(days="\\rhrDays{}")
#         + " $\\Delta = 100(\\overline{L}^{\\,\\text{pooled}}/\\overline{L}^{\\,\\text{per-bar}} - 1)$, the same estimator "
#         "and input set. "
#         + NOTE_DM.format(what="(pooled minus per-bar)")
#         + "Positive $\\Delta$ and DM favour the per-bar fit. $t$ is the Newey--West $t$ (lag 14) of the daily "
#         "difference of the midpoint straddle returns (pooled minus per-bar). Bold: $p < 0.05$. "
#         "Per-bar trees: fixed settings refit every session; pooled trees: the bank's walk refit at every bar "
#         "(24,000-bar window); linear models are pre-tuned in both fits. The pooled MinGRU is refit monthly (each refit "
#         "on its trailing 24,000 bars), not at every bar like the pooled trees; both MinGRU fits are ridge-initialised "
#         "and trained and selected on QLIKE. "
#         "$^\\dagger$The pooled forest on all features is interim (16:00 forecasts; the full walk is still running).",
#         "\\end{tablenotes}",
#         "\\end{threeparttable}",
#         "\\end{table}",
#         "",
#     ]
#     return "\n".join(out)
#
#
# def table_t3(rank: pd.DataFrame, keys: list[str]) -> str:
#     out = [
#         "% AUTO-GENERATED by writeup/make_rehaul_results.py -- do not edit.",
#         "\\begin{table}[htbp]",
#         "\\centering",
#         "\\small",
#         "\\setlength{\\tabcolsep}{5pt}",
#         "\\caption{The headline forecasts ranked by QLIKE, against the per-bar OLS on HAR + calendar. Rank is the "
#         "position among all \\rhrRankN{} forecasts of Table~\\ref{tab:rh_rank_full}.}",
#         "\\label{tab:rh_rank}",
#         "\\begin{threeparttable}",
#         "\\begin{tabular}{@{}r l l l r rrr@{}}",
#         "\\toprule",
#         " & & & & & \\multicolumn{3}{c}{vs per-bar OLS} \\\\",
#         "\\cmidrule(l){6-8}",
#         "Rank & Model & Fit & Inputs & QLIKE & $\\Delta$ (\\%) & DM & $p$ \\\\",
#         "\\midrule",
#     ]
#     sub = rank.loc[keys].sort_values("qlike")
#     for k, r in sub.iterrows():
#         name = tex_escape(r["name"]) + ("$^\\dagger$" if k in INTERIM else "")
#         if k == OLS:
#             out.append(
#                 f"{r['rank']} & \\textbf{{{name}}} (reference) & {r['fit']} & {r['inputs']} & \\textbf{{{q4(r['qlike'])}}} & & & \\\\"
#             )
#             continue
#         out.append(
#             f"{r['rank']} & {name} & {r['fit']} & {r['inputs']} & {q4(r['qlike'])} & {num(r['pct'], 1)} & "
#             f"{num(r['dm'], 2, r['p'] < SIG)} & {pval(r['p'])} \\\\"
#         )
#     out += [
#         "\\bottomrule",
#         "\\end{tabular}",
#         "\\begin{tablenotes}[flushleft]\\footnotesize",
#         "\\item \\textit{Notes.} "
#         + NOTE_CONV.format(days="\\rhrDays{}")
#         + " $\\Delta = 100(\\overline{L}/\\overline{L}^{\\,\\text{OLS}} - 1)$. "
#         + NOTE_DM.format(what="(model minus the per-bar OLS)")
#         + "Negative $\\Delta$ and DM favour the model. Bold: $p < 0.05$. Rows: the per-bar OLS, the canonical rung of each "
#         "family on each input set (per-bar trees with fixed settings refit every session) and the earlier paper's pooled "
#         "OLS and block-diagonal ridge. The per-bar ridge, lasso and elastic net on HAR + calendar are within "
#         "\\rhrLinBaseMaxPct\\,\\% of the OLS and are not shown. $^\\dagger$Interim (see Table~\\ref{tab:rh_axis1}). "
#         "The pooled MinGRU is refit monthly, the pooled trees at every bar. "
#         "Of the \\rhrNSet{} model-set forecasts other than the OLS, \\rhrNLower{} have lower QLIKE (\\rhrNLowerSig{} "
#         "with $p < 0.05$) and \\rhrNWorseSig{} are significantly worse.",
#         "\\end{tablenotes}",
#         "\\end{threeparttable}",
#         "\\end{table}",
#         "",
#     ]
#     return "\n".join(out)
#
#
# def table_t3_full(rank: pd.DataFrame) -> str:
#     head = "Rank & Model & Fit & Inputs & QLIKE & $\\Delta$ (\\%) & DM & $p$ \\\\"
#     out = [
#         "% AUTO-GENERATED by writeup/make_rehaul_results.py -- do not edit.",
#         "{\\footnotesize",
#         "\\setlength{\\tabcolsep}{3.5pt}",
#         "\\begin{longtable}{@{}r l l l r rrr@{}}",
#         "\\caption{Every forecast ranked by QLIKE, against the per-bar OLS on HAR + calendar (\\rhrRankN{} forecasts: "
#         "the \\rhrNModelSet{}-forecast model set, the per-bar and pooled MinGRU and the pooled trees outside the master "
#         "table).}",
#         "\\label{tab:rh_rank_full}\\\\",
#         "\\toprule",
#         head,
#         "\\midrule",
#         "\\endfirsthead",
#         "\\multicolumn{8}{@{}l}{\\textit{Table~\\ref{tab:rh_rank_full}, continued}} \\\\",
#         "\\toprule",
#         head,
#         "\\midrule",
#         "\\endhead",
#         "\\midrule",
#         "\\multicolumn{8}{r@{}}{\\textit{continued on the next page}} \\\\",
#         "\\endfoot",
#         "\\bottomrule",
#         "\\multicolumn{8}{@{}p{\\linewidth}@{}}{\\footnotesize\\raggedright\\textit{Notes.} Conventions as in "
#         "Table~\\ref{tab:rh_rank}: plain $\\hat y_t^2 B_t$ forecasts on \\rhrDays{} days; DM against the per-bar OLS "
#         "(lag 6, HLN), negative favours the model; bold $p < 0.05$. $^\\ast$Not in the master table's model set "
#         "(results/close\\_master\\_table/paper\\_model\\_set.csv). $^\\dagger$Interim. Per-bar tree rungs: fixed settings "
#         "refit every 10 sessions or every session (daily refit); random search on squared error (SE) or QLIKE, re-run "
#         "every 250 sessions; Optuna on SE or QLIKE re-run every 1, 5, 25 or 250 sessions (tune), keeping the best of the "
#         "first 10, 25 or 50 trials (top). Earlier panel: the pooled 48-bar models of the earlier draft. The pooled MinGRU "
#         "is refit monthly, the pooled trees at every bar.} \\tabularnewline",
#         "\\endlastfoot",
#     ]
#     for k, r in rank.sort_values("qlike").iterrows():
#         name = (
#             tex_escape(r["name"])
#             + ("$^\\dagger$" if k in INTERIM else "")
#             + ("$^\\ast$" if not r["in_set"] else "")
#         )
#         if k == OLS:
#             out.append(
#                 f"{r['rank']} & \\textbf{{{name}}} (reference) & {r['fit']} & {r['inputs']} & \\textbf{{{q4(r['qlike'])}}} & & & \\\\"
#             )
#             continue
#         out.append(
#             f"{r['rank']} & {name} & {r['fit']} & {r['inputs']} & {q4(r['qlike'])} & {num(r['pct'], 1)} & "
#             f"{num(r['dm'], 2, r['p'] < SIG)} & {pval(r['p'])} \\\\"
#         )
#     out += ["\\end{longtable}", "}", ""]
#     return "\n".join(out)


# ----------------------------------------------------------------------------- 2026-10-15 (table legibility)
# The four tables in the finance-journal style of NOTATION_GUIDE_2026-10-15.md section 8 (see the module docstring).
HEAD = "% AUTO-GENERATED by writeup/make_rehaul_results.py -- do not edit."
STAR_NOTE = "$^{*}$, $^{**}$ and $^{***}$ denote significance at the 10\\%, 5\\% and 1\\% levels."
# NOTE_SAMPLE = (  # 2026-10-15 (user): parked -- accuracy over every rolling forecast, trades on the option days
#     "The out-of-sample period runs from \\rhrSampleStart{} to \\rhrSampleEnd{} ($T_{\\mathrm{oos}} = \\rhrDays$ "
#     "trading days). QLIKE is the mean loss ...
NOTE_SAMPLE = (
    "Rolling forecasts from \\rhoosSpanStart{} to \\rhoosSpanEnd{} ($T_{\\mathrm{oos}} = \\rhoosDays$ trading "
    "days, each forecast from a model estimated on its trailing window; Section~\\ref{sec:rh_results_eval}); Sharpe "
    "ratios and HAC $t$-statistics on the \\rhrDays{} days of the option-data subsample. QLIKE is the mean loss $\\bar L$, with $L_t = \\QLIKE(\\RV_t, \\hat\\sigma^2_t)$, $\\RV_t$ the "
    "actual realized variance of the closing half hour and $\\hat\\sigma^2_t = \\hat y_t^{\\,2} s^2_{t,c}$ the "
    "variance forecast."
)
NOTE_TREES = (
    "Closing-interval trees use fixed hyperparameters and are re-estimated daily"
)
ITEM_EARLIER_FULL = (
    "\\item[a] Full predictor set: the earlier draft's intraday specification."
)
ITEM_PRELIM = (
    "\\item[$\\dagger$] Preliminary: the closing-interval forecasts of the intraday random forest on the full "
    "predictor set, from an exercise still in progress."
)
ITEM_PRELIM_SHORT = "\\item[$\\dagger$] Preliminary (see Table~\\ref{tab:rh_axis1})."
# T1 / T2 rows whose pooled intraday forecast on the full predictor set is the earlier draft's (the master table's
# lgbm / xgb rows; main() asserts the aliasing)
EARLIER_POOLED_FULL = ("LightGBM", "XGBoost")


def table_t1(rows: list[dict]) -> str:
    """T1: the full predictor set against the HAR benchmark, same model, one row per model: QLIKE of both, the % change
    (stars) and its DM, the change in the sign strategy's Sharpe ratio (stars) and its HAC t. (Under-the-estimate
    statistics made the float taller than the page, 2026-10-15; the statistics get their own columns instead.)"""
    body = ""
    for fit, title in (
        ("per-bar", "Panel A: Closing-interval models"),
        ("pooled", "Panel B: Intraday models"),
    ):
        body += panel_head(title, 7, first=fit == "per-bar")
        for r in [r for r in rows if r["fit"] == fit]:
            assert not r.get("pending"), r["name"]
            name = r["name"]
            if fit == "pooled" and name in EARLIER_POOLED_FULL:
                name += "\\tnote{a}"
            if r.get("mark"):
                name += "\\tnote{$\\dagger$}"
            cells = [
                q4(r["qb"]),
                q4(r["qa"]),
                cell(r["pct"], 1, n_stars(r["p"])),
                stat_col(r["dm"]),
                cell(r["dsh"], 2, n_stars(r["p_t"])),
                stat_col(r["t"]),
            ]
            body += f"{name} & " + " & ".join(cells) + " \\\\\n"
    notes = (
        "\\item \\textit{Notes:} "
        + NOTE_SAMPLE
        + " $\\Delta = 100(\\bar L^{\\mathrm{Full}}/\\bar L^{\\mathrm{HAR}} - 1)$, negative when the full predictor "
        "set is more accurate, and $\\DM$ is the Diebold--Mariano statistic of the daily loss differential "
        "(Harvey--Leybourne--Newbold correction; Newey--West lag $q_{\\mathrm{NW}} = \\rhrDMLag$). $\\Delta\\SR$ is the "
        "change in the annualized Sharpe ratio of the sign strategy at mid-quotes and $t$ the HAC $t$-statistic of the "
        "daily return differential ($q_{\\mathrm{NW}} = \\rhrHACLag$; Section~\\ref{sec:rh_methods_tests}). "
        + STAR_NOTE
        + " "
        + NOTE_TREES
        # 2026-10-15 (user): parked -- NN content parked for now
        # + ", and the MinGRU is ridge-initialized and trained and selected on QLIKE "
        # "(Section~\\ref{sec:rh_methods_models})."
        # 2026-10-15 (finish): parked -- the re-estimation schedule moved to Appendix B.2 (sec:rh_exp_schemes). Was:
        # + " (Section~\\ref{sec:rh_methods_models})."
        + " (Section~\\ref{sec:rh_exp_schemes})."
    )
    return "\n".join(
        [
            HEAD,
            "% tab:rh_axis1 (T1). The intraday LightGBM and XGBoost on the full predictor set are the master",
            "% table's lgbm and xgb rows, the earlier draft's pooled trees; their HAR-benchmark twins come from",
            "% results/pooled_trees_2026-10-07.",
            "\\begin{table}[htbp]",
            "\\centering",
            "\\small",
            "\\begin{threeparttable}",
            "\\caption{Forecast accuracy: HAR benchmark versus full predictor set}",
            "\\label{tab:rh_axis1}",
            "\\begin{tabular}{@{}lrrrrrr@{}}",
            "\\toprule",
            " & \\multicolumn{4}{c}{QLIKE} & \\multicolumn{2}{c}{Sharpe ratio} \\\\",
            "\\cmidrule(lr){2-5}\\cmidrule(l){6-7}",
            "Model & HAR & Full & $\\Delta$ (\\%) & $\\DM$ & $\\Delta\\SR$ & $t$ \\\\",
            "\\midrule",
            body + "\\bottomrule",
            "\\end{tabular}",
            "\\begin{tablenotes}[flushleft]",
            "\\footnotesize",
            notes,
            ITEM_EARLIER_FULL,
            ITEM_PRELIM,
            "\\end{tablenotes}",
            "\\end{threeparttable}",
            "\\end{table}",
            "",
        ]
    )


def table_t2(rows: list[dict]) -> str:
    """T2: the intraday model against the closing-interval model of the same class; Panel A on the HAR
    benchmark, Panel B on the full predictor set; one row per model class: the % change in QLIKE (stars) and its DM,
    the change in the sign strategy's Sharpe ratio (stars) and its HAC t."""
    body = ""
    for b, title in (
        ("baseline", "Panel A: HAR benchmark"),
        ("all_features", "Panel B: Full predictor set"),
    ):
        body += panel_head(title, 5, first=b == "baseline")
        for r in rows:
            assert not r.get("pending"), r["name"]
            name = r["name"]
            if b == "all_features" and name in EARLIER_POOLED_FULL:
                name += "\\tnote{a}"
            if b == "all_features" and r.get("mark"):
                name += "\\tnote{$\\dagger$}"
            x = r[b]
            cells = [
                cell(x["pct"], 1, n_stars(x["p"])),
                stat_col(x["dm"]),
                cell(x["dsh"], 2, n_stars(x["p_t"])),
                stat_col(x["t"]),
            ]
            body += f"{name} & " + " & ".join(cells) + " \\\\\n"
    notes = (
        "\\item \\textit{Notes:} Sample, loss and tests as in Table~\\ref{tab:rh_axis1}. Each row compares the pooled "
        "intraday model with the closing-interval model of the same class on the predictor set of the panel: "
        "$\\Delta = 100(\\bar L^{\\mathrm{pool}}/\\bar L^{\\mathrm{close}} - 1)$, positive when the closing-interval "
        "model is more accurate, and $\\Delta\\SR$ is the intraday minus the closing-interval Sharpe ratio. "
        + STAR_NOTE
        + " "
        + NOTE_TREES
        # 2026-10-15 (user): parked -- NN content parked for now
        # + "; the pooled intraday trees are re-estimated at every interval and the pooled intraday MinGRU monthly, "
        # "each on a rolling window of past intervals (Section~\\ref{sec:rh_methods_setup})."
        # 2026-10-15 (finish): parked -- the pooled trees' schedule moved to Appendix B.2 (sec:rh_exp_schemes). Was:
        # + "; the pooled intraday trees are re-estimated at every interval, on a rolling window of past intervals "
        # "(Section~\\ref{sec:rh_methods_setup})."
        + "; the intraday trees are re-estimated at every interval, on a rolling window of past intervals "
        "(Section~\\ref{sec:rh_exp_schemes})."
    )
    return "\n".join(
        [
            HEAD,
            "% tab:rh_axis2 (T2): intraday minus closing-interval, same model class and predictor set.",
            "\\begin{table}[htbp]",
            "\\centering",
            "\\small",
            "\\begin{threeparttable}",
            "\\caption{Forecast accuracy: closing-interval versus intraday models}",
            "\\label{tab:rh_axis2}",
            "\\begin{tabular}{@{}lrrrr@{}}",
            "\\toprule",
            " & \\multicolumn{2}{c}{QLIKE} & \\multicolumn{2}{c}{Sharpe ratio} \\\\",
            "\\cmidrule(lr){2-3}\\cmidrule(l){4-5}",
            "Model & $\\Delta$ (\\%) & $\\DM$ & $\\Delta\\SR$ & $t$ \\\\",
            "\\midrule",
            body + "\\bottomrule",
            "\\end{tabular}",
            "\\begin{tablenotes}[flushleft]",
            "\\footnotesize",
            notes,
            "\\item[a] The earlier draft's intraday specification.",
            ITEM_PRELIM_SHORT,
            "\\end{tablenotes}",
            "\\end{threeparttable}",
            "\\end{table}",
            "",
        ]
    )


def rank_name(k: str, r: pd.Series, tnotes: bool) -> str:
    """Model name of a ranking row, with its markers: a = the earlier draft's specification (T3 only; T3 full has a
    Specification column and Panel C), dagger = preliminary, double dagger = outside the model set (T3 full only)."""
    name = r["model"]
    if tnotes:
        if r["panel"] == "C" and r["spec"] != "--":
            name += f" ({r['spec'][0].lower()}{r['spec'][1:]})"
        if r["panel"] == "C" or k in EARLIER_TREES:
            name += "\\tnote{a}"
        if k in INTERIM:
            name += "\\tnote{$\\dagger$}"
        return name
    sup = ("\\dagger" if k in INTERIM else "") + ("" if r["in_set"] else "\\ddagger")
    return name + (f"$^{{{sup}}}$" if sup else "")


def table_t3(rank: pd.DataFrame, keys: list[str]) -> str:
    """T3: the main forecasts ranked by mean QLIKE, against the HAR benchmark (the closing-interval OLS on HAR)."""
    sub = rank.loc[keys]
    body = ""
    for fit, title in (
        ("per-bar", "Panel A: Closing-interval models"),
        ("pooled", "Panel B: Intraday models"),
    ):
        body += panel_head(title, 6, first=fit == "per-bar")
        for k, r in sub[sub["fit"] == fit].sort_values("qlike").iterrows():
            if k == OLS:
                body += f"{r['rank']} & OLS (benchmark) & HAR & {q4(r['qlike'])} & {cell(None, 1)} & -- \\\\\n"
                continue
            body += (
                f"{r['rank']} & {rank_name(k, r, True)} & {r['inputs']} & {q4(r['qlike'])} & "
                f"{cell(r['pct'], 1, n_stars(r['p']))} & {stat_col(r['dm'])} \\\\\n"
            )
    notes = (
        "\\item \\textit{Notes:} Sample, loss and tests as in Table~\\ref{tab:rh_axis1}. "
        "$\\Delta = 100(\\bar L/\\bar L^{\\mathrm{OLS}} - 1)$ compares each forecast with the HAR benchmark, the "
        "closing-interval OLS on the HAR predictor set, with $\\DM$ in parentheses; negative values favor the forecast. "
        "Rank is the position among the \\rhrRankN{} forecasts of Table~\\ref{tab:rh_rank_full}; the closing-interval "
        "ridge, lasso and elastic net on the HAR predictor set are within \\rhrLinBaseMaxPct\\% of the benchmark and are "
        "omitted. " + NOTE_TREES + ". " + STAR_NOTE
    )
    return "\n".join(
        [
            HEAD,
            "% tab:rh_rank (T3): Rank is the global rank of tab:rh_rank_full; the closing-interval rows rank above the",
            "% intraday rows, so the panels keep the sort order.",
            "\\begin{table}[htbp]",
            "\\centering",
            "\\small",
            "\\begin{threeparttable}",
            "\\caption{Ranking of the main forecasts by mean QLIKE}",
            "\\label{tab:rh_rank}",
            "\\begin{tabular}{@{}rllrrr@{}}",
            "\\toprule",
            " & & & & \\multicolumn{2}{c}{vs.\\ HAR benchmark} \\\\",
            "\\cmidrule(l){5-6}",
            "Rank & Model & Predictors & QLIKE & $\\Delta$ (\\%) & $\\DM$ \\\\",
            "\\midrule",
            body + "\\bottomrule",
            "\\end{tabular}",
            "\\begin{tablenotes}[flushleft]",
            "\\footnotesize",
            notes,
            "\\item[a] The earlier draft's intraday specification.",
            ITEM_PRELIM_SHORT,
            "\\end{tablenotes}",
            "\\end{threeparttable}",
            "\\end{table}",
            "",
        ]
    )


def table_t3_full(rank: pd.DataFrame) -> str:
    """T3 (full): every forecast, a longtable for the appendix; Panel A closing-interval, B intraday, C the
    earlier draft's intraday linear models, each sorted by QLIKE; Rank is the global rank."""
    head = [
        " & & & & & \\multicolumn{2}{c}{vs.\\ HAR benchmark} \\\\",
        "\\cmidrule(l){6-7}",
        "Rank & Model & Specification & Predictors & QLIKE & $\\Delta$ (\\%) & $\\DM$ \\\\",
    ]
    notes = (
        "\\textit{Notes:} Sample, loss, tests and significance markers as in Table~\\ref{tab:rh_axis1}; $\\Delta$ and "
        "$\\DM$ compare each forecast with the HAR benchmark, as in Table~\\ref{tab:rh_rank}. Specification of the "
        "closing-interval trees (Panel A): fixed hyperparameters, random search or Optuna, selecting on MSE or QLIKE, "
        "then the re-estimation frequency or, for Optuna, the re-tuning frequency and the number of trials searched; "
        "d: trading days (Section~\\ref{sec:rh_methods_tuning}). Panel C holds the earlier draft's intraday "
        "linear models, with a block, fixed or re-tuned penalty; ``no FOMC'' marks the earlier draft's predictor set, "
        # 2026-10-15 (user): parked -- NN content parked for now
        # "which lacks the FOMC predictors. The pooled intraday trees are re-estimated at every interval and the pooled "
        # "intraday MinGRU monthly (Section~\\ref{sec:rh_methods_setup}). $^{\\ddagger}$Not in the model set counted in "
        # 2026-10-15 (finish): parked -- the pooled trees' schedule moved to Appendix B.2 (sec:rh_exp_schemes). Was:
        # "(Section~\\ref{sec:rh_methods_setup}). $^{\\ddagger}$Not in the model set counted in "
        "which lacks the FOMC predictors. The intraday trees are re-estimated at every interval "
        "(Section~\\ref{sec:rh_exp_schemes}). $^{\\ddagger}$Not in the model set counted in "
        "the text; $^{\\dagger}$preliminary (see Table~\\ref{tab:rh_axis1})."
    )
    out = [
        HEAD,
        "% tab:rh_rank_full (T3, every forecast): a longtable, for the appendix (10_appendix_floats.tex).",
        "{\\small",
        "\\setlength{\\tabcolsep}{3.5pt}",  # 4pt overflows the text width by 4pt
        "\\begin{longtable}{@{}rlllrrr@{}}",
        "\\caption{Ranking of all forecasts by mean QLIKE}",
        "\\label{tab:rh_rank_full}\\\\",
        "\\toprule",
        *head,
        "\\midrule",
        "\\endfirsthead",
        "\\multicolumn{7}{@{}l}{\\textit{Table~\\ref{tab:rh_rank_full}, continued}} \\\\",
        "\\toprule",
        *head,
        "\\midrule",
        "\\endhead",
        "\\midrule",
        "\\multicolumn{7}{r@{}}{\\textit{Continued on next page}} \\\\",
        "\\endfoot",
        "\\bottomrule",
        f"\\multicolumn{{7}}{{@{{}}p{{\\linewidth}}@{{}}}}{{\\footnotesize\\raggedright {notes}}} \\tabularnewline",
        "\\endlastfoot",
    ]
    for panel, title in (
        ("A", "Panel A: Closing-interval models"),
        ("B", "Panel B: Intraday models"),
        ("C", "Panel C: Intraday models, earlier draft's specification"),
    ):
        out.append(panel_head(title, 7, first=panel == "A").rstrip("\n"))
        for k, r in rank[rank["panel"] == panel].sort_values("qlike").iterrows():
            if k == OLS:
                out.append(
                    f"{r['rank']} & OLS (benchmark) & -- & HAR & {q4(r['qlike'])} & {cell(None, 1)} & -- \\\\"
                )
                continue
            out.append(
                f"{r['rank']} & {rank_name(k, r, False)} & {r['spec']} & {r['inputs']} & {q4(r['qlike'])} & "
                f"{cell(r['pct'], 1, n_stars(r['p']))} & {stat_col(r['dm'])} \\\\"
            )
    out += ["\\end{longtable}", "}", ""]
    return "\n".join(out)


# --------------------------------------------------------------------------- figures

C = {
    LIN: "#2a78d6",
    TREE: "#d55e00",
    NN: "#009e73",
    POOL: "#b9479b",
}  # = make_rehaul_economics.py FAM_COLOR; validate_palette.js all-pairs on white: PASS (worst CVD dE 8.2)
MK = {LIN: "o", TREE: "s", NN: "^", POOL: "D"}  # = make_rehaul_economics.py FAM_MARKER
MS = {LIN: 5.0, TREE: 4.6, NN: 5.6, POOL: 4.8}  # marker sizes that read as equal areas
INK, MUTED, GRID, FRAME, LINK = "#1c1c1c", "#5e5a54", "#e8e8e8", "#bdbdbd", "#b5b5b5"
SHADE = "#efefef"
W = 6.5  # \linewidth of the 11pt article with 1in margins, inches


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.6,
            "axes.edgecolor": FRAME,
            "xtick.color": FRAME,
            "ytick.color": FRAME,
            "xtick.labelcolor": MUTED,
            "ytick.labelcolor": INK,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.size": 2.5,
            "ytick.major.size": 0,
            "xtick.major.width": 0.6,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "lines.solid_capstyle": "round",
        }
    )


def save(fig, stem: str) -> None:
    """Vector PDF at the figure's own size (constrained layout; W inches wide = \\linewidth), plus a PNG preview."""
    fig.savefig(FIG / f"{stem}.pdf")
    fig.savefig(FIG / f"{stem}.png", dpi=200)
    plt.close(fig)
    print(f"wrote figures/{stem}.pdf")


def minus(s: str) -> str:
    return s.replace("-", "−")


def legend_handles(sig_note: bool = True) -> list:
    h = [
        Line2D(
            [],
            [],
            ls="",
            marker=MK[LIN],
            ms=5,
            mfc=C[LIN],
            mec=C[LIN],
            label="per-bar linear",
        ),
        Line2D(
            [],
            [],
            ls="",
            marker=MK[TREE],
            ms=4.5,
            mfc=C[TREE],
            mec=C[TREE],
            label="per-bar tree",
        ),
        # 2026-10-15 (user): parked -- NN content parked for now
        # Line2D(
        #     [],
        #     [],
        #     ls="",
        #     marker=MK[NN],
        #     ms=5.5,
        #     mfc=C[NN],
        #     mec=C[NN],
        #     label="per-bar MinGRU",
        # ),
        Line2D(
            [],
            [],
            ls="",
            marker=MK[POOL],
            ms=4.5,
            mfc=C[POOL],
            mec=C[POOL],
            label="pooled (any family)",
        ),
    ]
    if sig_note:
        h += [
            Line2D(
                [],
                [],
                ls="",
                marker="o",
                ms=5,
                mfc=MUTED,
                mec=MUTED,
                label="filled: $p<0.05$",
            ),
            Line2D(
                [],
                [],
                ls="",
                marker="o",
                ms=5,
                mfc="white",
                mec=MUTED,
                mew=1.2,
                label="open: $p\\geq0.05$",
            ),
        ]
    return h


def fig_f1(rows: list[dict]) -> None:
    """Forest plot of T1: % change all vs HAR + cal., 95 % band from the DM statistic, DM in a right column."""
    fig, ax = plt.subplots(figsize=(W, 3.9), layout="constrained")
    ylabels, y, items = [], 0.0, []
    for fit, title in (("per-bar", "Per-bar"), ("pooled", "Pooled")):
        ylabels.append((y, title, True))
        y -= 1
        for r in [r for r in rows if r["fit"] == fit]:
            items.append((y, r))
            ylabels.append(
                (y, r["name"] + ("$^\\dagger$" if r.get("mark") else ""), False)
            )
            y -= 1
        y -= 0.5
    ax.axvline(0, color=INK, lw=0.7, zorder=1)
    for yy, r in items:
        fam = POOL if r["fit"] == "pooled" else r["family"]
        col = C[fam]
        if r.get("pending"):
            ax.text(
                -0.6,
                yy,
                "pending (walk running)",
                va="center",
                ha="right",
                color=MUTED,
                fontsize=8,
                style="italic",
            )
            continue
        sig = r["p"] < SIG
        ax.plot(
            [r["pct"] - r["hw"], r["pct"] + r["hw"]],
            [yy, yy],
            color=col,
            lw=1.4,
            zorder=2,
        )
        ax.plot(
            r["pct"],
            yy,
            ls="",
            marker=MK[fam],
            ms=MS[fam],
            mfc=col if sig else "white",
            mec=col,
            mew=1.2,
            zorder=3,
        )
    tr = blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(
        1.02,
        ylabels[0][0],
        "DM",
        transform=tr,
        ha="left",
        va="center",
        fontsize=8,
        color=INK,
        fontweight="bold",
    )
    for yy, r in items:
        if not r.get("pending"):
            ax.text(
                1.02,
                yy,
                minus(f"{r['dm']:+.2f}"),
                transform=tr,
                ha="left",
                va="center",
                fontsize=8,
                color=INK,
                fontweight="bold" if r["p"] < SIG else "normal",
            )
    ax.set_yticks([t[0] for t in ylabels])
    ax.set_yticklabels([t[1] for t in ylabels])
    for lab, t in zip(ax.get_yticklabels(), ylabels):
        if t[2]:
            lab.set_fontweight("bold")
    ax.set_ylim(y + 0.6, 0.6)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(-22, 12)
    ax.set_xlabel("Change in QLIKE, all features vs HAR + calendar (%), with 95% band")
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: minus(f"{v:+.0f}") if v else "0")
    )
    top = blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(
        -0.6,
        1.0,
        "← all features better",
        transform=top,
        ha="right",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )
    ax.text(
        0.6,
        1.0,
        "HAR + calendar better →",
        transform=top,
        ha="left",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )
    fig.legend(
        handles=legend_handles(),
        loc="outside lower center",
        ncol=3,
        handletextpad=0.3,
        columnspacing=1.4,
    )
    save(fig, "rehaul_results_f1_forest")


def fig_f2(rows: list[dict], levels: dict[str, dict]) -> None:
    """Dumbbell: per-bar and pooled QLIKE of each family, one panel per input set."""
    fig, axes = plt.subplots(
        1, 2, figsize=(W, 2.95), sharey=True, sharex=True, layout="constrained"
    )
    names = [r["name"] for r in rows]
    ys = -np.arange(len(rows), dtype=float)
    ols_q = levels[OLS]["qlike"]
    for ax, (bi, title) in zip(axes, ((0, "HAR + calendar"), (1, "All features"))):
        ax.axvline(ols_q, color=INK, lw=0.7, zorder=1)
        for yy, r in zip(ys, rows):
            fam = r["family"]
            pb = levels[r["perbar"][bi]]["qlike"]
            if r["pooled"] is None:
                ax.plot(
                    pb,
                    yy,
                    ls="",
                    marker=MK[fam],
                    ms=MS[fam],
                    mfc=C[fam],
                    mec=C[fam],
                    zorder=3,
                )
                ax.text(
                    pb + 0.0012,
                    yy,
                    "pooled: pending",
                    va="center",
                    ha="left",
                    fontsize=7.5,
                    color=MUTED,
                    style="italic",
                )
                continue
            pk = r["pooled"][bi]
            pq = levels[pk]["qlike"]
            sig = r[("baseline", "all_features")[bi]]["p"] < SIG
            ax.plot([pb, pq], [yy, yy], color=LINK, lw=1.6, zorder=2)
            ax.plot(
                pb,
                yy,
                ls="",
                marker=MK[fam],
                ms=MS[fam],
                mfc=C[fam],
                mec=C[fam],
                zorder=3,
            )
            ax.plot(
                pq,
                yy,
                ls="",
                marker=MK[POOL],
                ms=4.8,
                mfc=C[POOL] if sig else "white",
                mec=C[POOL],
                mew=1.2,
                zorder=3,
            )
            if pk in INTERIM:
                ax.text(
                    pq + 0.0008,
                    yy + 0.08,
                    "$\\dagger$",
                    va="center",
                    ha="left",
                    fontsize=8,
                    color=INK,
                )
        ax.set_title(title, fontsize=9, loc="left", color=INK, pad=4)
        ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("QLIKE (lower is better)")
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels(names)
    axes[0].set_ylim(ys[-1] - 0.6, 0.6)
    axes[0].set_xlim(0.0885, 0.1285)
    axes[1].tick_params(axis="y", left=False)
    # 2026-10-15 (user): parked -- NN content parked for now: [:3] took the MinGRU handle too; [:2] is the per-bar
    # linear and tree handles
    # h = legend_handles(sig_note=False)[:3] + [
    h = legend_handles(sig_note=False)[:2] + [
        Line2D([], [], color=INK, lw=0.7, label="per-bar OLS, HAR + cal."),
        Line2D(
            [],
            [],
            ls="",
            marker=MK[POOL],
            ms=4.8,
            mfc=C[POOL],
            mec=C[POOL],
            mew=1.2,
            label="pooled, $p<0.05$ vs per-bar",
        ),
        Line2D(
            [],
            [],
            ls="",
            marker=MK[POOL],
            ms=4.8,
            mfc="white",
            mec=C[POOL],
            mew=1.2,
            label="pooled, $p\\geq0.05$",
        ),
    ]
    fig.legend(
        handles=h,
        loc="outside lower center",
        ncol=3,
        handletextpad=0.3,
        columnspacing=1.4,
    )
    save(fig, "rehaul_results_f2_dumbbell")


def cum_diff(by: dict[str, pd.DataFrame], key: str) -> tuple[pd.Series, dict]:
    """Daily QLIKE loss differential of a forecast minus the closing-interval OLS, with its sums over all days and over
    the 24 Feb - 30 Apr 2020 window (F3; 2026-10-15: also the parked MinGRU's \\rhrCumNn* macros, off the figure)."""
    # ref = by[OLS]["qlike_raw"]  # 2026-10-15: parked -- accuracy on the rolling span (acc_of)
    # d = (by[key]["qlike_raw"] - ref).reindex(ref.index)
    ref = acc_of(by[OLS])["qlike_raw"]
    d = (acc_of(by[key])["qlike_raw"] - ref).reindex(ref.index)
    w = (d.index >= CRASH[0]) & (d.index <= CRASH[1])
    return d, {"total": float(d.sum()), "window": float(d[w].sum())}


def fig_f3(by: dict[str, pd.DataFrame], best: list[tuple[str, str, str]]) -> dict:
    """Cumulative daily QLIKE loss differential (model minus the closing-interval OLS on the HAR benchmark) for the
    closing-interval forecast with the lowest mean QLIKE in each model class."""
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(W, 2.75), layout="constrained")
    # ref = by[OLS]["qlike_raw"]  # 2026-10-15: parked -- accuracy on the rolling span (acc_of)
    ref = acc_of(by[OLS])["qlike_raw"]
    ax.axvspan(CRASH[0], CRASH[1], color=SHADE, lw=0, zorder=0)
    ax.axhline(0, color=INK, lw=0.7, zorder=1)
    out, ends = {}, []
    for key, fam, label in best:
        # d = (by[key]["qlike_raw"] - ref).reindex(ref.index)  # 2026-10-15: moved to cum_diff (same numbers)
        d, out[key] = cum_diff(by, key)
        cs = d.cumsum()
        ax.plot(cs.index, cs.to_numpy(), color=C[fam], lw=1.4, label=label, zorder=3)
        ends.append((float(cs.iloc[-1]), label.split(",")[0]))
        # w = (d.index >= CRASH[0]) & (d.index <= CRASH[1])  # 2026-10-15: moved to cum_diff (same numbers)
        # out[key] = {"total": float(d.sum()), "window": float(d[w].sum())}
    x_end = ref.index[-1] + pd.Timedelta(days=15)
    for v, lab in ends:
        ax.text(x_end, v, lab, color=INK, fontsize=8, va="center", ha="left")
    # ax.set_ylabel("Cumulative QLIKE difference\nvs per-bar OLS (sum over days)")  # 2026-10-15 (table legibility): parked -- guide terms
    ax.set_ylabel("Cumulative QLIKE loss differential\nvs closing-interval OLS")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(pd.Timestamp("2020-01-01"), ref.index[-1] + pd.Timedelta(days=5))
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: minus(f"{v:.1f}") if v else "0")
    )
    ax.tick_params(axis="y", labelcolor=MUTED, length=2.5, color=FRAME)
    ax.text(
        0.99,
        0.97,
        # "below 0: the model has lower total QLIKE than the OLS",  # 2026-10-15 (table legibility): parked -- guide terms
        "Below 0: lower cumulative loss than the closing-interval OLS",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        color=MUTED,
    )
    h, _ = ax.get_legend_handles_labels()
    h.append(Patch(facecolor=SHADE, edgecolor="none", label="24 Feb–30 Apr 2020"))
    fig.legend(
        handles=h,
        loc="outside lower center",
        ncol=4,
        handlelength=1.6,
        columnspacing=1.4,
    )
    save(fig, "rehaul_results_f3_cumulative")
    return out


def fig_f5(
    by: dict[str, pd.DataFrame], show: list[tuple[str, str | None, str]]
) -> dict:
    """Panel (a): each day's share of the OLS's total squared error; (b) share of total loss in the k worst days."""
    fig, (a, b) = plt.subplots(
        1,
        2,
        figsize=(W, 2.95),
        gridspec_kw={"width_ratios": [1.1, 1.0]},
        layout="constrained",
    )
    # f = by[OLS]  # 2026-10-15: parked -- accuracy on the rolling span (acc_of)
    f = acc_of(by[OLS])
    # se = (f["target"] - f["plain"]) ** 2  # 2026-10-15 (user): parked -- scored against the trimmed target; scores now use actual RV (data audit #10)
    se = (f["rv_unclipped"] - f["plain"]) ** 2
    share = 100 * se / se.sum()
    a.vlines(share.index, 0, share.to_numpy(), color=INK, lw=0.7)
    top = share.sort_values(ascending=False).iloc[:5]
    a.set_ylabel("Share of total squared error (%)")
    a.xaxis.set_major_locator(mdates.YearLocator())
    a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    a.set_xlim(pd.Timestamp("2020-01-01"), f.index[-1] + pd.Timedelta(days=5))
    a.set_ylim(0, None)
    a.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    a.set_axisbelow(True)
    a.tick_params(axis="y", labelcolor=MUTED, length=2.5, color=FRAME)
    span = f"{top.index.min().day}–{top.index.max().day} {top.index.max():%B %Y}"
    a.text(
        top.index.max() + pd.Timedelta(days=70),
        top.iloc[0] * 0.95,
        f"5 days, {span}:\n{top.sum():.0f}% of the total",
        fontsize=8,
        color=INK,
        va="top",
        ha="left",
    )
    a.set_title("(a) Per-bar OLS: each day's share", fontsize=9, loc="left", pad=4)
    out = {
        "ols_top5_share": float(top.sum()),
        "ols_top5_days": [str(d.date()) for d in top.index],
    }
    # k = np.arange(1, NDAYS + 1)  # 2026-10-15: parked -- the rolling span's day count
    k = np.arange(1, len(f) + 1)
    dash = (0, (3, 1.6))
    for key, fam, label in show:
        # g = by[key]  # 2026-10-15: parked -- accuracy on the rolling span (acc_of)
        g = acc_of(by[key])
        for loss, ls in (("se", "-"), ("ql", dash)):
            # v = ((g["target"] - g["plain"]) ** 2) if loss == "se" else g["qlike_raw"]  # 2026-10-15 (user): parked -- scored against the trimmed target; scores now use actual RV (data audit #10)
            v = (
                ((g["rv_unclipped"] - g["plain"]) ** 2)
                if loss == "se"
                else g["qlike_raw"]
            )
            c = 100 * np.cumsum(np.sort(v.to_numpy())[::-1]) / v.sum()
            b.plot(k, c, color=INK if fam is None else C[fam], lw=1.3, ls=ls, zorder=3)
            out[f"{key}_{loss}_top5"] = float(c[4])
            if loss == "se":
                out[f"{key}_se_top5_days"] = [
                    str(d.date()) for d in v.sort_values(ascending=False).index[:5]
                ]
    b.set_xscale("log")
    # b.set_xlim(1, NDAYS)  # 2026-10-15: parked -- the rolling span's day count
    b.set_xlim(1, len(f))
    b.set_ylim(0, 100)
    b.axvline(5, color=FRAME, lw=0.6, zorder=1)
    b.set_xlabel("Largest-loss days included, $k$ (log scale)")
    b.set_ylabel("Share of total loss (%)")
    b.grid(color=GRID, lw=0.6, zorder=0)
    b.set_axisbelow(True)
    b.tick_params(axis="y", labelcolor=MUTED, length=2.5, color=FRAME)
    b.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    b.set_title("(b) Concentration of the loss", fontsize=9, loc="left", pad=4)
    b.text(8, 84, "squared error", fontsize=8, color=INK, ha="left", va="top")
    b.text(45, 25, "QLIKE", fontsize=8, color=INK, ha="left", va="top")
    h = [
        Line2D([], [], color=INK if fam is None else C[fam], lw=1.3, label=label)
        for _, fam, label in show
    ]
    h += [
        Line2D([], [], color=MUTED, lw=1.3, label="squared error"),
        Line2D([], [], color=MUTED, lw=1.3, ls=dash, label="QLIKE"),
    ]
    fig.legend(
        handles=h,
        loc="outside lower center",
        ncol=3,
        handlelength=1.8,
        columnspacing=1.4,
        handletextpad=0.4,
    )
    save(fig, "rehaul_results_f5_se_share")
    return out


# --------------------------------------------------------------------------- main


def main() -> None:
    global SPAN, GEN, FIG
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--span",
        choices=("rolling", "mt866"),
        default="rolling",
        help="rolling: accuracy on the common rolling span (writes the paper's files); mt866: on the 866 option-data "
        "days, asserting the values of record (writes under results/rolling_eval_2026-10-15/mt866_check/ only)",
    )
    SPAN = ap.parse_args().span
    if SPAN == "mt866":
        GEN = FIG = REPO / "results" / "rolling_eval_2026-10-15" / "mt866_check"
    m, s, by = load_all()
    # lv = {k: level(f) for k, f in by.items()}  # 2026-10-15: parked -- rolling frames exist for the forecasts used
    lv = {k: level(f) for k, f in by.items() if SPAN == "mt866" or id(f) in ACC}
    gate_866 = (
        SPAN == "mt866"
    )  # the values of record below are the 866-day accuracy numbers

    # ---- known values (outline, section 6) ----
    # 2026-10-15 (data audit #10): every QLIKE / DM below is against the actual RV; the values asserted before
    # (trimmed target) are kept in the comments, old -> new also in PAPER_REHAUL_OUTLINE_2026-10-15.md
    def near(x, v, tol):
        # assert abs(x - v) <= tol, (x, v)  # 2026-10-15: accuracy values of record hold on the 866 days only
        if gate_866:
            assert abs(x - v) <= tol, (x, v)

    def near_t(x, v, tol):  # trade statistics: the 866 option-data days in every mode
        assert abs(x - v) <= tol, (x, v)

    near(lv[OLS]["qlike"], 0.0967, 5e-5)  # was 0.0946
    near(lv["subpt_enet_all_features"]["qlike"], 0.0958, 5e-5)  # was 0.0942
    near(lv["subpt_lasso_all_features"]["qlike"], 0.0931, 5e-5)  # was 0.0914
    near(lv["mingru_rinitq_all_features"]["qlike"], 0.0966, 5e-5)  # was 0.0948
    near(lv["mingru_rinitq_baseline"]["qlike"], 0.1006, 5e-5)  # was 0.0985
    near(lv["blk2"]["qlike"], 0.1092, 5e-5)  # was 0.1068
    near(lv["a0"]["qlike"], 0.1270, 5e-5)  # was 0.1232
    near_t(lv[OLS]["sharpe"], 1.37, 5e-3)
    for k in s["key"]:  # every model-set row reproduces master_table.csv
        near(lv[k]["qlike"], float(m.loc[k, "qlike_raw"]), 1e-12)
        near_t(lv[k]["sharpe"], float(m.loc[k, "Sharpe_mid"]), 1e-9)

    # ---- T1: all vs HAR + cal. within each family ----
    # 2026-10-15 (user): NN content parked for now -- the MinGRU (NN_ROW) is still paired, for the gates and its macros
    # (T1 / T2 lookups), and dropped from the table and figure rows below
    t1 = []
    for name, fam, pb, pool in FAMILY_ROWS + [NN_ROW]:
        t1.append(
            {
                "name": name,
                "family": fam,
                "fit": "per-bar",
                **pair(by[pb[1]], by[pb[0]]),
            }
        )
        if pool is None:
            t1.append({"name": name, "family": fam, "fit": "pooled", "pending": True})
        else:
            t1.append(
                {
                    "name": name,
                    "family": fam,
                    "fit": "pooled",
                    "mark": "$^\\dagger$" if pool[1] in INTERIM else "",
                    **pair(by[pool[1]], by[pool[0]]),
                }
            )
    t1 = [r for r in t1 if r["fit"] == "per-bar"] + [
        r for r in t1 if r["fit"] == "pooled"
    ]
    T1 = {(r["fit"], r["name"]): r for r in t1}
    # 2026-10-15 (user): parked -- NN content parked for now; T1 rows: 14 -> 12
    t1 = [r for r in t1 if r["family"] != NN]
    assert len(t1) == 2 * len(FAMILY_ROWS) == 12, len(t1)
    # 2026-10-15 (user): parked -- scored against the trimmed target; scores now use actual RV (data audit #10):
    # expect_t1 = {  # (fit, name): (QLIKE base, QLIKE all, % change, DM)  -- the outline's T1, trimmed target
    #     ("per-bar", "Ridge"): (0.0946, 0.0958, 1.2, 0.26),
    #     ("per-bar", "Lasso"): (0.0946, 0.0914, -3.4, -1.22),
    #     ("per-bar", "Elastic net"): (0.0946, 0.0942, -0.4, -0.10),
    #     ("per-bar", "LightGBM"): (0.0986, 0.0939, -4.7, -1.43),
    #     ("per-bar", "XGBoost"): (0.0976, 0.0935, -4.2, -1.41),
    #     ("per-bar", "Random forest"): (0.0968, 0.0966, -0.2, -0.05),
    #     ("per-bar", NN_NAME): (0.0985, 0.0948, -3.8, None),
    #     ("pooled", "Ridge"): (None, 0.102, -4.3, -1.29),
    #     ("pooled", "Lasso"): (None, 0.104, -1.7, -0.50),
    #     ("pooled", "Elastic net"): (None, 0.102, -4.7, -1.42),
    #     ("pooled", "LightGBM"): (None, 0.106, -14.3, -4.57),
    #     ("pooled", "XGBoost"): (None, 0.106, -12.4, -4.46),
    #     ("pooled", "Random forest"): (None, 0.107, -12.7, -4.57),
    # }
    expect_t1 = {  # (fit, name): (QLIKE base, QLIKE all, % change, DM)  -- the outline's T1, actual RV
        ("per-bar", "Ridge"): (0.0967, 0.0975, 0.8, 0.18),
        ("per-bar", "Lasso"): (0.0967, 0.0931, -3.7, -1.35),
        ("per-bar", "Elastic net"): (0.0967, 0.0958, -0.9, -0.20),
        ("per-bar", "LightGBM"): (0.1010, 0.0961, -4.9, -1.45),
        ("per-bar", "XGBoost"): (0.1001, 0.0957, -4.3, -1.45),
        ("per-bar", "Random forest"): (0.0992, 0.0991, -0.0, -0.01),
        ("per-bar", NN_NAME): (0.1006, 0.0966, -4.0, None),
        ("pooled", "Ridge"): (None, 0.1045, -4.8, -1.45),
        ("pooled", "Lasso"): (None, 0.107, -2.1, -0.63),
        ("pooled", "Elastic net"): (None, 0.104, -5.1, -1.57),
        ("pooled", "LightGBM"): (None, 0.109, -14.6, -4.64),
        ("pooled", "XGBoost"): (None, 0.109, -12.7, -4.55),
        ("pooled", "Random forest"): (None, 0.110, -13.0, -4.63),
    }
    for kk, (qb, qa, pct, dmv) in expect_t1.items():
        r = T1[kk]
        if qb is not None:
            near(r["qb"], qb, 5e-5)
        near(r["qa"], qa, 5e-5 if qa < 0.1 or kk[0] == "per-bar" else 5e-4)
        near(r["pct"], pct, 0.05 + 1e-9)
        if dmv is not None:
            near(
                r["dm"], dmv, 0.02
            )  # same-estimator base; the outline paired the per-bar linear rows with the OLS
    near(T1[("per-bar", NN_NAME)]["p"], 0.32, 0.005)  # was 0.34 (trimmed target)
    expect_dsh = {
        "Ridge": (0.05, 0.09),
        "Lasso": (0.50, 1.47),
        "Elastic net": (0.55, 1.34),
        "LightGBM": (0.54, 1.11),
        "XGBoost": (0.16, 0.30),
        "Random forest": (0.18, 0.34),
    }
    for nme, (dsh, t) in expect_dsh.items():
        near_t(T1[("per-bar", nme)]["dsh"], dsh, 0.005 + 1e-9)
        near_t(T1[("per-bar", nme)]["t"], t, 0.02)

    # all 51 input-set pairs of the model set (the fig_qlike_sharpe counts)
    allf = [k for k in s.loc[s["input_set"] == "all_features", "key"]]
    af = pd.DataFrame(
        [pair(by[k], by[k.replace("all_features", "baseline")]) for k in allf],
        index=allf,
    )
    n_af, n_lower, n_lower_sig = (
        len(af),
        int((af["pct"] < 0).sum()),
        int(((af["pct"] < 0) & (af["p"] < SIG)).sum()),
    )
    n_mse_up, n_mse_up_sig = (
        int((af["mse_pct"] > 0).sum()),
        int(((af["mse_pct"] > 0) & (af["p_mse"] < SIG)).sum()),
    )
    # (51, 38, 2) on the trimmed target (parked 2026-10-15, data audit #10)
    assert n_af == 51, n_af
    if gate_866:  # 2026-10-15: the 866-day values of record
        assert (n_af, n_lower, n_lower_sig) == (51, 38, 3), (n_af, n_lower, n_lower_sig)
        assert (n_mse_up, n_mse_up_sig) == (51, 0), (n_mse_up, n_mse_up_sig)
    assert (
        int((af["dsh"] > 0).sum()) == 50
        and int(((af["dsh"] > 0) & (af["p_t"] < SIG)).sum()) == 4
    )

    # ---- 2026-10-15: the trees against the closing-interval ridge on the same predictor set (Model classes, 6.5) ----
    # The 48 tree forecasts of the model set the campaign compared with the ridge (writeup/make_table_close_main.py
    # \cmTrVsRidge*: re-estimated every 10 sessions or daily, untuned or random search, and Optuna best of 50 at
    # TUNE_PER 1 / 5 / 25 / 250, MSE rule), each against the ridge of record subpt_ridge_<set>, re-scored here on this
    # run's span (the campaign's counts were written on the 866 days with the campaign-era XGBoost / forest).
    tr_keys = [
        k
        for k in s["key"]
        if re.fullmatch(
            r"subtree_(?:(?:lgbm|xgb|rf)_(?:baseline|all_features)|(?:daily_|tuned_|tuned_daily_|optuna_tp\d+_)"
            r"(?:baseline|all_features)_(?:lgbm|xgb|rf))",
            k,
        )
    ]
    assert len(tr_keys) == 48, len(tr_keys)
    tr = pd.DataFrame(
        [
            pair(
                by[k],
                by[
                    "subpt_ridge_"
                    + ("all_features" if "all_features" in k else "baseline")
                ],
            )
            for k in tr_keys
        ],
        index=tr_keys,
    )
    n_tr_above = int((tr["pct"] > 0).sum())
    n_tr_above_sig = int(((tr["pct"] > 0) & (tr["p"] < SIG)).sum())
    n_tr_below_sig = int(((tr["pct"] < 0) & (tr["p"] < SIG)).sum())
    print(
        f"trees vs the closing-interval ridge ({SPAN}): QLIKE higher in {n_tr_above} of {len(tr)}, "
        f"{n_tr_above_sig} significantly; lower and significant: {n_tr_below_sig}"
    )
    # 2026-10-15: the Feb-Apr 2020 claim of 6.3 ("excluding February--April 2020, the lasso on the full predictor set
    # has a lower MSE than the benchmark forecast"), on this run's span
    la_, ol_ = acc_of(by["subpt_lasso_all_features"]), acc_of(by[OLS])
    out_w = (la_.index < CRASH[0]) | (la_.index > CRASH[1])
    mse_ex = [
        float(((x["rv_unclipped"] - x["plain"])[out_w] ** 2).mean()) for x in (la_, ol_)
    ]
    lasso_mse_ex_pct = 100.0 * (mse_ex[0] / mse_ex[1] - 1.0)
    print(
        f"lasso (full) vs OLS, MSE outside 24 Feb - 30 Apr 2020: {lasso_mse_ex_pct:+.1f} %"
    )

    # ---- T2: pooled minus per-bar ----
    t2 = []
    # 2026-10-15 (user): NN content parked for now -- NN_ROW for the gates and macros only (dropped from the rows below)
    for name, fam, pb, pool in FAMILY_ROWS + [NN_ROW]:
        if pool is None:
            t2.append(
                {
                    "name": name,
                    "family": fam,
                    "pending": True,
                    "perbar": pb,
                    "pooled": None,
                }
            )
            continue
        r = {
            "name": name,
            "family": fam,
            "perbar": pb,
            "pooled": pool,
            "mark": "$^\\dagger$" if pool[1] in INTERIM else "",
        }
        for i, b in enumerate(("baseline", "all_features")):
            r[b] = pair(by[pool[i]], by[pb[i]])
        t2.append(r)
    T2 = {r["name"]: r for r in t2}
    # 2026-10-15 (user): parked -- NN content parked for now; T2 rows per panel: 7 -> 6
    t2 = [r for r in t2 if r["family"] != NN]
    assert len(t2) == len(FAMILY_ROWS) == 6, len(t2)
    # trimmed target (parked 2026-10-15): Ridge (12.8, 6.6, 1.56), Lasso (12.2, 14.2, 3.19), Elastic net (12.6, 7.8, 2.00)
    expect_t2 = {
        "Ridge": (13.6, 7.2, 1.61),
        "Lasso": (13.0, 14.9, 3.09),
        "Elastic net": (13.4, 8.5, 2.03),
    }
    for nme, (pb_, pa_, dma) in expect_t2.items():
        near(T2[nme]["baseline"]["pct"], pb_, 0.05 + 1e-9)
        near(T2[nme]["all_features"]["pct"], pa_, 0.05 + 1e-9)
        near(T2[nme]["all_features"]["dm"], dma, 0.005 + 1e-9)
        assert not gate_866 or 3.35 <= T2[nme]["baseline"]["dm"] <= 3.55
    for est, nme, sc in (
        ("lgbm", "LightGBM", "score_lgbm"),
        ("xgb", "XGBoost", "score_xgb"),
        ("rf", "Random forest", "score_rf_interim1600"),
    ):
        pp = pd.read_csv(POOLTREE / sc / "pairs.csv")
        pp = pp[pp["vs"] == "per-bar T1"].set_index("bucket")
        for b in ("baseline", "all_features"):
            near(T2[nme][b]["pct"], float(pp.loc[b, "qlike_pct"]), 1e-6)
            near(T2[nme][b]["dm"], float(pp.loc[b, "dm"]), 1e-6)
            near_t(T2[nme][b]["t"], float(pp.loc[b, "hac_t_mid"]), 1e-6)
        lvp = pd.read_csv(POOLTREE / sc / "levels.csv")
        lvp = lvp[lvp["forecast"].str.startswith("pooled")].set_index("bucket")
        near(
            lv[f"pooltree_{est}_baseline"]["qlike"],
            float(lvp.loc["baseline", "qlike"]),
            1e-12,
        )
        near(
            lv[f"pooltree_{est}_all_features"]["qlike"],
            float(lvp.loc["all_features", "qlike"]),
            1e-12,
        )
    for nme in ("LightGBM", "XGBoost", "Random forest"):
        # trimmed target (parked 2026-10-15): HAR + cal. 24.0-27.5 % / DM 4.3-4.8; all features 11.0-14.0 % / DM 2.7-3.0
        assert not gate_866 or (
            25.0 <= T2[nme]["baseline"]["pct"] <= 28.0
            and 4.2 <= T2[nme]["baseline"]["dm"] <= 4.6
        )
        assert not gate_866 or (
            11.0 <= T2[nme]["all_features"]["pct"] <= 14.2
            and 2.7 <= T2[nme]["all_features"]["dm"] <= 3.1
        )
    t2_t = [
        r[b]["t"]
        for r in t2
        if not r.get("pending") and r["family"] != NN
        for b in ("baseline", "all_features")
    ]
    assert max(t2_t) < 1 and max(abs(x) for x in t2_t) < Z95, (
        t2_t
    )  # outline: every linear / tree HAC t below 1, none significant

    # ---- the pooled ridge-init MinGRU (coordinator, 2026-10-15; pooled_rinit_study/at1600_*.csv) ----
    pa, pb0 = "mingru_pooled_rinitq_all_features", "mingru_pooled_rinitq_baseline"
    near(lv[pa]["qlike"], 0.1141, 5e-5)  # was 0.1121 (trimmed target)
    near(lv[pb0]["qlike"], 0.1177, 5e-5)  # was 0.1151
    near_t(lv[pa]["sharpe"], 0.34, 5e-3)
    near_t(lv[pa]["sharpe_x"], -0.13, 5e-3)
    near_t(lv[pb0]["sharpe"], 0.78, 5e-3)
    near_t(lv[pb0]["sharpe_x"], 0.31, 5e-3)
    sc = pd.read_csv(NN_POOLED / "at1600_scores.csv")
    sc = sc[sc["model"] == "pooled ridge-init QLIKE (MinGRU)"].set_index("design")
    for b, k in (("all_features", pa), ("baseline", pb0)):
        near(lv[k]["qlike"], float(sc.loc[b, "qlike"]), 1e-9)
        near_t(lv[k]["sharpe"], float(sc.loc[b, "sharpe_mid"]), 1e-6)
        near_t(lv[k]["sharpe_x"], float(sc.loc[b, "sharpe_crossed"]), 1e-6)
    near(T1[("pooled", NN_NAME)]["pct"], -3.0, 0.05)  # was -2.7 (trimmed target)
    near(T1[("pooled", NN_NAME)]["p"], 0.47, 0.005)  # was 0.52
    near(T2[NN_NAME]["all_features"]["pct"], 18.1, 0.05)  # was 18.2
    near(T2[NN_NAME]["baseline"]["pct"], 17.0, 0.05)  # was 16.8
    assert not gate_866 or (
        T2[NN_NAME]["all_features"]["p"] < 1e-4 * 1.5
        and T2[NN_NAME]["baseline"]["p"] < 2e-3  # was < 1.5e-3
    )
    cn = pd.read_csv(NN_POOLED / "at1600_contrasts.csv").set_index(["model", "vs"])
    who = "pooled ridge-init QLIKE (MinGRU) ({})"
    perbar = "per-bar ridge-init QLIKE (MinGRU) ({})"
    for got, (mod, vs) in (
        (T1[("pooled", NN_NAME)], (who.format("all_features"), who.format("baseline"))),
        (
            T1[("per-bar", NN_NAME)],
            (perbar.format("all_features"), perbar.format("baseline")),
        ),
        (
            T2[NN_NAME]["all_features"],
            (who.format("all_features"), perbar.format("all_features")),
        ),
        (T2[NN_NAME]["baseline"], (who.format("baseline"), perbar.format("baseline"))),
    ):
        c = cn.loc[(mod, vs)]
        near(got["pct"], float(c["qlike_pct"]), 1e-6)
        near(got["p"], float(c["p_qlike"]), 1e-6)  # the study's p is dm_test's: same DM
        near_t(got["t"], float(c["hac_t_mid"]), 1e-6)

    # ---- T3: ranking vs the per-bar OLS ----
    extra = [
        # 2026-10-15 (user): parked -- NN content parked for now (ranks recompute among the other forecasts)
        # "mingru_rinitq_baseline",
        # "mingru_rinitq_all_features",
        # "mingru_pooled_rinitq_baseline",
        # "mingru_pooled_rinitq_all_features",
        "pooltree_lgbm_baseline",
        "pooltree_xgb_baseline",
        "pooltree_rf_baseline",
        "pooltree_rf_all_features",
    ]
    keys = list(s["key"]) + extra
    rows = []
    for k in keys:
        # 2026-10-15 (table legibility): parked -- code labels; describe_key gives the guide's names and the panel
        # name, fam, fit, inputs = rung_label(k, m)
        # r = {
        #     "key": k,
        #     "name": name,
        #     "family": fam,
        #     "fit": fit,
        #     "inputs": inputs,
        #     "qlike": lv[k]["qlike"],
        #     "in_set": k in set(s["key"]),
        # }
        r = {
            "key": k,
            **describe_key(k),
            "qlike": lv[k]["qlike"],
            "in_set": k in set(s["key"]),
        }
        if k != OLS:
            p = pair(by[k], by[OLS])
            r.update(pct=p["pct"], dm=p["dm"], p=p["p"])
        rows.append(r)
    rank = pd.DataFrame(rows).set_index("key").sort_values("qlike")
    rank["rank"] = np.arange(1, len(rank) + 1)
    # 2026-10-15 (user): NN content parked for now -- the four MinGRU forecasts leave the ranking: forecasts ranked
    # 120 -> 116; the per-bar OLS ranks 9 -> 8 (the per-bar MinGRU on the full predictor set was 5th, above it)
    assert not rank.index.str.startswith("mingru_").any()
    assert len(rank) == 116, len(rank)  # was 120
    assert not gate_866 or int(rank.loc[OLS, "rank"]) == 8, rank.loc[
        OLS, "rank"
    ]  # was 9
    ms = rank[rank["in_set"] & (rank.index != OLS)]
    n_set, n_lower3, n_lower3_sig, n_worse_sig = (
        len(ms),
        int((ms["pct"] < 0).sum()),
        int(((ms["pct"] < 0) & (ms["p"] < SIG)).sum()),
        int(((ms["pct"] > 0) & (ms["p"] < SIG)).sum()),
    )
    # (111, 7, 0, 48) on the trimmed target (parked 2026-10-15)
    assert n_set == 111, n_set
    assert not gate_866 or (n_set, n_lower3, n_lower3_sig, n_worse_sig) == (
        111,
        7,
        0,
        51,
    ), (
        n_set,
        n_lower3,
        n_lower3_sig,
        n_worse_sig,
    )
    lin_base_max = float(
        rank.loc[
            ["subpt_ridge_baseline", "subpt_lasso_baseline", "subpt_enet_baseline"],
            "pct",
        ]
        .abs()
        .max()
    )
    assert not gate_866 or lin_base_max < 0.05
    head = [OLS, "a0", "blk2"]
    for _, _, pb, pool in FAMILY_ROWS:
        head += list(pb) + (list(pool) if pool else [])
    alias = {
        "pooltree_lgbm_all_features": "lgbm",
        "pooltree_xgb_all_features": "xgb",
    }  # the master table's keys
    head = [
        alias.get(k, k)
        for k in head
        if k
        not in ("subpt_ridge_baseline", "subpt_lasso_baseline", "subpt_enet_baseline")
    ]
    assert len(set(head)) == len(head)
    # 2026-10-15 (table legibility): the markers and panels the new tables print
    for (
        k_pool,
        k_master,
    ) in alias.items():  # tnote a of T1 / T2: the earlier draft's pooled trees
        assert by[k_pool] is by[k_master], k_pool
        assert describe_key(k_master)["spec"].startswith("Earlier draft"), k_master
    assert {n for n, *_, pool in FAMILY_ROWS if pool and pool[1] in alias} == set(
        EARLIER_POOLED_FULL
    )
    sub3 = rank.loc[
        head
    ]  # T3: the closing-interval rows rank above the pooled intraday rows (panels keep the order)
    t3_split = bool(
        sub3.loc[sub3["fit"] == "per-bar", "rank"].max()
        < sub3.loc[sub3["fit"] == "pooled", "rank"].min()
    )
    assert not gate_866 or t3_split  # 2026-10-15: the 866-day ordering of record
    if not t3_split:
        print(
            "T3 (parked): on this span an intraday headline row ranks above a closing-interval one"
        )
    assert (
        set(rank["panel"]) == {"A", "B", "C"}
        and (rank.loc[rank["panel"] == "A", "fit"] == "per-bar").all()
    )

    # ---- figures ----
    style()
    FIG.mkdir(exist_ok=True)
    fig_f1(t1)
    fig_f2(t2, lv)
    # 2026-10-15 (table legibility): parked -- "all features" in the legend; the caption says every line is on the
    # full predictor set
    # best = [
    #     ("subpt_lasso_all_features", LIN, "Lasso, all features"),
    #     ("subtree_daily_all_features_xgb", TREE, "XGBoost, all features"),
    #     ("mingru_rinitq_all_features", NN, "MinGRU, all features"),
    # ]
    best = [
        ("subpt_lasso_all_features", LIN, "Lasso"),
        ("subtree_daily_all_features_xgb", TREE, "XGBoost"),
        # ("mingru_rinitq_all_features", NN, "MinGRU"),  # 2026-10-15 (user): parked -- NN content parked for now
    ]
    for key, fam, _ in best:  # the best per-bar forecast of each family on QLIKE
        famrows = rank[(rank["family"] == fam) & (rank["fit"] == "per-bar")]
        assert not gate_866 or famrows.index[0] == key, (fam, famrows.index[0])
    cum = fig_f3(by, best)
    # 2026-10-15 (user): NN content parked for now -- the MinGRU is off F3 (and F5); its \rhrCumNn* sums stay defined
    nnk = NN_ROW[2][1]  # the per-bar MinGRU on the full predictor set (was best[2])
    cum[nnk] = cum_diff(by, nnk)[1]
    show: list[tuple[str, str | None, str]] = [(OLS, None, "OLS, HAR + cal."), *best]
    f5 = fig_f5(by, show)  # (typed for mypy, 2026-10-15; same list as before)
    se_top5 = [f5[f"{k}_se_top5"] for k, *_ in [(OLS,)] + best]
    assert len(se_top5) == 3  # F5 forecasts: 4 -> 3 (MinGRU parked)
    assert not gate_866 or all(0.85 <= x / 100 <= 0.925 for x in se_top5), (
        se_top5
    )  # outline: five March 2020 days carry 85-92 % (83-91 % on the trimmed target)
    for k, *_ in [
        (OLS,)
    ] + best:  # the five largest squared errors of each F5 forecast fall in March 2020
        assert all(d.startswith("2020-03") for d in f5[f"{k}_se_top5_days"]), (
            k,
            f5[f"{k}_se_top5_days"],
        )
    ql_top5 = [f5[f"{k}_ql_top5"] for k, *_ in [(OLS,)] + best]

    # ---- tables ----
    GEN.mkdir(exist_ok=True)
    (GEN / "rehaul_results_t1_axis1.tex").write_text(
        table_t1(t1), encoding="utf-8", newline="\n"
    )
    (GEN / "rehaul_results_t2_pooled.tex").write_text(
        table_t2(t2), encoding="utf-8", newline="\n"
    )
    (GEN / "rehaul_results_t3_ranking.tex").write_text(
        table_t3(rank, head), encoding="utf-8", newline="\n"
    )
    (GEN / "rehaul_results_t3_full.tex").write_text(
        table_t3_full(rank), encoding="utf-8", newline="\n"
    )
    print("wrote generated/rehaul_results_t{1_axis1,2_pooled,3_ranking,3_full}.tex")

    # ---- 2026-10-15: the claims the prose of 06_results.tex makes on the rolling span (asserted, so it can't drift) ----
    if SPAN == "rolling":
        sig_t1 = {(r["fit"], r["name"]) for r in t1 if r["p"] < SIG}
        assert sig_t1 == {
            ("pooled", "LightGBM"),
            ("pooled", "XGBoost"),
            ("pooled", "Random forest"),
            ("pooled", "Elastic net"),
        }, (
            sig_t1
        )  # 6.3: the pooled trees and the pooled elastic net are the significant reductions
        assert all(r["pct"] < 0 for r in t1), (
            "6.3: Full lowers QLIKE in every model class"
        )
        assert all(r[b]["pct"] > 0 for r in t2 for b in ("baseline", "all_features")), (
            "6.4: pooled > closing"
        )
        assert {
            r["name"] for r in t2 if r["family"] == LIN and r["all_features"]["p"] < SIG
        } == {"Lasso"}
        assert all(r["baseline"]["p"] < SIG for r in t2), (
            "6.4: every HAR-benchmark difference significant"
        )
        assert all(r["all_features"]["p"] < SIG for r in t2 if r["family"] == TREE), (
            "6.4: trees, full set"
        )
        top4 = set(rank.index[:4])
        assert top4 == {
            "subpt_lasso_all_features",
            "subpt_enet_all_features",
            "subpt_ridge_all_features",
            "subtree_daily_all_features_xgb",
        }, (
            top4
        )  # 6.5: the four most accurate, each closing-interval on the full set ...
        assert all(
            rank.loc[k, "p"] >= SIG for k in top4
        )  # ... none significantly better than the benchmark
        assert lasso_mse_ex_pct < 0, (
            "6.3: outside Feb-Apr 2020 the lasso (full) has the lower MSE"
        )
        assert (
            list(rank.index[-3:]).count("pooltree_lgbm_baseline")
            + list(rank.index[-3:]).count("pooltree_rf_baseline")
            == 2
        ), "6.5: the pooled trees on HAR are among the least accurate"

    # ---- 2026-10-15 (user, restructure): "I need the last 30 min scheme to be moved to the end of the economic
    # application section as a final cherry on top instead of competing with the exogenous feature signal story".
    # Experiments and the body of Economic value now present the pooled intraday models only (the 12 forecasts of T1
    # Panel B); the closing-interval (last-30-minutes) results move to the final subsection of Economic value
    # (sec:rh_econ_last30). Pooled-only statistics for that prose, each claim asserted. The model-set versions they
    # replace in the body are recomputed here from the same frames and gated against their macros of record
    # (close_main_numbers: \cmMasterVsShort*; rehaul_econ_numbers: \rheShortSh, \rheEnet*, \rheSz*, \rheTailNeg), so
    # the pooled numbers use the same construction.
    pool_keys = [
        k for *_, pool in FAMILY_ROWS for k in pool
    ]  # T1 Panel B: 6 classes x (HAR, Full)
    pool_full = [pool[1] for *_, pool in FAMILY_ROWS]
    pool_har = [pool[0] for *_, pool in FAMILY_ROWS]
    lin_n = [n for n, fam, *_ in FAMILY_ROWS if fam == LIN]
    tree_n = [n for n, fam, *_ in FAMILY_ROWS if fam == TREE]
    pool_of = {n: pool for n, _, _, pool in FAMILY_ROWS}
    assert len(pool_keys) == 12 and len(lin_n) == 3 and len(tree_n) == 3
    t1p = {n: T1[("pooled", n)] for n in lin_n + tree_n}
    t1c = {n: T1[("per-bar", n)] for n in lin_n + tree_n}

    # the short-straddle benchmark (q_t = -1) and each forecast's sign strategy against it (866 days)
    ref_ix = by[OLS].index
    assert np.allclose(np.abs(by[OLS]["q"]), 1.0)
    r_book = by[OLS]["ret_mid"] * by[OLS]["q"]  # the straddle return R_t (q = +-1)
    short_mid = -r_book
    near_t(sharpe(short_mid), 0.20, 5e-3)  # = \rheShortSh

    def vs_short(k: str) -> dict:
        f = by[k]
        assert f.index.equals(ref_ix), k
        d = f["ret_mid"] - short_mid
        t_, lag_ = asl.newey_west_t(d)
        assert lag_ == HAC_LAG
        return {
            "dsh": sharpe(f["ret_mid"]) - sharpe(short_mid),
            "t": float(t_),
            "p": float(math.erfc(abs(t_) / math.sqrt(2.0))),
            "tail": float(d.sum() - d.nlargest(TOP_DAYS).sum()),
        }

    vs_ms = {
        k: vs_short(k) for k in s["key"]
    }  # gate: the model set reproduces its macros of record
    assert sum(v["dsh"] > 0 for v in vs_ms.values()) == 112  # \cmMasterVsShortAbove
    assert (
        sum(v["t"] > Z95 for v in vs_ms.values()) == 20
    )  # \cmMasterVsShortAboveSig = \rheSigVsShort
    assert sum(v["tail"] < 0 for v in vs_ms.values()) == 102  # \rheTailNeg
    en = "subpt_enet_all_features"  # \rheEnetSh, \rheEnetShX, \rheEnetT
    near_t(lv[en]["sharpe"], 1.92, 5e-3)
    near_t(lv[en]["sharpe_x"], 1.43, 5e-3)
    near_t(vs_ms[en]["t"], 2.56, 5e-3)
    vs_sh = {
        k: vs_short(k)
        for _, _, pb, pool in FAMILY_ROWS + [NN_ROW]
        for k in (*pb, *pool)
    }
    pool_sig_short = {k for k in pool_keys if vs_sh[k]["t"] > Z95}

    # sizing: the GUI's scorer on the gap plain - iv_var, scored after the 252-day warm-up (= make_rehaul_economics.py)
    qbook = pd.read_parquet(
        QUOTES, columns=["R", "exit", "bid_c", "bid_p", "ask_c", "ask_p"]
    )
    qbook.index = pd.to_datetime(qbook.index).normalize()
    qbook = qbook.reindex(ref_ix)
    assert np.allclose(
        qbook["R"].to_numpy(float), r_book.to_numpy(float), rtol=1e-12, atol=1e-15
    )

    def sizing(k: str) -> dict:
        f = by[k]
        return gui._score_sizes(
            (f["plain"] - f["iv_var"]).to_numpy(float),
            qbook["R"].to_numpy(float),
            qbook["exit"].to_numpy(float),
            (qbook["bid_c"] + qbook["bid_p"]).to_numpy(float),
            (qbook["ask_c"] + qbook["ask_p"]).to_numpy(float),
        )

    sz_ms = {
        k: sizing(k) for k in s["key"]
    }  # gate: \rheSzSignMed, \rheSzClipMed, \rheSzLfssMed, \rheSzRankMed
    meds_ms = tuple(
        round(float(np.median([v[r]["sharpe_mid"] for v in sz_ms.values()])), 2)
        for r in ("sign", "clip", "lfss", "rank")
    )
    assert meds_ms == (0.92, 1.34, 1.12, 1.34), meds_ms
    assert (
        sum(v["clip"]["d_vs_sign"] > 0 for v in sz_ms.values()) == 98
    )  # \rheSzClipBeats
    sz_pool = {k: sizing(k) for k in pool_keys}

    def med_sz(r: str) -> float:
        return float(np.median([sz_pool[k][r]["sharpe_mid"] for k in pool_keys]))

    # squared-error concentration of the pooled forecasts on the full predictor set (rolling span)
    se_pool, ql_pool = [], []
    for k in pool_full:
        a_ = acc_of(by[k])
        se_ = (a_["rv_unclipped"] - a_["plain"]) ** 2
        top_ = se_.nlargest(5)
        se_pool.append(100.0 * float(top_.sum() / se_.sum()))
        ql_pool.append(
            100.0 * float(a_["qlike_raw"].loc[top_.index].sum() / a_["qlike_raw"].sum())
        )
        assert SPAN != "rolling" or all(
            str(d.date()).startswith("2020-03") for d in top_.index
        ), (k, top_.index)

    # the pooled forecasts against the benchmark forecast (the closing-interval OLS on HAR), and their QLIKE levels
    vb_pool = {k: pair(by[k], by[OLS]) for k in pool_keys}
    qv = {k: lv[k]["qlike"] for k in pool_keys}

    def qs(names: list[str], i: int) -> list[float]:
        return [qv[pool_of[n][i]] for n in names]

    vb_lin_full = [vb_pool[pool_of[n][1]]["pct"] for n in lin_n]
    vb_tree_full = [vb_pool[pool_of[n][1]]["pct"] for n in tree_n]
    vb_har = [vb_pool[k]["pct"] for k in pool_har]
    sh_full = [lv[k]["sharpe"] for k in pool_full]
    sh_har = [lv[k]["sharpe"] for k in pool_har]
    dsh_pool = [t1p[n]["dsh"] for n in lin_n + tree_n]
    if (
        SPAN == "rolling"
    ):  # the prose's claims (06_results.tex, 07_economics.tex), asserted
        # 5.3: the full predictor set lowers QLIKE in every pooled class; significant for the trees and the enet; the
        # trees' reductions are the largest; the squared error is mixed, never significant
        assert all(r["pct"] < 0 for r in t1p.values())
        assert {n for n, r in t1p.items() if r["p"] < SIG} == set(tree_n) | {
            "Elastic net"
        }
        assert min(abs(t1p[n]["pct"]) for n in tree_n) > max(
            abs(t1p[n]["pct"]) for n in lin_n
        )
        assert not any(r["p_mse"] < SIG for r in t1p.values())
        # 5.5: the three most accurate pooled forecasts are the linear models on the full predictor set; linear below
        # trees on each predictor set; every pooled forecast above the benchmark forecast, on the full predictor set
        # significantly for the trees only, on the HAR benchmark all at 1 %; the full set narrows the gap in each class
        assert set(sorted(pool_keys, key=lambda k: qv[k])[:3]) == {
            pool_of[n][1] for n in lin_n
        }
        assert max(qs(lin_n, 1)) < min(qs(tree_n, 1)) and max(qs(lin_n, 0)) < min(
            qs(tree_n, 0)
        )
        assert all(vb_pool[k]["pct"] > 0 for k in pool_keys)
        assert {n for n in lin_n + tree_n if vb_pool[pool_of[n][1]]["p"] < SIG} == set(
            tree_n
        )
        assert all(vb_pool[k]["p"] < 0.01 for k in pool_har)
        assert all(
            vb_pool[pool_of[n][1]]["pct"] < vb_pool[pool_of[n][0]]["pct"]
            for n in lin_n + tree_n
        )
        # final subsection: the full predictor set lowers QLIKE in every closing-interval class, none at 5 %
        assert all(r["pct"] < 0 for r in t1c.values()) and not any(
            r["p"] < SIG for r in t1c.values()
        )
    # 6.3 (trade statistics, 866 days in every mode): every pooled sign strategy above the short straddle, the
    # significant gains are the three trees on the full predictor set; the three highest pooled Sharpe ratios are the
    # elastic net, ridge and LightGBM on the full predictor set; the full set raises every pooled Sharpe ratio, never
    # significantly; without its 10 best days every pooled forecast's gain over the short straddle is negative
    assert all(vs_sh[k]["dsh"] > 0 for k in pool_keys)
    assert pool_sig_short == {pool_of[n][1] for n in tree_n}, pool_sig_short
    assert sorted(pool_keys, key=lambda k: -lv[k]["sharpe"])[:3] == [
        "poolpt_enet_all_features",
        "poolpt_ridge_all_features",
        "pooltree_lgbm_all_features",
    ]
    assert min(sh_full) > max(sh_har)
    assert all(x > 0 for x in dsh_pool) and not any(t1p[n]["p_t"] < SIG for n in t1p)
    assert all(vs_sh[k]["tail"] < 0 for k in pool_keys)
    assert sum(sz_pool[k]["clip"]["d_vs_sign"] > 0 for k in pool_keys) == 11
    # final subsection: the closing-interval elastic net and lasso on the full predictor set earn a higher Sharpe ratio
    # than every pooled intraday forecast
    assert min(
        lv[k]["sharpe"] for k in ("subpt_enet_all_features", "subpt_lasso_all_features")
    ) > max(lv[k]["sharpe"] for k in pool_keys)
    # ... and the elastic net's is the highest of the model set (= \cmMasterShMax)
    assert max(s["key"], key=lambda k: lv[k]["sharpe"]) == "subpt_enet_all_features"

    # ---- macros for the captions ----
    def pm(x: float, nd: int) -> str:
        s_ = f"{x:+.{nd}f}"
        return f"\\ensuremath{{{s_[0] if s_[0] == '+' else '-'}}}{s_[1:]}"

    # lasso, xgbk, nnk = (k for k, *_ in best)  # 2026-10-15 (user): parked -- NN content parked for now
    lasso, xgbk = (k for k, *_ in best)  # nnk: set above, off the figure
    nums = {
        # "Days": (str(NDAYS), "trade days scored"),  # 2026-10-15: parked -- the comment; the value is unchanged
        "Days": (
            str(NDAYS),
            "days of the option-data subsample: trades, Sharpe ratios (accuracy: rhoosDays of rehaul_oos_numbers)",
        ),
        "NNPooledSeeds": (
            str(NN_POOLED_SEEDS[0]),
            "seeds averaged in the pooled MinGRU forecast",
        ),
        "OlsQ": (q4(lv[OLS]["qlike"]), "QLIKE of the per-bar OLS on HAR + calendar"),
        "AfPairs": (
            str(n_af),
            "model-set input-set pairs (all features vs HAR + cal.)",
        ),
        "AfLower": (str(n_lower), "... QLIKE lower with all features"),
        "AfLowerSig": (str(n_lower_sig), "... and DM p < 0.05"),
        "AfMseHigher": (
            str(n_mse_up),
            "... variance-space MSE higher with all features",
        ),
        "AfMseHigherSig": (str(n_mse_up_sig), "... and DM (squared error) p < 0.05"),
        "NModelSet": (str(len(s)), "forecasts in the master table's model set"),
        "RankN": (str(len(rank)), "forecasts ranked in the full table"),
        "NSet": (str(n_set), "model-set forecasts other than the OLS"),
        "NLower": (str(n_lower3), "... with lower QLIKE than the OLS"),
        "NLowerSig": (str(n_lower3_sig), "... and DM p < 0.05"),
        "NWorseSig": (
            str(n_worse_sig),
            "... significantly worse than the OLS (DM p < 0.05)",
        ),
        "LinBaseMaxPct": (
            f"{lin_base_max:.2f}",
            "largest |% change| of the per-bar ridge / lasso / enet on HAR + cal. vs the OLS",
        ),
        "SETopFiveOls": (
            f"{f5['ols_top5_share']:.0f}",
            "% of the OLS's total squared error on its 5 largest days",
        ),
        "SETopFiveMin": (
            f"{min(se_top5):.0f}",
            "min over the F5 forecasts (OLS, lasso, XGBoost; MinGRU parked): % of squared error on the 5 largest days",
        ),
        "SETopFiveMax": (f"{max(se_top5):.0f}", "max over the F5 forecasts"),
        "QLTopFiveMin": (
            f"{min(ql_top5):.0f}",
            "min over the F5 forecasts (OLS, lasso, XGBoost; MinGRU parked): % of total QLIKE on the 5 largest days",
        ),
        "QLTopFiveMax": (f"{max(ql_top5):.0f}", "max over the F5 forecasts"),
        "CumLassoTotal": (
            pm(cum[lasso]["total"], 2),
            "lasso, all features: sum over days of the QLIKE difference vs the OLS",
        ),
        "CumLassoWindow": (
            pm(cum[lasso]["window"], 2),
            "... of which in 24 Feb - 30 Apr 2020",
        ),
        "CumXgbTotal": (
            pm(cum[xgbk]["total"], 2),
            "XGBoost (daily refit), all features: sum over days",
        ),
        "CumXgbWindow": (
            pm(cum[xgbk]["window"], 2),
            "... of which in 24 Feb - 30 Apr 2020",
        ),
        "CumNnTotal": (
            pm(cum[nnk]["total"], 2),
            "MinGRU, all features: sum over days (parked: off F3, unused by the floats)",
        ),
        "CumNnWindow": (
            pm(cum[nnk]["window"], 2),
            "... of which in 24 Feb - 30 Apr 2020",
        ),
    }
    # 2026-10-15 (data audit #10): macros for the numbers the section prose (01_introduction, 06_results) typed by hand,
    # so the text can quote the actual-RV values; every one is computed above
    rb = pair(
        by["subpt_ridge_baseline"], by["poolpt_ridge_baseline"]
    )  # per-bar minus pooled, HAR + cal.
    lin_t2 = [T2[n]["baseline"] for n in ("Ridge", "Lasso", "Elastic net")]
    lin_t2a = [T2[n]["all_features"] for n in ("Ridge", "Lasso", "Elastic net")]
    tr_t2 = [T2[n]["baseline"] for n in ("LightGBM", "XGBoost", "Random forest")]
    tr_t2a = [T2[n]["all_features"] for n in ("LightGBM", "XGBoost", "Random forest")]
    ptq = [lv[f"pooltree_{e}_baseline"]["qlike"] for e in ("lgbm", "xgb", "rf")]
    nums.update(
        {
            "OlsSharpe": (
                f"{lv[OLS]['sharpe']:.2f}",
                "sign(s) Sharpe mid of the per-bar OLS on HAR + calendar",
            ),
            "LassoAllQ": (q4(lv[lasso]["qlike"]), "QLIKE, per-bar lasso, all features"),
            "XgbAllQ": (
                q4(lv[xgbk]["qlike"]),
                "QLIKE, per-bar XGBoost (daily refit), all features",
            ),
            "LgbmAllQ": (
                q4(lv["subtree_daily_all_features_lgbm"]["qlike"]),
                "QLIKE, per-bar LightGBM (daily refit), all features",
            ),
            "EnetAllQ": (
                q4(lv["subpt_enet_all_features"]["qlike"]),
                "QLIKE, per-bar elastic net, all features",
            ),
            "RidgeAllQ": (
                q4(lv["subpt_ridge_all_features"]["qlike"]),
                "QLIKE, per-bar ridge, all features",
            ),
            "NnAllQ": (
                q4(lv[nnk]["qlike"]),
                "QLIKE, per-bar MinGRU (ridge-init, QLIKE-tuned), all features",
            ),
            "OlsRank": (
                str(int(rank.loc[OLS, "rank"])),
                "rank of the per-bar OLS among the RankN forecasts",
            ),
            "NnAfPct": (
                pm(T1[("per-bar", NN_NAME)]["pct"], 1),
                "per-bar MinGRU: QLIKE % change, all features vs HAR + cal.",
            ),
            "NnAfP": (f"{T1[('per-bar', NN_NAME)]['p']:.2f}", "... its DM p"),
            "AfMedPct": (
                pm(float(af["pct"].median()), 1),
                "median QLIKE % change over the AfPairs pairs",
            ),
            "AfRevSig": (
                str(int(((af["pct"] > 0) & (af["p"] < SIG)).sum())),
                "pairs with QLIKE significantly higher with all features (DM p < 0.05)",
            ),
            "PerBarRidgeBasePct": (
                pm(rb["pct"], 1),
                "per-bar minus pooled ridge, HAR + cal.: QLIKE % change (of the pooled)",
            ),
            "PerBarRidgeBaseDm": (pm(rb["dm"], 2), "... its DM"),
            "LinBasePoolPctMin": (
                f"{min(x['pct'] for x in lin_t2):.1f}",
                "T2 linear, HAR + cal.: min pooled-minus-per-bar %",
            ),
            "LinBasePoolPctMax": (f"{max(x['pct'] for x in lin_t2):.1f}", "... max"),
            "LinBasePoolDmMin": (f"{min(x['dm'] for x in lin_t2):.2f}", "... min DM"),
            "LinBasePoolDmMax": (f"{max(x['dm'] for x in lin_t2):.2f}", "... max DM"),
            "LinAllPoolPctMin": (
                f"{min(x['pct'] for x in lin_t2a):.1f}",
                "T2 linear, all features: min %",
            ),
            "LinAllPoolPctMax": (f"{max(x['pct'] for x in lin_t2a):.1f}", "... max"),
            "TreeBasePoolPctMin": (
                f"{min(x['pct'] for x in tr_t2):.1f}",
                "T2 trees, HAR + cal.: min %",
            ),
            "TreeBasePoolPctMax": (f"{max(x['pct'] for x in tr_t2):.1f}", "... max"),
            "TreeAllPoolPctMin": (
                f"{min(x['pct'] for x in tr_t2a):.1f}",
                "T2 trees, all features: min %",
            ),
            "TreeAllPoolPctMax": (f"{max(x['pct'] for x in tr_t2a):.1f}", "... max"),
            "PoolTreeBaseQMin": (
                f"{min(ptq):.3f}",
                "QLIKE of the pooled trees on HAR + cal.: min",
            ),
            "PoolTreeBaseQMax": (f"{max(ptq):.3f}", "... max"),
        }
    )
    # 2026-10-15 (table legibility): the sample constants the table notes cite, and per-model macros against the HAR
    # benchmark (NOTATION_GUIDE 6b), so the prose can drop its typed values; every value is the one the tables print.
    d0, d1 = by[OLS].index[0], by[OLS].index[-1]
    assert (str(d0.date()), str(d1.date())) == ("2020-01-03", "2024-04-30"), (d0, d1)
    cs_ = rolling.common_span()
    assert (cs_[0], cs_[-1]) == (d0, d1), (
        cs_[0],
        cs_[-1],
    )  # 2026-10-15: the rolling span has the same ends
    nums.update(
        {
            # 2026-10-15: parked -- "first / last out-of-sample day"; the dates are unchanged (asserted above)
            "SampleStart": (
                f"{d0.day} {d0:%B %Y}",
                "first day of the option-data subsample (= of the rolling span)",
            ),
            "SampleEnd": (
                f"{d1.day} {d1:%B %Y}",
                "last day of the option-data subsample (= of the rolling span)",
            ),
            "TrVsRidgeN": (
                str(len(tr)),
                "tree forecasts compared with the closing-interval ridge, same predictors",
            ),
            "TrVsRidgeQAbove": (str(n_tr_above), "... with the higher average QLIKE"),
            "TrVsRidgeQAboveSig": (str(n_tr_above_sig), "... and DM p < 0.05"),
            "TrVsRidgeQBelowSig": (str(n_tr_below_sig), "... lower and DM p < 0.05"),
            "LassoMseExCrashPct": (
                mnum(lasso_mse_ex_pct, 1),
                "lasso (full) vs the HAR benchmark: MSE % change outside 24 Feb - 30 Apr 2020",
            ),
            "DMLag": (
                str(DM_LAG),
                "Newey-West lag q_NW of every DM statistic (dm_test's automatic lag)",
            ),
            "HACLag": (
                str(HAC_LAG),
                "Newey-West lag q_NW of every HAC t (floor(1.5 T^(1/3)))",
            ),
        }
    )
    tok = {
        "Ridge": "Ridge",
        "Lasso": "Lasso",
        "Elastic net": "Enet",
        "LightGBM": "Lgbm",
        "XGBoost": "Xgb",
        "Random forest": "Rf",
        NN_NAME: "Nn",
    }
    xtok = {"baseline": "Har", "all_features": "Full"}

    def trio(pre: str, x: dict, what: str) -> dict[str, tuple[str, str]]:
        return {
            f"{pre}Pct": (mnum(x["pct"], 1), f"{what}: QLIKE % change"),
            f"{pre}Dm": (mnum(x["dm"], 2), "... its DM"),
            f"{pre}Pval": (mpval(x["p"]), "... its two-sided p (the whole statement)"),
            f"{pre}Stars": (mstars(x["p"]), "... its stars (10 / 5 / 1 %)"),
        }

    m_q: dict[str, tuple[str, str]] = {}
    m_bm: dict[str, tuple[str, str]] = {}
    m_af: dict[str, tuple[str, str]] = {}
    m_pc: dict[str, tuple[str, str]] = {}
    # 2026-10-15 (user): NN content parked for now -- NN_ROW keeps its Nn macros defined (unused by the floats); its
    # forecasts are no longer in the ranking, so only the others are checked against the T3 rows
    for name, _, pb, pool in FAMILY_ROWS + [NN_ROW]:
        mt = tok[name]
        for scope, ks in (("Close", pb), ("Pool", pool)):
            for i, b in enumerate(("baseline", "all_features")):
                k = ks[i]
                vb = pair(by[k], by[OLS])
                if name == NN_NAME:
                    assert alias.get(k, k) not in rank.index, k
                else:
                    rk = rank.loc[alias.get(k, k)]  # = the T3 rows
                    assert (
                        abs(vb["pct"] - rk["pct"]) < 1e-12
                        and abs(vb["dm"] - rk["dm"]) < 1e-12
                    ), k
                pre = f"{mt}{scope}{xtok[b]}"
                m_q[f"Q{pre}"] = (q4(lv[k]["qlike"]), f"mean QLIKE, {k}")
                m_bm.update(
                    trio(
                        f"VsBm{pre}",
                        vb,
                        f"{k} vs the HAR benchmark (closing-interval OLS)",
                    )
                )
        for scope, fit in (("Close", "per-bar"), ("Pool", "pooled")):
            m_af.update(
                trio(
                    f"Af{mt}{scope}", T1[(fit, name)], f"T1, {fit} {name}: Full vs HAR"
                )
            )
        for b in ("baseline", "all_features"):
            m_pc.update(
                trio(
                    f"PoolVsClose{mt}{xtok[b]}",
                    T2[name][b],
                    f"T2, {name} on {xtok[b]}: pooled vs per-bar",
                )
            )
    # 2026-10-15 (user, restructure): per-model trade macros (the sign strategy on the option-data subsample), for the
    # pooled intraday body of Economic value and its last-30-minutes final subsection
    m_sh: dict[str, tuple[str, str]] = {}
    for name, _, pb, pool in FAMILY_ROWS + [NN_ROW]:
        mt = tok[name]
        for scope, ks in (("Close", pb), ("Pool", pool)):
            for i, b in enumerate(("baseline", "all_features")):
                k = ks[i]
                pre = f"{mt}{scope}{xtok[b]}"
                m_sh[f"Sh{pre}"] = (
                    mnum(lv[k]["sharpe"], 2),
                    f"sign-strategy Sharpe ratio, mid-quotes, {k}",
                )
                m_sh[f"ShX{pre}"] = (
                    mnum(lv[k]["sharpe_x"], 2),
                    "... at quoted bid/ask prices",
                )
                m_sh[f"VsShort{pre}T"] = (
                    mnum(vs_sh[k]["t"], 2),
                    "... HAC t of the daily mid-return differential over the short straddle",
                )
        for scope, fit in (("Close", "per-bar"), ("Pool", "pooled")):
            r_ = T1[(fit, name)]
            m_sh[f"Af{mt}{scope}DSh"] = (
                pm(r_["dsh"], 2),
                f"T1, {fit} {name}: Sharpe mid, Full minus HAR",
            )
            m_sh[f"Af{mt}{scope}DShT"] = (
                mnum(r_["t"], 2),
                "... HAC t of the daily mid-return differential",
            )
    # 2026-10-15 (user, restructure): pooled-only aggregates (asserted above) and the closing-interval T1 summary
    m_pool = {
        "PoolN": (
            str(len(pool_keys)),
            "intraday forecasts (T1 Panel B: 6 classes x HAR, Full)",
        ),
        "AfPoolPairs": (str(len(t1p)), "pooled Full vs HAR pairs (one per class)"),
        "AfPoolLower": (
            str(sum(r["pct"] < 0 for r in t1p.values())),
            "... QLIKE lower with Full",
        ),
        "AfPoolLowerSig": (
            str(sum(r["pct"] < 0 and r["p"] < SIG for r in t1p.values())),
            "... and DM p < 0.05",
        ),
        "AfPoolMseHigher": (
            str(sum(r["mse_pct"] > 0 for r in t1p.values())),
            "... MSE higher with Full",
        ),
        "AfPoolMseHigherSig": (
            str(sum(r["mse_pct"] > 0 and r["p_mse"] < SIG for r in t1p.values())),
            "... and DM (squared error) p < 0.05",
        ),
        "AfPoolShHigher": (
            str(sum(x > 0 for x in dsh_pool)),
            "... Sharpe mid higher with Full",
        ),
        "AfPoolShHigherSig": (
            str(sum(r["dsh"] > 0 and r["p_t"] < SIG for r in t1p.values())),
            "... and HAC t p < 0.05",
        ),
        "AfPoolDShMin": (
            pm(min(dsh_pool), 2),
            "... Sharpe mid change, Full minus HAR: min",
        ),
        "AfPoolDShMax": (pm(max(dsh_pool), 2), "... max"),
        "AfPoolDShMed": (pm(float(np.median(dsh_pool)), 2), "... median"),
        "AfClosePairs": (
            str(len(t1c)),
            "closing-interval Full vs HAR pairs (one per class)",
        ),
        "AfCloseLower": (
            str(sum(r["pct"] < 0 for r in t1c.values())),
            "... QLIKE lower with Full",
        ),
        "AfCloseLowerSig": (
            str(sum(r["pct"] < 0 and r["p"] < SIG for r in t1c.values())),
            "... and DM p < 0.05",
        ),
        "AfClosePctMin": (
            mnum(min(r["pct"] for r in t1c.values()), 1),
            "... QLIKE % change: min",
        ),
        "AfClosePctMax": (mnum(max(r["pct"] for r in t1c.values()), 1), "... max"),
        "PoolSETopFiveMin": (
            f"{min(se_pool):.0f}",
            "pooled forecasts on Full: % of total squared error on their 5 largest days (all in March 2020), min",
        ),
        "PoolSETopFiveMax": (f"{max(se_pool):.0f}", "... max"),
        "PoolQLTopFiveMin": (
            f"{min(ql_pool):.0f}",
            "... % of total QLIKE on those days, min",
        ),
        "PoolQLTopFiveMax": (f"{max(ql_pool):.0f}", "... max"),
        "PoolQLinFullMin": (
            q4(min(qs(lin_n, 1))),
            "mean QLIKE, pooled linear models on Full: min",
        ),
        "PoolQLinFullMax": (q4(max(qs(lin_n, 1))), "... max"),
        "PoolQTreeFullMin": (
            q4(min(qs(tree_n, 1))),
            "mean QLIKE, pooled trees on Full: min",
        ),
        "PoolQTreeFullMax": (q4(max(qs(tree_n, 1))), "... max"),
        "PoolQLinHarMin": (
            q4(min(qs(lin_n, 0))),
            "mean QLIKE, pooled linear models on HAR: min",
        ),
        "PoolQLinHarMax": (q4(max(qs(lin_n, 0))), "... max"),
        "PoolQTreeHarMin": (
            q4(min(qs(tree_n, 0))),
            "mean QLIKE, pooled trees on HAR: min",
        ),
        "PoolQTreeHarMax": (q4(max(qs(tree_n, 0))), "... max"),
        "VsBmPoolLinFullPctMin": (
            f"{min(vb_lin_full):.1f}",
            "pooled linear on Full vs the benchmark forecast: QLIKE % change, min (all > 0)",
        ),
        "VsBmPoolLinFullPctMax": (f"{max(vb_lin_full):.1f}", "... max"),
        "VsBmPoolTreeFullPctMin": (
            f"{min(vb_tree_full):.1f}",
            "pooled trees on Full vs the benchmark forecast: min",
        ),
        "VsBmPoolTreeFullPctMax": (f"{max(vb_tree_full):.1f}", "... max"),
        "VsBmPoolHarPctMin": (
            f"{min(vb_har):.1f}",
            "pooled models on HAR vs the benchmark forecast: min",
        ),
        "VsBmPoolHarPctMax": (f"{max(vb_har):.1f}", "... max"),
        "PoolShortAbove": (
            str(sum(vs_sh[k]["dsh"] > 0 for k in pool_keys)),
            "pooled forecasts whose sign strategy has a higher Sharpe mid than the short straddle",
        ),
        "PoolShortSig": (
            str(len(pool_sig_short)),
            "... with HAC t > 1.96 (the three trees on Full)",
        ),
        "PoolShHarMin": (
            mnum(min(sh_har), 2),
            "Sharpe mid of the pooled forecasts on HAR: min",
        ),
        "PoolShHarMax": (mnum(max(sh_har), 2), "... max"),
        "PoolShFullMin": (
            mnum(min(sh_full), 2),
            "Sharpe mid of the pooled forecasts on Full: min",
        ),
        "PoolShFullMax": (mnum(max(sh_full), 2), "... max"),
        "PoolSzSignMed": (
            f"{med_sz('sign'):.2f}",
            "sizing sign: median Sharpe mid over the pooled forecasts, 614 days",
        ),
        "PoolSzClipMed": (f"{med_sz('clip'):.2f}", "sizing robust clip: median"),
        "PoolSzRankMed": (f"{med_sz('rank'):.2f}", "sizing rank of |gap|: median"),
        "PoolSzClipBeats": (
            str(sum(sz_pool[k]["clip"]["d_vs_sign"] > 0 for k in pool_keys)),
            "pooled forecasts with clip Sharpe mid above sign",
        ),
        "PoolTailNeg": (
            str(sum(vs_sh[k]["tail"] < 0 for k in pool_keys)),
            "pooled forecasts whose gain over the short straddle is < 0 without their 10 best days",
        ),
    }
    for fam_ in (m_q, m_bm, m_af, m_pc, m_sh, m_pool):
        assert not set(fam_) & set(nums), set(fam_) & set(nums)
        nums.update(fam_)
    # 2026-10-15 (table legibility): parked -- the header now lists the macro families (NOTATION_GUIDE 6b)
    # lines = [
    #     "% AUTO-GENERATED by writeup/make_rehaul_results.py -- do not edit.",
    #     "% Numbers for the captions of writeup/rehaul/06_results_floats.tex (plain f^2 B, 866 days, DM lag 6 HLN).",
    # ]
    lines = [
        "% AUTO-GENERATED by writeup/make_rehaul_results.py -- do not edit. Macro definitions only (no tables); loaded",
        # 2026-10-15 (user): parked -- accuracy over every rolling forecast. Was: "Every QLIKE is against the actual
        # realized variance of the closing half hour on the \\rhrDays out-of-sample trading days; DM: dm_test, ..."
        "% once by the preamble of paper_rehaul.tex. Every QLIKE, squared error and DM is against the actual realized",
        (
            "% variance of the closing half hour on the common rolling span (\\rhoosDays trading days, \\rhoosSpanStart"
            " to \\rhoosSpanEnd); every Sharpe ratio and HAC t on the \\rhrDays days of the option-data subsample."
            if SPAN == "rolling"
            else "% variance of the closing half hour. mt866 CHECK RUN: every number on the 866 option-data days."
        ),
        "% DM: dm_test, HLN-corrected, Newey-West lag \\rhrDMLag;",
        "% HAC t: newey_west_t, lag \\rhrHACLag; p two-sided normal; stars at the 10 / 5 / 1 % levels.",
        # 2026-10-15 (user): parked -- NN content parked for now
        # "% Per-model families (<M> = Ridge, Lasso, Enet, Lgbm, Xgb, Rf, Nn; <S> = Close | Pool, the closing-interval or",
        "% Per-model families (<M> = Ridge, Lasso, Enet, Lgbm, Xgb, Rf; Nn, the MinGRU, is parked from every table and",
        "% figure (2026-10-15) but its macros stay defined; <S> = Close | Pool, the closing-interval or",
        "% intraday model; <X> = Har | Full, the predictor set):",
        "%   \\rhrQ<M><S><X>                             mean QLIKE;",
        "%   \\rhrVsBm<M><S><X>{Pct,Dm,Pval,Stars}       against the HAR benchmark, the closing-interval OLS on HAR (T3);",
        "%   \\rhrAf<M><S>{Pct,Dm,Pval,Stars}            full predictor set against HAR, same model (T1);",
        "%   \\rhrPoolVsClose<M><X>{Pct,Dm,Pval,Stars}   intraday against closing-interval, same class (T2);",
        # 2026-10-15 (user, restructure): the trade families (option-data subsample)
        "%   \\rhrSh<M><S><X>, \\rhrShX<M><S><X>           sign-strategy Sharpe ratio at mid-quotes, at bid/ask;",
        "%   \\rhrVsShort<M><S><X>T                       HAC t of its daily return differential over the short straddle;",
        "%   \\rhrAf<M><S>{DSh,DShT}                      Sharpe ratio (mid), Full minus HAR, and its HAC t.",
        "% Pooled-only aggregates (\\rhrPool..., \\rhrAfPool..., \\rhrVsBmPool...) and \\rhrAfClose...: below.",
        "% Pct = 100 (Lbar_A / Lbar_B - 1), 1 decimal, negative favors A (the first-named forecast); Dm: 2 decimals;",
        "% Pval: the whole statement, $p = 0.18$ or $p < 0.001$; Stars: superscript stars, empty if not significant.",
    ]
    for k, (v, c) in nums.items():
        lines.append(f"\\providecommand{{\\rhr{k}}}{{{v}}}  % {c}")
    (GEN / "rehaul_results_numbers.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote generated/rehaul_results_numbers.tex ({len(nums)} macros)")
    for k, (v, c) in nums.items():
        print(f"  {k:16s} {v:28s} {c}")


if __name__ == "__main__":
    main()
