"""Table 1 (tab:rh_oos) from the pooled linear reruns, like for like, scored on every interval (2026-10-20).

User (2026-10-20): "we need to straighten this out so it's like for like"; "we need to evaluate on all 48 periods when
I say it's 48 periods"; "do tests for 13 interval and redo the table. trees can be held off for now because we need
clusters for that".

The table it writes replaces the table of writeup/make_rehaul_oos_table.py (which scored the 16:00 row of runs with
different windows, HAR horizon sets and data vintages; that generator now writes only Table 2, tab:rh_oos_last30).

ROWS     the OLS, ridge, lasso and elastic net of the two pooled linear reruns of 19-20 October 2026, one design:
         base-5 HAR horizons (1, 5, 25, 125, 625, 3125 intervals), the four FOMC series in the full predictor set, no
         trailing averages of the activity indicators (E6), the calendar cut at the overnight indicator (E9), every
         data-fix flag on, penalties chosen once on the pre-sample and frozen, a 2,000-trading-day window, re-estimated
         every interval.
           Panel A  trained and scored on all 48 half-hour intervals of the day
                    results/pooled_linear_rerun_2026-10-19_48/forecasts/<model>_<set>.parquet
           Panel B  trained and scored on the 13 regular-hours bars (end stamps 10:00 .. 16:00 ET)
                    results/pooled_linear_rerun_2026-10-19_13/forecasts.parquet
         Tree ensembles: held off until they are refit on the same design (cluster).
SCORED   every interval of the panel's sample on the 1,082 trading days of the evaluation span (3 January 2020 ..
         30 April 2024, the NYSE trading days with a positive closing-half-hour print, early closes dropped), by the
         calendar date of the interval end (00:00 .. 23:30 ET in Panel A); intervals with a non-positive or missing RV
         and any interval a model of the panel does not forecast are dropped, so every cell of a panel has the same
         intervals.
FORECAST sigma2 = yhat^2 x s^2 (the files' "plain"); TRUTH: the interval's actual RV, the sumret2 of
         data/core_stats.parquet (asserted equal to the files' rv / rv_actual).
METRICS  MSE (x 10^MSE_EXP), QLIKE = mean of r - ln r - 1 with r = RV / sigma2, both averaged over every scored
         interval; R2_OOS = 1 - sum (RV - sigma2)^2 / sum (RV - RVbar)^2 with RVbar the historical mean RV of the same
         time-of-day interval over every earlier day from 5 January 1998 (expanding, causal; early-close days and
         non-positive prints left out of the history), the Campbell-Thompson benchmark taken interval by interval.
TESTS    Diebold-Mariano on the daily average loss of each trading day (the intervals of a day are correlated, so the
         test runs on the 1,082 daily averages), src/evaluation/diebold_mariano.dm_test: Harvey-Leybourne-Newbold
         correction, Newey-West lag max(h - 1, its automatic bandwidth). The table prints the QLIKE test of the full
         predictor set against the HAR benchmark of the same model (negative: the full set has the lower loss).
         Tests against the OLS and the MSE tests are written to the CSV only.
CHECK    the 16:00 rows reproduce each rerun's own closing-interval QLIKE (its scores.csv) to 1e-10.
TABLE    2026-10-20 (user): "don't change the table so much. there shouldn't be 2 panels on it. just keep the 13 session
         panel on it" -- Table 1 prints the 13-bar sample only, in the earlier layout (a row a model, the OLS first,
         the predictor sets as column groups) with a DM column; both samples stay in metrics.csv and dm.csv.

WRITES   writeup/generated/rehaul_results_oos.tex           Table 1 (tab:rh_oos)
         writeup/generated/rehaul_oos_rerun_numbers.tex     \\rhor... macros (counts, tests), loaded by the preamble
         results/oos_rerun_2026-10-20/metrics.csv, dm.csv   every number of the table and the extra tests

SPAN     2026-10-20 (user): "switch the main window to 1,000 days, not 2,000. Table 1 should cover 2012-2024 as a
         single panel, with no 2012-2019 / 2020-2024 split table." The default (--span 2012) builds Table 1 from
         results/pooled_linear_rerun_2026-10-20_13_2012_w1000 (same design, a 1,000-trading-day window, penalties chosen
         once on the four validation folds 2008-2011 and frozen) on the 3,075 trading days 3 January 2012 .. 30 April
         2024, 13-bar sample only (there is no 48-interval rerun from 2012, so no Panel A macros). The 16:00 check runs
         on the 1,082 days 2020-01-03 .. 2024-04-30 that the run's own scores.csv scores. CSVs go to
         results/oos_span_check_2026-10-20/table1_2012_w1000/. "--span 2020" writes the earlier table (the 2,000-day
         reruns from 2020 above, both panels, CSVs in results/oos_rerun_2026-10-20/).
         2026-10-20 (user): parked -- the 2,000-day 2012 table with subperiod macros. Was: "--span 2012" read
         results/pooled_linear_rerun_2026-10-20_13_2012_w2000 and also wrote \\rhorSubOne... / \\rhorSubTwo...
         macros (2012-2019, 2020-2024); "--span 2020" was the default.

TREES    2026-10-20 (user): the pooled LightGBM, XGBoost and random forest of experiments/pooled_trees_rerun_13.py
         (same design and 1,000-day window, refit at every bar, hyperparameters chosen once on the 1998-2002
         pre-sample and frozen: cluster/pooltree_lgbm_config.json, xgbtop_xgb_config.json, pooltree_rf400_config.json)
         join Table 1 with --trees PATH (a forecasts.parquet, or the directory holding it), --span 2012 only. Off by
         default: without it every file this script writes is unchanged. With it:
           - three rows after the elastic net (LightGBM, XGBoost, Random forest), both predictor sets and the DM column;
           - the scored bars are those of the 7 linear and the 6 tree series together; the script stops, with the
             count and the first bars, if the trees drop any bar the linear rows score, so the linear rows (and every
             \\rhor... macro of the linear models) keep their numbers;
           - \\rhorTree... macros of the tree rows; the subperiod macros stay the linear models' only;
           - the notes describe the trees in place of "Tree ensembles are left out ...";
           - the CSVs go to <the span's CSV directory>/with_trees/ unless --res-dir says otherwise.
         The 16:00 check against a scores.csv runs on the linear rows only (the tree run has no scores.csv).
         Test without the real forecasts: experiments/test_table1_trees_standin.py.

Run:  python writeup/make_rehaul_oos_rerun.py                 (2012-2024, 1,000-day window)
      python writeup/make_rehaul_oos_rerun.py --span 2020     (the earlier 2020-2024 table; --out-dir writes elsewhere)
      python writeup/make_rehaul_oos_rerun.py --trees results/pooled_trees_rerun13_2026-10-20/w1000
                                                              (Table 1 with the pooled tree rows)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent  # writeup/
REPO = ROOT.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "notebooks"))
import atm_straddle_lib as asl  # noqa: E402
from src.evaluation.diebold_mariano import dm_test  # noqa: E402

GEN = ROOT / "generated"
OUT_TABLE = GEN / "rehaul_results_oos.tex"
OUT_NUMBERS = GEN / "rehaul_oos_rerun_numbers.tex"
OUT_RES = REPO / "results" / "oos_rerun_2026-10-20"
CORE = REPO / "data" / "core_stats.parquet"
R48 = REPO / "results" / "pooled_linear_rerun_2026-10-19_48"
R13 = REPO / "results" / "pooled_linear_rerun_2026-10-19_13"
FIRST_DAY, LAST_DAY, SPAN_DAYS = pd.Timestamp("2020-01-03"), pd.Timestamp("2024-04-30"), 1082
HIST_START = pd.Timestamp("1998-01-05")
MSE_EXP = 10
HEAD = "% AUTO-GENERATED by writeup/make_rehaul_oos_rerun.py -- do not edit."
EARLY = pd.DatetimeIndex(pd.to_datetime(list(asl.EARLY_CLOSE_DATES)))

MODELS = [("ols", "OLS"), ("ridge", "Ridge"), ("lasso", "Lasso"), ("enet", "Elastic net")]
SETS = [("baseline", "HAR benchmark"), ("all_features", "All features")]
# the pooled trees of --trees (model code, row label, macro tag)
TREES = [("lgbm", "LightGBM", "Lgbm"), ("xgb", "XGBoost", "Xgb"), ("rf", "Random forest", "Rf")]
PANELS = [
    ("48", "Panel A: Trained and scored on all 48 half-hour intervals"),
    ("13", "Panel B: Trained and scored on the 13 regular-hours bars (10:00--16:00)"),
]
# 2026-10-20 (user): parked -- the 2,000-day 2012 run. Was: R13_2012 = ... / "pooled_linear_rerun_2026-10-20_13_2012_w2000"
R13_2012 = REPO / "results" / "pooled_linear_rerun_2026-10-20_13_2012_w1000"
# span -> 13-bar run, panels, first and last evaluation day, day count, the days of the run's own 16:00 scores.csv
# (first, last, count), the CSV directory, the rolling window in trading days, and the subperiods of the
# \rhorSub... macros (tag, first, last, days)
SPANS = {
    "2020": dict(
        r13=R13, panels=["48", "13"], first=FIRST_DAY, last=LAST_DAY, n_days=SPAN_DAYS,
        check=(FIRST_DAY, LAST_DAY, SPAN_DAYS), res=OUT_RES, window=2000, subs=[],
    ),
    "2012": dict(
        r13=R13_2012, panels=["13"], first=pd.Timestamp("2012-01-03"), last=LAST_DAY, n_days=3075,
        check=(FIRST_DAY, LAST_DAY, SPAN_DAYS),
        res=REPO / "results" / "oos_span_check_2026-10-20" / "table1_2012_w1000", window=1000,
        # 2026-10-20 (user): parked -- "no 2012-2019 / 2020-2024 split table". Was: subs=[("One", 2012-01-03,
        # 2019-12-31, 1992 days), ("Two", 2020-01-03, 2024-04-30, 1082 days)]
        # 2026-10-20 (user): "add the one-sentence 2012-2019 / 2020-2024 split in the text (no table)" -- the split is
        # back as macros for that sentence only. Was: subs=[]
        subs=[
            ("A", pd.Timestamp("2012-01-03"), pd.Timestamp("2019-12-31"), 1992),
            ("B", FIRST_DAY, LAST_DAY, SPAN_DAYS),
        ],
    ),
}


# --------------------------------------------------------------------------- data


def core() -> pd.DataFrame:
    c = pd.read_parquet(CORE, columns=["endbartime", "sumret2"])
    c["endbartime"] = pd.to_datetime(c["endbartime"])
    assert c["endbartime"].dt.tz is None, "core_stats stamps are naive ET"
    assert not c["endbartime"].duplicated().any()
    return c.set_index("endbartime")["sumret2"].sort_index()


def evaluation_days(rv: pd.Series, first=FIRST_DAY, last=LAST_DAY, n=SPAN_DAYS) -> pd.DatetimeIndex:
    """The 1,082 trading days of the span: a positive closing-half-hour print, early closes dropped."""
    s = rv[(rv.index.hour == 16) & (rv.index.minute == 0)]
    d = pd.DatetimeIndex(s.index.normalize())
    keep = (s.to_numpy() > 0) & np.isfinite(s.to_numpy()) & ~d.isin(EARLY)
    days = d[keep]
    days = days[(days >= first) & (days <= last)]
    assert len(days) == n and days[0] == first and days[-1] == last, (len(days), days[0], days[-1])
    return days


def forecasts(panel: str, r13: Path = R13) -> dict[tuple[str, str], pd.DataFrame]:
    """(model, set) -> frame indexed by interval end with plain (forecast) and rv (truth)."""
    out = {}
    if panel == "48":
        for m, _ in MODELS:
            for s, _ in SETS:
                p = R48 / "forecasts" / f"{m}_{s}.parquet"
                if not p.is_file():
                    continue
                f = pd.read_parquet(p, columns=["date", "plain", "rv"])
                out[(m, s)] = f.set_index(pd.to_datetime(f["date"]))[["plain", "rv"]]
    else:
        g = pd.read_parquet(r13 / "forecasts.parquet", columns=["model", "predictor_set", "date", "plain", "rv_actual"])
        for (m, s), f in g.groupby(["model", "predictor_set"]):
            out[(m, s)] = f.set_index(pd.to_datetime(f["date"])).rename(columns={"rv_actual": "rv"})[["plain", "rv"]]
    assert set(out) == {(m, s) for m, _ in MODELS for s, _ in SETS if m != "ols" or s == "baseline"}, sorted(out)
    for k, f in out.items():
        assert not f.index.duplicated().any(), k
    return out


def tree_forecasts(path: Path) -> dict[tuple[str, str], pd.DataFrame]:
    """--trees: (model, set) -> frame indexed by bar end with plain and rv, the 6 tree series."""
    if path.is_dir():
        path = path / "forecasts.parquet"
    g = pd.read_parquet(path, columns=["model", "predictor_set", "date", "plain", "rv_actual"])
    out = {}
    for (m, s), f in g.groupby(["model", "predictor_set"]):
        out[(m, s)] = f.set_index(pd.to_datetime(f["date"])).rename(columns={"rv_actual": "rv"})[["plain", "rv"]]
    assert set(out) == {(m, s) for m, _, _ in TREES for s, _ in SETS}, sorted(out)
    for k, f in out.items():
        assert not f.index.duplicated().any(), k
    return out


def scored_set(fc: dict, rv: pd.Series, days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Interval ends scored in a panel: on an evaluation day, a positive RV, and a finite positive forecast of every
    model of the panel."""
    idx = None
    for f in fc.values():
        ok = f.index[np.isfinite(f["plain"].to_numpy()) & (f["plain"].to_numpy() > 0)]
        idx = ok if idx is None else idx.intersection(ok)
    idx = idx[pd.DatetimeIndex(idx.normalize()).isin(days)]
    t = rv.reindex(idx)
    idx = idx[(t.to_numpy() > 0) & np.isfinite(t.to_numpy())]
    for k, f in fc.items():  # the files' truth is the core_stats print
        assert np.array_equal(f["rv"].reindex(idx).to_numpy(), rv.reindex(idx).to_numpy()), k
    return idx.sort_values()


def slot_benchmark(rv: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    """RVbar of each scored interval: the mean RV of the same clock time over every earlier day from 1998-01-05
    (positive prints, early-close days out), causal by construction."""
    h = rv[(rv.index >= HIST_START) & (rv.to_numpy() > 0) & np.isfinite(rv.to_numpy())]
    h = h[~pd.DatetimeIndex(h.index.normalize()).isin(EARLY)]
    frame = pd.DataFrame({"rv": h.to_numpy(), "slot": h.index.strftime("%H:%M"), "t": h.index})
    frame["csum"] = frame.groupby("slot")["rv"].cumsum() - frame["rv"]  # earlier days only
    frame["n"] = frame.groupby("slot").cumcount()
    frame["bench"] = frame["csum"] / frame["n"].where(frame["n"] > 0)
    b = frame.set_index("t")["bench"]
    # a scored interval need not be a history interval (early closes are not scored either); take the last earlier
    # history value of its slot and check a sample by brute force
    out = {}
    for slot, g in frame.groupby("slot"):
        want = idx[idx.strftime("%H:%M") == slot]
        if len(want):
            gg = g.set_index("t")
            pos = gg.index.searchsorted(want, side="left")  # first history row at or after t: its prior mean
            means = []
            for t, p in zip(want, pos):
                csum = gg["rv"].iloc[:p].sum()
                means.append(csum / p if p > 0 else np.nan)
            out.update(dict(zip(want, means)))
    s = pd.Series(out).reindex(idx)
    h_slot, h_day = h.index.strftime("%H:%M"), h.index.normalize()
    for t in idx[:: max(1, len(idx) // 200)]:  # brute-force spot check
        prior = h[(h_slot == t.strftime("%H:%M")) & (h_day < t.normalize())]
        assert np.isclose(s[t], prior.mean(), rtol=1e-10, atol=0), t
    assert s.notna().all()
    del b
    return s


# --------------------------------------------------------------------------- statistics


def loss(f: pd.DataFrame, idx: pd.DatetimeIndex, rv: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    p, y = f["plain"].reindex(idx).to_numpy(float), rv.reindex(idx).to_numpy(float)
    r = y / p
    return r - np.log(r) - 1.0, (y - p) ** 2


def daily(x: np.ndarray, idx: pd.DatetimeIndex) -> np.ndarray:
    return pd.Series(x, index=pd.DatetimeIndex(idx.normalize())).groupby(level=0).mean().to_numpy()


def fmt(x: float, dp: int) -> str:
    v = round(float(x), dp)
    if v == 0:
        v = 0.0
    return f"{'$-$' if v < 0 else ''}{abs(v):.{dp}f}"


def fmt_p(p: float) -> str:
    """A p-value to three decimals, with values below 0.001 written as <0.001 (text or math mode)."""
    return "\\ensuremath{<}0.001" if p < 0.001 else f"{p:.3f}"


def thousands(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


# --------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    # 2026-10-20 (user): parked -- the 2020 default. Was: default="2020"
    ap.add_argument("--span", choices=sorted(SPANS), default="2012", help="evaluation span of Table 1 (default 2012)")
    ap.add_argument("--out-dir", type=Path, default=GEN, help="directory of the two .tex files (default generated/)")
    ap.add_argument("--trees", type=Path, default=None,
                    help="pooled tree forecasts.parquet (or its directory): adds the tree rows (default: off)")
    ap.add_argument("--res-dir", type=Path, default=None,
                    help="directory of the CSVs (default: the span's; <it>/with_trees/ under --trees)")
    args = ap.parse_args()
    cfg = SPANS[args.span]
    assert args.trees is None or args.span == "2012", "--trees: the tree rerun is on the 2012 span (1,000-day window)"
    out_table, out_numbers, out_res = args.out_dir / OUT_TABLE.name, args.out_dir / OUT_NUMBERS.name, cfg["res"]
    if args.res_dir is not None:
        out_res = args.res_dir
    elif args.trees is not None:
        out_res = cfg["res"] / "with_trees"
    tf = tree_forecasts(args.trees) if args.trees is not None else {}
    r13 = cfg["r13"]
    rv = core()
    days = evaluation_days(rv, cfg["first"], cfg["last"], cfg["n_days"])
    ck_first, ck_last, ck_days = cfg["check"]
    met_rows, dm_rows, panels, sub_rows = [], [], {}, []
    for pid, _ in PANELS:
        if pid not in cfg["panels"]:
            continue
        fc = forecasts(pid, r13)
        idx = scored_set(fc, rv, days)
        if tf and pid == "13":  # one set of bars for the 7 linear and the 6 tree series
            idx_all = scored_set({**fc, **tf}, rv, days)
            if not idx_all.equals(idx):
                drop = idx.difference(idx_all)
                raise SystemExit(
                    f"--trees: the tree forecasts drop {len(drop):,} of the {len(idx):,} bars the linear rows score "
                    f"(first: {', '.join(str(t) for t in drop[:5])}); the linear rows would change"
                )
        bench = slot_benchmark(rv, idx)
        y = rv.reindex(idx).to_numpy(float)
        den = float(((y - bench.to_numpy()) ** 2).sum())
        L = {}
        for k, f in fc.items():
            q, se = loss(f, idx, rv)
            L[k] = (q, se)
            # check: the 16:00 rows reproduce the rerun's own closing-interval QLIKE
            # (on the days that scores.csv scores: the whole span by default, 2020-01-03 .. 2024-04-30 under --span 2012)
            on16 = idx[(idx.hour == 16) & (idx.minute == 0)]
            on16 = on16[(on16.normalize() >= ck_first) & (on16.normalize() <= ck_last)]
            q16, _ = loss(f, on16, rv)
            sc = pd.read_csv((R48 if pid == "48" else r13) / "scores.csv")
            setcol = "set" if "set" in sc.columns else "predictor_set"
            ref = sc[(sc["model"] == k[0]) & (sc[setcol] == k[1])]["qlike"].iloc[0]
            assert len(on16) == ck_days and np.isclose(q16.mean(), ref, rtol=1e-10, atol=0), (pid, k, q16.mean(), ref)
            met_rows.append(
                dict(
                    panel=pid,
                    model=k[0],
                    set=k[1],
                    n_intervals=len(idx),
                    n_days=len(days),
                    mse=float(se.mean()),
                    qlike=float(q.mean()),
                    r2_oos_pct=float(100.0 * (1.0 - se.sum() / den)),
                    qlike_1600=float(q16.mean()),
                )
            )
        if pid == "13":  # the tree rows, on the same bars (no scores.csv to check against)
            for k, f in tf.items():
                q, se = loss(f, idx, rv)
                L[k] = (q, se)
                on16 = idx[(idx.hour == 16) & (idx.minute == 0)]
                on16 = on16[(on16.normalize() >= ck_first) & (on16.normalize() <= ck_last)]
                q16, _ = loss(f, on16, rv)
                met_rows.append(
                    dict(
                        panel=pid, model=k[0], set=k[1], n_intervals=len(idx), n_days=len(days),
                        mse=float(se.mean()), qlike=float(q.mean()),
                        r2_oos_pct=float(100.0 * (1.0 - se.sum() / den)), qlike_1600=float(q16.mean()),
                    )
                )
        for m, _ in MODELS + [(m, n) for m, n, _ in TREES if pid == "13" and tf]:  # tests on the daily average loss
            pairs = []
            if m != "ols":
                pairs.append(((m, "all_features"), (m, "baseline"), "all features vs HAR benchmark"))
            for s, _ in SETS:
                if (m, s) in L and m != "ols":
                    pairs.append(((m, s), ("ols", "baseline"), "vs OLS"))
            for a, b, what in pairs:
                for li, lname in ((0, "qlike"), (1, "mse")):
                    r = dm_test(daily(L[a][li], idx), daily(L[b][li], idx))
                    dm_rows.append(
                        dict(
                            panel=pid, model=a[0], set=a[1], against=f"{b[0]}_{b[1]}", comparison=what, loss=lname,
                            dm=float(r["dm"]), p=float(r["p"]), mean_diff=float(r["mean_diff"]),
                            T=int(r["T"]), hac_lag=int(r["hac_lag"]),
                        )
                    )
        for tag, s_first, s_last, s_days in cfg["subs"]:  # subperiods: the same scored bars, cut by date
            sidx = idx[(idx.normalize() >= s_first) & (idx.normalize() <= s_last)]
            n_sd = pd.DatetimeIndex(sidx.normalize()).nunique()
            assert n_sd == s_days, (tag, n_sd)
            sl = idx.isin(sidx)
            den_s = float(((y[sl] - bench.to_numpy()[sl]) ** 2).sum())  # R2_OOS of the subperiod, same benchmark
            for k in fc:
                sub_rows.append(dict(
                    subperiod=f"{s_first.date()}:{s_last.date()}", tag=tag, model=k[0], set=k[1], against="",
                    n_intervals=len(sidx), n_days=n_sd, qlike=float(L[k][0][sl].mean()),
                    mse=float(L[k][1][sl].mean()), r2_oos_pct=float(100.0 * (1.0 - L[k][1][sl].sum() / den_s)),
                    dm=np.nan, p=np.nan, hac_lag=np.nan,
                ))
            for m, _ in MODELS[1:]:
                r = dm_test(daily(L[(m, "all_features")][0][sl], sidx), daily(L[(m, "baseline")][0][sl], sidx))
                sub_rows.append(dict(
                    subperiod=f"{s_first.date()}:{s_last.date()}", tag=tag, model=m, set="all_features",
                    against=f"{m}_baseline", n_intervals=len(sidx), n_days=n_sd, qlike=np.nan, mse=np.nan,
                    r2_oos_pct=np.nan, dm=float(r["dm"]), p=float(r["p"]), hac_lag=int(r["hac_lag"]),
                ))
        panels[pid] = (idx, days)
    met = pd.DataFrame(met_rows)
    dms = pd.DataFrame(dm_rows)
    subs = pd.DataFrame(sub_rows)
    out_res.mkdir(parents=True, exist_ok=True)
    met.to_csv(out_res / "metrics.csv", index=False)
    dms.to_csv(out_res / "dm.csv", index=False)
    if len(subs):
        subs.to_csv(out_res / "subperiods.csv", index=False)
    span12 = args.span == "2012"

    # ------------------------------------------------------------------ the table
    def cell(pid, m, s):
        r = met[(met.panel == pid) & (met.model == m) & (met.set == s)]
        if r.empty:
            return ["--", "--", "--"]
        r = r.iloc[0]
        return [fmt(r.mse * 10.0**MSE_EXP, 2), f"{r.qlike:.4f}", fmt(r.r2_oos_pct, 1)]

    def test(pid, m):
        r = dms[(dms.panel == pid) & (dms.model == m) & (dms.comparison == "all features vs HAR benchmark")
                & (dms.loss == "qlike")]
        if r.empty:
            return "--"
        r = r.iloc[0]
        return f"{fmt(r.dm, 2)} ({fmt_p(r.p)})"

    # 2026-10-20 (user): parked -- one panel a sample ("there shouldn't be 2 panels on it"). Was: a "Panel A ... 48"
    # and a "Panel B ... 13" block, each with its OLS, ridge, lasso and elastic net rows.
    body = []
    for m, name in MODELS:
        line = [name] + cell("13", m, "baseline") + cell("13", m, "all_features") + [test("13", m)]
        body.append(" & ".join(line) + " \\\\")
        if m == "ols":
            body.append("\\addlinespace[3pt]")
    if tf:
        body.append("\\addlinespace[3pt]")
        for m, name, _ in TREES:
            line = [name] + cell("13", m, "baseline") + cell("13", m, "all_features") + [test("13", m)]
            body.append(" & ".join(line) + " \\\\")
    lag = sorted(set(dms["hac_lag"]))
    # span-dependent text; the defaults are the 2020 table as committed
    top = [
        "% tab:rh_oos: the pooled linear rerun on the 13 regular-hours bars (base 5, FOMC in, frozen penalties),",
        "% every bar scored on the 1,082 days of the span; Diebold-Mariano on daily average QLIKE (user 2026-10-20).",
    ]
    caption = "Forecast accuracy of the intraday models"
    pen = "penalties chosen once on the pre-sample and then held fixed"
    span_txt = ("Every bar is scored on the $T_{\\mathrm{oos}} = \\rhoosDays$ trading days from \\rhoosSpanStart{} to "
                "\\rhoosSpanEnd{} (early closes dropped)")
    if span12:
        top = [
            "% tab:rh_oos, 2012 span (--span 2012): the pooled linear rerun from 2012 on the 13 regular-hours bars (base 5,",
            f"% FOMC in, {cfg['window']:,}-day window, penalties chosen on the folds 2008-2011 and frozen), every bar "
            "scored on the",
            f"% {len(days):,} days 2012-01-03 .. 2024-04-30; Diebold-Mariano on daily average QLIKE.",
        ]
        caption = ("Forecast accuracy of the intraday models, \\rhorSpanStart{} to \\rhorSpanEnd{} "
                   f"({thousands(cfg['window'])}-day window, penalties frozen from 2008--2011)")
        pen = ("penalties chosen once on four validation folds covering 2008--2011, before the first forecast, "
               "and then frozen")
        span_txt = ("Every bar is scored on the $T_{\\mathrm{oos}} = \\rhorDays$ trading days from \\rhorSpanStart{} to "
                    "\\rhorSpanEnd{} (early closes dropped)")
    models_txt = "Pooled intraday linear models"
    trees_txt = "Tree ensembles are left out until they are re-estimated on this design."
    if tf:
        top.append("% With the pooled trees (--trees): LightGBM, XGBoost and random forest on the same design and window.")
        models_txt = "Pooled intraday linear models and tree ensembles"
        # 2026-10-20 (user): parked -- under --trees. Was: "Tree ensembles are left out until they are re-estimated
        # on this design."
        trees_txt = (f"LightGBM, XGBoost and the random forest use the same design and the same "
                     f"{thousands(cfg['window'])}-trading-day window, are refit at every bar, and keep hyperparameters "
                     "chosen once on the 1998--2002 pre-sample and then frozen.")
    table = "\n".join(
        [
            HEAD,
            *top,
            "\\begin{table}[htbp]",
            "\\centering",
            "\\small",
            # --trees: the "Random forest" label widens the first column (18pt over \linewidth at 6pt)
            *(["\\setlength{\\tabcolsep}{4.5pt}"] if tf else []),
            "\\begin{threeparttable}",
            f"\\caption{{{caption}}}",
            "\\label{tab:rh_oos}",
            "\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}lrrrrrrr@{}}",
            "\\toprule",
            " & \\multicolumn{3}{c}{HAR benchmark} & \\multicolumn{3}{c}{All features} & \\\\",
            "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
            "Model & MSE ($10^{-\\rhoosMseExp}$) & QLIKE & $R^2_{\\mathrm{OOS}}$ (\\%) & MSE ($10^{-\\rhoosMseExp}$) & "
            "QLIKE & $R^2_{\\mathrm{OOS}}$ (\\%) & DM \\\\",
            "\\midrule",
            *body,
            "\\bottomrule",
            "\\end{tabular*}",
            "\\begin{tablenotes}[flushleft]",
            "\\footnotesize",
            f"\\item \\textit{{Notes:}} {models_txt}, one design for every row: the HAR components at "
            "horizons $k \\in \\{1, 5, 25, 125, 625, 3125\\}$, the full predictor set with the FOMC series, "
            f"{pen}, a rolling window of {thousands(cfg['window'])} trading days, "
            "re-estimated every interval (Section~\\ref{sec:rh_methods}), trained and scored on the 13 regular-hours "
            f"half-hour bars ending 10:00 to 16:00~ET. {span_txt}, \\rhorNIntB{{}} bars "
            "(bars with a non-positive realized variance dropped); each forecast $\\hat\\sigma^2_{t,i} = \\hat y^{2}_{t,i}\\, s^2_{t,i}$ is "
            "scored against the actual realized variance $\\RV_{t,i}$ of its interval. MSE (in units of "
            "$10^{-\\rhoosMseExp}$) and QLIKE are averages over the scored intervals. "
            "$R^2_{\\mathrm{OOS}} = 1 - \\sum (\\RV_{t,i} - \\hat\\sigma^2_{t,i})^2 / \\sum (\\RV_{t,i} - "
            "\\overline{\\RV}_{t-1,i})^2$, in percent, where $\\overline{\\RV}_{t-1,i}$ is the historical mean "
            "realized variance of the same half-hour interval over every earlier trading day from "
            "\\rhoosBenchStart{} \\citep{CampbellThompson2008}. DM: the Diebold--Mariano statistic, with the "
            "correction of \\citet{HarveyLeybourneNewbold1997}, for the daily average QLIKE of the full predictor "
            "set minus that of the HAR benchmark of the same model (negative: the full set has the lower loss), "
            f"Newey--West lag {', '.join(str(x) for x in lag)}, $p$-value in parentheses. {trees_txt} "
            "Models trained on the last 30 minutes are in Table~\\ref{tab:rh_oos_last30}.",
            "\\end{tablenotes}",
            "\\end{threeparttable}",
            "\\end{table}",
            "",
        ]
    )
    out_table.parent.mkdir(parents=True, exist_ok=True)
    out_table.write_text(table, encoding="utf-8", newline="\n")

    # ------------------------------------------------------------------ macros for the prose
    def mac(name: str, value: str, comment: str) -> str:
        return f"\\providecommand{{\\{name}}}{{}}\\renewcommand{{\\{name}}}{{{value}}}  % {comment}"

    if not span12:
        lines = [
            HEAD,
            "% \\rhor... macros of Table 1 (tab:rh_oos): the pooled linear reruns scored on every interval.",
            mac("rhorNIntA", thousands(len(panels["48"][0])), "scored intervals, Panel A (48 intervals)"),
            mac("rhorNIntB", thousands(len(panels["13"][0])), "scored intervals, Panel B (13 regular-hours bars)"),
            mac("rhorDays", thousands(len(days)), "trading days of the span"),
            mac("rhorHacLag", ", ".join(str(x) for x in lag), "Newey-West lag of the daily tests"),
        ]
    else:  # --span 2012: no 48-interval rerun from 2012, so no Panel A macros
        lines = [
            HEAD,
            "% \\rhor... macros of Table 1 (tab:rh_oos), --span 2012: the pooled linear rerun from 2012 on the 13 bars",
            f"% ({cfg['window']:,}-day window, penalties chosen once on the folds 2008-2011 and frozen), scored on "
            "every bar.",
            mac("rhorNIntB", thousands(len(panels["13"][0])), "scored intervals, 13 regular-hours bars"),
            mac("rhorDays", thousands(len(days)), "trading days of the span"),
            mac("rhorSpanStart", f"{cfg['first'].day} {cfg['first']:%B %Y}", "first day of the span"),
            mac("rhorSpanEnd", f"{cfg['last'].day} {cfg['last']:%B %Y}", "last day of the span"),
            mac("rhorEarlyCloses", str(int(((EARLY >= days[0]) & (EARLY <= days[-1])).sum())),
                "NYSE early-close days inside the span, dropped (the rule of rhoosEarlyCloses)"),
            mac("rhorPenYears", "2008--2011", "years of the four validation folds on which the penalties are chosen"),
            mac("rhorNBars", str(len(set(panels["13"][0].strftime("%H:%M")))), "bars of the day in the sample"),
            mac("rhorPenFolds", str(len(json.loads((r13 / "frozen_penalties.json").read_text())["info"]["folds"])),
                "validation folds on which the penalties are chosen"),
            mac("rhorWindowDays", thousands(cfg["window"]), "rolling window, trading days"),
            mac("rhorWindowBars", thousands(13 * cfg["window"]), "rolling window, 13-bar observations"),
            mac("rhorHacLag", ", ".join(str(x) for x in lag), "Newey-West lag of the daily tests"),
        ]
    for pid, tag in (("48", "A"), ("13", "B")):
        if pid not in cfg["panels"]:
            continue
        for m, _ in MODELS[1:]:
            r = dms[(dms.panel == pid) & (dms.model == m) & (dms.comparison == "all features vs HAR benchmark")
                    & (dms.loss == "qlike")].iloc[0]
            cap = {"ridge": "Ridge", "lasso": "Lasso", "enet": "Enet"}[m]
            lines.append(mac(f"rhorDm{cap}{tag}", fmt(r.dm, 2), f"DM, QLIKE, all features vs HAR benchmark, {m}, panel {tag}"))
            lines.append(mac(f"rhorP{cap}{tag}", fmt_p(r.p), "its p-value"))
    # 2026-10-20 (user): parked -- the macros of every row, set and test of each subperiod. Was: \rhorSub{One,Two}
    # {Start,End,Days,NInt}, ...Qlike{Ols,Ridge,Lasso,Enet}{Har,Af}, ...Dm/P{Ridge,Lasso,Enet}. Now: what the one
    # sentence on the split quotes (06_results.tex), with A = 2012-2019 and B = 2020-2024.
    for tag, s_first, s_last, _ in cfg["subs"]:
        g = subs[(subs.tag == tag) & (subs.against == "")].set_index(["model", "set"])
        har = g.xs("baseline", level="set")

        def one_or_range(vals: list[str]) -> str:  # one HAR value if the four HAR rows agree to the printed digits
            lo, hi = vals[int(np.argmin([float(v.replace("$-$", "-")) for v in vals]))], \
                vals[int(np.argmax([float(v.replace("$-$", "-")) for v in vals]))]
            return lo if lo == hi else f"{lo}--{hi}"

        # the OLS and the penalized HAR rows were to agree to the printed digits; on 2020-2024 the lasso's QLIKE
        # (0.19515) prints 0.1952 against 0.1951, so the macro prints the range when they differ
        q_har = one_or_range([f"{v:.4f}" for v in har.qlike])
        r_har = one_or_range([fmt(v, 1) for v in har.r2_oos_pct])
        af = g.xs("all_features", level="set")
        lines.append(f"% subperiod {tag}: {s_first.date()} .. {s_last.date()}, the bars of Table 1 cut by date")
        lines.append(mac(f"rhorSub{tag}Label", f"{s_first.year}--{s_last.year}", "years of the subperiod"))
        lines.append(mac(f"rhorSub{tag}StartYear", str(s_first.year), "first year of the subperiod"))
        lines.append(mac(f"rhorSub{tag}Days", thousands(int(g.n_days.iloc[0])), "trading days of the subperiod"))
        lines.append(mac(f"rhorSub{tag}QlikeHar", q_har, "QLIKE, HAR benchmark (OLS, ridge, lasso, elastic net; a range if they differ)"))
        lines.append(mac(f"rhorSub{tag}RoosHar", r_har, "R2_OOS (%), HAR benchmark (the four rows; a range if they differ)"))
        lines.append(mac(f"rhorSub{tag}QlikeAfMin", f"{af.qlike.min():.4f}", "QLIKE, all features, lowest of ridge, lasso, elastic net"))
        lines.append(mac(f"rhorSub{tag}QlikeAfMax", f"{af.qlike.max():.4f}", "QLIKE, all features, highest of the three"))
        lines.append(mac(f"rhorSub{tag}RoosAfMin", fmt(af.r2_oos_pct.min(), 1), "R2_OOS (%), all features, lowest of the three"))
        lines.append(mac(f"rhorSub{tag}RoosAfMax", fmt(af.r2_oos_pct.max(), 1), "R2_OOS (%), all features, highest of the three"))
    if tf:  # the tree rows of Table 1
        lines.append("% \\rhorTree... macros: the pooled trees of Table 1 (--trees), on the bars of the linear rows")
        for m, _, tag in TREES:
            r = met[(met.panel == "13") & (met.model == m)].set_index("set")
            d = dms[(dms.panel == "13") & (dms.model == m) & (dms.comparison == "all features vs HAR benchmark")
                    & (dms.loss == "qlike")].iloc[0]
            lines.append(mac(f"rhorTreeQlikeHar{tag}", f"{r.loc['baseline', 'qlike']:.4f}", f"QLIKE, HAR benchmark, {m}"))
            lines.append(mac(f"rhorTreeQlikeAf{tag}", f"{r.loc['all_features', 'qlike']:.4f}", f"QLIKE, all features, {m}"))
            lines.append(mac(f"rhorTreeRoosHar{tag}", fmt(r.loc["baseline", "r2_oos_pct"], 1), f"R2_OOS (%), HAR benchmark, {m}"))
            lines.append(mac(f"rhorTreeRoosAf{tag}", fmt(r.loc["all_features", "r2_oos_pct"], 1), f"R2_OOS (%), all features, {m}"))
            lines.append(mac(f"rhorTreeDm{tag}", fmt(d.dm, 2), f"DM, QLIKE, all features vs HAR benchmark, {m}"))
            lines.append(mac(f"rhorTreeP{tag}", fmt_p(d.p), "its p-value"))
    out_numbers.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    print(met.to_string(index=False))
    print(dms[dms.loss == "qlike"].to_string(index=False))
    if len(subs):
        print(subs.to_string(index=False))
    print(f"wrote {out_table}, {out_numbers}, {out_res.relative_to(REPO) if out_res.is_relative_to(REPO) else out_res}/")


if __name__ == "__main__":
    main()
