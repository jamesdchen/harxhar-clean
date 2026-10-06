"""Load the committed export of the 2026-10-03 / 10-04 close studies (no _work/, no design cache).

The export lives in results/close_studies_2026-10-03/forecasts/ and is written by
experiments/close_studies_export.py (stage ``build``); its README.md lists every file and column.
Everything here reads only those files, plus the scorer code it calls
(experiments/dense_vs_sparse_1530.py research_frame / deck_panel / point).

    import close_studies_load as L
    cat = L.catalogue()                                   # one row for each series
    F = L.forecasts(study="trees_kfull")                  # dates x series, adjusted scale
    L.score(F)                                            # QLIKE, Sharpe mid / crossed, 866 trade days
    L.score(L.rebuild("trees_morebars/average.csv", "average of lgbm pools N = 1 .. 4"))
    th = L.coefficients("ridge_bb0")                      # dates x (628 design columns + intercept)
    gain = L.importance("lgbm_bars4_r8000_barmin_seed42")  # refits x columns
    D = L.design()                                        # 16:00 design: 3469 rows x (628 inputs + y, B)
    C = L.linear_contributions("ridge_bb0")               # linear SHAP: dates x 628 columns

Scoring: ``score`` puts every forecast through the master table's research scorer: the 16:00-bar
recalibration pred_clock = (pred_adj^2 + s) x B, s = the mean squared adjusted-scale error over
the previous 250 forecast rows (at least 63), lagged one row, computed from the rows ``start`` ..
1468 of each forecast; QLIKE of pred_clock against the realized variance and the 15:30 sign(s)
straddle (buy when pred_clock > iv_var) on the 866 trade days 2020-01-03 .. 2024-04-30.  The
default start is row 130 (2019-01), the first tree forecast; on the trade days the result does not
depend on the start as long as it is at or before row 132 (the first trade day is row 382).
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "experiments", REPO / "notebooks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

EXPORT = REPO / "results" / "close_studies_2026-10-03" / "forecasts"
TREE_START = 130  # first tree forecast row (close_trees_datasize.TREE_START)
N_FC = 1469  # 16:00 forecast rows, 2018-06-25 .. 2024-04-30
RIDGE_REF = "exog_penalty/ridge_bb0"  # ridge, HAR + calendar backbone unpenalized
DECK_COLS = ("iv_var", "R", "exit", "ask_c", "ask_p", "bid_c", "bid_p", "signal", "entry")


# ============================================================================ tables
@lru_cache(maxsize=None)
def _read(name: str) -> pd.DataFrame:
    f = EXPORT / name
    if f.suffix == ".csv":
        return pd.read_csv(f, keep_default_na=True, low_memory=False)
    t = pd.read_parquet(f)
    for c in t.columns:  # dictionary-encoded strings come back categorical
        if isinstance(t[c].dtype, pd.CategoricalDtype):
            t[c] = t[c].astype(str)
    return t


def catalogue(exported_only: bool = False) -> pd.DataFrame:
    """One row for each series (and each npz file left out, with the reason)."""
    c = _read("catalogue.csv").copy()
    return c[c["exported"] == "yes"].reset_index(drop=True) if exported_only else c


def study_arms() -> pd.DataFrame:
    """How each row of the studies' own CSVs maps to an exported series or a recipe over them."""
    return _read("study_arms.csv").copy()


def targets() -> pd.DataFrame:
    """One row for each 16:00 forecast row (index = 16:00 bar-end stamp, naive ET)."""
    t = _read("targets.parquet").copy()
    t["date"] = pd.to_datetime(t["date"]).astype("datetime64[ns]")
    return t.set_index("date")


def _resolve(series: str) -> str:
    """Accept a full series_id ('trees_kfull/lgbm_bars4_r8000_barmin_seed42') or a unique arm name."""
    c = catalogue(exported_only=True)
    if series in set(c["series_id"]):
        return series
    hit = c.loc[c["arm"] == series, "series_id"].tolist()
    if len(hit) == 1:
        return hit[0]
    raise KeyError(f"{series!r}: {'ambiguous: ' + ', '.join(hit) if hit else 'no such series'}")


def forecasts(
    series: str | list[str] | None = None,
    study: str | list[str] | None = None,
    model: str | list[str] | None = None,
    kind: str | list[str] | None = None,
    value: str = "pred_adj",
    include_exact: bool = False,
) -> pd.DataFrame:
    """Wide frame: 16:00 stamps (1469 rows) x series.  value = 'pred_adj' (the model's own output,
    adjusted scale) or 'pred_clock' (the scorer's recalibrated variance forecast).  Filters combine
    with AND; the exact-solution check series (exog_penalty '<arm>:exact_pred', every 25th row) are
    left out unless asked for (include_exact or by name)."""
    c = catalogue(exported_only=True)
    keep = np.ones(len(c), bool)

    def _in(col: str, v) -> np.ndarray:
        v = [v] if isinstance(v, str) else list(v)
        return c[col].isin(v).to_numpy()

    if series is not None:
        ids = [series] if isinstance(series, str) else list(series)
        ids = [_resolve(s) for s in ids]
        keep &= c["series_id"].isin(ids).to_numpy()
    if study is not None:
        keep &= _in("study", study)
    if model is not None:
        keep &= _in("model", model)
    if kind is not None:
        keep &= _in("kind", kind)
    if not include_exact and series is None:
        keep &= (c["is_exact_pred"] != "yes").to_numpy()
    want = c.loc[keep, "series_id"].tolist()
    f = _read("forecasts.parquet")
    f = f[f["series_id"].isin(want)]
    W = f.pivot(index="date", columns="series_id", values=value)
    W.index = pd.DatetimeIndex(pd.to_datetime(W.index)).astype("datetime64[ns]")
    W.columns.name = None
    order = [s for s in (ids if series is not None else want) if s in W.columns]
    return W[order]


def linear_columns() -> pd.DataFrame:
    """Columns of the linear coefficient vectors: 628 design columns + the intercept (last)."""
    return _read("linear_columns.csv").copy()


def tree_columns() -> pd.DataFrame:
    """Columns of the tree importance records: 628 design columns (+ bar_end_minute at 628)."""
    return _read("tree_columns.csv").copy()


def coefficients(arm: str) -> pd.DataFrame:
    """Coefficients of an exog_penalty arm at every forecast row (float32): index = 16:00 stamp,
    columns = the design names of design_bar1600_all_features + 'intercept' (the C kernel's
    layout: pred = [X_row, 1] . theta).  ols_baseline: the 22 backbone columns only (its intercept
    was not stored)."""
    sid = arm if "/" in arm else f"exog_penalty/{arm}"
    t = pd.read_parquet(EXPORT / "linear_coefficients.parquet", filters=[("series_id", "==", sid)])
    if not len(t):
        raise KeyError(f"no coefficients for {sid!r}")
    t = t.sort_values("row")
    t.index = pd.DatetimeIndex(pd.to_datetime(t.pop("date"))).astype("datetime64[ns]")
    t.index.name = "date"
    return t.drop(columns=["series_id", "row"])


DESIGN_META = ("row", "date", "forecast_row", "y", "baseline", "true_raw")


def design(forecast_rows_only: bool = False) -> pd.DataFrame:
    """The 16:00-bar all_features design, every row (index = 16:00 stamp): row, forecast_row
    (row - 2000; >= 0 on the 1469 forecast rows of every other table), y (= true_adj), baseline (B),
    true_raw, and the 628 prescaled inputs exactly as the 16:00-bar models received them."""
    t = _read("design_bar1600_all_features.parquet").copy()
    t.index = pd.DatetimeIndex(pd.to_datetime(t.pop("date"))).astype("datetime64[ns]")
    t.index.name = "date"
    return t[t["forecast_row"] >= 0] if forecast_rows_only else t


def design_names() -> list[str]:
    """The 628 input columns of the 16:00 design, in the coefficients' and importances' order."""
    return [c for c in _read("design_bar1600_all_features.parquet").columns if c not in DESIGN_META]


def linear_contributions(arm: str, window: int = 2000) -> pd.DataFrame:
    """Linear SHAP of an exog_penalty arm at every forecast row: theta_j x (x_j - mean of x_j over
    the arm's training window, the `window` rows before the forecast row).  Rows sum to the
    forecast minus the prediction at the window means.  theta is float32 in the export."""
    names = design_names()
    D = design()
    X = D[names].to_numpy(np.float64)
    first = int((D["forecast_row"] < 0).sum())
    th = coefficients(arm)
    cs = np.vstack([np.zeros(len(names)), np.cumsum(X, axis=0)])
    rows = np.arange(first, first + len(th))
    means = (cs[rows] - cs[rows - window]) / window
    phi = (X[rows] - means) * th[names].to_numpy(np.float64)
    return pd.DataFrame(phi, index=th.index, columns=names)


def rechoices(arm: str | None = None) -> pd.DataFrame:
    """Every penalty re-choice of the linear arms: alpha, ratio r, group penalties, validation
    MSE grid, locked and masked column sets (column indices into linear_columns())."""
    t = _read("linear_rechoices.parquet").copy()
    if arm is not None:
        sid = arm if "/" in arm else _resolve(arm)
        t = t[t["series_id"] == sid]
    return t.reset_index(drop=True)


def importance(arm: str, kind: str = "gain") -> pd.DataFrame:
    """Booster importance of a trees_kfull arm at every refit: kind = 'gain' (total gain, float32),
    'split' (split count) or 'kept' (column kept by the window mask).  Index = refit anchor row
    (forecast row index); columns = the columns the arm's design had."""
    sid = arm if "/" in arm else f"trees_kfull/{arm}"
    t = pd.read_parquet(EXPORT / "tree_importance.parquet", filters=[("series_id", "==", sid)])
    if not len(t):
        raise KeyError(f"no importance records for {sid!r}")
    cols = tree_columns()["name"].tolist()
    n_col = int(catalogue().set_index("series_id").loc[sid, "n_columns"])
    rf = refits(sid)
    anchors = rf["row"].to_numpy()
    if kind == "kept":
        M = np.zeros((len(anchors), n_col), bool)
        M[t["refit"].to_numpy(), t["col"].to_numpy()] = True
    else:
        M = np.zeros((len(anchors), n_col), np.float32 if kind == "gain" else np.int32)
        M[t["refit"].to_numpy(), t["col"].to_numpy()] = t[kind].to_numpy()
    return pd.DataFrame(M, index=pd.Index(anchors, name="anchor_row"), columns=cols[:n_col])


def refits(series: str | None = None) -> pd.DataFrame:
    """Fit-level records: tree refits (anchor row, fit seconds, training rows / sessions, kept
    columns, leaf minimum, linear-leaf sizes, the configuration in force) and linear solves
    (alpha in force, homotopy events) at every forecast row."""
    t = _read("refits.parquet").copy()
    if series is not None:
        t = t[t["series_id"] == _resolve(series)]
    return t.reset_index(drop=True)


def pretune_trials() -> pd.DataFrame:
    """The trees_pretune pre-tune trials, one row for each (study, trial, fold)."""
    return _read("pretune_trials.parquet").copy()


# ============================================================================ derived forecasts
def average(series: list[str]) -> pd.Series:
    """Equal-weight average on the adjusted scale, in the given order (as the studies' np.mean)."""
    F = forecasts(series=series)
    out = np.mean([F[s].to_numpy(float) for s in F.columns], axis=0)
    return pd.Series(out, index=F.index, name="average")


def seed_average(series: list[str]) -> pd.Series:
    """Average of seed replicates of one configuration (adjusted scale)."""
    return average(series).rename("seed average")


def pool_average(series: list[str]) -> pd.Series:
    """Average of different pools (adjusted scale)."""
    return average(series).rename("pool average")


def combine(a, b, w: float = 0.5) -> pd.Series:
    """w a + (1 - w) b on the adjusted scale (a, b: series ids or pd.Series of pred_adj)."""
    A = forecasts(series=a).iloc[:, 0] if isinstance(a, str) else a
    B = forecasts(series=b).iloc[:, 0] if isinstance(b, str) else b
    return pd.Series(w * A.to_numpy(float) + (1.0 - w) * B.to_numpy(float), index=A.index, name="combination")


@lru_cache(maxsize=None)
def _recipes() -> dict:
    sa = study_arms()
    return {(r.table, r.label): r for r in sa.itertuples(index=False)}


def rebuild(table: str, label: str) -> pd.Series:
    """The adjusted-scale forecast behind one row of a study CSV (e.g. table
    'trees_morebars/average.csv', label 'average of lgbm pools N = 1 .. 4'), from its recipe:
    single = an exported series; mean = np.mean of the members in order; weighted = sum of
    weight x member.  A member is a series_id or another label of the same table."""
    R = _recipes()
    r = R[(table, label)]
    if r.recipe == "single":
        return forecasts(series=r.series_id).iloc[:, 0].rename(label)
    if r.recipe in ("variance", "realtime"):
        raise ValueError(f"{label!r} combines on the variance level: use ridge_trees(trees, how={'variance' if r.recipe == 'variance' else 'realtime'!r})")
    members = json.loads(r.members)
    ids = set(catalogue(exported_only=True)["series_id"])

    def member(m: str) -> pd.Series:
        if m in ids:
            return forecasts(series=m).iloc[:, 0]
        if (table, m) in R:
            return rebuild(table, m)
        other = [t for (t, lab) in R if lab == m and t.split("/")[0] == table.split("/")[0]]
        if not other:
            raise KeyError(f"member {m!r} of {label!r}")
        return rebuild(other[0], m)

    arrs = [member(m) for m in members]
    idx = arrs[0].index
    if r.recipe == "mean":
        out = np.mean([a.to_numpy(float) for a in arrs], axis=0)
    elif r.recipe == "weighted":
        w = json.loads(r.weights)
        out = w[0] * arrs[0].to_numpy(float)
        for wi, a in zip(w[1:], arrs[1:]):
            out = out + wi * a.to_numpy(float)
    else:
        raise ValueError(r.recipe)
    return pd.Series(out, index=idx, name=label)


def ridge_trees(trees, ridge: str = RIDGE_REF, how: str = "adjusted", w: float = 0.5, start: int = TREE_START, rt_start: int = 250) -> pd.Series:
    """Ridge + trees combinations of the studies.
    how = 'adjusted': w ridge + (1 - w) trees on the adjusted scale (trees_morebars/average.csv,
          kfull_tests '... adjusted scale, rescored'); returns pred_adj (score it with score()).
    how = 'variance': w F_ridge + (1 - w) F_trees on the recalibrated variance level; returns the
          variance forecast on the 866 trade days (score it with score_clock()).
    how = 'realtime': F_ridge + lambda_t (F_trees - F_ridge), lambda_t minimizing the QLIKE over the
          earlier trade days (grid 0, 0.001, .., 1; NaN before trade day rt_start), as
          kfull_tests/realtime_combination.csv; returns the variance forecast on the trade days."""
    T = forecasts(series=trees).iloc[:, 0] if isinstance(trees, str) else trees
    R = forecasts(series=ridge).iloc[:, 0]
    if how == "adjusted":
        return pd.Series(w * R.to_numpy(float) + (1.0 - w) * T.to_numpy(float), index=R.index, name="ridge + trees")
    P = panel(pd.DataFrame({"ridge": R, "trees": T}), start=start)
    fL, fT = P["F"]["ridge"].to_numpy(float), P["F"]["trees"].to_numpy(float)
    if how == "variance":
        out = w * (fL + fT) if w == 0.5 else w * fL + (1.0 - w) * fT
    elif how == "realtime":
        import close_kfull_testlib as KT

        lam = KT.realtime_lambda(P["RV"].to_numpy(float), fL, fT, rt_start, np.linspace(0.0, 1.0, 1001))
        out = fL + np.nan_to_num(lam) * (fT - fL)
    else:
        raise ValueError(how)
    return pd.Series(out, index=P["F"].index, name=f"ridge + trees ({how})")


# ============================================================================ scorer
def deck() -> pd.DataFrame:
    """The straddle deck on the 866 trade days (index = trade day, as dense_vs_sparse_1530.deck_frame)."""
    t = targets()
    d = t[t["is_trade_day"]].copy()
    d.index = pd.DatetimeIndex(d.index).normalize().astype("datetime64[ns]")
    d.index.name = None
    return d[list(DECK_COLS)].sort_index()


def _as_frame(x) -> pd.DataFrame:
    if isinstance(x, str):
        return forecasts(series=x)
    if isinstance(x, (list, tuple)):
        return forecasts(series=list(x))
    if isinstance(x, pd.Series):
        return x.to_frame(x.name if x.name is not None else "forecast")
    return x


def panel(x, start: int = TREE_START) -> dict:
    """Daily records on the 866 trade days for each forecast (columns): pred_clock 'F', realized
    variance 'RV', QLIKE 'ql', mid P&L 'pnl', crossed P&L 'pnlx' (DataFrames), plus 'pre_ql':
    QLIKE on the forecast rows before the first trade day where pred_clock is defined."""
    import dense_vs_sparse_1530 as dvs

    F = _as_frame(x)
    t = targets()
    assert len(F) == len(t) and (pd.DatetimeIndex(F.index) == t.index).all(), "forecasts must be on the 1469 forecast rows"
    rows = np.arange(start, len(t))
    dz = dict(date=t.index.to_numpy()[rows], true_adj=t["true_adj"].to_numpy(float)[rows], true_raw=t["true_raw"].to_numpy(float)[rows])
    dk = deck()
    out = {k: {} for k in ("F", "RV", "ql", "pnl", "pnlx", "buy", "pre_ql")}
    first_trade = dk.index[0]
    for name in F.columns:
        p = F[name].to_numpy(float)[rows]
        fr = dvs.research_frame(dz, p)
        f = fr["pred_clock"].set_axis(fr.index.normalize())
        Fd = f.reindex(dk.index).to_numpy(float)
        if not np.isfinite(Fd).all():
            continue  # no forecast on some trade day (e.g. the exact-solution check series)
        P = dvs.deck_panel([fr], dk)
        out["F"][name] = Fd
        out["RV"] = fr["true_raw"].set_axis(fr.index.normalize()).reindex(dk.index)
        for k in ("ql", "pnl", "pnlx"):
            out[k][name] = P[k][:, 0]
        out["buy"][name] = float(P["buy"][0])
        m = (f.index < first_trade) & np.isfinite(f.to_numpy(float))
        rv = fr["true_raw"].set_axis(fr.index.normalize())
        r = rv[m].to_numpy(float) / f[m].to_numpy(float)
        out["pre_ql"][name] = pd.Series(r - np.log(r) - 1.0, index=f.index[m])
    for k in ("F", "ql", "pnl", "pnlx"):
        out[k] = pd.DataFrame(out[k], index=dk.index)
    return out


def score(x, start: int = TREE_START, daily: bool = False):
    """QLIKE and the sign(s) straddle Sharpe (mid, crossed) on the 866 trade days for each
    forecast.  x: a series_id / arm name, a list of them, a pd.Series of pred_adj on the 1469
    forecast rows, or a DataFrame (forecast rows x forecasts).  Also returns the MSE of pred_clock
    on the trade days and the QLIKE on the forecast rows of 2019 (before the first trade day)
    where pred_clock is defined (kfull_tests' k* selection rows).  daily=True also returns the
    panel() dict."""
    import dense_vs_sparse_1530 as dvs

    P = panel(x, start=start)
    names = list(P["ql"].columns)
    pt = dvs.point({k: P[k].to_numpy(float) for k in ("ql", "pnl", "pnlx")}) if names else {"ql": [], "sh": [], "shx": []}
    RV = P["RV"].to_numpy(float) if names else None
    tab = pd.DataFrame(
        dict(
            forecast=names,
            qlike=pt["ql"],
            sharpe_mid=pt["sh"],
            sharpe_crossed=pt["shx"],
            pct_buy=[100.0 * P["buy"][n] for n in names],
            mse=[float(np.mean((RV - P["F"][n].to_numpy(float)) ** 2)) for n in names],
            qlike_2019=[float(P["pre_ql"][n].mean()) for n in names],
            n_days=len(P["ql"]),
        )
    ).set_index("forecast")
    return (tab, P) if daily else tab


def score_clock(F, RV: pd.Series | None = None) -> pd.DataFrame:
    """QLIKE / MSE / Sharpe of variance forecasts already on the trade days (e.g. ridge_trees(...,
    how='variance')).  NaN days (the real-time combination's first 250) are left out."""
    import score_linear_subsection as base

    F = F.to_frame() if isinstance(F, pd.Series) else F
    dk = deck()
    if RV is None:
        RV = targets()["true_raw"].set_axis(targets().index.normalize()).reindex(dk.index)
    rows = []
    for n in F.columns:
        f = F[n].reindex(dk.index).to_numpy(float)
        m = np.isfinite(f)
        y = RV.to_numpy(float)[m]
        r = y / f[m]
        q = np.where(f[m] > dk["iv_var"].to_numpy(float)[m], 1.0, -1.0)
        pnl = q * dk["R"].to_numpy(float)[m]
        ask = (dk["ask_c"] + dk["ask_p"]).to_numpy(float)[m]
        bid = (dk["bid_c"] + dk["bid_p"]).to_numpy(float)[m]
        ex = dk["exit"].to_numpy(float)[m]
        pnlx = q * np.where(q > 0, ex / ask - 1.0, ex / bid - 1.0)
        rows.append(dict(forecast=n, qlike=float(np.mean(r - np.log(r) - 1.0)), mse=float(np.mean((y - f[m]) ** 2)),
                         sharpe_mid=pnl.mean() / pnl.std(ddof=1) * base.ANN, sharpe_crossed=pnlx.mean() / pnlx.std(ddof=1) * base.ANN,
                         n_days=int(m.sum())))
    return pd.DataFrame(rows).set_index("forecast")
