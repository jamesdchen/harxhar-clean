"""Rolling-origin forecast evaluation of the out-of-sample table tab:rh_oos (2026-10-15).

User (2026-10-15), verbatim: "Out of sample period doesn't make sense for forecasts because we're performing rolling
backtests. The forecasting results need to be calculated to reflect such (that the out of sample forecasts are done on
a rolling basis after the burn in training window)"

WHAT.  Every forecast of tab:rh_oos (writeup/make_rehaul_oos_table.py model_rows()) comes from a rolling-window
backtest: at each forecast origin the model is re-estimated on its trailing window, so every forecast on disk is an
out-of-sample forecast of its day.  The paper scored forecast ACCURACY only on the 866 trading days 2020-01-03 ..
2024-04-30 of results/close_master_table/master_table_daily.parquet -- the days with a same-day-expiry SPXW straddle,
which matter only for the straddle economics.  This script scores every rolling forecast on disk, from the first
forecast after the model's initial estimation window to the end of the data, and on the common span of all of them.

SOURCES (the files experiments/master_table_close.py reads for each key, arm root linear_subsection_dedup, the paper
run's PERBAR_SWITCH / PAPER_SWITCHED active; gate: its discover() resolves the same path for every master key)
  closing-interval linear  results/spxw_pnl/yhat_subpt_<est>_<set>.parquet (results/linear_pretune_2026-10-06,
                           experiments/stack_pretune_yhat.py; 2026-10-18, user, new-notes 24 per E23: the frozen arms,
                           c_frozen on the HAR benchmark and d_frozen_bb on all features, one pre-sample penalty and no
                           re-tune -- was arm f, the margin re-tune, which replaced the HAR-benchmark penalty from
                           2020-06-23 (ridge, elastic net) and 2023-06-16 (lasso) and never on all features, so only
                           the three subpt_*_baseline forecasts moved; outputs re-written 2026-10-22); the OLS on the HAR
                           benchmark is the per-bar incumbent arm file
                           results/linear_subsection_dedup/baseline/bar1600/reclasso/tw2000/causal_tune_linear/
                           incumbent_ols/results_bar1600.csv (B = true_raw / true_adj^2, compare_mfiv_harlag._prep);
                           since 2026-10-26 (integration plan D9) its rebuild under the current data chain,
                           results/ci_har_rebuild_2026-10-26/results_bar1600.csv (same model; forecasts from 2006-06-14);
  closing-interval trees   results/spxw_pnl/yhat_subtree_daily_<set>_lgbm.parquet,
                           results/trees_xgbtop_1600/yhat_subtree_daily_<set>_xgbtop.parquet,
                           results/trees_rf400_1600/yhat_subtree_daily_<set>_rf400.parquet (PERBAR_SWITCH);
  pooled linear            results/spxw_pnl/yhat_poolpt_<est>_<set>.parquet (arm f of
                           results/pool_linear_pretune_2026-10-06; NaN before 2020-01-03: the pre-tuned pooled walk
                           ran over the trade period only, experiments/stack_pool_pretune_yhat.py), a0 =
                           results/spxw_pnl/yhat_a0.parquet (pooled OLS on HAR + calendar);
  pooled trees             lgbm = results/spxw_pnl/yhat_tree00.parquet, xgb = yhat_tree10.parquet (PAPER_SWITCHED),
                           results/pooled_trees_2026-10-07/yhat_pooltree_<est>_baseline.parquet, and the forest's
                           INTERIM 16:00 table yhat_pooltree_rf_all_features_INTERIM1600.parquet (walk still running).

FORECAST  sigma2_t = yhat_t^2 * B_t on the 16:00 ET row (bar 15:30-16:00), B = the table's own baseline column (the
          arm file's true_raw / true_adj^2) -- the master table's "plain" forecast (asserted to 1e-10 relative on the
          866 master days for every master key; make_rehaul_oos_table.plain_16 for the pooltree keys).
TRUTH     RV_t = the 16:00 ET sumret2 of data/core_stats.parquet (= master_table_daily rv_unclipped = the production
          table's rv_raw, asserted on the 866 days; every table's own rv_raw equals it on every forecast day), the
          actual (unwinsorized) realized variance of the closing half hour.  NYSE early-close days are dropped (their
          16:00 row is a post-close bar; atm_straddle_lib.EARLY_CLOSE_DATES), as are days with a non-positive or
          missing print.
DAYS      scored days of a key = its finite 16:00 forecasts on non-early-close days with a positive truth.  Gaps are
          NYSE sessions (rule-based calendar below, checked against core_stats) inside the key's span without a scored
          forecast.
SPANS     own        every scored day of the key (first forecast after its initial estimation window .. 2024-04-30);
          own_clean  own, from the first day whose forecast uses no hyperparameter selected on data at or after that
                     day (HP_CLEAN: the pre-tuned linear penalties were selected on validation folds tiling the 1,000
                     16:00 sessions before 2020-01-03; the tree configurations on the dev span before 2003-01-01);
          common     the days on which EVERY key of the table has a scored forecast (the paired span of the table);
          mt866      the master table's 866 days (reproduces the current table's numbers).
METRICS   MSE = mean (RV_t - sigma2_t)^2; QLIKE = mean r - log r - 1, r = RV_t / sigma2_t; R2_OOS (Campbell and
          Thompson 2008) = 1 - sum (RV_t - sigma2_t)^2 / sum (RV_t - RVbar_{t-1})^2, RVbar_{t-1} = the mean of every
          closing-half-hour RV print strictly before day t from 1998-01-05 (expanding, causal; brute-force checked on
          every scored day; = make_rehaul_oos_table.ct_benchmark); DM = src.evaluation.diebold_mariano.dm_test as is
          (HLN, Bartlett Newey-West at its automatic lag; autocovariances divided by T since 2026-10-18, E13, so the
          DM columns of the 2026-10-22 re-run differ from the 10-15 run's for every key) on the daily loss
          differential key minus sub_ols_baseline,
          QLIKE and squared error, on the days BOTH have (dm < 0: the key has the lower loss).
CHECK     on the 866 master days every key's QLIKE equals master_table.csv qlike_raw (rtol 1e-12) and its daily QLIKE
          the daily frame's qlike_raw.

WRITES  results/rolling_eval_2026-10-15/spans.csv    one row per key: source, 16:00 rows, first / last forecast, count,
                                                      early closes dropped, gaps, hyperparameter-clean start;
        results/rolling_eval_2026-10-15/metrics.csv  one row per key x span (own, own_clean, common, mt866);
        results/rolling_eval_2026-10-15/common_span.csv  the common span (first, last, count, binding keys);
        results/rolling_eval_2026-10-15/gates.csv    every asserted check and its value.

API (imported by writeup/make_rehaul_oos_table.py, writeup/make_rehaul_results.py, writeup/make_rehaul_data_methods.py):
  common_span()  the common rolling span (cached); scored(key)  plain and actual RV of any key of tab:rh_oos or of the
  master table on its scored days; history()  the scoring truth; master_source(key)  the master table's source of a
  key.  main() gates that the API reproduces this run's frames and span.

Run:  python experiments/rolling_eval_2026_10_15.py      (CPU light: one process, BLAS threads 1)
"""

from __future__ import annotations

import os

for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ[_v] = "1"
os.environ.setdefault("MASTER_TABLE_ARM_ROOT", "linear_subsection_dedup")  # the paper run's arm root

import sys  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "notebooks", REPO / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import atm_straddle_lib as asl  # noqa: E402
from src.evaluation.diebold_mariano import dm_test  # noqa: E402

RES = REPO / "results"
SPXW = RES / "spxw_pnl"
MT = RES / "close_master_table"
CORE = REPO / "data" / "core_stats.parquet"
OUT = RES / "rolling_eval_2026-10-15"
OLS = "sub_ols_baseline"
# 2026-10-26 (integration plan D9): the CI-HAR is the rebuild under the current data chain (RULE.txt next to it), the
# file experiments/master_table_close.py reads for sub_ols_baseline (CI_HAR_REBUILD). Parked -- the incumbent_ols arm:
# OLS_ARM = (
#     RES / "linear_subsection_dedup" / "baseline" / "bar1600" / "reclasso" / "tw2000" / "causal_tune_linear"
#     / "incumbent_ols" / "results_bar1600.csv"
# )
OLS_ARM_OLD = (
    RES
    / "linear_subsection_dedup"
    / "baseline"
    / "bar1600"
    / "reclasso"
    / "tw2000"
    / "causal_tune_linear"
    / "incumbent_ols"
    / "results_bar1600.csv"
)
OLS_ARM = RES / "ci_har_rebuild_2026-10-26" / "results_bar1600.csv"
MT_FIRST, MT_LAST, MT_N = pd.Timestamp("2020-01-03"), pd.Timestamp("2024-04-30"), 866
PLAIN_RTOL = 1e-10  # the plain forecast vs master_table_daily plain on the 866 days
QLIKE_RTOL = 1e-12  # QLIKE vs master_table.csv qlike_raw
MSE_EXP = 10  # MSE printed x 10^MSE_EXP (as make_rehaul_oos_table)

# hyperparameter selection and the first day free of it
PRETUNE_END = pd.Timestamp("2020-01-03")  # close_linear_pretune / pool_linear_pretune: folds before the first trade day
DEV_END = pd.Timestamp("2003-01-01")  # experiments/tree_menu_dev.py DEV_END_DATE (score_unification TREE_DEV_END)
HP_NOTE = {
    "ols": "none (OLS)",
    "pretune": "penalty pre-tuned on 4 x 250-session validation folds tiling the 1,000 16:00 sessions before 2020-01-03 "
    "(experiments/close_linear_pretune.py pre_folds; pooled: pool_linear_pretune.py), light retune from 2020-06-23",
    # 2026-10-22: the closing-interval linear tables are the frozen arms since 2026-10-18 (new-notes 24, SOURCES)
    "frozen": "penalty pre-tuned on 4 x 250-session validation folds tiling the 1,000 16:00 sessions before "
    "2020-01-03 (experiments/close_linear_pretune.py pre_folds), frozen: no re-tune (arms c_frozen / d_frozen_bb)",
    "dev": "configuration chosen on the dev span before 2003-01-01 (experiments/tree_menu_dev.py; rf: the 400-trial "
    "dev-span search, trial 255)",
}


@dataclass(frozen=True)
class Key:
    key: str
    model: str
    pset: str  # predictor set: baseline (HAR benchmark) | all_features
    scheme: str  # close (last 30 minutes) | pool (pooled intraday)
    path: Path
    kind: str  # table | arm
    hp: str  # ols | pretune | frozen | dev

    @property
    def hp_clean_from(self) -> pd.Timestamp | None:
        return {
            "ols": None,
            "pretune": PRETUNE_END,
            "frozen": PRETUNE_END,
            "dev": DEV_END,
        }[self.hp]


def table_keys() -> list[Key]:
    """Every key of tab:rh_oos (= writeup/make_rehaul_oos_table.py model_rows() on SETS baseline / all_features)."""
    k: list[Key] = [
        Key(OLS, "OLS", "baseline", "close", OLS_ARM, "arm", "ols"),
        Key("a0", "OLS", "baseline", "pool", SPXW / "yhat_a0.parquet", "table", "ols"),
    ]
    for est, name in (("ridge", "Ridge"), ("lasso", "Lasso"), ("enet", "Elastic net")):
        for b in ("baseline", "all_features"):
            k.append(
                Key(
                    f"subpt_{est}_{b}",
                    name,
                    b,
                    "close",
                    SPXW / f"yhat_subpt_{est}_{b}.parquet",
                    "table",
                    "frozen",  # 2026-10-22: was "pretune" (arm f, light re-tune) before the 10-18 swap
                )
            )
            k.append(
                Key(
                    f"poolpt_{est}_{b}",
                    name,
                    b,
                    "pool",
                    SPXW / f"yhat_poolpt_{est}_{b}.parquet",
                    "table",
                    "pretune",
                )
            )
    close_tree = {
        "lgbm": lambda b: SPXW / f"yhat_subtree_daily_{b}_lgbm.parquet",
        "xgb": lambda b: RES / "trees_xgbtop_1600" / f"yhat_subtree_daily_{b}_xgbtop.parquet",
        "rf": lambda b: RES / "trees_rf400_1600" / f"yhat_subtree_daily_{b}_rf400.parquet",
    }
    pt = RES / "pooled_trees_2026-10-07"
    pool_tree = {
        ("lgbm", "baseline"): (
            "pooltree_lgbm_baseline",
            pt / "yhat_pooltree_lgbm_baseline.parquet",
        ),
        ("xgb", "baseline"): (
            "pooltree_xgb_baseline",
            pt / "yhat_pooltree_xgb_baseline.parquet",
        ),
        ("rf", "baseline"): (
            "pooltree_rf_baseline",
            pt / "yhat_pooltree_rf_baseline.parquet",
        ),
        ("lgbm", "all_features"): (
            "lgbm",
            SPXW / "yhat_tree00.parquet",
        ),  # the earlier draft's pooled LightGBM
        ("xgb", "all_features"): (
            "xgb",
            SPXW / "yhat_tree10.parquet",
        ),  # PAPER_SWITCHED: bank arm 10
        ("rf", "all_features"): (
            "pooltree_rf_all_features",
            pt / "yhat_pooltree_rf_all_features_INTERIM1600.parquet",
        ),
    }
    for est, name in (
        ("lgbm", "LightGBM"),
        ("xgb", "XGBoost"),
        ("rf", "Random forest"),
    ):
        for b in ("baseline", "all_features"):
            k.append(
                Key(
                    f"subtree_daily_{b}_{est}",
                    name,
                    b,
                    "close",
                    close_tree[est](b),
                    "table",
                    "dev",
                )
            )
            pk, pp = pool_tree[(est, b)]
            k.append(Key(pk, name, b, "pool", pp, "table", "dev"))
    assert len({x.key for x in k}) == len(k) == 26
    return k


# --------------------------------------------------------------------------- calendar


def nyse_sessions(first: str, last: str) -> pd.DatetimeIndex:
    """NYSE full and early-close sessions by rule (weekdays minus the exchange's holidays and one-off closures)."""
    from pandas.tseries.holiday import (
        AbstractHolidayCalendar,
        GoodFriday,
        Holiday,
        USLaborDay,
        USMartinLutherKingJr,
        USMemorialDay,
        USPresidentsDay,
        USThanksgivingDay,
        nearest_workday,
        sunday_to_monday,
    )

    class NYSE(AbstractHolidayCalendar):
        rules = [
            Holiday("New Year", month=1, day=1, observance=sunday_to_monday),  # Saturday: no Friday closure
            USMartinLutherKingJr,
            USPresidentsDay,
            GoodFriday,
            USMemorialDay,
            Holiday(
                "Juneteenth",
                month=6,
                day=19,
                start_date="2022-01-01",
                observance=nearest_workday,
            ),
            Holiday("Independence", month=7, day=4, observance=nearest_workday),
            USLaborDay,
            USThanksgivingDay,
            Holiday("Christmas", month=12, day=25, observance=nearest_workday),
        ]

    one_off = pd.to_datetime(
        [
            "2001-09-11",
            "2001-09-12",
            "2001-09-13",
            "2001-09-14",
            "2004-06-11",
            "2007-01-02",
            "2012-10-29",
            "2012-10-30",
            "2018-12-05",
        ]
    )
    hol = NYSE().holidays(first, last)
    days = pd.bdate_range(first, last)
    return days[~days.isin(hol) & ~days.isin(one_off)]


EARLY = pd.DatetimeIndex(pd.to_datetime(list(asl.EARLY_CLOSE_DATES)))


# --------------------------------------------------------------------------- data


def _day_index(t: pd.Series) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(t.dt.normalize().to_numpy()).as_unit("ns")


def truth_history() -> tuple[pd.Series, pd.Series]:
    """(every finite 16:00 print by day, the scoring history: early closes and non-positive prints dropped)."""
    c = pd.read_parquet(CORE, columns=["endbartime", "sumret2"])
    et = pd.to_datetime(c["endbartime"])
    assert et.dt.tz is None, "core_stats stamps are naive ET"
    m = ((et.dt.hour == 16) & (et.dt.minute == 0)).to_numpy()
    s = pd.Series(c.loc[m, "sumret2"].to_numpy(float), index=_day_index(et[m])).sort_index()
    assert not s.index.duplicated().any()
    s = s[np.isfinite(s.to_numpy())]
    hist = s[(s.to_numpy() > 0) & ~s.index.isin(EARLY)]
    return s, hist


def read_key(k: Key) -> pd.DataFrame:
    """16:00 rows of a key: day-indexed yhat, B, the table's own rv_raw (NaN for the arm file), finite yhat only."""
    if k.kind == "arm":
        r = pd.read_csv(k.path, parse_dates=["date"]).set_index("date").sort_index()
        r = r[(r["true_adj"] > 0) & (r["true_raw"] > 0)]  # compare_mfiv_harlag._prep
        assert (r.index.strftime("%H:%M") == "16:00").all()
        f = pd.DataFrame(
            {
                "yhat": r["pred_adj"].to_numpy(float),
                "B": (r["true_raw"] / r["true_adj"] ** 2).to_numpy(float),
                "rv_raw": np.nan,
            },
            index=pd.DatetimeIndex(r.index.normalize()).as_unit("ns"),
        )
        n16 = len(r)
    else:
        d = pd.read_parquet(k.path, columns=["t", "yhat", "baseline", "rv_raw"])
        t = pd.to_datetime(d["t"])
        assert t.dt.tz is not None, k.key
        et = t.dt.tz_convert("America/New_York").dt.tz_localize(None)
        m = ((et.dt.hour == 16) & (et.dt.minute == 0)).to_numpy()
        f = pd.DataFrame(
            {
                "yhat": d["yhat"].to_numpy(float)[m],
                "B": d["baseline"].to_numpy(float)[m],
                "rv_raw": d["rv_raw"].to_numpy(float)[m],
            },
            index=_day_index(et[m]),
        )
        n16 = int(m.sum())
    assert not f.index.duplicated().any(), k.key
    f = f.sort_index()
    f.attrs["n16"] = n16
    f.attrs["first16"], f.attrs["last16"] = f.index.min(), f.index.max()
    f = f[np.isfinite(f["yhat"].to_numpy())]
    assert np.isfinite(f["B"]).all() and (f["B"] > 0).all(), k.key
    f["plain"] = f["yhat"] ** 2 * f["B"]
    return f


# --------------------------------------------------------------------------- statistics


def ct_benchmark(hist: pd.Series, days: pd.DatetimeIndex) -> pd.Series:
    """RVbar_{t-1}: mean of hist strictly before t (expanding, causal); = make_rehaul_oos_table.ct_benchmark."""
    prior = hist.cumsum().shift(1) / pd.Series(np.arange(len(hist), dtype=float), index=hist.index)
    b = prior.replace([np.inf], np.nan).reindex(days)
    hv, hi = hist.to_numpy(float), hist.index
    cs = np.concatenate([[0.0], np.cumsum(hv)])
    for t, v in b.items():  # brute force: the value at t uses days < t only
        n = int(np.searchsorted(hi, t, side="left"))
        assert n > 0 and hi[n - 1] < t and np.isclose(v, hv[:n].mean(), rtol=1e-12, atol=0), t
        assert np.isclose(cs[n] / n, v, rtol=1e-12, atol=0), t
    return b


def qlike_daily(rv: np.ndarray, plain: np.ndarray) -> np.ndarray:
    r = rv / plain
    return r - np.log(r) - 1.0


def metrics(rv: pd.Series, plain: pd.Series, bench: pd.Series) -> dict[str, float]:
    e2 = (rv - plain) ** 2
    return {
        "mse": float(e2.mean()),
        "qlike": float(qlike_daily(rv.to_numpy(float), plain.to_numpy(float)).mean()),
        "r2_oos_pct": float(100.0 * (1.0 - e2.sum() / ((rv - bench.reindex(rv.index)) ** 2).sum())),
    }


def dm_pair(rv: pd.Series, a: pd.Series, b: pd.Series) -> dict[str, float]:
    """dm_test (as is) of key a against key b on the given days: QLIKE and squared-error loss, a minus b."""
    x, y, v = a.to_numpy(float), b.to_numpy(float), rv.to_numpy(float)
    q = dm_test(qlike_daily(v, x), qlike_daily(v, y))
    s = dm_test((v - x) ** 2, (v - y) ** 2)
    return {
        "dm_qlike": float(q["dm"]),
        "dm_qlike_p": float(q["p"]),
        "dm_qlike_mean_diff": float(q["mean_diff"]),
        "dm_se": float(s["dm"]),
        "dm_se_p": float(s["p"]),
        "dm_T": int(q["T"]),
        "dm_hac_lag": int(q.get("hac_lag", -1)),
    }


# --------------------------------------------------------------------------- API for the paper's generators
# (writeup/make_rehaul_oos_table.py, writeup/make_rehaul_results.py: every accuracy number on the common rolling span)

_MASTER: dict[str, tuple[Path, str]] = {}
_HIST: list[pd.Series] = []
_COMMON: list[pd.DatetimeIndex] = []


def master_source(key: str) -> tuple[Path, str]:
    """(path, kind) of a master-table key as experiments/master_table_close.py reads it in the paper run."""
    if not _MASTER:
        import master_table_close as mtc

        assert mtc.ARM_ROOT == "linear_subsection_dedup" and mtc.perbar_switch_active()
        _MASTER.update({s.key: (Path(s.path), s.source) for s in mtc.discover()[0]})
    return _MASTER[key]


def history() -> pd.Series:
    """The scoring truth by day: the closing-half-hour RV, early closes and non-positive prints dropped (cached)."""
    if not _HIST:
        _HIST.append(truth_history()[1])
    return _HIST[0]


def scored(key: str) -> pd.DataFrame:
    """plain (yhat^2 B) and the actual RV of a forecast key on its scored days (16:00 rows, early closes and missing
    prints dropped): a key of tab:rh_oos (table_keys), else any key of the master table (master_source)."""
    tk = {k.key: k for k in table_keys()}
    if key in tk:
        k = tk[key]
    else:
        path, kind = master_source(key)
        k = Key(key, "", "", "", path, kind, "ols")
    f = read_key(k)
    hist = history()
    f = f[~f.index.isin(EARLY) & f.index.isin(hist.index)]
    return pd.DataFrame({"plain": f["plain"], "rv": hist.reindex(f.index)})


def common_span() -> pd.DatetimeIndex:
    """The common rolling span: the days on which every forecast of tab:rh_oos has a scored forecast (cached)."""
    if not _COMMON:
        days: pd.DatetimeIndex | None = None
        for k in table_keys():
            idx = scored(k.key).index
            days = idx if days is None else days.intersection(idx)
        assert days is not None
        _COMMON.append(pd.DatetimeIndex(days.sort_values()).as_unit("ns"))
    return _COMMON[0]


# --------------------------------------------------------------------------- main


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    keys = table_keys()
    gates: list[dict] = []

    def gate(name: str, item: str, value: object, ok: bool) -> None:
        gates.append({"gate": name, "item": item, "value": value, "ok": bool(ok)})
        assert ok, (name, item, value)

    # the sources are the ones the master table reads (paper run: dedup arm root, switches active)
    import master_table_close as mtc

    assert mtc.ARM_ROOT == "linear_subsection_dedup" and mtc.perbar_switch_active()
    specs = {s.key: s for s in mtc.discover()[0]}
    for k in keys:
        if k.key in specs:
            got = Path(specs[k.key].path).resolve()
            gate(
                "source = master_table_close.discover()",
                k.key,
                str(got.relative_to(REPO)),
                got == k.path.resolve(),
            )

    # truth and calendar
    raw16, hist = truth_history()
    nyse = nyse_sessions("1998-01-01", "2024-04-30")
    core_days = raw16.index[raw16.index >= hist.index[0]]
    miss_core = nyse[(nyse >= hist.index[0]) & ~nyse.isin(raw16.index)]
    extra_core = core_days[~core_days.isin(nyse)]
    gate(
        "core_stats 16:00 prints on non-NYSE days (from 1998-01-05)",
        "core_stats",
        len(extra_core),
        len(extra_core) == 0,
    )
    print(
        f"truth: {hist.index[0].date()} .. {hist.index[-1].date()}, {len(hist)} scoring prints; NYSE sessions "
        f"without a finite core_stats 16:00 print since {hist.index[0].date()}: {len(miss_core)} "
        f"{[str(d.date()) for d in miss_core]}"
    )

    # forecasts
    data: dict[str, pd.DataFrame] = {}
    span_rows: list[dict] = []
    for k in keys:
        f = read_key(k)
        n_fin = len(f)
        first_fc, last_fc = f.index.min(), f.index.max()
        off = f.index[~f.index.isin(nyse)]
        gate("16:00 forecasts on NYSE sessions only", k.key, len(off), len(off) == 0)
        ec = f.index.isin(EARLY)
        f = f[~ec]
        has_truth = f.index.isin(hist.index)
        no_truth = f.index[~has_truth]
        f = f[has_truth]
        f["rv"] = hist.reindex(f.index)
        if k.kind == "table":
            rel = float(np.max(np.abs(f["rv_raw"] / f["rv"] - 1.0)))
            gate(
                "table rv_raw = core_stats 16:00 sumret2 on every scored day",
                k.key,
                rel,
                rel == 0.0,
            )
        exp = nyse[(nyse >= f.index.min()) & (nyse <= f.index.max()) & ~nyse.isin(EARLY)]
        gaps = exp[~exp.isin(f.index)]
        data[k.key] = f[["plain", "rv"]]
        hc = k.hp_clean_from
        clean = f.index if hc is None else f.index[f.index >= hc]
        span_rows.append(
            {
                "key": k.key,
                "model": k.model,
                "set": k.pset,
                "scheme": k.scheme,
                "source": str(k.path.relative_to(REPO)).replace("\\", "/"),
                "kind": k.kind,
                "n_16h_rows": f.attrs["n16"],
                "first_16h_row": f.attrs["first16"].date(),
                "last_16h_row": f.attrs["last16"].date(),
                "n_finite_forecasts": n_fin,
                "first_forecast": first_fc.date(),
                "last_forecast": last_fc.date(),
                "n_early_close_dropped": int(ec.sum()),
                "n_no_truth_dropped": len(no_truth),
                "no_truth_days": ";".join(str(d.date()) for d in no_truth),
                "first_scored": f.index.min().date(),
                "last_scored": f.index.max().date(),
                "n_scored": len(f),
                "nyse_sessions_in_span": len(exp),
                "n_gaps": len(gaps),
                "gap_days": ";".join(str(d.date()) for d in gaps),
                "hp_selection": HP_NOTE[k.hp],
                "hp_clean_from": "" if hc is None else hc.date(),
                "first_scored_hp_clean": clean.min().date() if len(clean) else "",
                "n_scored_hp_clean": len(clean),
            }
        )
    spans = pd.DataFrame(span_rows)

    # the master table's 866 days: plain, truth and QLIKE reproduce it
    mtab = pd.read_csv(MT / "master_table.csv").set_index("key")
    daily = pd.read_parquet(
        MT / "master_table_daily.parquet",
        columns=["key", "day", "plain", "rv_unclipped", "qlike_raw"],
    )
    daily["day"] = _day_index(pd.to_datetime(daily["day"]))
    mt_days = pd.DatetimeIndex(daily.loc[daily["key"] == OLS, "day"].sort_values().to_numpy()).as_unit("ns")
    gate(
        "master days",
        OLS,
        f"{len(mt_days)} {mt_days[0].date()} .. {mt_days[-1].date()}",
        len(mt_days) == MT_N and mt_days[0] == MT_FIRST and mt_days[-1] == MT_LAST,
    )
    for k in keys:
        f = data[k.key]
        gate(
            "master days inside the key's scored days",
            k.key,
            int((~mt_days.isin(f.index)).sum()),
            bool(mt_days.isin(f.index).all()),
        )
        on = f.reindex(mt_days)
        if k.key in mtab.index:
            g = daily[daily["key"] == k.key].set_index("day").sort_index()
            assert g.index.equals(mt_days), k.key
            rel = float(np.max(np.abs(on["plain"] / g["plain"] - 1.0)))
            gate(
                "plain = master_table_daily plain (866 days, rel)",
                k.key,
                rel,
                rel <= PLAIN_RTOL,
            )
            gate(
                "truth = master_table_daily rv_unclipped (866 days)",
                k.key,
                0,
                np.array_equal(on["rv"].to_numpy(), g["rv_unclipped"].to_numpy()),
            )
            qd = qlike_daily(on["rv"].to_numpy(float), on["plain"].to_numpy(float))
            gate(
                "daily QLIKE = master_table_daily qlike_raw",
                k.key,
                float(np.max(np.abs(qd - g["qlike_raw"]))),
                bool(np.allclose(qd, g["qlike_raw"], rtol=1e-10, atol=1e-14)),
            )
            q = float(qd.mean())
            gate(
                "QLIKE = master_table.csv qlike_raw (866 days)",
                k.key,
                q - float(mtab.loc[k.key, "qlike_raw"]),
                bool(np.isclose(q, mtab.loc[k.key, "qlike_raw"], rtol=QLIKE_RTOL, atol=0)),
            )
        else:  # pooltree_*: make_rehaul_oos_table.plain_16 is f^2 B on the table's own 16:00 rows -- the same rule
            gate(
                "pooltree key: forecast on every master day",
                k.key,
                int(on["plain"].isna().sum()),
                bool(on["plain"].notna().all()),
            )

    # the common span of the table
    common = data[keys[0].key].index
    for k in keys[1:]:
        common = common.intersection(data[k.key].index)
    common = pd.DatetimeIndex(common.sort_values()).as_unit("ns")
    exp = nyse[(nyse >= common.min()) & (nyse <= common.max()) & ~nyse.isin(EARLY)]
    c_gaps = exp[~exp.isin(common)]
    binding_first = [k.key for k in keys if data[k.key].index.min() == common.min()]
    binding_last = [k.key for k in keys if data[k.key].index.max() == common.max()]
    pd.DataFrame(
        [
            {
                "first": common.min().date(),
                "last": common.max().date(),
                "n_days": len(common),
                "nyse_sessions_ex_early_close": len(exp),
                "n_gaps": len(c_gaps),
                "gap_days": ";".join(str(d.date()) for d in c_gaps),
                "n_master_days_inside": int(mt_days.isin(common).sum()),
                "n_non_master_days": int((~common.isin(mt_days)).sum()),
                "keys_starting_at_first": ";".join(binding_first),
                "keys_ending_at_last": ";".join(binding_last),
                "n_keys": len(keys),
            }
        ]
    ).to_csv(OUT / "common_span.csv", index=False)
    gate(
        "master days inside the common span",
        "common",
        int((~mt_days.isin(common)).sum()),
        bool(mt_days.isin(common).all()),
    )
    gate(
        "common span = common_span() (the generators' API)",
        "common",
        len(common),
        bool(common.equals(common_span())),
    )
    for k in keys:
        a, b = scored(k.key), data[k.key]
        gate(
            "scored() = this run's frame",
            k.key,
            len(a),
            bool(a.index.equals(b.index) and np.array_equal(a.to_numpy(), b.to_numpy())),
        )

    # benchmark on every scored day of every key
    all_days = common
    for k in keys:
        all_days = all_days.union(data[k.key].index)
    gate(
        "every scored day is a history day",
        "all",
        int((~all_days.isin(hist.index)).sum()),
        bool(all_days.isin(hist.index).all()),
    )
    bench = ct_benchmark(hist, pd.DatetimeIndex(all_days))

    # metrics per key x span
    ols = data[OLS]
    rows: list[dict] = []
    for k in keys:
        f = data[k.key]
        hc = k.hp_clean_from
        sets = {
            "own": f.index,
            "own_clean": f.index if hc is None else f.index[f.index >= hc],
            "common": common,
            "mt866": mt_days,
        }
        for span, days in sets.items():
            if not len(days):
                continue
            g = f.reindex(days)
            assert g["plain"].notna().all(), (k.key, span)
            m = metrics(g["rv"], g["plain"], bench)
            paired = days[days.isin(ols.index)]
            dm = dm_pair(
                g["rv"].reindex(paired),
                g["plain"].reindex(paired),
                ols["plain"].reindex(paired),
            )
            rows.append(
                {
                    "key": k.key,
                    "model": k.model,
                    "set": k.pset,
                    "scheme": k.scheme,
                    "span": span,
                    "first": days.min().date(),
                    "last": days.max().date(),
                    "n": len(days),
                    "mse": m["mse"],
                    f"mse_e{MSE_EXP}": m["mse"] * 10.0**MSE_EXP,
                    "qlike": m["qlike"],
                    "r2_oos_pct": m["r2_oos_pct"],
                    "dm_first": paired.min().date(),
                    "dm_last": paired.max().date(),
                    **dm,
                    "hp_clean": bool(hc is None or days.min() >= hc),
                }
            )
            if span == "mt866" and k.key in mtab.index:
                gate(
                    "mt866 QLIKE = master_table.csv qlike_raw",
                    k.key,
                    m["qlike"],
                    bool(
                        np.isclose(
                            m["qlike"],
                            mtab.loc[k.key, "qlike_raw"],
                            rtol=QLIKE_RTOL,
                            atol=0,
                        )
                    ),
                )
    met = pd.DataFrame(rows)

    spans.to_csv(OUT / "spans.csv", index=False)
    met.to_csv(OUT / "metrics.csv", index=False)
    pd.DataFrame(gates).to_csv(OUT / "gates.csv", index=False)

    # console
    pd.set_option("display.width", 250)
    print(
        spans[
            [
                "key",
                "first_forecast",
                "last_forecast",
                "n_finite_forecasts",
                "n_early_close_dropped",
                "n_no_truth_dropped",
                "n_scored",
                "n_gaps",
                "first_scored_hp_clean",
                "n_scored_hp_clean",
            ]
        ].to_string(index=False)
    )
    print(
        f"common span: {common.min().date()} .. {common.max().date()}, {len(common)} days "
        f"({int(mt_days.isin(common).sum())} master days + {int((~common.isin(mt_days)).sum())} others), "
        f"{len(c_gaps)} NYSE gaps; first set by {binding_first}"
    )
    piv = met.pivot_table(
        index="key",
        columns="span",
        values=["qlike", "r2_oos_pct", "dm_qlike"],
        sort=False,
    )
    print(piv.round(4).to_string())
    print(
        f"{len(gates)} gates, all ok; wrote {OUT.relative_to(REPO)}/spans.csv, metrics.csv, common_span.csv, gates.csv"
    )


if __name__ == "__main__":
    main()
