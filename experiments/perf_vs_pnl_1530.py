"""Model performance vs P&L of the 15:30 last-30-min trade (checklist C3).

Question: does a better 16:00-bar variance forecast (lower QLIKE / MSE) make more money
in the 15:30 sign(s) straddle trade, and if not, which property of the forecast does?

Universe (the brief: "use the master table's rows when it exists"): the table-A rows of
the closing-strategy master table (results/close_master_table/master_table.csv, written by
experiments/master_table_close.py) -- every forecast on the same 866 deck days, check rows
(the rest-of-day tables, identical to the per-bar arms at 16:00 by construction), exact
duplicates and the always-short rule left out -- read day by day from its
master_table_daily.parquet.  Fallback when that table is absent: every forecast table
results/spxw_pnl/yhat_*.parquet put through the same scorer here (exact duplicates left
out).

ONE scorer for all: the RESEARCH scorer's 16:00-bar recalibration
(experiments/score_linear_subsection_causal.causal_forecasts: forecast = (yhat^2 + s) x B,
s = the forecast's own trailing-250-session mean squared error at 16:00, lagged one
session) and the deck's 15:30 sign(s) trade (score_linear_subsection.trade_1530: buy the
straddle when the forecast exceeds the 15:30 implied variance iv_var, sell otherwise; mid
and crossed fills).  The realized series every loss uses is the master table's 16:00
target (the per-bar spec's target; the fallback uses the tables' realized variance).
Near-zero adjusted-scale forecasts need no treatment here: the recalibrated forecast is
at least s x B (the master table flags none below its floor on a trade day).

The straddle: nearest out-of-the-money call + nearest out-of-the-money put, same-day
expiry, one position, entered at 15:30, held to the close.

Parts
  B1  cross-model rank relation (Spearman over forecasts) of Sharpe with QLIKE, MSE and
      sign accuracy; 95 % intervals from a circular block bootstrap over DAYS (the whole
      panel of forecasts resampled together, block 21 sessions as the research scorer).
  B2  which loss ranks the forecasts the way the Sharpe does: sign accuracy overall and
      on the TAIL days (the k days with the largest |straddle return|, k = 20, 50),
      QLIKE on the tail days, calibration (log Mincer-Zarnowitz slope, |slope - 1|).
  B3  per day, per forecast: the trade return on the forecast's log error and on its
      sign correctness, HAC (Newey-West) standard errors; R^2 = the share of the
      day-to-day P&L variance explained.
  B4  within each model family of the master table (the 16:00 ladder's tree rungs T10,
      T1, RS10, RS1 and the Optuna rungs; the per-bar and intraday-sequence LSTM; the
      per-bar linear families; the 48-bar ones), the rank agreement of QLIKE and of the
      tail-day sign accuracy with the Sharpe, with the same day-block intervals, and
      their mean over the families; paired differences of agreements (same draws).

The summary's qualitative claims are evaluated as checks on the numbers (asserted ones
stop the summary; the others choose the prose), on this run and on the committed
outputs of the 106-forecast run (PREV_COMMIT), and the two are set side by side.

Outputs (results/perf_vs_pnl/): forecast_metrics.csv, rank_vs_sharpe.csv,
rank_vs_sharpe_by_family.csv, rank_vs_sharpe_by_model_family.csv,
rank_vs_sharpe_differences.csv, per_day_regressions.csv, gates.csv, qlike_vs_sharpe.png,
rank_vs_sharpe.png, rank_vs_sharpe_by_model_family.png, claims.csv, numbers.json (every
number the SUMMARY quotes), run_output.txt.

Run:  python experiments/perf_vs_pnl_1530.py
"""

from __future__ import annotations

import io
import json
import re
import subprocess
import sys
import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "experiments", REPO / "notebooks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import atm_straddle_lib as asl  # noqa: E402
import score_linear_subsection as base  # noqa: E402
import score_linear_subsection_causal as slc  # noqa: E402

OUT = REPO / "results" / "perf_vs_pnl"
YHAT = REPO / "results" / "spxw_pnl"
MASTER = REPO / "results" / "close_master_table" / "master_table.csv"
MASTER_DAILY = REPO / "results" / "close_master_table" / "master_table_daily.parquet"
ET = "America/New_York"
TAIL_K = (20, 50)
B = 2000
SEED = 20260929
BLOCK = base.BOOT_BLOCK  # 21 sessions, the research scorer's block
ANN = base.ANN
# the smallest set of forecasts a within-set rank correlation is reported for (the
# classes' threshold since the first run; the smallest master-table family has 6)
MIN_SET = 5
# per-day regressions: a count of forecasts with p < 0.05 up to this share of them is
# read as chance (twice the test size, the allowance the first run's check used)
CHANCE_SHARE = 0.10
# the committed run whose claims this run re-checks: the 106-forecast master table
# (H2, before the 16:00 campaign); its outputs are read from git, never from disk
PREV_COMMIT = "6177b74"
PREV_DIR = "results/perf_vs_pnl"


def nw_lag(n: int) -> int:
    """Newey-West rule-of-thumb lag floor(4 (n/100)^(2/9)) for the HAC regressions."""
    return int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))


# forecasts drawn with a label in the scatter (the ones the write-up names)
KEY = {
    "sub_ridge_live_feasible": "per-bar ridge, live-feasible (headline)",
    "sub_lasso_live_feasible": "per-bar lasso, live-feasible",
    "sub_ridge_vix_only": "per-bar ridge, VIX only",
    "subtree_lgbm_live_feasible": "per-bar LightGBM, live-feasible",
    "blk2": "block ridge (paper reference)",
    "sub_enet_live_feasible_har_base3": "per-bar elastic net, base-3 HAR (best QLIKE)",
}
# display classes for the scatter (the master table's families, grouped to 6 slots)
TREES = "per-bar trees"
LSTM_CLASS = "LSTM (per-bar and intraday sequence)"
CLASS_OF = {
    "paper": "48-bar (paper + pooled twins)",
    "pooled twin": "48-bar (paper + pooled twins)",
    "per-bar linear": "per-bar linear",
    "VIX-only family": "per-bar linear",
    "implied-vol representations": "per-bar linear",
    "HAR-ladder variants": "per-bar linear, HAR-ladder variants",
    "LSTM": LSTM_CLASS,
    "LSTM (intraday sequence)": LSTM_CLASS,
    "other table": TREES,
}
CLASS_ORDER = (
    "48-bar (paper + pooled twins)",
    "per-bar linear",
    "per-bar linear, HAR-ladder variants",
    TREES,
    LSTM_CLASS,
    "other",
)
# validated categorical slots (the palette of model_diagnostics_1530_trees.py; the
# LSTM rows keep the slot they had as 'other' in the 106-forecast run; magenta = the
# reference palette's slot 5 for anything unmapped); marker shape is the second code
CLASS_COLOR = dict(
    zip(
        CLASS_ORDER,
        ("#2a78d6", "#4a3aa7", "#1baf7a", "#eb6834", "#eda100", "#e87ba4"),
    )
)
CLASS_MARK = dict(zip(CLASS_ORDER, ("s", "o", "D", "^", "v", "P")))
QLIKE_XMAX = 0.125  # the scatter's x range; forecasts beyond it are drawn at the edge

# the 16:00 ladder's tree rungs (writeup/CAMPAIGN_16H_2026-09-29.md) by master-table family
TREE_RUNG = {
    "per-bar tree (untuned)": "T10",
    "per-bar tree (untuned, daily refit)": "T1",
    "per-bar tree (tuned)": "RS10",
    "per-bar tree (random search, daily)": "RS1",
}
OPTUNA_FAM = re.compile(r"per-bar tree \(Optuna, TUNE_PER=(\d+), best-of-(\d+)\)")
# order of the master-table families in the within-family table (unlisted ones last)
FAMILY_ORDER = (
    "per-bar linear",
    "VIX-only family",
    "implied-vol representations",
    "HAR-ladder variants",
    *TREE_RUNG,
    "LSTM",
    "LSTM (intraday sequence)",
    "paper",
    "pooled twin",
)


def classify(fam: str, key: str) -> str:
    if fam in CLASS_OF:
        return CLASS_OF[fam]
    if fam.startswith("per-bar tree") or key.startswith("subtree_"):
        return TREES
    return "other"


def rung(fam: str) -> str:
    """The ladder rung of a tree family (T10 / T1 / RS10 / RS1 / OP_tp<N>_k<k>), else ''."""
    if fam in TREE_RUNG:
        return TREE_RUNG[fam]
    m = OPTUNA_FAM.fullmatch(fam)
    return f"OP_tp{m[1]}_k{m[2]}" if m else ""


def _family_key(fam: str) -> tuple[int, int, int, int]:
    rs1 = FAMILY_ORDER.index("per-bar tree (random search, daily)")
    if fam in FAMILY_ORDER:
        return (FAMILY_ORDER.index(fam), 0, 0, 0)
    m = OPTUNA_FAM.fullmatch(fam)
    if m:  # after the shipped / random-search rungs: best-of-50 first, then TUNE_PER
        return (rs1, 1, -int(m[2]), int(m[1]))
    return (len(FAMILY_ORDER), 0, 0, 0)


def model_family_sets(fam: np.ndarray, cls: np.ndarray) -> list[dict]:
    """The within-family sets of B4: every master-table family with >= MIN_SET forecasts,
    plus the pooled sets (each display class of more than one family; every Optuna rung)."""
    out: list[dict] = []
    for c in CLASS_ORDER:
        fams = sorted(dict.fromkeys(fam[cls == c]), key=_family_key)
        for f in fams:
            if (fam == f).sum() >= MIN_SET:
                out.append(dict(set=f, kind="family", rung=rung(f), sel=fam == f))
        if c == TREES:
            op = np.array([bool(OPTUNA_FAM.fullmatch(f)) for f in fam])
            if sum(bool(OPTUNA_FAM.fullmatch(f)) for f in fams) > 1:
                out.append(
                    dict(set="every Optuna rung", kind="pooled", rung="OP", sel=op)
                )
        if len(fams) > 1 and (cls == c).sum() >= MIN_SET:
            # named apart from a family of the same name (the class "per-bar linear")
            out.append(
                dict(set=f"{c}, all families", kind="class", rung="", sel=cls == c)
            )
    return out


# ---------------------------------------------------------------------------- panels
def panel_from_master() -> tuple[pd.DataFrame, dict, list[dict]]:
    """The master table's table-A rows, day by day."""
    mt = pd.read_csv(MASTER)
    n_trade = int(mt["n_days"].max())
    keep = (
        (mt["n_days"] == n_trade)
        & ~mt["family"].astype(str).str.contains("check")
        & (mt["family"] != "rule")
        & mt["duplicate_of"].isna()
    )
    meta = mt[keep].set_index("key")
    d = pd.read_parquet(MASTER_DAILY)
    d = d[d["key"].isin(meta.index)]
    names = list(meta.index)
    piv = {
        c: d.pivot(index="day", columns="key", values=c)[names]
        for c in (
            "pred_clock",
            "target",
            "iv_var",
            "q",
            "ret_mid",
            "ret_crossed",
            "qlike_recal",
        )
    }
    days = piv["q"].index
    assert len(days) == n_trade and all(p.notna().all().all() for p in piv.values())
    F, RV = piv["pred_clock"].to_numpy(float), piv["target"].to_numpy(float)
    iv, q = piv["iv_var"].to_numpy(float), piv["q"].to_numpy(float)
    pnl, pnlx = piv["ret_mid"].to_numpy(float), piv["ret_crossed"].to_numpy(float)
    R = pnl * q  # the straddle's own mid return, the same for every forecast
    gates = [
        dict(
            check="q = sign(pred_clock - iv_var) as trade_1530",
            max_abs=float(np.abs(np.where(F > iv, 1.0, -1.0) - q).max()),
        ),
        dict(
            check="straddle return identical across forecasts",
            max_abs=float(np.abs(R - R[:, [0]]).max()),
        ),
        dict(
            check="iv_var identical across forecasts",
            max_abs=float(np.abs(iv - iv[:, [0]]).max()),
        ),
        dict(
            check="target identical across forecasts (rel)",
            max_abs=float(np.abs(RV / RV[:, [0]] - 1).max()),
        ),
    ]
    ratio = RV / F
    ql = ratio - np.log(ratio) - 1.0
    gates.append(
        dict(
            check="per-day QLIKE = master qlike_recal",
            max_abs=float(np.abs(ql - piv["qlike_recal"].to_numpy()).max()),
        )
    )
    P = _panel(F, RV, iv, q, R, pnl, pnlx)
    info = pd.DataFrame(
        {
            "label": meta["label"],
            "family": meta["family"],
            "master_Sharpe_mid": meta["Sharpe_mid"],
            "master_Sharpe_crossed": meta["Sharpe_crossed"],
            "master_qlike_recal": meta["qlike_recal"],
        },
        index=names,
    )
    return (
        info,
        {"P": P, "days": days, "source": "master", "n_rows_master": int(len(mt))},
        gates,
    )


def load16(path: Path) -> pd.DataFrame:
    """The 16:00 rows of a forecast table, put through the research scorer's recalibration."""
    d = pd.read_parquet(path, columns=["t", "yhat", "baseline", "rv_raw"])
    t = pd.DatetimeIndex(d["t"]).tz_convert(ET).tz_localize(None).as_unit("ns")
    m = np.asarray((t.hour == 16) & (t.minute == 0))
    r = pd.DataFrame(
        {
            "pred_adj": d["yhat"].to_numpy(float)[m],
            "baseline": d["baseline"].to_numpy(float)[m],
            "true_raw": d["rv_raw"].to_numpy(float)[m],
        },
        index=t[m],
    ).sort_index()
    r = r[(r["true_raw"] > 0) & (r["baseline"] > 0) & r["pred_adj"].notna()].copy()
    r["true_adj"] = np.sqrt(r["true_raw"] / r["baseline"])
    r["e2"] = (r["true_adj"] - r["pred_adj"]) ** 2
    r["hhmm"] = r.index.strftime("%H:%M")
    r["day"] = r.index.normalize()
    return slc.causal_forecasts(r)


def panel_from_tables() -> tuple[pd.DataFrame, dict, list[dict]]:
    """Fallback: every yhat table on disk, the same scorer; exact duplicates left out."""
    dk = pd.read_parquet(base.DECK).sort_index()
    dk.index = pd.DatetimeIndex(pd.to_datetime(dk.index)).normalize().as_unit("ns")
    files = sorted(
        YHAT.glob("yhat_*.parquet"),
        key=lambda f: (f.stem.startswith("yhat_restofday"), f.stem),
    )
    fc: dict[str, pd.DataFrame] = {}
    seen: dict[str, np.ndarray] = {}
    gates = []
    for f in files:
        stem = f.stem.removeprefix("yhat_")
        r = load16(f)
        on = r["pred_clock"].set_axis(r.index.normalize()).reindex(dk.index)
        if on.notna().sum() < len(dk):
            continue
        ya = (
            r["pred_adj"]
            .set_axis(r.index.normalize())
            .reindex(dk.index)
            .to_numpy(float)
        )
        if any(np.max(np.abs(v / ya - 1)) < 1e-9 for v in seen.values()):
            continue
        seen[stem] = ya
        fc[stem] = r
        gates.append(
            dict(
                check=f"trade_1530 Sharpe {stem}",
                max_abs=0.0,
                sharpe=base.trade_1530(r["pred_clock"])["Sharpe_mid"],
            )
        )
    names = list(fc)
    F = np.column_stack(
        [
            fc[k]["pred_clock"]
            .set_axis(fc[k].index.normalize())
            .reindex(dk.index)
            .to_numpy(float)
            for k in names
        ]
    )
    RV = np.column_stack(
        [
            fc[k]["true_raw"]
            .set_axis(fc[k].index.normalize())
            .reindex(dk.index)
            .to_numpy(float)
            for k in names
        ]
    )
    iv = np.broadcast_to(dk["iv_var"].to_numpy(float)[:, None], F.shape)
    q = np.where(F > iv, 1.0, -1.0)
    R = np.broadcast_to(dk["R"].to_numpy(float)[:, None], F.shape)
    ex = dk["exit"].to_numpy(float)[:, None]
    ask = (dk["ask_c"] + dk["ask_p"]).to_numpy(float)[:, None]
    bid = (dk["bid_c"] + dk["bid_p"]).to_numpy(float)[:, None]
    P = _panel(
        F, RV, iv, q, R, q * R, q * np.where(q > 0, ex / ask - 1.0, ex / bid - 1.0)
    )
    fam = [
        "paper"
        if not k.startswith(("sub", "pool"))
        else (
            "per-bar tree (untuned)"
            if k.startswith("subtree")
            else ("pooled twin" if k.startswith("pool") else "per-bar linear")
        )
        for k in names
    ]
    info = pd.DataFrame({"label": names, "family": fam}, index=names)
    return info, {"P": P, "days": dk.index, "source": "tables"}, gates


def _panel(F, RV, iv, q, R, pnl, pnlx) -> dict:
    ratio = RV / F
    return dict(
        F=F,
        RV=RV,
        iv=iv,
        q=q,
        R=R,
        pnl=pnl,
        pnlx=pnlx,
        ql=ratio - np.log(ratio) - 1.0,
        se=(RV - F) ** 2,
        lerr=np.log(F) - np.log(RV),
        sgn=(q == np.sign(RV - iv)).astype(float),
        hit=(pnl > 0).astype(float),
    )


# ---------------------------------------------------------------------------- metrics
def wmean(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return (w[:, None] * x).sum(0) / w.sum()


def sharpe(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    n = w.sum()
    m = (w[:, None] * x).sum(0) / n
    v = (w[:, None] * (x - m) ** 2).sum(0) / (n - 1)
    return m / np.sqrt(v) * ANN


def mz_slope(y: np.ndarray, x: np.ndarray, w: np.ndarray) -> np.ndarray:
    mx, my = wmean(x, w), wmean(y, w)
    return wmean((x - mx) * (y - my), w) / wmean((x - mx) ** 2, w)


def metrics(
    P: dict, w: np.ndarray, tails: dict[int, np.ndarray]
) -> dict[str, np.ndarray]:
    """Every per-forecast metric under day weights w (ones: the sample; bootstrap
    multiplicities: one draw)."""
    out = {
        "Sharpe_mid": sharpe(P["pnl"], w),
        "Sharpe_crossed": sharpe(P["pnlx"], w),
        "QLIKE": wmean(P["ql"], w),
        "MSE": wmean(P["se"], w),
        "MSE_log": wmean(P["lerr"] ** 2, w),
        "sign_acc": wmean(P["sgn"], w),
        "hit_rate": wmean(P["hit"], w),
        "mz_log_slope": mz_slope(np.log(P["RV"]), np.log(P["F"]), w),
        "mean_pnl_mid": wmean(P["pnl"], w),
    }
    out["calib_err"] = np.abs(out["mz_log_slope"] - 1.0)
    for k, tm in tails.items():
        wt = w * tm
        out[f"sign_acc_top{k}"] = wmean(P["sgn"], wt)
        out[f"hit_top{k}"] = wmean(P["hit"], wt)
        out[f"QLIKE_top{k}"] = wmean(P["ql"], wt)
        out[f"QLIKE_rest{k}"] = wmean(P["ql"], w * (1 - tm))
        out[f"sign_acc_rest{k}"] = wmean(P["sgn"], w * (1 - tm))
    return out


# which way a metric should rank the forecasts if it tracks the money
DIRECTION = {
    "QLIKE": -1,
    "MSE": -1,
    "MSE_log": -1,
    "calib_err": -1,
    "sign_acc": 1,
    "hit_rate": 1,
    **{f"sign_acc_top{k}": 1 for k in TAIL_K},
    **{f"sign_acc_rest{k}": 1 for k in TAIL_K},
    **{f"hit_top{k}": 1 for k in TAIL_K},
    **{f"QLIKE_top{k}": -1 for k in TAIL_K},
    **{f"QLIKE_rest{k}": -1 for k in TAIL_K},
}


def descr(m: str, n: int) -> str:
    base_ = {
        "QLIKE": "QLIKE",
        "MSE": "MSE (variance units)",
        "MSE_log": "mean squared log error",
        "calib_err": "calibration error |log MZ slope - 1|",
        "sign_acc": "sign accuracy, all days",
        "hit_rate": "hit rate (trade made money), all days",
    }
    if m in base_:
        return base_[m]
    for k in TAIL_K:
        if m == f"sign_acc_top{k}":
            return f"sign accuracy, {k} largest-|return| days"
        if m == f"sign_acc_rest{k}":
            return f"sign accuracy, other {n - k} days"
        if m == f"hit_top{k}":
            return f"hit rate, {k} largest-|return| days"
        if m == f"QLIKE_top{k}":
            return f"QLIKE, {k} largest-|return| days"
        if m == f"QLIKE_rest{k}":
            return f"QLIKE, other {n - k} days"
    return m


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(stats.spearmanr(a, b).statistic)


def ols_hac(y: np.ndarray, X: np.ndarray) -> dict:
    import statsmodels.api as sm

    f = sm.OLS(y, sm.add_constant(X)).fit(
        cov_type="HAC", cov_kwds={"maxlags": nw_lag(len(y))}
    )
    out = dict(
        const=float(f.params[0]),
        slope=float(f.params[1]),
        t_hac=float(f.tvalues[1]),
        p_hac=float(f.pvalues[1]),
        r2=float(f.rsquared),
        n=int(len(y)),
    )
    if np.ndim(X) == 2 and X.shape[1] == 2:
        out.update(
            slope2=float(f.params[2]),
            t2_hac=float(f.tvalues[2]),
            p2_hac=float(f.pvalues[2]),
        )
    return out


# ---------------------------------------------------------------------------- main
def main() -> None:  # noqa: C901 - one linear report
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    lines: list[str] = []
    NUM: dict = {}

    def say(msg: str) -> None:
        print(msg, flush=True)
        lines.append(msg)

    if MASTER.is_file() and MASTER_DAILY.is_file():
        info, S, gates = panel_from_master()
        say(
            f"universe: master table {MASTER.relative_to(REPO).as_posix()} ({S['n_rows_master']} rows), table A: "
            f"{len(info)} forecasts on the same {len(S['days'])} trade days (check rows, duplicates, the rule left out)"
        )
    else:
        info, S, gates = panel_from_tables()
        say(f"universe: no master table; {len(info)} forecast tables on disk")
    P, days = S["P"], S["days"]
    names = list(info.index)
    n_days, M = P["F"].shape
    one = np.ones(n_days)
    absR = np.abs(P["R"][:, 0])
    order = np.argsort(-absR, kind="stable")
    tails = {}
    for k in TAIL_K:
        tm = np.zeros(n_days)
        tm[order[:k]] = 1.0
        tails[k] = tm
    Mt = metrics(P, one, tails)
    if S["source"] == "master":
        gates.append(
            dict(
                check="Sharpe mid = master table",
                max_abs=float(
                    np.abs(
                        Mt["Sharpe_mid"] - info["master_Sharpe_mid"].to_numpy()
                    ).max()
                ),
            )
        )
        gates.append(
            dict(
                check="Sharpe crossed = master table",
                max_abs=float(
                    np.abs(
                        Mt["Sharpe_crossed"] - info["master_Sharpe_crossed"].to_numpy()
                    ).max()
                ),
            )
        )
        gates.append(
            dict(
                check="QLIKE = master qlike_recal",
                max_abs=float(
                    np.abs(Mt["QLIKE"] - info["master_qlike_recal"].to_numpy()).max()
                ),
            )
        )
    GT = pd.DataFrame(gates)
    GT.to_csv(OUT / "gates.csv", index=False)
    for g in gates:
        say(f"GATE {g['check']}: max abs {g['max_abs']:.1e}")
    if S["source"] == "master":
        assert GT["max_abs"].max() < 1e-9, GT

    cls = np.array([classify(f, k) for f, k in zip(info["family"], names)])
    T = pd.DataFrame({k: v for k, v in Mt.items()}, index=names)
    T.insert(0, "label", info["label"])
    T.insert(1, "family", info["family"])
    T.insert(2, "display_class", cls)
    T["pct_buy"] = 100 * (P["q"] > 0).mean(0)
    tot = P["pnl"].sum(0)
    for k in TAIL_K:
        T[f"pnl_share_top{k}"] = (P["pnl"] * tails[k][:, None]).sum(0) / tot
    T["pnl_share_sign_right"] = (P["pnl"] * P["sgn"]).sum(0) / tot
    T = T.sort_values("Sharpe_mid", ascending=False)
    T.index.name = "forecast"
    T.to_csv(OUT / "forecast_metrics.csv")
    say(
        f"tail days: the {TAIL_K[0]} / {TAIL_K[1]} trade days with the largest |straddle return| "
        f"(|R| >= {absR[order[TAIL_K[0] - 1]]:.2f} / {absR[order[TAIL_K[1] - 1]]:.2f}); median |R| {np.median(absR):.2f}"
    )
    say(
        T[
            [
                "family",
                "QLIKE",
                "sign_acc",
                "hit_rate",
                "Sharpe_mid",
                f"sign_acc_top{TAIL_K[0]}",
                f"sign_acc_top{TAIL_K[1]}",
                "mz_log_slope",
                "pct_buy",
            ]
        ]
        .round(3)
        .head(15)
        .to_string()
    )

    # ---- B1 / B2: rank agreement with the Sharpe, bootstrap over days
    rng = np.random.default_rng(SEED)
    idx = asl.circular_block_bootstrap_idx(rng, n_days, BLOCK, B)
    mets = list(DIRECTION)
    shs = ("Sharpe_mid", "Sharpe_crossed")
    draws = {(m, s): np.empty(B) for m in mets for s in shs}
    CLASS_METS = (
        "QLIKE",
        "MSE",
        "sign_acc",
        f"QLIKE_top{TAIL_K[0]}",
        f"sign_acc_rest{TAIL_K[0]}",
        f"sign_acc_top{TAIL_K[0]}",
        f"sign_acc_top{TAIL_K[1]}",
    )
    big = [c for c in CLASS_ORDER if (cls == c).sum() >= MIN_SET]
    cdraws = {(c, m): np.empty(B) for c in big for m in CLASS_METS}
    # B4: within each model family (and the pooled sets), the same draws
    fam_arr = info["family"].astype(str).to_numpy()
    sets = model_family_sets(fam_arr, cls)
    FAM_METS = ("QLIKE", f"sign_acc_top{TAIL_K[0]}", f"sign_acc_top{TAIL_K[1]}")
    sdraws = {(i, m): np.empty(B) for i in range(len(sets)) for m in FAM_METS}
    with warnings.catch_warnings():
        # a small family can tie on a tail-day sign accuracy in a draw (-> nan, dropped)
        warnings.simplefilter("ignore", stats.ConstantInputWarning)
        for b in range(B):
            w = np.bincount(idx[b], minlength=n_days).astype(float)
            Mb = metrics(P, w, tails)
            for m in mets:
                for s in shs:
                    draws[(m, s)][b] = spearman(Mb[m], Mb[s])
            for c in big:
                sel = cls == c
                for m in CLASS_METS:
                    cdraws[(c, m)][b] = DIRECTION[m] * spearman(
                        Mb[m][sel], Mb["Sharpe_mid"][sel]
                    )
            for i, st in enumerate(sets):
                for m in FAM_METS:
                    sdraws[(i, m)][b] = DIRECTION[m] * spearman(
                        Mb[m][st["sel"]], Mb["Sharpe_mid"][st["sel"]]
                    )
    rows = []
    for m in mets:
        for s in shs:
            pt = spearman(Mt[m], Mt[s])
            lo, hi = np.nanpercentile(draws[(m, s)], [2.5, 97.5])
            dm = DIRECTION[m]
            rows.append(
                dict(
                    metric=m,
                    description=descr(m, n_days),
                    sharpe=s,
                    better="lower" if dm < 0 else "higher",
                    spearman=pt,
                    ci_lo=lo,
                    ci_hi=hi,
                    agreement=dm * pt,
                    agreement_lo=min(dm * lo, dm * hi),
                    agreement_hi=max(dm * lo, dm * hi),
                    n_forecasts=M,
                )
            )
    RK = pd.DataFrame(rows)
    RK.to_csv(OUT / "rank_vs_sharpe.csv", index=False)
    say(
        f"rank agreement with the Sharpe (Spearman x direction; +1 = the metric ranks the {M} forecasts exactly as "
        f"the Sharpe does), 95 % interval: circular block bootstrap over days, block {BLOCK}, B {B}:"
    )
    say(
        RK[RK.sharpe == "Sharpe_mid"][
            ["metric", "spearman", "agreement", "agreement_lo", "agreement_hi"]
        ]
        .round(3)
        .to_string(index=False)
    )

    # the same inside the display classes (point estimates; small groups)
    fam_rows = []
    for c in big:
        sel = cls == c
        for m in CLASS_METS:
            lo, hi = np.nanpercentile(cdraws[(c, m)], [2.5, 97.5])
            fam_rows.append(
                dict(
                    display_class=c,
                    n=int(sel.sum()),
                    metric=m,
                    description=descr(m, n_days),
                    agreement_with_sharpe_mid=DIRECTION[m]
                    * spearman(Mt[m][sel], Mt["Sharpe_mid"][sel]),
                    lo=lo,
                    hi=hi,
                )
            )
    FR = pd.DataFrame(fam_rows)
    FR.to_csv(OUT / "rank_vs_sharpe_by_family.csv", index=False)
    say(FR.round(3).to_string(index=False))

    # ---- B4: within each model family, the mean over families, paired differences
    def pt_agr(m: str, sel: np.ndarray) -> float:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", stats.ConstantInputWarning)
            return DIRECTION[m] * spearman(Mt[m][sel], Mt["Sharpe_mid"][sel])

    def pct(d: np.ndarray) -> tuple[float, float]:
        lo_, hi_ = np.nanpercentile(d, [2.5, 97.5])
        return float(lo_), float(hi_)

    mf_rows, pts = [], {}
    for i, st in enumerate(sets):
        sel = st["sel"]
        for m in FAM_METS:
            pts[(i, m)] = pt_agr(m, sel)
            lo, hi = pct(sdraws[(i, m)])
            mf_rows.append(
                dict(
                    set=st["set"],
                    kind=st["kind"],
                    rung=st["rung"],
                    n=int(sel.sum()),
                    n_families=int(len(set(fam_arr[sel]))),
                    metric=m,
                    description=descr(m, n_days),
                    agreement_with_sharpe_mid=pts[(i, m)],
                    lo=lo,
                    hi=hi,
                    n_draws=int(np.isfinite(sdraws[(i, m)]).sum()),
                    qlike_min=float(Mt["QLIKE"][sel].min()),
                    qlike_max=float(Mt["QLIKE"][sel].max()),
                    sharpe_min=float(Mt["Sharpe_mid"][sel].min()),
                    sharpe_max=float(Mt["Sharpe_mid"][sel].max()),
                )
            )
    fam_idx = [i for i, st in enumerate(sets) if st["kind"] == "family"]
    tree_idx = [i for i in fam_idx if sets[i]["rung"]]
    mean_sets = [
        (f"mean over the {len(fam_idx)} model families", fam_idx),
        (f"mean over the {len(tree_idx)} tree rungs", tree_idx),
    ]
    mdraws, mpts = {}, {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # a draw with every family nan
        for name, ii in mean_sets:
            sel = np.any([sets[i]["sel"] for i in ii], axis=0)
            for m in FAM_METS:
                mdraws[(name, m)] = np.nanmean(
                    np.column_stack([sdraws[(i, m)] for i in ii]), axis=1
                )
                mpts[(name, m)] = float(np.nanmean([pts[(i, m)] for i in ii]))
                lo, hi = pct(mdraws[(name, m)])
                mf_rows.append(
                    dict(
                        set=name,
                        kind="mean",
                        rung="",
                        n=int(sel.sum()),
                        n_families=len(ii),
                        metric=m,
                        description=descr(m, n_days),
                        agreement_with_sharpe_mid=mpts[(name, m)],
                        lo=lo,
                        hi=hi,
                        n_draws=int(np.isfinite(mdraws[(name, m)]).sum()),
                        qlike_min=float(Mt["QLIKE"][sel].min()),
                        qlike_max=float(Mt["QLIKE"][sel].max()),
                        sharpe_min=float(Mt["Sharpe_mid"][sel].min()),
                        sharpe_max=float(Mt["Sharpe_mid"][sel].max()),
                    )
                )
    MF = pd.DataFrame(mf_rows)
    assert not MF.duplicated(["set", "metric"]).any(), "set names must be unique"
    MF.to_csv(OUT / "rank_vs_sharpe_by_model_family.csv", index=False)
    say(
        f"B4 within each model family (agreement with the Sharpe, mid; {len(fam_idx)} families "
        f"with >= {MIN_SET} forecasts, pooled sets, means over families):"
    )
    say(
        MF.pivot_table(
            index=["set", "n"],
            columns="metric",
            values="agreement_with_sharpe_mid",
            sort=False,
        )
        .round(3)
        .to_string()
    )

    # paired differences of two agreements (the same draws: a positive difference = the
    # first metric orders the forecasts more like the Sharpe than the second)
    k1_ = TAIL_K[0]
    diff_rows = []
    for a_, b_ in (
        (f"sign_acc_top{k1_}", "sign_acc"),
        (f"sign_acc_top{k1_}", f"sign_acc_rest{k1_}"),
        (f"sign_acc_top{k1_}", "QLIKE"),
        ("QLIKE", "MSE_log"),
    ):
        d = (
            DIRECTION[a_] * draws[(a_, "Sharpe_mid")]
            - DIRECTION[b_] * draws[(b_, "Sharpe_mid")]
        )
        pt = DIRECTION[a_] * spearman(Mt[a_], Mt["Sharpe_mid"]) - DIRECTION[
            b_
        ] * spearman(Mt[b_], Mt["Sharpe_mid"])
        diff_rows.append(
            dict(set="all forecasts", n=M, metric_a=a_, metric_b=b_, diff=pt)
            | dict(zip(("lo", "hi"), pct(d)), n_draws=int(np.isfinite(d).sum()))
        )
    a_, b_ = f"sign_acc_top{k1_}", "QLIKE"
    for i, st in enumerate(sets):
        if st["kind"] in ("class", "pooled"):
            d = sdraws[(i, a_)] - sdraws[(i, b_)]
            diff_rows.append(
                dict(
                    set=st["set"],
                    n=int(st["sel"].sum()),
                    metric_a=a_,
                    metric_b=b_,
                    diff=pts[(i, a_)] - pts[(i, b_)],
                )
                | dict(zip(("lo", "hi"), pct(d)), n_draws=int(np.isfinite(d).sum()))
            )
    for name, ii in mean_sets:
        d = mdraws[(name, a_)] - mdraws[(name, b_)]
        diff_rows.append(
            dict(
                set=name,
                n=len(ii),
                metric_a=a_,
                metric_b=b_,
                diff=mpts[(name, a_)] - mpts[(name, b_)],
            )
            | dict(zip(("lo", "hi"), pct(d)), n_draws=int(np.isfinite(d).sum()))
        )
    RD = pd.DataFrame(diff_rows)
    RD.to_csv(OUT / "rank_vs_sharpe_differences.csv", index=False)
    say("paired differences of agreement with the Sharpe (mid):")
    say(RD.round(3).to_string(index=False))

    # ---- B3: per-day regressions, HAC
    reg = []
    for j, k in enumerate(names):
        y = P["pnl"][:, j]
        for xname, X in (
            ("log_error", P["lerr"][:, j]),
            ("abs_log_error", np.abs(P["lerr"][:, j])),
            ("sign_right", P["sgn"][:, j]),
            (
                "log_error + sign_right",
                np.column_stack([P["lerr"][:, j], P["sgn"][:, j]]),
            ),
        ):
            reg.append(
                dict(
                    forecast=k,
                    family=info.loc[k, "family"],
                    regressor=xname,
                    **ols_hac(y, X),
                )
            )
    RG = pd.DataFrame(reg)
    RG.to_csv(OUT / "per_day_regressions.csv", index=False)
    summ = RG.groupby("regressor", sort=False).agg(
        r2_median=("r2", "median"),
        r2_min=("r2", "min"),
        r2_max=("r2", "max"),
        slope_median=("slope", "median"),
        t_median=("t_hac", "median"),
        n_p_below_05=("p_hac", lambda p: int((p < 0.05).sum())),
        n=("p_hac", "size"),
    )
    say(
        f"per-day regressions of the trade return (mid) on the forecast, {M} forecasts, HAC lag {nw_lag(n_days)}:"
    )
    say(summ.round(4).to_string())
    for kk in KEY:
        if kk in names:
            say(
                f"  {kk}: "
                + "; ".join(
                    f"{r.regressor} slope {r.slope:+.4f} (HAC t {r.t_hac:+.2f}) R2 {r.r2:.3f}"
                    for r in RG[RG.forecast == kk].itertuples()
                )
            )
    # P&L on the days the sign was right vs wrong (headline forecast)
    hk = "sub_ridge_live_feasible" if "sub_ridge_live_feasible" in names else names[0]
    j = names.index(hk)
    ok = P["sgn"][:, j] > 0
    NUM["headline_split"] = dict(
        forecast=hk,
        n_right=int(ok.sum()),
        n_wrong=int((~ok).sum()),
        mean_pnl_right=float(P["pnl"][ok, j].mean()),
        mean_pnl_wrong=float(P["pnl"][~ok, j].mean()),
        sum_pnl=float(P["pnl"][:, j].sum()),
        sum_pnl_top20=float((P["pnl"][:, j] * tails[20]).sum()),
        sum_pnl_top50=float((P["pnl"][:, j] * tails[50]).sum()),
    )

    # ---- numbers for the summary
    NUM.update(
        source=S["source"],
        n_forecasts=M,
        n_days=n_days,
        first_day=str(pd.Timestamp(days[0]).date()),
        last_day=str(pd.Timestamp(days[-1]).date()),
        classes={c: int((cls == c).sum()) for c in CLASS_ORDER},
        gates=gates,
        rank=RK.to_dict("records"),
        by_class=FR.to_dict("records"),
        by_model_family=MF.to_dict("records"),
        differences=RD.to_dict("records"),
        families={f: int((fam_arr == f).sum()) for f in dict.fromkeys(fam_arr)},
        regressions_summary=summ.reset_index().to_dict("records"),
        key=T.loc[[k for k in KEY if k in T.index]].reset_index().to_dict("records"),
        range={
            c: [float(T[c].min()), float(T[c].median()), float(T[c].max())]
            for c in (
                "QLIKE",
                "Sharpe_mid",
                "Sharpe_crossed",
                "sign_acc",
                "hit_rate",
                f"sign_acc_top{TAIL_K[0]}",
                f"sign_acc_top{TAIL_K[1]}",
                "mz_log_slope",
                f"pnl_share_top{TAIL_K[0]}",
                f"pnl_share_top{TAIL_K[1]}",
                "pnl_share_sign_right",
                "pct_buy",
            )
        },
        tail_R_threshold={k: float(absR[order[k - 1]]) for k in TAIL_K},
        median_absR=float(np.median(absR)),
        best_qlike={
            "forecast": str(T["QLIKE"].idxmin()),
            **T.loc[
                T["QLIKE"].idxmin(),
                ["QLIKE", "Sharpe_mid", "Sharpe_crossed", "sign_acc"],
            ]
            .astype(float)
            .to_dict(),
        },
        best_sharpe={
            "forecast": str(T["Sharpe_mid"].idxmax()),
            **T.loc[
                T["Sharpe_mid"].idxmax(),
                ["QLIKE", "Sharpe_mid", "Sharpe_crossed", "sign_acc"],
            ]
            .astype(float)
            .to_dict(),
        },
        qlike_rank_of_best_sharpe=int(T["QLIKE"].rank().loc[T["Sharpe_mid"].idxmax()]),
        sharpe_rank_of_best_qlike=int(
            T["Sharpe_mid"].rank(ascending=False).loc[T["QLIKE"].idxmin()]
        ),
        tail_days=[str(pd.Timestamp(days[i]).date()) for i in order[: TAIL_K[0]]],
        boot=dict(B=B, block=BLOCK, seed=SEED),
    )

    # ---- figures
    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    for c in CLASS_ORDER:
        sel = T["display_class"] == c
        if not sel.any():
            continue
        x = T.loc[sel, "QLIKE"].clip(upper=QLIKE_XMAX)
        ax.scatter(
            x,
            T.loc[sel, "Sharpe_mid"],
            s=40,
            marker=CLASS_MARK[c],
            color=CLASS_COLOR[c],
            edgecolor="white",
            linewidth=0.8,
            label=f"{c} ({int(sel.sum())})",
            zorder=3,
        )
    beyond = T[T["QLIKE"] > QLIKE_XMAX]
    if len(beyond):
        ax.annotate(
            f"{len(beyond)} forecasts at QLIKE {beyond['QLIKE'].min():.2f}-{beyond['QLIKE'].max():.2f}\n"
            "drawn at the edge",
            (QLIKE_XMAX, beyond["Sharpe_mid"].mean()),
            xytext=(-110, -4),
            textcoords="offset points",
            fontsize=7,
            color="0.3",
        )
    for kk, lab in KEY.items():
        if kk in T.index:
            ax.annotate(
                lab,
                (min(T.loc[kk, "QLIKE"], QLIKE_XMAX), T.loc[kk, "Sharpe_mid"]),
                xytext=(5, 3),
                textcoords="offset points",
                fontsize=6.5,
                color="0.2",
            )
    r = RK[(RK.metric == "QLIKE") & (RK.sharpe == "Sharpe_mid")].iloc[0]
    ax.set_xlabel(
        f"16:00-bar QLIKE, research scorer, the {n_days} trade days (lower = better forecast)",
        fontsize=8,
    )
    ax.set_ylabel("sign(s) Sharpe, mid fill (annualized)", fontsize=8)
    ax.set_title(
        f"{M} forecasts: Spearman(QLIKE, Sharpe) {r.spearman:+.2f}, 95 % interval "
        f"[{r.ci_lo:+.2f}, {r.ci_hi:+.2f}] (bootstrap over days)",
        fontsize=8.5,
    )
    ax.tick_params(labelsize=7)
    ax.grid(color="0.9", lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=6.5, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "qlike_vs_sharpe.png", dpi=160)
    plt.close(fig)

    show = [
        "QLIKE",
        "MSE",
        "MSE_log",
        "calib_err",
        "sign_acc",
        "hit_rate",
        *[f"QLIKE_top{k}" for k in TAIL_K],
        *[f"sign_acc_rest{k}" for k in TAIL_K],
        *[f"sign_acc_top{k}" for k in TAIL_K],
    ]
    sub = (
        RK[(RK.sharpe == "Sharpe_mid") & RK.metric.isin(show)]
        .set_index("metric")
        .loc[show]
    )
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    yy = np.arange(len(sub))
    ax.errorbar(
        sub["agreement"],
        yy,
        xerr=[
            sub["agreement"] - sub["agreement_lo"],
            sub["agreement_hi"] - sub["agreement"],
        ],
        fmt="o",
        ms=5,
        color="#2a78d6",
        ecolor="#2a78d6",
        lw=1.4,
        capsize=0,
    )
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_yticks(yy)
    ax.set_yticklabels([descr(m, n_days) for m in show], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(-1, 1)
    ax.set_xlabel(
        f"rank agreement with the sign(s) Sharpe across {M} forecasts (Spearman, signed: +1 = same order)",
        fontsize=7.5,
    )
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", color="0.9", lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "rank_vs_sharpe.png", dpi=160)
    plt.close(fig)

    # B4 forest plot: within each model family, QLIKE and the tail-day sign accuracy
    ymets = ("QLIKE", f"sign_acc_top{TAIL_K[0]}")
    rows_ = list(dict.fromkeys(zip(MF["set"], MF["kind"], MF["rung"], MF["n"])))
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 0.2 * len(rows_) + 1.1), sharey=True)
    yy = np.arange(len(rows_))
    for ax, m in zip(axes, ymets):
        for y, (s_, kind, rg, n_) in zip(yy, rows_):
            r = MF[(MF["set"] == s_) & (MF["kind"] == kind) & (MF["metric"] == m)].iloc[
                0
            ]
            if kind == "family":
                c = CLASS_COLOR[classify(s_, "")]
                mk = CLASS_MARK[classify(s_, "")]
            else:
                c, mk = "0.25", "D"
            ax.errorbar(
                r["agreement_with_sharpe_mid"],
                y,
                xerr=[
                    [r["agreement_with_sharpe_mid"] - r["lo"]],
                    [r["hi"] - r["agreement_with_sharpe_mid"]],
                ],
                fmt=mk,
                ms=4.5 if kind == "family" else 5,
                color=c,
                ecolor=c,
                lw=1.2,
                capsize=0,
            )
        ax.axvline(0, color="0.5", lw=0.8)
        ax.set_xlim(-1, 1)
        ax.set_title(descr(m, n_days), fontsize=7.5)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="x", color="0.9", lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_yticks(yy)
    axes[0].set_yticklabels(
        [
            (f"{rg}  ({n_})" if rg and kind == "family" else f"{s_}  ({n_})")
            if kind != "mean"
            else s_
            for s_, kind, rg, n_ in rows_
        ],
        fontsize=6.3,
    )
    axes[0].invert_yaxis()
    fig.supxlabel(
        "rank agreement with the sign(s) Sharpe within the set (Spearman, signed: +1 = same order); "
        f"95 % day-block interval, B {B}",
        fontsize=7,
    )
    fig.tight_layout()
    fig.savefig(OUT / "rank_vs_sharpe_by_model_family.png", dpi=160)
    plt.close(fig)

    (OUT / "numbers.json").write_text(
        json.dumps(NUM, indent=1, default=float), encoding="utf-8"
    )
    (OUT / "run_output.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------- summary
def _iv(lo: float, hi: float, nd: int = 2) -> str:
    return f"[{lo:+.{nd}f}, {hi:+.{nd}f}]"


def _sig(lo: float, hi: float) -> bool:
    return bool(lo > 0 or hi < 0)


def esc(text: str) -> str:
    """A table cell: a literal | would end the cell."""
    return str(text).replace("|", r"\|")


def _reg_summary(RG: pd.DataFrame) -> pd.DataFrame:
    return RG.groupby("regressor").agg(
        r2_med=("r2", "median"),
        r2_min=("r2", "min"),
        r2_max=("r2", "max"),
        t_min=("t_hac", "min"),
        t_max=("t_hac", "max"),
        t_med=("t_hac", "median"),
        n_sig=("p_hac", lambda p: int((p < 0.05).sum())),
        n=("p_hac", "size"),
        slope_med=("slope", "median"),
    )


def _git_text(rel: str) -> str | None:
    """A file as committed at PREV_COMMIT (None when git or the file is unavailable)."""
    try:
        r = subprocess.run(
            ["git", "show", f"{PREV_COMMIT}:{rel}"],
            cwd=REPO,
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return r.stdout.decode("utf-8")


def load_outputs(read: Callable[[str], str | None]) -> dict | None:
    """One run's outputs as the summary reads them; None if a core file is missing.
    The B4 files (model families, paired differences) are optional: older runs lack them."""
    core = (
        "rank_vs_sharpe.csv",
        "rank_vs_sharpe_by_family.csv",
        "per_day_regressions.csv",
        "forecast_metrics.csv",
        "numbers.json",
    )
    got = {n: read(n) for n in core}
    txt = {n: v for n, v in got.items() if v is not None}
    if len(txt) < len(core):
        return None

    def csv(name: str, **kw) -> pd.DataFrame:
        return pd.read_csv(io.StringIO(txt[name]), **kw)

    RG = csv("per_day_regressions.csv")
    out = dict(
        RK=csv("rank_vs_sharpe.csv"),
        FR=csv("rank_vs_sharpe_by_family.csv"),
        RG=RG,
        reg=_reg_summary(RG),
        T=csv("forecast_metrics.csv", index_col=0),
        N=json.loads(txt["numbers.json"]),
    )
    for key, name in (
        ("RD", "rank_vs_sharpe_differences.csv"),
        ("MF", "rank_vs_sharpe_by_model_family.csv"),
    ):
        t = read(name)
        out[key] = None if t is None else pd.read_csv(io.StringIO(t))
    return out


def check_claims(o: dict) -> dict[str, dict]:
    """Every qualitative claim of the prose, evaluated on one run's outputs.
    holds = the check's outcome (None: not computable from that run's outputs);
    statement = the clause the prose prints for that outcome; asserted = the summary
    refuses to be written when it fails; asserted_prev = asserted by the script of
    PREV_COMMIT (so a claim with asserted_prev and not holds is an assertion that fired)."""
    RK, FR, reg, T, N, RD, MF = (
        o[k] for k in ("RK", "FR", "reg", "T", "N", "RD", "MF")
    )
    M, nd = N["n_forecasts"], N["n_days"]
    k1, k2 = TAIL_K
    PL = "per-bar linear"
    top1, rest1 = f"sign_acc_top{k1}", f"sign_acc_rest{k1}"

    def rk(m: str) -> pd.Series:
        return RK[(RK.metric == m) & (RK.sharpe == "Sharpe_mid")].iloc[0]

    def agr(m: str) -> str:
        r = rk(m)
        return f"{r['agreement']:+.2f} {_iv(r['agreement_lo'], r['agreement_hi'])}"

    def sig(m: str) -> bool:
        return _sig(rk(m)["agreement_lo"], rk(m)["agreement_hi"])

    def fr(c: str, m: str) -> pd.Series:
        return FR[(FR.display_class == c) & (FR.metric == m)].iloc[0]

    def fagr(c: str, m: str) -> str:
        r = fr(c, m)
        return f"{r['agreement_with_sharpe_mid']:+.2f} {_iv(r['lo'], r['hi'])}"

    def rd(s: str, a: str, b: str) -> pd.Series | None:
        if RD is None:
            return None
        x = RD[(RD["set"] == s) & (RD.metric_a == a) & (RD.metric_b == b)]
        return None if x.empty else x.iloc[0]

    def dtxt(d: pd.Series | None) -> str:
        return (
            "not computed" if d is None else f"{d['diff']:+.2f} {_iv(d['lo'], d['hi'])}"
        )

    C: dict[str, dict] = {}

    def claim(cid, text, rule, asserted, asserted_prev, holds, value, statement):
        C[cid] = dict(
            id=cid,
            claim=text,
            rule=rule,
            asserted=asserted,
            asserted_prev=asserted_prev,
            holds=None if holds is None else bool(holds),
            value=value,
            statement=statement,
        )

    q = rk("QLIKE")
    h = q["agreement_lo"] > 0
    claim(
        "qlike_all",
        "across all forecasts a lower QLIKE goes with a higher Sharpe",
        "QLIKE agreement interval above 0",
        True,
        True,
        h,
        f"Spearman {q['spearman']:+.2f} {_iv(q['ci_lo'], q['ci_hi'])}",
        f"Across all {M} forecasts a lower QLIKE goes with a higher Sharpe (Spearman {q['spearman']:+.2f}, "
        f"interval {_iv(q['ci_lo'], q['ci_hi'])})"
        if h
        else f"Across all {M} forecasts QLIKE does not order the Sharpe (Spearman {q['spearman']:+.2f}, "
        f"interval {_iv(q['ci_lo'], q['ci_hi'])})",
    )
    r = fr(PL, "QLIKE")
    h = not _sig(r["lo"], r["hi"])
    claim(
        "qlike_linear",
        "but not among forecasts of similar accuracy (the per-bar linear class)",
        "per-bar linear class: QLIKE agreement interval covers 0",
        True,
        True,
        h,
        f"{int(r['n'])} forecasts: {fagr(PL, 'QLIKE')}",
        "but not among forecasts of similar accuracy"
        if h
        else "and among forecasts of similar accuracy too",
    )
    h = rk("MSE")["agreement"] < 0 and not sig("MSE")
    claim(
        "mse",
        "MSE in variance units orders the forecasts the wrong way round",
        "MSE agreement < 0, interval covers 0",
        True,
        True,
        h,
        agr("MSE"),
        f"MSE in variance units orders the forecasts the wrong way round (agreement {agr('MSE')})",
    )
    h = max(rk("MSE_log")["agreement"], rk("calib_err")["agreement"]) <= q["agreement"]
    claim(
        "other_losses",
        "the other accuracy losses do no better than QLIKE",
        "agreement of squared log error and calibration error <= QLIKE's (point)",
        False,
        False,
        h,
        f"QLIKE {agr('QLIKE')}; squared log error {agr('MSE_log')}; calibration {agr('calib_err')}; "
        f"QLIKE minus squared log error, paired {dtxt(rd('all forecasts', 'QLIKE', 'MSE_log'))}",
        "The other accuracy losses do no better."
        if h
        else "Another accuracy loss ranks the forecasts at least as well as QLIKE.",
    )
    d_all = rd("all forecasts", top1, "sign_acc")
    h = sig("sign_acc")
    claim(
        "all_day_sign",
        "the all-day sign accuracy ranks the forecasts",
        "all-day sign-accuracy agreement interval excludes 0",
        False,
        False,
        h,
        agr("sign_acc"),
        f"does not rank them (agreement {agr('sign_acc')})"
        if not h
        else (
            f"ranks them only weakly (agreement {agr('sign_acc')})"
            if d_all is None or d_all["lo"] > 0
            else f"ranks them (agreement {agr('sign_acc')})"
        ),
    )
    h = rk(top1)["agreement_lo"] > 0
    claim(
        "tail_sign",
        f"the {k1}-day tail sign accuracy ranks the forecasts",
        f"{k1}-day sign-accuracy agreement interval above 0",
        True,
        True,
        h,
        agr(top1),
        "it does" if h else "it does not",
    )
    h = rk(top1)["agreement"] > rk("sign_acc")["agreement"]
    claim(
        "tail_stronger",
        f"the {k1}-day sign accuracy ranks them more strongly than the all-day one",
        f"{k1}-day agreement > all-day agreement (point)",
        True,
        True,
        h,
        f"{rk(top1)['agreement']:+.2f} vs {rk('sign_acc')['agreement']:+.2f}",
        ("and more strongly" if h else "but less strongly")
        if d_all is None or d_all["lo"] > 0 or not h
        else "and more strongly in the point estimate",
    )
    claim(
        "tail_stronger_paired",
        f"... and the paired difference ({k1}-day minus all-day agreement) excludes 0",
        "paired-difference interval above 0 (same draws)",
        False,
        False,
        None if d_all is None else d_all["lo"] > 0,
        dtxt(d_all),
        ""
        if d_all is None
        else f"; the paired difference from the all-day agreement is {dtxt(d_all)}"
        + ("" if d_all["lo"] > 0 else ", an interval that covers zero"),
    )
    d_rest = rd("all forecasts", top1, rest1)
    h = rk(top1)["agreement"] > rk(rest1)["agreement"]
    claim(
        "tail_over_rest",
        f"the {k1}-day sign accuracy ranks them more strongly than the other days' one",
        f"{k1}-day agreement > other-days agreement (point)",
        True,
        False,
        h,
        f"{rk(top1)['agreement']:+.2f} vs {rk(rest1)['agreement']:+.2f}; paired {dtxt(d_rest)}",
        "",
    )
    h = not sig(rest1)
    claim(
        "rest_sign",
        f"on the other {nd - k1} days the sign accuracy does not rank the forecasts",
        "other-days sign-accuracy agreement interval covers 0",
        False,
        True,
        h,
        agr(rest1),
        f"on the other {nd - k1} days it does not ({agr(rest1)})"
        if h
        else f"on the other {nd - k1} days it does as well, less strongly in the point estimate ({agr(rest1)}; "
        f"paired difference of the {k1}-day agreement over it {dtxt(d_rest)})",
    )
    paired_ok = d_all is None or d_all["lo"] > 0
    h = C["rest_sign"]["holds"] and paired_ok
    claim(
        "tail_story",
        "what tracks the trade is calling the side right on the few largest-move days (and not on the others)",
        "rest_sign holds and tail_stronger_paired holds (or was not computed)",
        False,
        False,
        h,
        f"other days {agr(rest1)}; {k1}-day minus all-day {dtxt(d_all)}",
        "What does track the trade is calling the side right on the few days the straddle moves most."
        if h
        else (
            "What tracks the trade is calling the side right: the few days the straddle moves most rank the "
            "forecasts, the other days do not."
            if C["rest_sign"]["holds"]
            else f"Calling the side right tracks the trade: on the few days the straddle moves most and, across "
            f"these {M} forecasts, on the other days as well."
        ),
    )
    r = fr(PL, top1)
    pt, lo, hi = r["agreement_with_sharpe_mid"], r["lo"], r["hi"]
    h = pt > 0 and not _sig(lo, hi)
    claim(
        "linear_tail",
        f"within the per-bar linear class the {k1}-day sign accuracy is suggestive, its interval reaching zero",
        f"per-bar linear class: {k1}-day agreement > 0, interval covers 0",
        False,
        False,
        h,
        f"{int(r['n'])} forecasts: {fagr(PL, top1)}",
        "suggestive, with an interval that reaches zero"
        if h
        else ("and its interval excludes zero" if lo > 0 else "no sign of it either"),
    )
    rs, rl = reg.loc["sign_right"], reg.loc["log_error"]
    h = rs["n_sig"] == M and rl["n_sig"] <= CHANCE_SHARE * M
    claim(
        "sign_right_all",
        "per day, a right sign explains the P&L for every forecast; the log error only at the chance rate",
        f"sign indicator p < 0.05 for all {M}; log error p < 0.05 for <= {CHANCE_SHARE:.0%} of them",
        True,
        True,
        h,
        f"sign {int(rs['n_sig'])} of {M}; log error {int(rl['n_sig'])} of {M}",
        "every forecast significant",
    )
    h = rl["r2_max"] < rs["r2_min"]
    claim(
        "r2_order",
        "the log error explains less of the P&L than the sign for every forecast",
        "max R² (log error) < min R² (sign indicator)",
        True,
        True,
        h,
        f"{100 * rl['r2_max']:.2f} % < {100 * rs['r2_min']:.2f} %",
        "",
    )
    ra_ = reg.loc["abs_log_error"]
    h = ra_["slope_med"] < 0
    claim(
        "abs_cost",
        "larger misses cost a little",
        "median slope on the absolute log error < 0",
        False,
        False,
        h,
        f"median slope {ra_['slope_med']:+.2f}, {int(ra_['n_sig'])} of {M} with p < 0.05",
        "larger misses cost a little" if h else "larger misses cost nothing on average",
    )
    rj = reg.loc["log_error + sign_right"]
    h = rj["n_sig"] <= CHANCE_SHARE * M
    claim(
        "joint_nothing",
        "adding the log error to the sign indicator adds nothing",
        f"log error p < 0.05 in the joint regression for <= {CHANCE_SHARE:.0%} of the forecasts",
        False,
        False,
        h,
        f"{int(rj['n_sig'])} of {M}; median R² {100 * rj['r2_med']:.2f} % vs {100 * rs['r2_med']:.2f} %",
        "adds nothing"
        if h
        else f"adds something for {int(rj['n_sig'])} of {M} forecasts",
    )
    r = fr(TREES, "QLIKE")
    pt = r["agreement_with_sharpe_mid"]
    h = pt > 0 and _sig(r["lo"], r["hi"])
    claim(
        "trees_qlike",
        "among the per-bar trees the QLIKE ordering holds",
        "per-bar tree class: QLIKE agreement interval above 0",
        False,
        False,
        h,
        f"{int(r['n'])} forecasts: {fagr(TREES, 'QLIKE')}",
        "holds as well"
        if h
        else ("points the same way" if pt > 0 else "does not hold"),
    )
    is48 = T["display_class"] == "48-bar (paper + pooled twins)"
    mq = T.groupby(is48)["QLIKE"].median()
    ms = T.groupby(is48)["Sharpe_mid"].median()
    h = mq[True] > mq[False] and ms[True] < ms[False]
    claim(
        "bar48_worse",
        "the 48-bar forecasts forecast worse and trade worse than the per-bar ones",
        "48-bar median QLIKE > per-bar median and 48-bar median Sharpe < per-bar median",
        False,
        False,
        h,
        f"median QLIKE {mq[True]:.4f} vs {mq[False]:.4f}; median Sharpe {ms[True]:.2f} vs {ms[False]:.2f}",
        f"the 48-bar forecasts ({int(is48.sum())}) "
        + ("forecast worse" if mq[True] > mq[False] else "forecast no worse")
        + f" (median QLIKE {mq[True]:.4f} against {mq[False]:.4f}) and "
        + ("trade worse" if ms[True] < ms[False] else "trade no worse")
        + f" (median Sharpe {ms[True]:.2f} against {ms[False]:.2f}) than the per-bar ones ({int((~is48).sum())})",
    )
    hk, vx = N["headline_split"]["forecast"], "sub_ridge_vix_only"
    h = (
        vx in T.index
        and T.loc[vx, "QLIKE"] < T.loc[hk, "QLIKE"]
        and T.loc[vx, "Sharpe_mid"] < T.loc[hk, "Sharpe_mid"]
    )
    claim(
        "vx_example",
        "example: the VIX-only ridge forecasts better and trades worse than the headline",
        "QLIKE below the headline's and Sharpe below the headline's",
        False,
        False,
        h,
        f"QLIKE {T.loc[vx, 'QLIKE']:.4f} vs {T.loc[hk, 'QLIKE']:.4f}; Sharpe {T.loc[vx, 'Sharpe_mid']:.2f} vs "
        f"{T.loc[hk, 'Sharpe_mid']:.2f}"
        if vx in T.index
        else "not in the table",
        f" (e.g. the per-bar ridge on the VIX-only bucket: QLIKE {T.loc[vx, 'QLIKE']:.4f} against the headline's "
        f"{T.loc[hk, 'QLIKE']:.4f}, Sharpe {T.loc[vx, 'Sharpe_mid']:.2f} against {T.loc[hk, 'Sharpe_mid']:.2f})"
        if h
        else "",
    )
    # B4 (runs that have the model-family table)
    for m, cid, what in (
        ("QLIKE", "fam_qlike", "QLIKE"),
        (top1, "fam_tail", f"the {k1}-day sign accuracy"),
    ):
        if MF is None:
            claim(
                cid,
                f"within the model families, on average, {what} orders the trade",
                "",
                False,
                False,
                None,
                "not computed",
                "",
            )
            continue
        mr = MF[(MF.kind == "mean") & (MF.metric == m)].iloc[0]
        h = mr["lo"] > 0
        claim(
            cid,
            f"within the model families, on average, {what} orders the trade",
            "mean over the families of the within-family agreement: interval above 0",
            False,
            False,
            h,
            f"{mr['agreement_with_sharpe_mid']:+.2f} {_iv(mr['lo'], mr['hi'])}",
            f"{what} " + ("orders the trade" if h else "does not order the trade"),
        )
    return C


def write_summary() -> None:  # noqa: C901 - one linear report
    """SUMMARY.md from this folder's own outputs; the claims the verdict rests on are
    asserted against the numbers first, the others choose the prose, and the same checks
    on the 106-forecast run's committed outputs are set beside them."""

    def read_here(name: str) -> str | None:
        p = OUT / name
        return p.read_text(encoding="utf-8") if p.is_file() else None

    o = load_outputs(read_here)
    assert o is not None, "run main() first"
    RK, FR, RG, reg, T, N, RD, MF = (
        o[k] for k in ("RK", "FR", "RG", "reg", "T", "N", "RD", "MF")
    )
    M, nd = N["n_forecasts"], N["n_days"]
    k1, k2 = TAIL_K
    C = check_claims(o)
    for c in C.values():
        assert c["holds"] or not c["asserted"], c
    prev = load_outputs(lambda n: _git_text(f"{PREV_DIR}/{n}"))
    CP = check_claims(prev) if prev is not None else {}
    M_prev = prev["N"]["n_forecasts"] if prev is not None else None

    def S(cid: str) -> str:
        return C[cid]["statement"]

    def rk(m: str, s: str = "Sharpe_mid") -> pd.Series:
        return RK[(RK.metric == m) & (RK.sharpe == s)].iloc[0]

    def fr(c: str, m: str) -> pd.Series:
        return FR[(FR.display_class == c) & (FR.metric == m)].iloc[0]

    def agr(m: str, s: str = "Sharpe_mid") -> str:
        r = rk(m, s)
        return f"{r['agreement']:+.2f} {_iv(r['agreement_lo'], r['agreement_hi'])}"

    PL = "per-bar linear"
    hs = N["headline_split"]
    hk = hs["forecast"]
    h = T.loc[hk]
    bq, bs = N["best_qlike"], N["best_sharpe"]
    lab = T["label"].to_dict()

    # ---- claims.csv: both runs side by side
    rows = []
    for cid, c in C.items():
        p = CP.get(cid, {})
        rows.append(
            dict(
                id=cid,
                claim=c["claim"],
                rule=c["rule"],
                asserted_now=c["asserted"],
                asserted_in_prev_script=c["asserted_prev"],
                holds_prev=p.get("holds"),
                value_prev=p.get("value", "not computed"),
                statement_prev=p.get("statement", ""),
                holds_this=c["holds"],
                value_this=c["value"],
                statement_this=c["statement"],
            )
        )
    CL = pd.DataFrame(rows)
    CL.to_csv(OUT / "claims.csv", index=False)

    L: list[str] = []
    a = L.append
    a("# Forecast accuracy vs P&L of the 15:30 last-30-min trade (checklist C3)")
    a("")
    a(
        "Written by `experiments/perf_vs_pnl_1530.py` from its own outputs in this folder (`forecast_metrics.csv`, "
        "`rank_vs_sharpe.csv`, `rank_vs_sharpe_by_family.csv`, `rank_vs_sharpe_by_model_family.csv`, "
        "`rank_vs_sharpe_differences.csv`, `per_day_regressions.csv`, `numbers.json`); every qualitative claim is a "
        "check on those numbers (`claims.csv`): the asserted ones are verified before this file is written, the others "
        "choose the wording."
    )
    a("")
    src = (
        "the table-A rows of the closing-strategy master table (`results/close_master_table/master_table.csv`, "
        "`master_table_daily.parquet`; check rows, exact duplicates and the always-short rule left out)"
        if N["source"] == "master"
        else "every forecast table on disk (`results/spxw_pnl/yhat_*.parquet`)"
    )
    fams = N.get("families", {})
    a(
        f"**Universe:** {M} forecasts of the bar ending 16:00: {src}"
        + (f", in {len(fams)} model families" if fams else "")
        + ". By display class: "
        + ", ".join(f"{c} {n}" for c, n in N["classes"].items() if n)
        + f". Same {nd} trade days for every forecast "
        f"({N['first_day']} .. {N['last_day']}). The master table is the one rebuilt after the 16:00 campaign "
        "(de-duplicated per-bar design; every tree and LSTM fit with the per-window mask; tree rungs T10, T1, RS10, RS1 "
        "and the Optuna rungs; the per-bar and the intraday-sequence LSTM)."
    )
    a("")
    a(
        "**One scorer, the research scorer:** the 16:00 bar recalibrated alone (forecast = (ŷ² + s)·B, s = the forecast's "
        "own trailing-250-session mean squared error, lagged one session); every loss is against the per-bar spec's "
        "16:00 target. The trade is the deck's 15:30 **sign(s)** rule: buy the **straddle** (nearest out-of-the-money call "
        "+ nearest out-of-the-money put, same-day expiry, one position) when the forecast exceeds the 15:30 implied "
        "variance, sell it otherwise, hold to the close; Sharpe annualized with √252, mid and crossed fills. The per-day "
        "trade and QLIKE reproduce the master table's Sharpe and QLIKE to "
        f"{max(g['max_abs'] for g in N['gates']):.0e}."
    )
    a("")
    a(
        f"**Metrics per forecast:** QLIKE; MSE (variance units); mean squared log error; calibration error |b − 1| with b "
        f"the slope of log realized on log forecast; **sign accuracy** = the share of days on which sign(forecast − "
        f"implied) equals sign(realized − implied); hit rate = the share of days the trade made money; the same on the "
        f"**tail days** = the {k1} / {k2} trade days with the largest |straddle return| (|R| ≥ "
        f"{N['tail_R_threshold'][str(k1)]:.2f} / {N['tail_R_threshold'][str(k2)]:.2f} premium, against a median of "
        f"{N['median_absR']:.2f}; the same days for every forecast). **Rank agreement** = Spearman correlation across "
        "the forecasts between a metric and the Sharpe, signed so that +1 means the metric orders the forecasts exactly "
        f"as the Sharpe does; 95 % intervals from a circular block bootstrap over days (block {N['boot']['block']} "
        f"sessions, {N['boot']['B']:,} draws; the whole panel of forecasts resampled together); a **paired difference** "
        "of two agreements is computed on the same draws."
    )
    a("")
    a("## Answer in brief")
    a("")
    pl_q, tr_q = fr(PL, "QLIKE"), fr(TREES, "QLIKE")
    a(
        f"1. **{S('qlike_all')}, {S('qlike_linear')}.** Within the "
        f"{int(pl_q['n'])} per-bar linear forecasts (all buckets, the VIX-only and implied-vol families) the rank "
        f"agreement of QLIKE with the Sharpe is {pl_q['agreement_with_sharpe_mid']:+.2f} {_iv(pl_q['lo'], pl_q['hi'])}; "
        f"within the {int(tr_q['n'])} per-bar trees {tr_q['agreement_with_sharpe_mid']:+.2f} {_iv(tr_q['lo'], tr_q['hi'])}. "
        f"The best-QLIKE forecast ({lab[bq['forecast']]}, QLIKE {bq['QLIKE']:.4f}) ranks {N['sharpe_rank_of_best_qlike']} "
        f"of {M} on the Sharpe ({bq['Sharpe_mid']:.2f}); the best-Sharpe forecast ({lab[bs['forecast']]}, "
        f"{bs['Sharpe_mid']:.2f}) ranks {N['qlike_rank_of_best_sharpe']} of {M} on QLIKE."
    )
    a(
        f"2. **{S('other_losses')}** {S('mse')}; the squared log error {agr('MSE_log')}; the calibration error "
        f"{agr('calib_err')}."
    )
    rt1, rt2 = rk(f"sign_acc_top{k1}"), rk(f"sign_acc_top{k2}")
    rng_s = N["range"]["sign_acc"]
    pl_t = fr(PL, f"sign_acc_top{k1}")
    a(
        f"3. **{S('tail_story')}** Sign accuracy over all days hardly separates the forecasts ({100 * rng_s[0]:.0f}–"
        f"{100 * rng_s[2]:.0f} %, median {100 * rng_s[1]:.0f} %) and {S('all_day_sign')}. On the {k1} largest-|return| "
        f"days {S('tail_sign')}, {S('tail_stronger')} ({rt1['agreement']:+.2f} "
        f"{_iv(rt1['agreement_lo'], rt1['agreement_hi'])}; {k2} days {rt2['agreement']:+.2f} "
        f"{_iv(rt2['agreement_lo'], rt2['agreement_hi'])}){S('tail_stronger_paired')}; and {S('rest_sign')}. "
        f"Within the per-bar linear class, where QLIKE says nothing, the {k1}-day sign accuracy has agreement "
        f"{pl_t['agreement_with_sharpe_mid']:+.2f} {_iv(pl_t['lo'], pl_t['hi'])}: {S('linear_tail')}."
    )
    rs, rl, ra_ = reg.loc["sign_right"], reg.loc["log_error"], reg.loc["abs_log_error"]
    a(
        f"4. **Per day, being on the right side explains the P&L; the size of the forecast error does not.** Regressing "
        f"the day's trade return on an indicator of a right sign gives R² {100 * rs['r2_med']:.1f} % (median; "
        f"{100 * rs['r2_min']:.1f}–{100 * rs['r2_max']:.1f} % across the {M} forecasts; HAC t {rs['t_min']:.1f} to "
        f"{rs['t_max']:.1f}, {S('sign_right_all')}). On the forecast's log error it gives R² "
        f"{100 * rl['r2_med']:.2f} % (median; {int(rl['n_sig'])} of {M} with p < 0.05, the rate chance gives); on its "
        f"absolute log error {100 * ra_['r2_med']:.2f} % (median slope {ra_['slope_med']:+.2f}, {int(ra_['n_sig'])} of {M} "
        f"with p < 0.05: {S('abs_cost')}). Adding the log error to the sign indicator {S('joint_nothing')} (median "
        f"R² {100 * reg.loc['log_error + sign_right', 'r2_med']:.1f} %). For the headline per-bar ridge "
        f"(`{hk}`) the sign is right on {hs['n_right']} of {nd} days (mean return {hs['mean_pnl_right']:+.2f} per unit "
        f"premium) and wrong on {hs['n_wrong']} ({hs['mean_pnl_wrong']:+.2f}); its {k1} tail days carry "
        f"{100 * hs['sum_pnl_top20'] / hs['sum_pnl']:.0f} % of its total P&L ({k2} days "
        f"{100 * hs['sum_pnl_top50'] / hs['sum_pnl']:.0f} %)."
    )
    if MF is not None:
        fam = MF[MF.kind == "family"]
        nF = fam["set"].nunique()

        def members(m: str, side: str) -> pd.DataFrame:
            x = fam[fam.metric == m]
            return x[x.lo > 0] if side == "above" else x[x.hi < 0]

        def fam_name(r: pd.Series) -> str:
            return r["rung"] if isinstance(r["rung"], str) and r["rung"] else r["set"]

        def listing(x: pd.DataFrame) -> str:
            if x.empty:
                return ""
            return (
                " ("
                + "; ".join(
                    f"{fam_name(r)} {r['agreement_with_sharpe_mid']:+.2f} {_iv(r['lo'], r['hi'])}"
                    for _, r in x.iterrows()
                )
                + ")"
            )

        def mean_row(m: str, which: str) -> pd.Series:
            x = MF[(MF.kind == "mean") & (MF.metric == m)]
            return x[x["set"].str.contains(which)].iloc[0]

        def mtxt(m: str, which: str) -> str:
            r = mean_row(m, which)
            return f"{r['agreement_with_sharpe_mid']:+.2f} {_iv(r['lo'], r['hi'])}"

        def rdlohi(which: str) -> tuple[float, float]:
            x = RD[
                RD["set"].str.contains(which) & RD["set"].str.startswith("mean")
            ].iloc[0]
            return float(x["lo"]), float(x["hi"])

        def rdtxt(which: str) -> str:
            x = RD[
                RD["set"].str.contains(which) & RD["set"].str.startswith("mean")
            ].iloc[0]
            return f"{x['diff']:+.2f} {_iv(x['lo'], x['hi'])}"

        nt = int(mean_row("QLIKE", "tree rungs")["n_families"])
        parts = []
        for m, what in (
            ("QLIKE", "QLIKE's agreement with the Sharpe"),
            (f"sign_acc_top{k1}", f"the {k1}-day sign accuracy's agreement"),
        ):
            ab, be = members(m, "above"), members(m, "below")
            parts.append(
                f"{what} has an interval wholly above zero in {len(ab)} of the {nF} families{listing(ab)} and wholly "
                f"below zero in {len(be)}{listing(be)}; its mean over the {nF} families is {mtxt(m, 'model families')}, "
                f"over the {nt} tree rungs {mtxt(m, 'tree rungs')}"
            )
        a(
            "5. **Within the model families** (`rank_vs_sharpe_by_model_family.csv`, figure "
            "`rank_vs_sharpe_by_model_family.png`; a family = the forecasts of one model class on one rung of the ladder, "
            "differing in input set, estimator or selection rule): "
            + "; ".join(parts)
            + f". Within a family, on average, {S('fam_qlike')} and {S('fam_tail')}; paired, the {k1}-day sign "
            f"accuracy's agreement minus QLIKE's is {rdtxt('model families')} over the families and "
            f"{rdtxt('tree rungs')} over the tree rungs"
            + (
                ", intervals that cover zero: on these days the two are not told apart."
                if not any(_sig(*rdlohi(w)) for w in ("model families", "tree rungs"))
                else "."
            )
        )
    a("")
    a(
        "**Verdict.** QLIKE is the right loss for forecasting the 15:30–16:00 variance, and across model families a "
        f"better QLIKE comes with a better trade: {S('bar48_worse')}, and among the per-bar trees the ordering "
        f"{S('trees_qlike')} ({tr_q['agreement_with_sharpe_mid']:+.2f} {_iv(tr_q['lo'], tr_q['hi'])}). Among the "
        f"{int(pl_q['n'])} per-bar linear forecasts, the candidates for the headline, it does not "
        f"({pl_q['agreement_with_sharpe_mid']:+.2f} {_iv(pl_q['lo'], pl_q['hi'])}): there a QLIKE gain does not imply a "
        f"trade gain{S('vx_example')}. The trade is decided by the sign of forecast minus implied"
        + (
            " on a handful of large-move days"
            if C["tail_story"]["holds"]
            else ", with a large share of the P&L on a handful of large-move days"
        )
        + f" (the {k1} largest carry {100 * N['range'][f'pnl_share_top{k1}'][1]:.0f} % of total P&L for the median "
        f"forecast), and QLIKE, an average over all {nd} days of how far the forecast misses, gives them little weight "
        f"(they make up {100 * float((k1 * T[f'QLIKE_top{k1}'] / (nd * T['QLIKE'])).median()):.0f} % of the median "
        "forecast's summed QLIKE). The trade's own measure (sign accuracy weighted by the size of the move, i.e. the "
        f"P&L) is the relevant one for choosing among comparable forecasts, and on {nd} days it is too noisy to rank "
        "them reliably."
    )
    a("")
    a("## Rank agreement with the Sharpe, all forecasts")
    a("")
    a(
        "| metric | better | agreement with Sharpe (mid) | agreement with Sharpe (crossed) |"
    )
    a("|---|---|---|---|")
    order = [
        "QLIKE",
        "MSE",
        "MSE_log",
        "calib_err",
        "sign_acc",
        "hit_rate",
        f"QLIKE_top{k1}",
        f"QLIKE_rest{k1}",
        f"QLIKE_top{k2}",
        f"QLIKE_rest{k2}",
        f"sign_acc_top{k1}",
        f"sign_acc_rest{k1}",
        f"sign_acc_top{k2}",
        f"sign_acc_rest{k2}",
        f"hit_top{k1}",
        f"hit_top{k2}",
    ]
    for m in order:
        r = rk(m)
        a(
            f"| {esc(r['description'])} | {r['better']} | {agr(m)} | {agr(m, 'Sharpe_crossed')} |"
        )
    a("")
    a(
        f"Hit rate on the tail days is almost the P&L itself (a call on a day with |R| ≥ {N['tail_R_threshold'][str(k1)]:.2f} "
        f"wins or loses at least that many premiums), so its agreement is partly mechanical; the tail-day sign accuracy "
        "is the forecast-side version of the same quantity."
    )
    if RD is not None:
        a("")
        a(
            "Paired differences of agreement with the Sharpe (mid), same draws (`rank_vs_sharpe_differences.csv`; "
            "positive = the first metric orders the forecasts more like the Sharpe):"
        )
        a("")
        a("| set | n | first metric | minus | difference [95 %] |")
        a("|---|---:|---|---|---|")
        for _, r in RD.iterrows():
            a(
                f"| {esc(r['set'])} | {int(r['n'])} | {esc(descr(r['metric_a'], nd))} | {esc(descr(r['metric_b'], nd))} | "
                f"{r['diff']:+.2f} {_iv(r['lo'], r['hi'])} |"
            )
        a("")
        a("(For the mean rows n is the number of families averaged.)")
    a("")
    a("## Within classes of forecasts (figure `qlike_vs_sharpe.png` shows the classes)")
    a("")
    cls = list(dict.fromkeys(FR["display_class"]))
    mets = list(dict.fromkeys(FR["metric"]))
    a(
        "| class | n | "
        + " | ".join(esc(FR[FR.metric == m]["description"].iloc[0]) for m in mets)
        + " |"
    )
    a("|---|---:|" + "---|" * len(mets))
    for c in cls:
        cells = []
        for m in mets:
            r = fr(c, m)
            cells.append(
                f"{r['agreement_with_sharpe_mid']:+.2f} {_iv(r['lo'], r['hi'])}"
            )
        a(f"| {c} | {int(fr(c, mets[0])['n'])} | " + " | ".join(cells) + " |")
    if MF is not None:
        a("")
        a(
            "## Within each model family (B4; figure `rank_vs_sharpe_by_model_family.png`)"
        )
        a("")
        a(
            f"Every master-table family with at least {MIN_SET} forecasts, the pooled sets (a display class of several "
            "families; every Optuna rung together) and the mean of the within-family agreement over the families (its "
            "interval from the same draws). Rungs: T10 / T1 = the shipped tree configuration refit every 10 sessions / "
            "every session; RS10 / RS1 = random search (32 candidates, every 250 sessions) refit every 10 / every "
            "session; OP_tp<N>_k<k> = Optuna TPE tuned every N sessions, best of the first k of 50 trials, refit every "
            "session. Agreement with the Sharpe (mid), 95 % day-block interval."
        )
        a("")
        mm = ("QLIKE", f"sign_acc_top{k1}", f"sign_acc_top{k2}")
        a(
            "| set | rung | n | QLIKE range | Sharpe (mid) range | "
            + " | ".join(esc(descr(m, nd)) for m in mm)
            + " |"
        )
        a("|---|---|---:|---|---|" + "---|" * len(mm))
        for s_ in dict.fromkeys(MF["set"]):
            x = MF[MF["set"] == s_].set_index("metric")
            r0 = x.iloc[0]
            rg = r0["rung"] if isinstance(r0["rung"], str) else ""
            n_txt = (
                f"{int(r0['n_families'])} families"
                if r0["kind"] == "mean"
                else f"{int(r0['n'])}"
            )
            a(
                f"| {esc(s_)} | {rg} | {n_txt} | {r0['qlike_min']:.4f} – {r0['qlike_max']:.4f} | "
                f"{r0['sharpe_min']:.2f} – {r0['sharpe_max']:.2f} | "
                + " | ".join(
                    f"{x.loc[m, 'agreement_with_sharpe_mid']:+.2f} {_iv(x.loc[m, 'lo'], x.loc[m, 'hi'])}"
                    if np.isfinite(x.loc[m, "agreement_with_sharpe_mid"])
                    else "n/a (tie)"
                    for m in mm
                )
                + " |"
            )
        a("")
        a(
            "A family of 6–9 forecasts gives a rank correlation an interval of about ±0.7 on these days; the mean over "
            "the families pools them."
        )
    a("")
    a("## Named forecasts")
    a("")
    a(
        f"| forecast | QLIKE | sign accuracy | sign accuracy, {k1} tail days | {k2} tail days | hit rate | "
        f"Sharpe mid / crossed | share of P&L on the {k1} tail days |"
    )
    a("|---|---:|---:|---:|---:|---:|---|---:|")
    for k in KEY:
        if k in T.index:
            r = T.loc[k]
            a(
                f"| {r['label']} (`{k}`) | {r['QLIKE']:.4f} | {100 * r['sign_acc']:.1f} % | "
                f"{100 * r[f'sign_acc_top{k1}']:.0f} % | {100 * r[f'sign_acc_top{k2}']:.0f} % | {100 * r['hit_rate']:.1f} % | "
                f"{r['Sharpe_mid']:.2f} / {r['Sharpe_crossed']:.2f} | {100 * r[f'pnl_share_top{k1}']:.0f} % |"
            )
    a("")
    a(
        "## Per-day regressions (trade return, mid, on the forecast; HAC standard errors)"
    )
    a("")
    a(
        f"Newey–West lag {nw_lag(nd)} (the rule floor(4 (n/100)^(2/9)), n = {nd}); one regression per forecast "
        "(`per_day_regressions.csv`)."
    )
    a("")
    a(
        "| regressor | R² median (range) | slope median | HAC t range | forecasts with p < 0.05 |"
    )
    a("|---|---|---:|---|---:|")
    for x in ("sign_right", "log_error", "abs_log_error", "log_error + sign_right"):
        r = reg.loc[x]
        a(
            f"| {x.replace('_', ' ')} | {100 * r['r2_med']:.2f} % ({100 * r['r2_min']:.2f}–{100 * r['r2_max']:.2f} %) | "
            f"{r['slope_med']:+.3f} | {r['t_min']:+.1f} to {r['t_max']:+.1f} | {int(r['n_sig'])} of {int(r['n'])} |"
        )
    a("")
    a("(For the joint regression the slope and t are the log error's.)")
    a("")
    a(
        f"## The claims re-checked against the {M_prev}-forecast run (commit {PREV_COMMIT})"
    )
    a("")
    if prev is None:
        a(
            f"The outputs of commit {PREV_COMMIT} could not be read from git; nothing to compare."
        )
    else:
        fired = [c for c in C.values() if c["asserted_prev"] and not c["holds"]]
        a(
            f"Every claim above is a check on the numbers (`claims.csv`). The same checks run on the committed outputs of "
            f"the {M_prev}-forecast run (commit {PREV_COMMIT}, read with `git show`; that run had no model-family table "
            "and no paired differences, so those checks read 'not computed' there). 'asserted' = this script stops if "
            f"the check fails; 'asserted at {PREV_COMMIT}' = the script of that commit did."
        )
        a("")
        a(
            "**Assertions of the "
            f"{PREV_COMMIT} script that fire on this table:** "
            + (
                "; ".join(
                    f"{c['claim']} ({c['rule']}: {c['value']}) — the prose now follows the check"
                    for c in fired
                )
                if fired
                else "none"
            )
            + "."
        )
        a("")
        a(
            f"| claim | check | asserted (now / at {PREV_COMMIT}) | {M_prev} forecasts | {M} forecasts (this run) |"
        )
        a("|---|---|---|---|---|")

        def cell(c: dict | None) -> str:
            if not c or c.get("holds") is None:
                return "not computed"
            return (
                "holds" if c["holds"] else "**does not hold**"
            ) + f": {esc(c['value'])}"

        for cid, c in C.items():
            a(
                f"| {esc(c['claim'])} | {esc(c['rule'])} | {'yes' if c['asserted'] else 'no'} / "
                f"{'yes' if c['asserted_prev'] else 'no'} | {cell(CP.get(cid))} | {cell(c)} |"
            )
        a("")

        def shape(text: str) -> str:  # a statement's wording with its numbers masked
            return re.sub(r"[+\-−]?\d[\d.,]*", "#", text)

        changed = [
            cid
            for cid, c in C.items()
            if cid in CP
            and CP[cid]["holds"] is not None
            and c["holds"] is not None
            and (
                CP[cid]["holds"] != c["holds"]
                or shape(CP[cid]["statement"]) != shape(c["statement"])
            )
        ]
        a(
            "**Statements that changed** (the check's outcome or the wording that follows from the numbers; old "
            "statement → new statement, each rendered by this script from its run's numbers; for the "
            f"{M_prev}-forecast run these are the statements of its committed SUMMARY.md):"
        )
        a("")
        if not changed:
            a("- none")
        for cid in changed:
            a(
                f"- *{C[cid]['claim']}*: {M_prev} forecasts: “{CP[cid]['statement'].strip(' ;.')}” → {M} forecasts: "
                f"“{C[cid]['statement'].strip(' ;.')}”."
            )
        pc = prev["N"]["classes"]
        a(
            f"- *display classes*: {M_prev} forecasts: "
            + ", ".join(f"{c} {n}" for c, n in pc.items() if n)
            + f" → {M} forecasts: "
            + ", ".join(f"{c} {n}" for c, n in N["classes"].items() if n)
            + " (the LSTM rows, 'other' before, are a class of their own; every new tree rung joins the per-bar trees)."
        )
    a("")
    a("## Notes")
    a("")
    a(
        f"* The headline forecast `{hk}`: QLIKE {h['QLIKE']:.4f}, Sharpe {h['Sharpe_mid']:.2f} mid / "
        f"{h['Sharpe_crossed']:.2f} crossed, the master table's numbers."
    )
    a(
        "* The intervals are percentile intervals of the bootstrap distribution of the rank correlation; resampling days "
        "adds noise to every forecast's metrics, which pulls the resampled correlations toward zero, so the point estimate "
        "can sit near one end of its interval."
    )
    intra = T[T.index.str.contains("har_intra")]
    a(
        "* The classes are the master table's families grouped for the figure: 48-bar = the paper's forecasts and the "
        "pooled twins; per-bar linear = per-bar ridge / lasso / elastic net on every bucket, the VIX-only family and the "
        "implied-vol representations; HAR-ladder variants apart"
        + (
            f" ({len(intra)} of them build the ladder within the session and sit at QLIKE {intra['QLIKE'].min():.2f}–"
            f"{intra['QLIKE'].max():.2f}, drawn at the edge of the figure)"
            if len(intra)
            else ""
        )
        + "; per-bar trees = LightGBM / XGBoost / random forest on every rung of the 16:00 ladder (T10, T1, RS10, RS1, "
        "the Optuna rungs); LSTM = the per-bar LSTM and the intraday-sequence LSTM."
    )
    (OUT / "SUMMARY.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {OUT / 'SUMMARY.md'} ({len(L)} lines)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        write_summary()
    else:
        main()
        write_summary()
