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

Outputs (results/perf_vs_pnl/): forecast_metrics.csv, rank_vs_sharpe.csv,
rank_vs_sharpe_by_family.csv, per_day_regressions.csv, gates.csv, qlike_vs_sharpe.png,
rank_vs_sharpe.png, numbers.json (every number the SUMMARY quotes), run_output.txt.

Run:  python experiments/perf_vs_pnl_1530.py
"""

from __future__ import annotations

import json
import sys
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
# display classes for the scatter (the master table's families, grouped to 5 slots)
CLASS_OF = {
    "paper": "48-bar (paper + pooled twins)",
    "pooled twin": "48-bar (paper + pooled twins)",
    "per-bar linear": "per-bar linear",
    "VIX-only family": "per-bar linear",
    "implied-vol representations": "per-bar linear",
    "HAR-ladder variants": "per-bar linear, HAR-ladder variants",
    "per-bar tree (untuned)": "per-bar trees",
    "per-bar tree (tuned)": "per-bar trees",
    "other table": "per-bar trees",
}
CLASS_ORDER = (
    "48-bar (paper + pooled twins)",
    "per-bar linear",
    "per-bar linear, HAR-ladder variants",
    "per-bar trees",
    "other",
)
# validated categorical slots (the palette of model_diagnostics_1530_trees.py)
CLASS_COLOR = dict(
    zip(CLASS_ORDER, ("#2a78d6", "#4a3aa7", "#1baf7a", "#eb6834", "#eda100"))
)
CLASS_MARK = dict(zip(CLASS_ORDER, ("s", "o", "D", "^", "v")))
QLIKE_XMAX = 0.125  # the scatter's x range; forecasts beyond it are drawn at the edge


def classify(fam: str, key: str) -> str:
    if fam in CLASS_OF:
        return CLASS_OF[fam]
    if key.startswith("subtree_"):
        return "per-bar trees"
    return "other"


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
    big = [c for c in CLASS_ORDER if (cls == c).sum() >= 5]
    cdraws = {(c, m): np.empty(B) for c in big for m in CLASS_METS}
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


def write_summary() -> None:  # noqa: C901 - one linear report
    """SUMMARY.md from this folder's own outputs; the claims the verdict rests on are
    asserted against the numbers first."""
    RK = pd.read_csv(OUT / "rank_vs_sharpe.csv")
    FR = pd.read_csv(OUT / "rank_vs_sharpe_by_family.csv")
    RG = pd.read_csv(OUT / "per_day_regressions.csv")
    T = pd.read_csv(OUT / "forecast_metrics.csv", index_col=0)
    N = json.loads((OUT / "numbers.json").read_text(encoding="utf-8"))
    M, nd = N["n_forecasts"], N["n_days"]
    k1, k2 = TAIL_K

    def rk(m: str, s: str = "Sharpe_mid") -> pd.Series:
        return RK[(RK.metric == m) & (RK.sharpe == s)].iloc[0]

    def fr(c: str, m: str) -> pd.Series:
        return FR[(FR.display_class == c) & (FR.metric == m)].iloc[0]

    def agr(m: str, s: str = "Sharpe_mid") -> str:
        r = rk(m, s)
        return f"{r['agreement']:+.2f} {_iv(r['agreement_lo'], r['agreement_hi'])}"

    reg = RG.groupby("regressor").agg(
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
    PL = "per-bar linear"
    # ---- the claims, checked
    q = rk("QLIKE")
    assert q["agreement_lo"] > 0, "QLIKE ranks like the Sharpe across all forecasts"
    assert rk("MSE")["agreement"] < 0 and not _sig(
        rk("MSE")["agreement_lo"], rk("MSE")["agreement_hi"]
    )
    # all-day sign accuracy may or may not rank the forecasts (with the 100-forecast
    # table it did not; the six LSTM rows, weak on both counts, pull the interval
    # just past zero) -- the claim the verdict rests on is that the tail-day sign
    # accuracy ranks them, and more strongly than the all-day one
    assert rk(f"sign_acc_top{k1}")["agreement_lo"] > 0
    assert rk(f"sign_acc_top{k1}")["agreement"] > rk("sign_acc")["agreement"]
    assert not _sig(
        rk(f"sign_acc_rest{k1}")["agreement_lo"],
        rk(f"sign_acc_rest{k1}")["agreement_hi"],
    )
    assert not _sig(fr(PL, "QLIKE")["lo"], fr(PL, "QLIKE")["hi"])
    assert (
        reg.loc["sign_right", "n_sig"] == M and reg.loc["log_error", "n_sig"] <= 0.1 * M
    )
    assert reg.loc["log_error", "r2_max"] < reg.loc["sign_right", "r2_min"]
    hs = N["headline_split"]
    hk = hs["forecast"]
    h = T.loc[hk]
    bq, bs = N["best_qlike"], N["best_sharpe"]
    lab = T["label"].to_dict()

    L: list[str] = []
    a = L.append
    a("# Forecast accuracy vs P&L of the 15:30 last-30-min trade (checklist C3)")
    a("")
    a(
        "Written by `experiments/perf_vs_pnl_1530.py` from its own outputs in this folder (`forecast_metrics.csv`, "
        "`rank_vs_sharpe.csv`, `rank_vs_sharpe_by_family.csv`, `per_day_regressions.csv`, `numbers.json`); the claims the "
        "verdict rests on are asserted against those numbers before this file is written."
    )
    a("")
    src = (
        "the table-A rows of the closing-strategy master table (`results/close_master_table/master_table.csv`, "
        "`master_table_daily.parquet`; check rows, exact duplicates and the always-short rule left out)"
        if N["source"] == "master"
        else "every forecast table on disk (`results/spxw_pnl/yhat_*.parquet`)"
    )
    a(
        f"**Universe:** {M} forecasts of the bar ending 16:00: {src}. By class: "
        + ", ".join(f"{c} {n}" for c, n in N["classes"].items() if n)
        + f". Same {nd} trade days for every forecast "
        f"({N['first_day']} .. {N['last_day']})."
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
        f"sessions, {N['boot']['B']:,} draws; the whole panel of forecasts resampled together)."
    )
    a("")
    a("## Answer in brief")
    a("")
    pl_q, tr_q = fr(PL, "QLIKE"), fr("per-bar trees", "QLIKE")
    a(
        f"1. **Across all {M} forecasts a lower QLIKE goes with a higher Sharpe** (Spearman {q['spearman']:+.2f}, "
        f"interval {_iv(q['ci_lo'], q['ci_hi'])}), **but not among forecasts of similar accuracy.** Within the "
        f"{int(pl_q['n'])} per-bar linear forecasts (all buckets, the VIX-only and implied-vol families) the rank "
        f"agreement of QLIKE with the Sharpe is {pl_q['agreement_with_sharpe_mid']:+.2f} {_iv(pl_q['lo'], pl_q['hi'])}; "
        f"within the {int(tr_q['n'])} per-bar trees {tr_q['agreement_with_sharpe_mid']:+.2f} {_iv(tr_q['lo'], tr_q['hi'])}. "
        f"The best-QLIKE forecast ({lab[bq['forecast']]}, QLIKE {bq['QLIKE']:.4f}) ranks {N['sharpe_rank_of_best_qlike']} "
        f"of {M} on the Sharpe ({bq['Sharpe_mid']:.2f}); the best-Sharpe forecast ({lab[bs['forecast']]}, "
        f"{bs['Sharpe_mid']:.2f}) ranks {N['qlike_rank_of_best_sharpe']} of {M} on QLIKE."
    )
    a(
        f"2. **The other accuracy losses do no better.** MSE in variance units orders the forecasts the wrong way round "
        f"(agreement {agr('MSE')}); the squared log error {agr('MSE_log')}; the calibration error {agr('calib_err')}."
    )
    ra, rt1, rt2, rr1 = (
        rk("sign_acc"),
        rk(f"sign_acc_top{k1}"),
        rk(f"sign_acc_top{k2}"),
        rk(f"sign_acc_rest{k1}"),
    )
    rng_s = N["range"]["sign_acc"]
    pl_t = fr(PL, f"sign_acc_top{k1}")
    all_day_ranks = _sig(ra["agreement_lo"], ra["agreement_hi"])
    all_day_clause = (
        f"ranks them only weakly (agreement {ra['agreement']:+.2f} {_iv(ra['agreement_lo'], ra['agreement_hi'])})"
        if all_day_ranks
        else f"does not rank them (agreement {ra['agreement']:+.2f} {_iv(ra['agreement_lo'], ra['agreement_hi'])})"
    )
    a(
        f"3. **What does track the trade is calling the side right on the few days the straddle moves most.** Sign "
        f"accuracy over all days hardly separates the forecasts ({100 * rng_s[0]:.0f}–{100 * rng_s[2]:.0f} %, median "
        f"{100 * rng_s[1]:.0f} %) and {all_day_clause}. On the {k1} largest-|return| days it does, and more strongly "
        f"({rt1['agreement']:+.2f} {_iv(rt1['agreement_lo'], rt1['agreement_hi'])}; {k2} days {rt2['agreement']:+.2f} "
        f"{_iv(rt2['agreement_lo'], rt2['agreement_hi'])}), and on the other {nd - k1} days it does not "
        f"({rr1['agreement']:+.2f} {_iv(rr1['agreement_lo'], rr1['agreement_hi'])}). Within the per-bar linear class, "
        f"where QLIKE says nothing, the {k1}-day sign accuracy has agreement {pl_t['agreement_with_sharpe_mid']:+.2f} "
        f"{_iv(pl_t['lo'], pl_t['hi'])}: suggestive, with an interval that reaches zero."
    )
    rs, rl, ra_ = reg.loc["sign_right"], reg.loc["log_error"], reg.loc["abs_log_error"]
    a(
        f"4. **Per day, being on the right side explains the P&L; the size of the forecast error does not.** Regressing "
        f"the day's trade return on an indicator of a right sign gives R² {100 * rs['r2_med']:.1f} % (median; "
        f"{100 * rs['r2_min']:.1f}–{100 * rs['r2_max']:.1f} % across the {M} forecasts; HAC t {rs['t_min']:.1f} to "
        f"{rs['t_max']:.1f}, every forecast significant). On the forecast's log error it gives R² "
        f"{100 * rl['r2_med']:.2f} % (median; {int(rl['n_sig'])} of {M} with p < 0.05, the rate chance gives); on its "
        f"absolute log error {100 * ra_['r2_med']:.2f} % (median slope {ra_['slope_med']:+.2f}, {int(ra_['n_sig'])} of {M} "
        f"with p < 0.05: larger misses cost a little). Adding the log error to the sign indicator adds nothing (median "
        f"R² {100 * reg.loc['log_error + sign_right', 'r2_med']:.1f} %). For the headline per-bar ridge "
        f"(`{hk}`) the sign is right on {hs['n_right']} of {nd} days (mean return {hs['mean_pnl_right']:+.2f} per unit "
        f"premium) and wrong on {hs['n_wrong']} ({hs['mean_pnl_wrong']:+.2f}); its {k1} tail days carry "
        f"{100 * hs['sum_pnl_top20'] / hs['sum_pnl']:.0f} % of its total P&L ({k2} days "
        f"{100 * hs['sum_pnl_top50'] / hs['sum_pnl']:.0f} %)."
    )
    a("")
    vx = "sub_ridge_vix_only"
    trees_holds = _sig(tr_q["lo"], tr_q["hi"])
    vx_txt = (
        f" (e.g. the per-bar ridge on the VIX-only bucket: QLIKE {T.loc[vx, 'QLIKE']:.4f} against the headline's "
        f"{h['QLIKE']:.4f}, Sharpe {T.loc[vx, 'Sharpe_mid']:.2f} against {h['Sharpe_mid']:.2f})"
        if vx in T.index
        and T.loc[vx, "QLIKE"] < h["QLIKE"]
        and T.loc[vx, "Sharpe_mid"] < h["Sharpe_mid"]
        else ""
    )
    a(
        "**Verdict.** QLIKE is the right loss for forecasting the 15:30–16:00 variance, and across model families a "
        "better QLIKE comes with a better trade: the 48-bar forecasts forecast worse and trade worse than the per-bar "
        "ones, and among the per-bar trees the ordering "
        + ("holds as well" if trees_holds else "points the same way")
        + f" ({tr_q['agreement_with_sharpe_mid']:+.2f} {_iv(tr_q['lo'], tr_q['hi'])}). Among the {int(pl_q['n'])} "
        f"per-bar linear forecasts, the candidates for the headline, it does not ({pl_q['agreement_with_sharpe_mid']:+.2f} "
        f"{_iv(pl_q['lo'], pl_q['hi'])}): there a QLIKE gain does not imply a trade gain{vx_txt}. The trade is decided by "
        f"the sign of forecast minus implied on a handful of large-move days (the {k1} largest carry "
        f"{100 * N['range'][f'pnl_share_top{k1}'][1]:.0f} % of total P&L for the median forecast), and QLIKE, an average "
        f"over all {nd} days of how far the forecast misses, gives them little weight (they make up "
        f"{100 * float((k1 * T[f'QLIKE_top{k1}'] / (nd * T['QLIKE'])).median()):.0f} % of the median forecast's summed "
        f"QLIKE). The trade's own measure (sign "
        f"accuracy weighted by the size of the move, i.e. the P&L) is the relevant one for choosing among comparable "
        f"forecasts, and on {nd} days it is too noisy to rank them reliably."
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
        + "; per-bar trees = untuned and tuned LightGBM / XGBoost / random forest."
    )
    (OUT / "SUMMARY.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {OUT / 'SUMMARY.md'} ({len(L)} lines)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        write_summary()
    else:
        main()
        write_summary()
