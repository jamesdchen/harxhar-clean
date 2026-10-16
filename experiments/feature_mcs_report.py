"""Feature Model Confidence Set (2026-10-08): scoring, losses, the MCS, SUMMARY.md.

Reads the candidate forecasts of experiments/feature_mcs_runner.py (results/feature_mcs_2026-10-08/_work/
models[_ridge|_lasso]/<model>.npz, run on CARC, every model on an EPYC 7513): the paper's pre-tuned per-bar
elastic net, ridge and lasso (arm f of the 2026-10-06 study) on 96 input designs each.  Scores them like
the master table and the 2026-10-06 study (close_linear_pretune.Scorer, read-only: plain forecast f^2 x B
with NO recalibration on the 866 trade days 2020-01-03 .. 2024-04-30, the master table's 16:00 realized
variance and 15:30 implied variance, the daily_a0 straddle quotes).  Four daily losses:
  qlike    r - log r - 1, r = realized variance / plain forecast
  mse_fit  squared error in the fit space: (adjusted-scale forecast - adjusted-scale target)^2, the
           squared error the feature-importance study scores (experiments/feature_importance_1530.py:
           m0 = mean((yt - yh0)^2), yt the design's target y, yh0 the forecast pred_adj)
  es1      the elementary score at the trade's threshold, close_murphy_splits_headline.elementary with
           c = 1, forecast and target divided by the 15:30 implied variance (the paper's D.17):
           |y/s - 1| 1{plain/s and y/s on opposite sides of 1}
  trade    minus the daily mid P&L of the sign(s) straddle (q = sign(plain - iv), P&L = q R; the
           master table's ret_mid)
Trees (2026-10-08, user): the tree agent's per-bar LightGBM (lgbm_opt_00), XGBoost (xgb_opt_00) and random
forest (trial 255) on the same 96 designs (fixed settings, refit every session), read from

  results/feature_mcs_2026-10-08/trees/<est>/<model>.npz     est in {lgbm, xgb, rf}; model = the 96 names of
                                                             _work/inputs/models.csv (feature_mcs_runner designs)
      pred   float64, the ADJUSTED-SCALE forecast, the scale of the linear runs' `pred` (plain forecast =
             pred^2 x B; the fit-space MSE is (pred - y)^2 with y the design's target), either
             (a) length 1469: every 16:00 forecast row of the design (rows W .. W+1468, dates = inputs.json
                 dates[W:]), or
             (b) length 866: the trade days only, in the order of `date` (or, without `date`, in the order of
                 the scorer's 866 trade days)
      date   optional for (a), recommended for (b): the forecast timestamps, strings 'YYYY-MM-DD 16:00:00' (or
             'YYYY-MM-DD'; tz-aware stamps are converted to New York time), same length as pred
  results/feature_mcs_2026-10-08/trees/<est>/<model>.json    optional: {"cpu": CPU model string, "host": ...,
                                                             "cpu_total": CPU seconds, "wall": seconds}
  results/feature_mcs_2026-10-08/trees/TREES_DONE            written last, when all 3 x 96 files are in place

The trees are read only when TREES_DONE exists and all 288 files are present; the report checks that every
trade day is covered, that the forecasts are finite and positive in plain terms, and (when "cpu" is given)
that the trees ran on one CPU class.
Sensitivity losses (2026-10-12, user: does any loss give the MCS power?), on the same forecasts:
  log_mse  (log(f^2 B) - log RV)^2 on the 16:00 bar -- a SENSITIVITY loss: not robust to noise in the
           realized-variance proxy (Patton 2011, J. Econometrics 160), so a ranking by it may differ from the
           ranking by the true conditional variance
  hit      1{sign(f^2 B - IV) != sign(RV - IV)}, IV the 15:30 implied variance: the trade's direction without
           P&L weighting (= the scorer's q against the sign of RV - IV)
  qlike_w  QLIKE winsorized in differentials, and trade_w the same for the trade loss: for a scope with daily
  trade_w  loss matrix L (866 days x m candidates), D[t, j] = L[t, j] - mean_k L[t, k] (the cross-candidate mean
           of that scope on day t); q01 / q99 = the 1 % / 99 % points (numpy linear interpolation) of all
           866 m pooled values of D; the MCS runs on mean_k L[t, k] + clip(D[t, j], q01, q99).  The MCS
           depends on L only through between-candidate differences, so adding the daily mean back changes
           nothing but keeps the scale readable.  The clip points are fixed once per scope (before any
           elimination) and recorded in winsorize.csv; each scope (estimator, joint, grand) is winsorized
           on its own candidates.
MCS scopes: each estimator's 96 candidates; JOINT over the 288 linear candidates (estimator x design);
with the trees, JOINT_TREES over the 288 tree candidates and GRAND over all 576.
MCS: src/evaluation/model_confidence_set.py with resample=True (the circular moving-block bootstrap,
restored for this study), B = 10,000, block lengths 5 / 10 / 21, seed 20261008; T_max (the module's rule)
and T_R (range statistic) on the same replicates; HLN MCS p-values by full elimination (the last model
p = 1); the set at level alpha = the models with p >= alpha (gate: the module's own T_max loop at alpha
0.10 / 0.25 returns the same set).  T_R is computed incrementally (each replicate's row maxima over the
active set are kept across elimination rounds and only the rows whose maximum was the eliminated model are
recomputed): the same numbers as the brute-force pair scan (gate), O(B m^2) instead of O(B m^3).  Each (scope,
loss, block) is computed once and cached in _work/mcs_cache/; a run stops starting new ones after --budget
seconds and exits 3 until every one is cached.  Context: mean losses, Sharpe mid, Diebold-Mariano
(src.evaluation.diebold_mariano.dm_test) of each candidate against its own estimator's HAR + calendar and
(leave-one-out designs) against its own estimator's all-features model, the Newey-West t of the daily mid
P&L difference (atm_straddle_lib.newey_west_t).

    python experiments/feature_mcs_report.py [--budget SECONDS] [--workers N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
for _p in (
    REPO,
    REPO / "experiments",
    REPO / "notebooks",
    REPO / "specs",
    REPO / "writeup",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import close_linear_pretune as CLP  # noqa: E402
import feature_mcs_runner as FR  # noqa: E402

from src.evaluation import model_confidence_set as MCSM  # noqa: E402
from src.evaluation.diebold_mariano import dm_test  # noqa: E402

OUT = FR.OUT
WORK = FR.WORK
CACHE = WORK / "mcs_cache"
B_BOOT = 10_000
BLOCKS = (5, 10, 21)
ALPHAS = (0.10, 0.25)
SEED = 20261008
LIN = ("enet", "ridge", "lasso")
TREES = ("lgbm", "xgb", "rf")  # the tree agent's forecasts (2026-10-08)
SHORTS = LIN + TREES
EST_OF = {"enet": "reclasticnet", "ridge": "ridge", "lasso": "reclasso"}
EST_LABEL = {
    "enet": "elastic net",
    "ridge": "ridge",
    "lasso": "lasso",
    "lgbm": "LightGBM (lgbm_opt_00)",
    "xgb": "XGBoost (xgb_opt_00)",
    "rf": "random forest (trial 255)",
}
TREE_DIR = OUT / "trees"
TREES_DONE = TREE_DIR / "TREES_DONE"
JOINTS = {"joint": LIN, "joint_trees": TREES, "grand": SHORTS}
JOINT_LABEL = {
    "joint": "joint, the 288 linear candidates",
    "joint_trees": "joint, the 288 tree candidates",
    "grand": "grand joint, all 576 candidates",
}
LOSSES = ("qlike", "mse_fit", "es1", "trade")
LOSS_LABEL = {
    "qlike": "QLIKE of the plain forecast",
    "mse_fit": "squared error in the fit space (adjusted scale)",
    "es1": "elementary score at the trade's threshold (c = 1, scaled by the 15:30 implied variance)",
    "trade": "trade loss = - daily mid P&L of sign(s)",
}
LOSS_SHORT = {"qlike": "QLIKE", "mse_fit": "MSE fit", "es1": "ES(1)", "trade": "trade"}
NEW_LOSSES = ("log_mse", "hit", "qlike_w", "trade_w")  # 2026-10-12, sensitivity losses
ALL_LOSSES = LOSSES + NEW_LOSSES
BASE_LOSSES = (
    "qlike",
    "mse_fit",
    "es1",
    "trade",
    "log_mse",
    "hit",
)  # per-candidate daily losses
WINSOR = {
    "qlike_w": "qlike",
    "trade_w": "trade",
}  # winsorized in differentials, per scope
WIN_Q = (0.01, 0.99)
LOSS_LABEL |= {
    "log_mse": "log-MSE, (log plain - log RV)^2 (sensitivity loss, not robust to proxy noise)",
    "hit": "hit loss, 1{sign(plain - IV) != sign(RV - IV)}",
    "qlike_w": "QLIKE, differentials vs the cross-candidate mean winsorized at 1 % / 99 %",
    "trade_w": "trade loss, differentials vs the cross-candidate mean winsorized at 1 % / 99 %",
}
LOSS_SHORT |= {
    "log_mse": "log-MSE",
    "hit": "hit",
    "qlike_w": "QLIKE (w)",
    "trade_w": "trade (w)",
}
FAVORED = (
    "add_sumvolume",
    "add_sumabsret_vwstock",
    "add_sumabsret_ewstock",
)  # the pairwise evidence's inputs


def winsorize(L: np.ndarray) -> tuple[np.ndarray, float, float, float, float]:
    """The scope's loss matrix with each day's differentials vs the cross-candidate mean clipped at the 1 % /
    99 % points of the pooled differentials (module docstring).  -> (matrix, q01, q99, share low, share high)."""
    mu = L.mean(axis=1, keepdims=True)
    D = L - mu
    q1, q9 = np.quantile(D, WIN_Q)
    return (
        mu + np.clip(D, q1, q9),
        float(q1),
        float(q9),
        float((D < q1).mean()),
        float((D > q9).mean()),
    )


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================================ MCS (both statistics)
def _tr_dist(Z: np.ndarray, a: np.ndarray, ia: np.ndarray) -> np.ndarray:
    """max over pairs i < j of |Z_i - Z_j| / sd_ij for every replicate (the upper triangle, row by row)."""
    Za = np.ascontiguousarray(Z[:, a])
    out = np.zeros(Za.shape[0])
    for i in range(len(a) - 1):
        d = np.abs(Za[:, i + 1 :] - Za[:, i : i + 1])
        d *= ia[i, i + 1 :]
        np.maximum(out, d.max(axis=1), out=out)
    return out


def _isd(Z: np.ndarray) -> np.ndarray:
    """1 / sd of every bootstrap pair difference Lstar_i - Lstar_j about the sample (0 on the diagonal)."""
    S = Z.T @ Z / len(Z)
    dg = np.diag(S)
    isd = 1.0 / np.sqrt(np.maximum(dg[:, None] + dg[None, :] - 2 * S, 1e-300))
    np.fill_diagonal(isd, 0.0)
    return isd


def mcs_tr(L: np.ndarray, Lstar: np.ndarray) -> dict:
    """T_R by full elimination, incrementally: rowmax[b, i] = max over active j of |Z_bi - Z_bj| / sd_ij and its
    argmax are kept; eliminating e recomputes only the (b, i) whose argmax was e.  The replicate statistic is
    max_i rowmax[b, i] over the active i -- the same maxima as mcs_full(stat='TR'), exactly."""
    m = L.shape[1]
    Lbar = L.mean(axis=0)
    Z = Lstar - Lbar
    B = len(Z)
    isd = _isd(Z)
    rowmax = np.empty((B, m))
    rowarg = np.empty((B, m), dtype=np.int32)
    rb = np.arange(B)
    for i in range(m):
        d = np.abs(Z - Z[:, i : i + 1])
        d *= isd[i]
        rowarg[:, i] = d.argmax(axis=1)
        rowmax[:, i] = d[rb, rowarg[:, i]]
    active = np.ones(m, bool)
    p_run, pv = 0.0, np.ones(m)
    order, tests = [], []
    while active.sum() > 1:
        a = np.flatnonzero(active)
        tij = (Lbar[a][:, None] - Lbar[a][None, :]) * isd[np.ix_(a, a)]
        tstat = float(np.abs(tij).max())
        tdist = rowmax[:, a].max(axis=1)
        worst = int(a[int(np.argmax(tij.max(axis=1)))])
        p = float(np.mean(tdist >= tstat))
        p_run = max(p_run, p)
        tests.append(p)
        pv[worst] = p_run
        order.append(worst)
        active[worst] = False
        rowmax[:, worst] = -np.inf
        bi, ii = np.nonzero((rowarg == worst) & active[None, :])
        act = active.astype(float)
        for c0 in range(0, len(bi), 4000):
            b_, i_ = bi[c0 : c0 + 4000], ii[c0 : c0 + 4000]
            d = np.abs(Z[b_] - Z[b_, i_][:, None])
            d *= isd[i_] * act[None, :]
            j_ = d.argmax(axis=1)
            rowarg[b_, i_] = j_
            rowmax[b_, i_] = d[np.arange(len(b_)), j_]
    last = int(np.flatnonzero(active)[0])
    pv[last] = 1.0
    order.append(last)
    return dict(p=pv, order=np.array(order), tests=np.array(tests))


def mcs_full(L: np.ndarray, Lstar: np.ndarray, stat: str) -> dict:
    """HLN MCS p-values for every column of L (T x m losses) by full elimination, on the bootstrap
    column means Lstar (B x m; the module's _boot_col_means(resample=True)).  stat: 'Tmax' (the module's
    rule) or 'TR' (range statistic).  Returns p (m,), the elimination order, each round's test p-value."""
    m = L.shape[1]
    Lbar = L.mean(axis=0)
    Z = Lstar - Lbar  # (B, m): bootstrap deviations about the sample means
    active = list(range(m))
    p_run, pv = 0.0, np.ones(m)
    order, tests = [], []
    if stat == "TR":
        S = Z.T @ Z / len(Z)
        dg = np.diag(S)
        V = np.maximum(
            dg[:, None] + dg[None, :] - 2 * S, 1e-300
        )  # var of Lstar_i - Lstar_j
        isd = 1.0 / np.sqrt(V)
        np.fill_diagonal(isd, 0.0)
    while len(active) > 1:
        a = np.asarray(active)
        if stat == "Tmax":
            dbar = Lbar[a] - Lbar[a].mean()
            dz = Z[:, a] - Z[:, a].mean(
                axis=1, keepdims=True
            )  # = dstar - dbar of the module
            var = np.maximum((dz**2).mean(axis=0), 1e-300)
            t = dbar / np.sqrt(var)
            tstat = float(t.max())
            tdist = (dz / np.sqrt(var)).max(axis=1)
            worst = int(np.argmax(t))
        else:
            ia = isd[np.ix_(a, a)]
            tij = (Lbar[a][:, None] - Lbar[a][None, :]) * ia
            tstat = float(np.abs(tij).max())
            tdist = _tr_dist(Z, a, ia)
            worst = int(np.argmax(tij.max(axis=1)))
        p = float(np.mean(tdist >= tstat))
        p_run = max(p_run, p)
        tests.append(p)
        pv[active[worst]] = p_run
        order.append(active[worst])
        del active[worst]
    pv[active[0]] = 1.0
    order.append(active[0])
    return dict(p=pv, order=np.array(order), tests=np.array(tests))


def mcs_job(job: dict) -> str:
    """One (scope, loss, block): bootstrap means once, T_max, T_R and the module's own T_max loop at
    both alphas -> CACHE/<scope>_<loss>_b<block>.npz."""
    t0 = time.time()
    L, names, blk = job["L"], job["names"], job["block"]
    Lstar = MCSM._boot_col_means(L, B_BOOT, blk, SEED, resample=True)
    res = {"Tmax": mcs_full(L, Lstar, "Tmax")}
    if job.get("no_tr"):
        res["TR"] = dict(
            p=np.full(len(names), np.nan),
            order=np.arange(len(names)),
            tests=np.full(len(names) - 1, np.nan),
        )
    else:
        res["TR"] = mcs_tr(L, Lstar)
    mod = {}
    for al in ALPHAS:
        r = MCSM.model_confidence_set(
            {n: L[:, j] for j, n in enumerate(names)},
            alpha=al,
            B=B_BOOT,
            block=blk,
            seed=SEED,
            resample=True,
        )
        mod[al] = sorted(r["mcs"])
    f = CACHE / f"{job['key']}.npz"
    tmp = CACHE / f"{job['key']}.tmp.npz"
    np.savez(
        tmp,
        names=np.array(names),
        **{f"{st}_{k}": v for st, r_ in res.items() for k, v in r_.items()},
        mod10=np.array(mod[0.10]),
        mod25=np.array(mod[0.25]),
        sec=np.array(time.time() - t0),
    )
    tmp.replace(f)
    return f"{job['key']}: {time.time() - t0:.0f}s"


# ============================================================================ data
CORE_KINDS = (
    "anchor",
    "add",
    "loo",
)  # the 96 designs of the MCS; addb / loob (2026-10-12) are studied apart


def core_models() -> pd.DataFrame:
    """models.csv without the bucket designs (the 96 MCS designs, in their order)."""
    md = pd.read_csv(FR.INP / "models.csv").drop(columns=["cols"])
    return md[md["kind"].isin(CORE_KINDS)].reset_index(drop=True)


def cand_losses(S, pred: np.ndarray, y_adj: np.ndarray, day_row: np.ndarray) -> dict:
    """Every daily loss of one forecast (adjusted scale, 1469 forecast rows) on the 866 trade days: the scorer's
    plain / q / QLIKE / mid P&L plus the fit-space squared error, ES(1), log-MSE and the hit loss."""
    from close_murphy_splits_headline import elementary

    d = S.daily(pred)
    # d["mse_fit"] = (pred[day_row] - y_adj[day_row]) ** 2  # 2026-10-15 (user): parked -- scored against the trimmed target; scores now use actual RV (data audit #10)
    d["mse_fit"] = (
        pred[day_row] - S.y_adj
    ) ** 2  # vs sqrt(actual RV / B) on the trade days
    # S.target is the actual RV since 2026-10-15 (close_linear_pretune.Scorer), so ES(1), log-MSE and hit follow it
    d["es1"] = elementary(d["plain"] / S.iv, S.target / S.iv, 1.0)
    d["log_mse"] = (np.log(d["plain"]) - np.log(S.target)) ** 2
    d["hit"] = (d["q"] != np.where(S.target > S.iv, 1.0, -1.0)).astype(np.float64)
    d["pred"] = pred
    return d


def tree_pred(
    f: Path, fdates: pd.DatetimeIndex, day_row: np.ndarray
) -> tuple[np.ndarray, str]:
    """One tree forecast file -> the adjusted-scale forecast on the 1469 forecast rows (NaN where not given)."""
    z = np.load(f, allow_pickle=False)
    pred = np.asarray(z["pred"], np.float64)
    out = np.full(len(fdates), np.nan)
    fday = pd.DatetimeIndex(CLP.as_day(fdates))
    if "date" in z.files:
        idx = fday.get_indexer(
            pd.DatetimeIndex(CLP.as_day(np.asarray(z["date"]).astype(str)))
        )
        assert (idx >= 0).all() and len(set(idx.tolist())) == len(idx), (
            f"{f}: dates off the forecast rows"
        )
        out[idx] = pred
        return out, "date"
    if len(pred) == len(fdates):
        return pred, "rows"
    assert len(pred) == len(day_row), (
        f"{f}: {len(pred)} values, neither 1469 rows nor 866 trade days"
    )
    out[day_row] = pred
    return out, "trade-day order"


def load_tree(
    sh: str, fdates: pd.DatetimeIndex, day_row: np.ndarray
) -> tuple[pd.DataFrame, dict] | None:
    """The 96 tree candidates of one tree model (layout: module docstring), or None until TREES_DONE."""
    md = core_models()
    D = TREE_DIR / sh
    if not TREES_DONE.is_file() or not all(
        (D / f"{m}.npz").is_file() for m in md["model"]
    ):
        return None
    preds, infos = {}, []
    for m in md["model"]:
        preds[m], how = tree_pred(D / f"{m}.npz", fdates, day_row)
        jf = D / f"{m}.json"
        info = json.loads(jf.read_text()) if jf.is_file() else {}
        infos.append(
            dict(
                model=m,
                pick=np.nan,
                alphas="[]",
                r=np.nan,
                n_switch=0,
                cpu=str(info.get("cpu", "")),
                host=str(info.get("host", "")),
                cpu_sec=float(info.get("cpu_total", np.nan)),
                wall_sec=float(info.get("wall", np.nan)),
                numpy="",
                kernel="",
                align=how,
            )
        )
    return md.merge(pd.DataFrame(infos), on="model").assign(est=sh), preds


def load_est(sh: str) -> tuple[pd.DataFrame, dict] | None:
    """The 96 candidates of one linear estimator, or None while any is missing."""
    md = core_models()
    D = FR.models_dir(EST_OF[sh])
    if not all((D / f"{m}.npz").is_file() for m in md["model"]):
        return None
    preds, infos = {}, []
    for m in md["model"]:
        info = json.loads((D / f"{m}.json").read_text())
        preds[m] = np.load(D / f"{m}.npz")["pred"]
        infos.append(
            dict(
                model=m,
                pick=info["pick"],
                alphas=json.dumps(info["alphas"]),
                r=info["r"],
                n_switch=int(sum(x["switched"] for x in info["retune_log"])),
                cpu=info["cpu"],
                host=info["host"],
                cpu_sec=info["cpu_total"],
                wall_sec=info["wall"],
                numpy=info["numpy"],
                kernel=info["kernel"],
            )
        )
    return md.merge(pd.DataFrame(infos), on="model").assign(est=sh), preds


def main() -> None:  # noqa: C901 -- one report
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--budget",
        type=float,
        default=1e9,
        help="seconds after which no new MCS job starts",
    )
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--no-tr", action="store_true", help="skip T_R in new MCS jobs")
    args = ap.parse_args()
    t0 = time.time()
    from close_murphy_splits_headline import elementary

    inp = json.loads((FR.INP / "inputs.json").read_text())
    W = int(inp["W"])
    fdates = pd.DatetimeIndex(pd.to_datetime(inp["dates"][W:]))
    y_adj = np.load(FR.INP / "bar_y.npy")[W:]
    S = CLP.Scorer()
    assert (pd.DatetimeIndex(S.dates) == fdates).all()
    ref = np.load(FR.INP / "ref_armf.npz")
    day_row = ref["day_row"]
    loaded_ = {sh: load_est(sh) for sh in LIN} | {
        sh: load_tree(sh, fdates, day_row) for sh in TREES
    }
    loaded: dict[str, tuple[pd.DataFrame, dict]] = {
        sh: v for sh, v in loaded_.items() if v is not None
    }
    ests = [sh for sh in SHORTS if sh in loaded]
    say(f"estimators with all 96 candidates: {ests}")
    gates: list[dict] = []

    def gate(g: str, item: str, value: str, ok: bool) -> None:
        gates.append(dict(gate=g, item=item, value=value, ok=bool(ok)))

    # ---- gates: scorer conventions (plain, QLIKE, the master table's mid P&L) and the two new losses
    master = pd.read_csv(CLP.MASTER).set_index("key")
    mdq = pd.read_parquet(
        CLP.DAILY, columns=["key", "day", "plain", "qlike_raw", "ret_mid"]
    )
    mdq["day"] = CLP.as_day(mdq["day"])
    for key in ("sub_ols_baseline", "sub_enet_all_features", "sub_enet_baseline"):
        mk = mdq[mdq["key"] == key].set_index("day").reindex(S.days)
        d = S.daily_from_plain(mk["plain"].to_numpy(float))
        st = S.stats(d)
        r_ = master.loc[key]
        dif = max(
            abs(st["qlike"] - r_["qlike_raw"]), abs(st["sharpe_mid"] - r_["Sharpe_mid"])
        )
        gate(
            "scorer: master daily plain -> master_table.csv QLIKE / Sharpe mid",
            key,
            f"{st['qlike']:.4f} / {st['sharpe_mid']:.2f} (max abs diff {dif:.1e})",
            dif < 1e-9,
        )
        dm_ = float(np.max(np.abs(d["mid"] - mk["ret_mid"].to_numpy(float))))
        dq_ = float(np.max(np.abs(d["ql"] - mk["qlike_raw"].to_numpy(float))))
        gate(
            "losses: daily mid P&L / QLIKE == master_table_daily ret_mid / qlike_raw",
            key,
            f"max abs diff {dm_:.1e} / {dq_:.1e}",
            dm_ < 1e-12 and dq_ < 1e-12,
        )
        if key == "sub_ols_baseline":
            es = elementary(d["plain"] / S.iv, S.target / S.iv, 1.0).mean()
            ref_es = float(
                pd.read_csv(
                    REPO / "results" / "close_murphy_splits" / "scores_modelset.csv"
                )
                .set_index("key")
                .loc[key, "ES_at_threshold"]
            )
            gate(
                "ES(1): the elementary score at c = 1 == close_murphy_splits scores_modelset.csv ES_at_threshold",
                key,
                f"{es:.6f} vs {ref_es:.6f}",
                abs(es - ref_es) < 1e-12,
            )
    tg = pd.read_parquet(CLP.EXPORT / "targets.parquet")
    gate(
        "MSE fit: the design's target y (forecast rows) == targets.parquet true_adj",
        "1469 rows",
        f"max abs diff {np.abs(tg['true_adj'].to_numpy(float) - y_adj).max():.1e}",
        np.array_equal(tg["true_adj"].to_numpy(float), y_adj),
    )
    gate(
        "trade days: forecast row of each scorer day",
        f"{len(day_row)} days",
        f"rows {int(day_row.min())} .. {int(day_row.max())}",
        len(day_row) == len(S.days) == 866
        and np.array_equal(
            pd.DatetimeIndex(CLP.as_day(fdates))[day_row], pd.DatetimeIndex(S.days)
        ),
    )

    # ---- per estimator: anchors vs the stored arm f, one CPU class / build, losses
    arms = pd.read_csv(CLP.OUT / "arms.csv").set_index("series_id")
    MDs: list[pd.DataFrame] = []
    cand_names: list[str] = []
    Lall: dict[str, list[np.ndarray]] = {lo: [] for lo in BASE_LOSSES}
    daily_all: dict[tuple[str, str], dict] = {}
    for sh in ests:
        md, preds = loaded[sh]
        if sh in TREES:
            cov = all(np.isfinite(preds[m][day_row]).all() for m in md["model"])
            gate(
                f"trees: {sh} forecasts cover the 866 trade days (finite)",
                f"{len(md)} designs",
                f"aligned by {sorted(set(md['align']))}",
                cov,
            )
            tg_ = (
                TREE_DIR / f"gate_{sh}.csv"
            )  # the tree agent's own anchor gate, when shipped with the forecasts
            if tg_.is_file():
                gt = pd.read_csv(tg_)
                okc = (
                    gt["ok"].astype(str).str.lower().isin(["true", "1", "yes"])
                    if "ok" in gt
                    else None
                )
                gate(
                    f"trees: the tree agent's anchor gate trees/gate_{sh}.csv",
                    f"{len(gt)} rows",
                    "all ok" if okc is not None and okc.all() else "see the file",
                    okc is not None and bool(okc.all()),
                )
            MDs.append(md)
        else:
            gfile, _ = FR.gate_names(EST_OF[sh])
            cg = pd.read_csv(gfile).set_index("model")
        for m in FR.ANCHORS if sh in LIN else ():
            mine, st_ = preds[m], ref[f"{sh}_{m}_pred"]
            dmine, dr_ = S.daily(mine), S.daily(st_)
            rel = float(np.max(np.abs(mine / st_ - 1)))
            npos = int((dmine["q"] != dr_["q"]).sum())
            bit = bool(np.array_equal(mine, st_))
            tol = 1e-8 if sh == "ridge" else FR.TOL_REL
            sid = FR.ref_sid(EST_OF[sh], m)
            gate(
                "anchor (CARC) == stored arm f forecast",
                f"{sh} {m} vs {sid}",
                f"{'bit for bit' if bit else f'max rel {rel:.1e}'}; positions differ on {npos} of 866 days; "
                f"QLIKE {dmine['ql'].mean():.4f} vs {dr_['ql'].mean():.4f}; cluster gate: "
                f"{'ok' if cg.loc[m, 'ok'] else 'FAILED'}",
                (bit or (rel < tol and npos == 0)) and cg.loc[m, "ok"],
            )
            a = arms.loc[sid]
            stt = S.stats(dr_)
            gate(
                "scorer on the stored arm f == the 2026-10-06 arms.csv (QLIKE / Sharpe mid)",
                f"{sh} {m}",
                f"{stt['qlike']:.4f} / {stt['sharpe_mid']:.2f} vs {a['qlike']:.4f} / {a['sharpe_mid']:.2f}",
                abs(stt["qlike"] - a["qlike"]) < 1e-12
                and abs(stt["sharpe_mid"] - a["sharpe_mid"]) < 1e-9,
            )
        for m in md["model"]:
            d = cand_losses(S, preds[m], y_adj, day_row)
            assert np.isfinite(d["plain"]).all() and (d["plain"] > 0).all(), (sh, m)
            daily_all[(sh, m)] = d
            cand_names.append(f"{sh}:{m}")
            for lo in BASE_LOSSES:
                Lall[lo].append(
                    np.asarray(
                        {"qlike": d["ql"], "trade": -d["mid"]}.get(lo, d.get(lo))
                    )
                )
        if sh in LIN:
            MDs.append(md)
    md_all = pd.concat(MDs, ignore_index=True)
    md_lin = md_all[md_all["est"].isin(LIN)]
    cpus = sorted(set(md_lin["cpu"]))
    gate(
        "one CPU class: every linear candidate's CPU",
        f"{len(md_lin)} candidates",
        "; ".join(cpus),
        len(cpus) == 1 and "7513" in cpus[0],
    )
    gate(
        "one build: every linear candidate's kernel / numpy",
        f"{len(md_lin)} candidates",
        f"{sorted(set(md_lin['kernel']))} / numpy {sorted(set(md_lin['numpy']))}",
        md_lin["kernel"].nunique() == 1 and md_lin["numpy"].nunique() == 1,
    )
    md_tr = md_all[md_all["est"].isin(TREES)]
    for sh in [t for t in TREES if t in ests]:
        # CPU classes per tree (a file's "cpu" may join several chunks' CPUs with " | "): LightGBM strictly one
        # class; XGBoost and the forest may mix CARC epyc-7513 and epyc-7542, proven byte-identical for both
        # (results/trees_rfdev_preview163/xcpu_carc/SUMMARY.md: forest 620 / 620 chunks;
        # results/trees_xgbtop_1600/RUN_RECORD.md: every compared XGBoost chunk), coordinator 2026-10-09
        mt = md_tr[md_tr["est"] == sh]
        tc = sorted(
            {c.strip() for v in mt["cpu"] for c in str(v).split("|") if c.strip()}
        )
        if sh == "lgbm":
            ok_cpu = len(tc) == 1
            rule = "one class"
        else:
            ok_cpu = len(tc) >= 1 and all(
                ("EPYC 7513" in c) or ("EPYC 7542" in c) for c in tc
            )
            rule = "{EPYC 7513, EPYC 7542}: byte-identical (xcpu_carc/SUMMARY.md, trees_xgbtop_1600/RUN_RECORD.md)"
        gate(
            f"CPU class: every {sh} candidate's CPU ({rule})",
            f"{int((mt['cpu'] != '').sum())} of {len(mt)} with a CPU record",
            "; ".join(tc) or "not recorded",
            ok_cpu,
        )
    for bf, where in (
        ("bindings.csv", "elastic-net canary node"),
        ("bindings_rl.csv", "ridge / lasso job node"),
    ):
        if not (WORK / bf).is_file():
            continue
        bc = pd.read_csv(WORK / bf)
        kb = bc[
            bc["src"].str.contains("close_exogpen") & bc["sym"].isin(FR.KERNEL_SYMS)
        ]
        blas_like = bc["sym"].str.fullmatch(
            r"[sdcz][a-z0-9]+_|lsame_|xerbla_|ilaenv_|[sd]lamch_"
        )
        n_bad = int((blas_like & ~bc["dst"].str.contains("libnetlib_")).sum())
        gate(
            f"kernel BLAS / LAPACK bindings (LD_DEBUG on the {where})",
            bf,
            f"{len(bc)} bindings; {sorted(set(kb['sym']))} -> {sorted(set(kb['dst'].map(lambda s: Path(s).name)))}; "
            f"BLAS-like symbols bound outside the reference libraries: {n_bad}",
            set(kb["sym"]) == set(FR.KERNEL_SYMS)
            and kb["dst"].str.contains("libnetlib_").all()
            and n_bad == 0,
        )
    Lmat = {lo: np.column_stack(Lall[lo]) for lo in BASE_LOSSES}
    gate(
        "losses finite on the 866 trade days",
        f"{len(cand_names)} candidates x {len(BASE_LOSSES)} losses",
        f"{len(S.days)} days",
        all(np.isfinite(v).all() for v in Lmat.values()) and len(S.days) == 866,
    )

    # ---- the MCS jobs (cached)
    scopes = {sh: [f"{sh}:{m}" for m in loaded[sh][0]["model"]] for sh in ests}
    for jn, fam in JOINTS.items():
        if all(sh in ests for sh in fam):
            scopes[jn] = [c for c in cand_names if c.split(":", 1)[0] in fam]
    CACHE.mkdir(parents=True, exist_ok=True)
    jobs: list[dict] = []
    win_rows: list[dict] = []
    scope_mat: dict[
        tuple[str, str], np.ndarray
    ] = {}  # the scope's loss matrix (winsorized where the loss says)
    for sc, members in scopes.items():
        idx = [cand_names.index(c) for c in members]
        names_sc = members if sc in JOINTS else [c.split(":", 1)[1] for c in members]
        for lo in ALL_LOSSES:
            if lo in WINSOR:
                Lw, q1, q9, sl, sh_ = winsorize(Lmat[WINSOR[lo]][:, idx])
                win_rows.append(
                    dict(
                        scope=sc,
                        loss=lo,
                        q01=q1,
                        q99=q9,
                        share_below=sl,
                        share_above=sh_,
                        n_values=Lw.size,
                    )
                )
                scope_mat[(sc, lo)] = Lw
            else:
                scope_mat[(sc, lo)] = Lmat[lo][:, idx]
            for blk in BLOCKS:
                key = f"{sc}_{lo}_b{blk}"
                if (CACHE / f"{key}.npz").is_file():
                    z = np.load(CACHE / f"{key}.npz")
                    if list(z["names"]) == list(names_sc):
                        continue
                jobs.append(
                    dict(
                        key=key,
                        L=np.ascontiguousarray(scope_mat[(sc, lo)]),
                        names=names_sc,
                        block=blk,
                    )
                )
    pd.DataFrame(win_rows).to_csv(OUT / "winsorize.csv", index=False)
    jobs.sort(key=lambda j: -j["L"].shape[1])  # the joint ones first
    if args.no_tr:  # T_max only for the jobs still to compute (T_R stored as NaN)
        for j in jobs:
            j["no_tr"] = True
    say(
        f"MCS: {len(jobs)} (scope, loss, block) jobs to compute, {args.workers} processes, budget {args.budget:.0f}s"
    )
    if jobs:
        if args.workers <= 1:
            for j in jobs:
                if time.time() - t0 > args.budget:
                    break
                say("  " + mcs_job(j))
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed

            with ProcessPoolExecutor(args.workers) as ex:
                futs: dict = {}
                k = 0
                while k < len(jobs) or futs:
                    while (
                        k < len(jobs)
                        and len(futs) < args.workers
                        and time.time() - t0 < args.budget
                    ):
                        futs[ex.submit(mcs_job, jobs[k])] = jobs[k]["key"]
                        k += 1
                    if not futs:
                        break
                    done = next(as_completed(futs))
                    say("  " + done.result())
                    del futs[done]
        missing = [j["key"] for j in jobs if not (CACHE / f"{j['key']}.npz").is_file()]
        if missing:
            say(
                f"{len(missing)} MCS jobs still to run (rerun to continue): {missing[:6]} ..."
            )
            sys.exit(3)

    # ---- outputs: losses, forecasts
    long, fc = [], []
    for sh in ests:
        for m in loaded[sh][0]["model"]:
            d = daily_all[(sh, m)]
            long.append(
                pd.DataFrame(
                    dict(
                        est=sh,
                        model=m,
                        day=S.days,
                        plain=d["plain"],
                        q=d["q"].astype(np.int8),
                        qlike=d["ql"],
                        mse_fit=d["mse_fit"],
                        es1=d["es1"],
                        trade_loss=-d["mid"],
                        mid_pnl=d["mid"],
                        log_mse=d["log_mse"],
                        hit=d["hit"],
                    )
                )
            )
            fc.append(
                pd.DataFrame(
                    dict(
                        series_id=f"feature_mcs/{sh}_{m}",
                        est=sh,
                        model=m,
                        row=np.arange(len(fdates), dtype=np.int16),
                        date=fdates.astype("datetime64[ns]"),
                        pred_adj=d["pred"],
                    )
                )
            )
    pd.concat(long, ignore_index=True).to_parquet(
        OUT / "losses.parquet", index=False, compression="zstd"
    )
    pd.concat(fc, ignore_index=True).to_parquet(
        OUT / "forecasts.parquet", index=False, compression="zstd"
    )

    # ---- context: mean losses, DM vs the estimator's HAR + calendar / all features
    import atm_straddle_lib as asl

    def lossv(d: dict, lo: str) -> np.ndarray:
        return np.asarray({"qlike": d["ql"], "trade": -d["mid"]}.get(lo, d.get(lo)))

    wcol = {}  # winsorized losses: the estimator scope's matrix, by model
    for sh in ests:
        mods = list(loaded[sh][0]["model"])
        for lo in WINSOR:
            wcol[(sh, lo)] = {m: scope_mat[(sh, lo)][:, k] for k, m in enumerate(mods)}

    def lv(sh: str, m: str, lo: str) -> np.ndarray:
        return wcol[(sh, lo)][m] if lo in WINSOR else lossv(daily_all[(sh, m)], lo)

    rows = []
    for sh in ests:
        base = daily_all[(sh, "baseline")]
        for m in loaded[sh][0]["model"]:
            d = daily_all[(sh, m)]
            st = S.stats(d)
            row = dict(
                est=sh,
                model=m,
                cand=f"{sh}:{m}",
                sharpe_mid=st["sharpe_mid"],
                sharpe_crossed=st["sharpe_crossed"],
                pct_buy=st["pct_buy"],
                n_pos_diff_vs_baseline=int((d["q"] != base["q"]).sum()),
            )
            for lo in ALL_LOSSES:
                v, vb, va = (
                    lv(sh, m, lo),
                    lv(sh, "baseline", lo),
                    lv(sh, "all_features", lo),
                )
                row[lo] = float(v.mean())
                row[f"dpct_{lo}_vs_baseline"] = (
                    float(100 * (v.mean() / vb.mean() - 1))
                    if lo not in ("trade", "trade_w")
                    else np.nan
                )
                row[f"dm_{lo}_vs_baseline"] = (
                    float(dm_test(v, vb)["dm"]) if m != "baseline" else np.nan
                )
                row[f"dm_{lo}_vs_all"] = (
                    float(dm_test(v, va)["dm"]) if m != "all_features" else np.nan
                )
            row["hac_t_mid_vs_baseline"] = (
                float(asl.newey_west_t(d["mid"] - base["mid"])[0])
                if m != "baseline"
                else np.nan
            )
            rows.append(row)
    ctx = md_all.merge(pd.DataFrame(rows), on=["est", "model"])
    big = max(scopes, key=lambda k: len(scopes[k]))
    tail_rows = []
    for lo in ALL_LOSSES:
        Lb = scope_mat[(big, lo)]
        k5 = round(0.05 * Lb.shape[0])
        D = Lb - Lb.mean(axis=1, keepdims=True)
        D2 = np.sort(D**2, axis=0)[::-1]
        dshare = D2[:k5].sum(axis=0) / np.maximum(D2.sum(axis=0), 1e-300)
        nonneg = lo in ("qlike", "mse_fit", "es1", "log_mse", "hit")
        lshare = (
            np.sort(Lb, axis=0)[::-1][:k5].sum(axis=0) / Lb.sum(axis=0)
            if nonneg
            else np.full(Lb.shape[1], np.nan)
        )
        tail_rows.append(
            dict(
                loss=lo,
                scope=big,
                n=Lb.shape[1],
                top_days=k5,
                loss_share_median=float(np.nanmedian(lshare)) if nonneg else np.nan,
                diff_var_share_median=float(np.median(dshare)),
                diff_var_share_p90=float(np.quantile(dshare, 0.9)),
            )
        )
    pd.DataFrame(tail_rows).to_csv(OUT / "tail_concentration.csv", index=False)

    # ---- MCS tables from the cache
    prow, srow, trow = [], [], []
    for sc, members in scopes.items():
        for lo in ALL_LOSSES:
            for blk in BLOCKS:
                z = np.load(CACHE / f"{sc}_{lo}_b{blk}.npz")
                nm = [str(x) for x in z["names"]]
                cands = nm if sc in JOINTS else [f"{sc}:{x}" for x in nm]
                for stat_ in ("Tmax", "TR"):
                    p, order = z[f"{stat_}_p"], list(z[f"{stat_}_order"])
                    for jj, c in enumerate(cands):
                        prow.append(
                            dict(
                                scope=sc,
                                loss=lo,
                                block=blk,
                                stat=stat_,
                                cand=c,
                                p_mcs=float(p[jj]),
                                elim_rank=order.index(jj) + 1,
                            )
                        )
                    trow.append(
                        dict(
                            scope=sc,
                            loss=lo,
                            block=blk,
                            stat=stat_,
                            round_p=json.dumps(
                                [round(float(x), 4) for x in z[f"{stat_}_tests"]]
                            ),
                        )
                    )
                    for al in ALPHAS:
                        if not np.isfinite(p).all():
                            continue  # T_R skipped (--no-tr)
                        keep = [nm[j] for j in range(len(nm)) if p[j] >= al]
                        srow.append(
                            dict(
                                scope=sc,
                                loss=lo,
                                block=blk,
                                stat=stat_,
                                alpha=al,
                                n=len(nm),
                                n_in=len(keep),
                                models=json.dumps(keep),
                            )
                        )
                        if stat_ == "Tmax":
                            mod = [
                                str(x) for x in z["mod10" if al == 0.10 else "mod25"]
                            ]
                            gate(
                                "MCS: module's T_max loop (resample=True) == full-elimination p >= alpha",
                                f"{sc} {lo} block {blk} alpha {al}",
                                f"{len(mod)} vs {len(keep)} models",
                                sorted(mod) == sorted(keep),
                            )
    jl = [cand_names.index(c) for c in scopes[ests[0]]]
    Lg = np.ascontiguousarray(Lmat["qlike"][:, jl])
    Lsg = MCSM._boot_col_means(Lg, B_BOOT, 21, SEED, resample=True)
    rb, ri = mcs_full(Lg, Lsg, "TR"), mcs_tr(Lg, Lsg)
    gate(
        "T_R: the incremental elimination == the brute-force pair scan (p, order, every round)",
        f"{ests[0]} QLIKE block 21 ({len(jl)} candidates)",
        f"max |p diff| {np.abs(rb['p'] - ri['p']).max():.1e}",
        all(np.array_equal(rb[k], ri[k]) for k in ("p", "order", "tests")),
    )
    P = pd.DataFrame(prow)
    P.to_csv(OUT / "mcs_pvalues.csv", index=False)
    sets = pd.DataFrame(srow)
    sets.to_csv(OUT / "mcs_sets.csv", index=False)
    pd.DataFrame(trow).to_csv(OUT / "mcs_rounds.csv", index=False)
    # own-estimator MCS p-values (p_<loss>_<stat>_b<k>) and the joint MCS's (pjoint_...)
    own = P[~P["scope"].isin(list(JOINTS))].pivot_table(
        index="cand", columns=["loss", "stat", "block"], values="p_mcs"
    )
    own.columns = [f"p_{lo}_{st}_b{b}" for lo, st, b in own.columns]
    ctx = ctx.merge(own, left_on="cand", right_index=True, how="left")
    for jn in (
        JOINTS
    ):  # p<scope>_<loss>_<stat>_b<k>: pjoint_ (linear), pjoint_trees_, pgrand_
        if jn in scopes:
            jt = P[P["scope"] == jn].pivot_table(
                index="cand", columns=["loss", "stat", "block"], values="p_mcs"
            )
            jt.columns = [f"p{jn}_{lo}_{st}_b{b}" for lo, st, b in jt.columns]
            ctx = ctx.merge(jt, left_on="cand", right_index=True, how="left")
    ctx.to_csv(OUT / "models_scored.csv", index=False)
    G = pd.DataFrame(gates)
    G.to_csv(OUT / "gates.csv", index=False)
    write_summary(
        ctx,
        G,
        sets,
        md_all,
        scopes,
    )
    say(
        f"report done in {time.time() - t0:.0f}s; gates {int(G['ok'].sum())} / {len(G)} ok"
    )


# ============================================================================ SUMMARY.md
def _f(v, nd=4, sign=False) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ""
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def _cap(t: str) -> str:
    return t[:1].upper() + t[1:]


def _mark(p: float) -> str:
    return f"{p:.3f}" + ("**" if p >= 0.25 else "*" if p >= 0.10 else "")


def set_out(
    sets: pd.DataFrame, scopes: dict, sc: str, lo: str, st: str, al: float
) -> tuple[list[str], list[str], list[int]]:
    """(out at every block, out at some blocks only, set sizes by block + n) of one scope / loss / statistic /
    level; names as the sets store them (the model for an estimator's scope, est:model for the joint)."""
    s = sets[
        (sets["scope"] == sc)
        & (sets["loss"] == lo)
        & (sets["stat"] == st)
        & (sets["alpha"] == al)
    ]
    s = s.sort_values("block")
    universe = set(
        scopes[sc] if sc in JOINTS else [c.split(":", 1)[1] for c in scopes[sc]]
    )
    ins = [set(json.loads(x)) for x in s["models"]]
    outs = [universe - x for x in ins]
    always = set.intersection(*outs)
    return (
        sorted(always),
        sorted(set.union(*outs) - always),
        [len(x) for x in ins] + [len(universe)],
    )


def write_summary(
    ctx: pd.DataFrame,
    G: pd.DataFrame,
    sets: pd.DataFrame,
    md: pd.DataFrame,
    scopes: dict,
) -> None:  # noqa: C901
    L: list[str] = []
    w = L.append
    has_trees = any(sh in scopes for sh in TREES)
    cpu_h = md[md["est"].isin(LIN)]["cpu_sec"].sum() / 3600
    stamps = []
    for nm in ("submitted_feature_mcs.txt", "submitted_feature_mcs_rl.txt"):
        sf = WORK / "cluster_logs" / nm
        if sf.is_file():
            stamps.append(sf.read_text().split("  ")[0].strip())
    w(
        "# Which inputs matter for the 16:00 forecast: a Model Confidence Set over "
        + ("linear and tree input sets" if has_trees else "linear input sets")
    )
    w("")
    w(
        f"Written by `experiments/feature_mcs_report.py` on {time.strftime('%Y-%m-%d %H:%M')}; every number below is "
        "read from the CSVs in this folder."
    )
    w("")
    w("## Design")
    w("")
    w(
        "User (2026-10-08, the professor's suggestion, in place of the paper's lean set): which exogenous inputs matter "
        "for the 16:00 forecast, by a Model Confidence Set (Hansen, Lunde & Nason 2011) instead of feature importance. "
        "Same day: more powerful losses, and the same analysis for ridge and lasso, with one joint MCS over all "
        "linear candidates"
        + (
            "; then the trees (LightGBM, XGBoost, random forest) on the same designs, with a joint MCS over the 288 "
            "trees and a grand joint over all 576 candidates."
            if has_trees
            else "."
        )
    )
    w("")
    w(
        "- **Models**: the paper's pre-tuned per-bar elastic net (l1 ratio 0.5), ridge and lasso: arm (f) of "
        "`results/linear_pretune_2026-10-06/` (`experiments/close_linear_pretune.py`, its functions called "
        "read-only): 16:00-bar design, window 2000 sessions, refit every session through the C port, 250-session "
        "blocks; HAR + calendar backbone penalized at 1/100 of the exogenous penalty (on HAR + calendar alone, one "
        "penalty); alpha pre-tuned on 4 x 250 pre-2020 sessions (the spec grids: ridge 1e-2 .. 1e3, lasso / elastic "
        "net 1e-6 .. 1e-2; mean validation MSE), then the margin-guarded light retune at the four block starts in the "
        "trade period. Every candidate runs the whole procedure on its own design."
    )
    if has_trees:
        tcpu = md[md["est"].isin(TREES)]
        w(
            "- **Trees** (produced by the tree agent, read from `trees/<est>/<model>.npz`, layout in the report "
            "script's docstring): LightGBM `lgbm_opt_00`, XGBoost `xgb_opt_00` and the random forest of trial 255, "
            "fixed settings, refit every session, on the same 96 designs (the runner's design definitions); "
            "CPU classes recorded: "
            + "; ".join(
                f"{t} "
                + (
                    " + ".join(
                        sorted(
                            {
                                c.strip()
                                for v in tcpu[tcpu["est"] == t]["cpu"]
                                for c in str(v).split("|")
                                if c.strip()
                            }
                        )
                    )
                    or "not recorded"
                )
                for t in TREES
            )
            + ". LightGBM is held to one CPU class; XGBoost and the forest may mix CARC epyc-7513 and epyc-7542, "
            "proven byte-identical for both (`results/trees_rfdev_preview163/xcpu_carc/SUMMARY.md`: forest 620 of "
            "620 chunks; `results/trees_xgbtop_1600/RUN_RECORD.md`: every compared XGBoost chunk)."
        )
    w(
        "- **Designs (96 per estimator, 288 in all)**: `baseline` (HAR + calendar, 22 columns) and `all_features` "
        "(628); for each of the 47 exogenous series X of the feature-importance study (the 49 series of "
        "`results/feature_importance_1530_dedup/across_time_refits_series.parquet` minus har_ma and calendar; a "
        "series = every column of one input: its 6 moving averages, plus availability / activity flags where it has "
        "them) `add_X` = HAR + calendar + X and `loo_X` = all features minus X. Designs are order-preserving column "
        "subsets of the all_features 16:00-bar design of record (as every 2026-10-06 bucket is; checked)."
    )
    w(
        "- **Losses** on the 866 trade days 2020-01-03 .. 2024-04-30 (plain forecast f^2 x B, no recalibration; "
        "`close_linear_pretune.Scorer`, the master table's conventions): **QLIKE**; **MSE fit** = squared error of "
        "the adjusted-scale forecast against the design's adjusted-scale target (the feature-importance study's "
        "squared error); **ES(1)** = the elementary score at the trade's threshold, "
        "`close_murphy_splits_headline.elementary` with c = 1 on forecast and target divided by the 15:30 implied "
        "variance (the paper's D.17); **trade** = minus the mid P&L of the sign(s) straddle (the master table's "
        "`ret_mid`)."
    )
    w(
        f"- **MCS**: `src/evaluation/model_confidence_set.py` with `resample=True` (its circular moving-block "
        f"bootstrap restored for this study through that argument; the default, used by every other caller, is "
        f"unchanged), B = {B_BOOT:,}, block lengths {', '.join(map(str, BLOCKS))}, seed {SEED}; T_max (the module's "
        "rule) and T_R (range statistic) on the same replicates; HLN MCS p-values by full elimination (the last "
        "model p = 1); the set at level alpha = the candidates with p >= alpha. Scopes: each estimator's 96 "
        "candidates, the joint 288 linear"
        + (", the joint 288 trees and the grand joint 576" if has_trees else "")
        + ". `*` = in the 90 % set (p >= 0.10), `**` = in the 75 % set (p >= 0.25)."
    )
    w(
        f"- **Compute**: CARC, every candidate on an AMD EPYC 7513 (`--constraint=epyc-7513`, nice 0), one kernel "
        f"build (gcc 13.3 -O3 -march=haswell -ffp-contract=off, the 2026-10-06 flags, against the reference LAPACK / "
        f"BLAS 3.12 of conda-forge's netlib builds, renamed so OpenBLAS cannot stand in); {cpu_h:.1f} CPU-hours of "
        f"model fits in total (jobs: {'; '.join(stamps)}). Files: `forecasts.parquet` (every candidate), "
        "`losses.parquet` (daily losses), `models_scored.csv` (schedules, mean losses, DM, own-estimator MCS "
        "p-values), `mcs_pvalues.csv` (every MCS p-value, every scope), `mcs_sets.csv`, `mcs_rounds.csv` (each "
        "elimination round's test p-value)."
    )
    w("")
    w("## Gates")
    w("")
    w(_md(G.assign(ok=G["ok"].map({True: "yes", False: "NO"}))))
    w("")
    w(f"{int(G['ok'].sum())} of {len(G)} gates pass.")
    w("")
    # ---- set sizes
    w("## Size of the confidence sets")
    w("")
    w("Candidates with MCS p >= alpha, at block lengths 5 / 10 / 21 (of n):")
    w("")
    rows = []
    for sc in scopes:
        for lo in LOSSES:
            r = {
                "scope": EST_LABEL.get(sc, JOINT_LABEL.get(sc, sc)),
                "loss": LOSS_SHORT[lo],
            }
            for st in ("Tmax", "TR"):
                for al in ALPHAS:
                    _, _, sz = set_out(sets, scopes, sc, lo, st, al)
                    r[f"{st} {int(100 * (1 - al))} %"] = (
                        " / ".join(map(str, sz[:-1])) + f" (of {sz[-1]})"
                    )
            rows.append(r)
    w(_md(pd.DataFrame(rows)))
    w("")
    # ---- who is out
    w("## Who is excluded")
    w("")
    w(
        "Per scope and loss, the candidates OUTSIDE the set (T_max; T_R where it differs): excluded at every block "
        "length, and [at some block lengths only]. A loss with nobody excluded separates nothing."
    )
    w("")
    for sc in scopes:
        w(f"### {_cap(EST_LABEL.get(sc, JOINT_LABEL.get(sc, sc)))}")
        w("")
        for lo in LOSSES:
            parts = []
            for st in ("Tmax", "TR"):
                for al in ALPHAS:
                    always, some, sz = set_out(sets, scopes, sc, lo, st, al)
                    n = sz[-1]
                    if len(always) + len(some) == 0:
                        txt = "none"
                    elif len(always) > n / 2:
                        ins = sorted(
                            set(
                                json.loads(
                                    sets[
                                        (sets["scope"] == sc)
                                        & (sets["loss"] == lo)
                                        & (sets["stat"] == st)
                                        & (sets["alpha"] == al)
                                        & (sets["block"] == 10)
                                    ]["models"].iloc[0]
                                )
                            )
                        )
                        txt = (
                            f"{len(always)} at every block; the block-10 set keeps {len(ins)}: "
                            + ", ".join(f"`{m}`" for m in ins[:25])
                            + (" ..." if len(ins) > 25 else "")
                        )
                    else:
                        txt = (
                            ", ".join(f"`{m}`" for m in always)
                            if always
                            else "none at every block"
                        ) + (f" [{', '.join(f'`{m}`' for m in some)}]" if some else "")
                    parts.append(f"{st} {int(100 * (1 - al))} %: {txt}")
            w(f"- **{LOSS_SHORT[lo]}**: " + "; ".join(parts))
        w("")
    # ---- per estimator: per series
    cx = ctx.set_index("cand")
    for sh in [s for s in SHORTS if s in scopes]:
        w(f"## {_cap(EST_LABEL[sh])}: per series")
        w("")
        b, a = cx.loc[f"{sh}:baseline"], cx.loc[f"{sh}:all_features"]

        def al_(row) -> str:
            v = json.loads(row["alphas"])
            return f", alphas {' '.join(f'{x:g}' for x in v)}" if v else ""

        w(
            f"HAR + calendar: QLIKE {b['qlike']:.4f}, MSE fit {b['mse_fit']:.4f}, ES(1) {b['es1']:.4f}, mean mid P&L "
            f"{-b['trade']:+.4f} (Sharpe {b['sharpe_mid']:.2f}){al_(b)}. "
            f"All features: QLIKE {a['qlike']:.4f}, MSE fit {a['mse_fit']:.4f}, ES(1) {a['es1']:.4f}, mean mid P&L "
            f"{-a['trade']:+.4f} (Sharpe {a['sharpe_mid']:.2f}){al_(a)}; "
            f"DM vs HAR + calendar: QLIKE {a['dm_qlike_vs_baseline']:+.2f}, MSE fit {a['dm_mse_fit_vs_baseline']:+.2f}, "
            f"ES(1) {a['dm_es1_vs_baseline']:+.2f}, trade {a['dm_trade_vs_baseline']:+.2f}."
        )
        w("")
        w(
            "Diebold-Mariano t of the add-one model against HAR + calendar and of the leave-one-out model against all "
            "features, per loss (negative: the first has the lower loss); a mark after the value = that candidate is "
            "outside its estimator's T_max set at block 10 (`x` 90 %, `o` 75 % only):"
        )
        w("")
        rws = []
        for x in [
            c.split(":", 1)[1][4:] for c in cx.index if c.startswith(f"{sh}:add_")
        ]:
            ad, lv = cx.loc[f"{sh}:add_{x}"], cx.loc[f"{sh}:loo_{x}"]
            rr: dict[str, object] = {"series": x, "cols": int(ad["n_series_cols"])}
            for lo in LOSSES:

                def mk(row) -> str:
                    p = row[f"p_{lo}_Tmax_b10"]
                    return " x" if p < 0.10 else " o" if p < 0.25 else ""

                rr[f"add {LOSS_SHORT[lo]}"] = _f(
                    ad[f"dm_{lo}_vs_baseline"], 2, True
                ) + mk(ad)
            for lo in LOSSES:

                def mk2(row) -> str:
                    p = row[f"p_{lo}_Tmax_b10"]
                    return " x" if p < 0.10 else " o" if p < 0.25 else ""

                rr[f"loo {LOSS_SHORT[lo]}"] = _f(lv[f"dm_{lo}_vs_all"], 2, True) + mk2(
                    lv
                )
            rr["_k"] = ad["dm_mse_fit_vs_baseline"]
            rws.append(rr)
        w(_md(pd.DataFrame(rws).sort_values("_k").drop(columns="_k")))
        w("")
    # ---- joint scopes: best candidates per loss
    Pall = pd.read_csv(OUT / "mcs_pvalues.csv")
    for jn in [j for j in JOINTS if j in scopes]:
        w(
            f"## {JOINT_LABEL[jn][0].upper() + JOINT_LABEL[jn][1:]}: the best candidates per loss"
        )
        w("")
        Pj = Pall[Pall["scope"] == jn]
        for lo in LOSSES:
            q = Pj[(Pj["loss"] == lo)].pivot_table(
                index="cand", columns=["stat", "block"], values="p_mcs"
            )
            top = ctx.set_index("cand").loc[q.index].sort_values(lo).head(12)
            t = pd.DataFrame(
                {
                    "candidate": top.index,
                    LOSS_SHORT[lo]: top[lo].map(lambda v: _f(v, 5)),
                    "DM vs own HAR+cal": top[f"dm_{lo}_vs_baseline"].map(
                        lambda v: _f(v, 2, True)
                    ),
                    **{
                        f"p {st} {bk}": [_mark(q.loc[c, (st, bk)]) for c in top.index]
                        for st in ("Tmax", "TR")
                        for bk in BLOCKS
                    },
                }
            )
            w(f"**{LOSS_LABEL[lo]}** (the 12 lowest mean losses of {len(scopes[jn])}):")
            w("")
            w(_md(t))
            w("")
    L.extend(sensitivity_section(ctx, sets, scopes))
    L.extend(spa_section())
    L.extend(bucket_section())
    L.extend(reading(ctx, sets, scopes))
    if has_trees:
        L.extend(reading_trees(ctx, sets, scopes))
    (OUT / "SUMMARY.md").write_bytes(("\n".join(L) + "\n").encode())


def reading(ctx: pd.DataFrame, sets: pd.DataFrame, scopes: dict) -> list[str]:
    """The reading, written after the numbers (2026-10-08); every figure is filled in from the tables."""
    rounds = pd.read_csv(OUT / "mcs_rounds.csv")
    cx = ctx[ctx["est"].isin(LIN)].set_index("cand")
    ests = [s for s in LIN if s in scopes]

    def first_p(sc: str, lo: str, st: str) -> str:
        r = rounds[
            (rounds["scope"] == sc) & (rounds["loss"] == lo) & (rounds["stat"] == st)
        ].sort_values("block")
        return " / ".join(f"{json.loads(x)[0]:.2f}" for x in r["round_p"])

    def n_out(sc: str, lo: str, st: str, al: float) -> str:
        _, _, sz = set_out(sets, scopes, sc, lo, st, al)
        return " / ".join(str(sz[-1] - k) for k in sz[:-1])

    def sig(sh: str, lo: str, kind: str) -> list[tuple[str, float]]:
        x = ctx[(ctx["est"] == sh) & (ctx["kind"] == kind)]
        col = f"dm_{lo}_vs_baseline" if kind == "add" else f"dm_{lo}_vs_all"
        x = x[x[col] < -1.96] if kind == "add" else x[x[col] > 1.96]
        x = x.sort_values(col, ascending=kind == "add")
        return [(m[4:], float(v)) for m, v in zip(x["model"], x[col])]

    Ld = pd.read_parquet(OUT / "losses.parquet")
    Ld = Ld[Ld["est"].isin(LIN)]
    b0 = Ld[(Ld["est"] == ests[0]) & (Ld["model"] == "baseline")]
    tail = {}
    for lo in ("qlike", "mse_fit", "es1"):
        v = np.sort(b0[lo].to_numpy())[::-1]
        tail[lo] = (
            100 * v[: round(0.01 * len(v))].sum() / v.sum(),
            100 * v[: round(0.05 * len(v))].sum() / v.sum(),
        )
    span = {}
    for lo in ("qlike", "mse_fit"):
        mm = Ld.groupby(["est", "model"])[lo].mean()
        span[lo] = 100 * (mm.max() / mm.min() - 1)
    nz = Ld.groupby(["est", "model"])["es1"].apply(lambda x: 100 * (x > 0).mean())
    nz_lo, nz_hi = float(nz.min()), float(nz.max())
    has_trees = any(sh in scopes for sh in TREES)
    L = ["## Reading" + (": the linear models" if has_trees else ""), ""]
    if "joint" in scopes:
        L.append(
            "**Does any loss give the MCS power? No.** In the joint MCS over the 288 linear candidates (excluded "
            "under T_max: at 90 % for block lengths 5 / 10 / 21, then at 75 % for the same three): "
            + "; ".join(
                f"{LOSS_SHORT[lo]} {n_out('joint', lo, 'Tmax', 0.10)} / {n_out('joint', lo, 'Tmax', 0.25)}"
                for lo in LOSSES
            )
            + " (T_R: "
            + "; ".join(
                f"{LOSS_SHORT[lo]} {n_out('joint', lo, 'TR', 0.10)} / {n_out('joint', lo, 'TR', 0.25)}"
                for lo in LOSSES
            )
            + "). The first elimination test (the worst candidate against the set) has T_max p = "
            + "; ".join(
                f"{first_p('joint', lo, 'Tmax')} ({LOSS_SHORT[lo]})" for lo in LOSSES
            )
            + ". The joint set therefore cannot pick the estimator or the inputs. Everything it excludes at "
            "75 % at any block length: "
            + "; ".join(
                f"{LOSS_SHORT[lo]} "
                + ", ".join(
                    f"`{m}`"
                    for m in sorted(
                        set().union(
                            *[
                                set(
                                    sum(
                                        set_out(sets, scopes, "joint", lo, st, 0.25)[
                                            :2
                                        ],
                                        [],
                                    )
                                )
                                for st in ("Tmax", "TR")
                            ]
                        )
                    )
                )
                for lo in LOSSES
                if any(
                    sum(set_out(sets, scopes, "joint", lo, st, 0.25)[:2], [])
                    for st in ("Tmax", "TR")
                )
            )
            + ". These are add-one designs built on stock-market trading activity (buy / sell turnover), with "
            "one or two others. Neither new loss does better than QLIKE. Against HAR + calendar (elastic net), "
            f"the top 1 % / 5 % of days carry {tail['mse_fit'][0]:.0f} % / {tail['mse_fit'][1]:.0f} % of the "
            f"fit-space squared error and {tail['qlike'][0]:.0f} % / {tail['qlike'][1]:.0f} % of QLIKE, and the "
            f"means of the 288 candidates span {span['mse_fit']:.1f} % (fit space) against {span['qlike']:.1f} % "
            "(QLIKE). The threshold score is nonzero only on the days the forecast and the realized variance fall "
            f"on opposite sides of the implied variance ({nz_lo:.0f} .. {nz_hi:.0f} % of days, depending on the "
            f"candidate); its top 5 % of days carry {tail['es1'][1]:.0f} % of it. All three losses are dominated "
            "by a few days."
        )
        L.append("")
    per = []
    for sh in ests:
        per.append(
            f"{EST_LABEL[sh]}: "
            + ", ".join(
                f"{LOSS_SHORT[lo]} {n_out(sh, lo, 'Tmax', 0.10)} / {n_out(sh, lo, 'Tmax', 0.25)}"
                for lo in LOSSES
            )
        )
    L.append(
        "**Per estimator** (excluded of 96 under T_max: at 90 % for blocks 5 / 10 / 21, then at 75 % for the same three): "
        + "; ".join(per)
        + ". "
        "T_R compares every pair and sometimes excludes more. Most of its exclusions are near-copies of HAR + "
        "calendar: add-one designs whose penalty keeps the added series near zero, so the pair differs by small "
        "but steady amounts. Rarely are they the same ones at every block length. For example, under ridge the "
        "fit-space T_R excludes `add_fomc_release`, `add_stocktwits_attention` and `add_stocktwits_sentcount` at "
        "90 %, and HAR + calendar itself at 75 %, at every block length. The exclusions that stay put across "
        "block lengths are the T_max ones on the trade loss and on ES(1): mostly add-one designs on trading "
        "activity (turnover, order flow) and, for the elastic net, the VIX level (see 'Who is excluded')."
    )
    L.append("")
    lines = []
    for sh in ests:
        parts = []
        for lo in ("qlike", "mse_fit"):
            s_ = sig(sh, lo, "add")
            if s_:
                parts.append(
                    f"{LOSS_SHORT[lo]} added alone: "
                    + ", ".join(f"`{x}` ({v:+.2f})" for x, v in s_[:7])
                    + (f" and {len(s_) - 7} more" if len(s_) > 7 else "")
                )
            s_ = sig(sh, lo, "loo")
            if s_:
                parts.append(
                    f"{LOSS_SHORT[lo]} removed: "
                    + ", ".join(f"`{x}` ({v:+.2f})" for x, v in s_)
                )
        lines.append(f"{EST_LABEL[sh]}: " + "; ".join(parts))
    L.append(
        "**What the pairwise evidence says (DM against the same estimator's HAR + calendar or all features, "
        "|DM| > 1.96; context only, not the MCS).** "
        + ". ".join(lines)
        + ". Across the three estimators, two series appear every time: ES volume (`sumvolume`) and the "
        "value-weighted stock cross-section's sum of absolute returns (`sumabsret_vwstock`). Both help when added "
        "alone on QLIKE and on the fit-space error, and removing `sumabsret_vwstock` from all features costs QLIKE "
        "under every estimator. The implied-volatility series VVIX and VIX3M help alone on the fit-space error, "
        "and under the lasso on QLIKE as well. On ES(1) and on the trade loss almost nothing is pairwise "
        "significant."
    )
    L.append("")
    if len(ests) == 3:
        best = {lo: cx[lo].idxmin() for lo in LOSSES}
        L.append(
            "**Estimators.** By mean loss alone, the lowest QLIKE and fit-space error belong to lasso leave-one-out "
            f"designs (`{best['qlike']}` {cx.loc[best['qlike'], 'qlike']:.4f}, `{best['mse_fit']}` "
            f"{cx.loc[best['mse_fit'], 'mse_fit']:.5f}), the lowest ES(1) to ridge leave-one-out designs "
            f"(`{best['es1']}` {cx.loc[best['es1'], 'es1']:.4f}) and the best trade to an elastic-net leave-one-out "
            f"design (`{best['trade']}`, Sharpe {cx.loc[best['trade'], 'sharpe_mid']:.2f}). All three are in every "
            "joint set, and so are the three HAR + calendar models (QLIKE "
            + " / ".join(f"{cx.loc[f'{s}:baseline', 'qlike']:.4f}" for s in ests)
            + " for elastic net / ridge / lasso)."
        )
        L.append("")
    if has_trees:
        return L
    L.append(
        "**If trees were run.** The answer would probably not change. The MCS has no power because the losses are "
        "noisy over 866 days, not because the models are linear, and four losses and three estimators all agree "
        "on that. Trees could change the pairwise ranking: they can use implied-volatility and activity series "
        "through thresholds and interactions, and they are invariant to monotone transforms. Leave-one-out stays "
        "a weak probe because split selection moves to correlated substitutes. A tree MCS costs the Optuna "
        f"pre-tune per candidate (orders of magnitude more than these {ctx['cpu_sec'].sum() / 3600:.1f} "
        "CPU-hours), and LightGBM's selections change across CPU classes, so all candidates would have to be "
        "pinned to one class, as here. Power would come from more independent days (other bars or horizons "
        "pooled), not from another loss."
    )
    return L


def reading_trees(ctx: pd.DataFrame, sets: pd.DataFrame, scopes: dict) -> list[str]:
    """The trees and the grand joint: the facts, filled in from the tables (the interpretation is written
    after the numbers, in tree_reading_text())."""
    rounds = pd.read_csv(OUT / "mcs_rounds.csv")
    trees = [s for s in TREES if s in scopes]

    def first_p(sc: str, lo: str, st: str) -> str:
        r = rounds[
            (rounds["scope"] == sc) & (rounds["loss"] == lo) & (rounds["stat"] == st)
        ].sort_values("block")
        return " / ".join(f"{json.loads(x)[0]:.2f}" for x in r["round_p"])

    def n_out(sc: str, lo: str, st: str, al: float) -> str:
        _, _, sz = set_out(sets, scopes, sc, lo, st, al)
        return " / ".join(str(sz[-1] - k) for k in sz[:-1])

    def outs(sc: str, lo: str, al: float) -> list[str]:
        return sorted(
            set().union(
                *[
                    set(sum(set_out(sets, scopes, sc, lo, st, al)[:2], []))
                    for st in ("Tmax", "TR")
                ]
            )
        )

    def sig(sh: str, lo: str, kind: str) -> list[tuple[str, float]]:
        x = ctx[(ctx["est"] == sh) & (ctx["kind"] == kind)]
        col = f"dm_{lo}_vs_baseline" if kind == "add" else f"dm_{lo}_vs_all"
        x = x[x[col] < -1.96] if kind == "add" else x[x[col] > 1.96]
        x = x.sort_values(col, ascending=kind == "add")
        return [(m[4:], float(v)) for m, v in zip(x["model"], x[col])]

    L = ["## Reading: the trees and the grand joint", ""]
    per = [
        f"{EST_LABEL[sh]}: "
        + ", ".join(
            f"{LOSS_SHORT[lo]} {n_out(sh, lo, 'Tmax', 0.10)} / {n_out(sh, lo, 'Tmax', 0.25)}"
            for lo in LOSSES
        )
        for sh in trees
    ]
    L.append(
        "**Per tree** (excluded of 96 under T_max: at 90 % for blocks 5 / 10 / 21, then at 75 % for the same "
        "three): " + "; ".join(per) + "."
    )
    L.append("")
    for jn in ("joint_trees", "grand"):
        if jn not in scopes:
            continue
        ex = {lo: outs(jn, lo, 0.25) for lo in LOSSES}
        L.append(
            f"**{JOINT_LABEL[jn][0].upper() + JOINT_LABEL[jn][1:]}** (excluded under T_max: at 90 % for blocks 5 / 10 "
            "/ 21, then at 75 %): "
            + "; ".join(
                f"{LOSS_SHORT[lo]} {n_out(jn, lo, 'Tmax', 0.10)} / {n_out(jn, lo, 'Tmax', 0.25)}"
                for lo in LOSSES
            )
            + " (T_R: "
            + "; ".join(
                f"{LOSS_SHORT[lo]} {n_out(jn, lo, 'TR', 0.10)} / {n_out(jn, lo, 'TR', 0.25)}"
                for lo in LOSSES
            )
            + "). First elimination test, T_max p: "
            + "; ".join(
                f"{first_p(jn, lo, 'Tmax')} ({LOSS_SHORT[lo]})" for lo in LOSSES
            )
            + ". Excluded at 75 % at any block length (either statistic): "
            + "; ".join(
                f"{LOSS_SHORT[lo]} "
                + (
                    ", ".join(f"`{m}`" for m in ex[lo][:30])
                    + (f" and {len(ex[lo]) - 30} more" if len(ex[lo]) > 30 else "")
                )
                for lo in LOSSES
                if ex[lo]
            )
            + ("." if any(ex.values()) else "none.")
        )
        if jn == "grand":
            fam = []
            for lo in LOSSES:
                r = sets[
                    (sets["scope"] == "grand")
                    & (sets["loss"] == lo)
                    & (sets["stat"] == "Tmax")
                    & (sets["alpha"] == 0.10)
                    & (sets["block"] == 10)
                ]
                kept = json.loads(r["models"].iloc[0])
                fam.append(
                    f"{LOSS_SHORT[lo]} {sum(c.split(':')[0] in LIN for c in kept)} linear + "
                    f"{sum(c.split(':')[0] in TREES for c in kept)} trees"
                )
            L.append("")
            L.append(
                "Composition of the grand 90 % T_max set at block 10: "
                + "; ".join(fam)
                + "."
            )
        L.append("")
    lines = []
    for sh in trees:
        parts = []
        for lo in ("qlike", "mse_fit"):
            s_ = sig(sh, lo, "add")
            if s_:
                parts.append(
                    f"{LOSS_SHORT[lo]} added alone: "
                    + ", ".join(f"`{x}` ({v:+.2f})" for x, v in s_[:7])
                    + (f" and {len(s_) - 7} more" if len(s_) > 7 else "")
                )
            s_ = sig(sh, lo, "loo")
            if s_:
                parts.append(
                    f"{LOSS_SHORT[lo]} removed: "
                    + ", ".join(f"`{x}` ({v:+.2f})" for x, v in s_[:7])
                    + (f" and {len(s_) - 7} more" if len(s_) > 7 else "")
                )
        lines.append(f"{EST_LABEL[sh]}: " + ("; ".join(parts) if parts else "none"))
    L.append(
        "**Pairwise evidence for the trees (DM against the same tree's HAR + calendar or all features, |DM| > "
        "1.96; context only).** " + ". ".join(lines) + "."
    )
    L.append("")
    L.extend(tree_reading_text(ctx, sets, scopes))
    return L


BUCKET_ORDER = (
    "moments",
    "liquidity",
    "market_ew",
    "market_vw",
    "sentiment",
    "implied_vol",
    "vol_demand",
    "fomc",
)
FAM_LABEL = {
    "pool_linear": "pooled linear (ridge, lasso, elastic net)",
    "pool_trees": "pooled trees",
    "pool_all": "pooled, all six estimators",
    "pool_xgb_rf": "pooled trees WITHOUT LightGBM (XGBoost, forest)",
    "pool_5": "pooled, 5 estimators (ridge, lasso, elastic net, XGBoost, forest; LightGBM pending)",
    "pool_4": "pooled, 4 estimators (INTERIM: ridge, lasso, elastic net, LightGBM; XGBoost / forest pending)",
}


def bucket_verdicts() -> pd.DataFrame:
    """The story check per family and loss from buckets_stepm.csv / buckets_pairwise.csv (2026-10-13).
    single bucket helps = StepM (test A, FWER 5 %) names some addb at every block length;
    all features helps = StepM (test A, FWER 5 %) names all_features at every block length;
    removing matters = StepM (test C, FWER 5 %) names some loob at every block length."""
    st = pd.read_csv(OUT / "buckets_stepm.csv")
    rows = []
    for (fam, lo), g in st[st["fwer"] == 0.05].groupby(["family", "loss"], sort=False):

        def always(test: str) -> set:
            q = g[g["test"] == test].sort_values("block")
            return (
                set.intersection(*[set(json.loads(x)) for x in q["rejected"]])
                if len(q)
                else set()
            )

        a = always("A: beat HAR + calendar")
        c = always("C: removing hurts")
        b_help = sorted(x for x in a if x.startswith("addb_"))
        all_help = "all_features" in a
        verdict = (
            "holds"
            if (not b_help and all_help)
            else "fails: a single bucket beats HAR + calendar"
            + (" and so does all features" if all_help else "")
            if b_help
            else "fails: neither a bucket nor all features beats HAR + calendar"
        )
        rows.append(
            dict(
                family=fam,
                loss=lo,
                buckets_beating_har=json.dumps(b_help),
                all_features_beats_har=all_help,
                removal_hurts=json.dumps(sorted(c)),
                verdict=verdict,
            )
        )
    return pd.DataFrame(rows)


def bucket_section() -> list[str]:
    """2026-10-13: the bucket story check (experiments/feature_mcs_buckets.py)."""
    if not (OUT / "buckets_pairwise.csv").is_file():
        return []
    pw = pd.read_csv(OUT / "buckets_pairwise.csv")
    sp = pd.read_csv(OUT / "buckets_spa.csv")
    gg = pd.read_csv(OUT / "buckets_gates.csv")
    vd = bucket_verdicts()
    vd.to_csv(OUT / "buckets_verdicts.csv", index=False)
    fams = list(dict.fromkeys(pw["family"]))
    lab = {**{k: v.split(" (")[0] for k, v in EST_LABEL.items()}, **FAM_LABEL}
    L = [
        '## The bucket story check (2026-10-13): "no single bucket helps, all features does"',
        "",
    ]
    L.append(
        "`experiments/feature_mcs_buckets.py`. Designs: `addb_<B>` = HAR + calendar + every series of bucket B, "
        "`loob_<B>` = all features minus every series of bucket B, for the 8 buckets of `src/data/loading.py` "
        "SUBGROUPS that partition the 47 series (moments 7, liquidity 14, market_ew 6, market_vw 6, sentiment 3, "
        "implied_vol 3, vol_demand 4, fomc 4), each run through the same arm-f procedure per estimator (linear: "
        "CARC EPYC 7513 with the reference-BLAS kernel, from a fresh root after a /scratch1 storage fault; the "
        "job re-ran the six linear anchors and gated them against the stored arm f). Families: each estimator, "
        "and pools that average each design's daily loss (and daily mid P&L) over the estimators. Pairwise: DM on "
        "QLIKE, winsorized QLIKE (the MCS rule over the family's 112 designs) and the fit-space MSE; sign(s) "
        "Sharpe mid / crossed and the Newey-West t of the daily mid P&L difference. Joint: SPA / White RC / StepM "
        "(the SPA section's code and bootstrap, B = 10,000, blocks 5 / 10 / 21) on QLIKE, winsorized QLIKE and "
        "the trade loss: **A** benchmark HAR + calendar vs {8 addb + all features}; **B** benchmark all features "
        "vs the 8 loob (does removing a bucket help?); **C** B reversed (does removing a bucket hurt?). Trees "
        "(2026-10-13): the tree agent's bucket designs (flag `trees/BUCKETS_DONE`) -- LightGBM on CARC EPYC "
        "7513, XGBoost and the forest on Hoffman2 E5-2670 / E5-2650 v2, CPU classes proven byte-identical to "
        "EPYC 7513 for both "
        "(`results/trees_rfdev_preview163/xcpu_carc/SUMMARY.md`, `results/h2check_rfdev/compare_by_class.csv`). "
        "`pool_5` = ridge, lasso, elastic net, XGBoost, forest (the version specified while LightGBM was "
        "pending), `pool_all` = all six."
    )
    L.append("")
    L.append(_md(gg.assign(ok=gg["ok"].map({True: "yes", False: "NO"}))))
    L.append("")
    L.append(
        "**Story check** (StepM at FWER 5 %, at every block length; 'holds' = no bucket alone beats HAR + "
        "calendar while all features does):"
    )
    L.append("")
    show = vd.copy()
    show["family"] = show["family"].map(lambda f: lab.get(f, f))
    show["loss"] = show["loss"].map(lambda x: LOSS_SHORT.get(x, x))
    show["buckets_beating_har"] = show["buckets_beating_har"].map(
        lambda x: ", ".join(f"`{m}`" for m in json.loads(x)) or "none"
    )
    show["removal_hurts"] = show["removal_hurts"].map(
        lambda x: ", ".join(f"`{m}`" for m in json.loads(x)) or "none"
    )
    show["all_features_beats_har"] = show["all_features_beats_har"].map(
        {True: "yes", False: "no"}
    )
    L.append(
        _md(
            show.rename(
                columns={
                    "buckets_beating_har": "buckets that beat HAR + cal.",
                    "all_features_beats_har": "all features beats HAR + cal.",
                    "removal_hurts": "removing hurts (test C)",
                }
            )
        )
    )
    L.append("")
    L.append("**SPA consistent p** (blocks 5 / 10 / 21) per family, loss and test:")
    L.append("")
    rows = []
    for fam in fams:
        for lo in ("qlike", "qlike_w", "trade"):
            r = {"family": lab.get(fam, fam), "loss": LOSS_SHORT.get(lo, lo)}
            for t in (
                "A: beat HAR + calendar",
                "B: removing helps",
                "C: removing hurts",
            ):
                q = sp[
                    (sp["family"] == fam) & (sp["loss"] == lo) & (sp["test"] == t)
                ].sort_values("block")
                r[t] = " / ".join(f"{x:.3f}" for x in q["p_consistent"])
            rows.append(r)
    L.append(_md(pd.DataFrame(rows)))
    L.append("")
    for fam in fams:
        q = pw[pw["family"] == fam]
        L.append(
            f"**{lab.get(fam, fam)}** -- pairwise (DM < 0 / HAC t > 0: the design is better than its reference):"
        )
        L.append("")
        rows = []
        for _, r in q.iterrows():
            rows.append(
                {
                    "design": f"`{r['design']}`",
                    "vs": "HAR + cal."
                    if r["reference"] == "baseline"
                    else "all features",
                    "dQLIKE %": f"{r['dpct_qlike']:+.2f}",
                    "DM QLIKE": f"{r['dm_qlike']:+.2f}",
                    "DM QLIKE (w)": f"{r['dm_qlike_w']:+.2f}",
                    "DM MSE fit": f"{r['dm_mse_fit']:+.2f}",
                    "Sharpe mid (ref)": f"{r['sharpe_mid']:.2f} ({r['sharpe_mid_ref']:.2f})",
                    "Sharpe crossed (ref)": f"{r['sharpe_crossed']:.2f} ({r['sharpe_crossed_ref']:.2f})",
                    "HAC t mid": f"{r['hac_t_mid']:+.2f}",
                }
            )
        L.append(_md(pd.DataFrame(rows)))
        L.append("")
    return L


def spa_section() -> list[str]:
    """2026-10-12: SPA, Reality Check and StepM against HAR + calendar (experiments/feature_mcs_spa.py)."""
    if not (OUT / "spa_rc.csv").is_file():
        return []
    sp = pd.read_csv(OUT / "spa_rc.csv")
    st = pd.read_csv(OUT / "stepm.csv")
    gg = pd.read_csv(OUT / "spa_gates.csv")
    order_sc = [x for x in (*SHORTS, "grand") if x in set(sp["scope"])]
    lab = {
        **{k: v.split(" (")[0] for k, v in EST_LABEL.items()},
        "grand": "grand (vs the OLS)",
    }
    L = ["## SPA, Reality Check and StepM against HAR + calendar (2026-10-12)", ""]
    L.append(
        "`experiments/feature_mcs_spa.py`. H0: no alternative has a lower expected loss than the benchmark; "
        "d = loss of the benchmark - loss of the alternative. **Per estimator**: benchmark = that estimator's "
        "HAR + calendar (`baseline`), alternatives = its other 95 designs. **Grand**: benchmark = the paper's HAR "
        "+ calendar OLS (master table `sub_ols_baseline`; its adjusted-scale forecast = the 2026-10-03 export's "
        "`exog_penalty/ols_baseline`, plain forecast equal to the master table's on the 866 days), alternatives "
        "= all 576. The eight losses of this report (winsorized losses: the MCS scope's winsorization; the OLS "
        "is clipped with the grand scope's daily mean and clip points). **SPA**: Hansen (2005), studentized, "
        "p consistent / lower / upper (`close_kfull_testlib.spa_rc`, read-only); **RC**: White's (2000) Reality "
        "Check (not studentized); **StepM**: Romano & Wolf (2005), studentized, FWER 5 % and 10 %. Bootstrap: "
        "the circular block bootstrap of the MCS module's `resample=True` path written out as resampling counts, "
        f"B = {B_BOOT:,}, blocks {', '.join(map(str, BLOCKS))}, seed {SEED}. Alternatives identical to the "
        "benchmark on a loss (d = 0 every day) are dropped from that test (`n_identical_dropped` in spa_rc.csv). "
        "Files: `spa_rc.csv`, `stepm.csv`, `spa_tstats.csv` (every alternative's t), `spa_gates.csv`."
    )
    L.append("")
    L.append(_md(gg.assign(ok=gg["ok"].map({True: "yes", False: "NO"}))))
    L.append("")
    L.append(
        "**SPA consistent p** (blocks 5 / 10 / 21; `*` < 0.10, `**` < 0.05 at every block length):"
    )
    L.append("")
    rows = []
    for sc in order_sc:
        r = {"family": lab[sc]}
        for lo in ALL_LOSSES:
            q = sp[(sp["scope"] == sc) & (sp["loss"] == lo)].sort_values("block")
            ps = q["p_consistent"].tolist()
            mk = "**" if max(ps) < 0.05 else "*" if max(ps) < 0.10 else ""
            r[LOSS_SHORT[lo]] = " / ".join(f"{x:.3f}" for x in ps) + mk
        rows.append(r)
    L.append(_md(pd.DataFrame(rows)))
    L.append("")
    L.append(
        "**SPA lower / upper and the Reality Check** (block 10: p lower / p upper / RC p; the best "
        "alternative and its t):"
    )
    L.append("")
    rows = []
    for sc in order_sc:
        r = {"family": lab[sc]}
        for lo in ALL_LOSSES:
            q = sp[(sp["scope"] == sc) & (sp["loss"] == lo) & (sp["block"] == 10)].iloc[
                0
            ]
            bn = q["best"] if sc == "grand" else q["best"].split(":", 1)[1]
            r[LOSS_SHORT[lo]] = (
                f"{q['p_lower']:.3f} / {q['p_upper']:.3f} / {q['p_rc']:.3f}; `{bn}` {q['best_t']:.2f}"
            )
        rows.append(r)
    L.append(_md(pd.DataFrame(rows)))
    L.append("")
    L.append(
        "**StepM**: the designs that beat HAR + calendar at FWER 5 % (at every block length, then [at some "
        "only]); designs added at 10 % in the last column:"
    )
    L.append("")

    def nm(x: str, sc: str) -> str:
        return f"`{x if sc == 'grand' else x.split(':', 1)[1]}`"

    rows = []
    for sc in order_sc:
        for lo in ALL_LOSSES:
            q5 = st[(st["scope"] == sc) & (st["loss"] == lo) & (st["fwer"] == 0.05)]
            q10 = st[(st["scope"] == sc) & (st["loss"] == lo) & (st["fwer"] == 0.10)]
            s5 = [set(json.loads(x)) for x in q5.sort_values("block")["rejected"]]
            s10 = set().union(*[set(json.loads(x)) for x in q10["rejected"]])
            if not s10:
                continue
            always, some = set.intersection(*s5), set.union(*s5) - set.intersection(*s5)
            extra = s10 - set.union(*s5)
            rows.append(
                {
                    "family": lab[sc],
                    "loss": LOSS_SHORT[lo],
                    "beat HAR + calendar at 5 %": (
                        ", ".join(nm(x, sc) for x in sorted(always)) or "none"
                    )
                    + (
                        f" [{', '.join(nm(x, sc) for x in sorted(some))}]"
                        if some
                        else ""
                    ),
                    "added at 10 %": ", ".join(nm(x, sc) for x in sorted(extra)),
                }
            )
    L.append(_md(pd.DataFrame(rows)) if rows else "No rejection at 10 % anywhere.")
    L.append("")
    b10 = sp[sp["block"] == 10]
    rej = {
        lo: [
            sc
            for sc in order_sc
            if (
                sp[(sp["scope"] == sc) & (sp["loss"] == lo)]["p_consistent"] < 0.05
            ).all()
        ]
        for lo in ALL_LOSSES
    }
    names5: dict[str, int] = {}
    for _, r in st[st["fwer"] == 0.05].iterrows():
        for x in json.loads(r["rejected"]):
            k = x.split(":", 1)[1]
            names5[k] = names5.get(k, 0) + 1
    top = sorted(names5.items(), key=lambda kv: -kv[1])
    loo_or_all = [k for k in names5 if not k.startswith("add_")]
    L.append(
        "**Reading.** Unlike the MCS, SPA does reject HAR + calendar. The SPA consistent p is below 5 % at every "
        "block length for: "
        + "; ".join(
            f"{LOSS_SHORT[lo]} -- {', '.join(lab[s_] for s_ in rej[lo])}"
            for lo in ALL_LOSSES
            if rej[lo]
        )
        + ". Nowhere for "
        + ", ".join(LOSS_SHORT[lo] for lo in ALL_LOSSES if not rej[lo])
        + ". White's Reality Check, which is not studentized, rarely rejects (block 10: RC p < 0.05 in "
        + f"{int((b10['p_rc'] < 0.05).sum())} of {len(b10)} tests), because one noisy alternative dominates its "
        + "maximum. StepM names few designs"
        + (
            ", all of them add-one designs at 5 %"
            if not loo_or_all
            else ", nearly all of them add-one designs"
        )
        + ". Counted over every "
        + "family, loss and block at FWER 5 %: "
        + ", ".join(f"`{k}` {v}" for k, v in top[:10])
        + (
            f"; non-add-one designs named: {', '.join(f'`{k}`' for k in loo_or_all)}"
            if loo_or_all
            else "; never all features or a leave-one-out design"
        )
        + ". The losses that measure the variance forecast (QLIKE, winsorized QLIKE, fit-space MSE, log-MSE) "
        "agree with the pairwise evidence: under the linear models, ES volume and the stock cross-section's "
        "absolute returns, alone on top of HAR + calendar, beat HAR + calendar after the multiple-testing "
        "correction; under the lasso VVIX and VIX3M join them. The trees reject mainly on the log-MSE "
        "(LightGBM, XGBoost: ES volume, `sumautocov`), and XGBoost also on the trade, hit and winsorized "
        "trade losses (`sumabsret_ewstock`, `sumpret2`, the VIX level). The decision-based losses do not "
        "reject for the linear models. The MCS and SPA answer different questions. The MCS asks which "
        "candidates are indistinguishable from the best, and it eliminates from the bottom. SPA and StepM ask "
        "whether any candidate beats a fixed benchmark, one-sided and studentized. Against HAR + calendar that "
        "test has power, and it names the same few inputs throughout."
    )
    L.append("")
    return L


def sensitivity_section(
    ctx: pd.DataFrame, sets: pd.DataFrame, scopes: dict
) -> list[str]:
    """2026-10-12: the four sensitivity losses next to the first four (sets, first-elimination p, tails, who
    is excluded: HAR + calendar, the favored add-ones, the others)."""
    rounds = pd.read_csv(OUT / "mcs_rounds.csv")
    tails = pd.read_csv(OUT / "tail_concentration.csv").set_index("loss")
    win = pd.read_csv(OUT / "winsorize.csv")
    order_sc = [s_ for s_ in ("grand", "joint", "joint_trees", *SHORTS) if s_ in scopes]
    L = ["## Sensitivity losses (2026-10-12): does any loss give the MCS power?", ""]
    L.append(
        "Four more daily losses on the same 576 forecasts (definitions in the report script's docstring): "
        "**log-MSE** = (log(f^2 B) - log RV)^2, a *sensitivity* loss -- not robust to noise in the realized-"
        "variance proxy (Patton 2011), so it can rank forecasts differently from the true conditional variance; "
        "**hit** = 1{sign(f^2 B - IV) != sign(RV - IV)} with the 15:30 implied variance, the trade's direction "
        "without P&L weighting; **QLIKE (w)** and **trade (w)** = QLIKE and the trade loss with each "
        "candidate's daily differential against the scope's cross-candidate mean clipped at the 1 % / 99 % "
        "points of that scope's pooled differentials (fixed once per scope before the MCS; the daily mean is "
        "added back, which the MCS ignores). Clip points (grand scope): "
        + "; ".join(
            f"{LOSS_SHORT[r['loss']]} [{r['q01']:+.4g}, {r['q99']:+.4g}], {100 * (r['share_below'] + r['share_above']):.1f} % "
            "of values clipped"
            for _, r in win[win["scope"] == order_sc[0]].iterrows()
        )
        + ". MCS: T_max only for these four losses (T_R was not run for lack of time; their T_R columns in "
        "mcs_pvalues.csv are empty); B = 10,000, blocks 5 / 10 / 21, alpha 0.10 / 0.25."
    )
    L.append("")
    L.append(
        f"**Tail concentration** ({order_sc[0]} scope, {int(tails['n'].iloc[0])} candidates; medians across "
        "candidates): share of a candidate's total loss on its worst 5 % of days, and share of its squared "
        "daily differential against the cross-candidate mean on its top 5 % of days."
    )
    L.append("")
    L.append(
        _md(
            pd.DataFrame(
                [
                    {
                        "loss": LOSS_SHORT[lo],
                        "loss on top 5 % of days": ""
                        if not np.isfinite(tails.loc[lo, "loss_share_median"])
                        else f"{100 * tails.loc[lo, 'loss_share_median']:.0f} %",
                        "differential variance on top 5 % of days": f"{100 * tails.loc[lo, 'diff_var_share_median']:.0f} % "
                        f"(90th pct {100 * tails.loc[lo, 'diff_var_share_p90']:.0f} %)",
                    }
                    for lo in ALL_LOSSES
                ]
            )
        )
    )
    L.append("")
    L.append(
        "**First elimination test** (T_max p of the worst candidate against the whole set, blocks 5 / 10 / 21; "
        "below 0.10 means the 90 % set drops at least one candidate):"
    )
    L.append("")
    rows = []
    for sc in order_sc:
        r = {"scope": EST_LABEL.get(sc, JOINT_LABEL.get(sc, sc))}
        for lo in ALL_LOSSES:
            q = rounds[
                (rounds["scope"] == sc)
                & (rounds["loss"] == lo)
                & (rounds["stat"] == "Tmax")
            ].sort_values("block")
            r[LOSS_SHORT[lo]] = " / ".join(
                f"{json.loads(x)[0]:.2f}" for x in q["round_p"]
            )
        rows.append(r)
    L.append(_md(pd.DataFrame(rows)))
    L.append("")
    L.append(
        "**Who is excluded** (T_max; the union over block lengths 5 / 10 / 21; HAR + calendar = the scope's "
        "`baseline` model(s); favored add-ones = "
        + ", ".join(f"`{m}`" for m in FAVORED)
        + "):"
    )
    L.append("")
    rows = []
    for sc in order_sc:
        for lo in ALL_LOSSES:
            r = {
                "scope": EST_LABEL.get(sc, JOINT_LABEL.get(sc, sc)),
                "loss": LOSS_SHORT[lo],
            }
            for al in ALPHAS:
                a_, s_, sz = set_out(sets, scopes, sc, lo, "Tmax", al)
                outm = a_ + s_
                base_ = [m for m in outm if m.split(":")[-1] == "baseline"]
                fav = [m for m in outm if m.split(":")[-1] in FAVORED]
                add_o = [
                    m
                    for m in outm
                    if m.split(":")[-1].startswith("add_")
                    and m.split(":")[-1] not in FAVORED
                ]
                loo_o = [m for m in outm if m.split(":")[-1].startswith("loo_")]
                pc = f"{int(100 * (1 - al))} %"
                r[f"out {pc} (b5/b10/b21)"] = " / ".join(
                    str(sz[-1] - k) for k in sz[:-1]
                )
                r[f"HAR+cal out {pc}"] = ", ".join(base_) if base_ else "no"
                r[f"favored add-ones out {pc}"] = ", ".join(fav) if fav else "none"
                r[f"other add-ones / loo out {pc}"] = f"{len(add_o)} / {len(loo_o)}"
            if lo in NEW_LOSSES or sc in ("grand", "joint", "joint_trees"):
                rows.append(r)
    L.append(_md(pd.DataFrame(rows)))
    L.append("")
    L.append(
        "**The favored add-ones against their own estimator's HAR + calendar** (Diebold-Mariano, all eight "
        "losses; negative = the add-one model has the lower loss; context only):"
    )
    L.append("")
    cxs = ctx.set_index("cand")
    rws = []
    for sh in [s_ for s_ in SHORTS if s_ in scopes]:
        for m in FAVORED:
            c_ = f"{sh}:{m}"
            if c_ in cxs.index:
                rws.append(
                    {
                        "candidate": f"`{c_}`",
                        **{
                            LOSS_SHORT[lo]: _f(
                                cxs.loc[c_, f"dm_{lo}_vs_baseline"], 2, True
                            )
                            for lo in ALL_LOSSES
                        },
                    }
                )
    L.append(_md(pd.DataFrame(rws)))
    L.append("")
    L.extend(sensitivity_reading(ctx, sets, scopes, rounds, order_sc))
    return L


def sensitivity_reading(
    ctx: pd.DataFrame,
    sets: pd.DataFrame,
    scopes: dict,
    rounds: pd.DataFrame,
    order_sc: list,
) -> list[str]:
    """The answer to 'does any loss give the MCS power?' (2026-10-12), from the tables."""
    tails = pd.read_csv(OUT / "tail_concentration.csv").set_index("loss")

    def fp(sc: str, lo: str) -> list[float]:
        q = rounds[
            (rounds["scope"] == sc)
            & (rounds["loss"] == lo)
            & (rounds["stat"] == "Tmax")
        ].sort_values("block")
        return [json.loads(x)[0] for x in q["round_p"]]

    def outs(sc: str, lo: str, al: float) -> list[str]:
        a_, s_, _ = set_out(sets, scopes, sc, lo, "Tmax", al)
        return a_ + s_

    big = order_sc[0]
    hc, fav, oth, sizes = [], [], {}, {}
    for sc in order_sc:
        for lo in ALL_LOSSES:
            o = outs(sc, lo, 0.25)
            o90 = outs(sc, lo, 0.10)
            sizes[(sc, lo)] = (len(o90), len(o))
            hc += [
                f"{m if sc in JOINTS else f'{sc}:{m}'} ({LOSS_SHORT[lo]}, {sc})"
                for m in o
                if m.split(":")[-1] == "baseline"
            ]
            fav += [
                f"{m if sc in JOINTS else f'{sc}:{m}'} ({LOSS_SHORT[lo]}, {sc})"
                for m in o
                if m.split(":")[-1] in FAVORED
            ]
            oth[(sc, lo)] = sum(
                m.split(":")[-1].startswith("add_") and m.split(":")[-1] not in FAVORED
                for m in o
            )
    n_pow = {lo: sum(min(fp(sc, lo)) < 0.10 for sc in order_sc) for lo in ALL_LOSSES}
    g_out = {lo: outs(big, lo, 0.25) for lo in ALL_LOSSES}
    loo_out = sorted(
        {
            m
            for sc in order_sc
            for lo in ALL_LOSSES
            for m in outs(sc, lo, 0.25)
            if m.split(":")[-1].startswith("loo_")
        }
    )
    sharp = [lo for lo in NEW_LOSSES if max(fp(big, lo)) < 0.05]
    L = []
    L.append(
        "**Answer: some extra power, but not the separation that was asked for.** "
        + (
            "In the grand MCS ("
            + str(len(scopes[big]))
            + " candidates) "
            + ", ".join(LOSS_SHORT[lo] for lo in sharp)
            + " reject the first elimination test at 5 % at every block length (T_max p: "
            + "; ".join(
                f"{' / '.join(f'{x:.2f}' for x in fp(big, lo))} {LOSS_SHORT[lo]}"
                for lo in sharp
            )
            + "), where QLIKE, ES(1) and the trade loss do not ("
            + "; ".join(
                f"{' / '.join(f'{x:.2f}' for x in fp(big, lo))} {LOSS_SHORT[lo]}"
                for lo in ("qlike", "es1", "trade")
            )
            + "), and neither does the hit loss ("
            + " / ".join(f"{x:.2f}" for x in fp(big, "hit"))
            + "). "
            if sharp
            else ""
        )
        + "What they remove is a handful of single add-one designs. Out of the grand 75 % set at any block "
        "length: "
        + "; ".join(
            f"{LOSS_SHORT[lo]} {len(g_out[lo])}"
            + (
                f" ({', '.join(f'`{m}`' for m in g_out[lo])})"
                if g_out[lo] and len(g_out[lo]) <= 9
                else ""
            )
            for lo in NEW_LOSSES
        )
        + ". These are mostly order flow (`ofi_*`), third moments (`sumret3*`), turnover and attention series, "
        "and mostly trees. Over all scopes and all eight losses, a first test with p < 0.10 at some block length "
        "occurs in "
        + ", ".join(
            f"{n_pow[lo]} of {len(order_sc)} scopes on {LOSS_SHORT[lo]}"
            for lo in ALL_LOSSES
        )
        + ". Leave-one-out designs excluded at 75 % anywhere: "
        + (", ".join(f"`{m}`" for m in loo_out) if loo_out else "none")
        + "."
    )
    L.append("")
    L.append(
        "**HAR + calendar** is excluded at 75 % (T_max, any block length, any scope, any of the eight losses) "
        "only as: "
        + (", ".join(hc) if hc else "never")
        + ". That is LightGBM's HAR + calendar on the fit-space error at block 21 only, where the trees as a "
        "family fit worst; it is not evidence that the inputs matter. **The "
        "favored add-ones** (ES volume and the cross-sectional absolute returns) survive everywhere except: "
        + (", ".join(fav) if fav else "nowhere")
        + ". Apart from the elastic net's trade losses, the favored designs stay in the set while some "
        "single-series designs drop. But the MCS does not put them ahead of HAR + calendar or of the "
        "all-features designs: those stay in every set (bar the LightGBM case above). This is not because HAR "
        "+ calendar looks good. By mean QLIKE (w) it ranks "
        + ", ".join(
            f"{int((g.sort_values('qlike_w')['model'].tolist()).index('baseline')) + 1}"
            for _, g in ctx.groupby("est")
        )
        + " of 96 within the six estimators, and the favored add-ones beat it pairwise on QLIKE (w) and "
        "log-MSE (table above). But the MCS eliminates from the bottom and stops after a handful of "
        "candidates, well before it reaches HAR + calendar."
    )
    L.append("")
    L.append(
        "**Why the gain is small.** Clipping the differentials at 1 % / 99 % lowers the share of the "
        f"differential variance on the top 5 % of days (QLIKE {100 * tails.loc['qlike', 'diff_var_share_median']:.0f} % "
        f"-> {100 * tails.loc['qlike_w', 'diff_var_share_median']:.0f} %, trade "
        f"{100 * tails.loc['trade', 'diff_var_share_median']:.0f} % -> {100 * tails.loc['trade_w', 'diff_var_share_median']:.0f} %), "
        "and that is what moves the first test below 5 %. But only the worst single-series designs stand out "
        "from the noise that is left. "
        "The hit loss is the trade's direction alone. Two candidates' hit losses differ only on the days their "
        "positions differ: against each estimator's HAR + calendar that is a median of "
        f"{int(ctx.groupby('est')['n_pos_diff_vs_baseline'].median().min())} to "
        f"{int(ctx.groupby('est')['n_pos_diff_vs_baseline'].median().max())} of 866 days, depending on the "
        "estimator. The log-MSE "
        f"(differential variance on the top 5 % of days: {100 * tails.loc['log_mse', 'diff_var_share_median']:.0f} %) "
        "weighs relative misses evenly, but it inherits the proxy noise of a single day's realized variance and "
        "is not a robust loss (Patton 2011)."
    )
    L.append("")
    return L


def tree_reading_text(ctx: pd.DataFrame, sets: pd.DataFrame, scopes: dict) -> list[str]:
    """The interpretation of the tree results (written 2026-10-09 once the numbers were in; figures from the tables)."""
    if not all(s in scopes for s in SHORTS) or "grand" not in scopes:
        return []
    med = ctx.groupby("est")[["qlike", "mse_fit", "es1"]].median()
    mn = ctx.groupby("est")[["qlike", "mse_fit"]].min()
    af = ctx[ctx["model"] == "all_features"].set_index("est")
    best_t = ctx[ctx["est"].isin(TREES)].sort_values("trade").iloc[0]
    _, _, sz = set_out(sets, scopes, "grand", "mse_fit", "Tmax", 0.25)
    out25 = sorted(
        set().union(
            *[
                set(sum(set_out(sets, scopes, "grand", "mse_fit", "Tmax", al)[:2], []))
                for al in (0.25,)
            ]
        )
    )
    n_lgbm = sum(m.startswith("lgbm:") for m in out25)
    return [
        "**Interpretation.** The trees do not change the answer about inputs. Under each tree, and in the joint "
        "MCS over the 288 trees, at most a handful of designs leave the 75 % set on any loss, and ES(1) "
        "excludes nothing under T_max (T_R drops two or three XGBoost add-ones at 75 %). The few exclusions that hold at every block length are single add-one designs; "
        "the steadiest is `add_ofi_ewstock` (equal-weighted order flow), out of the LightGBM and XGBoost sets on "
        "QLIKE. The pairwise evidence names the same inputs as the linear models: ES volume (`sumvolume`) and the "
        "stock cross-section's absolute returns (`sumabsret_ewstock` for the boosted trees), which help when added "
        "alone; removing ES volume costs the forest on the fit-space error.",
        "",
        "**The grand joint has power on one loss only, and it separates model families, not inputs.** On the "
        f"fit-space squared error T_max excludes {sz[-1] - sz[1]} of 576 at 75 % (block 10; "
        f"{' / '.join(str(sz[-1] - k) for k in sz[:-1])} at blocks 5 / 10 / 21). Over the three block lengths "
        f"{len(out25)} distinct candidates leave the 75 % set: {n_lgbm} LightGBM candidates, all add-one designs "
        "except LightGBM's own HAR + calendar (block 21 only), plus "
        + ", ".join(f"`{m}`" for m in out25 if not m.startswith("lgbm:"))
        + ". At 90 % it "
        "excludes 3 to 5, all LightGBM add-one designs. The trees fit the adjusted-scale target worse than the "
        "linear models across the board. Median fit-space MSE: "
        + ", ".join(
            f"{EST_LABEL[s].split(' (')[0]} {med.loc[s, 'mse_fit']:.4f}" for s in SHORTS
        )
        + ". Every tree's all-features model is above every linear one ("
        + ", ".join(f"{s} {af.loc[s, 'mse_fit']:.4f}" for s in SHORTS)
        + "). LightGBM on HAR + calendar plus one series is the weakest group. On QLIKE the families "
        "overlap: the best tree candidates ("
        + ", ".join(f"{s} {mn.loc[s, 'qlike']:.4f}" for s in TREES)
        + ") sit next to the best linear ones ("
        + ", ".join(f"{s} {mn.loc[s, 'qlike']:.4f}" for s in LIN)
        + "), and the grand MCS keeps all but one. On the trade loss the best tree, "
        f"`{best_t['cand']}` (Sharpe {best_t['sharpe_mid']:.2f}), ranks among the elastic-net leave-one-out "
        "designs, and no family can be excluded. The fit-space loss sees the trees' larger errors because it "
        "weighs the adjusted-scale miss directly. QLIKE and the trade loss weigh the variance forecast and its "
        "side of the implied variance, where the families do not differ beyond noise.",
        "",
        "**Bottom line.** Across 576 candidates (6 estimators x 96 input sets) and four losses, the MCS cannot "
        "single out the input set. Its only firm exclusions are weak small-design trees on the fit-space error "
        "and a few activity add-ones on ES(1) and the trade loss. The pairwise evidence keeps pointing at ES "
        "volume and the stock cross-section's absolute returns. No MCS confirms that over these 866 days. The "
        "SPA and StepM tests against HAR + calendar do, on the variance losses (section above).",
    ]


if __name__ == "__main__":
    main()
