"""Report stage of experiments/close_trees_pretune_rfxgb.py: scores, pick stability, SUMMARY.md, export.

Reads every model x input set run under _work/<RXP_TAG>/<model>_<bucket>/ (pre-tune studies, plan,
retune logs, walk arms) and writes under results/trees_pretune_2026-10-06/:

  scorer_gate.csv            the plain scorer against the master table (plain forecasts of the stored
                             tree rows and the HAR + calendar OLS; Sharpe / Sortino / DM of the rows
                             the master-table GUI shows, writeup/master_table_gui/rows.py load_rows)
  gates.csv                  the cluster gates of every cell (fold fitter == the Optuna spec;
                             control refits == the stored masked tables) and the control arms
                             against the stored tables over all 1469 forecast rows
  arms.csv                   every arm, the stored master-table rows and the 2026-10-03 LightGBM
                             pre-tune arms on the 866 trade days: QLIKE, DM vs the control and vs
                             sub_ols_baseline, sign(s) Sharpe / Sortino mid and crossed, HAC t of the
                             daily P&L difference vs the control (lag 14), days with another position
  <cell>_pretune_trials.csv  every trial of every study (configuration, fold MSE, rounds, seconds)
  <cell>_pretune_curve.csv   best-so-far against the trial count (each study, merged), held-out folds
  <cell>_pretune_checkpoints.csv  the configuration chosen after k trials of every study
  <cell>_study_picks.csv     each study's best configuration; pick_stability.csv: the spread of the
                             picks across studies and folds for every cell
  <cell>_retune_log.csv / <cell>_retune_trials.csv / <cell>_frozen_picks.csv   (copied from _work)
  cpu.csv                    fit seconds of every stage and the allocated CPU time (sacct, if pulled)
  forecasts.parquet          every arm's forecasts in the export format (series_id, date, pred_adj)
  SUMMARY.md                 generated from the files above

Scorer (the master table's, writeup/master_table_gui/rows.py): plain forecast f^2 * B (no
recalibration; B = the export's diurnal scale), QLIKE = RV / plain - log(RV / plain) - 1 against the
master table's 16:00 target, sign(s): buy the straddle when plain > iv_var, else sell; mid P&L =
position x R, crossed = position x (exit / ask - 1) buying, (exit / bid - 1) selling (rows.
_crossed_return); Sharpe = mean / sd x sqrt(252); Sortino = rows._sortino; DM =
src.evaluation.diebold_mariano.dm_test on daily QLIKE; HAC t = notebooks/atm_straddle_lib.newey_west_t
(Bartlett, lag floor(1.5 n^(1/3)) = 14 at 866 days) of the paired daily P&L difference.  Point
estimates only.
"""

from __future__ import annotations

import itertools
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import close_trees_pretune_rfxgb as C

REPO = C.REPO
OUT = C.OUT
EXPORT = REPO / "results" / "close_studies_2026-10-03" / "forecasts"
DAILY = REPO / "results" / "close_master_table" / "master_table_daily.parquet"
QUOTES = REPO / "results" / "atm_straddle_0dte_1530" / "daily_a0.parquet"
OLS_KEY = "sub_ols_baseline"
FOLD_SPLITS = (
    ((0, 2), (1, 3)),
    ((1, 3), (0, 2)),
)  # (select on, score on): the LightGBM study's
LADDER = (1, 2, 5, 10, 25, 50, 100, 200, 300, 500, 1000)
MODELS = ("rf", "xgb")
MODEL_NAME = {"rf": "random forest", "xgb": "XGBoost", "lgbm": "LightGBM"}
ARM_ORDER = (
    "control",
    "frozen_k10",
    "frozen_k100",
    "frozen_all",
    "retune_any",
    "retune_margin",
)
ARM_LABEL = {
    "control": "shipped configuration (control)",
    "frozen_all": "pre-tuned, frozen",
    "retune_any": "pre-tuned + light retune (any gain)",
    "retune_margin": "pre-tuned + light retune (margin)",
}
LGBM_SERIES = {  # the 2026-10-03 LightGBM pre-tune arms in the export -> this report's arm names
    "trees_pretune/control_r10": "control_r10",
    "trees_pretune/frozen_k10_r10": "frozen_k10_r10",
    "trees_pretune/frozen_k45_r10": "frozen_all_r10",
    "trees_pretune/retune_any_r10": "retune_any_r10",
    "trees_pretune/retune_margin_r10": "retune_margin_r10",
}
TOL = 1e-9


def rows_module():
    gui = str(REPO / "writeup" / "master_table_gui")
    if gui not in sys.path:
        sys.path.insert(0, gui)
    import rows

    return rows


def nw_t(x) -> float:
    nb = str(REPO / "notebooks")
    if nb not in sys.path:
        sys.path.insert(0, nb)
    import atm_straddle_lib as asl

    return float(asl.newey_west_t(np.asarray(x, dtype=float))[0])


# ============================================================================ scorer
class Scorer:
    """The master table's plain sign(s) scorer on the 866 trade days."""

    def __init__(self) -> None:
        tg = pd.read_parquet(EXPORT / "targets.parquet")
        self.dates = pd.to_datetime(tg["date"]).to_numpy()
        self.B = pd.Series(
            tg["baseline"].to_numpy(float),
            index=pd.to_datetime(tg["date"]).dt.normalize(),
        )
        d = pd.read_parquet(DAILY, columns=["key", "day", "plain", "target", "iv_var"])
        d["day"] = pd.to_datetime(d["day"])
        self.master = d
        base = d[d["key"] == OLS_KEY].set_index("day").sort_index()
        q = pd.read_parquet(
            QUOTES, columns=["R", "exit", "bid_c", "bid_p", "ask_c", "ask_p"]
        )
        q.index = pd.to_datetime(q.index).normalize()
        self.days = base.index
        self.target = base["target"].to_numpy(float)
        self.iv = base["iv_var"].to_numpy(float)
        j = q.reindex(self.days)
        assert j.notna().all().all(), "quote book misses a trade day"
        self.R = j["R"].to_numpy(float)
        self.exit = j["exit"].to_numpy(float)
        self.bid = (j["bid_c"] + j["bid_p"]).to_numpy(float)
        self.ask = (j["ask_c"] + j["ask_p"]).to_numpy(float)
        self.ols_plain = base["plain"].to_numpy(float)
        self.ols = self.daily_from_plain(self.ols_plain)

    def plain(self, pred_adj: np.ndarray) -> np.ndarray:
        """f^2 * B on the trade days from a 1469-row adjusted-scale forecast."""
        s = pd.Series(
            np.asarray(pred_adj, float), index=pd.DatetimeIndex(self.dates).normalize()
        )
        f = s.reindex(self.days).to_numpy(float)
        return f**2 * self.B.reindex(self.days).to_numpy(float)

    def daily_from_plain(self, plain: np.ndarray) -> dict:
        rows = rows_module()
        ratio = self.target / plain
        q = np.where(plain > self.iv, 1.0, -1.0)
        return dict(
            plain=plain,
            ql=ratio - np.log(ratio) - 1.0,
            q=q,
            mid=q * self.R,
            crossed=rows._crossed_return(q, self.exit, self.bid, self.ask),
        )

    def daily(self, pred_adj: np.ndarray) -> dict:
        return self.daily_from_plain(self.plain(pred_adj))

    def stats(self, a: dict, ctrl: dict | None) -> dict:
        from src.evaluation.diebold_mariano import dm_test

        rows = rows_module()
        out = dict(
            n_days=len(a["ql"]),
            qlike=float(np.mean(a["ql"])),
            sharpe_mid=rows._sharpe_ann(a["mid"]),
            sharpe_crossed=rows._sharpe_ann(a["crossed"]),
            sortino_mid=rows._sortino(a["mid"]),
            sortino_crossed=rows._sortino(a["crossed"]),
            mean_mid=float(np.mean(a["mid"])),
            mean_crossed=float(np.mean(a["crossed"])),
            pct_buy=float(100 * np.mean(a["q"] > 0)),
        )
        o = dm_test(a["ql"], self.ols["ql"])
        out |= dict(dm_vs_ols=o["dm"], dm_vs_ols_p=o["p"])
        if ctrl is not None:
            c = dm_test(a["ql"], ctrl["ql"])
            if (
                np.max(np.abs(a["ql"] - ctrl["ql"])) < 1e-12
            ):  # the same forecast up to rounding: no test
                c = dict(dm=0.0, p=1.0)
            out |= dict(
                qlike_pct_vs_control=(out["qlike"] / float(np.mean(ctrl["ql"])) - 1)
                * 100,
                dm_vs_control=c["dm"],
                dm_vs_control_p=c["p"],
                dsharpe_mid_vs_control=out["sharpe_mid"]
                - rows._sharpe_ann(ctrl["mid"]),
                hac_t_mid_vs_control=nw_t(a["mid"] - ctrl["mid"])
                if np.any(a["mid"] != ctrl["mid"])
                else 0.0,
                hac_t_crossed_vs_control=nw_t(a["crossed"] - ctrl["crossed"])
                if np.any(a["crossed"] != ctrl["crossed"])
                else 0.0,
                days_other_position=int(np.sum(a["q"] != ctrl["q"])),
            )
        return out


def stored_pred(info_dates: np.ndarray, fname: str) -> np.ndarray:
    d = pd.read_parquet(REPO / "results" / "spxw_pnl" / fname)
    t = pd.to_datetime(d["t"]).dt.tz_convert("America/New_York").dt.tz_localize(None)
    s = pd.Series(d["yhat"].to_numpy(float), index=t)
    s = s[s.index.strftime("%H:%M") == "16:00"]
    want = pd.to_datetime(info_dates)
    assert len(s) == len(want) and (s.index == want).all(), (fname, len(s))
    return s.to_numpy(float)


def stored_tables(model: str, bucket: str) -> dict[str, tuple[str, str]]:
    """(stored table file, master-table key) of the shipped configuration's masked cluster runs."""
    return {
        "stored_T10": (
            f"yhat_subtree_{model}_{bucket}.parquet",
            f"subtree_{model}_{bucket}",
        ),
        "stored_T1": (
            f"yhat_subtree_daily_{bucket}_{model}.parquet",
            f"subtree_daily_{bucket}_{model}",
        ),
    }


def scorer_gate(S: Scorer) -> pd.DataFrame:
    rows = rows_module()
    out = []
    keys = sorted(S.master["key"].unique())
    gui = {
        r["key"]: r for r in rows.load_rows() if r.get("rule_name") == "sign(s)"
    }  # the sign(s) row of each key
    for m in MODELS + ("lgbm",):
        for b in ("all_features", "baseline"):
            for rung, (fname, key) in stored_tables(m, b).items():
                if (
                    key not in keys
                    or not (REPO / "results" / "spxw_pnl" / fname).is_file()
                ):
                    continue
                pred = stored_pred(S.dates, fname)
                mine = S.plain(pred)
                ref = (
                    S.master[S.master["key"] == key]
                    .set_index("day")
                    .sort_index()["plain"]
                    .reindex(S.days)
                    .to_numpy(float)
                )
                rel = float(np.max(np.abs(mine / ref - 1)))
                out.append(
                    dict(
                        check="plain f^2 B == master table daily plain",
                        item=key,
                        value=rel,
                        reference=0.0,
                        abs_diff=rel,
                        ok=rel <= TOL,
                    )
                )
                if key in gui:
                    st = S.stats(S.daily_from_plain(mine), None)
                    g = gui[key]
                    for mine_k, gui_k in (
                        ("sharpe_mid", "Sharpe_mid"),
                        ("sharpe_crossed", "Sharpe_crossed"),
                        ("sortino_mid", "sortino"),
                        ("sortino_crossed", "sortino_x"),
                        ("dm_vs_ols", "dm_vs_ref"),
                    ):
                        if not g.get(gui_k):
                            continue
                        v, r = float(st[mine_k]), float(g[gui_k])
                        out.append(
                            dict(
                                check=f"{mine_k} == master-table GUI {gui_k}",
                                item=key,
                                value=v,
                                reference=r,
                                abs_diff=abs(v - r),
                                ok=abs(v - r) <= TOL,
                            )
                        )
    g = gui.get(OLS_KEY)
    if g is not None:
        st = S.stats(S.ols, None)
        for mine_k, gui_k in (
            ("sharpe_mid", "Sharpe_mid"),
            ("sharpe_crossed", "Sharpe_crossed"),
            ("sortino_mid", "sortino"),
            ("sortino_crossed", "sortino_x"),
        ):
            v, r = float(st[mine_k]), float(g[gui_k])
            out.append(
                dict(
                    check=f"{mine_k} == master-table GUI {gui_k}",
                    item=OLS_KEY,
                    value=v,
                    reference=r,
                    abs_diff=abs(v - r),
                    ok=abs(v - r) <= TOL,
                )
            )
    df = pd.DataFrame(out)
    df.to_csv(OUT / "scorer_gate.csv", index=False)
    return df


# ============================================================================ pre-tune records
def axis_dist(model: str, a: np.ndarray, b: np.ndarray) -> float:
    """Largest axis difference of two encoded configurations as a share of the axis range (log
    units on log axes; categorical: steps in the ordered choice list over the steps available)."""
    import causal_tune_trees_optuna_jobs as OJ

    out = 0.0
    for j, (_, s) in enumerate(OJ.SPACES[model].items()):
        if s[0] == "cat":
            ch = [np.nan if c is None else float(c) for c in s[1]]

            def idx(v: float) -> int:
                return next(
                    i
                    for i, c in enumerate(ch)
                    if (np.isnan(c) and np.isnan(v)) or c == v
                )

            d = abs(idx(a[j]) - idx(b[j])) / (len(ch) - 1)
        else:
            _, lo, hi, log = s
            d = (
                abs(math.log(a[j]) - math.log(b[j])) / (math.log(hi) - math.log(lo))
                if log
                else abs(a[j] - b[j]) / (hi - lo)
            )
        out = max(out, d)
    return out


def fmt_cfg(model: str, vec) -> str:
    import causal_tune_trees_optuna_jobs as OJ

    lab = {
        "max_features": "col. share",
        "min_samples_leaf": "min leaf rows",
        "max_depth": "depth",
        "min_child_weight": "min leaf weight",
        "subsample": "row share",
        "colsample_bytree": "col. share",
        "reg_lambda": "L2",
        "learning_rate": "lr",
    }
    parts = []
    for v, (a, s) in zip(np.asarray(vec, float), OJ.SPACES[model].items()):
        if s[0] == "cat":
            t = "none" if np.isnan(v) else f"{v:.0f}"
        elif s[0] == "int":
            t = f"{v:.0f}"
        else:
            t = f"{v:.3g}"
        parts.append(f"{lab.get(a, a)} {t}")
    return ", ".join(parts)


def curve_for(
    model: str, name: str, P: np.ndarray, F: np.ndarray, ladder: bool = False
) -> pd.DataFrame:
    v = F.mean(axis=1)
    final = int(np.argmin(v))
    rows = []
    n = len(v)
    ks = sorted({k for k in LADDER if k <= n} | {n}) if ladder else range(1, n + 1)
    for k in ks:
        b = int(np.argmin(v[:k]))
        d = dict(
            sequence=name,
            k=k,
            best_trial=b,
            best_val_mse=v[b],
            rel_gap_to_final=v[b] / v[final] - 1,
            val_rel_to_shipped=v[b] / v[0],
            dist_to_final=axis_dist(model, P[b], P[final]),
        )
        if F.shape[1] == 4:
            ho = []
            for sel, sc in FOLD_SPLITS:
                bs = int(np.argmin(F[:k, list(sel)].mean(axis=1)))
                ho.append(F[bs, list(sc)].mean() / F[0, list(sc)].mean())
            d["heldout_rel_to_shipped"] = float(np.mean(ho))
        rows.append(d)
    return pd.DataFrame(rows)


def pretune_tables(model: str, bucket: str, run: Path) -> dict:
    seeds = sorted(
        int(p.stem.split("_s")[1])
        for p in run.glob("pretune_s*.npz")
        if ".tmp" not in p.name
    )
    if not seeds:
        return {}
    import causal_tune_trees_optuna_jobs as OJ

    recs = []
    for s in seeds:
        z = np.load(run / f"pretune_s{s}.npz", allow_pickle=False)
        d = {k: z[k] for k in z.files if k != "meta"}
        d["meta"], d["seed"] = json.loads(str(z["meta"])), s
        recs.append(d)
    cell = f"{model}_{bucket}"
    axes = list(OJ.SPACES[model])
    trows = []
    for r in recs:
        for i in range(len(r["val_mse"])):
            d = dict(study=f"s{r['seed']}", trial=i)
            d |= {a: r["params"][i, j] for j, a in enumerate(axes)}
            d["val_mse"] = r["val_mse"][i]
            for f in range(r["fold_val_mse"].shape[1]):
                d[f"fold{f}_mse"] = r["fold_val_mse"][i, f]
            d["val_qlike"] = float(np.mean(r["fold_val_qlike"][i]))
            d["rounds_median"] = float(np.median(r["fold_rounds"][i]))
            d["cap_hits"] = int(
                np.sum(
                    (r["fold_rounds_max"][i] > 0)
                    & (r["fold_rounds"][i] >= r["fold_rounds_max"][i])
                )
            )
            d["fit_sec"] = float(np.sum(r["fold_sec"][i]))
            trows.append(d)
    tt = pd.DataFrame(trows)
    tt.to_csv(OUT / f"{cell}_pretune_trials.csv", index=False)
    # curves: each study (ladder points) and the merged sequence (every k on the ladder)
    parts = [
        curve_for(model, f"s{r['seed']}", r["params"], r["fold_val_mse"], ladder=True)
        for r in recs
    ]
    C.MODEL = model  # merge_seeds / best_after are model-free
    m = C.merge_seeds(recs)
    n_k = int(m["trial"].max()) + 1
    mrows = []
    for k in sorted({k for k in LADDER if k < n_k} | {n_k}):
        sel = m["trial"] < k
        idx = np.flatnonzero(sel)
        F = m["fold_val_mse"][idx]
        v = F.mean(axis=1)
        b = idx[int(np.argmin(v))]
        ho = []
        for s_on, s_sc in FOLD_SPLITS:
            bs = idx[int(np.argmin(F[:, list(s_on)].mean(axis=1)))]
            ho.append(
                m["fold_val_mse"][bs, list(s_sc)].mean()
                / m["fold_val_mse"][0, list(s_sc)].mean()
            )
        fin = int(np.argmin(m["val_mse"]))
        mrows.append(
            dict(
                sequence="merged",
                k=k,
                trials_total=int(sel.sum()),
                best_study=f"s{m['study'][b]}",
                best_trial=int(m["trial"][b]),
                best_val_mse=float(m["val_mse"][b]),
                rel_gap_to_final=float(m["val_mse"][b] / m["val_mse"][fin] - 1),
                val_rel_to_shipped=float(m["val_mse"][b] / m["val_mse"][0]),
                heldout_rel_to_shipped=float(np.mean(ho)),
                dist_to_final=axis_dist(model, m["params"][b], m["params"][fin]),
                rounds=float(np.median(m["fold_rounds"][b])),
                configuration=fmt_cfg(model, m["params"][b]),
            )
        )
    chk = pd.DataFrame(mrows)
    chk.to_csv(OUT / f"{cell}_pretune_checkpoints.csv", index=False)
    cur = pd.concat(
        parts
        + [chk.drop(columns=["configuration", "best_study", "trials_total", "rounds"])],
        ignore_index=True,
    )
    cur.to_csv(OUT / f"{cell}_pretune_curve.csv", index=False)
    # pick stability: each study's best; pairwise distances; per-fold picks; fold rank agreement
    picks = []
    for r in recs:
        b = int(np.argmin(r["val_mse"]))
        picks.append(
            dict(
                study=f"s{r['seed']}",
                trials=len(r["val_mse"]),
                best_trial=b,
                val_mse=float(r["val_mse"][b]),
                rel_to_shipped=float(r["val_mse"][b] / r["val_mse"][0]),
                configuration=fmt_cfg(model, r["params"][b]),
                **{a: r["params"][b, j] for j, a in enumerate(axes)},
            )
        )
    pk = pd.DataFrame(picks)
    pk.to_csv(OUT / f"{cell}_study_picks.csv", index=False)
    P = np.vstack([r["params"][int(np.argmin(r["val_mse"]))] for r in recs])
    pair = [
        axis_dist(model, P[i], P[j])
        for i, j in itertools.combinations(range(len(P)), 2)
    ]
    F = m["fold_val_mse"]
    fold_best = [int(np.argmin(F[:, f])) for f in range(F.shape[1])]
    fold_pair = [
        axis_dist(model, m["params"][i], m["params"][j])
        for i, j in itertools.combinations(fold_best, 2)
    ]
    rk = pd.DataFrame(F).rank()
    rho = rk.corr().to_numpy()[np.triu_indices(F.shape[1], 1)]
    sec = np.concatenate([r["fold_sec"].ravel() for r in recs])
    stab = dict(
        cell=cell,
        studies=len(recs),
        trials_total=int(sum(len(r["val_mse"]) for r in recs)),
        trials_min=int(min(len(r["val_mse"]) for r in recs)),
        trials_max=int(max(len(r["val_mse"]) for r in recs)),
        best_val_rel_to_shipped=float(m["val_mse"].min() / m["val_mse"][0]),
        study_best_rel_min=float(pk["rel_to_shipped"].min()),
        study_best_rel_max=float(pk["rel_to_shipped"].max()),
        pick_dist_median=float(np.median(pair)) if pair else np.nan,
        pick_dist_max=float(np.max(pair)) if pair else np.nan,
        fold_pick_dist_max=float(np.max(fold_pair)) if fold_pair else np.nan,
        fold_rank_corr_min=float(rho.min()),
        fold_rank_corr_mean=float(rho.mean()),
        heldout_rel_to_shipped_final=float(chk["heldout_rel_to_shipped"].iloc[-1]),
        fold_fit_core_hours=float(sec.sum() / 3600),
        fold_fit_sec_mean=float(sec.mean()),
    )
    for f in ("retune_log.csv", "retune_trials.csv", "frozen_picks.csv"):
        if (run / f).is_file():
            shutil.copy(run / f, OUT / f"{cell}_{f}")
    return dict(
        recs=recs, merged=m, checkpoints=chk, picks=pk, stability=stab, trials=tt
    )


# ============================================================================ arms
def walk_preds(run: Path) -> dict[str, np.ndarray]:
    out = {}
    for f in sorted(run.glob("walk_*_r*.npz")):
        if ".tmp" in f.name:
            continue
        z = np.load(f, allow_pickle=False)
        name = f.stem[len("walk_") :]
        p = z["pred"]
        if not np.isfinite(p).all():
            continue  # a restricted (smoke) walk
        out[name] = p
    return out


def arm_sort_key(name: str) -> tuple:
    base, cad = name.rsplit("_r", 1)
    o = (
        ARM_ORDER.index(base)
        if base in ARM_ORDER
        else (
            ARM_ORDER.index("frozen_k10") + 0.5 if base.startswith("frozen_k") else 99
        )
    )
    return (-int(cad), o, name)


def cells_found(tag: str) -> list[tuple[str, str]]:
    base = C.WORK / tag
    return [(m, b) for m in MODELS for b in C.BUCKETS if (base / f"{m}_{b}").is_dir()]


def score_cells(
    S: Scorer, tag: str
) -> tuple[pd.DataFrame, pd.DataFrame, list[pd.DataFrame]]:
    rows, gates, export = [], [], []
    for m, b in cells_found(tag):
        run = C.WORK / tag / f"{m}_{b}"
        preds = walk_preds(run)
        for d in (
            C.WORK / "canary" / f"{m}_{b}",
            run,
        ):  # the cluster gate runs in the canary
            if (d / f"gate_{m}_{b}.csv").is_file():
                gates.append(pd.read_csv(d / f"gate_{m}_{b}.csv"))
                break
        if not preds:
            continue
        for cad in (10, 1):
            ck = f"control_r{cad}"
            if ck not in preds:
                continue
            ctrl = S.daily(preds[ck])
            names = sorted(
                [k for k in preds if k.endswith(f"_r{cad}")], key=arm_sort_key
            )
            for k in names:
                st = S.stats(S.daily(preds[k]), ctrl)
                rows.append(
                    dict(
                        model=m,
                        bucket=b,
                        cadence=cad,
                        arm=k.rsplit("_r", 1)[0],
                        series_id=f"{m}_{b}/{k}",
                        source="this run",
                    )
                    | st
                )
            # the stored masked cluster tables of the shipped configuration, against this control
            rung = "stored_T10" if cad == 10 else "stored_T1"
            fname, key = stored_tables(m, b)[rung]
            if (REPO / "results" / "spxw_pnl" / fname).is_file():
                sp = stored_pred(S.dates, fname)
                st = S.stats(S.daily(sp), ctrl)
                rows.append(
                    dict(
                        model=m,
                        bucket=b,
                        cadence=cad,
                        arm=rung,
                        series_id=f"spxw_pnl/{fname}",
                        source=f"stored ({key})",
                    )
                    | st
                )
                rel = np.abs(preds[ck] / sp - 1)
                gates.append(
                    pd.DataFrame(
                        [
                            dict(
                                gate=f"control_r{cad} == stored {rung[7:]} table over all 1469 rows",
                                item=fname,
                                value=float(np.nanmax(rel)),
                                reference=0.0,
                                max_rel_diff=float(np.nanmax(rel)),
                                ok=bool(np.nanmax(rel) <= TOL),
                                model=m,
                                bucket=b,
                                rows_differing=int(np.sum(rel > 0)),
                                rows_beyond_rounding=int(np.sum(rel > 1e-12)),
                                rows_beyond_list=" ".join(
                                    str(i) for i in np.flatnonzero(rel > 1e-12)[:20]
                                ),
                                trade_days_other_position=int(
                                    np.sum(S.daily(preds[ck])["q"] != S.daily(sp)["q"])
                                ),
                                qlike_abs_diff=abs(
                                    float(np.mean(S.daily(preds[ck])["ql"]))
                                    - float(np.mean(S.daily(sp)["ql"]))
                                ),
                            )
                        ]
                    )
                )
        tg = pd.read_parquet(EXPORT / "targets.parquet", columns=["row", "date"])
        for k, p in preds.items():
            export.append(
                pd.DataFrame(
                    dict(
                        series_id=f"{m}_{b}/{k}",
                        study="trees_pretune_2026-10-06",
                        row=tg["row"].to_numpy(),
                        date=tg["date"].to_numpy(),
                        pred_adj=p,
                    )
                )
            )
    # the 2026-10-03 LightGBM pre-tune arms, re-scored on the plain forecast
    fc = pd.read_parquet(
        EXPORT / "forecasts.parquet", columns=["series_id", "date", "pred_adj"]
    )
    fc = fc[fc["series_id"].isin(list(LGBM_SERIES))]
    wide = fc.pivot(index="date", columns="series_id", values="pred_adj").sort_index()
    assert (pd.to_datetime(wide.index).to_numpy() == S.dates).all()
    ctrl = S.daily(wide["trees_pretune/control_r10"].to_numpy(float))
    for sid, arm in LGBM_SERIES.items():
        st = S.stats(S.daily(wide[sid].to_numpy(float)), ctrl)
        rows.append(
            dict(
                model="lgbm",
                bucket="all_features",
                cadence=10,
                arm=arm.rsplit("_r", 1)[0],
                series_id=sid,
                source="2026-10-03 export",
            )
            | st
        )
    fname, key = stored_tables("lgbm", "all_features")["stored_T10"]
    st = S.stats(S.daily(stored_pred(S.dates, fname)), ctrl)
    rows.append(
        dict(
            model="lgbm",
            bucket="all_features",
            cadence=10,
            arm="stored_T10",
            series_id=f"spxw_pnl/{fname}",
            source=f"stored ({key})",
        )
        | st
    )
    st = S.stats(S.ols, None)
    rows.append(
        dict(
            model="ols",
            bucket="baseline",
            cadence=1,
            arm=OLS_KEY,
            series_id=f"master_table/{OLS_KEY}",
            source="master table",
        )
        | st
    )
    arms = pd.DataFrame(rows)
    arms.to_csv(OUT / "arms.csv", index=False)
    gt = pd.concat(gates, ignore_index=True) if gates else pd.DataFrame()
    gt.to_csv(OUT / "gates.csv", index=False)
    return arms, gt, export


def cpu_table(tag: str, pt: dict) -> pd.DataFrame:
    rows = []
    for m, b in cells_found(tag):
        run = C.WORK / tag / f"{m}_{b}"
        cell = f"{m}_{b}"
        if cell in pt:
            rows.append(
                dict(
                    cell=cell,
                    stage="pre-tune fold fits",
                    core_hours=pt[cell]["stability"]["fold_fit_core_hours"],
                )
            )
        if (run / "retune_trials.csv").is_file():
            rows.append(
                dict(
                    cell=cell,
                    stage="retune fold fits",
                    core_hours=pd.read_csv(run / "retune_trials.csv")["sec"].sum()
                    / 3600,
                )
            )
        sec = 0.0
        for f in (run / "refits").glob("*_p*.npz") if (run / "refits").is_dir() else []:
            if ".tmp" not in f.name:
                sec += float(np.load(f)["fit_sec"].sum())
        rows.append(dict(cell=cell, stage="walk refits", core_hours=sec / 3600))
    df = pd.DataFrame(rows)
    sa = OUT / "sacct.csv"
    if sa.is_file():
        s = pd.read_csv(sa)
        s = s[s["elapsed_s"] > 0]
        st, en = (
            pd.to_datetime(s["start"], errors="coerce"),
            pd.to_datetime(s["end"], errors="coerce"),
        )
        ok = st.notna() & en.notna()
        ev = sorted(
            [(t, c) for t, c in zip(st[ok], s.loc[ok, "alloc_cpus"])]
            + [(t, -c) for t, c in zip(en[ok], s.loc[ok, "alloc_cpus"])],
            key=lambda x: (x[0], x[1]),
        )
        run_c, peak = 0, 0
        for _, c in ev:
            run_c += c
            peak = max(peak, run_c)
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        dict(
                            cell="all",
                            stage="allocated (sacct: elapsed x CPUs)",
                            core_hours=float(s["cpu_hours"].sum()),
                        ),
                        dict(
                            cell="all",
                            stage="wall-clock hours (first start to last end)",
                            core_hours=float(
                                (en[ok].max() - st[ok].min()).total_seconds() / 3600
                            ),
                        ),
                        dict(
                            cell="all",
                            stage="peak concurrent CPUs",
                            core_hours=float(peak),
                        ),
                        dict(
                            cell="all",
                            stage="jobs / array tasks",
                            core_hours=float(len(s)),
                        ),
                    ]
                ),
            ],
            ignore_index=True,
        )
    df.to_csv(OUT / "cpu.csv", index=False)
    return df


# ============================================================================ summary
def f4(v) -> str:
    return (
        "" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.4f}"
    )


def f2(v, sign: bool = False) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ""
    return f"{v:+.2f}" if sign else f"{v:.2f}"


def write_summary(
    tag: str,
    sg: pd.DataFrame,
    gt: pd.DataFrame,
    arms: pd.DataFrame,
    pt: dict,
    cpu: pd.DataFrame,
) -> None:
    L: list[str] = []
    w = L.append
    w(
        "# Random forest and XGBoost on the 16:00 bar: heavy pre-2020 pre-tune, light retune every 250 sessions"
    )
    w("")
    w(
        "Generated by `experiments/close_trees_pretune_rfxgb.py report` (`experiments/close_trees_pretune_rfxgb_report.py`) from "
        "this folder's CSVs. The 2026-10-03 LightGBM pre-tune study (`results/close_studies_2026-10-03/trees_pretune/`) extended "
        "to the random forest and XGBoost, same protocol: the tree spec's model (`specs/causal_tune_trees.py`) on the 16:00-bar "
        "design of one input set, 2000-session window, window column mask on, every fit single-threaded; search space, round "
        "rules and model construction from `specs/causal_tune_trees_optuna_jobs.py`. Run tag `"
        + tag
        + "`."
    )
    w("")
    w(
        "Scorer: the master table's plain forecast f^2 x B (no recalibration), QLIKE against the master table's 16:00 target and "
        "the 15:30 sign(s) straddle (buy when the plain forecast exceeds the implied variance, else sell; mid and crossed fills as "
        "`writeup/master_table_gui/rows.py`), the 866 trade days 2020-01-03 .. 2024-04-30, point estimates. DM = "
        "`src/evaluation/diebold_mariano.dm_test` on daily QLIKE (negative = lower QLIKE); HAC t = Newey-West t (lag 14) of the "
        "paired daily mid P&L difference against the control of the same model, input set and refit cadence (positive = more P&L)."
    )
    w("")
    w("## Gates")
    w("")
    w(
        f"- Scorer vs the master table (`scorer_gate.csv`): {int(sg['ok'].sum())} of {len(sg)} checks pass (plain forecasts "
        f"of the stored rows within {TOL:g} relative; Sharpe, Sortino and DM of the rows the master-table GUI shows within {TOL:g}); "
        f"largest relative plain difference {sg.loc[sg['check'].str.startswith('plain'), 'abs_diff'].max():.1e}, largest "
        f"statistic difference {sg.loc[~sg['check'].str.startswith('plain'), 'abs_diff'].max():.1e}."
    )
    dc = OUT / "design_check.csv"
    if dc.is_file():
        d = pd.read_csv(dc)
        mm = d[
            (d["reference_sha256"].notna())
            & (d["reference_sha256"] != "")
            & (~d["matches_reference"])
        ]
        w(
            f"- Design caches (`design_check.csv`): {len(d[d['array'] != 'X vs the 2026-10-03 export (stored cache)'])} arrays hashed; "
            f"against `design_hashes.json` the arrays that differ: {', '.join(sorted(set(mm['design'] + ':' + mm['array']))) or 'none'} "
            f"({d.loc[d['array'].str.startswith('X vs'), 'sha256'].iloc[0]} on the bar1600 all_features X: the AVX2 build of the "
            "design, the one the stored masked cluster runs were fitted on; the gate below compares with those runs)."
        )
    if not gt.empty:
        for (m, b), g in gt.groupby(["model", "bucket"], sort=False):
            ff = g[g["gate"].str.startswith("fold fitter")]
            cr = g[g["gate"].str.contains("over all 1469")]
            sm = g[g["gate"].str.startswith("control refit")]
            parts = []
            if len(ff):
                parts.append(
                    f"fold fitter == spec fit_trial: {'pass' if ff['ok'].all() else 'FAIL'}"
                )
            if len(sm) and sm["max_rel_diff"].notna().any():
                parts.append(
                    f"control refits at sample rows vs stored tables: max rel. diff {sm['max_rel_diff'].max():.1e}"
                )
            elif len(sm):
                parts.append(
                    "the cluster canary had no stored table for this input set (the masked lean_vol_vix tree tables were written by another workstream at 18:52, after the ship)"
                )
            for _, r in cr.iterrows():
                nb = int(r["rows_beyond_rounding"])
                parts.append(
                    f"{r['gate'].split(' ==')[0]} vs stored over 1469 rows: max rel. diff {r['max_rel_diff']:.1e}"
                    + (
                        f" ({nb} rows beyond 1e-12: {r['rows_beyond_list']}; positions differ on {int(r['trade_days_other_position'])} trade days, QLIKE differs by {r['qlike_abs_diff']:.1e})"
                        if nb
                        else " (rounding only)"
                    )
                )
            w(f"- {MODEL_NAME[m]} / {b}: " + "; ".join(parts) + ".")
    w("")
    w("## Pre-tune (sessions before 2020-01-03 only) and pick stability")
    w("")
    if pt:
        any_cell = next(iter(pt.values()))
        meta = any_cell["recs"][0]["meta"]
        w(
            f"Independent Optuna TPE studies per model x input set (sampler seed {C.TPE_SEED_BASE} + study seed; trial 0 = the shipped "
            "configuration), merged by interleaving. Objective: mean validation MSE on the transformed target over "
            f"{len(meta['folds'])} blocked walk-forward folds of {meta['fold_len']} sessions, each fitted on the 2000 sessions before a "
            f"{meta['embargo']}-session embargo: "
            + "; ".join(
                f"fold {f['fold']} validates {f['val_first']} .. {f['val_last']}"
                for f in meta["folds"]
            )
            + ". Rows: the 16:00 sessions of the last-hour design cache. Each study stopped at its trial cap or time budget. "
            "'held-out' = select on folds {0, 2}, score on {1, 3}, and the reverse, averaged, as a ratio to the shipped configuration "
            "(the LightGBM study: 1.1346 after 71 trials). 'pick spread' = the largest axis difference between two studies' best "
            "configurations, share of the axis range (LightGBM: 47 % between its 2 studies); 'fold picks' = the same between the "
            "best trials of the single folds; 'fold rank corr.' = Spearman correlation of the trials' validation MSE between folds."
        )
        w("")
        w(
            "| model | input set | studies | trials (min-max a study) | best / shipped (4 folds) | held-out / shipped | pick spread median / max | fold picks max | fold rank corr. min / mean | fold-fit core-h |"
        )
        w("|---|---|---|---|---|---|---|---|---|---|")
        for cell, d in pt.items():
            s = d["stability"]
            m, b = cell.split("_", 1)
            w(
                f"| {MODEL_NAME[m]} | {b} | {s['studies']} | {s['trials_total']} ({s['trials_min']}-{s['trials_max']}) | "
                f"{s['best_val_rel_to_shipped']:.4f} | {s['heldout_rel_to_shipped_final']:.4f} | {s['pick_dist_median'] * 100:.0f} % / "
                f"{s['pick_dist_max'] * 100:.0f} % | {s['fold_pick_dist_max'] * 100:.0f} % | {s['fold_rank_corr_min']:.2f} / "
                f"{s['fold_rank_corr_mean']:.2f} | {s['fold_fit_core_hours']:.1f} |"
            )
        w("")
        w(
            "Best-so-far of the merged studies against the trial count (`<cell>_pretune_checkpoints.csv`; k = trials of every study):"
        )
        w("")
        w(
            "| model | input set | k (trials in all) | best (study #trial) | val MSE / shipped | held-out / shipped | distance to final | configuration |"
        )
        w("|---|---|---|---|---|---|---|---|")
        for cell, d in pt.items():
            m, b = cell.split("_", 1)
            for _, r in d["checkpoints"].iterrows():
                w(
                    f"| {MODEL_NAME[m]} | {b} | {int(r['k'])} ({int(r['trials_total'])}) | {r['best_study']} #{int(r['best_trial'])} | {r['val_rel_to_shipped']:.4f} | {r['heldout_rel_to_shipped']:.4f} | {r['dist_to_final'] * 100:.0f} % | {r['configuration']} |"
                )
        w("")
    rl = [
        (c, pd.read_csv(OUT / f"{c}_retune_log.csv"))
        for c in pt
        if (OUT / f"{c}_retune_log.csv").is_file()
    ]
    if rl:
        w("## Light retunes (every 250 sessions from 2020)")
        w("")
        w(
            f"At each retune row: {C.RETUNE_TRIALS} trials (trial 0 = the incumbent; all challengers are TPE start-up draws, i.e. "
            f"random in a box of {C.NEIGHBOURHOOD_FRACTION:g} of each axis's range around the incumbent; random forest depth: the "
            "incumbent's neighbours in {2, 4, 8, 16, 32, none}), validated on the 250 sessions before the row as 2 folds of 125 "
            "sessions on the 16:00-bar design. Switches (rows 380 / 630 / 880 / 1130 / 1380 = 2019-12-31 / 2020-12-28 / 2021-12-27 / "
            f"2022-12-22 / 2023-12-21); margin rule: HAC t below -z({C.MARGIN_ALPHA:g} / 9)."
        )
        w("")
        w(
            "| model | input set | path | switched at | relative gains of the best challenger | HAC t |"
        )
        w("|---|---|---|---|---|---|")
        for c, df in rl:
            m, b = c.split("_", 1)
            for p, g in df.groupby("path", sort=False):
                sw = (
                    ", ".join(str(int(r)) for r in g.loc[g["switched"], "row"])
                    or "never"
                )
                w(
                    f"| {MODEL_NAME[m]} | {b} | {p} | {sw} | {', '.join(f'{v * 100:+.1f} %' for v in g['rel_gain'])} | {', '.join(f'{v:+.2f}' for v in g['hac_t'])} |"
                )
        w("")
    w("## Out of sample: the 866 trade days (plain forecast)")
    w("")
    w(
        "| model | input set | refit | arm | QLIKE | vs control | DM vs control (p) | DM vs OLS (p) | Sharpe mid / crossed | Sortino mid / crossed | HAC t vs control mid / crossed | days other position | buy % |"
    )
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in arms.iterrows():
        ctl = "dm_vs_control" in r and pd.notna(r.get("dm_vs_control"))
        w(
            f"| {MODEL_NAME.get(r['model'], r['model'])} | {r['bucket']} | {r['cadence']} | {r['arm']} | {r['qlike']:.4f} | "
            + (
                f"{r['qlike_pct_vs_control']:+.2f} % | {r['dm_vs_control']:+.2f} ({r['dm_vs_control_p']:.3f}) | "
                if ctl
                else " |  | "
            )
            + f"{r['dm_vs_ols']:+.2f} ({r['dm_vs_ols_p']:.3f}) | {r['sharpe_mid']:.2f} / {r['sharpe_crossed']:.2f} | "
            f"{r['sortino_mid']:.2f} / {r['sortino_crossed']:.2f} | "
            + (
                f"{r['hac_t_mid_vs_control']:+.2f} / {r['hac_t_crossed_vs_control']:+.2f} | {int(r['days_other_position'])} | "
                if ctl
                else " |  | "
            )
            + f"{r['pct_buy']:.1f} |"
        )
    w("")
    w(
        "Arms: `control` the shipped configuration; `frozen_k<k>` the best of the first k trials of every study; `frozen_all` the "
        "best of all trials (the pre-tuned configuration), frozen; `retune_any` / `retune_margin` frozen_all plus the light retunes "
        "above. `stored_T10` / `stored_T1`: the master table's masked cluster tables of the shipped configuration (refit every 10 / "
        "every session). LightGBM rows: the 2026-10-03 study's exported arms (`trees_pretune/*`, refit every 10; its frozen_all = "
        "frozen_k45) re-scored here on the plain forecast. `sub_ols_baseline`: the HAR + calendar OLS (the DM reference)."
    )
    w("")
    w("## CPU")
    w("")
    if not cpu.empty:
        tot = cpu[cpu["cell"] != "all"].groupby("stage")["core_hours"].sum()
        for st, v in tot.items():
            w(f"- {st}: {v:.1f} fit core-hours")
        al = cpu[cpu["cell"] == "all"].set_index("stage")["core_hours"]
        if len(al):
            w(
                f"- CARC (`sacct.csv`, {al.get('jobs / array tasks', float('nan')):.0f} tasks): {al.iloc[0]:.0f} allocated CPU-hours, "
                f"wall-clock {al.get('wall-clock hours (first start to last end)', float('nan')):.1f} h from the first start to the last end, "
                f"peak {al.get('peak concurrent CPUs', float('nan')):.0f} CPUs at once"
            )
    (OUT / "SUMMARY.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tag = C.TAG
    S = Scorer()
    sg = scorer_gate(S)
    pt = {}
    for m, b in cells_found(tag):
        d = pretune_tables(m, b, C.WORK / tag / f"{m}_{b}")
        if d:
            pt[f"{m}_{b}"] = d
    if pt:
        pd.DataFrame([d["stability"] for d in pt.values()]).to_csv(
            OUT / "pick_stability.csv", index=False
        )
    arms, gt, export = score_cells(S, tag)
    if export:
        fc = pd.concat(export, ignore_index=True)
        fc["row"] = fc["row"].astype("int16")
        fc.to_parquet(OUT / "forecasts.parquet", index=False, compression="zstd")
    cpu = cpu_table(tag, pt)
    write_summary(tag, sg, gt, arms, pt, cpu)
    print(
        arms[
            [
                "model",
                "bucket",
                "cadence",
                "arm",
                "qlike",
                "sharpe_mid",
                "dm_vs_control",
                "hac_t_mid_vs_control",
            ]
        ].to_string(index=False)
    )
    C.say(f"report written to {OUT}")


if __name__ == "__main__":
    sys.exit(main())
