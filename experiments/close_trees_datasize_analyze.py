"""Analysis stage of experiments/close_trees_datasize.py (``python experiments/close_trees_datasize.py analyze``).

Scores every finished arm in _work/ with the master table's research convention
(dense_vs_sparse_1530.research_frame -> deck_panel -> point) on the 866 trade days, and writes:

  gates.csv          the stored reference forecasts re-scored here against the master table; the
                     stored LightGBM scored on the tree forecast rows only (TREE_START on) against
                     all 1469 rows; the local controls against their stored forecasts
  arms.csv           every arm: QLIKE, sign(s) Sharpe mid / crossed, share of buys, Diebold-Mariano on
                     daily QLIKE and a HAC t on the paired daily P&L difference against the local
                     LightGBM control (lgbm_bar1600_w2000), CPU time
  vs_reference.csv   every arm against the same model's arm with 2000 sessions of 16:00 rows, on the
                     shipped one-bar design (bar1600) and on the last30 design's 16:00 rows (h16)
  trees_vs_linear.csv  the change from more rows for the trees minus the same change for each linear
                     model (difference in differences; DM on the daily QLIKE series of differences,
                     HAC t on the daily P&L series of differences)
  curve.csv          the learning curve on the 16:00 rows (h16), with the pooled and one-bar arms
  learning_curve.png QLIKE and Sharpe mid against training sessions
  SUMMARY.md         written from the CSVs above

The circular block bootstrap is off in this session (commit b761b28): point estimates only; every
difference carries the DM statistic (src.evaluation.diebold_mariano.dm_test, the master table's
call) and the HAC t (atm_straddle_lib.newey_west_t, the master table's t_hac_mid_vs_ref).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import close_trees_datasize as c
import dense_vs_sparse_1530 as dvs
from src.evaluation.diebold_mariano import dm_test

CTRL = "lgbm_bar1600_w2000"
MASTER = c.REPO / "results" / "close_master_table" / "master_table.csv"
STORED = c.REPO / "results" / "spxw_pnl"
# the stored forecasts behind the brief's reference rows (master-table keys); our local arm of each
STORED_KEYS = {
    "subtree_lgbm_all_features": "lgbm_bar1600_w2000",
    "subtree_xgb_all_features": "xgb_bar1600_w2000",
    "sub_lasso_all_features": "lasso_bar1600_w2000",
    "sub_ridge_all_features": "ridge_bar1600_w2000",
    "subtree_daily_all_features_lgbm": "",
    "sub_lasso_baseline": "",
}
REPRO_TOL = 1e-9  # the master table stores full-precision floats
LABEL = {
    "lgbm": "LightGBM",
    "xgb": "XGBoost",
    "rf": "random forest",
    "ridge": "ridge",
    "lasso": "lasso",
}
SOURCE_TEXT = {
    "bar1600": "16:00 rows, one-bar design (shipped)",
    "pool": "both last-hour bars (last30 design)",
    "h16": "16:00 rows of the last30 design",
}


def stored_pred(key: str, dz: dict) -> np.ndarray:
    d = pd.read_parquet(STORED / f"yhat_{key}.parquet")
    t = pd.to_datetime(d["t"]).dt.tz_convert("America/New_York").dt.tz_localize(None)
    s = pd.Series(d["yhat"].to_numpy(float), index=pd.DatetimeIndex(t))
    out = s.reindex(pd.DatetimeIndex(pd.to_datetime(dz["date"]))).to_numpy(float)
    assert np.isfinite(out).all(), key
    return out


def load_arm(arm: str) -> dict | None:
    f = c.WORK / f"{arm}.npz"
    if not f.is_file():
        return None
    z = np.load(f, allow_pickle=False)
    out = {k: z[k] for k in z.files}
    out["meta"] = json.loads(str(out["meta"]))
    return out


def window_text(arm: str, run: dict) -> tuple[str, float, int, int]:
    """(label, sessions on the first trade day, smallest and largest training rows)."""
    _, src, win = c.ARMS[arm]
    nt = run["n_train"]
    lo, hi = int(nt.min()), int(nt.max())
    if src == "pool":
        return "2000 sessions (4000 rows)", 2000.0, lo, hi
    if win == "exp":
        return f"expanding ({lo}..{hi} sessions)", float("nan"), lo, hi
    return f"{win} sessions", float(win), lo, hi


def compare(P: dict, a: int, b: int) -> dict:
    dm = dm_test(P["ql"][:, a], P["ql"][:, b])
    t, lag = c_asl().newey_west_t(P["pnl"][:, a] - P["pnl"][:, b])
    return dict(dm=dm["dm"], dm_p=dm["p"], dm_hac_lag=dm.get("hac_lag", np.nan), t_hac=t, hac_lag=lag)


def c_asl():
    import atm_straddle_lib as asl

    return asl


def analyze() -> None:  # noqa: C901 - one linear report
    c.REPORT.mkdir(parents=True, exist_ok=True)
    S = c.sources()
    dz_all = S["dz"]
    n_fc = S["n_fc"]
    dk = dvs.deck_frame()
    fc_dates = pd.DatetimeIndex(pd.to_datetime(dz_all["date"]))
    first_trade = int(fc_dates.normalize().get_indexer([dk.index[0]])[0])
    import score_linear_subsection_causal as slc

    # the tree forecasts start TREE_START rows in; the recalibration on the first trade day must
    # still see a full window of forecasts
    assert first_trade - c.TREE_START >= slc.SMEAR_W, (first_trade, c.TREE_START)
    sub = np.arange(c.TREE_START, n_fc)
    dz = {k: v[sub] for k, v in dz_all.items()}

    # ---------------------------------------------------------------- gates
    master = pd.read_csv(MASTER).drop_duplicates("key").set_index("key")
    gates = []
    keys = list(STORED_KEYS)
    st_full = [dvs.research_frame(dz_all, stored_pred(k, dz_all)) for k in keys]
    st_sub = [dvs.research_frame(dz, stored_pred(k, dz_all)[sub]) for k in keys]
    Pf = dvs.deck_panel(st_full, dk)
    Ps = dvs.deck_panel(st_sub, dk)
    pf, ps = dvs.point(Pf), dvs.point(Ps)
    for i, k in enumerate(keys):
        for what, ours, theirs in (
            ("QLIKE", pf["ql"][i], master.loc[k, "qlike_recal"]),
            ("Sharpe mid", pf["sh"][i], master.loc[k, "Sharpe_mid"]),
        ):
            gates.append(
                dict(
                    gate=f"stored {k}: {what} re-scored here = master table",
                    value=ours,
                    reference=theirs,
                    abs_diff=abs(ours - theirs),
                    passed=bool(abs(ours - theirs) < REPRO_TOL),
                )
            )
        for what, j in (("QLIKE", "ql"), ("Sharpe mid", "sh")):
            gates.append(
                dict(
                    gate=f"stored {k}: {what} on forecast rows {c.TREE_START}.. = on all rows",
                    value=ps[j][i],
                    reference=pf[j][i],
                    abs_diff=abs(ps[j][i] - pf[j][i]),
                    passed=bool(abs(ps[j][i] - pf[j][i]) < REPRO_TOL),
                )
            )

    # ---------------------------------------------------------------- arms
    arms = [a for a in c.ARMS if load_arm(a) is not None]
    runs = {a: load_arm(a) for a in arms}
    assert CTRL in runs, "the local control has not run"
    frames = [dvs.research_frame(dz, runs[a]["pred"][sub]) for a in arms]
    # the stored forecasts on the same rows, in the same panel (for local-vs-stored)
    stored_in = [k for k, v in STORED_KEYS.items() if v in runs]
    frames += [dvs.research_frame(dz, stored_pred(k, dz_all)[sub]) for k in stored_in]
    names = arms + [f"stored:{k}" for k in stored_in]
    P = dvs.deck_panel(frames, dk)
    pt = dvs.point(P)
    col = {n: i for i, n in enumerate(names)}
    n_days = P["ql"].shape[0]

    for k in stored_in:
        a = STORED_KEYS[k]
        d = np.abs(runs[a]["pred"][sub] - stored_pred(k, dz_all)[sub])
        cm = compare(P, col[a], col[f"stored:{k}"])
        gates.append(
            dict(
                gate=f"local {a} vs stored {k}: max |forecast difference| (fit space)",
                value=float(d.max()),
                reference=0.0,
                abs_diff=float(d.max()),
                passed=bool(d.max() < REPRO_TOL),
            )
        )
        for what, j in (("QLIKE", "ql"), ("Sharpe mid", "sh")):
            gates.append(
                dict(
                    gate=f"local {a} vs stored {k}: {what} (DM {cm['dm']:.2f}, "
                    + ("HAC t: identical daily P&L" if not np.isfinite(cm["t_hac"]) else f"HAC t {cm['t_hac']:.2f}")
                    + ")",
                    value=pt[j][col[a]],
                    reference=pt[j][col[f"stored:{k}"]],
                    abs_diff=abs(pt[j][col[a]] - pt[j][col[f"stored:{k}"]]),
                    passed=bool(abs(pt[j][col[a]] - pt[j][col[f"stored:{k}"]]) < REPRO_TOL),
                )
            )
    G = pd.DataFrame(gates)
    G.to_csv(c.REPORT / "gates.csv", index=False)

    rows = []
    for a in arms:
        model, src, win = c.ARMS[a]
        r = runs[a]
        wt, sess, lo, hi = window_text(a, r)
        cm = compare(P, col[a], col[CTRL])
        i = col[a]
        rows.append(
            dict(
                arm=a,
                model=LABEL[model],
                family="tree" if model in c.TREES else "linear",
                rows=SOURCE_TEXT[src],
                source=src,
                window=wt,
                window_sessions=sess,
                train_rows_min=lo,
                train_rows_max=hi,
                fits=len(r["fit_sec"]) if model in c.TREES else int(r["n_solves"]),
                leaf_min=(
                    f"{r['leaf_min'].min():g}..{r['leaf_min'].max():g}"
                    if model in c.TREES and r["leaf_min"].min() != r["leaf_min"].max()
                    else (f"{r['leaf_min'][0]:.4g}" if model in c.TREES else "")
                ),
                qlike=pt["ql"][i],
                sharpe_mid=pt["sh"][i],
                sharpe_crossed=pt["shx"][i],
                pct_buy=100.0 * P["buy"][i],
                d_qlike_vs_ctrl=pt["ql"][i] - pt["ql"][col[CTRL]],
                dm_vs_ctrl=cm["dm"],
                dm_p_vs_ctrl=cm["dm_p"],
                d_sharpe_vs_ctrl=pt["sh"][i] - pt["sh"][col[CTRL]],
                t_hac_vs_ctrl=cm["t_hac"],
                hac_lag=cm["hac_lag"],
                cpu_min=r["meta"]["cpu_sec"] / 60.0,
                fit_sec_mean=float(np.mean(r["fit_sec"])) if model in c.TREES else np.nan,
                n_days=n_days,
            )
        )
    A = pd.DataFrame(rows)
    A.to_csv(c.REPORT / "arms.csv", index=False)

    # ---------------------------------------------------------------- vs the 2000-session arms
    vr = []
    for a in arms:
        model, src, win = c.ARMS[a]
        for ref in (f"{model}_bar1600_w2000", f"{model}_h16_w2000"):
            if ref == a or ref not in runs:
                continue
            cm = compare(P, col[a], col[ref])
            vr.append(
                dict(
                    arm=a,
                    reference=ref,
                    model=LABEL[model],
                    d_qlike=pt["ql"][col[a]] - pt["ql"][col[ref]],
                    pct_qlike=100.0 * (pt["ql"][col[a]] / pt["ql"][col[ref]] - 1.0),
                    dm=cm["dm"],
                    dm_p=cm["dm_p"],
                    d_sharpe_mid=pt["sh"][col[a]] - pt["sh"][col[ref]],
                    t_hac=cm["t_hac"],
                )
            )
    V = pd.DataFrame(vr)
    V.to_csv(c.REPORT / "vs_reference.csv", index=False)

    # ---------------------------------------------------------------- trees minus linear
    asl = c_asl()
    dd = []
    steps = [("pool_w4000", "h16_w2000"), ("pool_w4000", "bar1600_w2000")]
    steps += [(f"h16_w{w}", "h16_w2000") for w in (500, 1000, 3000, "exp")]
    for tree in c.TREES:
        for lin in c.LINEAR:
            for new, old in steps:
                arms4 = [f"{tree}_{new}", f"{tree}_{old}", f"{lin}_{new}", f"{lin}_{old}"]
                if not all(x in runs for x in arms4):
                    continue
                tn, to, ln, lo_ = (col[x] for x in arms4)
                dq_t = P["ql"][:, tn] - P["ql"][:, to]
                dq_l = P["ql"][:, ln] - P["ql"][:, lo_]
                dp_t = P["pnl"][:, tn] - P["pnl"][:, to]
                dp_l = P["pnl"][:, ln] - P["pnl"][:, lo_]
                dm = dm_test(dq_t, dq_l)
                t, lag = asl.newey_west_t(dp_t - dp_l)
                dd.append(
                    dict(
                        tree=LABEL[tree],
                        linear=LABEL[lin],
                        change=f"{old} -> {new}",
                        d_qlike_tree=dq_t.mean(),
                        d_qlike_linear=dq_l.mean(),
                        did_qlike=dq_t.mean() - dq_l.mean(),
                        dm=dm["dm"],
                        dm_p=dm["p"],
                        d_sharpe_tree=pt["sh"][tn] - pt["sh"][to],
                        d_sharpe_linear=pt["sh"][ln] - pt["sh"][lo_],
                        t_hac_pnl=t,
                    )
                )
    D = pd.DataFrame(dd)
    D.to_csv(c.REPORT / "trees_vs_linear.csv", index=False)

    # ---------------------------------------------------------------- tree minus linear on the same rows
    tl = []
    for tree in c.TREES:
        for lin in c.LINEAR:
            for a in arms:
                m, src, win = c.ARMS[a]
                if m != tree:
                    continue
                b = a.replace(f"{tree}_", f"{lin}_", 1)
                if b not in runs:
                    continue
                cm = compare(P, col[a], col[b])
                tl.append(
                    dict(
                        tree_arm=a,
                        linear_arm=b,
                        rows=SOURCE_TEXT[src],
                        window=window_text(a, runs[a])[0],
                        qlike_tree=pt["ql"][col[a]],
                        qlike_linear=pt["ql"][col[b]],
                        d_qlike=pt["ql"][col[a]] - pt["ql"][col[b]],
                        dm=cm["dm"],
                        dm_p=cm["dm_p"],
                        sharpe_tree=pt["sh"][col[a]],
                        sharpe_linear=pt["sh"][col[b]],
                        d_sharpe_mid=pt["sh"][col[a]] - pt["sh"][col[b]],
                        t_hac=cm["t_hac"],
                    )
                )
    T = pd.DataFrame(tl)
    T.to_csv(c.REPORT / "tree_minus_linear.csv", index=False)

    # ---------------------------------------------------------------- learning curve
    cur = A[["arm", "model", "source", "window", "window_sessions", "train_rows_min", "train_rows_max", "qlike", "sharpe_mid"]].copy()
    # the expanding window plotted at its mean training size over the trade days' refits
    trade_mean = {}
    for a in arms:
        _, src, win = c.ARMS[a]
        if win == "exp":
            trade_mean[a] = float(np.mean(runs[a]["n_train"]))
    cur["x_sessions"] = [trade_mean.get(a, s) for a, s in zip(cur["arm"], cur["window_sessions"])]
    cur.to_csv(c.REPORT / "curve.csv", index=False)
    plot_curve(cur)
    write_summary(G, A, V, D, cur, T)
    print(A[["arm", "qlike", "sharpe_mid", "dm_vs_ctrl", "t_hac_vs_ctrl", "cpu_min"]].to_string(index=False))


def plot_curve(cur: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    color = {"LightGBM": dvs.COLOR["lgbm"], "XGBoost": dvs.COLOR["xgb"], "random forest": dvs.COLOR["rf"], "ridge": dvs.COLOR["ridge"], "lasso": dvs.COLOR["reclasso"]}
    dash = {"LightGBM": dvs.DASH["lgbm"], "XGBoost": dvs.DASH["xgb"], "random forest": dvs.DASH["rf"], "ridge": dvs.DASH["ridge"], "lasso": dvs.DASH["reclasso"]}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    for ax, ycol, ylab in ((axes[0], "qlike", "QLIKE (866 trade days)"), (axes[1], "sharpe_mid", "sign(s) Sharpe, mid")):
        for m in ("LightGBM", "XGBoost", "random forest", "lasso", "ridge"):
            h = cur[(cur["model"] == m) & (cur["source"] == "h16")].sort_values("x_sessions")
            if len(h):
                ax.plot(h["x_sessions"], h[ycol], color=color[m], linestyle=dash[m], linewidth=2, marker="o", markersize=6, label=f"{m}, 16:00 rows")
            b = cur[(cur["model"] == m) & (cur["source"] == "bar1600")]
            if len(b):
                ax.plot(b["x_sessions"], b[ycol], linestyle="none", marker="s", markersize=8, markerfacecolor="white", markeredgecolor=color[m], markeredgewidth=2, label=f"{m}, one-bar design")
            p = cur[(cur["model"] == m) & (cur["source"] == "pool")]
            if len(p):
                ax.plot(p["x_sessions"], p[ycol], linestyle="none", marker="*", markersize=13, color=color[m], markeredgecolor="white", label=f"{m}, both last-hour bars")
        ax.set_xscale("log")
        ex = cur.loc[cur["window"].str.startswith("expanding"), "x_sessions"]
        ticks, labels = [500, 1000, 2000, 3000], ["500", "1000", "2000", "3000"]
        if len(ex):
            ticks.append(float(ex.iloc[0]))
            labels.append(f"expanding\n(mean {ex.iloc[0]:.0f})")
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        ax.minorticks_off()
        ax.set_xlim(420, 6200)
        ax.set_xlabel("training sessions")
        ax.set_ylabel(ylab)
        ax.grid(True, color="#d9d9d9", linewidth=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle("16:00 forecast, all features: training sessions vs accuracy and P&L", fontsize=11)
    fig.savefig(c.REPORT / "learning_curve.png", dpi=150)
    plt.close(fig)


def _f(v: float, nd: int = 4) -> str:
    return "" if pd.isna(v) else f"{v:.{nd}f}"


def write_summary(G, A, V, D, cur, T) -> None:  # noqa: C901 - one linear report
    ctrl = A.set_index("arm").loc[CTRL]
    L = []
    L.append("# Trees and training-set size at the 16:00 bar (close study, 2026-10-03)")
    L.append("")
    L.append(
        "Question (the user's note): *not enough training data for trees? -> last hour (2 x 30-min bars)?* "
        "Do the 16:00-bar trees lose to the linear models because 2000 rows (one a session) is too little data for them? "
        "Written by `experiments/close_trees_datasize_analyze.py` from the CSVs in this folder; every number below is in them."
    )
    L.append("")
    L.append("## Setup")
    L.append(
        "- Design: the captured design of the 16:00 arms (`experiments/capture_design_close.py`, all features). "
        "`bar1600` = the shipped one-bar design (one row a session, 2010-07 on); `last30` = both last-hour bars (ending 15:30 and 16:00), "
        "whose 16:00 rows reach back to 2004-04-13 and carry the same target as `bar1600` (asserted) but a different rolling scaling of 279 of 628 columns."
    )
    L.append(
        "- Arms: *pool* = train on both last-hour bars, 2000 sessions = 4000 rows, forecast the 16:00 rows (the same day's 15:30 row may enter the fit; nothing later); "
        "*h16* = the 16:00 rows of `last30`, windows of 500 / 1000 / 2000 / 3000 sessions and an expanding window."
    )
    L.append(
        f"- Trees: the tree spec's shipped configs, refit every {c.REFIT_EVERY} sessions, single-threaded, window mask on; the leaf minimum scaled to the fit's rows by the spec's rule "
        f"(LightGBM `min_child_samples` = round(98 x rows / 24000): 8 at 2000 rows, 16 at 4000). Tree forecasts start at forecast row {c.TREE_START} (2019-01) so that every refit anchor is the stored run's; "
        "the recalibration window is full on the first trade day (gate below)."
    )
    L.append(
        "- Linear control: the linear spec's ridge and lasso (warm updates every row, penalty re-chosen every 250 solves), on the same rows and windows; on *pool* they solve at every row, 15:30 rows included (not scored)."
    )
    L.append(
        "- Scorer: the master table's research convention (16:00 recalibration over the previous 250 sessions, the 15:30 sign(s) straddle trade), 866 trade days 2020-01-03 .. 2024-04-30. "
        "The circular block bootstrap is off (commit b761b28): point estimates; differences carry a Diebold-Mariano statistic on daily QLIKE (negative = the first forecast has lower loss) "
        "and a HAC t on the paired daily P&L difference (positive = the first forecast earns more)."
    )
    L.append("")
    L.append("## Reading (numbers from the tables below)")
    Ai = A.set_index("arm")
    mods = [m for m in ("lgbm", "xgb", "rf", "lasso", "ridge") if f"{m}_bar1600_w2000" in Ai.index]
    for m in mods:
        pool, h2, b2 = f"{m}_pool_w4000", f"{m}_h16_w2000", f"{m}_bar1600_w2000"
        if pool in Ai.index and h2 in Ai.index:
            v = V[(V["arm"] == pool) & (V["reference"] == h2)].iloc[0]
            L.append(
                f"- {LABEL[m]}, both last-hour bars (4000 rows) vs 2000 sessions of 16:00 rows on the same scaling: QLIKE {Ai.loc[h2, 'qlike']:.4f} -> "
                f"{Ai.loc[pool, 'qlike']:.4f} (DM {v['dm']:.2f}), Sharpe mid {Ai.loc[h2, 'sharpe_mid']:.2f} -> {Ai.loc[pool, 'sharpe_mid']:.2f} (HAC t {v['t_hac']:.2f})."
            )
    for m in mods:
        ws = [w for w in (500, 1000, 2000, 3000, "exp") if f"{m}_h16_w{w}" in Ai.index]
        if len(ws) > 1:
            L.append(
                f"- {LABEL[m]}, learning curve on the 16:00 rows ({' / '.join('expanding' if w == 'exp' else str(w) for w in ws)} sessions): QLIKE "
                + " / ".join(f"{Ai.loc[f'{m}_h16_w{w}', 'qlike']:.4f}" for w in ws)
                + "; Sharpe mid "
                + " / ".join(f"{Ai.loc[f'{m}_h16_w{w}', 'sharpe_mid']:.2f}" for w in ws)
                + "."
            )
    for m in mods:
        h2, b2 = f"{m}_h16_w2000", f"{m}_bar1600_w2000"
        if h2 in Ai.index:
            v = V[(V["arm"] == h2) & (V["reference"] == b2)].iloc[0]
            L.append(
                f"- {LABEL[m]}, same 2000 sessions, one-bar design vs the last30 design's 16:00 rows (only the scaling of 279 columns differs): QLIKE "
                f"{Ai.loc[b2, 'qlike']:.4f} vs {Ai.loc[h2, 'qlike']:.4f} (DM {v['dm']:.2f}), Sharpe mid {Ai.loc[b2, 'sharpe_mid']:.2f} vs {Ai.loc[h2, 'sharpe_mid']:.2f} (HAC t {v['t_hac']:.2f})."
            )
    L.append("")
    L.append("## Gates")
    for _, g in G.iterrows():
        L.append(f"- {g['gate']}: {g['value']:.6g} vs {g['reference']:.6g} (|diff| {g['abs_diff']:.2e}) {'PASS' if g['passed'] else 'differs'}")
    L.append(
        "- Library versions here: " + ", ".join(f"{k} {v}" for k, v in json.loads(json.dumps(load_arm(CTRL)["meta"]["versions"])).items())
        + " (the stored campaign ran LightGBM 4.6.0 / XGBoost 3.2.0). Every arm below is compared with the LOCAL control."
    )
    L.append("")
    L.append(f"## Every arm against the local LightGBM control (`{CTRL}`: QLIKE {ctrl['qlike']:.4f}, Sharpe mid {ctrl['sharpe_mid']:.2f})")
    L.append("")
    L.append("| arm | model | rows | window | QLIKE | Sharpe mid | Sharpe crossed | DM vs control | HAC t vs control | CPU min |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in A.iterrows():
        dm = "" if r["arm"] == CTRL else f"{r['dm_vs_ctrl']:.2f}"
        th = "" if r["arm"] == CTRL else f"{r['t_hac_vs_ctrl']:.2f}"
        L.append(
            f"| `{r['arm']}` | {r['model']} | {r['rows']} | {r['window']} | {r['qlike']:.4f} | {r['sharpe_mid']:.2f} | {r['sharpe_crossed']:.2f} | {dm} | {th} | {r['cpu_min']:.1f} |"
        )
    L.append("")
    if len(V):
        L.append("## Each arm against the same model with 2000 sessions of 16:00 rows")
        L.append("")
        L.append("`bar1600_w2000` = the shipped one-bar design; `h16_w2000` = the 16:00 rows of the last30 design (same scaling as *pool* and the other *h16* windows).")
        L.append("")
        L.append("| arm | reference | dQLIKE | dQLIKE % | DM | dSharpe mid | HAC t |")
        L.append("|---|---|---|---|---|---|---|")
        for _, r in V.iterrows():
            L.append(
                f"| `{r['arm']}` | `{r['reference']}` | {r['d_qlike']:+.4f} | {r['pct_qlike']:+.1f} | {r['dm']:.2f} | {r['d_sharpe_mid']:+.2f} | {r['t_hac']:.2f} |"
            )
        L.append("")
    if len(T):
        L.append("## Tree minus linear on the same rows and window")
        L.append("")
        L.append("dQLIKE < 0 and DM < 0: the tree has the lower loss; dSharpe > 0 and HAC t > 0: the tree earns more.")
        L.append("")
        L.append("| tree arm | linear arm | rows | window | QLIKE tree | QLIKE linear | dQLIKE | DM | Sharpe tree | Sharpe linear | HAC t |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for _, r in T.iterrows():
            L.append(
                f"| `{r['tree_arm']}` | `{r['linear_arm']}` | {r['rows']} | {r['window']} | {r['qlike_tree']:.4f} | {r['qlike_linear']:.4f} | {r['d_qlike']:+.4f} | "
                f"{r['dm']:.2f} | {r['sharpe_tree']:.2f} | {r['sharpe_linear']:.2f} | {r['t_hac']:.2f} |"
            )
        L.append("")
    if len(D):
        L.append("## Does more data help the trees more than the linear models? (difference in differences)")
        L.append("")
        L.append(
            "dQLIKE tree = the tree's QLIKE change from the reference window to the new rows; the same for the linear model; "
            "DiD = tree change minus linear change (negative = more rows help the tree more). DM is on the daily series of differences; HAC t on the daily P&L series of differences "
            "(positive = more rows raise the tree's P&L more than the linear model's)."
        )
        L.append("")
        L.append("| tree | linear | change | dQLIKE tree | dQLIKE linear | DiD | DM | dSharpe tree | dSharpe linear | HAC t |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for _, r in D.iterrows():
            L.append(
                f"| {r['tree']} | {r['linear']} | {r['change']} | {r['d_qlike_tree']:+.4f} | {r['d_qlike_linear']:+.4f} | {r['did_qlike']:+.4f} | {r['dm']:.2f} | "
                f"{r['d_sharpe_tree']:+.2f} | {r['d_sharpe_linear']:+.2f} | {r['t_hac_pnl']:.2f} |"
            )
        L.append("")
    L.append("## Learning curve (16:00 rows of the last30 design)")
    L.append("")
    L.append("![learning curve](learning_curve.png)")
    L.append("")
    h = cur[cur["source"] == "h16"].copy()
    if len(h):
        piv_q = h.pivot_table(index="window", columns="model", values="qlike", aggfunc="first")
        piv_s = h.pivot_table(index="window", columns="model", values="sharpe_mid", aggfunc="first")
        order = sorted(piv_q.index, key=lambda w: h.loc[h["window"] == w, "x_sessions"].iloc[0])
        mods = [m for m in ("LightGBM", "XGBoost", "random forest", "lasso", "ridge") if m in piv_q.columns]
        L.append("| window | " + " | ".join(f"{m} QLIKE" for m in mods) + " | " + " | ".join(f"{m} Sharpe mid" for m in mods) + " |")
        L.append("|---|" + "---|" * (2 * len(mods)))
        for w in order:
            L.append(
                f"| {w} | "
                + " | ".join(_f(piv_q.loc[w, m]) for m in mods)
                + " | "
                + " | ".join(_f(piv_s.loc[w, m], 2) for m in mods)
                + " |"
            )
        L.append("")
    L.append("## Not run")
    done = set(A["arm"])
    missing = [a for a in c.ORDER if a not in done]
    L.append(
        "- Arms of the plan not run in this container's CPU budget: "
        + (", ".join(f"`{a}`" for a in missing) if missing else "none")
        + ". The `closing` segment (5 bars, 14:00-16:00) was not built."
    )
    tf = c.REPO / "cluster" / "close_trees_datasize_h2_tasks.txt"
    if tf.is_file():
        tasks = pd.read_csv(tf, sep=" ", header=None, names=["arm", "refit", "chunk", "n"])
        g = tasks.groupby(["refit", "arm"], sort=False).size().reset_index()
        L.append(
            f"- Hoffman2 version (written, smoke-tested locally at tiny size, NOT submitted): `cluster/close_trees_datasize_h2_task.sh` + "
            f"`cluster/submit_close_trees_datasize_h2.sh`, {len(tasks)} single-slot tasks in `cluster/close_trees_datasize_h2_tasks.txt` "
            f"(each arm cut into {tasks['n'].iloc[0]} time chunks, joined by `merge`): "
            + "; ".join(
                f"refit every {r} session{'s' if r > 1 else ''}: " + ", ".join(f"`{x}`" for x in g.loc[g['refit'] == r, 'arm'])
                for r in g["refit"].unique()
            )
            + f". The local control's LightGBM fit took {ctrl['fit_sec_mean']:.1f} s on one core here, so one arm refit every session "
            f"({c.sources()['n_fc'] - c.TREE_START} fits) is about {ctrl['fit_sec_mean'] * (c.sources()['n_fc'] - c.TREE_START) / 3600:.1f} CPU-hours."
        )
    L.append("")
    L.append("## Files")
    L.append("- `experiments/close_trees_datasize.py` (run stage), `experiments/close_trees_datasize_analyze.py` (this report)")
    L.append("- `arms.csv`, `vs_reference.csv`, `tree_minus_linear.csv`, `trees_vs_linear.csv`, `curve.csv`, `gates.csv`, `learning_curve.png`; forecasts in `_work/<arm>.npz` (not committed)")
    L.append(f"- CPU: {A['cpu_min'].sum():.0f} min over {len(A)} arms, one single-threaded process at a time.")
    (c.REPORT / "SUMMARY.md").write_text("\n".join(L) + "\n", encoding="utf-8")
