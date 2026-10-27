"""Gates, scores, readings and figures of the window x feature-count grid (RULE.txt ADDENDUM 1), on the laptop.

Inputs: --work-grid <dir> (the fetched CARC grid work: w<W>/{rank_*.json, trees/, linear/, design.json, dates_oos.csv}
for W = 1,000 / 2,000 / 3,500) and --work-ladder <dir> (the ladder's W = 3,000 work: its primary rungs).
Outputs (results/ladder_2026-10-26/grid/): gates.json, scores.csv, readings.csv, ranks.csv, tables.txt,
fig_complexity.(png|pdf) (a), fig_data_scaling.(png|pdf) (b), fig_feature_scaling.(png|pdf) (c).

usage: python results/ladder_2026-10-26/grid/code/grid_score.py --work-grid D1 --work-ladder D2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
GRID = HERE.parent
LADDER = GRID.parent
ROOT = LADDER.parents[1]
for _p in (str(LADDER / "code"), str(ROOT), str(ROOT / "notebooks")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ladder_common as C  # noqa: E402
import ladder_score as LS  # noqa: E402
from src.evaluation.diebold_mariano import dm_test  # noqa: E402

WS = (1000, 2000, 3000, 3500)
MODELS = ("lgbm", "xgb", "ridge", "lasso", "enet")
KS = (0, 1, 3, 5, 10, 20, 46)
RID = {46: "full", 0: "har", 20: "P20", 10: "P10", 5: "P5", 3: "P3", 1: "P1"}
NAME = {"lgbm": "LightGBM", "xgb": "XGBoost", "ridge": "Ridge", "lasso": "Lasso", "enet": "Elastic net"}
LIN_T = ROOT / "results" / "window_sweep_2014start_2026-10-22" / "w{W}" / "forecasts.parquet"
TREE_T = {
    1000: ROOT / "results" / "pooled_trees_lc_2014start_w1000" / "forecasts.parquet",
    2000: ROOT / "results" / "pooled_trees_lc_2014start_w2000" / "forecasts.parquet",
    3000: ROOT / "results" / "pooled_trees_final_2014start_w3000" / "forecasts.parquet",
    3500: ROOT / "results" / "pooled_trees_final_2014start_w3500" / "forecasts.parquet",
}

LABEL = "PRE-PATCH design: the FOMC calendar features are those before the pending FOMC fix."
LABEL_SHORT = "PRE-PATCH design (FOMC features before the pending fix)."


def say(msg: str) -> None:
    print(msg, flush=True)


def origins_w(dd: pd.DatetimeIndex, wd: int) -> pd.DataFrame:
    o = pd.read_csv(LADDER / "code" / "fiw_origins.csv")
    o["oos_row"] = o[f"oos_row_w{wd}"].astype(int)
    o["r"] = wd * 13 + o["oos_row"]
    o["r_end"] = list(o["r"].iloc[1:]) + [C.N_ROWS]
    assert (pd.DatetimeIndex(dd[o["oos_row"]]) == pd.to_datetime(o["origin"])).all()
    assert (o["r"] == 39000 + o["oos_row_w3000"]).all() and (o["r_end"] - o["r"]).sum() == 33970
    return o


def load_cell(work: Path, wd: int, m: str, o: pd.DataFrame, tag: str = "") -> tuple[dict[int, np.ndarray], dict, dict]:
    """{k: pred_adj over the 33,970 rows}, the sets file, and gate inputs."""
    out: dict[int, np.ndarray] = {}
    if m in ("lgbm", "xgb"):
        d = work / "trees" / m
        sets: dict = {}
        for f in sorted(d.glob(f"sets_{tag}_part*.json" if tag else "sets_part*.json")):
            s = json.loads(f.read_text())
            sets = s if not sets else sets
            assert s["sets"] == sets["sets"] or wd == 3000, f
        parts: dict[str, list] = {rid: [] for rid in RID.values()}
        p0 = []
        for row in o.itertuples():
            z = np.load(d / f"k{int(row.k):03d}.npz")
            for k, rid in RID.items():
                fitted = sets["alias"][rid]
                parts[rid].append(z[f"pred_{fitted}"])
            p0.append(float(z[f"p0_{sets['alias']['full']}"]))
        for k, rid in RID.items():
            out[k] = np.concatenate(parts[rid])
        return out, sets, {"p0_full": np.array(p0)}
    d = work / "linear" / m
    sets = json.loads((d / (f"sets_{tag}.json" if tag else "sets.json")).read_text())
    rep, gs = [], []
    for k, rid in RID.items():
        fitted = sets["alias"][rid]
        arr = []
        for row in o.itertuples():
            z = np.load(d / fitted / f"k{int(row.k):03d}.npz")
            arr.append(z["pred"])
            if rid == "full":
                rep.append(z["replay"])
                gs.append(int(z["g"]))
        out[k] = np.concatenate(arr)
    return out, sets, {"replay": rep, "g": gs}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-grid", required=True)
    ap.add_argument("--work-ladder", required=True)
    # 2026-10-26 (RULE.txt addendum 3, the FOMC-patched rerun; every default = the pre-patch run unchanged)
    ap.add_argument("--sets-tag", default="", help="pri: the patched runner's sets files")
    ap.add_argument("--out", default="")
    ap.add_argument("--walk-lin-main", default="", help="patched linear walks (W = 3,000) for GL1")
    ap.add_argument("--walk-trees-main", default="", help="patched tree walks (W = 3,000) for GT")
    ap.add_argument("--prepatch-grid", default="", help="pre-patch grid work (W 1,000 / 2,000 / 3,500) for gate GP")
    ap.add_argument("--prepatch-ladder", default="", help="pre-patch ladder work (W = 3,000) for gate GP")
    a = ap.parse_args()
    global GRID, LABEL, LABEL_SHORT
    if a.out:
        GRID = Path(a.out)
        GRID.mkdir(parents=True, exist_ok=True)
    patched = bool(a.sets_tag)
    if patched:
        LABEL = LABEL_SHORT = "PATCHED design (FOMC fix adopted: FOMC_SCHEDULE_2020_FIX, FOMC_PROXIMITY_NO_WINSOR)."
    rv = LS.core_rv()
    days = LS.evaluation_days(rv)
    panel: dict[tuple[int, str, int], pd.Series] = {}
    gates: dict = {}
    ranks = []
    for wd in WS:
        work = Path(a.work_ladder) if wd == 3000 else Path(a.work_grid) / f"w{wd}"
        dd = pd.DatetimeIndex(pd.read_csv(work / "dates_oos.csv", parse_dates=["date"])["date"])
        assert len(dd) == C.N_ROWS - 13 * wd
        o = origins_w(dd, wd)
        rows_dates = dd[C.R_FIRST - 13 * wd :]
        assert len(rows_dates) == 33970 and rows_dates[0] == pd.Timestamp("2014-01-02 10:00")
        lt = pd.read_parquet(str(LIN_T).format(W=wd), columns=["model", "predictor_set", "date", "pred_adj", "B"])
        B = lt[(lt.model == "ridge") & (lt.predictor_set == "all_features")].set_index("date")["B"].reindex(rows_dates)
        assert np.isfinite(B.to_numpy()).all()
        tt = pd.read_parquet(TREE_T[wd])
        tb = tt[(tt.model == "lgbm") & (tt.predictor_set == "all_features")].set_index("date")["B"]
        assert np.array_equal(tb.reindex(rows_dates).to_numpy(), B.to_numpy()), f"tree / linear B differ at W {wd}"
        nm = list(json.loads((work / "design.json").read_text())["names"])
        tie = C.series_list(nm)
        for m in MODELS:
            pr, sets, gi = load_cell(work, wd, m, o, a.sets_tag)
            gg: dict = {}
            if patched and not (wd == 3000):  # no patched walk at this window: gate D (design) and GP only
                gg = {"walk": "none at this window", "pass": True}
                fam = m if m in ("lgbm", "xgb") else "linear"
            elif m in ("lgbm", "xgb"):
                src = pd.read_parquet(a.walk_trees_main) if patched else tt
                walk = src[(src.model == m) & (src.predictor_set == "all_features")].set_index("date")["pred_adj"]
                ref = walk.reindex(pd.DatetimeIndex(o["origin"])).to_numpy()
                gg = dict(
                    origin_bit_equal=int((gi["p0_full"] == ref).sum()),
                    n=len(ref),
                    maxabs=float(np.abs(gi["p0_full"] - ref).max()),
                )
                gg["pass"] = gg["origin_bit_equal"] == 126
                fam = m
            else:
                src = pd.read_parquet(a.walk_lin_main) if patched else lt
                walk = src[(src.model == m) & (src.predictor_set == "all_features")].set_index("date")["pred_adj"]
                mx, n = 0.0, 0
                for g0, rep, i in zip(gi["g"], gi["replay"], o["oos_row"]):
                    w_ = walk.reindex(dd[g0 : int(i) + 1]).to_numpy()
                    mx, n = max(mx, float(np.abs(rep - w_).max())), n + len(rep)
                gg = dict(rows=n, maxabs=mx, **{"pass": bool(mx <= 1e-12)})
                fam = "linear"
            if patched:  # gate GP: rungs without the FOMC proximity series = their pre-patch twins, bit for bit
                pw = Path(a.prepatch_ladder) if wd == 3000 else Path(a.prepatch_grid) / f"w{wd}"
                ppr, psets, _ = load_cell(pw, wd, m, o)
                twin = {frozenset(psets["sets"][RID[k]]): k for k in KS if k != 46}
                n_cmp, n_eq, bad = 0, 0, []
                for k in KS:
                    ss = frozenset(sets["sets"][RID[k]])
                    if k == 46 or ss & {"fomc_until_inv", "fomc_since_inv"} or ss not in twin:
                        continue
                    n_cmp += 1
                    if np.array_equal(pr[k], ppr[twin[ss]]):
                        n_eq += 1
                    else:
                        bad.append(k)
                gg.update(GP_compared=n_cmp, GP_bit_equal=n_eq, GP_differ=bad)
                gg["pass"] = bool(gg["pass"] and not bad)
            gates[f"w{wd}/{m}"] = gg
            imp = json.loads((work / f"rank_{fam}.json").read_text())["importance"][m]
            for r_, s in enumerate(C.order_from_importance(imp, tie), 1):
                ranks.append(dict(wdays=wd, model=m, rank=r_, series=s, importance=imp[s]))
            for k in KS:
                panel[(wd, m, k)] = pd.Series(pr[k] ** 2 * B.to_numpy(), index=rows_dates)
    (GRID / "gates.json").write_text(json.dumps(gates, indent=1))
    pd.DataFrame(ranks).to_csv(GRID / "ranks.csv", index=False)
    say("gates: " + json.dumps({k: v["pass"] for k, v in gates.items()}))

    idx = None
    for f in panel.values():
        ok = f.index[np.isfinite(f.to_numpy()) & (f.to_numpy() > 0)]
        idx = ok if idx is None else idx.intersection(ok)
    assert idx is not None
    idx = idx[pd.DatetimeIndex(idx.normalize()).isin(days)]
    t = rv.reindex(idx)
    idx = idx[(t.to_numpy() > 0) & np.isfinite(t.to_numpy())].sort_values()
    assert len(idx) == LS.N_BARS, len(idx)
    spans = {"all": idx, "c16": idx[(idx.hour == 16) & (idx.minute == 0)]}
    spans["c16_2020"] = spans["c16"][spans["c16"].normalize() >= LS.SUB_FIRST]
    yv = {sp: rv.reindex(ix).to_numpy(float) for sp, ix in spans.items()}

    def ql(f: pd.Series, sp: str) -> np.ndarray:
        r = yv[sp] / f.reindex(spans[sp]).to_numpy(float)
        return r - np.log(r) - 1.0

    def daily(x: np.ndarray, sp: str) -> np.ndarray:
        return pd.Series(x, index=pd.DatetimeIndex(spans[sp].normalize())).groupby(level=0).mean().to_numpy()

    rows = []
    for (wd, m, k), f in panel.items():
        for sp in spans:
            q = ql(f, sp)
            q0 = ql(panel[(wd, m, 0)], sp)
            if k:
                d = dm_test(daily(q, sp), daily(q0, sp)) if sp == "all" else dm_test(q, q0)
            else:
                d = {"dm": np.nan, "p": np.nan, "hac_lag": -1}
            P = 12 + 12 * k
            T = 13 * wd
            rows.append(
                dict(
                    wdays=wd,
                    T=T,
                    model=m,
                    k=k,
                    P=P,
                    c=P / T,
                    span=sp,
                    qlike=float(q.mean()),
                    gain=float(q0.mean() - q.mean()),
                    dm_vs_k0=float(d["dm"]),
                    p_vs_k0=float(d["p"]),
                    hac_lag=int(d.get("hac_lag", -1)),
                )
            )
    S = pd.DataFrame(rows).sort_values(["model", "span", "wdays", "k"]).reset_index(drop=True)
    full = S[S.k == 46].set_index(["model", "span", "wdays"])["gain"]
    S["gain_pct_of_full"] = [
        g / full[(m, sp, w)] * 100.0 if full[(m, sp, w)] > 0 else np.nan
        for g, m, sp, w in zip(S.gain, S.model, S.span, S.wdays)
    ]
    S.to_csv(GRID / "scores.csv", index=False)

    def seq_label(v: list[int]) -> str:
        dif = np.diff(v)
        if (dif == 0).all():
            return "FLAT"
        if (dif >= 0).all():
            return "RISES"
        if (dif <= 0).all():
            return "FALLS"
        return "MIXED"

    rd = []
    for m in MODELS:
        for sp in ("all", "c16"):
            f = S[(S.model == m) & (S.span == sp)]
            kstar, k90, slopes, full_best = [], [], [], 0
            for wd in WS:
                g = f[f.wdays == wd].set_index("k")
                kb = int(g["qlike"].idxmin())
                kstar.append(kb)
                full_best += int(kb == 46)
                pc = g["gain_pct_of_full"]
                hit = [k for k in KS if k > 0 and pc.get(k, np.nan) >= 90]
                k90.append(min(hit) if hit else 46)
                x = np.log(g.loc[list(KS), "c"].to_numpy())
                slopes.append(float(np.polyfit(x, g.loc[list(KS), "gain"].to_numpy(), 1)[0]))
            rd.append(
                dict(
                    model=m,
                    span=sp,
                    kstar_by_W=" ".join(map(str, kstar)),
                    kstar_reading=seq_label(kstar),
                    k90_by_W=" ".join(map(str, k90)),
                    k90_reading=seq_label(k90),
                    windows_full_best=full_best,
                    slope_gain_logc_by_W=" ".join(f"{s:.5f}" for s in slopes),
                    gain_full_by_W=" ".join(
                        f"{float(f[(f.wdays == w) & (f.k == 46)]['gain'].iloc[0]):.5f}" for w in WS
                    ),
                    gain_best_by_W=" ".join(f"{float(f[f.wdays == w]['gain'].max()):.5f}" for w in WS),
                )
            )
    R = pd.DataFrame(rd)
    R.to_csv(GRID / "readings.csv", index=False)
    lines = [LABEL, ""]
    for sp in ("all", "c16"):
        lines.append(f"QLIKE gain over the same window's k = 0 rung (x 1e3), span {sp}; * p < 0.05 vs k = 0")
        for m in MODELS:
            lines.append(f"  {NAME[m]}")
            lines.append("    W      " + "".join(f"{'k=' + str(k):>10s}" for k in KS[1:]) + "   best k")
            for wd in WS:
                g = S[(S.model == m) & (S.span == sp) & (S.wdays == wd)].set_index("k")
                cells = "".join(
                    f"{g.loc[k, 'gain'] * 1e3:9.2f}{'*' if g.loc[k, 'p_vs_k0'] < 0.05 else ' '}" for k in KS[1:]
                )
                lines.append(f"    {wd:<6d} {cells}   {int(g['qlike'].idxmin())}")
        lines.append("")
    lines.append(R.to_string(index=False))
    lines.append("")
    lines.append("Gates: " + json.dumps(gates))
    (GRID / "tables.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    say("\n".join(lines))
    figures(S)
    return 0


FLOOR = (
    -6.0
)  # gain x 1e3 below which a point is drawn at the axis floor and labelled (off-scale held-forecast blow-ups)


def clipped(ax, x, y, color):
    """Plot y clipped at FLOOR; mark and label the clipped points."""
    import numpy as np

    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    low = y < FLOOR
    for xi, yi in zip(x[low], y[low]):
        ax.plot([xi], [FLOOR], marker="v", ms=6, color=color, zorder=4)
        ax.annotate(
            f"{yi:,.0f}", (xi, FLOOR), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=6, color=color
        )
    return np.where(low, FLOOR, y)


def figures(S: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, INK2, GRIDC = "#0b0b0b", "#52514e", "#e4e3df"
    WCOL = {1000: "#9ec5f0", 2000: "#5c9ee6", 3000: "#2a78d6", 3500: "#0d3f80"}  # one hue, light -> dark (W ordered)
    KCOL = {1: "#c6dbf5", 3: "#9ec5f0", 5: "#6ea8e8", 10: "#3f8ade", 20: "#1f62b8", 46: "#0d3f80"}

    def style(ax) -> None:
        ax.grid(axis="y", color=GRIDC, lw=0.6)
        ax.axhline(0, color=INK2, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(INK2)
        ax.tick_params(colors=INK2, labelsize=8)

    def grid_of(xcol: str, fname: str, xlabel: str) -> None:
        fig, axes = plt.subplots(2, len(MODELS), figsize=(3.0 * len(MODELS), 6.0), squeeze=False)
        for j, m in enumerate(MODELS):
            for i, sp in enumerate(("all", "c16")):
                ax = axes[i, j]
                for wd in WS:
                    g = S[(S.model == m) & (S.span == sp) & (S.wdays == wd)].sort_values("k")
                    ax.plot(
                        g[xcol],
                        clipped(ax, g[xcol], g["gain"] * 1e3, WCOL[wd]),
                        color=WCOL[wd],
                        lw=2,
                        marker="o",
                        ms=4,
                        label=f"W = {wd:,}",
                        markeredgecolor="white",
                        markeredgewidth=0.8,
                    )
                ax.set_xscale("log")
                style(ax)
                if i == 0:
                    ax.set_title(NAME[m], fontsize=10, color=INK)
                if j == 0:
                    ax.set_ylabel(("every bar" if sp == "all" else "16:00 bar") + ": QLIKE gain x 1e3", fontsize=9)
                if i == 1:
                    ax.set_xlabel(xlabel, fontsize=8, color=INK2)
        h, lb = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, lb, loc="upper center", ncol=4, frameon=False, fontsize=9)
        fig.text(
            0.5,
            0.005,
            LABEL_SHORT + " Gain = QLIKE of the same window's k = 0 rung (12 HAR-benchmark columns) minus the rung's; "
            "rungs k = 0, 1, 3, 5, 10, 20, 46 series joined within a window. Monthly refits, 33,527 bars.",
            ha="center",
            fontsize=7.5,
            color=INK2,
        )
        fig.tight_layout(rect=(0, 0.03, 1, 0.94))
        fig.savefig(GRID / f"{fname}.png", dpi=160)
        fig.savefig(GRID / f"{fname}.pdf")
        plt.close(fig)

    grid_of("c", "fig_complexity", "complexity c = P / T (log)")
    grid_of("P", "fig_feature_scaling", "predictors P = 12 + 12k (log)")
    fig, axes = plt.subplots(2, len(MODELS), figsize=(3.0 * len(MODELS), 6.0), squeeze=False)
    for j, m in enumerate(MODELS):
        for i, sp in enumerate(("all", "c16")):
            ax = axes[i, j]
            for k in KS[1:]:
                g = S[(S.model == m) & (S.span == sp) & (S.k == k)].sort_values("wdays")
                ax.plot(
                    g["T"],
                    clipped(ax, g["T"], g["gain"] * 1e3, KCOL[k]),
                    color=KCOL[k],
                    lw=2,
                    marker="o",
                    ms=4,
                    label=f"k = {k}",
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                )
            style(ax)
            ax.set_xticks([13 * w for w in WS])
            ax.set_xticklabels([f"{13 * w // 1000}k" for w in WS])
            if i == 0:
                ax.set_title(NAME[m], fontsize=10, color=INK)
            if j == 0:
                ax.set_ylabel(("every bar" if sp == "all" else "16:00 bar") + ": QLIKE gain x 1e3", fontsize=9)
            if i == 1:
                ax.set_xlabel("window T (rows; W = T / 13 days)", fontsize=8, color=INK2)
    h, lb = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lb, loc="upper center", ncol=6, frameon=False, fontsize=9)
    fig.text(
        0.5,
        0.005,
        LABEL_SHORT + " Gain over the same window's k = 0 rung at fixed k; W = 1,000 / 2,000 / 3,000 / 3,500 days.",
        ha="center",
        fontsize=7.5,
        color=INK2,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(GRID / "fig_data_scaling.png", dpi=160)
    fig.savefig(GRID / "fig_data_scaling.pdf")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
