"""Gates, scores, figure and tables of the feature-count ladder (RULE.txt sections 4-6), on the laptop.

Reads the fetched CARC work directory (--work: rank_*.json, trees/<est>/k###.npz, linear/<model>/<rung>/k###.npz,
dates_oos.csv) and the repository's templates, and writes into results/ladder_2026-10-26/:
  rank_primary.csv, rank_secondary.csv   the two rankings per model (importance, rank)
  gates.json                             GT (trees), GL1 / GL2 (linear)
  scores.csv                             model x ranking x k x span: QLIKE, MSE, DM vs k = 46 and vs k = 0, R_k
  readings.csv                           RULE section 6 per model x ranking x span
  ladder.png, ladder.pdf                 QLIKE vs k per model (every bar; 16:00), full and HAR lines
  tables.txt                             the tables in text form

usage: python results/ladder_2026-10-26/code/ladder_score.py --work <dir> [--models lgbm,xgb,...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE.parent
ROOT = RES.parents[1]
for _p in (str(HERE), str(ROOT), str(ROOT / "notebooks")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ladder_common as C  # noqa: E402
from src.evaluation.diebold_mariano import dm_test  # noqa: E402

FIRST, LAST, N_DAYS, N_BARS = pd.Timestamp("2014-01-02"), pd.Timestamp("2024-04-30"), 2579, 33527
SUB_FIRST, SUB_DAYS = pd.Timestamp("2020-01-03"), 1082
LIN_T = ROOT / "results" / "window_sweep_2014start_2026-10-22" / "w3000" / "forecasts.parquet"
TABLE1 = ROOT / "results" / "oos_final_2014start" / "table1_w3000" / "metrics.csv"
SHAP_GATE = ROOT / "results" / "tree_shap_w3000_2014start_2026-10-23" / "gate.csv"
ABL_WIDE = Path(
    r"C:\Users\james\AppData\Local\Temp\ablate_wt\results\ablations_2014start_w3000\forecasts\linear_wide.parquet"
)
T1_NAME = {"rf_skx": "rf"}
NAME = {
    "lgbm": "LightGBM",
    "xgb": "XGBoost",
    "rf_skx": "Random forest",
    "ridge": "Ridge",
    "lasso": "Lasso",
    "enet": "Elastic net",
}
SPANS = ("all", "c16", "c16_2020")


def say(msg: str) -> None:
    print(msg, flush=True)


def core_rv() -> pd.Series:
    c = pd.read_parquet(ROOT / "data" / "core_stats.parquet", columns=["endbartime", "sumret2"])
    s = pd.Series(c["sumret2"].to_numpy(float), index=pd.DatetimeIndex(pd.to_datetime(c["endbartime"])))
    return s[~s.index.duplicated()].sort_index()


def evaluation_days(rv: pd.Series) -> pd.DatetimeIndex:
    import atm_straddle_lib as asl

    early = pd.DatetimeIndex(pd.to_datetime(list(asl.EARLY_CLOSE_DATES)))
    s = rv[(rv.index.hour == 16) & (rv.index.minute == 0)]
    d = pd.DatetimeIndex(s.index.normalize())
    keep = (s.to_numpy() > 0) & np.isfinite(s.to_numpy()) & ~d.isin(early)
    days = d[keep]
    days = days[(days >= FIRST) & (days <= LAST)]
    assert len(days) == N_DAYS and days[0] == FIRST and days[-1] == LAST, (len(days), days[0], days[-1])
    return days


def origins_from_dates(dd: pd.DatetimeIndex) -> pd.DataFrame:
    """ladder_common.origins on the fetched dates_oos.csv (the same checks)."""
    import hashlib

    f = HERE / "fiw_origins.csv"
    assert hashlib.sha256(f.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == C.ORIGINS_SHA256
    o = pd.read_csv(f)
    assert len(o) == 126 and len(dd) == C.N_ROWS - C.W
    o["oos_row"] = o["oos_row_w3000"].astype(int)
    o["r"] = C.W + o["oos_row"]
    o["r_end"] = list(o["r"].iloc[1:]) + [C.N_ROWS]
    assert (pd.DatetimeIndex(dd[o["oos_row"]]) == pd.to_datetime(o["origin"])).all()
    assert (o["r_end"] - o["r"]).sum() == 33970 and o["r"].iloc[0] == C.R_FIRST
    return o[["k", "origin", "oos_row", "r", "r_end"]].reset_index(drop=True)


def rung_k(rid: str) -> tuple[str, int]:
    """(ranking, k) of a rung id; full / har belong to both rankings."""
    if rid == "full":
        return "both", 46
    if rid == "har":
        return "both", 0
    return rid[0], int(rid[1:])


def load_trees(work: Path, est: str, o: pd.DataFrame, tag: str = "") -> tuple[dict[str, np.ndarray], dict, dict]:
    d = work / "trees" / est
    sets: dict = {}
    for f in sorted(d.glob(f"sets_{tag}_part*.json" if tag else "sets_part*.json")):
        s = json.loads(f.read_text())
        assert not sets or s == sets, f"sets differ across parts of {est}"
        sets = s
    preds: dict[str, list[np.ndarray]] = {rid: [] for rid in sets["fitted"]}
    p0: dict[str, list[float]] = {rid: [] for rid in sets["fitted"]}
    flags: dict = {"shap_pred_equal": [], "shap_p0_equal": [], "n": 0, "cpus": set()}
    for row in o.itertuples():
        z = np.load(d / f"k{int(row.k):03d}.npz")
        assert int(z["r"]) == int(row.r) and int(z["r_end"]) == int(row.r_end)
        for rid in sets["fitted"]:
            preds[rid].append(z[f"pred_{rid}"])
            p0[rid].append(float(z[f"p0_{rid}"]))
        if "shap_pred_equal" in z.files:
            flags["shap_pred_equal"].append(bool(z["shap_pred_equal"]))
            flags["shap_p0_equal"].append(bool(z["shap_p0_equal"]))
        flags["n"] += 1
        flags["cpus"].add(str(z["cpu"]))
    out = {rid: np.concatenate(v) for rid, v in preds.items()}
    return out, sets, {"p0": p0, **flags}


def load_linear(work: Path, m: str, o: pd.DataFrame, tag: str = "") -> tuple[dict[str, np.ndarray], dict, dict]:
    d = work / "linear" / m
    sets = json.loads((d / (f"sets_{tag}.json" if tag else "sets.json")).read_text())
    preds: dict[str, np.ndarray] = {}
    extra: dict = {"replay": [], "g": [], "reseeds": {}}
    for rid in sets["fitted"]:
        parts = []
        rs = [0, 0, 0]
        for row in o.itertuples():
            z = np.load(d / rid / f"k{int(row.k):03d}.npz")
            assert int(z["r"]) == int(row.r) and int(z["r_end"]) == int(row.r_end)
            parts.append(z["pred"])
            rs[0] += int(z["kkt_reseeds"])
            rs[1] += int(z["binom_reseeds"])
            rs[2] += int(z["evid_reseeds"])
            if rid == "full":
                extra["replay"].append(z["replay"])
                extra["g"].append(int(z["g"]))
        preds[rid] = np.concatenate(parts)
        extra["reseeds"][rid] = rs
    return preds, sets, extra


def main() -> int:
    global RES
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--models", default=",".join(C.MODELS))
    ap.add_argument("--out", default="")
    # 2026-10-26 (RULE.txt addendum 3, the FOMC-patched rerun; every default = the pre-patch run unchanged)
    ap.add_argument("--sets-tag", default="", help="pri / sec: the patched runner's sets files")
    ap.add_argument("--secondary-json", default=str(HERE / "secondary_importance.json"), help="or 'none'")
    ap.add_argument("--walk-lin", default="", help="linear walks to gate GL1 against (default: LIN_T)")
    ap.add_argument(
        "--walk-trees", default="", help="comma-separated tree walk parquets for GT (default: SHAP gate.csv)"
    )
    ap.add_argument("--prepatch-wide", default="", help="pre-patch forecasts_wide.parquet for gate GP")
    ap.add_argument("--prepatch-work", default="", help="pre-patch work dir (its sets) for gate GP")
    a = ap.parse_args()
    if a.out:
        RES = Path(a.out)
    RES.mkdir(parents=True, exist_ok=True)
    work = Path(a.work)
    models = [m for m in a.models.split(",") if m]
    dd = pd.DatetimeIndex(pd.read_csv(work / "dates_oos.csv", parse_dates=["date"])["date"])
    o = origins_from_dates(dd)
    rows_dates = dd[C.R_FIRST - C.W :]  # design rows 50,777 .. 84,746
    assert len(rows_dates) == 33970
    lt = pd.read_parquet(LIN_T, columns=["model", "predictor_set", "date", "pred_adj", "B", "plain"])
    B = lt[(lt.model == "ridge") & (lt.predictor_set == "all_features")].set_index("date")["B"]
    b = B.reindex(rows_dates).to_numpy()
    assert np.isfinite(b).all()
    nm_order = list(json.loads((work / "design.json").read_text())["names"])
    tie = C.series_list(nm_order)

    # ---------------------------------------------------------------- rankings
    rk_rows = []
    for m in models:
        fam = "linear" if m in C.LINEAR else m
        imp_p = json.loads((work / f"rank_{fam}.json").read_text())["importance"][m]
        pairs = [("primary", imp_p)]
        if a.secondary_json != "none":
            pairs.append(("secondary", json.loads(Path(a.secondary_json).read_text())[m]))
        for tag, imp in pairs:
            order = C.order_from_importance(imp, tie)
            tot = sum(imp.values())
            for r_, s in enumerate(order, 1):
                rk_rows.append(dict(model=m, ranking=tag, rank=r_, series=s, importance=imp[s], share=imp[s] / tot))
    RK = pd.DataFrame(rk_rows)
    RK[RK.ranking == "primary"].to_csv(RES / "rank_primary.csv", index=False)
    RK[RK.ranking == "secondary"].to_csv(RES / "rank_secondary.csv", index=False)

    # ---------------------------------------------------------------- forecasts and gates
    panel: dict[tuple[str, str], pd.Series] = {}
    sets_of: dict[str, dict] = {}
    gates: dict = {}
    wide: dict[str, np.ndarray] = {}
    gsh = pd.read_csv(SHAP_GATE)
    abl = pd.read_parquet(ABL_WIDE) if (ABL_WIDE.is_file() and not a.walk_lin) else None
    wt = pd.concat([pd.read_parquet(f) for f in a.walk_trees.split(",")], ignore_index=True) if a.walk_trees else None
    lwalk = pd.read_parquet(a.walk_lin) if a.walk_lin else lt
    pre_w = pd.read_parquet(a.prepatch_wide).set_index("date") if a.prepatch_wide else None
    for m in models:
        if m in C.TREES:
            pr, sets, fl = load_trees(work, m, o, a.sets_tag)
            if wt is not None:  # the patched every-bar walk at the origin rows
                ww = wt[(wt.model == m) & (wt.predictor_set == "all_features")].set_index("date")["pred_adj"]
                ref = ww.reindex(pd.DatetimeIndex(o["origin"])).to_numpy()
            else:
                g = gsh[gsh.est == m].sort_values("k")
                ref = np.array([float.fromhex(h) for h in g["pred_origin_hex"]])
            mine = np.array(fl["p0"]["full"])
            first = np.array([pr["full"][int(r - C.R_FIRST)] for r in o["r"]])
            gates[m] = dict(
                GT_origin_bit_equal=int((mine == ref).sum()),
                GT_origin_n=len(ref),
                GT_origin_maxabs=float(np.nanmax(np.abs(mine - ref))) if np.isfinite(ref).any() else float("nan"),
                GT_origin_walk_missing=int((~np.isfinite(ref)).sum()),
                GT_block_first_equals_p0=bool(np.array_equal(first, mine)),
                GT_block_vs_shap_equal=int(sum(fl["shap_pred_equal"])),
                GT_block_vs_shap_n=len(fl["shap_pred_equal"]),
                cpus=sorted(fl["cpus"]),
            )
            gates[m]["pass"] = bool(
                gates[m]["GT_origin_bit_equal"] == 126
                and gates[m]["GT_block_vs_shap_equal"] == gates[m]["GT_block_vs_shap_n"]
            )
        else:
            pr, sets, ex = load_linear(work, m, o, a.sets_tag)
            f = lwalk[(lwalk.model == m) & (lwalk.predictor_set == "all_features")].set_index("date")["pred_adj"]
            mx, n = 0.0, 0
            for g0, rep, i in zip(ex["g"], ex["replay"], o["oos_row"]):
                walk = f.reindex(dd[g0 : int(i) + 1]).to_numpy()
                assert np.isfinite(walk).all()
                mx, n = max(mx, float(np.abs(rep - walk).max())), n + len(rep)
            gates[m] = dict(GL1_rows=n, GL1_maxabs=mx, GL1_pass=bool(mx <= 1e-12), reseeds=ex["reseeds"])
            if abl is not None:
                ab = abl.set_index("date")
                for arm in C.GATE_ARMS:
                    rid = f"gate_{arm}"
                    if rid not in pr:
                        continue
                    at = np.array([pr[rid][int(r - C.R_FIRST)] for r in o["r"]])
                    w = ab[f"{m}|{arm}"].reindex(pd.DatetimeIndex(o["origin"])).to_numpy()
                    gates[m][f"GL2_{arm}_maxabs"] = float(np.abs(at - w).max())
                    gates[m][f"GL2_{arm}_pass_1e12"] = bool(np.abs(at - w).max() <= 1e-12)
            gates[m]["pass"] = gates[m]["GL1_pass"] and all(
                v for k_, v in gates[m].items() if k_.startswith("GL2_") and k_.endswith("pass_1e12")
            )
        if pre_w is not None:  # gate GP: rungs without the FOMC proximity series = their pre-patch twins, bit for bit
            pw = Path(a.prepatch_work)
            if m in C.TREES:
                pre_sets = json.loads(sorted((pw / "trees" / m).glob("sets_part*.json"))[0].read_text())
            else:
                pre_sets = json.loads((pw / "linear" / m / "sets.json").read_text())
            pre_of = {frozenset(pre_sets["sets"][r]): r for r in pre_sets["fitted"]}
            n_cmp, n_eq, bad = 0, 0, []
            for rid in sets["fitted"]:
                ss = frozenset(sets["sets"][rid])
                if rid == "full" or ss & {"fomc_until_inv", "fomc_since_inv"} or ss not in pre_of:
                    continue
                n_cmp += 1
                twin = pre_w[f"{m}|{pre_of[ss]}"].reindex(rows_dates).to_numpy()
                if np.array_equal(pr[rid], twin):
                    n_eq += 1
                else:
                    bad.append(rid)
            gates[m].update(GP_compared=n_cmp, GP_bit_equal=n_eq, GP_differ=bad)
            gates[m]["pass"] = bool(gates[m]["pass"] and not bad)
        sets_of[m] = sets
        for rid, fitted in sets["alias"].items():
            panel[(m, rid)] = pd.Series(pr[fitted] ** 2 * b, index=rows_dates)
        for rid in sets["fitted"]:
            wide[f"{m}|{rid}"] = pr[rid]
    (RES / "gates.json").write_text(json.dumps(gates, indent=1, default=str))
    pd.DataFrame({"date": rows_dates, **wide}).to_parquet(RES / "forecasts_wide.parquet", index=False)
    say(json.dumps({m: g["pass"] for m, g in gates.items()}))

    # ---------------------------------------------------------------- scored bars (Table 1's)
    rv = core_rv()
    days = evaluation_days(rv)
    idx = None
    for f in panel.values():
        ok = f.index[np.isfinite(f.to_numpy()) & (f.to_numpy() > 0)]
        idx = ok if idx is None else idx.intersection(ok)
    assert idx is not None
    idx = idx[pd.DatetimeIndex(idx.normalize()).isin(days)]
    t = rv.reindex(idx)
    idx = idx[(t.to_numpy() > 0) & np.isfinite(t.to_numpy())].sort_values()
    assert len(idx) == N_BARS, len(idx)
    spans = {"all": idx, "c16": idx[(idx.hour == 16) & (idx.minute == 0)]}
    spans["c16_2020"] = spans["c16"][spans["c16"].normalize() >= SUB_FIRST]
    assert len(spans["c16"]) == N_DAYS and len(spans["c16_2020"]) == SUB_DAYS

    def losses(f: pd.Series, ix: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
        p, y = f.reindex(ix).to_numpy(float), rv.reindex(ix).to_numpy(float)
        r = y / p
        return r - np.log(r) - 1.0, (y - p) ** 2

    def daily(x: np.ndarray, ix: pd.DatetimeIndex) -> np.ndarray:
        return pd.Series(x, index=pd.DatetimeIndex(ix.normalize())).groupby(level=0).mean().to_numpy()

    L = {(key, sp): losses(f, ix) for key, f in panel.items() for sp, ix in spans.items()}

    def dm(k1: tuple, k2: tuple, sp: str) -> dict:
        x1, x2, ix = L[(k1, sp)][0], L[(k2, sp)][0], spans[sp]
        r = dm_test(daily(x1, ix), daily(x2, ix)) if sp == "all" else dm_test(x1, x2)
        return dict(dm=float(r["dm"]), p=float(r["p"]), lag=int(r.get("hac_lag", -1)))

    t1 = pd.read_csv(TABLE1)
    rows = []
    for m in models:
        for rid in sets_of[m]["alias"]:
            if rid.startswith("gate_"):
                continue
            rank_, k = rung_k(rid)
            ss = sets_of[m]["sets"][rid]
            for sp in SPANS:
                q, se = L[((m, rid), sp)]
                lq_full = float(L[((m, "full"), sp)][0].mean())
                lq_har = float(L[((m, "har"), sp)][0].mean())
                vf = dm((m, rid), (m, "full"), sp) if rid != "full" else dict(dm=np.nan, p=np.nan, lag=-1)
                vh = dm((m, rid), (m, "har"), sp) if rid != "har" else dict(dm=np.nan, p=np.nan, lag=-1)
                gain = lq_har - lq_full
                rows.append(
                    dict(
                        model=m,
                        ranking=rank_,
                        k=k,
                        rung=rid,
                        fitted_as=sets_of[m]["alias"][rid],
                        span=sp,
                        n_bars=len(spans[sp]),
                        qlike=float(q.mean()),
                        mse=float(se.mean()),
                        dm_vs_full=vf["dm"],
                        p_vs_full=vf["p"],
                        dm_vs_har=vh["dm"],
                        p_vs_har=vh["p"],
                        hac_lag=max(vf["lag"], vh["lag"]),
                        retained_pct=(lq_har - float(q.mean())) / gain * 100.0 if gain > 0 else np.nan,
                        series=" ".join(ss) if k not in (46,) else "(all 46)",
                    )
                )
    S = pd.DataFrame(rows)
    # reference: Table 1's every-bar walks (full set and HAR benchmark of each model; the OLS HAR)
    refrows: list[dict] = []
    for m in models:
        for st, lab in (("all_features", "walk_full"), ("baseline", "walk_har")):
            r = t1[(t1.model == T1_NAME.get(m, m)) & (t1.set == st)]
            if len(r):
                refrows.append(dict(model=m, rung=lab, span="all", qlike=float(r["qlike"].iloc[0])))
                refrows.append(dict(model=m, rung=lab, span="c16_2020", qlike=float(r["qlike_1600"].iloc[0])))
    r = t1[(t1.model == "ols") & (t1.set == "baseline")]
    refrows.append(dict(model="ols", rung="walk_har", span="all", qlike=float(r["qlike"].iloc[0])))
    refrows.append(dict(model="ols", rung="walk_har", span="c16_2020", qlike=float(r["qlike_1600"].iloc[0])))
    S = pd.concat([S, pd.DataFrame(refrows)], ignore_index=True)
    S.to_csv(RES / "scores.csv", index=False)

    # ---------------------------------------------------------------- readings (RULE section 6)
    RANKS = ["P"] + (["S"] if (S.ranking == "S").any() else [])
    rd = []
    for m in models:
        for rank_ in RANKS:
            for sp in ("all", "c16"):
                f = S[(S.model == m) & (S.span == sp) & (S.ranking.isin([rank_, "both"]))].set_index("k")
                lf, lh = float(f.loc[46, "qlike"]), float(f.loc[0, "qlike"])
                ret = f["retained_pct"]
                kx = {}
                for thr in (50, 75, 90):
                    hit = [k for k in (0, 1, 3, 5, 10, 20, 46) if k < 46 and ret.get(k, np.nan) >= thr]
                    kx[thr] = min(hit) if hit else 46
                if lf >= lh:
                    verdict = "NO GAIN TO RETAIN"
                elif kx[90] <= 5:
                    verdict = "CONCENTRATED"
                elif kx[90] <= 20:
                    verdict = "INTERMEDIATE"
                else:
                    verdict = "DENSE AND WEAK"
                better = [int(k) for k in (1, 3, 5, 10, 20) if float(f.loc[k, "qlike"]) < lf]
                rd.append(
                    dict(
                        model=m,
                        ranking={"P": "primary", "S": "secondary (look-ahead)"}[rank_],
                        span=sp,
                        qlike_full=lf,
                        qlike_har=lh,
                        gain_full_vs_har=lh - lf,
                        k50=kx[50],
                        k75=kx[75],
                        k90=kx[90],
                        verdict=verdict,
                        rungs_below_full=" ".join(map(str, better)),
                        **{f"R{k}": float(ret.get(k, np.nan)) for k in (1, 3, 5, 10, 20)},
                        **{f"p{k}_vs_full": float(f.loc[k, "p_vs_full"]) for k in (20, 10, 5)},
                        **{f"dm{k}_vs_full": float(f.loc[k, "dm_vs_full"]) for k in (20, 10, 5)},
                    )
                )
    R = pd.DataFrame(rd)
    R.to_csv(RES / "readings.csv", index=False)

    # ---------------------------------------------------------------- tables.txt
    lines = ["PRE-PATCH design: the FOMC calendar features are those before the pending FOMC fix.", ""]
    for sp in ("all", "c16", "c16_2020"):
        lines.append(f"QLIKE by rung, span {sp} (P = primary, pre-sample ranking; S = secondary, look-ahead)")
        hdr = f"{'model':14s}" + "".join(f"{'k=' + str(k):>10s}" for k in C.KS)
        for rank_ in RANKS:
            lines.append(f"  ranking {rank_}")
            lines.append("  " + hdr)
            for m in models:
                f = S[(S.model == m) & (S.span == sp) & (S.ranking.isin([rank_, "both"]))].set_index("k")
                lines.append("  " + f"{NAME[m]:14s}" + "".join(f"{float(f.loc[k, 'qlike']):10.5f}" for k in C.KS))
        lines.append("")
    lines.append("Retained % of the full set's QLIKE gain over the model's own HAR rung, and DM p vs full (span all)")
    for rank_ in RANKS:
        lines.append(f"  ranking {rank_}")
        for m in models:
            f = S[(S.model == m) & (S.span == "all") & (S.ranking.isin([rank_, "both"]))].set_index("k")
            cells = " ".join(
                f"k{k}: {float(f.loc[k, 'retained_pct']):6.1f}% (p {float(f.loc[k, 'p_vs_full']):.3f})"
                for k in (20, 10, 5, 3, 1)
            )
            lines.append(f"  {NAME[m]:14s} {cells}")
    lines.append("")
    lines.append(R.to_string(index=False))
    lines.append("")
    lines.append("Gates: " + json.dumps(gates, default=str))
    (RES / "tables.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    say("\n".join(lines[:40]))
    figure(S, models)
    return 0


def figure(S: pd.DataFrame, models: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
    BLUE, ORANGE = "#2a78d6", "#eb6834"
    xs = {46: 6, 20: 5, 10: 4, 5: 3, 3: 2, 1: 1, 0: 0}
    fig, axes = plt.subplots(2, len(models), figsize=(3.0 * len(models), 6.4), squeeze=False)
    for j, m in enumerate(models):
        for i, (sp, lab) in enumerate((("all", "every bar"), ("c16", "16:00 bar"))):
            ax = axes[i, j]
            f = S[(S.model == m) & (S.span == sp) & S.k.notna()]
            full = float(f[f.k == 46]["qlike"].iloc[0])
            har = float(f[f.k == 0]["qlike"].iloc[0])
            ax.axhline(full, color=INK2, lw=1.0, ls="-", zorder=1)
            ax.axhline(har, color=INK2, lw=1.0, ls=":", zorder=1)
            for rank_, col, ls, nm in (
                ("S", ORANGE, "--", "secondary (look-ahead)"),
                ("P", BLUE, "-", "primary (pre-sample)"),
            ):
                if not (f.ranking == rank_).any():  # a run without the secondary ranking
                    continue
                g = f[f.ranking.isin([rank_, "both"])].sort_values("k")
                ax.plot(
                    [xs[int(k)] for k in g.k],
                    g.qlike,
                    color=col,
                    lw=2,
                    ls=ls,
                    marker="o",
                    ms=4.5,
                    markeredgecolor="white",
                    markeredgewidth=1.0,
                    label=nm,
                    zorder=3,
                )
            ax.set_xticks(list(xs.values()))
            ax.set_xticklabels([str(k) for k in xs])
            ax.grid(axis="y", color=GRID, lw=0.6)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(INK2)
            ax.tick_params(colors=INK2, labelsize=8)
            ax.text(6.1, full, "full", color=INK2, fontsize=7, va="bottom", ha="right")
            ax.text(6.1, har, "HAR", color=INK2, fontsize=7, va="bottom", ha="right")
            if i == 0:
                ax.set_title(NAME[m], fontsize=10, color=INK)
            if j == 0:
                ax.set_ylabel(f"QLIKE, {lab}", fontsize=9, color=INK)
            if i == 1:
                ax.set_xlabel("k (exogenous series kept)", fontsize=8, color=INK2)
    h, lb = axes[0, 0].get_legend_handles_labels()
    fig.legend(h[::-1], lb[::-1], loc="upper center", ncol=2, frameon=False, fontsize=9)
    fig.text(
        0.5,
        0.005,
        "PRE-PATCH design (FOMC features before the pending fix). Monthly refits (126 origins, held 21 sessions); "
        "Table 1's 33,527 bars.\n"
        "Gray lines: the model's full set (solid) and its HAR-benchmark rung (dotted).",
        ha="center",
        fontsize=7.5,
        color=INK2,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.95))
    fig.savefig(RES / "ladder.png", dpi=160)
    fig.savefig(RES / "ladder.pdf")


if __name__ == "__main__":
    raise SystemExit(main())
