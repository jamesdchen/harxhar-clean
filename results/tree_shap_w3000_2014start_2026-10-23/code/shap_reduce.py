"""Gate and measures M1-M5 of RULE_shap.txt from the per-origin SHAP files (shap_run.py) -> the results directory:
gate.csv, m1_*.csv, m2_*.csv, m3_*.csv, m4_*.csv, m5_*.csv, tables.txt, run_info.json.

    python shap_reduce.py [--work DIR]   (DIR holds lgbm/, xgb/, rf_skx/ and x_forecast_rows.npz or the design)
"""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import shap_common as C

B_BOOT, BLOCK = 2000, 12
TXT: list[str] = []


def out(s: str = "") -> None:
    TXT.append(s)
    print(s, flush=True)


# ============================================================================ loading
def load_est(work: Path, est: str, o: pd.DataFrame, M: int) -> dict:
    """Row-level variable SHAP (33,970 x 48), per-origin column |phi| sums, gate and check quantities."""
    PG, PRED, BASE = [], [], []
    colabs = np.zeros((len(o), M))
    live = np.zeros((len(o), M), dtype=bool)
    used = np.zeros((len(o), M), dtype=bool)
    g = []
    G, cm, ci, ierr = [], [], [], []
    for ii, k in enumerate(o["k"]):
        z = np.load(work / est / f"k{k:03d}.npz")
        n = int(z["r_end"] - z["r"])
        assert z["phi"].shape == (n, M) and z["phi_group"].shape == (n, 48), (est, k)
        PG.append(z["phi_group"])
        PRED.append(z["pred"])
        BASE.append(z["base"])
        colabs[ii] = np.abs(z["phi"].astype(np.float64)).sum(0)
        live[ii], used[ii] = z["live"], z["used"]
        f = np.abs(z["pred"])
        tol = 1e-5 * np.maximum(1.0, f) if est == "xgb" else np.full(n, 1e-9)
        rec = dict(
            est=est,
            k=int(k),
            pred_origin=float(z["pred_origin"]),
            pred_origin_hex=float(z["pred_origin"]).hex(),
            add_err_max=float(np.abs(z["add_err"]).max()),
            add_pass=bool((np.abs(z["add_err"]) <= tol).all()),
            pred_origin_eq_pred0=bool(z["pred_origin"] == z["pred"][0]),
            cpu=str(z["cpu"]) if "cpu" in z.files else "laptop (pre-guard run)",
        )
        if "tree_mean" in z.files:
            rel = np.abs(z["pred"] - z["tree_mean"]) / np.maximum(
                np.abs(z["pred"]), 1e-300
            )
            rec["onedal_vs_trees_rel_max"] = float(rel.max())
        if "repeat_pred_equal" in z.files:
            rec["repeat_pred_equal"] = bool(z["repeat_pred_equal"])
            rec["repeat_phi_equal"] = bool(z["repeat_phi_equal"])
        if "G" in z.files:
            G.append(z["G"])
            cm.append(z["col_main_abs"])
            ci.append(z["col_inter_abs"])
            ierr.append(float(np.abs(z["inter_sum_err"]).max()))
            rec["inter_sum_err_max"] = ierr[-1]
        g.append(rec)
    res = dict(
        PG=np.vstack(PG),
        pred=np.concatenate(PRED),
        base=np.concatenate(BASE),
        colabs=colabs,
        live=live,
        used=used,
        gate=pd.DataFrame(g),
    )
    if G:
        res.update(
            G=np.concatenate(G),
            col_main=np.concatenate(cm),
            col_inter=np.concatenate(ci),
        )
    return res


def boot_counts(K: int, seed: int = C.SEED) -> np.ndarray:
    """Circular moving-block bootstrap over origins: B x K counts (block 12)."""
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(K / BLOCK))
    cnt = np.zeros((B_BOOT, K))
    for b in range(B_BOOT):
        idx = (rng.integers(0, K, nb)[:, None] + np.arange(BLOCK)[None, :]).ravel()[
            :K
        ] % K
        np.add.at(cnt[b], idx, 1.0)
    return cnt


def ci(x: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def pct(v: float, d: int = 1) -> str:
    return f"{100 * v:.{d}f}"


# ============================================================================ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=str(C.WORK))
    ap.add_argument("--est", default=",".join(C.EST))
    ap.add_argument(
        "--gate-only",
        action="store_true",
        help="the gate, nothing else (read before any measure)",
    )
    a = ap.parse_args()
    work = Path(a.work)
    ests = a.est.split(",")
    t0 = time.time()
    T = C.column_table()
    M = len(T)
    o = C.origins()
    K = len(o)
    d = C.oos_dates()
    rows = np.concatenate([np.arange(r, e) for r, e in zip(o["r"], o["r_end"])])
    dates = pd.DatetimeIndex(d[rows - C.W])
    korig = np.concatenate(
        [np.full(e - r, k) for k, r, e in zip(o["k"], o["r"], o["r_end"])]
    )
    year = dates.year.to_numpy()
    bar = dates.strftime("%H:%M").to_numpy()
    GI = {g: i for i, g in enumerate(C.GROUPS)}
    EX = np.array([GI[g] for g in C.EXOG])
    NS = np.array([GI[g] for g in C.NONSTANDARD])
    NS_EX = np.array(sorted(set(NS) | {GI[g] for g in C.EMINI_ACTIVITY}))
    if a.gate_only:  # the origins computed so far
        have = {
            e: sorted(int(f.stem[1:]) for f in (work / e).glob("k[0-9][0-9][0-9].npz"))
            for e in ests
        }
        o = o[
            o["k"].isin(set.intersection(*[set(v) for v in have.values()]))
        ].reset_index(drop=True)
    data = {e: load_est(work, e, o, M) for e in ests}
    RES = C.RES
    out(
        f"tree SHAP, W = 3,000, 2014 start: {K} origins, {len(rows)} forecast rows, models {ests}"
    )

    # ------------------------------------------------------------------ gate
    walk = pd.read_csv(RES / "inputs" / "walk_origin_rows.csv", parse_dates=["date"])
    walk = walk[walk["run_dir"].str.endswith("w3000")]
    gate = []
    for e in ests:
        gt = data[e]["gate"].merge(o[["k", "origin"]], on="k")
        w = walk[walk["est"] == e].drop_duplicates("date").set_index("date")
        gt["walk_hex"] = gt["origin"].map(w["pred_adj_hex"])
        gt["walk_fit_row"] = gt["origin"].map(w["fit_row"])
        gt["fit_row"] = o.set_index("k").loc[gt["k"], "r"].to_numpy() - C.W
        gt["walk_equal"] = [
            (float.fromhex(h) == p) if isinstance(h, str) else np.nan
            for h, p in zip(gt["walk_hex"], gt["pred_origin"])
        ]
        gt["walk_abs_diff"] = [
            abs(float.fromhex(h) - p) if isinstance(h, str) else np.nan
            for h, p in zip(gt["walk_hex"], gt["pred_origin"])
        ]
        gt["walk_rel_diff"] = gt["walk_abs_diff"] / gt["pred_origin"].abs()
        gate.append(gt)
    gate = pd.concat(gate, ignore_index=True)
    gate.to_csv(RES / "gate.csv", index=False)
    out("\n== GATE")
    for e in ests:
        g = gate[gate["est"] == e]
        have = g["walk_hex"].notna()
        fr = (g.loc[have, "walk_fit_row"] == g.loc[have, "fit_row"]).all()
        out(
            f"{C.EST_NAME[e]:14s} walk rows {have.sum():3d}/{len(g)}; bit-equal {int(g.loc[have, 'walk_equal'].sum())}"
            f"/{have.sum()}; max |diff| {g['walk_abs_diff'].max():.3g} (rel {g['walk_rel_diff'].max():.3g}); fit rows "
            f"equal {fr}; additivity pass {int(g['add_pass'].sum())}/{len(g)} (max {g['add_err_max'].max():.2g})"
            + (
                f"; interaction rows sum to SHAP: max {g['inter_sum_err_max'].max():.2g}"
                if g.get("inter_sum_err_max", pd.Series(dtype=float)).notna().any()
                else ""
            )
            + (
                f"; oneDAL vs estimators_ mean rel max {g['onedal_vs_trees_rel_max'].max():.2g}"
                if g.get("onedal_vs_trees_rel_max", pd.Series(dtype=float))
                .notna()
                .any()
                else ""
            )
        )
        if g.get("repeat_pred_equal", pd.Series(dtype=float)).notna().any():
            rp = g.dropna(subset=["repeat_pred_equal"])
            out(
                f"{'':14s} repeat refits at k {rp['k'].tolist()}: predictions equal {rp['repeat_pred_equal'].tolist()}, "
                f"SHAP equal {rp['repeat_phi_equal'].tolist()}"
            )
        out(f"{'':14s} cpus: {sorted(set(g['cpu']))}")

    if a.gate_only:
        (RES / "gate.txt").write_text("\n".join(TXT) + "\n", encoding="utf-8")
        return 0
    cnt = boot_counts(K)
    # per-origin sums of |phi_A| (rows equal weight in the pooled shares)
    SA = {e: np.zeros((K, 48)) for e in ests}
    for e in ests:
        np.add.at(SA[e], korig, np.abs(data[e]["PG"]))

    # ------------------------------------------------------------------ M1
    out(
        "\n== M1 importance shares (share of total mean|SHAP| over the 48 variables; % ; 95% block-bootstrap CI)"
    )
    m1 = []
    for e in ests:
        PG = np.abs(data[e]["PG"])
        Imean = PG.mean(0)
        s = Imean / Imean.sum()
        sx = Imean / Imean[EX].sum()
        Icol = data[e]["colabs"].sum(0) / len(rows)
        Ivar_col = np.array(
            [Icol[(T["group"] == g).to_numpy()].sum() for g in C.GROUPS]
        )
        boot = cnt @ SA[e]
        bs = boot / boot.sum(1, keepdims=True)
        lv = np.array(
            [
                [
                    data[e]["live"][k][(T["group"] == g).to_numpy()].any()
                    for g in C.GROUPS
                ]
                for k in range(K)
            ]
        )
        us = np.array(
            [
                [
                    data[e]["used"][k][(T["group"] == g).to_numpy()].any()
                    for g in C.GROUPS
                ]
                for k in range(K)
            ]
        )
        for i, g in enumerate(C.GROUPS):
            lo, hi = ci(bs[:, i])
            m1.append(
                dict(
                    est=e,
                    variable=g,
                    bucket=C.CAT_OF[g],
                    cls=C.CLASS_OF[g],
                    mean_abs_shap=Imean[i],
                    share=s[i],
                    share_lo=lo,
                    share_hi=hi,
                    share_exog=(sx[i] if g in C.EXOG else np.nan),
                    share_col=Ivar_col[i] / Ivar_col.sum(),
                    K_live=int(lv[:, i].sum()),
                    K_used=int(us[:, i].sum()),
                )
            )
    m1 = pd.DataFrame(m1)
    m1["rank"] = (
        m1.groupby("est")["share"].rank(ascending=False, method="first").astype(int)
    )
    m1.to_csv(RES / "m1_variables.csv", index=False)
    cls_rows, bk_rows = [], []
    for e in ests:
        boot = cnt @ SA[e]
        tot = boot.sum(1)
        for name, idx in (
            ("NON-STANDARD", NS),
            ("STANDARD", np.setdiff1d(np.arange(48), NS)),
            ("NON-STANDARD incl. sumvolume/numobs", NS_EX),
        ):
            v = SA[e][:, idx].sum() / SA[e].sum()
            lo, hi = ci(boot[:, idx].sum(1) / tot)
            vx = SA[e][:, np.intersect1d(idx, EX)].sum() / SA[e][:, EX].sum()
            lox, hix = ci(boot[:, np.intersect1d(idx, EX)].sum(1) / boot[:, EX].sum(1))
            cls_rows.append(
                dict(
                    est=e,
                    cls=name,
                    share=v,
                    lo=lo,
                    hi=hi,
                    share_exog=vx,
                    lo_exog=lox,
                    hi_exog=hix,
                )
            )
        for b in C.BUCKETS:
            idx = np.array([GI[g] for g in C.GROUPS if C.CAT_OF[g] == b])
            lo, hi = ci(boot[:, idx].sum(1) / tot)
            bk_rows.append(
                dict(
                    est=e,
                    bucket=b,
                    share=SA[e][:, idx].sum() / SA[e].sum(),
                    lo=lo,
                    hi=hi,
                )
            )
    cls = pd.DataFrame(cls_rows)
    bk = pd.DataFrame(bk_rows)
    cls.to_csv(RES / "m1_classes.csv", index=False)
    bk.to_csv(RES / "m1_buckets.csv", index=False)
    out(f"{'bucket':28s}" + "".join(f"{C.EST_NAME[e]:>24s}" for e in ests))
    for b in C.BUCKETS:
        r = bk[bk["bucket"] == b].set_index("est")
        out(
            f"{C.BUCKET_NAME[b]:28s}"
            + "".join(
                f"{pct(r.at[e, 'share']):>9s} [{pct(r.at[e, 'lo'])}, {pct(r.at[e, 'hi'])}]".rjust(
                    24
                )
                for e in ests
            )
        )
    for name in ("STANDARD", "NON-STANDARD", "NON-STANDARD incl. sumvolume/numobs"):
        r = cls[cls["cls"] == name].set_index("est")
        out(
            f"{name[:28]:28s}"
            + "".join(
                f"{pct(r.at[e, 'share']):>9s} [{pct(r.at[e, 'lo'])}, {pct(r.at[e, 'hi'])}]".rjust(
                    24
                )
                for e in ests
            )
        )
        out(
            f"{'  of the exogenous total':28s}"
            + "".join(
                f"{pct(r.at[e, 'share_exog']):>9s} [{pct(r.at[e, 'lo_exog'])}, {pct(r.at[e, 'hi_exog'])}]".rjust(
                    24
                )
                for e in ests
            )
        )
    out(
        "\nTop 15 variables by share (%), NON-STANDARD marked *; [K_used of 126 origins]"
    )
    for e in ests:
        t = m1[m1["est"] == e].sort_values("share", ascending=False).head(15)
        out(
            f"{C.EST_NAME[e]}: "
            + "; ".join(
                f"{r.variable}{'*' if r.cls == 'NON-STANDARD' else ''} {pct(r.share)} [{r.K_used}]"
                for r in t.itertuples()
            )
        )
    out("NON-STANDARD variables by share (%):")
    for e in ests:
        t = m1[(m1["est"] == e) & (m1["cls"] == "NON-STANDARD")].sort_values(
            "share", ascending=False
        )
        out(
            f"{C.EST_NAME[e]}: "
            + "; ".join(f"{r.variable} {pct(r.share, 2)}" for r in t.itertuples())
        )
    # by year
    yrs = sorted(set(year))
    by_year = []
    for e in ests:
        PG = np.abs(data[e]["PG"])
        for y in yrs:
            Imean = PG[year == y].mean(0)
            s = Imean / Imean.sum()
            for i, g in enumerate(C.GROUPS):
                by_year.append(
                    dict(
                        est=e,
                        year=int(y),
                        variable=g,
                        bucket=C.CAT_OF[g],
                        cls=C.CLASS_OF[g],
                        share=s[i],
                        rows=int((year == y).sum()),
                    )
                )
    by_year = pd.DataFrame(by_year)
    by_year.to_csv(RES / "m1_by_year.csv", index=False)
    out("\nNON-STANDARD share by year (%)")
    out(f"{'':14s}" + "".join(f"{y:>7d}" for y in yrs))
    for e in ests:
        t = (
            by_year[(by_year["est"] == e) & (by_year["cls"] == "NON-STANDARD")]
            .groupby("year")["share"]
            .sum()
        )
        out(f"{C.EST_NAME[e]:14s}" + "".join(f"{pct(t[y]):>7s}" for y in yrs))
    out("HAR terms share by year (%)")
    for e in ests:
        t = by_year[
            (by_year["est"] == e) & (by_year["variable"] == "har_ma")
        ].set_index("year")["share"]
        out(f"{C.EST_NAME[e]:14s}" + "".join(f"{pct(t[y]):>7s}" for y in yrs))

    # ------------------------------------------------------------------ M2
    out(
        "\n== M2 rank agreement with the linear models (W = 3,000 walks, same 126 origins)"
    )
    z = np.load(
        io.BytesIO(
            C.git_show(
                f"{C.FIW_SHA}:results/fi_by_window_2014start_2026-10-22/fiw_gamma.npz"
            )
        )
    )
    assert list(z["columns"]) == T["column"].tolist()
    assert list(pd.to_datetime(z["origins"])) == list(o["origin"])
    P = C.group_matrix(T)
    LIN = {
        m: np.abs(z[f"walk_w3000_{m}_gamma"]) @ P.T for m in ("ridge", "lasso", "enet")
    }  # K x 48
    fv = pd.read_csv(RES / "inputs" / "fiw_variables_w3000_walk.csv")
    chk = fv[fv["model"] == "ridge"].set_index("variable")["pooled_share"]
    rshare = LIN["ridge"][:, EX].sum(0) / LIN["ridge"][:, EX].sum()
    assert np.allclose(rshare, chk.reindex(C.EXOG).to_numpy(), rtol=1e-9, atol=1e-12), (
        "ridge share != fiw_variables"
    )
    m2 = []
    pooled_lin = {m: LIN[m].sum(0) for m in LIN}
    pd.DataFrame(
        [
            dict(
                linear=m,
                variable=g,
                share_all=pooled_lin[m][i] / pooled_lin[m].sum(),
                share_exog=(
                    pooled_lin[m][i] / pooled_lin[m][EX].sum()
                    if g in C.EXOG
                    else np.nan
                ),
            )
            for m in LIN
            for i, g in enumerate(C.GROUPS)
        ]
    ).to_csv(RES / "m2_linear_shares.csv", index=False)

    def top10(v: np.ndarray, names: list[str]) -> list[str]:
        return [names[i] for i in np.argsort(-v)[:10]]

    for e in ests:
        tree_ex = SA[e][:, EX].sum(0)
        tree_all = SA[e].sum(0)
        tree_col = (
            m1[m1["est"] == e]
            .set_index("variable")["share_col"]
            .reindex(C.GROUPS)
            .to_numpy()
        )
        for m in ("ridge", "lasso", "enet"):
            lex, lall = pooled_lin[m][EX], pooled_lin[m]
            for scope, tv, lv_, names in (
                ("46 exogenous", tree_ex, lex, C.EXOG),
                ("48 groups", tree_all, lall, C.GROUPS),
                ("46 exogenous, tree column-level", tree_col[EX], lex, C.EXOG),
            ):
                rho = spearmanr(tv, lv_).statistic
                ov = len(set(top10(tv, names)) & set(top10(lv_, names)))
                m2.append(
                    dict(
                        est=e,
                        linear=m,
                        scope=scope,
                        spearman=rho,
                        top10_overlap=ov,
                        tree_top10=" ".join(top10(tv, names)),
                        linear_top10=" ".join(top10(lv_, names)),
                    )
                )
    for i, e1 in enumerate(ests):
        for e2 in ests[i + 1 :]:
            v1, v2 = SA[e1][:, EX].sum(0), SA[e2][:, EX].sum(0)
            m2.append(
                dict(
                    est=e1,
                    linear=e2,
                    scope="46 exogenous (tree vs tree)",
                    spearman=spearmanr(v1, v2).statistic,
                    top10_overlap=len(set(top10(v1, C.EXOG)) & set(top10(v2, C.EXOG))),
                )
            )
    m2 = pd.DataFrame(m2)
    m2.to_csv(RES / "m2_rank.csv", index=False)
    for r in m2[m2["scope"] == "46 exogenous"].itertuples():
        out(
            f"{C.EST_NAME[r.est]:14s} vs {r.linear:6s}: Spearman {r.spearman:+.2f}, top-10 overlap {r.top10_overlap}/10"
        )
    out("secondary (48 groups / tree column-level / tree vs tree):")
    for r in m2[m2["scope"] != "46 exogenous"].itertuples():
        out(
            f"  {r.est:7s} vs {r.linear:7s} {r.scope:34s} Spearman {r.spearman:+.2f}, top-10 overlap {r.top10_overlap}/10"
        )
    out(
        f"ridge top 10 (46 exogenous): {' '.join(top10(pooled_lin['ridge'][EX], C.EXOG))}"
    )
    for e in ests:
        out(f"{C.EST_NAME[e]} top 10: {' '.join(top10(SA[e][:, EX].sum(0), C.EXOG))}")
    # D: NON-STANDARD share, tree minus ridge
    out(
        "\nD = NON-STANDARD share, tree minus linear (pooled over origins; paired block bootstrap)"
    )
    md = []
    for e in ests:
        bt = cnt @ SA[e]
        for m in ("ridge", "lasso", "enet"):
            bl = cnt @ LIN[m]
            for label, num, den in (
                ("of exogenous total", np.intersect1d(NS, EX), EX),
                ("of all 48 (backbone in denominator)", NS, np.arange(48)),
                (
                    "of exogenous, sumvolume/numobs NON-STANDARD",
                    np.intersect1d(NS_EX, EX),
                    EX,
                ),
            ):
                tv = SA[e][:, num].sum() / SA[e][:, den].sum()
                lv_ = LIN[m][:, num].sum() / LIN[m][:, den].sum()
                dd = bt[:, num].sum(1) / bt[:, den].sum(1) - bl[:, num].sum(1) / bl[
                    :, den
                ].sum(1)
                lo, hi = ci(dd)
                verdict = (
                    "MORE" if lo > 0 else "LESS" if hi < 0 else "NO CLEAR DIFFERENCE"
                )
                md.append(
                    dict(
                        est=e,
                        linear=m,
                        denominator=label,
                        tree=tv,
                        linear_share=lv_,
                        D=tv - lv_,
                        lo=lo,
                        hi=hi,
                        verdict=verdict,
                    )
                )
    md = pd.DataFrame(md)
    md.to_csv(RES / "m2_nonstandard_D.csv", index=False)
    for r in md.itertuples():
        if r.linear == "ridge" or r.denominator == "of exogenous total":
            out(
                f"{C.EST_NAME[r.est]:14s} vs {r.linear:5s} {r.denominator:45s} tree {pct(r.tree)} linear "
                f"{pct(r.linear_share)}  D {100 * r.D:+.1f} [{100 * r.lo:+.1f}, {100 * r.hi:+.1f}] -> {r.verdict}"
            )

    # ------------------------------------------------------------------ M3
    out(
        "\n== M3 response shape of the NON-STANDARD variables (mean SHAP by bin of the dominant value column)"
    )
    xr = load_x(work, rows)
    m3, m3b = [], []
    for e in ests:
        sh = m1[m1["est"] == e].set_index("variable")
        sel = [g for g in C.NONSTANDARD if sh.at[g, "share"] >= 0.005]
        selnote = {g: "" for g in sel}
        if len(sel) < 5:
            top = (
                sh.loc[C.NONSTANDARD]
                .sort_values("share", ascending=False)
                .index[:5]
                .tolist()
            )
            for g in top:
                if g not in sel:
                    sel.append(g)
                    selnote[g] = "below 0.5%"
        Icol = data[e]["colabs"].sum(0) / len(rows)
        dom = {}
        for g in sel:
            cand = T.index[(T["group"] == g) & (T["kind"] == "value")].to_numpy()
            dom[g] = int(cand[np.argmax(Icol[cand])])
        phi_dom = load_cols(work, e, o, [dom[g] for g in sel])
        meanf = float(np.mean(data[e]["pred"]))
        for g in sel:
            j = dom[g]
            x = xr[:, j].astype(np.float64)
            yA = data[e]["PG"][:, GI[g]]
            yj = phi_dom[j]
            nun = len(np.unique(x))
            if nun <= 10:
                codes = np.searchsorted(np.unique(x), x)
            else:
                codes = pd.qcut(x, 10, labels=False, duplicates="drop")
            nb = int(codes.max()) + 1
            mb = np.array([yA[codes == b].mean() for b in range(nb)])
            I_A = float(np.abs(yA).mean())
            shape, flag, rho, R = classify(mb, I_A)
            m3.append(
                dict(
                    est=e,
                    variable=g,
                    bucket=C.CAT_OF[g],
                    share=sh.at[g, "share"],
                    selection=selnote[g] or ">= 0.5%",
                    dominant_column=T.at[j, "column"],
                    bins=nb,
                    distinct_values=nun,
                    I_A=I_A,
                    range=R,
                    spearman=rho,
                    shape=shape,
                    asymmetry=flag,
                    mean_forecast=meanf,
                )
            )
            for bi in range(nb):
                mk = codes == bi
                m3b.append(
                    dict(
                        est=e,
                        variable=g,
                        column=T.at[j, "column"],
                        bin=bi + 1,
                        n=int(mk.sum()),
                        x_lo=float(x[mk].min()),
                        x_hi=float(x[mk].max()),
                        x_mid=float(np.median(x[mk])),
                        mean_shap_var=float(yA[mk].mean()),
                        p10_shap_var=float(np.percentile(yA[mk], 10)),
                        p90_shap_var=float(np.percentile(yA[mk], 90)),
                        mean_shap_col=float(yj[mk].mean()),
                        mean_shap_var_pct_forecast=100 * float(yA[mk].mean()) / meanf,
                    )
                )
    m3 = pd.DataFrame(m3)
    m3b = pd.DataFrame(m3b)
    m3.to_csv(RES / "m3_shapes.csv", index=False)
    m3b.to_csv(RES / "m3_bins.csv", index=False)
    for e in ests:
        out(f"{C.EST_NAME[e]}:")
        for r in m3[m3["est"] == e].itertuples():
            out(
                f"  {r.variable:30s} {pct(r.share, 2):>5s}% {r.selection:10s} x={r.dominant_column:34s} bins {r.bins:2d} "
                f"-> {r.shape}{(' + ' + r.asymmetry) if r.asymmetry else ''}  (range {100 * r.range / r.mean_forecast:.2f}% of "
                f"mean forecast; R/I {r.range / r.I_A:.2f}; rho {r.spearman:+.2f})"
            )

    # ------------------------------------------------------------------ M4
    out(
        "\n== M4 conditional importance (NON-STANDARD share of the total / of the exogenous total, %)"
    )
    vix = pd.read_parquet(C.REPO / "data" / "free_feed" / "vix_1d_yahoo.parquet")
    vix["t"] = pd.to_datetime(vix["t"]).dt.normalize()
    vix = vix.set_index("t")["close"].sort_index()
    vix_o = np.array(
        [float(vix[vix.index < dd.normalize()].iloc[-1]) for dd in o["origin"]]
    )
    terc = pd.qcut(vix_o, 3, labels=False)
    rel = pd.read_parquet(
        C.REPO / "data" / "releases.parquet", columns=["endbartime", "fomc release"]
    )
    src = (C.REPO / "src" / "data" / "loading.py").read_text(encoding="utf-8")
    assert 'for d in ("2020-03-03",)' in src  # UNSCHEDULED_FOMC_DAYS
    fdays = set(
        pd.to_datetime(rel.loc[rel["fomc release"] > 0, "endbartime"]).dt.normalize()
    ) - {pd.Timestamp("2020-03-03")}
    fomc = dates.normalize().isin(fdays)
    oinfo = o.assign(vix_prev_close=vix_o, vix_tercile=terc)
    oinfo.to_csv(RES / "m4_origins_vix.csv", index=False)
    out(
        f"VIX terciles at the origin (prior close): cut points {np.quantile(vix_o, [1 / 3, 2 / 3]).round(2).tolist()}, "
        f"origins per tercile {np.bincount(terc).tolist()}; FOMC statement days among forecast rows: "
        f"{len(set(dates[fomc].normalize()))} days, {int(fomc.sum())} rows"
    )
    rows_t = terc[korig]
    subsets = [
        ("vix", f"VIX {['low', 'mid', 'high'][t]}", rows_t == t) for t in range(3)
    ]
    subsets += [("bar", b, bar == b) for b in sorted(set(bar))]
    subsets += [("fomc", "FOMC day", fomc), ("fomc", "other day", ~fomc)]
    m4 = []
    sub_sums = {}
    for e in ests:
        PG = np.abs(data[e]["PG"])
        for kind, name, mk in subsets:
            S = np.zeros((K, 48))
            np.add.at(S, korig[mk], PG[mk])
            sub_sums[e, name] = S
            bt = cnt @ S
            tot = S.sum()
            ns, nsx = (
                S[:, NS].sum() / tot,
                S[:, np.intersect1d(NS, EX)].sum() / S[:, EX].sum(),
            )
            lo, hi = ci(bt[:, NS].sum(1) / bt.sum(1))
            sh = S.sum(0) / tot
            top = [C.NONSTANDARD[i] for i in np.argsort(-sh[NS])[:5]]
            rec = dict(
                est=e,
                kind=kind,
                subset=name,
                rows=int(mk.sum()),
                ns_share=ns,
                ns_lo=lo,
                ns_hi=hi,
                ns_share_exog=nsx,
                top5_nonstandard=" ".join(f"{g}:{pct(sh[GI[g]], 2)}" for g in top),
            )
            for b in C.BUCKETS:
                rec[f"bucket_{b}"] = sum(
                    sh[GI[g]] for g in C.GROUPS if C.CAT_OF[g] == b
                )
            m4.append(rec)
    m4 = pd.DataFrame(m4)
    m4.to_csv(RES / "m4_subsets.csv", index=False)
    # contrasts
    con = []
    for e in ests:
        for label, a_, b_ in (
            ("VIX high - low", "VIX high", "VIX low"),
            ("FOMC day - other day", "FOMC day", "other day"),
            ("16:00 - 10:00", "16:00", "10:00"),
        ):
            Sa, Sb = sub_sums[e, a_], sub_sums[e, b_]
            ba, bb = cnt @ Sa, cnt @ Sb
            for scope, num, den in (
                ("of total", NS, np.arange(48)),
                ("of exogenous", np.intersect1d(NS, EX), EX),
            ):
                v = (
                    Sa[:, num].sum() / Sa[:, den].sum()
                    - Sb[:, num].sum() / Sb[:, den].sum()
                )
                dd = ba[:, num].sum(1) / ba[:, den].sum(1) - bb[:, num].sum(1) / bb[
                    :, den
                ].sum(1)
                lo, hi = ci(dd)
                con.append(
                    dict(
                        est=e,
                        contrast=label,
                        scope=scope,
                        diff=v,
                        lo=lo,
                        hi=hi,
                        verdict="higher"
                        if lo > 0
                        else "lower"
                        if hi < 0
                        else "no clear difference",
                    )
                )
            for b in ("fomc", "implied_vol", "har_ma", "calendar"):
                idx = np.array([GI[g] for g in C.GROUPS if C.CAT_OF[g] == b])
                v = Sa[:, idx].sum() / Sa.sum() - Sb[:, idx].sum() / Sb.sum()
                dd = ba[:, idx].sum(1) / ba.sum(1) - bb[:, idx].sum(1) / bb.sum(1)
                lo, hi = ci(dd)
                con.append(
                    dict(
                        est=e,
                        contrast=label,
                        scope=f"bucket {b} of total",
                        diff=v,
                        lo=lo,
                        hi=hi,
                        verdict="higher"
                        if lo > 0
                        else "lower"
                        if hi < 0
                        else "no clear difference",
                    )
                )
    con = pd.DataFrame(con)
    con.to_csv(RES / "m4_contrasts.csv", index=False)
    for e in ests:
        t = m4[m4["est"] == e]
        out(f"{C.EST_NAME[e]}:")
        for r in t.itertuples():
            out(
                f"  {r.subset:10s} rows {r.rows:6d}  NS {pct(r.ns_share)} [{pct(r.ns_lo)}, {pct(r.ns_hi)}]  NS of exog "
                f"{pct(r.ns_share_exog)}  FOMC bucket {pct(r.bucket_fomc, 2)}  HAR {pct(r.bucket_har_ma)}  top NS: "
                f"{r.top5_nonstandard}"
            )
        for r in con[con["est"] == e].itertuples():
            out(
                f"  {r.contrast:22s} {r.scope:24s} {100 * r.diff:+.2f} [{100 * r.lo:+.2f}, {100 * r.hi:+.2f}] {r.verdict}"
            )

    # ------------------------------------------------------------------ M5
    out("\n== M5 interactions (LightGBM, XGBoost; 8 rows a neighbourhood, 1,008 rows)")
    Q = np.zeros((len(C.BUCKETS), 48))
    for i, g in enumerate(C.GROUPS):
        Q[C.BUCKETS.index(C.CAT_OF[g]), i] = 1.0
    m5s, m5v, m5b, m5p = [], [], [], []
    for e in [e for e in ests if "G" in data[e]]:
        G = data[e]["G"]
        n = len(G)
        aG = np.abs(G)
        diag = np.abs(np.einsum("naa->na", G))
        main_v, inter_v = diag.sum(), aG.sum() - diag.sum()
        cm, cint = data[e]["col_main"].sum(), data[e]["col_inter"].sum()
        xa_num = aG.sum(2) - diag  # n x 48
        xa_den = aG.sum(2)
        cls_x = {}
        for name, idx in (
            ("NON-STANDARD", NS),
            ("STANDARD", np.setdiff1d(np.arange(48), NS)),
        ):
            cls_x[name] = xa_num[:, idx].sum() / xa_den[:, idx].sum()
        m5s.append(
            dict(
                est=e,
                rows=n,
                inter_share_variable=inter_v / (main_v + inter_v),
                inter_share_column=cint / (cm + cint),
                inter_frac_nonstandard=cls_x["NON-STANDARD"],
                inter_frac_standard=cls_x["STANDARD"],
            )
        )
        for i, g in enumerate(C.GROUPS):
            m5v.append(
                dict(
                    est=e,
                    variable=g,
                    cls=C.CLASS_OF[g],
                    bucket=C.CAT_OF[g],
                    main_abs=diag[:, i].mean(),
                    inter_abs=xa_num[:, i].mean(),
                    inter_frac=xa_num[:, i].sum() / max(xa_den[:, i].sum(), 1e-300),
                )
            )
        Gb = Q @ G @ Q.T  # n x 10 x 10
        Tm = np.zeros((len(C.BUCKETS), len(C.BUCKETS)))
        for p in range(len(C.BUCKETS)):
            for q in range(len(C.BUCKETS)):
                Tm[p, q] = (
                    np.abs(Gb[:, p, q]).mean()
                    if p == q
                    else np.abs(2 * Gb[:, p, q]).mean()
                )
        tri = np.triu(Tm).sum()
        for p, bp in enumerate(C.BUCKETS):
            for q, bq in enumerate(C.BUCKETS):
                if q >= p:
                    m5b.append(
                        dict(
                            est=e,
                            bucket_a=bp,
                            bucket_b=bq,
                            T=Tm[p, q],
                            share=Tm[p, q] / tri,
                        )
                    )
        TA = np.abs(2 * G).mean(0)
        np.fill_diagonal(TA, np.abs(np.einsum("naa->na", G)).mean(0))
        triA = np.triu(TA).sum()
        for i in range(48):
            for j in range(i + 1, 48):
                gi, gj = C.GROUPS[i], C.GROUPS[j]
                if gi in C.NONSTANDARD or gj in C.NONSTANDARD:
                    m5p.append(
                        dict(
                            est=e,
                            var_a=gi,
                            var_b=gj,
                            T=TA[i, j],
                            share=TA[i, j] / triA,
                            both_nonstandard=gi in C.NONSTANDARD
                            and gj in C.NONSTANDARD,
                        )
                    )
    m5s, m5v, m5b, m5p = (pd.DataFrame(x) for x in (m5s, m5v, m5b, m5p))
    m5s.to_csv(RES / "m5_summary.csv", index=False)
    m5v.to_csv(RES / "m5_variables.csv", index=False)
    m5b.to_csv(RES / "m5_buckets.csv", index=False)
    m5p = m5p.sort_values(["est", "T"], ascending=[True, False])
    m5p.to_csv(RES / "m5_pairs_nonstandard.csv", index=False)
    for r in m5s.itertuples():
        out(
            f"{C.EST_NAME[r.est]:14s} interaction share: variable level {pct(r.inter_share_variable)}%, column level "
            f"{pct(r.inter_share_column)}%; share of a variable's attribution through interactions: NON-STANDARD "
            f"{pct(r.inter_frac_nonstandard)}%, STANDARD {pct(r.inter_frac_standard)}%"
        )
    for e in m5s["est"]:
        b = m5b[m5b["est"] == e].sort_values("share", ascending=False).head(8)
        out(
            f"  {e} largest bucket cells: "
            + "; ".join(
                f"{r.bucket_a} x {r.bucket_b} {pct(r.share)}" for r in b.itertuples()
            )
        )
        p = m5p[m5p["est"] == e].head(10)
        out(
            f"  {e} top 10 pairs with a NON-STANDARD variable: "
            + "; ".join(
                f"{r.var_a} x {r.var_b} {pct(r.share, 2)}" for r in p.itertuples()
            )
        )
        v = (
            m5v[(m5v["est"] == e) & (m5v["cls"] == "NON-STANDARD")]
            .sort_values("main_abs", ascending=False)
            .head(8)
        )
        out(
            f"  {e} NON-STANDARD interaction fraction (largest 8 by main effect): "
            + "; ".join(f"{r.variable} {pct(r.inter_frac)}" for r in v.itertuples())
        )

    (RES / "tables.txt").write_text("\n".join(TXT) + "\n", encoding="utf-8")
    info = dict(
        created=time.strftime("%Y-%m-%d %H:%M"),
        rule="RULE_shap.txt",
        origins=K,
        forecast_rows=int(len(rows)),
        bootstrap=dict(
            kind="circular moving blocks over origins",
            block=BLOCK,
            B=B_BOOT,
            seed=C.SEED,
        ),
        models=ests,
        runner_sha=C.RUNNER_SHA,
        fiw_sha=C.FIW_SHA,
        seconds=round(time.time() - t0, 1),
    )
    for e in ests:
        for f in sorted((work / e).glob("run_info*.json")):
            info.setdefault("runs", {})[f"{e}/{f.name}"] = json.loads(f.read_text())
    (RES / "run_info.json").write_text(json.dumps(info, indent=1, default=str))
    return 0


def classify(m: np.ndarray, I_A: float) -> tuple[str, str, float, float]:
    """RULE_shap.txt M3 shape rule on the bin means m_1..m_B."""
    B = len(m)
    R = float(m.max() - m.min())
    rho = (
        float(spearmanr(np.arange(B), m).statistic)
        if B > 2
        else float(np.sign(m[-1] - m[0]))
    )
    if R < 0.5 * I_A:
        return "FLAT", "", rho, R
    if B <= 3:
        shape = "STEP " + ("up" if m[-1] > m[0] else "down")
    elif abs(rho) >= 0.9:
        tot = abs(m[-1] - m[0])
        shape = (
            "THRESHOLD" if np.abs(np.diff(m)).max() >= 0.5 * tot else "MONOTONE"
        ) + (" up" if rho > 0 else " down")
    else:
        kmin, kmax = int(np.argmin(m)), int(np.argmax(m))
        if (
            0 < kmin < B - 1
            and m[0] - m[kmin] >= 0.25 * R
            and m[-1] - m[kmin] >= 0.25 * R
        ):
            shape = "U-SHAPED"
        elif (
            0 < kmax < B - 1
            and m[kmax] - m[0] >= 0.25 * R
            and m[kmax] - m[-1] >= 0.25 * R
        ):
            shape = "INVERTED-U"
        else:
            shape = "IRREGULAR"
    mc = m[B // 2] if B % 2 else 0.5 * (m[B // 2 - 1] + m[B // 2])
    lo, hi = abs(m[0] - mc), abs(m[-1] - mc)
    flag = ""
    if max(lo, hi) >= 2 * min(lo, hi):
        flag = "ASYMMETRIC (" + ("top-tail" if hi > lo else "bottom-tail") + ")"
    return shape, flag, rho, R


def load_x(work: Path, rows: np.ndarray) -> np.ndarray:
    f = work / "x_forecast_rows.npz"
    if f.is_file():
        z = np.load(f)
        assert int(z["r0"]) == rows[0] and int(z["r1"]) == rows[-1] + 1
        return z["x"]
    X = np.load(C.DESIGN / "_work" / "design_all_features" / "X.npy", mmap_mode="r")
    return np.asarray(X[rows[0] : rows[-1] + 1], dtype=np.float32)


def load_cols(
    work: Path, est: str, o: pd.DataFrame, cols: list[int]
) -> dict[int, np.ndarray]:
    parts: dict[int, list] = {j: [] for j in cols}
    for k in o["k"]:
        ph = np.load(work / est / f"k{k:03d}.npz")["phi"]
        for j in cols:
            parts[j].append(ph[:, j].astype(np.float64))
    return {j: np.concatenate(v) for j, v in parts.items()}


if __name__ == "__main__":
    raise SystemExit(main())
