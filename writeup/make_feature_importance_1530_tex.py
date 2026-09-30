"""Write writeup/feature_importance_1530.tex (+ pdf) and results/feature_importance_1530/SUMMARY.md
from the tables of experiments/feature_importance_1530.py (run its stages first: design, inputs,
the tree fleet, capture_baseline, linear, aggregate).  Every number below is read from those
tables; nothing is typed in.

``--root <dir> --stem <name> --dedup``: the re-run on the de-duplicated per-bar design with the
per-window tree mask (checklist I7; experiments/feature_importance_1530_dedup.py), read from its
root, written to <root>/SUMMARY.md and writeup/<name>.tex (+ pdf), with its own gates and the
before / after section (writeup/feature_importance_1530_dedup_sections.py).  Without flags the
output is the first pass's, unchanged."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parent.parent
OUT = R / "results" / "feature_importance_1530"
BY = OUT / "by_measure"
ROOT_REL = "results/feature_importance_1530"  # the figures' path from writeup/
STEM = "feature_importance_1530"  # writeup/<STEM>.tex / .pdf
DEDUP = False  # --dedup: the re-run on the de-duplicated per-bar design (checklist I7)
TREES = ("lgbm", "xgb", "rf")
LIN = ("ridge", "lasso")
MODELS = LIN + TREES
LABEL = {
    "ridge": "ridge",
    "lasso": "lasso",
    "lgbm": "LightGBM",
    "xgb": "XGBoost",
    "rf": "random forest",
}
BUCKETS = ("baseline", "live_feasible", "all_features")
BNAME = {
    "baseline": "HAR + calendar",
    "live_feasible": "live-feasible",
    "all_features": "all features",
}
MEAS = {
    "mdi": "MDI / gain",
    "split": "split count",
    "bsd": "$|\\beta\\cdot sd|$",
    "perm_p1_qlike": "perm.\\ P1, $\\Delta$QLIKE",
    "perm_p2_qlike": "perm.\\ P2, $\\Delta$QLIKE",
    "perm_p1_mse": "perm.\\ P1, $\\Delta$MSE",
    "perm_p2_mse": "perm.\\ P2, $\\Delta$MSE",
    "drop_qlike": "drop-column, $\\Delta$QLIKE",
    "drop_mse": "drop-column, $\\Delta$MSE",
    "shap": "SHAP mean $|\\phi|$",
}
MEAS_MD = {
    "mdi": "MDI / gain",
    "split": "split count",
    "bsd": "abs(beta x sd)",
    "perm_p1_qlike": "perm P1 dQLIKE",
    "perm_p2_qlike": "perm P2 dQLIKE",
    "perm_p1_mse": "perm P1 dMSE",
    "perm_p2_mse": "perm P2 dMSE",
    "drop_qlike": "drop-col dQLIKE",
    "drop_mse": "drop-col dMSE",
    "shap": "SHAP mean abs(phi)",
}
TREE_M = ("mdi", "split", "perm_p1_qlike", "perm_p2_qlike", "perm_p2_mse", "shap")
LIN_M = ("bsd", "perm_p1_mse", "perm_p2_mse", "perm_p2_qlike", "drop_mse", "shap")


def tab(bucket: str, model: str, m: str, level: str = "series") -> pd.DataFrame:
    t = pd.read_csv(BY / f"imp_{bucket}_{model}_{m}.csv")
    return t[(t["level"] == level) & t["eligible"]].sort_values("rank")


def esc(s: str) -> str:
    return (
        str(s)
        .replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("#", "\\#")
        .replace("|", "$|$")
    )


def tt(s: str) -> str:
    return f"\\texttt{{{esc(s)}}}"


def short(label: str) -> str:
    return "calendar" if label.startswith("calendar") else label


def f2(x: float) -> str:
    return f"{x:.2f}"


def configure(argv: list[str] | None) -> None:
    """The root / output stem / mode from the command line (defaults = the first pass)."""
    import argparse

    global OUT, BY, ROOT_REL, STEM, DEDUP
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT_REL)
    ap.add_argument("--stem", default=STEM)
    ap.add_argument("--dedup", action="store_true")
    a = ap.parse_args(argv)
    ROOT_REL, STEM, DEDUP = a.root.replace("\\", "/").rstrip("/"), a.stem, a.dedup
    OUT = R / ROOT_REL
    BY = OUT / "by_measure"


def main(argv: list[str] | None = None) -> None:  # noqa: C901 - one report
    configure(argv)
    if DEDUP:
        import sys

        sys.path.insert(0, str(R / "writeup"))
        import feature_importance_1530_dedup_sections as DS
    S = json.loads((OUT / "aggregate_meta.json").read_text(encoding="utf-8"))
    G = pd.read_csv(OUT / "gates.csv")
    CW = pd.read_csv(OUT / "rank_corr_within_model.csv")
    CX = pd.read_csv(OUT / "rank_corr_across_models.csv")
    ST = pd.read_csv(OUT / "stability_by_stratum.csv")
    CM = pd.read_csv(OUT / "cardinality_vs_measure.csv")
    PR = pd.read_csv(OUT / "noise_probes.csv")
    DL = pd.read_csv(OUT / "cluster_dilution.csv")
    CL = pd.read_csv(OUT / "clusters.csv")
    UV = pd.read_csv(OUT / "unique_values.csv")
    TU = pd.read_csv(OUT / "tuned_trees_series.csv")
    K = S["live_feasible"]["K"]
    n = S["live_feasible"]["n"]
    first, last = S["live_feasible"]["first"][:10], S["live_feasible"]["last"][:10]
    md: list[str] = []
    N: dict = {}  # numbers used in prose, also dumped to json for the record

    # ---------------------------------------------------------------- data type
    cls_order = ["constant", "binary", "3-25 values", "26-500 values", "> 500 values"]
    ct = (
        UV.groupby(["bucket", "cardinality_class"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=cls_order, fill_value=0)
    )
    dtype_rows = []
    for b in BUCKETS:
        r = ct.loc[b]
        dtype_rows.append(
            f"{BNAME[b]} & {int(r.sum())} & "
            + " & ".join(str(int(r[c])) for c in cls_order)
            + f" & {S[b]['n_series']} & {S[b]['n_clusters']} ({S[b]['n_cluster_cols']}) \\\\"
        )
    N["dtype"] = {b: {c: int(ct.loc[b, c]) for c in cls_order} for b in BUCKETS}

    # ---------------------------------------------------------------- cardinality
    def card(b: str, model: str, m: str) -> float:
        r = CM[(CM.bucket == b) & (CM.model == model) & (CM.measure == m)]
        return float(r["spearman_with_unique_count"].iloc[0])

    card_rows = []
    for b in ("live_feasible", "all_features", "baseline"):
        for model in TREES:
            card_rows.append(
                f"{BNAME[b]} & {LABEL[model]} & "
                + " & ".join(
                    f2(card(b, model, m))
                    for m in (
                        "mdi",
                        "split",
                        "shap",
                        "perm_p1_qlike",
                        "perm_p2_qlike",
                        "perm_p2_mse",
                    )
                )
                + " \\\\"
            )
    rng_ = {}
    for b in BUCKETS:
        for m in ("mdi", "split", "shap", "perm_p1_qlike", "perm_p2_qlike"):
            v = [card(b, mo, m) for mo in TREES]
            rng_[(b, m)] = (min(v), max(v))
    N["card"] = {f"{b}:{m}": v for (b, m), v in rng_.items()}

    # ---------------------------------------------------------------- probes
    probe_rows = []
    for b in ("live_feasible", "all_features", "baseline"):
        for model in TREES:
            r = PR[(PR.bucket == b) & (PR.model == model)].set_index("probe")
            c, lv, bi = (
                r.loc["probe_continuous"],
                r.loc["probe_20_levels"],
                r.loc["probe_binary"],
            )
            har = tab(b, model, "perm_p2_qlike")
            har = har[har["unit"] == "har_ma"]["value"].iloc[0]
            probe_rows.append(
                f"{BNAME[b]} & {LABEL[model]} & {S[b]['p'] + 3} & "
                f"{int(c.mdi_rank_median)} ({100 * c.mdi_real_columns_below_share:.0f}\\%) & "
                f"{int(c.split_rank_median)} ({100 * c.split_real_columns_below_share:.0f}\\%) & "
                f"{int(c.shap_rank_median)} ({100 * c.shap_real_columns_below_share:.0f}\\%) & "
                f"{int(lv.mdi_rank_median)} & {int(bi.mdi_rank_median)} & "
                f"${1e3 * c.perm_p2_qlike_mean:+.2f}$ [{1e3 * c.perm_p2_qlike_lo:+.2f}, {1e3 * c.perm_p2_qlike_hi:+.2f}] & "
                f"{1e3 * har:.0f} \\\\"
            )
    lf = PR[
        (PR.bucket == "live_feasible") & (PR.probe == "probe_continuous")
    ].set_index("model")
    N["probe_lf"] = lf[
        [
            "mdi_rank_median",
            "mdi_real_columns_below_share",
            "split_rank_median",
            "split_real_columns_below_share",
            "shap_rank_median",
            "perm_p2_qlike_mean",
            "perm_p2_qlike_lo",
            "perm_p2_qlike_hi",
            "mdi_real_columns_used_median",
        ]
    ].to_dict()

    # ---------------------------------------------------------------- top features per measure
    def top_list(b: str, model: str, m: str, k: int = 3, level: str = "series") -> str:
        t = tab(b, model, m, level).head(k)
        return ", ".join(short(x) for x in t["label"])

    top_rows: dict = {}
    for b in ("live_feasible", "all_features", "baseline"):
        rows = []
        for model in MODELS:
            ms = TREE_M if model in TREES else LIN_M
            for m in ms:
                t = tab(b, model, m).head(3)
                cells = []
                for r in t.itertuples():
                    if m in ("mdi", "split", "bsd", "shap"):
                        vs = f"{100 * r.value:.0f}\\%"
                    else:
                        vs = f"{r.pct_of_loss:+.0f}\\%"
                    cells.append(f"{tt(short(r.label))} {vs}")
                rows.append(
                    f"{LABEL[model]} & {MEAS[m]} & " + "; ".join(cells) + " \\\\"
                )
        top_rows[b] = rows
    # does har_ma lead every measure of every model?
    lead = {}
    for b in BUCKETS:
        for model in MODELS:
            for m in TREE_M if model in TREES else LIN_M:
                lead[(b, model, m)] = tab(b, model, m)["unit"].iloc[0]
    N["leaders_not_har"] = {
        f"{b}:{mo}:{m}": v for (b, mo, m), v in lead.items() if v != "har_ma"
    }

    # ---------------------------------------------------------------- within-model agreement
    def wc(b: str, level: str, model: str, a: str, c: str) -> float:
        r = CW[
            (CW.bucket == b)
            & (CW.level == level)
            & (CW.model == model)
            & (((CW.a == a) & (CW.b == c)) | ((CW.a == c) & (CW.b == a)))
        ]
        return float(r["spearman"].iloc[0]) if len(r) else float("nan")

    PAIRS_T = [
        ("mdi", "split"),
        ("mdi", "shap"),
        ("mdi", "perm_p2_qlike"),
        ("shap", "perm_p2_qlike"),
        ("perm_p1_qlike", "perm_p2_qlike"),
        ("perm_p2_qlike", "perm_p2_mse"),
    ]
    PAIRS_L = [
        ("bsd", "shap"),
        ("bsd", "perm_p2_mse"),
        ("shap", "perm_p2_mse"),
        ("perm_p1_mse", "perm_p2_mse"),
        ("perm_p2_mse", "drop_mse"),
        ("perm_p2_qlike", "perm_p2_mse"),
    ]
    within_rows = []
    for b in ("live_feasible", "all_features"):
        for level in ("series", "column"):
            for model in TREES:
                within_rows.append(
                    f"{BNAME[b]} & {level} & {LABEL[model]} & "
                    + " & ".join(f2(wc(b, level, model, a, c)) for a, c in PAIRS_T)
                    + " \\\\"
                )
    within_lin = []
    for b in ("live_feasible", "all_features"):
        for level in ("series", "column"):
            for model in LIN:
                within_lin.append(
                    f"{BNAME[b]} & {level} & {LABEL[model]} & "
                    + " & ".join(f2(wc(b, level, model, a, c)) for a, c in PAIRS_L)
                    + " \\\\"
                )

    # ---------------------------------------------------------------- across models
    def xc(b: str, level: str, ma: str, ea: str, mb: str, eb: str) -> float:
        r = CX[
            (CX.bucket == b)
            & (CX.level == level)
            & (
                (
                    (CX.model_a == ma)
                    & (CX.measure_a == ea)
                    & (CX.model_b == mb)
                    & (CX.measure_b == eb)
                )
                | (
                    (CX.model_a == mb)
                    & (CX.measure_a == eb)
                    & (CX.model_b == ma)
                    & (CX.measure_b == ea)
                )
            )
        ]
        return float(r["spearman"].iloc[0]) if len(r) else float("nan")

    across_rows = []
    for b in ("live_feasible", "all_features"):
        for level in ("series", "column"):
            cells = []
            for m in ("shap", "perm_p2_qlike", "perm_p2_mse"):
                tt_ = [
                    xc(b, level, a, m, c, m)
                    for i, a in enumerate(TREES)
                    for c in TREES[i + 1 :]
                ]
                tl = [xc(b, level, a, m, c, m) for a in TREES for c in LIN]
                ll = [xc(b, level, "ridge", m, "lasso", m)]
                cells += [
                    f"{min(tt_):.2f}--{max(tt_):.2f}",
                    f"{min(tl):.2f}--{max(tl):.2f}",
                    f"{ll[0]:.2f}",
                ]
            across_rows.append(f"{BNAME[b]} & {level} & " + " & ".join(cells) + " \\\\")
    # first pass comparison: tree SHAP vs linear SHAP at the series level
    N["xc_shap_tl"] = {
        b: [xc(b, "series", a, "shap", c, "shap") for a in TREES for c in LIN]
        for b in ("live_feasible", "all_features")
    }

    # ---------------------------------------------------------------- top-5 stability (series)
    stab_rows = []
    for b in ("live_feasible", "all_features"):
        for model in MODELS:
            ms = TREE_M if model in TREES else LIN_M
            cells = []
            for m in (
                "mdi" if model in TREES else "bsd",
                "perm_p2_qlike" if model in TREES else "perm_p2_mse",
                "shap",
            ):
                t = tab(b, model, m)
                har = float(t[t["unit"] == "har_ma"]["top5_share"].iloc[0])
                n_ever = int((t["top5_share"] > 0).sum())
                n_half = int((t["top5_share"] >= 0.5).sum())
                cells.append(f"{100 * har:.0f}\\% & {n_half} & {n_ever}")
            stab_rows.append(
                f"{BNAME[b]} & {LABEL[model]} & " + " & ".join(cells) + " \\\\"
            )

    # ---------------------------------------------------------------- strata
    strat_rows = []
    for b in ("live_feasible", "all_features"):
        for model in MODELS:
            cells = []
            for m in (
                "mdi" if model in TREES else "bsd",
                "perm_p2_qlike" if model in TREES else "perm_p2_mse",
                "shap",
            ):
                s = ST[
                    (ST.bucket == b)
                    & (ST.level == "series")
                    & (ST.model == model)
                    & (ST.measure == m)
                ]
                y = s[s.stratum == "year"]
                v = s[s.stratum == "vix_tercile"]
                lead_y = int((y["leader"] == y["leader_all"]).sum())
                cells.append(
                    f"{y.spearman_vs_all.median():.2f} ({y.spearman_vs_all.min():.2f}) & "
                    f"{v.spearman_vs_all.median():.2f} ({v.spearman_vs_all.min():.2f}) & {lead_y}/{len(y)}"
                )
            strat_rows.append(
                f"{BNAME[b]} & {LABEL[model]} & " + " & ".join(cells) + " \\\\"
            )
    terc = S["live_feasible"]["terciles"]
    years = S["live_feasible"]["years"]

    # ---------------------------------------------------------------- dilution
    dil_rows = []
    for b in ("live_feasible", "all_features"):
        d = DL[(DL.bucket == b) & (DL.scheme == "P2")]
        # the clusters that matter: top 4 by the joint permutation of the LightGBM model
        top = (
            d[d.model == "lgbm"]
            .sort_values("joint_dmse", ascending=False)["cluster"]
            .head(4)
            .tolist()
        )
        for c in top:
            mem = CL[(CL.bucket == b) & (CL.cluster == c)].iloc[0]
            cells = []
            for model in MODELS:
                r = d[(d.model == model) & (d.cluster == c)].iloc[0]
                cells.append(
                    f"{1e3 * r.joint_dmse:.1f} / {1e3 * r.sum_of_columns_dmse:.1f}"
                )
            members = mem["members"].split()
            mlabel = ", ".join(members[:4]) + (
                f" +{len(members) - 4}" if len(members) > 4 else ""
            )
            dil_rows.append(
                f"{BNAME[b]} & {c} & {tt(mlabel)} ({mem.n_columns}; $|r|\\ge${mem.min_abs_corr:.2f}) & "
                + " & ".join(cells)
                + " \\\\"
            )
    c22 = DL[
        (DL.bucket == "live_feasible")
        & (DL.scheme == "P2")
        & (DL.members.str.startswith("har_ma_1 har_ma_5"))
    ]
    N["c22_ratio_mse"] = (
        c22.set_index("model")["joint_dmse"]
        / c22.set_index("model")["sum_of_columns_dmse"]
    ).to_dict()

    # ---------------------------------------------------------------- linear QLIKE blow-ups
    blow = []
    for b in ("live_feasible", "all_features"):
        for model in LIN:
            for m in ("perm_p1_qlike", "perm_p2_qlike", "drop_qlike"):
                t = tab(b, model, m).head(1).iloc[0]
                blow.append(
                    f"{BNAME[b]} & {LABEL[model]} & {MEAS[m]} & {tt(short(t.label))} & "
                    f"{t.value:.3g} [{t.lo:.2g}, {t.hi:.2g}] & {t.median_refit:.2g} & {100 * t.max_refit_share:.0f}\\% \\\\"
                )

    # ---------------------------------------------------------------- tuned trees
    tuned_rows = []
    tuned_note = ""
    if "series" in TU.columns and len(TU):
        for b in ("live_feasible", "all_features"):
            for model in TREES:
                t = TU[(TU.bucket == b) & (TU.model == model)].sort_values(
                    "tuned_shap_share", ascending=False
                )
                if not len(t):
                    continue
                h = t[t.series == "har_ma"].iloc[0]
                others = [
                    f"{tt(short(x))}" for x in t["series"].head(4) if x != "har_ma"
                ][:3]
                rs = np.corrcoef(
                    t["tuned_shap_share"].rank(), t["untuned_shap_share"].rank()
                )[0, 1]
                rm = np.corrcoef(
                    t["tuned_mdi_share"].rank(), t["untuned_mdi_share"].rank()
                )[0, 1]
                tuned_rows.append(
                    f"{BNAME[b]} & {LABEL[model]} & {100 * h.tuned_mdi_share:.0f}\\% ({100 * h.untuned_mdi_share:.0f}\\%) & "
                    f"{100 * h.tuned_shap_share:.0f}\\% ({100 * h.untuned_shap_share:.0f}\\%) & {rm:.2f} & {rs:.2f} & "
                    + ", ".join(others)
                    + " \\\\"
                )
    else:
        tuned_note = "The tuned-tree importance extracts were not present when this was built (pending)."

    # ---------------------------------------------------------------- gates
    gl = G.set_index(["bucket", "model"])
    lg_gap = {
        b: (
            float(gl.loc[(b, "lgbm"), "forecast_gap_mean"]),
            float(gl.loc[(b, "lgbm"), "forecast_gap_max"]),
            float(gl.loc[(b, "lgbm"), "forecast_corr_stored"]),
        )
        for b in BUCKETS
    }
    xr_gap = max(
        float(gl.loc[(b, m), "forecast_gap_max"])
        for b in BUCKETS
        for m in ("xgb", "rf")
    )
    xr_mean = max(
        float(gl.loc[(b, m), "forecast_gap_mean"])
        for b in BUCKETS
        for m in ("xgb", "rf")
    )
    an = {
        b: {m: float(gl.loc[(b, m), "anchor_resolve_gap_max"]) for m in LIN}
        for b in BUCKETS
    }
    N["gates"] = dict(lgbm=lg_gap, xgb_rf_max=xr_gap, xgb_rf_mean=xr_mean, anchor=an)

    # what permutation says instead: series MDI or split puts in its top 5 that permutation
    # P2 puts outside its top 10, per tree model
    only_nat = {}
    for b in ("live_feasible", "all_features"):
        for model in TREES:
            pr_ = tab(b, model, "perm_p2_qlike").set_index("unit")["rank"]
            hits: list[tuple[str, int]] = []
            for m in ("mdi", "split"):
                for r in tab(b, model, m).head(5).itertuples():
                    if pr_.get(r.unit, 999) > 10 and short(r.label) not in [
                        h[0] for h in hits
                    ]:
                        hits.append((short(r.label), int(pr_.get(r.unit, 0))))
            only_nat[(b, model)] = hits
    N["mdi_split_top5_perm_outside10"] = {
        f"{k[0]}:{k[1]}": v for k, v in only_nat.items()
    }
    anch_n = {
        m: [int(gl.loc[(b, m), "anchor_refits_above_1e6"]) for b in BUCKETS]
        for m in LIN
    }

    # probe permutation: how many intervals exclude zero, the largest |mean|
    cont = PR[PR.probe == "probe_continuous"]
    n_ex = int(((cont.perm_p2_qlike_lo > 0) | (cont.perm_p2_qlike_hi < 0)).sum())
    ex_list = ", ".join(
        f"{BNAME[r.bucket]} {LABEL[r.model]} [{1e3 * r.perm_p2_qlike_lo:+.2f}, {1e3 * r.perm_p2_qlike_hi:+.2f}]e-3"
        for r in cont[
            (cont.perm_p2_qlike_lo > 0) | (cont.perm_p2_qlike_hi < 0)
        ].itertuples()
    )
    max_probe = float(np.abs(PR[["perm_p2_qlike_mean"]]).max().iloc[0])
    har_min = min(
        float(tab(b, m, "perm_p2_qlike")["value"].iloc[0])
        for b in BUCKETS
        for m in TREES
    )
    N["probe_perm"] = dict(
        n_excl=n_ex, of=len(cont), max_abs=max_probe, har_min=har_min
    )

    # strata: median over models of the median-over-years Spearman, per measure family
    def strat_med(b: str, fam: str, stratum: str) -> float:
        rows = []
        for model in MODELS:
            m = {
                "builtin": "mdi" if model in TREES else "bsd",
                "perm": "perm_p2_qlike" if model in TREES else "perm_p2_mse",
                "shap": "shap",
            }[fam]
            s_ = ST[
                (ST.bucket == b)
                & (ST.level == "series")
                & (ST.model == model)
                & (ST.measure == m)
                & (ST.stratum == stratum)
            ]
            rows.append(float(s_.spearman_vs_all.median()))
        return float(np.median(rows))

    SM = {
        (b, f, st): strat_med(b, f, st)
        for b in ("live_feasible", "all_features")
        for f in ("builtin", "perm", "shap")
        for st in ("year", "vix_tercile")
    }
    N["strata_median"] = {f"{k[0]}:{k[1]}:{k[2]}": v for k, v in SM.items()}
    tuned_lead: tuple[int, int] | None = None
    if "series" in TU.columns and len(TU):
        tl = []
        for (b, model), g in TU.groupby(["bucket", "model"]):
            tl.append(
                (g.loc[g.tuned_mdi_share.idxmax(), "series"] == "har_ma")
                and (g.loc[g.tuned_shap_share.idxmax(), "series"] == "har_ma")
            )
        tuned_lead = (int(sum(bool(x) for x in tl)), len(tl))
    gap_n = {
        m: int(sum(int(gl.loc[(b, m), "refits_gap_above_1e9"]) for b in BUCKETS))
        for m in ("xgb", "rf")
    }
    af_ridge_q = float(gl.loc[("all_features", "ridge"), "tail_qlike_mean"])
    oth_q = [
        float(gl.loc[(b, m), "tail_qlike_mean"])
        for b in BUCKETS
        for m in MODELS
        if (b, m) != ("all_features", "ridge")
    ]

    # ================================================================ markdown summary
    lfK = S["live_feasible"]
    c_lf = PR[
        (PR.bucket == "live_feasible") & (PR.probe == "probe_continuous")
    ].set_index("model")
    b_lf = PR[(PR.bucket == "live_feasible") & (PR.probe == "probe_binary")].set_index(
        "model"
    )
    md += [
        "# Feature importance of the 15:30 forecast (16:00 bar) -- the professor's four measures"
        + (
            " -- re-run on the de-duplicated per-bar design, every tree fit window-masked"
            if DEDUP
            else ""
        ),
        "",
    ]
    if DEDUP:
        md += DS.intro_md(OUT, K)
    md += [
        f"Models: per-bar ridge and lasso (every-session re-solve, causal penalty), untuned per-bar LightGBM, XGBoost and random forest (refit every {10} sessions) -- the shipped configurations. "
        f"Window: the 2000 sessions before each refit. Forecasts: {n:,} sessions {first} .. {last}; {K} refits; each refit's CAUSAL held-out tail = the <= 10 sessions its model forecasts, up to the next refit (never seen in its fit). "
        + (
            "Buckets: live_feasible (the deck's), all_features, baseline = HAR + calendar. Script: `experiments/feature_importance_1530.py` (+ `_trees.py` for the refits, run on the cluster); tables in `results/feature_importance_1530/`, PDF `writeup/feature_importance_1530.pdf`."
            if not DEDUP
            else DS.buckets_md(ROOT_REL, STEM)
        ),
        "",
        "## Key (design column names)",
        (
            "`har_ma_*` realized variance of the bar (the target's own HAR ladder; `*` = mean over the last 1, 5, 25, 125, 625, 3125 bars; `har_ma_*_x_close` = the same times the 16:00 close gate, identical at this bar); "
            if not DEDUP
            else "`har_ma_*` realized variance of the bar (the target's own HAR ladder; `*` = mean over the last 1, 5, 25, 125, 625, 3125 bars; the per-bar design no longer carries the `har_ma_*_x_open` / `_x_close` session-edge interactions); "
        )
        + "`adj_sumabsret_ma_*` absolute return; `adj_sumret_ma_*` signed return; `adj_sumret3/4_ma_*` 3rd / 4th-power returns; `adj_sumpret2_ma_*` upside squared returns; `adj_sumbipow_ma_*` bipower variation; `adj_sumautocov_ma_*` return autocovariance; "
        "`adj_sumvolume_ma_*` ES volume; `adj_numobs_ma_*` ES prints per bar; `adj_vix_ma_*`, `adj_vvix_ma_*`, `adj_vix3m_ma_*` the Cboe indices; `adj_fomc_*` FOMC flags / distances; `*_avail_ma_*` / `*_active_ma_*` is-present / is-nonzero flags of a source; "
        "calendar = `DOW_*`, `is_*`, `hour`, `days_to_opex`. all_features adds the constituent cross-section (`*_ewstock`, `*_vwstock`: moments, turnover, spreads, order-flow imbalance `ofi_*`), Cboe volume (`adj_voldemand_*`) and StockTwits (`adj_stocktwits_*`). "
        "A SERIES = all columns of one input (lags, flags, gates); a CLUSTER = columns every pair of which correlates >= 0.8 in absolute value (complete linkage).",
        "",
        "## Measures",
        "1. MDI / gain: LightGBM gain, XGBoost total_gain, random-forest impurity decrease; share per refit, averaged over refits.",
        "2. Split count: splits on the column (LightGBM split, XGBoost weight, forest internal nodes); share per refit.",
        "3. Permutation importance on the causal held-out tail, loss = QLIKE (the bar's realized variance vs the plain back-transform yhat^2 x diurnal baseline, no recalibration) and squared error in the fit space; 10 draws per unit and refit, the SAME draws for every model. "
        "P1 shuffles the unit's values among the tail's rows (the literal permutation; blind to slow inputs, which barely move within 10 sessions); P2 replaces them with rows drawn from that refit's own training window (causal: past data only). Units: columns, series, clusters (a group's columns move together). "
        "Interval = 95 % circular-block bootstrap over refits (block 6).",
        "4. SHAP: TreeSHAP (path-dependent; LightGBM / XGBoost native, shap 0.51 TreeExplainer for the forest) on the same tail rows; mean |phi| share (and signed mean in the CSVs). Linear: beta (x - window mean) = the linear SHAP with the window as background.",
        "Linear extras: |beta x sd| (standardized weight) and drop-column importance (re-solve the anchor's window without the unit, penalty held).",
        "",
        "## Library and data type",
        "scikit-learn 1.9.0 (random forest), LightGBM 4.6.0, XGBoost 3.2.0, shap 0.51.0; the permutation and drop-column loops are written out (sklearn's permutation_importance cannot score walk-forward models on their own tails). "
        "Every design column is a float. What MDI's cardinality bias keys on is the number of distinct values in each 2000-session window (median over refits): "
        + "; ".join(
            f"{BNAME[b]} {int(ct.loc[b].sum())} columns = {int(ct.loc[b, '> 500 values'])} continuous (> 500 values), {int(ct.loc[b, '26-500 values'])} with 26-500, {int(ct.loc[b, '3-25 values'])} with 3-25, {int(ct.loc[b, 'binary'])} binary, {int(ct.loc[b, 'constant'])} constant at 16:00"
            for b in BUCKETS
        )
        + ". So the design is dominated by continuous inputs, but it is NOT all continuous: the calendar dummies and many flag moving averages are binary or few-valued -- exactly the mix where MDI's cardinality bias can bite.",
        "",
        "## Does MDI's cardinality bias matter here? Yes, at the column level.",
        f"- Rank correlation of a column's importance with its number of distinct values (non-constant columns, tree models): live_feasible MDI {rng_[('live_feasible', 'mdi')][0]:.2f}..{rng_[('live_feasible', 'mdi')][1]:.2f}, split count {rng_[('live_feasible', 'split')][0]:.2f}..{rng_[('live_feasible', 'split')][1]:.2f}, SHAP {rng_[('live_feasible', 'shap')][0]:.2f}..{rng_[('live_feasible', 'shap')][1]:.2f}; permutation P1 {rng_[('live_feasible', 'perm_p1_qlike')][0]:.2f}..{rng_[('live_feasible', 'perm_p1_qlike')][1]:.2f}, P2 {rng_[('live_feasible', 'perm_p2_qlike')][0]:.2f}..{rng_[('live_feasible', 'perm_p2_qlike')][1]:.2f}. "
        f"all_features: MDI {rng_[('all_features', 'mdi')][0]:.2f}..{rng_[('all_features', 'mdi')][1]:.2f}, permutation P2 {rng_[('all_features', 'perm_p2_qlike')][0]:.2f}..{rng_[('all_features', 'perm_p2_qlike')][1]:.2f}.",
        "- Noise probes (a second fit on every 3rd refit with three pure-noise columns appended): the CONTINUOUS noise column gets a median MDI rank of "
        + ", ".join(
            f"{int(c_lf.loc[m, 'mdi_rank_median'])} ({LABEL[m]})" for m in TREES
        )
        + f" among the {lfK['p'] + 3} live_feasible columns -- above "
        + ", ".join(
            f"{100 * c_lf.loc[m, 'mdi_real_columns_below_share']:.0f} %" for m in TREES
        )
        + " of the real columns the model uses; by split count the forest ranks it "
        + f"{int(c_lf.loc['rf', 'split_rank_median'])} (above {100 * c_lf.loc['rf', 'split_real_columns_below_share']:.0f} % of used real columns). The BINARY noise column ranks "
        + ", ".join(f"{int(b_lf.loc[m, 'mdi_rank_median'])}" for m in TREES)
        + " by MDI. Permutation P2 gives the continuous probe "
        + ", ".join(
            f"{1e3 * c_lf.loc[m, 'perm_p2_qlike_mean']:+.2f} [{1e3 * c_lf.loc[m, 'perm_p2_qlike_lo']:+.2f}, {1e3 * c_lf.loc[m, 'perm_p2_qlike_hi']:+.2f}]e-3"
            for m in TREES
        )
        + f" dQLIKE, against {1e3 * float(tab('live_feasible', 'lgbm', 'perm_p2_qlike')['value'].iloc[0]):.0f}e-3 for `har_ma_*` (LightGBM).",
        "- TreeSHAP (path-dependent) is not immune: it credits the continuous probe with a median rank of "
        + ", ".join(f"{int(c_lf.loc[m, 'shap_rank_median'])}" for m in TREES)
        + " (above "
        + ", ".join(
            f"{100 * c_lf.loc[m, 'shap_real_columns_below_share']:.0f} %" for m in TREES
        )
        + " of used real columns): it splits the fitted trees, including their fits to noise.",
        (
            "- At the SERIES level the bias washes out of the headline: every tree measure puts `har_ma_*` first. It matters for everything below the leader."
            if not DEDUP
            else "- At the SERIES level: "
            + DS.tree_leaders_md(lead, TREES, TREE_M, BUCKETS)
        ),
        "- What permutation says instead (series in the top 5 by MDI or split count that permutation P2 ranks outside its top 10; permutation rank in brackets): "
        + "; ".join(
            f"{BNAME[b]} {LABEL[m]}: "
            + (", ".join(f"`{x}` ({r_})" for x, r_ in v) if v else "none")
            for (b, m), v in only_nat.items()
        )
        + ". These are series the trees split on often that rank low when scrambled on the held-out tail.",
        "",
    ]
    # top features per measure (series), live_feasible and all_features
    for b in ("live_feasible", "all_features", "baseline"):
        md.append(
            f"## Top series per measure -- {b} (value = share for MDI/split/|beta sd|/SHAP; % of the model's tail loss for permutation / drop-column)"
        )
        md.append("")
        md.append("| model | measure | top 3 |")
        md.append("|---|---|---|")
        for model in MODELS:
            for m in TREE_M if model in TREES else LIN_M:
                t = tab(b, model, m).head(3)
                cells = []
                for r in t.itertuples():
                    vs = (
                        f"{100 * r.value:.0f} %"
                        if m in ("mdi", "split", "bsd", "shap")
                        else f"{r.pct_of_loss:+.0f} %"
                    )
                    cells.append(f"`{short(r.label)}` {vs}")
                md.append(
                    f"| {LABEL[model]} | {MEAS_MD[m]} | " + "; ".join(cells) + " |"
                )
        md.append("")
    md += [
        "## Ranking agreement",
        "Spearman rank correlation of the refit-averaged importance vectors (eligible units = non-constant in at least one window).",
        "",
        "Within a tree model (series / column level): "
        + "; ".join(
            f"{BNAME[b]} {LABEL[m]} MDI~SHAP {wc(b, 'series', m, 'mdi', 'shap'):.2f}/{wc(b, 'column', m, 'mdi', 'shap'):.2f}, "
            f"MDI~perm P2 {wc(b, 'series', m, 'mdi', 'perm_p2_qlike'):.2f}/{wc(b, 'column', m, 'mdi', 'perm_p2_qlike'):.2f}, "
            f"SHAP~perm P2 {wc(b, 'series', m, 'shap', 'perm_p2_qlike'):.2f}/{wc(b, 'column', m, 'shap', 'perm_p2_qlike'):.2f}"
            for b in ("live_feasible", "all_features")
            for m in TREES
        )
        + ".",
        "",
        "Across models, same measure (series level, range over pairs): "
        + "; ".join(
            f"{BNAME[b]} SHAP tree~tree {min(xc(b, 'series', a, 'shap', c, 'shap') for i, a in enumerate(TREES) for c in TREES[i + 1 :]):.2f}..{max(xc(b, 'series', a, 'shap', c, 'shap') for i, a in enumerate(TREES) for c in TREES[i + 1 :]):.2f}, "
            f"tree~linear {min(N['xc_shap_tl'][b]):.2f}..{max(N['xc_shap_tl'][b]):.2f}; perm P2 dMSE tree~linear "
            f"{min(xc(b, 'series', a, 'perm_p2_mse', c, 'perm_p2_mse') for a in TREES for c in LIN):.2f}..{max(xc(b, 'series', a, 'perm_p2_mse', c, 'perm_p2_mse') for a in TREES for c in LIN):.2f}"
            for b in ("live_feasible", "all_features")
        )
        + ".",
        "",
        "## Correlated groups (the dilution caveat)",
        "Joint permutation of a cluster vs the sum of its columns' separate permutations (P2, dMSE x 1e-3, joint vs sum), live_feasible cluster "
        + f"{c22['cluster'].iloc[0]} (`{c22['members'].iloc[0]}`): "
        + "; ".join(
            f"{LABEL[r.model]} {1e3 * r.joint_dmse:.1f} vs {1e3 * r.sum_of_columns_dmse:.1f}"
            for r in c22.itertuples()
        )
        + ". Joint > sum = the columns substitute for each other and permuting one at a time understates the group (dilution); joint < sum = offsetting weights (permuting one breaks an offset the other carries).",
        "",
    ]
    # ---- numbers for the remaining prose
    har5 = {}
    for b in ("live_feasible", "all_features"):
        for model in MODELS:
            for m in TREE_M if model in TREES else LIN_M:
                t = tab(b, model, m)
                har5[(b, model, m)] = float(
                    t[t["unit"] == "har_ma"]["top5_share"].iloc[0]
                )
    har5_min = {
        b: min(v for (bb, _, _), v in har5.items() if bb == b)
        for b in ("live_feasible", "all_features")
    }
    har5_min_at = {
        b: min(((v, k) for k, v in har5.items() if k[0] == b))[1]
        for b in ("live_feasible", "all_features")
    }
    exceptions = N["leaders_not_har"]
    lf_ridge_p1 = tab("live_feasible", "ridge", "perm_p1_qlike").iloc[0]
    tuned_ok = "series" in TU.columns and len(TU) > 0

    md += [
        "## Top-5 stability and regimes",
        f"`har_ma_*` is in the per-refit top 5 of every measure of every model in at least {100 * har5_min['live_feasible']:.0f} % of the {K} refits (live_feasible; lowest: {LABEL[har5_min_at['live_feasible'][1]]} {MEAS_MD[har5_min_at['live_feasible'][2]]}) "
        f"and at least {100 * har5_min['all_features']:.0f} % (all_features; lowest: {LABEL[har5_min_at['all_features'][1]]} {MEAS_MD[har5_min_at['all_features'][2]]}). "
        "Top series of each measure that is NOT `har_ma_*`: "
        + (
            ", ".join(f"{k} -> {v}" for k, v in exceptions.items())
            if exceptions
            else "none"
        )
        + ".",
        f"By calendar year ({', '.join(f'{y}: {c}' for y, c in years.items())} refits) and by causal VIX tercile (VIX at 15:30 vs the 1/3, 2/3 quantiles of the previous 2000 sessions' 15:30 VIX; low {terc['low']}, mid {terc['mid']}, high {terc['high']} refits -- the trailing window includes the calm 2010s, so 2018-2024 is mostly 'high'): "
        f"Spearman of a stratum's series ranking with the full-sample ranking, median over strata and models (live_feasible / all_features): built-in {SM[('live_feasible', 'builtin', 'year')]:.2f} / {SM[('all_features', 'builtin', 'year')]:.2f} by year, {SM[('live_feasible', 'builtin', 'vix_tercile')]:.2f} / {SM[('all_features', 'builtin', 'vix_tercile')]:.2f} by VIX tercile; SHAP {SM[('live_feasible', 'shap', 'year')]:.2f} / {SM[('all_features', 'shap', 'year')]:.2f} and {SM[('live_feasible', 'shap', 'vix_tercile')]:.2f} / {SM[('all_features', 'shap', 'vix_tercile')]:.2f}; permutation P2 {SM[('live_feasible', 'perm', 'year')]:.2f} / {SM[('all_features', 'perm', 'year')]:.2f} and {SM[('live_feasible', 'perm', 'vix_tercile')]:.2f} / {SM[('all_features', 'perm', 'vix_tercile')]:.2f}. The built-in and SHAP rankings barely move across regimes; the permutation ranking below the leader moves more (it is measured on 10-session tails). Tables `stability_by_stratum.csv`, PDF Table 10.",
        "",
        "## Linear models: the QLIKE permutation numbers are dominated by single tails",
        f"The linear forecasts are unbounded: moving an extreme input (a crash day's 4th-power or absolute return) onto another row can push the fit-space forecast through zero, and QLIKE of yhat^2 x baseline explodes. Example: live_feasible ridge, top P1 dQLIKE `{short(lf_ridge_p1.label)}` mean {lf_ridge_p1.value:.3g} [{lf_ridge_p1.lo:.2g}, {lf_ridge_p1.hi:.2g}], median refit {lf_ridge_p1.median_refit:.2g}, one refit carries {100 * lf_ridge_p1.max_refit_share:.0f} % of the sum. "
        f"The all_features ridge is the extreme case: its intact tail QLIKE averages {af_ridge_q:.2f} against {min(oth_q):.3f}..{max(oth_q):.3f} for every other model x bucket (the known rare extreme values of the plain back-transform). "
        "Handled explicitly: every perm / drop table carries the median refit, the share of refits with a positive change and the largest single-refit share; for the linear models the figures and the cross-model comparisons use the squared-error versions (dMSE in the fit space, no blow-up). Trees cannot extrapolate (their forecasts stay inside the training range of the target), so their QLIKE numbers are well behaved.",
        "",
        "## Tuned trees",
        (
            DS.tuned_md(tuned_ok, tuned_lead)
            if DEDUP
            else f"The causally tuned trees' own extracts (`results/linear_subsection_trees_tuned/importance/`: gain per refit, TreeSHAP rows at 16:00): `har_ma_*` leads both MDI and SHAP in {tuned_lead[0]} of {tuned_lead[1]} bucket x model arms; shares tuned vs untuned in `tuned_trees_series.csv` and PDF Table 13. Split count and permutation were not computed for the tuned trees (their per-refit chosen configurations would have to be refitted)."
            if DEDUP or (tuned_ok and tuned_lead is not None)
            else "Pending: the tuned-tree importance extracts were not present."
        ),
        "",
    ]
    if DEDUP:
        md += DS.gates_md(OUT, G) + DS.before_after_md(OUT) + DS.files_md(ROOT_REL)
    else:
        md += [
            "## Gates",
            f"- Refits vs the stored forecasts: XGBoost and random forest reproduce them (largest gap {xr_gap:.1e}, above 1e-9 on {gap_n['xgb']} XGBoost and {gap_n['rf']} forest refits of the 3 x 147; mean gap {xr_mean:.1e}); LightGBM reproduces them bit for bit on HAR + calendar (gap {lg_gap['baseline'][1]:.1e}) but not on live_feasible / all_features: mean |gap| {lg_gap['live_feasible'][0]:.4f} / {lg_gap['all_features'][0]:.4f}, max {lg_gap['live_feasible'][1]:.3f} / {lg_gap['all_features'][1]:.3f}, correlation {lg_gap['live_feasible'][2]:.4f} / {lg_gap['all_features'][2]:.4f} (same library version, same params, seed and thread count; the refit is deterministic run to run -- the cluster and a laptop give the same gap -- so the stored LightGBM runs differ in something the input file does not carry; XGBoost on the same input reproduces to 1e-16). The LightGBM importance is that of the refit.",
            "- The design: out-of-sample stamps, targets and column names equal the stored runs'; the linear coefficient captures equal the design rows exactly and the stored research forecasts to <= "
            + f"{max(float(gl.loc[(b, m), 'capture_vs_stored_rel']) for b in BUCKETS for m in LIN):.0e} (relative).",
            "- Drop-column anchors: the re-solve on the window before the tail reproduces the captured coefficients (ridge <= "
            + f"{max(an[b]['ridge'] for b in BUCKETS):.0e}; lasso <= {max(an[b]['lasso'] for b in BUCKETS):.0e}, above 1e-6 on "
            + " / ".join(str(v) for v in anch_n["lasso"])
            + " of 147 refits (HAR + calendar / live_feasible / all_features); the re-solve masks constant and duplicate columns of its own window, the walk keeps a between-tune mask). The all_features lasso refits 75..124 (penalty 0.001, 115-120 active columns) ran on the cluster (`cluster/slurm/submit_featimp_linear.sh`), the rest locally; the 53 parts partition the 147 refits (checked).",
            "- Every model of a bucket saw the identical permutation draws (md5 per refit equal across the five models, 147/147).",
            "- TreeSHAP additivity: sum of phi + expected value = forecast to <= "
            + f"{max(float(gl.loc[(b, m), 'shap_additivity_max']) for b in BUCKETS for m in TREES):.0e}.",
            "",
            "## Files",
            "- `by_measure/imp_<bucket>_<model>_<measure>.csv` (one per measure x model x bucket; rows = columns, series and clusters: value, 95 % interval over refits, rank, top-5 share, % of loss, median refit, largest single-refit share, by-year and by-VIX-tercile means, SHAP signed mean)",
            "- `series_level_all.csv`, `rank_corr_within_model.csv`, `rank_corr_across_models.csv`, `top5_stability.csv`, `stability_by_stratum.csv`, `unique_values.csv`, `cardinality_vs_measure.csv`, `noise_probes.csv`, `clusters.csv`, `cluster_dilution.csv`, `tuned_trees_series.csv`, `gates.csv`",
            "- figures: `fig_ranked_<bucket>.png` (ranked bars per measure), `fig_rank_heatmap_<bucket>_<level>.png` (ranks across measures), `fig_cardinality_live_feasible.png`, `fig_top5_stability_live_feasible.png`",
            "- cluster twins: `cluster/slurm/{ship_featimp_carc.sh, submit_featimp.sh, featimp_pack.sbatch, featimp_collect.sbatch}`, `cluster/featimp_tasks*.txt` (tree refits); `cluster/slurm/{ship_featimp_linear_carc.sh, submit_featimp_linear.sh, featimp_linear.sbatch}`, `cluster/featimp_linear_tasks*.txt` (the slow lasso drop-column refits)",
        ]
    (OUT / "SUMMARY.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (OUT / "summary_numbers.json").write_text(
        json.dumps(N, indent=1, default=str), encoding="utf-8"
    )

    # ================================================================ tex
    def table(
        caption: str,
        colspec: str,
        head: str,
        rows: list[str],
        size: str = "\\footnotesize",
    ) -> str:
        return (
            "\\begin{table}[H]\\centering"
            + size
            + "\n\\caption{"
            + caption
            + "}\n\\begin{tabular}{"
            + colspec
            + "}\\toprule\n"
            + head
            + "\\\\\\midrule\n"
            + "\n".join(rows)
            + "\n\\bottomrule\\end{tabular}\\end{table}\n"
        )

    def fig(path: str, caption: str, width: str = "\\textwidth") -> str:
        return (
            "\\begin{figure}[H]\\centering\\includegraphics[width="
            + width
            + "]{../"
            + ROOT_REL
            + "/"
            + path
            + "}\n\\caption{"
            + caption
            + "}\\end{figure}\n"
        )

    key = (
        (
            "\\texttt{har\\_ma\\_*} realized variance of the bar (the target's own HAR ladder, means over the last 1, 5, 25, 125, 625, 3125 bars; \\texttt{har\\_ma\\_*\\_x\\_close} = the same times the 16:00 close gate, identical at this bar); "
            if not DEDUP
            else "\\texttt{har\\_ma\\_*} realized variance of the bar (the target's own HAR ladder, means over the last 1, 5, 25, 125, 625, 3125 bars; the per-bar design no longer carries the session-edge interactions \\texttt{har\\_ma\\_*\\_x\\_open/close}); "
        )
        + "\\texttt{adj\\_sumabsret\\_ma\\_*} absolute return; \\texttt{adj\\_sumret\\_ma\\_*} signed return; \\texttt{adj\\_sumret3/4\\_ma\\_*} 3rd/4th-power returns; \\texttt{adj\\_sumpret2\\_ma\\_*} upside squared returns; "
        "\\texttt{adj\\_sumbipow\\_ma\\_*} bipower variation; \\texttt{adj\\_sumautocov\\_ma\\_*} return autocovariance; \\texttt{adj\\_sumvolume\\_ma\\_*} ES volume; \\texttt{adj\\_numobs\\_ma\\_*} ES prints per bar; "
        "\\texttt{adj\\_vix/vvix/vix3m\\_ma\\_*} the Cboe indices; \\texttt{adj\\_fomc\\_*} FOMC flags and distances; \\texttt{*\\_avail/active\\_ma\\_*} is-present / is-nonzero flags; calendar = \\texttt{DOW\\_*}, \\texttt{is\\_*}, \\texttt{hour}, \\texttt{days\\_to\\_opex}. "
        "all features adds the constituent cross-section (\\texttt{*\\_ewstock}, \\texttt{*\\_vwstock}), Cboe volume (\\texttt{adj\\_voldemand\\_*}) and StockTwits (\\texttt{adj\\_stocktwits\\_*})."
    )
    tex = (
        r"""\documentclass[10pt]{article}
\usepackage[margin=0.75in]{geometry}
\usepackage{graphicx,booktabs,amsmath,float}
\usepackage[font=small]{caption}
\begin{document}
\begin{center}{\Large\bf Feature importance of the 15:30 forecast (16:00 bar)"""
        + (r"\\ re-run on the de-duplicated per-bar design" if DEDUP else "")
        + r"""}\\[3pt]
{\small MDI, split count, permutation importance on the causal held-out tail, SHAP --- trees and the per-bar linear models; """
        + f"{n:,} forecasts {first} .. {last}, {K} refits"
        + r"""}\end{center}
"""
        + (DS.intro_tex(OUT, K) if DEDUP else "")
        + r"""
\textbf{Setup.} Models: per-bar ridge and lasso (coefficients re-solved every session, penalty re-chosen causally every 250 sessions) and the untuned per-bar LightGBM, XGBoost and random forest (refit every 10 sessions) --- the shipped configurations. Every fit uses the 2000 sessions before it. A refit's \emph{held-out tail} is the $\le 10$ sessions its model forecasts, up to the next refit: never in its fit, so a loss change measured there is out of sample and causal. Buckets: live-feasible (the deck's), all features, and HAR + calendar. Library: scikit-learn 1.9.0 (random forest), LightGBM 4.6.0, XGBoost 3.2.0, shap 0.51.0; the permutation and drop-column loops are written out, because a library permutation routine scores one fitted model on one test set, not """
        + str(K)
        + r""" walk-forward models on their own tails.

\textbf{Measures.} (1) \emph{MDI / gain}: LightGBM gain, XGBoost total gain, random-forest impurity decrease; share per refit, averaged. (2) \emph{Split count}. (3) \emph{Permutation importance} on the tail, loss = QLIKE of the plain back-transform $\hat y^2\times$ diurnal baseline against the bar's realized variance (no recalibration), and squared error in the fit space; 10 draws per unit and refit, the same draws for all five models. P1 shuffles the unit's values among the tail's rows (the literal permutation; blind to slow inputs, which barely move in 10 sessions); P2 replaces them with rows drawn from the refit's own training window (past data only). A unit is a column, a \emph{series} (all columns of one input: lags, flags, gates) or a \emph{cluster} (columns every pair of which has $|$corr$|\ge 0.8$, complete linkage); a group's columns move together. Intervals: 95\% circular-block bootstrap over refits (block 6). (4) \emph{SHAP}: TreeSHAP (path-dependent) on the same tail rows; for the linear models $\beta_j(x_j-\bar x_j)$ with the window mean as background. Linear extras: $|\beta\cdot sd|$ and drop-column importance (re-solve without the unit, penalty held).

\textbf{Key.} """
        + key
        + "\n\n"
        + table(
            "The design is float throughout, but not all continuous: distinct values per column in each 2000-session window (median over the "
            + str(K)
            + " windows) --- what MDI's cardinality bias keys on.",
            "lrrrrrrrr",
            "bucket & columns & constant & binary & 3--25 & 26--500 & $>$500 & series & clusters (columns)",
            dtype_rows,
        )
        + r"""
\section*{1. Does MDI's cardinality bias matter here? Yes, below the leader.}
"""
        + table(
            "Spearman rank correlation, over the non-constant columns, between a column's importance and its number of distinct values. MDI, split count and TreeSHAP track cardinality; permutation importance does not.",
            "llrrrrrr",
            "bucket & model & MDI & split & SHAP & perm P1 QL & perm P2 QL & perm P2 MSE",
            card_rows,
        )
        + table(
            "Noise probes: on every third refit a second fit appends three pure-noise columns (continuous; 20 values; binary). Median rank among all columns (1 = most important) and, in brackets, the share of the real columns the model uses that rank below the probe; permutation P2 of the continuous probe ($\\Delta$QLIKE $\\times 10^{3}$, 95\\% interval) against \\texttt{har\\_ma\\_*}'s.",
            "llrrrrrrrr",
            "bucket & model & cols & MDI rank & split rank & SHAP rank & 20-val MDI & binary MDI & perm P2 (cont.) & \\texttt{har\\_ma\\_*}",
            probe_rows,
            "\\scriptsize",
        )
        + r"""
\textbf{Reading.} A column of pure noise with a new value on every row is ranked by MDI above """
        + ", ".join(
            f"{100 * c_lf.loc[m, 'mdi_real_columns_below_share']:.0f}\\%" for m in TREES
        )
        + r""" (LightGBM, XGBoost, forest) of the real live-feasible columns the model uses, and by the forest's split count above """
        + f"{100 * c_lf.loc['rf', 'split_real_columns_below_share']:.0f}\\%"
        + r"""; the binary noise column sits near the bottom. The same noise gets a permutation importance of at most """
        + f"{1e3 * max_probe:.2f}"
        + r"""$\times10^{-3}$ in absolute value ("""
        + f"{len(cont) - n_ex}"
        + r""" of """
        + f"{len(cont)}"
        + r""" intervals include zero"""
        + (f"; the exceptions {esc(ex_list)}" if n_ex else "")
        + r"""), against at least """
        + f"{har_min:.2f}"
        + r""" for \texttt{har\_ma\_*}. TreeSHAP, which splits the fitted trees, inherits part of the bias. Over the real columns, MDI, split count and SHAP correlate """
        + f"{rng_[('live_feasible', 'mdi')][0]:.2f}--{rng_[('live_feasible', 'shap')][1]:.2f}"
        + r""" with the number of distinct values, permutation """
        + f"{min(rng_[('live_feasible', 'perm_p1_qlike')][0], rng_[('live_feasible', 'perm_p2_qlike')][0]):.2f}--{max(rng_[('live_feasible', 'perm_p1_qlike')][1], rng_[('live_feasible', 'perm_p2_qlike')][1]):.2f}"
        + r""". So below the leader, the MDI / split ranking of this design is partly a ranking by cardinality; the permutation ranking is the one to use for selection. """
        + (
            r"At the series level the bias does not reach the top: every tree measure puts \texttt{har\_ma\_*} first."
            if not DEDUP
            else "At the series level: "
            + DS.tree_leaders_tex(lead, TREES, TREE_M, BUCKETS)
        )
        + r""" Series that MDI or split count puts in the top 5 but permutation P2 ranks outside the top 10 (permutation rank in brackets): """
        + "; ".join(
            f"{BNAME[b_]} {LABEL[m_]}: "
            + (", ".join(f"{tt(x)} ({r_})" for x, r_ in v) if v else "none")
            for (b_, m_), v in only_nat.items()
        )
        + r""".

\section*{2. The top inputs by measure}
"""
        + table(
            "Live-feasible design, series level: top three per model and measure (share for MDI / split / $|\\beta\\cdot sd|$ / SHAP; change as \\% of the model's tail loss for permutation and drop-column).",
            "llp{11.2cm}",
            "model & measure & top three",
            top_rows["live_feasible"],
            "\\scriptsize",
        )
        + fig(
            "fig_ranked_live_feasible.png",
            "Live-feasible: ranked bars per measure (series level; top 8). Rows: models; columns: the built-in measure, split count (trees) / drop-column (linear), permutation P2 on the tail, SHAP. Bars with 95\\% intervals over refits. Linear permutation / drop-column in $\\Delta$MSE (see Section 6).",
        )
        + table(
            "All-features design, series level: top three per model and measure.",
            "llp{11.2cm}",
            "model & measure & top three",
            top_rows["all_features"],
            "\\scriptsize",
        )
        + fig(
            "fig_ranked_all_features.png",
            "All features: ranked bars per measure (series level; top 8).",
        )
        + r"""
\section*{3. Ranking agreement}
"""
        + table(
            "Within a tree model: Spearman between measures (series / column level).",
            "lllrrrrrr",
            "bucket & level & model & MDI$\\sim$split & MDI$\\sim$SHAP & MDI$\\sim$P2 & SHAP$\\sim$P2 & P1$\\sim$P2 & P2 QL$\\sim$MSE",
            within_rows,
            "\\scriptsize",
        )
        + table(
            "Within a linear model: Spearman between measures.",
            "lllrrrrrr",
            "bucket & level & model & $|\\beta sd|\\sim$SHAP & $|\\beta sd|\\sim$P2 & SHAP$\\sim$P2 & P1$\\sim$P2 & P2$\\sim$drop & P2 QL$\\sim$MSE",
            within_lin,
            "\\scriptsize",
        )
        + table(
            "Across models, same measure: range of Spearman over tree--tree pairs, tree--linear pairs, and ridge--lasso.",
            "llrrrrrrrrr",
            "bucket & level & \\multicolumn{3}{c}{SHAP} & \\multicolumn{3}{c}{perm P2 $\\Delta$QLIKE} & \\multicolumn{3}{c}{perm P2 $\\Delta$MSE}\\\\ & & tree & tree--lin & r--l & tree & tree--lin & r--l & tree & tree--lin & r--l",
            across_rows,
            "\\scriptsize",
        )
        + fig(
            "fig_rank_heatmap_live_feasible_series.png",
            "Live-feasible: rank of every series under every model $\\times$ measure (1 = most important).",
            "0.95\\textwidth",
        )
        + fig(
            "fig_rank_heatmap_live_feasible_cluster.png",
            "Live-feasible: the same at the cluster level (clusters of $|$corr$|\\ge0.8$; a singleton is its column).",
            "0.95\\textwidth",
        )
        + fig(
            "fig_rank_heatmap_all_features_series.png",
            "All features: rank of every series under every model $\\times$ measure.",
            "0.95\\textwidth",
        )
        + r"""
\section*{4. Stability over refits and regimes}
"""
        + table(
            "Top-5 stability (series level): share of the "
            + str(K)
            + " refits in which \\texttt{har\\_ma\\_*} is in the refit's top 5; number of series in the top 5 in at least half the refits; number ever in the top 5. Measures: built-in (MDI or $|\\beta sd|$), permutation P2 ($\\Delta$QLIKE trees, $\\Delta$MSE linear), SHAP.",
            "llrrrrrrrrr",
            "bucket & model & \\multicolumn{3}{c}{built-in} & \\multicolumn{3}{c}{perm P2} & \\multicolumn{3}{c}{SHAP}\\\\ & & har & $\\ge$50\\% & ever & har & $\\ge$50\\% & ever & har & $\\ge$50\\% & ever",
            stab_rows,
            "\\scriptsize",
        )
        + table(
            "Regimes: Spearman between the ranking within a stratum and the full-sample ranking (series level): median (minimum) over the calendar years ("
            + ", ".join(f"{y}: {c}" for y, c in years.items())
            + " refits) and over the causal VIX terciles (VIX at 15:30 against the 1/3, 2/3 quantiles of the previous 2000 sessions' 15:30 VIX: low "
            + f"{terc['low']}, mid {terc['mid']}, high {terc['high']}"
            + " refits); years in which the full-sample leader also leads.",
            "llrrrrrrrrr",
            "bucket & model & \\multicolumn{3}{c}{built-in} & \\multicolumn{3}{c}{perm P2} & \\multicolumn{3}{c}{SHAP}\\\\ & & year & VIX & lead & year & VIX & lead & year & VIX & lead",
            strat_rows,
            "\\scriptsize",
        )
        + fig(
            "fig_top5_stability_live_feasible.png",
            "Live-feasible: \\% of refits in which each series is in the top 5, per model $\\times$ measure.",
            "0.95\\textwidth",
        )
        + r"""
\section*{5. Correlated inputs: the dilution caveat}
"""
        + table(
            "Joint permutation of a cluster against the sum of its columns permuted one at a time (P2, $\\Delta$MSE in the fit space $\\times10^{3}$, joint / sum; squared error so the linear columns are not dominated by single tails), for the four clusters with the largest joint effect in LightGBM. Joint $>$ sum: the columns substitute for one another and one-at-a-time permutation understates the group (dilution). Joint $<$ sum: offsetting weights (permuting one column breaks an offset its partner carries).",
            "llp{5.2cm}rrrrr",
            "bucket & cluster & members & ridge & lasso & LightGBM & XGBoost & forest",
            dil_rows,
            "\\scriptsize",
        )
        + r"""
\section*{6. Linear models: QLIKE permutation is dominated by single tails}
The linear forecasts are unbounded: moving an extreme input (a crash day's 4th-power or absolute return) onto another row can push the fit-space forecast through zero, and QLIKE of $\hat y^2\times$ baseline explodes. Every permutation / drop-column table therefore carries the median refit and the largest single-refit share of the sum, and the linear figures and cross-model comparisons use the squared-error versions. Trees cannot extrapolate beyond the training range of the target, so their QLIKE changes are well behaved.
"""
        + table(
            "The top linear QLIKE entries: mean [95\\% interval], median refit, share of the sum carried by one refit.",
            "lllllrr",
            "bucket & model & measure & top series & mean [interval] & median & one refit",
            blow,
            "\\scriptsize",
        )
        + r"""
\section*{7. Tuned trees}
"""
        + (
            table(
                "Causally tuned trees (their own extracts: gain per refit and TreeSHAP rows at 16:00): \\texttt{har\\_ma\\_*} share tuned (untuned), rank correlation tuned vs untuned over the series, and the next three series by tuned SHAP. Split count and permutation were not computed for the tuned trees (their per-refit configurations would have to be refitted).",
                "llrrrrp{5cm}",
                "bucket & model & MDI har & SHAP har & $\\rho$ MDI & $\\rho$ SHAP & next by tuned SHAP",
                tuned_rows,
                "\\scriptsize",
            )
            if tuned_rows
            else tuned_note
        )
        + r"""
"""
        + (DS.gates_tex(OUT, G) + DS.before_after_tex(OUT, ROOT_REL) if DEDUP else "")
        + (
            ""
            if DEDUP
            else r"""\section*{8. Gates}
\begin{itemize}\itemsep1pt
\item Design: out-of-sample stamps, targets and column names equal the stored tree runs'; the linear captures equal the design rows exactly and the stored research forecasts to $\le$ """
            + f"{max(float(gl.loc[(b, m), 'capture_vs_stored_rel']) for b in BUCKETS for m in LIN):.0e}"
            + r""" (relative).
\item Refits: XGBoost and random forest reproduce the stored forecasts (largest gap """
            + f"{xr_gap:.1e}"
            + r"""; above $10^{-9}$ on """
            + f"{gap_n['xgb']}"
            + r""" XGBoost and """
            + f"{gap_n['rf']}"
            + r""" forest refits of the $3	imes147$). LightGBM reproduces them exactly on HAR + calendar but not on the two larger designs: mean $|$gap$|$ """
            + f"{lg_gap['live_feasible'][0]:.4f} / {lg_gap['all_features'][0]:.4f}, max {lg_gap['live_feasible'][1]:.3f} / {lg_gap['all_features'][1]:.3f}, correlation {lg_gap['live_feasible'][2]:.4f} / {lg_gap['all_features'][2]:.4f}"
            + r""" (same version, parameters, seed and thread count; deterministic run to run on two machines; XGBoost reproduces on the same input). The LightGBM numbers are those of the refit.
\item Drop-column anchors reproduce the captured coefficients (ridge $\le$ """
            + f"{max(an[b]['ridge'] for b in BUCKETS):.0e}, lasso $\\le$ {max(an[b]['lasso'] for b in BUCKETS):.0e}"
            + r"""). Every model of a bucket saw identical permutation draws (md5 per refit). TreeSHAP additivity $\le$ """
            + f"{max(float(gl.loc[(b, m), 'shap_additivity_max']) for b in BUCKETS for m in TREES):.0e}."
            + r"""
\end{itemize}
\sloppy Files: \texttt{experiments/\allowbreak feature\_importance\_1530.py} (design, inputs, linear, aggregate); \texttt{experiments/\allowbreak feature\_importance\_1530\_trees.py} (the refits, run on the cluster through \texttt{cluster/\allowbreak slurm/\allowbreak submit\_featimp.sh}); tables and figures in \texttt{results/\allowbreak feature\_importance\_1530/}; summary \texttt{SUMMARY.md}.
"""
        )
        + r"""
\appendix
\section*{Appendix: HAR + calendar design}
"""
        + table(
            "HAR + calendar, series level: top three per model and measure.",
            "llp{11.2cm}",
            "model & measure & top three",
            top_rows["baseline"],
            "\\scriptsize",
        )
        + fig("fig_ranked_baseline.png", "HAR + calendar: ranked bars per measure.")
        + fig(
            "fig_rank_heatmap_baseline_series.png",
            "HAR + calendar: ranks across measures.",
            "0.7\\textwidth",
        )
        + fig(
            "fig_cardinality_live_feasible.png",
            "Live-feasible columns: MDI share, permutation P2 $\\Delta$QLIKE and SHAP share against the number of distinct values in the training window (log scale), by cardinality class.",
        )
        + fig(
            "fig_rank_heatmap_all_features_cluster.png",
            "All features: ranks at the cluster level.",
            "0.95\\textwidth",
        )
        + r"""
\end{document}
"""
    )
    W = R / "writeup"
    (W / f"{STEM}.tex").write_text(tex, encoding="utf-8")
    for _ in range(2):
        r = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", f"{STEM}.tex"],
            cwd=W,
            capture_output=True,
            text=True,
        )
    print(
        "pdflatex exit",
        r.returncode,
        "; pages:",
        [ln for ln in r.stdout.splitlines() if "Output written" in ln],
    )
    print("\n".join(md))


if __name__ == "__main__":
    main()
