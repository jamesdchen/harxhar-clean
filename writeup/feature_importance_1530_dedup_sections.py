"""The de-dup re-run's own parts of the feature-importance report (checklist I7), used by
writeup/make_feature_importance_1530_tex.py --dedup: the design / mask / cadence note, the gates
and the before / after section, markdown and LaTeX.  Every number is read from the re-run's CSVs
(experiments/feature_importance_1530_dedup.py stages gate_design, cadence, canonical,
before_after, and the aggregates of both roots); nothing is typed in."""

from __future__ import annotations

import re
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parent.parent
FIRST = R / "results" / "feature_importance_1530"
NOMASK = R / "results" / "feature_importance_1530_dedup_nomask"
BUCKETS = ("all_features",)  # baseline commented out; live_feasible commented out
BNAME = {
    "baseline": "HAR + calendar",
    "live_feasible": "live-feasible",
    "all_features": "all features",
}
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
# the measures of the before / after tables: built-in, SHAP, permutation P2 (and drop-column)
HEAD_T = ("mdi", "split", "shap", "perm_p2_qlike", "perm_p2_mse")
HEAD_L = ("bsd", "shap", "perm_p2_mse", "drop_mse")
MNAME = {
    "mdi": "MDI",
    "split": "split",
    "bsd": "abs(b sd)",
    "shap": "SHAP",
    "perm_p1_qlike": "perm P1 QL",
    "perm_p2_qlike": "perm P2 QL",
    "perm_p1_mse": "perm P1 MSE",
    "perm_p2_mse": "perm P2 MSE",
    "drop_qlike": "drop QL",
    "drop_mse": "drop MSE",
}
SHARE = ("mdi", "split", "bsd", "shap")


def _read(p: Path) -> pd.DataFrame | None:
    return pd.read_csv(p) if p.is_file() else None


# ============================================================================ tree settings (2026-10-08)
# The trees' settings are read from the refit files on disk (every part's meta records its params).  A model
# re-run on new settings after the stored campaign runs (XGBoost 2026-10-07: xgb_opt_00; the forest 2026-10-08:
# trial 255) is gated by its own file gates_<token>.csv: the stored-run gates (no-mask reproduction, the mask's
# effect, the canonical tables, cadence) compare its old settings and leave it out.
RERUN_GATES = {"xgb": "gates_xgb10.csv", "rf": "gates_rf255.csv"}
CHANGE = {
    "xgb": "XGBoost now runs xgb_opt_00 (was xgb_opt_06; 2026-10-07)",
    "rf": "the random forest now runs trial 255 of its 400-trial dev-span search (was the library defaults; 2026-10-08)",
}


def _trees_module():  # type: ignore[no-untyped-def]
    import sys

    sys.path.insert(0, str(R / "experiments"))
    import feature_importance_1530_trees as T  # type: ignore

    return T


def tree_settings(out: Path) -> dict:
    """model -> {params, name, note, interim, cpus} from the all_features refit parts of the root."""
    import json

    T = _trees_module()
    res: dict = {}
    for m in TREES:
        fs = sorted(
            (out / "_work" / "trees").glob(
                f"trees_all_features_{T.PART_TAG[m]}_*_*.npz"
            )
        )
        fs = [
            f
            for f in fs
            if int(f.stem.split("_")[-1]) - int(f.stem.split("_")[-2]) == 21
        ]
        if not fs:
            continue
        zs = [np.load(f, allow_pickle=True) for f in fs]
        metas = [json.loads(str(z["meta"])) for z in zs]
        prm = metas[0]["params"]
        assert all(x["params"] == prm for x in metas), (
            m,
            "parts disagree on the settings",
        )
        name, note, interim = "", "", False
        if m == "lgbm" and prm == T.LGBM_PARAMS:
            name, note = "lgbm_opt_00", "the top entry of the 1998-2002 dev-span search"
        elif m == "xgb" and prm == T.XGB_PARAMS:
            name = "xgb_opt_00"
            note = "the top entry of the 1998-2002 dev-span search (since 2026-10-07; was the 7th-ranked xgb_opt_06)"
        elif m == "rf" and prm == T.RF255_PARAMS:
            name = "trial 255"
            note = "the top entry of the forest's 400-trial 1998-2002 dev-span search (since 2026-10-08; was the library defaults)"
        elif m == "rf" and prm == T.RF_DEFAULT_PARAMS:
            name, interim = "library defaults", True
            note = (
                "INTERIM: this build still shows the default forest; its refits on trial 255 of the forest's 400-trial "
                "1998-2002 dev-span search are running and replace it in the final build"
            )
        cpus = sorted(
            {str(c).split("|", 1)[-1] for z in zs if "cpu" in z.files for c in z["cpu"]}
        )
        res[m] = dict(params=prm, name=name, note=note, interim=interim, cpus=cpus)
    return res


def _fmt_params(m: str, prm: dict) -> str:
    if m == "rf":
        if prm == {"n_estimators": 100}:
            return "100 trees, unlimited depth, min leaf 1, every feature at every split, bootstrap of all rows"
        return (
            f"{prm['n_estimators']} trees, depth {prm['max_depth']}, min leaf {prm['min_samples_leaf']}, "
            f"{100 * prm['max_features']:.0f}% of the features at a split, bootstrap of {100 * prm['max_samples']:.0f}% of the rows"
        )
    if m == "xgb":
        return (
            f"{prm['n_estimators']} rounds, depth {prm['max_depth']}, learning rate {prm['learning_rate']:.4f}, "
            f"min child weight {prm['min_child_weight']:.2f}"
        )
    return (
        f"{prm['n_estimators']} rounds, {prm['num_leaves']} leaves, learning rate {prm['learning_rate']:.4f}, "
        f"min child samples {prm['min_child_samples']}"
    )


def _tx(s: str) -> str:
    """Markdown -> LaTeX for the settings sentences (underscores and percent signs)."""
    return s.replace("_", "\\_").replace("%", "\\%")


def settings_sentence(out: Path, tex: bool) -> str:
    """Each tree's settings, read from its refit files (markdown or LaTeX), without a leading label."""
    S = tree_settings(out)
    parts = []
    for m in TREES:
        if m not in S:
            continue
        s = S[m]
        parts.append(
            f"{LABEL[m]}: {s['name'] or 'settings as recorded'} ({_fmt_params(m, s['params'])}"
            + (f"; {s['note']}" if s["note"] else "")
            + ")"
        )
    txt = (
        "; ".join(parts)
        + ". Leaf minimums are scaled from the pooled 24,000-row window to the 2,000-row window as in the per-bar spec."
    )
    return _tx(txt) if tex else txt


def rerun_models(out: Path) -> dict:
    """model -> its re-run gate table (rows of the shown buckets), for the models re-run on new settings."""
    S = tree_settings(out)
    res: dict = {}
    for m, f in RERUN_GATES.items():
        if m not in S or S[m]["interim"] or not S[m]["name"]:
            continue
        g = _read(out / f)
        res[m] = (
            g[g.bucket.isin(BUCKETS)] if g is not None and "bucket" in g.columns else g
        )
    return res


def _rerun_sentences(out: Path) -> list[str]:
    L = []
    S = tree_settings(out)
    for m, g in rerun_models(out).items():
        if g is None:
            L.append(
                f"{LABEL[m]} ({S[m]['name']}): its re-run gate file `{RERUN_GATES[m]}` is missing."
            )
            continue
        asr = g[g["asserted"].astype(bool)]
        rep_ = g[~g["asserted"].astype(bool)]
        s = (
            f"{LABEL[m]} re-run on its new settings ({S[m]['name']}; CPU "
            + ", ".join(S[m]["cpus"] or ["not recorded"])
            + f"; `{RERUN_GATES[m]}`): asserted gates "
            + ("pass" if asr["ok"].all() else "FAIL")
            + " ("
            + "; ".join(f"{r.gate}: {r.detail}" for r in asr.itertuples())
            + ")"
        )
        if len(rep_):
            s += "; reported: " + "; ".join(
                f"{str(r.gate).replace(' [reported]', '')}: {r.detail}"
                for r in rep_.itertuples()
            )
        L.append(
            s
            + ". Its stored-run gates (no-mask reproduction, the mask's effect, canonical table, cadence) compare its old settings and are left out here."
        )
    return L


def _settings_change_note(out: Path, tex: bool) -> str:
    """The tree settings changed since the first pass, read from the refit files."""
    S = tree_settings(out)
    ch = [m for m in ("xgb", "rf") if m in S and S[m]["name"] and not S[m]["interim"]]
    s = (
        "Settings changed since the first pass: "
        + ("; ".join(CHANGE[m] for m in ch) or "none")
        + "."
    )
    if ch:
        s += (
            " For "
            + " and the ".join(LABEL[m] for m in ch)
            + " the before / after comparison mixes the settings change with the design and the mask."
        )
    if "rf" in S and S["rf"]["interim"]:
        s += " INTERIM: the random forest is still the library-default forest in this build; its refits on trial 255 are running."
    return _tx(s) if tex else s


def _esc(s: str) -> str:
    return (
        str(s)
        .replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("#", "\\#")
        .replace("|", "$|$")
    )


def _fmt(v: float, m: str, pct: float | None = None) -> str:
    """A share in %, a loss change as % of the model's tail loss (or x 1e-3 when absent)."""
    if not np.isfinite(v):
        return "--"
    if m in SHARE:
        return f"{100 * v:.0f}%"
    if pct is not None and np.isfinite(pct):
        return f"{pct:+.0f}%"
    return f"{1e3 * v:+.2f}e-3"


# ============================================================================ intro
def _design_numbers(out: Path) -> dict:
    D = _read(out / "gates_design.csv")
    G = _read(out / "gates.csv")
    n: dict = {}
    if D is not None:
        n["p"] = {r.bucket: (int(r.p_old), int(r.p_new)) for r in D.itertuples()}
        n["design_ok"] = bool(D["ok"].all())
    if G is not None and "kept_columns_median" in G.columns:
        k = G[G.model.isin(TREES)].groupby("bucket")
        n["kept"] = {
            b: (
                int(g.kept_columns_min.min()),
                float(g.kept_columns_median.median()),
                int(g.kept_columns_max.max()),
            )
            for b, g in k
        }
    return n


def intro_md(out: Path, K: int) -> list[str]:
    n = _design_numbers(out)
    p = n.get("p", {})
    kept = n.get("kept", {})
    return [
        "## What changed from the first pass (`results/feature_importance_1530/`)",
        "- The design (commit 47f7f9c): the per-bar design no longer carries the twelve HAR x open / close session-edge interactions. At 16:00 the six `har_ma_k_x_open` columns were all zero and the six `har_ma_k_x_close` columns exact copies of `har_ma_1 .. har_ma_3125` (checked bit for bit, gate below). Columns "
        + ", ".join(f"{b} {p[b][0]} -> {p[b][1]}" for b in BUCKETS if b in p)
        + ".",
        "- The trees (user decision 2026-09-29, commit f9a19b6): every tree refit and probe fit uses only the columns `src/models/window_mask.window_keep` keeps on its own 2000-session training window (constant columns and exact copies of an earlier kept column removed) -- the per-bar linear arms' identifiability mask applied to the trees. MDI / gain, split count and TreeSHAP are computed on the kept columns and mapped back to all p columns with 0 for a dropped column (the model never saw it); permuting a dropped column leaves the forecast unchanged (0, exactly). Kept columns per refit (min / median / max over refits and tree models): "
        + "; ".join(
            f"{b} {kept[b][0]} / {kept[b][1]:.0f} / {kept[b][2]} of {p.get(b, (0, 0))[1]}"
            for b in BUCKETS
            if b in kept
        )
        + ". The linear models are unchanged by the mask: their identifiability mask already removed constant columns and copies (including every session-edge column) at every tune.",
        f"- Everything else is the first pass's: the same {K} refits (every 10 sessions), the same causal held-out tails, the same permutation generator and seeds (by bucket and refit), the same five models, parameters, seeds and two designs; one thread per fit. So before / after differs only by the design (and, for the trees, the mask) -- and by the permutation draws of most units: the stream is laid out unit by unit (every unit's P1 draws, then every unit's P2 draws), and the old design's twelve extra columns were its last twelve, so a column's P1 draws are identical in both runs while the P2 draws and the series / cluster units' draws are fresh draws of the same scheme. The ridge fits are the same problem in both runs (coefficients equal to 1e-10), so the ridge rows of the before / after tables measure that Monte Carlo noise alone. The intermediate rung -- de-dup, no mask -- is `results/feature_importance_1530_dedup_nomask/` (same code, FEATIMP_WINDOW_MASK=0).",
        "- Cadence: the campaign's trees now refit every session (`yhat_subtree_daily_*`). At an importance refit the model is identical under either cadence (the same 2000-session window, the same configuration), so the importance recorded at every 10th refit is the daily-refit model's importance on those days (checked on the stored runs, gate below).",
        "- " + _settings_change_note(out, tex=False),  # 2026-10-08
        "",
    ]


def intro_tex(out: Path, K: int) -> str:
    n = _design_numbers(out)
    p = n.get("p", {})
    kept = n.get("kept", {})
    return (
        r"""
\textbf{What changed from the first pass.} (i) The design: the per-bar design no longer carries the twelve HAR$\times$open/close session-edge interactions (at 16:00 the six \texttt{har\_ma\_k\_x\_open} columns were all zero and the six \texttt{har\_ma\_k\_x\_close} columns exact copies of \texttt{har\_ma\_1 .. har\_ma\_3125}); columns """
        + ", ".join(f"{BNAME[b]} {p[b][0]}$\\to${p[b][1]}" for b in BUCKETS if b in p)
        + r""". (ii) The trees: every tree fit uses only the columns kept by the per-window mask of its own 2000-session training window (constant columns and exact copies removed; the linear arms' identifiability mask applied to the trees); MDI, split count and TreeSHAP are mapped back to all columns with 0 for a dropped column, and permuting a dropped column changes nothing. Kept columns per refit (min / median / max): """
        + "; ".join(
            f"{BNAME[b]} {kept[b][0]} / {kept[b][1]:.0f} / {kept[b][2]} of {p.get(b, (0, 0))[1]}"
            for b in BUCKETS
            if b in kept
        )
        + r""". The linear models are unchanged by the mask (theirs already removed these columns at every tune). Everything else is the first pass's ("""
        + str(K)
        + r""" refits, the same held-out tails, permutation generator and seeds, models and parameters), so before/after differs only by the design and the mask --- and by the permutation draws of most units (the stream is laid out unit by unit and the unit count changed: a column's P1 draws are identical in both runs, the P2 draws and the group units' draws are fresh; the ridge fits are the same problem in both runs, so the ridge rows measure that Monte Carlo noise). (iii) Cadence: the campaign's trees now refit every session; at an importance refit the model is identical under either cadence (same window, same configuration), so the importance at every 10th refit is the daily-refit model's importance on those days (Section 8). Before/after rankings: Section 9. (iv) """
        + _settings_change_note(out, tex=True)  # 2026-10-08
        + r"""
"""
    )


def buckets_md(root_rel: str, stem: str) -> str:
    return (
        "Buckets: all_features. Scripts: `experiments/feature_importance_1530_dedup.py` (the re-run's roots and stages) driving the first pass's `experiments/feature_importance_1530.py` (+ `_trees.py` for the refits, run on the cluster; the per-window mask behind FEATIMP_WINDOW_MASK=1); "
        f"tables in `{root_rel}/`, PDF `writeup/{stem}.pdf`."
    )


def tuned_md(tuned_ok: bool, tuned_lead: tuple[int, int] | None) -> str:
    if tuned_ok and tuned_lead is not None:
        return f"The causally tuned trees' own extracts (random search, refit every 10; importance per refit and TreeSHAP rows at 16:00, from the masked tuned runs): `har_ma_*` leads both MDI and SHAP in {tuned_lead[0]} of {tuned_lead[1]} bucket x model arms; shares tuned vs untuned in `tuned_trees_series.csv`. Split count and permutation were not computed for the tuned trees."
    return "Pending: the masked tuned trees (random search, the campaign's rung RS10 on the de-dup design WITH the window mask) had not been merged when this was built, so the tuned-tree table is left out rather than filled with the unmasked runs; the no-mask root carries the unmasked tuned trees' table (`results/feature_importance_1530_dedup_nomask/tuned_trees_series.csv`)."


# ============================================================================ gates
def _gate_numbers(out: Path, G: pd.DataFrame) -> dict:
    n: dict = {}
    RR = rerun_models(
        out
    )  # 2026-10-08: re-run models have their own gate (RERUN_GATES)
    gm = G.set_index(["bucket", "model"])
    GN = _read(NOMASK / "gates.csv")
    if GN is not None:
        gn = GN.set_index(["bucket", "model"])
        n["repro"] = {
            (b, m): (
                float(gn.loc[(b, m), "forecast_gap_max"]),
                float(gn.loc[(b, m), "forecast_gap_mean"]),
                float(gn.loc[(b, m), "forecast_corr_stored"]),
                int(gn.loc[(b, m), "refits_gap_above_1e9"]),
            )
            for b in BUCKETS
            for m in TREES
            if m not in RR  # 2026-10-08
        }
    n["mask_effect"] = {
        (b, m): (
            float(gm.loc[(b, m), "forecast_gap_max"]),
            float(gm.loc[(b, m), "forecast_gap_mean"]),
            float(gm.loc[(b, m), "forecast_corr_stored"]),
        )
        for b in BUCKETS
        for m in TREES
        if m not in RR  # 2026-10-08
    }

    def _rows(b: str, m: str) -> int:
        """Rows above 1e-9 (a walk run before the count was recorded: 0 when its max is below)."""
        v = gm.loc[(b, m)].get("capture_vs_stored_rows_1e9", np.nan)
        if np.isfinite(v):
            return int(v)
        return 0 if float(gm.loc[(b, m), "capture_vs_stored_rel"]) <= 1e-9 else -1

    n["cap"] = {
        (b, m): (float(gm.loc[(b, m), "capture_vs_stored_rel"]), _rows(b, m))
        for b in BUCKETS
        for m in LIN
    }
    n["anchor"] = {
        (b, m): (
            float(gm.loc[(b, m), "anchor_resolve_gap_max"]),
            int(gm.loc[(b, m), "anchor_refits_above_1e6"]),
        )
        for b in BUCKETS
        for m in LIN
    }
    n["shap_add"] = max(
        float(gm.loc[(b, m), "shap_additivity_max"]) for b in BUCKETS for m in TREES
    )
    shown = G[G.bucket.isin(BUCKETS)]
    n["draws"] = bool((shown["draws_equal_to_lgbm"] == shown["refits"]).all())
    C = _read(out / "gates_cadence.csv")
    if C is not None:
        n["cadence"] = C[C.bucket.isin(BUCKETS)] if "bucket" in C.columns else C
        if "model" in n["cadence"].columns:  # 2026-10-08
            n["cadence"] = n["cadence"][~n["cadence"].model.isin(list(RR))]
    Q = _read(out / "gates_canonical.csv")
    if Q is not None:
        n["canonical"] = Q[Q.bucket.isin(BUCKETS)] if "bucket" in Q.columns else Q
        if "model" in n["canonical"].columns:  # 2026-10-08
            n["canonical"] = n["canonical"][~n["canonical"].model.isin(list(RR))]
    D = _read(out / "gates_design.csv")
    if D is not None:
        n["design"] = D[D.bucket.isin(BUCKETS)] if "bucket" in D.columns else D
    return n


def _cadence_sentence(C: pd.DataFrame | None) -> str:
    if C is None or not len(C):
        return (
            "Cadence identity: not checked (the every-session runs were not present)."
        )
    parts = []
    for v, g in C.groupby("variant", sort=False):
        if (
            "t1_vs_t10_forecast_max_abs" not in g.columns
            or g["t1_vs_t10_forecast_max_abs"].isna().all()
        ):
            parts.append(
                f"{v}: the every-session run of this variant was not merged when this was built"
                if v == "mask"
                else f"{v}: every-session run not present"
            )
            continue
        g = g.dropna(subset=["t1_vs_t10_forecast_max_abs"])
        s = f"{v} ({len(g)} bucket x model arms, {int(g['refits'].iloc[0])} importance refits each): every-session vs every-10 run on the importance-refit days, forecast max |gap| {g['t1_vs_t10_forecast_max_abs'].max():.1e}, native importance max |gap| {g['t1_vs_t10_importance_max_abs'].max():.1e}"
        if (
            "t1_vs_refit_forecast_max_abs" in g.columns
            and g["t1_vs_refit_forecast_max_abs"].notna().any()
        ):
            ok = g[g["t1_vs_refit_forecast_max_abs"] <= 1e-9]
            s += f"; this study's refit vs the every-session run: forecast equal (<= 1e-9) in {len(ok)} of {len(g)} arms"
            bad = g[g["t1_vs_refit_forecast_max_abs"] > 1e-9]
            if len(bad):
                s += (
                    " (not in "
                    + ", ".join(
                        f"{BNAME[r.bucket]} {LABEL[r.model]}: {r.t1_vs_refit_forecast_max_abs:.1e}"
                        for r in bad.itertuples()
                    )
                    + ")"
                )
        parts.append(s)
    return "Cadence identity -- " + "; ".join(parts) + "."


def gates_md(out: Path, G: pd.DataFrame) -> list[str]:
    n = _gate_numbers(out, G)
    L = ["## Gates"]
    D = n.get("design")
    if D is not None:
        L.append(
            "- Design (`gates_design.csv`): "
            + "; ".join(
                f"{r.bucket} p {r.p_new} (want {r.p_expected}; old {r.p_old}), no `_x_` column: {r.session_edge_columns_new == 0}, old minus new = the 12 session-edge columns: {r.old_minus_new_is_session_edge}"
                for r in D.itertuples()
            )
            + f"; the same out-of-sample stamps, the same targets and every other column bit for bit as the first pass's design: {bool(D['oos_stamps_equal'].all() and D['target_bitwise_equal'].all() and D['other_columns_bitwise_equal'].all())}; the old x_open columns all zero and the old x_close columns = `har_ma_k` bit for bit: {bool(D['old_x_open_all_zero'].all() and D['old_x_close_equal_har_ma'].all())}. Stamps, targets and column names also equal the stored de-dup tree runs'."
        )
    if "repro" in n:
        rp = n["repro"]
        exact = [(b, m) for (b, m), v in rp.items() if v[0] <= 1e-9]
        notx = [(b, m) for (b, m), v in rp.items() if v[0] > 1e-9]
        L.append(
            "- Refits vs the stored forecasts (the reproduction gate, no-mask root vs the stored de-dup T10 runs, unmasked, agent B): bit for bit (<= 1e-9) in "
            + f"{len(exact)} of {len(rp)} bucket x model arms"
            + (
                "; not in "
                + "; ".join(
                    f"{BNAME[b]} {LABEL[m]} (max |gap| {rp[(b, m)][0]:.3f}, mean {rp[(b, m)][1]:.4f}, correlation {rp[(b, m)][2]:.4f}, {rp[(b, m)][3]} of 147 refits above 1e-9)"
                    for b, m in notx
                )
                if notx
                else ""
            )
            + ". The LightGBM gap is the first pass's again, now with ONE thread on both sides (the thread count is ruled out), and forcing LightGBM's column-wise or row-wise histograms leaves it unchanged; the refit is deterministic run to run and the cluster and a laptop give the same gap, so the stored LightGBM runs differ in something the input file does not carry. The LightGBM importance is that of the refit."
        )
    me = n["mask_effect"]
    L.append(
        "- The mask's effect on the forecasts (masked refits vs the same unmasked stored runs): max |gap| "
        + ", ".join(
            # f"{BNAME[b]} {LABEL[m]} {me[(b, m)][0]:.3f}" for b in BUCKETS for m in TREES
            f"{BNAME[b]} {LABEL[m]} {me[(b, m)][0]:.3f}"
            for (b, m) in me  # 2026-10-08
        )
        + "; correlation "
        + f"{min(v[2] for v in me.values()):.4f}..{max(v[2] for v in me.values()):.4f}."
    )
    Q = n.get("canonical")
    if Q is not None and len(Q):
        if (
            "table_equals_unmasked_t10" in Q.columns
            and Q["table_equals_unmasked_t10"].fillna(True).astype(bool).all()
        ):
            L.append(
                "- Canonical masked tree tables: `results/spxw_pnl/yhat_subtree_<model>_<bucket>` still equal the unmasked T10 runs bit for bit ("
                + f"{len(Q)} of {len(Q)}"
                + ") when this was built -- the masked re-run of the campaign (agent H) had not replaced them, so the masked refits are gated against their own unmasked twins (above) and the design; `gates_canonical.csv` repeats the check once the masked tables land."
            )
        else:
            col = "refit_mask_vs_table_max_abs"
            ok = Q[Q[col] <= 1e-9] if col in Q.columns else Q.iloc[0:0]
            L.append(
                f"- Masked refits vs the canonical masked tree tables (`results/spxw_pnl/yhat_subtree_<model>_<bucket>`, agent H): bit for bit (<= 1e-9) in {len(ok)} of {len(Q)} arms"
                + (
                    "; otherwise max |gap| "
                    + ", ".join(
                        f"{BNAME[r.bucket]} {LABEL[r.model]} {getattr(r, col):.3g}"
                        for r in Q.itertuples()
                        if getattr(r, col) > 1e-9
                    )
                    if len(ok) < len(Q)
                    else ""
                )
                + " (`gates_canonical.csv`)."
            )
    L.append("- " + _cadence_sentence(n.get("cadence")))
    L += ["- " + s for s in _rerun_sentences(out)]  # 2026-10-08
    cap = n["cap"]
    L.append(
        "- The linear coefficient captures (re-run on the new design through the spec's own class): coefficient x row + intercept = the forecast; capture vs the stored de-dup research forecasts (agent A, `results/linear_subsection_dedup/arms_hoffman2`): ridge max rel "
        + ", ".join(f"{BNAME[b]} {cap[(b, 'ridge')][0]:.0e}" for b in BUCKETS)
        + "; lasso "
        + ", ".join(
            f"{BNAME[b]} {cap[(b, 'lasso')][0]:.1e}"
            + (
                f" ({cap[(b, 'lasso')][1]} of 1469 rows above 1e-9)"
                if cap[(b, "lasso")][1] > 0
                else ""
            )
            for b in BUCKETS
        )
        + ". The lasso gaps are the warm-homotopy float path of the spec's lasso: it moves with the CPU architecture (agent A: a re-run on the same architecture is bit-identical, another machine moves 16:00 forecasts by 5e-4..8e-2), and the stored arms ran on the other cluster; the lasso importance is that of this capture. Drop-column anchors reproduce the captures: ridge <= "
        + f"{max(v[0] for (b, m), v in n['anchor'].items() if m == 'ridge'):.0e}, lasso <= {max(v[0] for (b, m), v in n['anchor'].items() if m == 'lasso'):.0e} (above 1e-6 on "
        + " / ".join(str(n["anchor"][(b, "lasso")][1]) for b in BUCKETS)
        + " of 147 refits, "
        + " / ".join(BNAME[b] for b in BUCKETS)
        + "). The all_features lasso refits 75..124 (penalty 0.001) ran on the cluster (`cluster/slurm/submit_featimp_dedup.sh linear`), the rest locally; the parts partition the 147 refits (checked by the merge)."
    )
    L.append(
        f"- Every model of a bucket saw the identical permutation draws (md5 per refit equal across the five models): {n['draws']}. TreeSHAP additivity: sum of phi + expected value = forecast to <= {n['shap_add']:.0e}."
    )
    L.append("")
    return L


def gates_tex(out: Path, G: pd.DataFrame) -> str:
    md = gates_md(out, G)[1:]
    items = [x[2:] for x in md if x.startswith("- ")]
    return (
        "\n\\section*{7. Gates}\n\\begin{itemize}\\itemsep1pt\n"  # 2026-10-08: was 8 (sections 5 and 9 commented out)
        + "\n".join("\\item " + _tex_inline(x) for x in items)
        + "\n\\end{itemize}\n"
    )


def _show(s: str) -> str:
    """A design name as the PDF presents it: without the leading ``adj_`` (user decision
    2026-10-06; display only, the CSVs keep the stored names)."""
    return re.sub(r"\badj_", "", str(s))


def _tex_inline(s: str) -> str:
    """Markdown bullet -> LaTeX: `code` -> texttt (names as _show presents them), escapes."""
    out, code = [], False
    for part in s.split("`"):
        out.append(("\\texttt{" + _esc(_show(part)) + "}") if code else _esc(part))
        code = not code
    return (
        "".join(out)
        .replace("->", "$\\to$")
        .replace("<=", "$\\le$")
        .replace(">=", "$\\ge$")
    )


# ============================================================================ before / after
def _ba(out: Path) -> dict:
    return dict(
        sp=_read(out / "before_after_spearman.csv"),
        har=_read(out / "before_after_har_ma.csv"),
        harc=_read(out / "before_after_har_ma_columns.csv"),
        mv=_read(out / "before_after_movers.csv"),
        cl=_read(out / "before_after_clusters.csv"),
    )


def _har_rows(har: pd.DataFrame, harc: pd.DataFrame, b: str) -> list[tuple]:
    rows = []
    for model in MODELS:
        ms = HEAD_T if model in TREES else HEAD_L
        for m in ms:
            h = har[(har.bucket == b) & (har.model == model) & (har.measure == m)]
            c = harc[(harc.bucket == b) & (harc.model == model) & (harc.measure == m)]
            if not len(h):
                continue
            h, c = h.iloc[0], (c.iloc[0] if len(c) else None)
            rows.append(
                (
                    model,
                    m,
                    _fmt(h.value_old, m, h.get("pct_of_loss_old")),
                    _fmt(h.value_nomask, m, h.get("pct_of_loss_nomask"))
                    if model in TREES and np.isfinite(h.value_nomask)
                    else "",
                    _fmt(h.value_new, m, h.get("pct_of_loss_new")),
                    f"{100 * c.old_x_close_share_of_har_columns:.0f}%"
                    if c is not None and np.isfinite(c.old_x_close_share_of_har_columns)
                    else "--",
                    int(h.rank_old),
                    int(h.rank_new),
                )
            )
    return rows


def before_after_md(out: Path) -> list[str]:
    X = _ba(out)
    if X["sp"] is None or X["har"] is None:
        return ["## Before / after", "Pending: `before_after_*.csv` not built.", ""]
    sp, har, harc, mv, cl = (
        _shown(X["sp"]),
        _shown(X["har"]),
        _shown(X["harc"]),
        _shown(X["mv"]),
        _shown(X["cl"]),
    )
    if sp is None or har is None or harc is None:
        return ["## Before / after", "Pending: `before_after_*.csv` not built.", ""]
    L = [
        "## Before / after: first pass (old design, trees unmasked) vs de-dup + mask",
        "Tables `before_after_ranks_<bucket>_<model>.csv` (every unit of every measure and level: value and rank old / new [/ de-dup no mask for the trees], the rank among the units common to both runs and its change), `before_after_spearman.csv`, `before_after_movers.csv`, `before_after_har_ma.csv`, `before_after_har_ma_columns.csv`, `before_after_clusters.csv` (clusters are renumbered per run; matched by their members once the session-edge columns are removed from the old cluster).",
        _settings_change_note(out, tex=False),  # 2026-10-08
        "",
    ]
    # headline: har_ma_* in the trees, and the ridge rows as the draw-noise yardstick
    for b in ("all_features",):  # live_feasible commented out
        cells = []
        for model in TREES:
            h = har[(har.bucket == b) & (har.model == model)].set_index("measure")
            c = harc[(harc.bucket == b) & (harc.model == model)].set_index("measure")
            cells.append(
                f"{LABEL[model]} MDI {100 * h.loc['mdi', 'value_old']:.0f} -> {100 * h.loc['mdi', 'value_new']:.0f} %, "
                f"SHAP {100 * h.loc['shap', 'value_old']:.0f} -> {100 * h.loc['shap', 'value_new']:.0f} %, "
                f"perm P2 dQLIKE {h.loc['perm_p2_qlike', 'pct_of_loss_old']:+.0f} -> {h.loc['perm_p2_qlike', 'pct_of_loss_new']:+.0f} % of tail loss "
                f"(the old `_x_close` copies carried {100 * c.loc['mdi', 'old_x_close_share_of_har_columns']:.0f} % of the old har_ma column MDI)"
            )
        L.append(f"- `har_ma_*` in the trees, {b}: " + "; ".join(cells) + ".")
    rr = har[(har.model == "ridge") & (har.measure == "perm_p2_mse")]
    d_mse = (rr["pct_of_loss_new"] - rr["pct_of_loss_old"]).abs()
    L.append(
        f"- The draw-noise yardstick: the ridge fits are the same problem in both runs, so its `har_ma_*` change is the Monte Carlo of fresh permutation draws alone: perm P2 dMSE moves by at most {d_mse.max():.1f} points of % tail loss over the {rr['bucket'].nunique()} buckets (abs(beta x sd), SHAP and drop-column unchanged)."
    )
    # Spearman summary per level
    for level in ("series", "cluster", "column"):
        s = sp[sp.level == level]
        tr = s[s.model.isin(TREES)]["spearman_old_new"]
        li = s[s.model.isin(LIN)]["spearman_old_new"]
        line = f"- Spearman old vs new ({level} level, common units, every bucket x measure): trees median {tr.median():.2f} (min {tr.min():.2f}), linear median {li.median():.2f} (min {li.min():.2f})"
        if "spearman_old_nomask" in s.columns:
            a = s[s.model.isin(TREES)]["spearman_old_nomask"]
            c = s[s.model.isin(TREES)]["spearman_nomask_new"]
            line += f"; trees old vs de-dup no mask {a.median():.2f} (min {a.min():.2f}), no mask vs mask {c.median():.2f} (min {c.min():.2f})"
        L.append(line + ".")
    lead = sp[(sp.level == "series") & (sp.leader_old != sp.leader_new)]
    L.append(
        "- Series leader changed in "
        + f"{len(lead)} of {int((sp.level == 'series').sum())} bucket x model x measure rows"
        + (
            ": "
            + "; ".join(
                f"{r.bucket} {LABEL[r.model]} {MNAME.get(r.measure, r.measure)} `{r.leader_old}` -> `{r.leader_new}`"
                for r in lead.itertuples()
            )
            if len(lead)
            else ""
        )
        + "."
    )
    if cl is not None:
        for b in BUCKETS:
            c = cl[cl.bucket == b]
            L.append(
                f"- Clusters {b}: "
                + ", ".join(f"{k} {v}" for k, v in c["status"].value_counts().items())
                + "."
            )
    L.append("")
    L.append(
        "### `har_ma_*` once its duplicates are gone (series value old / de-dup no mask / new; permutation and drop-column as % of the model's tail loss; `x_close share` = the old `_x_close` copies' part of the old har_ma column total)"
    )
    for b in ("all_features",):  # baseline commented out; live_feasible commented out
        L += [
            "",
            f"{b}:",
            "",
            "| model | measure | old | no mask | new | x_close share (old) | rank old | rank new |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in _har_rows(har, harc, b):
            L.append(
                f"| {LABEL[r[0]]} | {MNAME[r[1]]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} | {r[7]} |"
            )
    L.append("")
    if harc is not None:
        L.append(
            "Column level, the dilution the copies caused (permutation P2 dQLIKE x 1e-3 of `har_ma_1` / `har_ma_5` alone, old -> new; old = its `_x_close` copy left intact): "
            + "; ".join(
                f"{BNAME[b]} {LABEL[m]} "
                + f"{1e3 * float(harc[(harc.bucket == b) & (harc.model == m) & (harc.measure == 'perm_p2_qlike')]['old_har_ma_1'].iloc[0]):.1f} -> {1e3 * float(harc[(harc.bucket == b) & (harc.model == m) & (harc.measure == 'perm_p2_qlike')]['new_har_ma_1'].iloc[0]):.1f} / "
                + f"{1e3 * float(harc[(harc.bucket == b) & (harc.model == m) & (harc.measure == 'perm_p2_qlike')]['old_har_ma_5'].iloc[0]):.1f} -> {1e3 * float(harc[(harc.bucket == b) & (harc.model == m) & (harc.measure == 'perm_p2_qlike')]['new_har_ma_5'].iloc[0]):.1f}"
                for b in ("all_features",)  # live_feasible commented out
                for m in TREES
            )
            + "."
        )
        L.append("")
    if mv is not None:
        L.append(
            "### Rows that moved most (series level, top-20 zone of either run, rank among common units old -> new)"
        )
        for b in ("all_features",):  # live_feasible commented out
            L.append("")
            L.append(f"{b}:")
            for model in MODELS:
                ms = HEAD_T if model in TREES else HEAD_L
                cells = []
                for m in ms:
                    t = mv[
                        (mv.bucket == b)
                        & (mv.model == model)
                        & (mv.measure == m)
                        & (mv.level == "series")
                        & (mv.rank_change != 0)
                    ].head(2)
                    if len(t):
                        cells.append(
                            f"{MNAME[m]}: "
                            + ", ".join(
                                f"`{r.unit}` {int(r.rank_old_common)}->{int(r.rank_new_common)}"
                                for r in t.itertuples()
                            )
                        )
                L.append(
                    f"- {LABEL[model]}: "
                    + ("; ".join(cells) if cells else "no series moved")
                )
        L.append("")
    return L


def _mm(v: pd.Series) -> str:
    """median (minimum) of a Spearman column; -- when undefined (HAR + calendar has two series)."""
    v = v.dropna()
    return f"{v.median():.2f} ({v.min():.2f})" if len(v) else "--"


def _shown(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or "bucket" not in df.columns:
        return df
    return df[df.bucket.isin(BUCKETS)].copy()


def _har_figure(out: Path) -> str | None:
    """Two-bucket copy of the har_ma before/after figure. The stored PNG still
    has the live-feasible panel; that file is left in place."""
    src = out / "before_after_har_ma.csv"
    if not src.is_file():
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    H = pd.read_csv(src)
    color = {
        "ridge": "#2a78d6",
        "lasso": "#4a3aa7",
        "lgbm": "#eb6834",
        "xgb": "#1baf7a",
        "rf": "#eda100",
    }
    short = {
        "mdi": "MDI",
        "split": "split",
        "bsd": "|b sd|",
        "shap": "SHAP",
        "perm_p2_qlike": "perm P2 QLIKE",
        "perm_p2_mse": "perm P2 MSE",
    }
    panels = (
        (
            "share of the model's importance (%)",
            {"tree": ("mdi", "split", "shap"), "linear": ("bsd", "shap")},
        ),
        (
            "permutation P2, % of the model's tail loss",
            {"tree": ("perm_p2_qlike",), "linear": ("perm_p2_mse",)},
        ),
    )
    fig, axes = plt.subplots(
        len(BUCKETS),
        2,
        figsize=(11, 4.1 * len(BUCKETS)),
        gridspec_kw={"width_ratios": [1.25, 1]},
        squeeze=False,
    )
    grid = cast(np.ndarray, np.asarray(axes))
    for r, b in enumerate(BUCKETS):
        for c, (xlab, ms) in enumerate(panels):
            ax = grid[r, c]
            rows = []
            measures = cast(dict[str, tuple[str, ...]], ms)
            for model in MODELS:
                fam = "tree" if model in TREES else "linear"
                for m in measures[fam]:
                    h = H[(H.bucket == b) & (H.model == model) & (H.measure == m)]
                    if len(h):
                        rows.append((model, m, h.iloc[0]))
            y = np.arange(len(rows))[::-1]
            for yy, (model, m, h) in zip(y, rows):
                share = m in SHARE
                if share:
                    vo, vn = 100 * h.value_old, 100 * h.value_new
                    vm = 100 * h.value_nomask
                else:
                    vo, vn = h.pct_of_loss_old, h.pct_of_loss_new
                    vm = h.pct_of_loss_nomask
                ax.plot([vo, vn], [yy, yy], color="0.75", lw=1.2, zorder=1)
                ax.scatter([vo], [yy], s=40, color="0.45", zorder=3, lw=0)
                if np.isfinite(vm):
                    ax.scatter(
                        [vm],
                        [yy],
                        s=46,
                        facecolor="none",
                        edgecolor=color[model],
                        lw=1.4,
                        zorder=3,
                    )
                ax.scatter(
                    [vn],
                    [yy],
                    s=46,
                    color=color[model],
                    zorder=4,
                    edgecolor="white",
                    lw=1.0,
                )
                ax.text(
                    max(vo, vn),
                    yy,
                    f"  {vo:.0f} -> {vn:.0f}",
                    va="center",
                    ha="left",
                    fontsize=6.5,
                    color="0.3",
                )
            ax.set_yticks(y)
            ax.set_yticklabels(
                [f"{LABEL[mo]}: {short[m]}" for mo, m, _ in rows], fontsize=7.5
            )
            ax.set_xlabel(xlab, fontsize=8)
            ax.tick_params(axis="x", labelsize=7)
            ax.grid(axis="x", color="0.92", lw=0.6)
            ax.set_axisbelow(True)
            for sp_ in ("top", "right"):
                ax.spines[sp_].set_visible(False)
            lo, hi = ax.get_xlim()
            ax.set_xlim(min(0, lo), hi + 0.18 * (hi - lo))
            if c == 0:
                ax.set_title(
                    f"{BNAME[b]}: har_ma_* before -> after", fontsize=9, loc="left"
                )
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            color="0.45",
            lw=0,
            markersize=6,
            label="first pass (old design, trees unmasked)",
        ),
        Line2D(
            [],
            [],
            marker="o",
            markerfacecolor="none",
            markeredgecolor="0.2",
            lw=0,
            markersize=6,
            label="de-dup, no mask (trees)",
        ),
        Line2D(
            [],
            [],
            marker="o",
            color="0.2",
            lw=0,
            markersize=6,
            label="de-dup + mask (model colour)",
        ),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7.5, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    name = "fig_before_after_har_ma_shown.png"
    fig.savefig(out / name, dpi=150)
    plt.close(fig)
    return name


def before_after_tex(out: Path, root_rel: str) -> str:
    X = _ba(out)
    if X["sp"] is None or X["har"] is None:
        return "\n\\section*{9. Before / after}\nPending.\n"
    sp, har, harc = _shown(X["sp"]), _shown(X["har"]), _shown(X["harc"])
    if sp is None or har is None or harc is None:
        return "\n\\section*{9. Before / after}\nPending.\n"
    body = [
        "\n\\section*{9. Before / after: first pass (old design) vs de-dup + mask}\n"
    ]
    body.append(
        "Every unit of every measure and level, old vs new: \\texttt{before\\_after\\_ranks\\_<bucket>\\_<model>.csv}; clusters are renumbered per run and matched by their members once the session-edge columns are removed from the old cluster. "
        + _settings_change_note(out, tex=True)  # 2026-10-08
        + " Spearman between the old and the new importance over the units common to both runs:\n"
    )
    rows = []
    for b in BUCKETS:
        for level in ("series", "cluster", "column"):
            s_ = sp[(sp.bucket == b) & (sp.level == level)]
            cells = [
                _mm(s_[s_.model.isin(TREES)]["spearman_old_new"]),
                _mm(s_[s_.model.isin(LIN)]["spearman_old_new"]),
            ]
            if "spearman_old_nomask" in s_.columns:
                t = s_[s_.model.isin(TREES)]
                cells += [_mm(t["spearman_old_nomask"]), _mm(t["spearman_nomask_new"])]
            rows.append(f"{BNAME[b]} & {level} & " + " & ".join(cells) + " \\\\")
    body.append(
        "\\begin{table}[H]\\centering\\scriptsize\n\\caption{Spearman, old vs new importance over the common units: median (minimum) over the model's measures. Trees also old vs de-dup without the mask and no mask vs mask.}\n\\begin{tabular}{llrrrr}\\toprule\nbucket & level & trees old--new & linear old--new & trees old--no mask & trees no mask--mask\\\\\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\\end{tabular}\\end{table}\n"
    )
    for b in ("all_features",):  # baseline commented out; live_feasible commented out
        hr = _har_rows(har, harc, b)
        trows = [
            f"{LABEL[r[0]]} & {MNAME[r[1]].replace('|', '$|$')} & {_esc(r[2])} & {_esc(r[3])} & {_esc(r[4])} & {_esc(r[5])} & {r[6]} & {r[7]} \\\\"
            for r in hr
        ]
        body.append(
            "\\begin{table}[H]\\centering\\scriptsize\n\\caption{"
            + BNAME[b]
            + ": \\texttt{har\\_ma\\_*} (series) old / de-dup no mask / new: share for MDI, split, $|\\beta sd|$, SHAP; permutation and drop-column as \\% of the model's tail loss. \\emph{x\\_close share}: the old \\texttt{\\_x\\_close} copies' part of the old \\texttt{har\\_ma} column total.}\n\\begin{tabular}{llrrrrrr}\\toprule\nmodel & measure & old & no mask & new & x\\_close share & rank old & rank new\\\\\\midrule\n"
            + "\n".join(trows)
            + "\n\\bottomrule\\end{tabular}\\end{table}\n"
        )
    shown = _har_figure(out)
    if shown:
        body.append(
            "\\begin{figure}[H]\\centering\\includegraphics[width=\\textwidth]{../"
            + root_rel
            + "/"
            + shown
            + "}\n\\caption{\\texttt{har\\_ma\\_*} before (old design) and after (de-dup + mask; open marker: de-dup, no mask), per model and measure; left: shares, right: permutation P2 as \\% of the model's tail loss.}\\end{figure}\n"
        )
    md = before_after_md(out)
    txt = [x for x in md if x.startswith("- ") or x.startswith("Column level")]
    body.append(
        "\\begin{itemize}\\itemsep1pt\n"
        + "\n".join("\\item " + _tex_inline(x.lstrip("- ")) for x in txt)
        + "\n\\end{itemize}\n"
    )
    return "".join(body)


def files_md(root_rel: str) -> list[str]:
    return [
        "## Files",
        f"- `{root_rel}/by_measure/imp_<bucket>_<model>_<measure>.csv` and the first pass's tables (`series_level_all.csv`, `rank_corr_*.csv`, `top5_stability.csv`, `stability_by_stratum.csv`, `unique_values.csv`, `cardinality_vs_measure.csv`, `noise_probes.csv`, `clusters.csv`, `cluster_dilution.csv`, `tuned_trees_series.csv`, `gates.csv`) on the new design",
        "- the re-run's own: `gates_design.csv`, `gates_cadence.csv`, `gates_canonical.csv`, `before_after_*.csv`, `fig_before_after_*.png`; the no-mask rung in `results/feature_importance_1530_dedup_nomask/` (same tables)",
        "- scripts: `experiments/feature_importance_1530_dedup.py` (roots, stages, gates), `experiments/feature_importance_1530_dedup_compare.py` (before / after), `writeup/make_feature_importance_1530_tex.py --root results/feature_importance_1530_dedup --stem feature_importance_1530_dedup --dedup` (+ `writeup/feature_importance_1530_dedup_sections.py`)",
        "- cluster twins: `cluster/slurm/{ship_featimp_dedup_carc.sh, submit_featimp_dedup.sh, featimp_dedup_pack.sbatch, featimp_dedup_collect.sbatch, featimp_dedup_linear.sbatch, pull_featimp_dedup_carc.sh}`, `cluster/featimp_dedup_tasks*.txt`, `cluster/featimp_dedup_linear_tasks.txt`",
    ]


# ============================================================================ series leaders
def _tree_leaders(
    lead: dict, trees: tuple, tree_m: tuple, buckets: tuple
) -> tuple[int, int, list]:
    rows = [(b, m, x) for b in buckets for m in trees for x in tree_m]
    exc = [
        (b, m, x, lead[(b, m, x)]) for b, m, x in rows if lead[(b, m, x)] != "har_ma"
    ]
    return len(rows) - len(exc), len(rows), exc


def tree_leaders_md(lead: dict, trees: tuple, tree_m: tuple, buckets: tuple) -> str:
    n_ok, n, exc = _tree_leaders(lead, trees, tree_m, buckets)
    return (
        f"`har_ma_*` is the leader of {n_ok} of {n} tree bucket x model x measure rows"
        + (
            "; not of "
            + ", ".join(
                f"{BNAME[b]} {LABEL[m]} {MNAME.get(x, x)} (leader `{u}`)"
                for b, m, x, u in exc
            )
            if exc
            else ""
        )
        + ". The cardinality bias matters for everything below the leader."
    )


def tree_leaders_tex(lead: dict, trees: tuple, tree_m: tuple, buckets: tuple) -> str:
    return _tex_inline(tree_leaders_md(lead, trees, tree_m, buckets))


# ============================================================================ importance over time (2026-10-08)
# User request 2026-10-08 (meeting): each tree's importance over the 147 refits, all features, series level, from the
# aggregate's per-refit table (across_time_refits_series.parquet): SHAP mean |phi| share and MDI / gain share per
# refit (rows 1-2), and the same series' rank among the 49 (rows 3-4, capped at 20).  har_ma_* plus the next five
# series by overall share (mean over the refits and the three trees, per measure); the rest summed as "other".
OVER_TIME_FIG = "feature_importance_1530_over_time"  # writeup/figures/<name>.pdf / .png
OVER_TIME_MEASURES = (("shap", "SHAP mean |phi| share"), ("mdi", "MDI / gain share"))
# 2026-10-08 (user): the same figure for the per-bar linear models; |beta x sd| in place of MDI
OVER_TIME_FIG_LIN = "feature_importance_1530_over_time_linear"
OVER_TIME_MEASURES_LIN = (
    ("shap", "SHAP mean |phi| share"),
    ("bsd", "|beta x sd| share"),
)
OVER_TIME_TOP = 5
OVER_TIME_SMOOTH = 5  # refits, centred rolling mean
OVER_TIME_RANK_CAP = 20
CRASH_REFITS = (42, 43)  # tails starting 2020-02-28 and 2020-03-13 (the 2020 crash)
# colour-blind-safe (Paul Tol "muted"); har_ma_* black, the rest grey
_TOL_MUTED = (
    "#CC6677",
    "#332288",
    "#DDCC77",
    "#117733",
    "#88CCEE",
    "#882255",
    "#44AA99",
    "#999933",
    "#AA4499",
)


def over_time_data(
    out: Path, models: tuple = TREES, measures: tuple = OVER_TIME_MEASURES
) -> dict:
    """{measure: (frame of model x refit x series share / rank, top series, n_series)} for all features."""
    a = pd.read_parquet(out / "across_time_refits_series.parquet")
    a = a[
        (a.bucket == "all_features")
        & a.model.isin(models)
        & a.measure.isin([m for m, _ in measures])
    ]
    res = {}
    for meas, _ in measures:
        s = a[a.measure == meas].copy()
        s["date"] = pd.to_datetime(s["date"])
        overall = s.groupby("unit")["value"].mean().sort_values(ascending=False)
        top = [u for u in overall.index if u != "har_ma"][:OVER_TIME_TOP]
        res[meas] = (
            s,
            ["har_ma"] + top,
            int(s.groupby(["model", "refit"]).unit.count().max()),
        )
    return res


def over_time_figure(
    out: Path,
    models: tuple = TREES,
    measures: tuple = OVER_TIME_MEASURES,
    fig_name: str = OVER_TIME_FIG,
) -> tuple[Path, dict]:
    """Draw writeup/figures/<fig_name>.pdf / .png; return the path and the numbers used."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    D = over_time_data(out, models, measures)
    series = list(dict.fromkeys(u for meas in D for u in D[meas][1] if u != "har_ma"))
    colour = {
        "har_ma": "black",
        **{u: _TOL_MUTED[i % len(_TOL_MUTED)] for i, u in enumerate(series)},
    }
    nrow = 2 * len(measures)
    fig, axes = plt.subplots(
        nrow,
        len(models),
        figsize=(3.65 * len(models) + 0.05, 2.35 * nrow),
        sharex=True,
        squeeze=False,
    )
    info: dict = {}
    for j, model in enumerate(models):
        for i, (meas, mname) in enumerate(measures):
            s, top, n_ser = D[meas]
            g = s[s.model == model]
            dates = g.groupby("refit")["date"].first().sort_index()
            share = g.pivot_table(
                index="refit", columns="unit", values="value"
            ).sort_index()
            rank = g.pivot_table(
                index="refit", columns="unit", values="rank"
            ).sort_index()
            other = share.drop(columns=top).sum(axis=1)
            ax_s, ax_r = axes[i, j], axes[len(measures) + i, j]
            for ax in (ax_s, ax_r):
                ax.axvspan(
                    dates.loc[CRASH_REFITS[0]],
                    dates.loc[CRASH_REFITS[-1] + 1],
                    color="0.85",
                    lw=0,
                    zorder=0,
                )
            for u in top + ["other"]:
                y = other if u == "other" else share[u]
                c = "0.55" if u == "other" else colour[u]
                lab = (
                    f"other {n_ser - len(top)}"
                    if u == "other"
                    else _show(str(g[g.unit == u].label.iloc[0]))
                )
                ax_s.plot(dates.values, y.values, color=c, lw=0.5, alpha=0.45)
                ax_s.plot(
                    dates.values,
                    y.rolling(OVER_TIME_SMOOTH, center=True, min_periods=1)
                    .mean()
                    .values,
                    color=c,
                    lw=1.5,
                    label=lab,
                )
                if u != "other":
                    r = rank[u].clip(upper=OVER_TIME_RANK_CAP)
                    ax_r.plot(dates.values, r.values, color=c, lw=0.5, alpha=0.45)
                    ax_r.plot(
                        dates.values,
                        r.rolling(OVER_TIME_SMOOTH, center=True, min_periods=1)
                        .mean()
                        .values,
                        color=c,
                        lw=1.5,
                    )
            ax_s.set_ylim(0, 1)
            ax_r.set_ylim(OVER_TIME_RANK_CAP + 0.5, 0.5)
            ax_r.set_yticks([1, 5, 10, 15, 20])
            ax_r.set_yticklabels(["1", "5", "10", "15", "20+"])
            if j == 0:
                ax_s.set_ylabel(mname.replace(" share", "\nshare"), fontsize=8)
                ax_r.set_ylabel(
                    mname.replace(" share", f"\nrank of {n_ser}"), fontsize=8
                )
            if i == 0:
                ax_s.set_title(LABEL[model], fontsize=10)
            if j == len(models) - 1:
                ax_s.legend(
                    fontsize=6.5,
                    loc="upper left",
                    bbox_to_anchor=(1.01, 1.0),
                    frameon=False,
                )
            for ax in (ax_s, ax_r):
                ax.tick_params(labelsize=7)
                ax.grid(alpha=0.25, lw=0.4)
            info[(model, meas)] = dict(
                har_share_mean=float(share["har_ma"].mean()),
                har_share_crash=float(share.loc[list(CRASH_REFITS), "har_ma"].mean()),
                har_rank1=int((rank["har_ma"] == 1).sum()),
                n_refits=int(len(share)),
            )
    fig.tight_layout()
    path = R / "writeup" / "figures" / f"{fig_name}.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=150)
    plt.close(fig)
    info["series"] = {meas: D[meas][1] for meas in D}
    info["n_series"] = max(D[meas][2] for meas in D)
    return path, info


def over_time_tex(out: Path) -> str:
    """The figure (drawn here, so every build of the PDF redraws it) and its caption."""
    path, info = over_time_figure(out)
    _, li = over_time_figure(
        out, LIN, OVER_TIME_MEASURES_LIN, OVER_TIME_FIG_LIN
    )  # 2026-10-08: linear models
    T = tree_settings(out)
    names = ", ".join(f"{LABEL[m]} ({T[m]['name']})" for m in TREES if m in T)
    har = "; ".join(
        f"{LABEL[m]} {100 * info[(m, 'shap')]['har_share_mean']:.0f}\\% SHAP / {100 * info[(m, 'mdi')]['har_share_mean']:.0f}\\% MDI, "
        f"rank 1 in {info[(m, 'shap')]['har_rank1']} / {info[(m, 'mdi')]['har_rank1']} of {info[(m, 'shap')]['n_refits']} refits"
        for m in TREES
        if (m, "shap") in info
    )
    return (
        "\n\\section*{6. Importance over time}\n"
        "\\begin{figure}[H]\\centering\\includegraphics[width=\\textwidth]{figures/"
        + OVER_TIME_FIG
        + ".pdf}\n\\caption{Importance over time, all features, series level (a series = the sum of its columns), one column per tree: "
        + _tx(names)
        + ". Rows 1--2: each series' share of the refit's total importance, SHAP mean $|\\phi|$ on the refit's held-out tail and MDI / gain of the fitted model; rows 3--4: the same series' rank among the "
        + str(info["n_series"])
        + " series (1 = most important; ranks beyond 20 drawn at 20). Lines: \\texttt{har\\_ma\\_*} and the next "
        + str(OVER_TIME_TOP)
        + " series by overall share (mean over the refits and the three trees), the rest summed as `other'. x axis: the start date of each of the 147 refits' tails. Thin lines: one point per refit --- the trees are refit every 10 sessions, and each point is that refit's model on its own held-out tail (the $\\le 10$ sessions it forecasts before the next refit): SHAP averaged over those tail rows, MDI / gain of the fitted model; thick lines: "
        + str(OVER_TIME_SMOOTH)
        + "-refit centred rolling mean. Grey band: the 2020 crash tails (refits 42--43, tails starting 2020-02-28 and 2020-03-13). Mean \\texttt{har\\_ma\\_*} share and rank-1 count: "
        + har
        + ". SHAP and MDI shares are split credit: \\texttt{har\\_ma\\_*} shares its credit with every input correlated with the HAR ladder, and MDI and SHAP inflate continuous, many-valued columns (the noise-probe and cardinality checks above); the shares below \\texttt{har\\_ma\\_*} are therefore not a ranked feature list.}\\end{figure}\n"
        + over_time_linear_tex(li)
    )


def over_time_linear_tex(info: dict) -> str:
    """Caption of the linear models' figure (2026-10-08, user request)."""
    har = "; ".join(
        f"{LABEL[m]} {100 * info[(m, 'shap')]['har_share_mean']:.0f}\\% SHAP / {100 * info[(m, 'bsd')]['har_share_mean']:.0f}\\% "
        f"$|\\beta\\cdot sd|$, rank 1 in {info[(m, 'shap')]['har_rank1']} / {info[(m, 'bsd')]['har_rank1']} of {info[(m, 'shap')]['n_refits']} refits"
        for m in LIN
        if (m, "shap") in info
    )
    return (
        "\\begin{figure}[H]\\centering\\includegraphics[width=0.8\\textwidth]{figures/"
        + OVER_TIME_FIG_LIN
        + ".pdf}\n\\caption{The same for the per-bar linear models (ridge and lasso, as in Setup), one column per model. Rows 1--2: "
        "each series' share of the refit's SHAP mean $|\\phi|$ ($\\phi_j=\\beta_j(x_j-\\bar x_j)$, window mean as background) on the "
        "refit's held-out tail, and of $|\\beta\\cdot sd|$ (coefficient times the column's standard deviation in the refit's 2000-session "
        "window, summed over a series' columns); rows 3--4: ranks among the "
        + str(info["n_series"])
        + " series. The same 147 refit points and tails as the trees (one point per 10 sessions), the same smoothing and grey crash band. "
        "Mean \\texttt{har\\_ma\\_*} share and rank-1 count: "
        + har
        + ". A linear model's credit is split across correlated columns differently from a tree's (the penalty spreads or "
        "concentrates it), so the shares compare over time within one model, not across the two figures.}\\end{figure}\n"
    )
