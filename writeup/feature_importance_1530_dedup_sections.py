"""The de-dup re-run's own parts of the feature-importance report (checklist I7), used by
writeup/make_feature_importance_1530_tex.py --dedup: the design / mask / cadence note, the gates
and the before / after section, markdown and LaTeX.  Every number is read from the re-run's CSVs
(experiments/feature_importance_1530_dedup.py stages gate_design, cadence, canonical,
before_after, and the aggregates of both roots); nothing is typed in."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parent.parent
FIRST = R / "results" / "feature_importance_1530"
NOMASK = R / "results" / "feature_importance_1530_dedup_nomask"
BUCKETS = ("baseline", "live_feasible", "all_features")
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
        f"- Everything else is the first pass's: the same {K} refits (every 10 sessions), the same causal held-out tails, the same permutation generator and seeds (by bucket and refit), the same five models, parameters, seeds and three designs; one thread per fit. So before / after differs only by the design (and, for the trees, the mask) -- and by the permutation draws of most units: the stream is laid out unit by unit (every unit's P1 draws, then every unit's P2 draws), and the old design's twelve extra columns were its last twelve, so a column's P1 draws are identical in both runs while the P2 draws and the series / cluster units' draws are fresh draws of the same scheme. The ridge fits are the same problem in both runs (coefficients equal to 1e-10), so the ridge rows of the before / after tables measure that Monte Carlo noise alone. The intermediate rung -- de-dup, no mask -- is `results/feature_importance_1530_dedup_nomask/` (same code, FEATIMP_WINDOW_MASK=0).",
        "- Cadence: the campaign's trees now refit every session (`yhat_subtree_daily_*`). At an importance refit the model is identical under either cadence (the same 2000-session window, the same configuration), so the importance recorded at every 10th refit is the daily-refit model's importance on those days (checked on the stored runs, gate below).",
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
        + r""" refits, the same held-out tails, permutation generator and seeds, models and parameters), so before/after differs only by the design and the mask --- and by the permutation draws of most units (the stream is laid out unit by unit and the unit count changed: a column's P1 draws are identical in both runs, the P2 draws and the group units' draws are fresh; the ridge fits are the same problem in both runs, so the ridge rows measure that Monte Carlo noise). (iii) Cadence: the campaign's trees now refit every session; at an importance refit the model is identical under either cadence (same window, same configuration), so the importance at every 10th refit is the daily-refit model's importance on those days (Section 8). Before/after rankings: Section 9.
"""
    )


def buckets_md(root_rel: str, stem: str) -> str:
    return (
        "Buckets: live_feasible (the deck's), all_features, baseline = HAR + calendar. Scripts: `experiments/feature_importance_1530_dedup.py` (the re-run's roots and stages) driving the first pass's `experiments/feature_importance_1530.py` (+ `_trees.py` for the refits, run on the cluster; the per-window mask behind FEATIMP_WINDOW_MASK=1); "
        f"tables in `{root_rel}/`, PDF `writeup/{stem}.pdf`."
    )


def tuned_md(tuned_ok: bool, tuned_lead: tuple[int, int] | None) -> str:
    if tuned_ok and tuned_lead is not None:
        return f"The causally tuned trees' own extracts (random search, refit every 10; importance per refit and TreeSHAP rows at 16:00, from the masked tuned runs): `har_ma_*` leads both MDI and SHAP in {tuned_lead[0]} of {tuned_lead[1]} bucket x model arms; shares tuned vs untuned in `tuned_trees_series.csv`. Split count and permutation were not computed for the tuned trees."
    return "Pending: the masked tuned trees (random search, the campaign's rung RS10 on the de-dup design WITH the window mask) had not been merged when this was built, so the tuned-tree table is left out rather than filled with the unmasked runs; the no-mask root carries the unmasked tuned trees' table (`results/feature_importance_1530_dedup_nomask/tuned_trees_series.csv`)."


# ============================================================================ gates
def _gate_numbers(out: Path, G: pd.DataFrame) -> dict:
    n: dict = {}
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
        }
    n["mask_effect"] = {
        (b, m): (
            float(gm.loc[(b, m), "forecast_gap_max"]),
            float(gm.loc[(b, m), "forecast_gap_mean"]),
            float(gm.loc[(b, m), "forecast_corr_stored"]),
        )
        for b in BUCKETS
        for m in TREES
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
    n["draws"] = bool((G["draws_equal_to_lgbm"] == G["refits"]).all())
    C = _read(out / "gates_cadence.csv")
    if C is not None:
        n["cadence"] = C
    Q = _read(out / "gates_canonical.csv")
    if Q is not None:
        n["canonical"] = Q
    D = _read(out / "gates_design.csv")
    if D is not None:
        n["design"] = D
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
            f"{BNAME[b]} {LABEL[m]} {me[(b, m)][0]:.3f}" for b in BUCKETS for m in TREES
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
                "- Canonical masked tree tables: `results/spxw_pnl/yhat_subtree_<model>_<bucket>` still equal the unmasked T10 runs bit for bit (9 of 9) when this was built -- the masked re-run of the campaign (agent H) had not replaced them, so the masked refits are gated against their own unmasked twins (above) and the design; `gates_canonical.csv` repeats the check once the masked tables land."
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
        + " of 147 refits, HAR + calendar / live-feasible / all features). The all_features lasso refits 75..124 (penalty 0.001) ran on the cluster (`cluster/slurm/submit_featimp_dedup.sh linear`), the rest locally; the parts partition the 147 refits (checked by the merge)."
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
        "\n\\section*{8. Gates}\n\\begin{itemize}\\itemsep1pt\n"
        + "\n".join("\\item " + _tex_inline(x) for x in items)
        + "\n\\end{itemize}\n"
    )


def _tex_inline(s: str) -> str:
    """Markdown bullet -> LaTeX: `code` -> texttt, escapes."""
    out, code = [], False
    for part in s.split("`"):
        out.append(("\\texttt{" + _esc(part) + "}") if code else _esc(part))
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
    sp, har, harc, mv, cl = X["sp"], X["har"], X["harc"], X["mv"], X["cl"]
    L = [
        "## Before / after: first pass (old design, trees unmasked) vs de-dup + mask",
        "Tables `before_after_ranks_<bucket>_<model>.csv` (every unit of every measure and level: value and rank old / new [/ de-dup no mask for the trees], the rank among the units common to both runs and its change), `before_after_spearman.csv`, `before_after_movers.csv`, `before_after_har_ma.csv`, `before_after_har_ma_columns.csv`, `before_after_clusters.csv` (clusters are renumbered per run; matched by their members once the session-edge columns are removed from the old cluster).",
        "",
    ]
    # headline: har_ma_* in the trees, and the ridge rows as the draw-noise yardstick
    for b in ("live_feasible", "all_features"):
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
        f"- The draw-noise yardstick: the ridge fits are the same problem in both runs, so its `har_ma_*` change is the Monte Carlo of fresh permutation draws alone: perm P2 dMSE moves by at most {d_mse.max():.1f} points of % tail loss over the three buckets (abs(beta x sd), SHAP and drop-column unchanged)."
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
    for b in ("live_feasible", "all_features", "baseline"):
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
                for b in ("live_feasible", "all_features")
                for m in TREES
            )
            + "."
        )
        L.append("")
    if mv is not None:
        L.append(
            "### Rows that moved most (series level, top-20 zone of either run, rank among common units old -> new)"
        )
        for b in ("live_feasible", "all_features"):
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


def before_after_tex(out: Path, root_rel: str) -> str:
    X = _ba(out)
    if X["sp"] is None or X["har"] is None:
        return "\n\\section*{9. Before / after}\nPending.\n"
    sp, har, harc = X["sp"], X["har"], X["harc"]
    body = [
        "\n\\section*{9. Before / after: first pass (old design) vs de-dup + mask}\n"
    ]
    body.append(
        "Every unit of every measure and level, old vs new: \\texttt{before\\_after\\_ranks\\_<bucket>\\_<model>.csv}; clusters are renumbered per run and matched by their members once the session-edge columns are removed from the old cluster. Spearman between the old and the new importance over the units common to both runs:\n"
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
    for b in ("live_feasible", "all_features", "baseline"):
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
    f = out / "fig_before_after_har_ma.png"
    if f.is_file():
        body.append(
            "\\begin{figure}[H]\\centering\\includegraphics[width=\\textwidth]{../"
            + root_rel
            + "/fig_before_after_har_ma.png}\n\\caption{\\texttt{har\\_ma\\_*} before (old design) and after (de-dup + mask; open marker: de-dup, no mask), per model and measure; left: shares, right: permutation P2 as \\% of the model's tail loss.}\\end{figure}\n"
        )
    f = out / "fig_before_after_ranks_live_feasible.png"
    if f.is_file():
        body.append(
            "\\begin{figure}[H]\\centering\\includegraphics[width=\\textwidth]{../"
            + root_rel
            + "/fig_before_after_ranks_live_feasible.png}\n\\caption{Live-feasible, series level: rank among the common series, old (x) vs new (y), per model and measure; on the diagonal = unchanged.}\\end{figure}\n"
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
