"""Before / after rankings of the C1 feature-importance re-run (checklist I7).

BEFORE = the first pass on the old per-bar design (results/feature_importance_1530/: the six
``har_ma_k_x_open`` columns all zero at 16:00, the six ``har_ma_k_x_close`` columns exact copies
of ``har_ma_k``; trees unmasked).  AFTER = the de-duplicated design with the per-window mask on
every tree fit (results/feature_importance_1530_dedup/).  The intermediate rung, de-dup without
the mask (results/feature_importance_1530_dedup_nomask/), is carried as an extra column for the
trees (the linear models are the same under both: their identifiability mask always applied).

Per bucket x model x measure x level (column / series / cluster), every unit old vs new:
value, stored rank (among the level's eligible units of its own run), rank among the units
common to both runs (eligible in both) and its change; Spearman between the old and new values
over the common units; the units that moved most.  Clusters (|corr| >= 0.8, complete linkage,
renumbered by each run) are matched by membership once the session-edge columns are removed
from the old cluster.  ``har_ma_*``: its series value old / new, and at the column level the old
total split into the plain ladder ``har_ma_k`` and its ``_x_close`` copies.

Called by ``python experiments/feature_importance_1530_dedup.py before_after`` (roots set there).
Writes results/feature_importance_1530_dedup/before_after_*.csv.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OLD = REPO / "results" / "feature_importance_1530"
NEW = REPO / "results" / "feature_importance_1530_dedup"
MID = REPO / "results" / "feature_importance_1530_dedup_nomask"
BUCKETS = ("baseline", "live_feasible", "all_features")
MODELS = ("ridge", "lasso", "lgbm", "xgb", "rf")
TREE_MODELS = ("lgbm", "xgb", "rf")
TREE_MEASURES = (
    "mdi",
    "split",
    "perm_p1_qlike",
    "perm_p2_qlike",
    "perm_p1_mse",
    "perm_p2_mse",
    "shap",
)
LIN_MEASURES = (
    "bsd",
    "perm_p1_qlike",
    "perm_p2_qlike",
    "perm_p1_mse",
    "perm_p2_mse",
    "drop_qlike",
    "drop_mse",
    "shap",
)
LEVELS = ("column", "series", "cluster")
HAR_LAGS = (1, 5, 25, 125, 625, 3125)
SESSION_EDGE = frozenset(
    f"har_ma_{k}_x_{g}" for g in ("open", "close") for k in HAR_LAGS
)
TOP_MOVE = 5  # movers listed per model x measure x level
TOP_ZONE = (
    20  # ... among the units ranked in the top 20 of either run (the part people read)
)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def _truthy(v: object) -> bool:
    """An eligibility flag after an outer merge (NaN = the unit is absent from that run)."""
    return bool(v) if v == v else False


def _rank_desc(v: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    return rankdata(-v, method="min").astype(int)


def _cluster_members(root: Path, bucket: str, names: list[str]) -> dict[str, frozenset]:
    """Cluster unit -> member columns (multi-column clusters from clusters.csv; a singleton
    cluster is its column)."""
    C = pd.read_csv(root / "clusters.csv")
    C = C[C["bucket"] == bucket]
    out = {r.cluster: frozenset(r.members.split()) for r in C.itertuples()}
    inside = set().union(*out.values()) if out else set()
    for c in names:
        if c not in inside:
            out[c] = frozenset([c])
    return out


def _names(root: Path, bucket: str) -> list[str]:
    U = pd.read_csv(root / "unique_values.csv")
    return U.loc[U["bucket"] == bucket, "column"].astype(str).tolist()


def _key_display(members: frozenset) -> str:
    m = sorted(members, key=lambda c: (not c.startswith("har_ma"), c))
    return m[0] if len(m) == 1 else f"{m[0]} +{len(m) - 1}"


def _table(
    root: Path, bucket: str, model: str, m: str, level: str
) -> pd.DataFrame | None:
    f = root / "by_measure" / f"imp_{bucket}_{model}_{m}.csv"
    if not f.is_file():
        return None
    t = pd.read_csv(f)
    return t[t["level"] == level].reset_index(drop=True)


def main() -> None:  # noqa: C901 - one comparison
    has_mid = (MID / "by_measure").is_dir()
    long_rows: dict[tuple[str, str], list[pd.DataFrame]] = {}
    sp_rows, mv_rows, har_rows, harc_rows, cl_rows = [], [], [], [], []
    for b in BUCKETS:
        n_old, n_new = _names(OLD, b), _names(NEW, b)
        cm_old = _cluster_members(OLD, b, n_old)
        cm_new = _cluster_members(NEW, b, n_new)
        # the no-mask root reads the mask root's inputs (same design, same clusters)
        cm_mid = _cluster_members(MID, b, n_new) if has_mid else {}
        assert not has_mid or cm_mid == cm_new, b
        # cluster match: old members minus the session-edge columns == new members
        new_by_key = {v: k for k, v in cm_new.items()}
        mid_by_key = {v: k for k, v in cm_mid.items()}
        cmap: dict[str, str] = {}
        for u, mem in cm_old.items():
            key = frozenset(mem - SESSION_EDGE)
            status = (
                "removed (session-edge only)"
                if not key
                else ("matched" if key in new_by_key else "regrouped")
            )
            if status == "matched":
                cmap[u] = new_by_key[key]
            cl_rows.append(
                dict(
                    bucket=b,
                    cluster_old=u,
                    members_old=" ".join(sorted(mem)),
                    session_edge_members=len(mem & SESSION_EDGE),
                    status=status,
                    cluster_new=new_by_key.get(key, ""),
                    cluster_nomask=mid_by_key.get(key, "") if has_mid else "",
                )
            )
        matched_new = set(cmap.values())
        for u, mem in cm_new.items():
            if u not in matched_new:
                cl_rows.append(
                    dict(
                        bucket=b,
                        cluster_old="",
                        members_old="",
                        session_edge_members=0,
                        status="new only",
                        cluster_new=u,
                        cluster_nomask=mid_by_key.get(mem, "") if has_mid else "",
                        members_new=" ".join(sorted(mem)),
                    )
                )
        for model in MODELS:
            meas = TREE_MEASURES if model in TREE_MODELS else LIN_MEASURES
            frames = []
            for m in meas:
                for level in LEVELS:
                    to = _table(OLD, b, model, m, level)
                    tn = _table(NEW, b, model, m, level)
                    tm = (
                        _table(MID, b, model, m, level)
                        if has_mid and model in TREE_MODELS
                        else None
                    )
                    if to is None or tn is None:
                        continue
                    # the key every run shares (a cluster: its first member + size in the
                    # new design; an unmatched cluster keeps a marked key of its own)
                    if level == "cluster":
                        to["key"] = [
                            _key_display(cm_new[cmap[u]])
                            if u in cmap
                            else f"(old) {_key_display(cm_old[u])}"
                            for u in to["unit"].astype(str)
                        ]
                        for t in (tn,) + ((tm,) if tm is not None else ()):
                            t["key"] = [
                                _key_display(cm_new[u])
                                if u in matched_new
                                else f"(new) {_key_display(cm_new[u])}"
                                for u in t["unit"].astype(str)
                            ]
                    else:
                        for t in (to, tn) + ((tm,) if tm is not None else ()):
                            t["key"] = t["unit"].astype(str)
                    cols = [
                        "key",
                        "unit",
                        "label",
                        "eligible",
                        "value",
                        "lo",
                        "hi",
                        "rank",
                    ]
                    extra = [c for c in ("pct_of_loss",) if c in to.columns]
                    M_ = to[cols + extra].merge(
                        tn[cols + extra],
                        on="key",
                        how="outer",
                        suffixes=("_old", "_new"),
                    )
                    if tm is not None:
                        tmm = tm[["key", "value", "rank", "eligible"] + extra].rename(
                            columns={
                                "value": "value_nomask",
                                "rank": "rank_nomask",
                                "eligible": "eligible_nomask",
                                **(
                                    {"pct_of_loss": "pct_of_loss_nomask"}
                                    if extra
                                    else {}
                                ),
                            }
                        )
                        M_ = M_.merge(tmm, on="key", how="left")
                    both = M_["eligible_old"].map(_truthy) & M_["eligible_new"].map(
                        _truthy
                    )
                    M_["status"] = np.where(
                        both,
                        "common",
                        np.where(
                            M_["value_new"].isna(),
                            "old only",
                            np.where(M_["value_old"].isna(), "new only", "ineligible"),
                        ),
                    )
                    M_["rank_old_common"] = 0
                    M_["rank_new_common"] = 0
                    if both.any():
                        M_.loc[both, "rank_old_common"] = _rank_desc(
                            M_.loc[both, "value_old"].to_numpy(float)
                        )
                        M_.loc[both, "rank_new_common"] = _rank_desc(
                            M_.loc[both, "value_new"].to_numpy(float)
                        )
                    M_["rank_change"] = np.where(
                        both, M_["rank_new_common"] - M_["rank_old_common"], 0
                    )
                    M_.insert(0, "level", level)
                    M_.insert(0, "measure", m)
                    M_.insert(0, "model", model)
                    M_.insert(0, "bucket", b)
                    frames.append(M_)
                    C_ = M_[both]
                    r = dict(
                        bucket=b,
                        model=model,
                        measure=m,
                        level=level,
                        n_eligible_old=int(M_["eligible_old"].map(_truthy).sum()),
                        n_eligible_new=int(M_["eligible_new"].map(_truthy).sum()),
                        n_common=int(both.sum()),
                        n_old_only=int((M_["status"] == "old only").sum()),
                        spearman_old_new=_spearman(
                            C_["value_old"].to_numpy(float),
                            C_["value_new"].to_numpy(float),
                        ),
                        leader_old=str(M_.loc[M_["rank_old"] == 1, "key"].iloc[0])
                        if (M_["rank_old"] == 1).any()
                        else "",
                        leader_new=str(M_.loc[M_["rank_new"] == 1, "key"].iloc[0])
                        if (M_["rank_new"] == 1).any()
                        else "",
                        median_abs_rank_change=float(C_["rank_change"].abs().median())
                        if len(C_)
                        else np.nan,
                        top5_overlap=int(
                            len(
                                set(C_.nsmallest(5, "rank_old_common")["key"])
                                & set(C_.nsmallest(5, "rank_new_common")["key"])
                            )
                        ),
                    )
                    if tm is not None:
                        bm = both & M_["eligible_nomask"].map(_truthy)
                        r["spearman_old_nomask"] = _spearman(
                            M_.loc[bm, "value_old"].to_numpy(float),
                            M_.loc[bm, "value_nomask"].to_numpy(float),
                        )
                        r["spearman_nomask_new"] = _spearman(
                            M_.loc[bm, "value_nomask"].to_numpy(float),
                            M_.loc[bm, "value_new"].to_numpy(float),
                        )
                    sp_rows.append(r)
                    zone = C_[
                        (C_["rank_old_common"] <= TOP_ZONE)
                        | (C_["rank_new_common"] <= TOP_ZONE)
                    ]
                    order = zone.assign(a=zone["rank_change"].abs()).sort_values(
                        ["a", "rank_old_common"], ascending=[False, True]
                    )
                    for q in order.head(TOP_MOVE).itertuples():
                        mv_rows.append(
                            dict(
                                bucket=b,
                                model=model,
                                measure=m,
                                level=level,
                                unit=q.key,
                                rank_old_common=int(q.rank_old_common),
                                rank_new_common=int(q.rank_new_common),
                                rank_change=int(q.rank_change),
                                value_old=q.value_old,
                                value_new=q.value_new,
                            )
                        )
                    # har_ma: the series, and the old column total split plain / x_close copies
                    if level == "series":
                        h = M_[M_["key"] == "har_ma"]
                        if len(h):
                            h = h.iloc[0]
                            har_rows.append(
                                dict(
                                    bucket=b,
                                    model=model,
                                    measure=m,
                                    value_old=h.value_old,
                                    value_new=h.value_new,
                                    value_nomask=h.get("value_nomask", np.nan),
                                    change=h.value_new - h.value_old,
                                    rank_old=int(h.rank_old),
                                    rank_new=int(h.rank_new),
                                    pct_of_loss_old=h.get("pct_of_loss_old", np.nan),
                                    pct_of_loss_new=h.get("pct_of_loss_new", np.nan),
                                    pct_of_loss_nomask=h.get(
                                        "pct_of_loss_nomask", np.nan
                                    ),
                                )
                            )
                    if level == "column":
                        vo = to.set_index("unit")["value"]
                        vn = tn.set_index("unit")["value"]
                        vm = tm.set_index("unit")["value"] if tm is not None else None
                        plain = [f"har_ma_{k}" for k in HAR_LAGS]
                        xclose = [f"har_ma_{k}_x_close" for k in HAR_LAGS]
                        xopen = [f"har_ma_{k}_x_open" for k in HAR_LAGS]
                        row: dict = dict(
                            bucket=b,
                            model=model,
                            measure=m,
                            old_plain_total=float(vo.reindex(plain).sum()),
                            old_x_close_total=float(vo.reindex(xclose).sum()),
                            old_x_open_total=float(vo.reindex(xopen).sum()),
                            new_plain_total=float(vn.reindex(plain).sum()),
                            nomask_plain_total=float(vm.reindex(plain).sum())
                            if vm is not None
                            else np.nan,
                        )
                        tot = row["old_plain_total"] + row["old_x_close_total"]
                        row["old_x_close_share_of_har_columns"] = (
                            row["old_x_close_total"] / tot if tot != 0 else np.nan
                        )
                        for k in HAR_LAGS:
                            row[f"old_har_ma_{k}"] = float(
                                vo.get(f"har_ma_{k}", np.nan)
                            )
                            row[f"old_har_ma_{k}_x_close"] = float(
                                vo.get(f"har_ma_{k}_x_close", np.nan)
                            )
                            row[f"new_har_ma_{k}"] = float(
                                vn.get(f"har_ma_{k}", np.nan)
                            )
                        harc_rows.append(row)
            if frames:
                long_rows[(b, model)] = frames
    NEW.mkdir(parents=True, exist_ok=True)
    for (b, model), frames in long_rows.items():
        pd.concat(frames, ignore_index=True).to_csv(
            NEW / f"before_after_ranks_{b}_{model}.csv",
            index=False,
            float_format="%.5g",
        )
    pd.DataFrame(sp_rows).to_csv(
        NEW / "before_after_spearman.csv", index=False, float_format="%.4f"
    )
    pd.DataFrame(mv_rows).to_csv(
        NEW / "before_after_movers.csv", index=False, float_format="%.5g"
    )
    pd.DataFrame(har_rows).to_csv(
        NEW / "before_after_har_ma.csv", index=False, float_format="%.6g"
    )
    pd.DataFrame(harc_rows).to_csv(
        NEW / "before_after_har_ma_columns.csv", index=False, float_format="%.6g"
    )
    pd.DataFrame(cl_rows).to_csv(NEW / "before_after_clusters.csv", index=False)
    S = pd.DataFrame(sp_rows)
    print(
        S.groupby(["level"])["spearman_old_new"].describe().to_string(),
        flush=True,
    )
    figures()


if __name__ == "__main__":
    main()


# ============================================================================ figures
# the first pass's validated categorical slots (one per model; validator: all checks pass on
# the light surface, contrast WARN -> every mark is named on its axis and the numbers are in
# the CSVs)
COLOR = {
    "ridge": "#2a78d6",
    "lasso": "#4a3aa7",
    "lgbm": "#eb6834",
    "xgb": "#1baf7a",
    "rf": "#eda100",
}
LABEL = {
    "ridge": "ridge",
    "lasso": "lasso",
    "lgbm": "LightGBM",
    "xgb": "XGBoost",
    "rf": "random forest",
}
SHORT = {
    "mdi": "MDI",
    "split": "split",
    "bsd": "|b sd|",
    "shap": "SHAP",
    "perm_p2_qlike": "perm P2 QLIKE",
    "perm_p2_mse": "perm P2 MSE",
    "drop_mse": "drop-column MSE",
}


def figures() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    H = pd.read_csv(NEW / "before_after_har_ma.csv")
    panels: tuple[tuple[str, dict[str, tuple[str, ...]]], ...] = (
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
        2, 2, figsize=(11, 8.2), gridspec_kw={"width_ratios": [1.25, 1]}
    )
    for r, b in enumerate(("live_feasible", "all_features")):
        for c, (xlab, ms) in enumerate(panels):
            ax = axes[r, c]
            rows = []
            for model in MODELS:
                fam = "tree" if model in TREE_MODELS else "linear"
                for m in ms[fam]:
                    h = H[(H.bucket == b) & (H.model == model) & (H.measure == m)]
                    if len(h):
                        rows.append((model, m, h.iloc[0]))
            y = np.arange(len(rows))[::-1]
            for yy, (model, m, h) in zip(y, rows):
                share = m in ("mdi", "split", "bsd", "shap")
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
                        edgecolor=COLOR[model],
                        lw=1.4,
                        zorder=3,
                    )
                ax.scatter(
                    [vn],
                    [yy],
                    s=46,
                    color=COLOR[model],
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
                [f"{LABEL[mo]}: {SHORT[m]}" for mo, m, _ in rows], fontsize=7.5
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
                ax.set_title(f"{b}: har_ma_* before -> after", fontsize=9, loc="left")
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
    fig.savefig(NEW / "fig_before_after_har_ma.png", dpi=150)
    plt.close(fig)

    # rank scatter, live_feasible series: old vs new rank among the common series
    b = "live_feasible"
    heads = {
        "tree": ("mdi", "split", "perm_p2_qlike", "shap"),
        "linear": ("bsd", "perm_p2_mse", "drop_mse", "shap"),
    }
    fig, axes = plt.subplots(len(MODELS), 4, figsize=(10.5, 12.5))
    for r, model in enumerate(MODELS):
        T = pd.read_csv(NEW / f"before_after_ranks_{b}_{model}.csv")
        fam = "tree" if model in TREE_MODELS else "linear"
        for c, m in enumerate(heads[fam]):
            ax = axes[r, c]
            t = T[(T.measure == m) & (T.level == "series") & (T.status == "common")]
            n = len(t)
            ax.plot([1, n], [1, n], color="0.8", lw=0.8, zorder=1)
            ax.scatter(
                t.rank_old_common,
                t.rank_new_common,
                s=14,
                color=COLOR[model],
                lw=0,
                zorder=2,
            )
            h = t[t.key == "har_ma"]
            if len(h):
                ax.scatter(
                    h.rank_old_common,
                    h.rank_new_common,
                    s=40,
                    facecolor="none",
                    edgecolor="0.15",
                    lw=1.0,
                    zorder=3,
                )
            rho = _spearman(t.value_old.to_numpy(float), t.value_new.to_numpy(float))
            ax.set_title(
                f"{LABEL[model]}: {SHORT.get(m, m)}  (Spearman {rho:.2f})", fontsize=7.5
            )
            ax.set_xlim(0, n + 1)
            ax.set_ylim(n + 1, 0)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.tick_params(labelsize=6)
            for sp_ in ("top", "right"):
                ax.spines[sp_].set_visible(False)
            if r == len(MODELS) - 1:
                ax.set_xlabel("rank, first pass", fontsize=7)
            if c == 0:
                ax.set_ylabel("rank, de-dup + mask", fontsize=7)
    fig.suptitle(
        "live_feasible, series level: rank among the common series, before (x) vs after (y); ring = har_ma_*",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(NEW / "fig_before_after_ranks_live_feasible.png", dpi=140)
    plt.close(fig)
