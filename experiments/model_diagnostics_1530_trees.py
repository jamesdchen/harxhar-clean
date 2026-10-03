"""Feature importance of the 16:00-bar forecast: tree models against the linear ones.

Same bar (ending 16:00 = the forecast the 15:30 trade uses), same 2000-session
window, same design columns, same 1,469 forecast rows.  For every model the
importance of an input SERIES is its share of mean |SHAP|:

  SHAP of series g on a row  = the sum of the SHAP values of g's columns (all lags,
                               the is-present / is-nonzero flags, the open/close
                               interactions of the HAR ladder)
  share of g                 = mean over rows of |SHAP of g| / sum over series

Linear (per-bar ridge, per-bar lasso): SHAP = beta_j (x_j - window mean_j) from the
coefficient captures of ``experiments/model_diagnostics_1530.py capture`` (the
identity with the shap library is checked in ``model_diagnostics_1530_regimes.py``);
base value = the forecast at the window mean.
Trees (LightGBM, XGBoost, random forest): the per-row TreeSHAP values the tree
campaign saved (``specs/causal_tune_trees.py``, one model per refit every 10
sessions); base value = the explainer's expected value (last column).

Gates (printed and written to trees_shap.json):
  1. every model's SHAP row sums to its prediction minus its base value;
  2. the tree npz forecasts equal the tree results CSV; the linear captures equal
     the stored research forecasts (checked by the capture's own analysis);
  3. every model covers the same 1,469 stamps;
  4. the linear shares reproduce the ridge / lasso deck-day shares already in
     regimes.json (live_feasible).

Outputs (results/model_diagnostics_1530/):
  trees_shap_share_<bucket>.csv       per series x model: share (all rows, deck days),
                                      rank, native tree importance share
  trees_shap_spearman_<bucket>.csv    Spearman rank correlation of the share vectors,
                                      every pair of models (all rows, deck days)
  trees_shap_rolling63_<bucket>.csv   trailing-63-session shares, every model
  trees_shap_share.png, trees_shap_rolling.png, trees_shap.json, trees_shap.log

Run:  python experiments/model_diagnostics_1530_trees.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO / "notebooks"))
import model_diagnostics_1530 as md  # noqa: E402

OUT = md.OUT
TREES = REPO / "results" / "linear_subsection_trees"
STORED = (
    REPO / "results" / "linear_subsection" / "arms_hoffman2"
)  # stored linear forecasts
STORED_EST = {"ridge": "ridge", "lasso": "reclasso"}
SEG = "bar1600"
TW = 2000
ROLL = 63  # a quarter of sessions, the window of the rolling-SHAP page
BUCKETS = ("all_features",)  # live_feasible commented out
# model key -> (label, capture tag or tree dir); order = the fixed colour order
LINEAR: dict[str, dict[str, str | None]] = {
    "live_feasible": {"ridge": None, "lasso": "reclasso"},
    "all_features": {"ridge": "all_features", "lasso": "all_features_reclasso"},
}
TREE_MODELS = {"lgbm": "LightGBM", "xgb": "XGBoost", "rf": "random forest"}
LABEL = {"ridge": "ridge", "lasso": "lasso", **TREE_MODELS}
ORDER = ("ridge", "lasso", "lgbm", "xgb", "rf")
# categorical slots validated for adjacent-pair separation (dataviz validator):
# blue, violet, orange, aqua, yellow; lines add a dash pattern (trees dashed)
COLOR = {
    "ridge": "#2a78d6",
    "lasso": "#4a3aa7",
    "lgbm": "#eb6834",
    "xgb": "#1baf7a",
    "rf": "#eda100",
}
DASH = {
    "ridge": "-",
    "lasso": (0, (1, 1)),
    "lgbm": (0, (5, 2)),
    "xgb": (0, (3, 1, 1, 1)),
    "rf": (0, (8, 3)),
}
N_FIG = 12  # series drawn individually in the bar figure; the rest are summed
# series that measure the SIZE of recent returns (realized variance, absolute, 4th-power,
# bipower, upside squared): correlated inputs a linear fit can trade off against
# each other; their SHAP summed before |.| is the family's share
SIZE_FAMILY = ("har_ma", "sumabsret", "sumret4", "sumbipow", "sumpret2")


def deck_days() -> pd.DatetimeIndex:
    d = pd.read_parquet(
        REPO / "results" / "atm_straddle_0dte_1530" / "daily_sub_live_ridge.parquet"
    )
    return pd.DatetimeIndex(d.index).normalize().as_unit("ns")


def load_linear(tag: str | None, bucket: str, key: str) -> dict | None:
    f = OUT / ("capture_bar1600.npz" if tag is None else f"capture_bar1600_{tag}.npz")
    if not f.is_file():
        return None
    z = np.load(f, allow_pickle=True)
    th, X, mu = z["th"], z["x"], z["mu"]
    phi = th[:, :-1] * (X - mu)
    base = th[:, -1] + np.einsum("nk,nk->n", th[:, :-1], mu)
    date = np.asarray(z["date"]).astype(str)
    pred = np.asarray(z["pred"], float)
    # the capture against the stored research forecast of the same arm
    stored = pd.read_csv(
        STORED / bucket / STORED_EST[key] / f"tw{TW}" / f"results_{SEG}.csv"
    )
    s = stored.set_index(stored["date"].astype(str))["pred_adj"]
    assert s.index.is_unique and set(date) == set(s.index), (f, len(s), len(date))
    sv = s.reindex(date).to_numpy(float)
    return dict(
        names=[str(v) for v in z["names"]],
        date=date,
        phi=phi,
        base=base,
        pred=pred,
        importance=None,
        stored_rel=float(np.max(np.abs(pred - sv) / np.abs(sv))),
        src=f.name,
    )


def load_tree(bucket: str, model: str) -> dict | None:
    d = TREES / bucket / SEG / model / f"tw{TW}" / "causal_tune_trees" / model / bucket
    f = d / f"trees_{SEG}.npz"
    if not f.is_file():
        return None
    z = np.load(f, allow_pickle=True)
    sh = np.asarray(z["shap"], dtype=np.float64)
    names = [str(v) for v in z["feature_names"]]
    assert sh.shape[1] == len(names) + 1, (f, sh.shape, len(names))
    csv = pd.read_csv(d / f"results_{SEG}.csv")
    date = np.asarray(z["date"]).astype(str)
    assert (csv["date"].astype(str).to_numpy() == date).all(), f
    pred = np.asarray(z["pred_adj"], float)
    csv_gap = float(np.max(np.abs(pred - csv["pred_adj"].to_numpy(float))))
    imp = np.asarray(z["importance"], dtype=np.float64)
    return dict(
        names=names,
        date=date,
        phi=sh[:, :-1],
        base=sh[:, -1],
        pred=pred,
        importance=imp.mean(axis=0),
        csv_gap=csv_gap,
        src=str(f.relative_to(REPO)),
    )


def series_of(names: list[str]) -> tuple[np.ndarray, dict[str, str]]:
    grp = np.array([md.parse_feature(x)[0] for x in names])
    stem = {g: md.var_stem(g, names) for g in sorted(set(grp))}
    stem["calendar"] = "calendar (DOW_*, is_*, hour, days_to_opex)"
    return grp, stem


def main() -> None:  # noqa: C901 - one linear report
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lines: list[str] = []

    def say(msg: str) -> None:
        print(msg, flush=True)
        lines.append(msg)

    deck = deck_days()
    J: dict = {"gates": {}, "buckets": {}}
    shares_all: dict[str, pd.DataFrame] = {}
    rolling_all: dict[str, pd.DataFrame] = {}
    stems_all: dict[str, dict[str, str]] = {}

    for bucket in BUCKETS:
        M: dict[str, dict] = {}
        for key, tag in LINEAR[bucket].items():
            o = load_linear(tag, bucket, key)
            if o is None:
                say(f"note: no {bucket} {key} capture; {key} left out of {bucket}")
            else:
                M[key] = o
        for key in TREE_MODELS:
            o = load_tree(bucket, key)
            if o is None:
                say(f"note: no {bucket} {key} npz")
            else:
                M[key] = o
        keys = [k for k in ORDER if k in M]
        if not any(k in TREE_MODELS for k in keys):
            continue
        # gate 3: same names, same stamps
        ref = M[keys[0]]
        for k in keys[1:]:
            assert M[k]["names"] == ref["names"], (bucket, k)
            assert (M[k]["date"] == ref["date"]).all(), (bucket, k)
        stamp = pd.DatetimeIndex(pd.to_datetime(ref["date"])).as_unit("ns")
        assert ((stamp.hour == 16) & (stamp.minute == 0)).all()
        in_deck = np.asarray(stamp.normalize().isin(deck))
        say(
            f"[{bucket}] {len(keys)} models ({', '.join(LABEL[k] for k in keys)}); "
            f"{len(ref['names'])} design columns; {len(stamp):,} rows "
            f"({stamp[0].date()} .. {stamp[-1].date()}), {int(in_deck.sum())} deck days; "
            "every model on the same stamps and columns"
        )
        # gate 1: additivity; gate 2 (trees): npz == CSV
        g1 = {}
        for k in keys:
            o = M[k]
            gap = np.abs(o["phi"].sum(axis=1) + o["base"] - o["pred"])
            g1[k] = dict(
                max_abs=float(gap.max()),
                max_rel=float((gap / np.abs(o["pred"])).max()),
                base_mean=float(o["base"].mean()),
                pred_mean=float(o["pred"].mean()),
                csv_gap=o.get("csv_gap"),
                stored_rel=o.get("stored_rel"),
                src=o["src"],
            )
            say(
                f"  GATE {LABEL[k]}: max |sum SHAP + base - forecast| {gap.max():.1e} "
                f"(relative {g1[k]['max_rel']:.1e}); base mean {o['base'].mean():.4f}, "
                f"forecast mean {o['pred'].mean():.4f}"
                + (
                    f"; npz forecast vs results CSV {o['csv_gap']:.1e}"
                    if o.get("csv_gap") is not None
                    else ""
                )
            )
            assert g1[k]["max_rel"] < 1e-5, (bucket, k, g1[k])
            if o.get("csv_gap") is not None:
                assert o["csv_gap"] < 1e-12, (bucket, k, o["csv_gap"])
            if o.get("stored_rel") is not None:
                say(
                    f"  GATE {LABEL[k]}: captured forecast vs the stored research forecast, "
                    f"{len(o['date']):,} rows: max relative difference {o['stored_rel']:.1e}"
                )
                assert o["stored_rel"] < 1e-6, (bucket, k, o["stored_rel"])
        J["gates"][bucket] = g1

        grp, stem = series_of(ref["names"])
        gset = sorted(set(grp))
        stems_all[bucket] = stem
        S: dict[str, pd.DataFrame] = {}  # |series SHAP| per row
        rows = []
        for k in keys:
            o = M[k]
            A = pd.DataFrame(
                {g: np.abs(o["phi"][:, grp == g].sum(axis=1)) for g in gset},
                index=stamp,
            )
            S[k] = A
            for smp, m in (("all rows", np.ones(len(A), bool)), ("deck days", in_deck)):
                mm = A[m].mean()
                sh = mm / mm.sum()
                rk = sh.rank(ascending=False, method="min").astype(int)
                imp = None
                if o["importance"] is not None:
                    ii = pd.Series(o["importance"], index=grp).groupby(level=0).sum()
                    imp = ii / ii.sum()
                for g in gset:
                    rows.append(
                        dict(
                            bucket=bucket,
                            sample=smp,
                            model=k,
                            series=g,
                            columns=stem[g],
                            n_columns=int((grp == g).sum()),
                            mean_abs_shap=float(mm[g]),
                            share=float(sh[g]),
                            rank=int(rk[g]),
                            native_importance_share=float(imp[g])
                            if imp is not None
                            else np.nan,
                        )
                    )
        T = pd.DataFrame(rows)
        T.to_csv(OUT / f"trees_shap_share_{bucket}.csv", index=False)
        shares_all[bucket] = T

        # the return-size family as one series
        fam = np.isin(grp, SIZE_FAMILY)
        # how correlated the family's level columns are with realized variance, same lag
        zc = np.load(OUT / "capture_bar1600.npz", allow_pickle=True)
        names_ = ref["names"]
        xr = None
        if bucket == "live_feasible":
            xr = np.asarray(zc["x"], float)
            assert [str(v) for v in zc["names"]] == names_
        size_corr = []
        if xr is not None:
            for lag in md.LAGS:
                h = xr[:, names_.index(f"har_ma_{lag}")]
                for g in SIZE_FAMILY[1:]:
                    c = f"adj_{g}_ma_{lag}"
                    size_corr.append(
                        float(np.corrcoef(h, xr[:, names_.index(c)])[0, 1])
                    )
            say(
                "  correlation of each return-size column with har_ma_<k> at the same lag k "
                f"({len(size_corr)} pairs, the {len(stamp):,} rows): "
                f"{min(size_corr):.2f} .. {max(size_corr):.2f}"
            )
        other = [g for g in gset if g not in SIZE_FAMILY]
        size_share = {}
        for k in keys:
            o = M[k]
            fa = np.abs(o["phi"][:, fam].sum(axis=1))
            num = {"all rows": fa.mean(), "deck days": fa[in_deck].mean()}
            den = {
                "all rows": num["all rows"] + S[k][other].mean().sum(),
                "deck days": num["deck days"] + S[k][other][in_deck].mean().sum(),
            }
            size_share[LABEL[k]] = {s: float(num[s] / den[s]) for s in num}
        say(
            "  return-size family ("
            + ", ".join(stem[g] for g in SIZE_FAMILY if g in stem)
            + ") as one series, share all rows / deck days: "
            + "; ".join(
                f"{m} {v['all rows']:.1%} / {v['deck days']:.1%}"
                for m, v in size_share.items()
            )
        )
        # the trees' own importance (split gain / impurity) against their SHAP shares
        nat = {}
        for k in keys:
            if M[k]["importance"] is None:
                continue
            t_ = T[(T["sample"] == "all rows") & (T["model"] == k)]
            r = stats.spearmanr(t_["share"], t_["native_importance_share"])
            lead_nat = t_.loc[t_["native_importance_share"].idxmax(), "series"]
            nat[LABEL[k]] = dict(
                spearman=float(r.statistic),
                leader=stem[lead_nat],
                leader_share=float(t_["native_importance_share"].max()),
            )
            say(
                f"  {LABEL[k]} native importance vs SHAP share: Spearman {r.statistic:.3f}; "
                f"native leader {stem[lead_nat]} ({t_['native_importance_share'].max():.1%})"
            )

        # gate 4: the linear shares reproduce regimes.json (live_feasible deck days)
        if bucket == "live_feasible" and (OUT / "regimes.json").is_file():
            R = json.loads((OUT / "regimes.json").read_text(encoding="utf-8"))
            g4 = {}
            for k, jv in (
                ("ridge", R.get("ridge_share_deck", {})),
                ("lasso", R.get("lasso", {}).get("share", {})),
            ):
                if k not in keys or not jv:
                    continue
                mine = T[(T["sample"] == "deck days") & (T["model"] == k)].set_index(
                    "series"
                )
                d_ = max(
                    abs(
                        float(mine.loc[g, "share"])
                        - float(jv[md.var_stem(g, ref["names"])])
                    )
                    for g in gset
                )
                g4[k] = d_
                say(
                    f"  GATE {k} deck-day shares vs regimes.json: max difference {d_:.1e}"
                )
                assert d_ < 1e-12, (k, d_)
            J["gates"]["regimes_json"] = g4

        # rank agreement
        sp = []
        for smp in ("all rows", "deck days"):
            W = T[T["sample"] == smp].pivot(
                index="series", columns="model", values="share"
            )
            for i, a in enumerate(keys):
                for b in keys[i + 1 :]:
                    r = stats.spearmanr(W[a], W[b])
                    ta = set(W[a].nlargest(5).index)
                    tb = set(W[b].nlargest(5).index)
                    sp.append(
                        dict(
                            bucket=bucket,
                            sample=smp,
                            a=a,
                            b=b,
                            n_series=len(W),
                            spearman=float(r.statistic),
                            pval=float(r.pvalue),
                            top5_overlap=len(ta & tb),
                            same_leader=bool(W[a].idxmax() == W[b].idxmax()),
                        )
                    )
        SP = pd.DataFrame(sp)
        SP.to_csv(OUT / f"trees_shap_spearman_{bucket}.csv", index=False)

        # rolling shares (every model is one forecast per session; SHAP per row)
        rl = []
        for k in keys:
            rm = S[k].rolling(ROLL, min_periods=ROLL).mean()
            sh = rm.div(rm.sum(axis=1), axis=0)
            rl.append(sh.assign(model=k))
        RL = pd.concat(rl)
        RL.index.name = "stamp"
        RL.to_csv(OUT / f"trees_shap_rolling{ROLL}_{bucket}.csv")
        rolling_all[bucket] = RL

        # report
        W_all = T[T["sample"] == "all rows"].pivot(
            index="series", columns="model", values="share"
        )[keys]
        W_dk = T[T["sample"] == "deck days"].pivot(
            index="series", columns="model", values="share"
        )[keys]
        order = W_all.mean(axis=1).sort_values(ascending=False).index
        say(f"  share of mean |SHAP| by series, all {len(stamp):,} rows (%):")
        say(
            (100 * W_all.loc[order].rename(index=stem).rename(columns=LABEL))
            .round(1)
            .to_string()
        )
        say(f"  deck days ({int(in_deck.sum())}) (%):")
        say(
            (100 * W_dk.loc[order].rename(index=stem).rename(columns=LABEL))
            .round(1)
            .head(12)
            .to_string()
        )
        say("  Spearman rank correlation of the share vectors:")
        say(SP.round(3).to_string(index=False))
        lead = {}
        for k in keys:
            sh = RL[RL["model"] == k].drop(columns="model").dropna()
            vc = sh.idxmax(axis=1).value_counts()
            lead[k] = {stem[g]: int(c) for g, c in vc.items()}
            say(
                f"  rolling {ROLL}: leading series for {LABEL[k]}: "
                + ", ".join(f"{stem[g]} {c / vc.sum():.0%}" for g, c in vc.items())
            )
        # the lasso's realized-variance share before / after its penalty re-choice
        lasso_split: dict[str, float | str] | None = None
        if (
            "lasso" in keys
            and bucket == "live_feasible"
            and (OUT / "regimes.json").is_file()
        ):
            Rj = json.loads((OUT / "regimes.json").read_text(encoding="utf-8"))
            sw = pd.Timestamp(Rj["lasso"]["switches"][0][0])
            A = S["lasso"]
            lasso_split = {}
            for part, m in (("before", A.index < sw), ("after", A.index >= sw)):
                mm = A[m].mean()
                lasso_split[part] = float(mm["har_ma"] / mm.sum())
            lasso_split["switch"] = str(sw.date())
            say(
                f"  lasso har_ma_* share before / after its penalty re-choice {sw.date()}: "
                f"{lasso_split['before']:.1%} / {lasso_split['after']:.1%}"
            )
        # the tree-vs-linear gap in the realized-variance share over time
        har = {k: RL[RL["model"] == k]["har_ma"].dropna() for k in keys}
        J["buckets"][bucket] = dict(
            models=keys,
            n=int(len(stamp)),
            deck_n=int(in_deck.sum()),
            n_columns=len(ref["names"]),
            n_series=len(gset),
            order=[stem[g] for g in order],
            share_all={
                LABEL[k]: {stem[g]: float(W_all.loc[g, k]) for g in gset} for k in keys
            },
            share_deck={
                LABEL[k]: {stem[g]: float(W_dk.loc[g, k]) for g in gset} for k in keys
            },
            rolling_leader=lead,
            rolling_har_range={
                LABEL[k]: [float(v.min()), float(v.max())] for k, v in har.items()
            },
            lasso_split=lasso_split,
            size_family=[stem[g] for g in SIZE_FAMILY if g in stem],
            size_corr=[min(size_corr), max(size_corr)] if size_corr else None,
            size_share=size_share,
            native=nat,
            spearman=SP.to_dict("records"),
        )

    # ---- figure 1: share by series, one bar group per model
    buckets = [b for b in BUCKETS if b in shares_all]
    fig, axes = plt.subplots(
        len(buckets), 1, figsize=(11, 3.3 * len(buckets) + 0.4), squeeze=False
    )
    for ax, bucket in zip(axes[:, 0], buckets):
        T = shares_all[bucket]
        keys = [k for k in ORDER if k in set(T["model"])]
        W = T[T["sample"] == "all rows"].pivot(
            index="series", columns="model", values="share"
        )[keys]
        order = list(W.mean(axis=1).sort_values(ascending=False).index)
        show = order[:N_FIG]
        rest = order[N_FIG:]
        Wp = W.loc[show]
        if rest:
            Wp = pd.concat(
                [Wp, W.loc[rest].sum().to_frame(f"other {len(rest)} series").T]
            )
        stem = stems_all[bucket]
        labels = ["calendar" if s == "calendar" else stem.get(s, s) for s in Wp.index]
        x = np.arange(len(Wp))
        wbar = 0.8 / len(keys)
        for i, k in enumerate(keys):
            ax.bar(
                x + (i - (len(keys) - 1) / 2) * wbar,
                100 * Wp[k].to_numpy(),
                width=wbar * 0.9,
                color=COLOR[k],
                label=LABEL[k],
                edgecolor="white",
                linewidth=0.4,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
        ax.set_ylabel("share of mean |SHAP| (%)", fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", color="0.9", lw=0.6)
        ax.set_axisbelow(True)
        for sp_ in ("top", "right"):
            ax.spines[sp_].set_visible(False)
        n_cols = J["buckets"][bucket]["n_columns"]
        ax.set_title(
            f"{bucket} design ({n_cols} columns, {J['buckets'][bucket]['n_series']} series), "
            f"16:00 bar, all {J['buckets'][bucket]['n']:,} forecasts",
            fontsize=8,
        )
        ax.legend(fontsize=7, ncol=len(keys), frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "trees_shap_share.png", dpi=150)
    plt.close(fig)

    # ---- figure 2: rolling shares of the four largest live_feasible series, every model
    bucket = "live_feasible"
    RL = rolling_all[bucket]
    T = shares_all[bucket]
    keys = [k for k in ORDER if k in set(T["model"])]
    W = T[T["sample"] == "all rows"].pivot(
        index="series", columns="model", values="share"
    )
    top4 = list(W.mean(axis=1).sort_values(ascending=False).index[:4])
    stem = stems_all[bucket]
    d0, d1 = deck.min(), deck.max()
    fig, axes = plt.subplots(2, 2, figsize=(11, 5.0), sharex=True)
    for ax, g in zip(axes.ravel(), top4):
        ax.axvspan(d0, d1, color="0.94", zorder=0, lw=0)
        for k in keys:
            s = RL[RL["model"] == k][g]
            ax.plot(
                s.index,
                100 * s.to_numpy(),
                color=COLOR[k],
                ls=DASH[k],
                lw=1.1,
                label=LABEL[k],
            )
        ax.set_title(stem[g], fontsize=8)
        ax.set_ylabel(f"share, trailing {ROLL} sessions (%)", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", color="0.9", lw=0.6)
        for sp_ in ("top", "right"):
            ax.spines[sp_].set_visible(False)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=len(keys), fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "trees_shap_rolling.png", dpi=150)
    plt.close(fig)
    J["rolling_top4"] = [stem[g] for g in top4]
    J["roll"] = ROLL

    (OUT / "trees_shap.json").write_text(
        json.dumps(J, indent=1, default=str), encoding="utf-8"
    )
    (OUT / "trees_shap.log").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
