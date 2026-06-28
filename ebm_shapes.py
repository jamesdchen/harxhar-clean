"""IN-SAMPLE PROBE (NOT a forecast): regime-EBM SHAPE interpretation.

Fits ONE ExplainableBoostingRegressor on the h16-19 close/AH leftover that remains
AFTER the Hero-A d8 global stage, to make the regime correction's MECHANISM legible.

This is a single GLOBAL in-sample fit over ALL OOS h16-19 rows at once. It is NOT
walk-forward and NOT an out-of-sample forecast -- it is a structural probe. It
mirrors the per-block regime EBM in resid_amortized.preds_chunk(arm='resid_regime')
(same EBM cfg, same h16-19 gate, same adj-space leftover target r2 = y_true -
pred_adj, same survivor-mask column set -- here the masks UNION across blocks as the
faithful global analogue of the per-block survivor masks) but fit ONCE so
explain_global() yields a single coherent set of shape functions.

Outputs results/ebm_shapes.json: top main-effect shapes (bin edges + per-bin scores,
named) and top pairwise interactions, each with an auto-described functional form.
"""

import glob
import json
import sys
from typing import Any

import numpy as np
import pandas as pd

from resid_amortized import _tree_factory, load_cache

# CELL / SUB / OUT overridable via argv so the same probe runs on any dug cell (e.g. the
# signature cells): argv = CELL [SUB [OUT]]. Defaults = the original enetreg2 Hero-A probe.
CELL = (
    sys.argv[1] if len(sys.argv) > 1 else "xgb_all_buckets_tw1000_enetreg2_rf480_slim"
)
SUB = sys.argv[2] if len(sys.argv) > 2 else "resid_subset_heroA_d8"
CFG = dict(
    learning_rate=0.02,
    max_leaves=3,
    interactions=10,
    max_bins=256,
    min_samples_leaf=10,
    max_rounds=500,
    outer_bags=4,
)
N_TOP_MAIN = 10
N_TOP_INT = 10
OUT = sys.argv[3] if len(sys.argv) > 3 else "results/ebm_shapes.json"


def _json_default(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(repr(type(o)))


def _load_d8_leftover(cache: dict) -> tuple[np.ndarray, np.ndarray]:
    """Concat d8 chunk preds -> adj-space leftover r2 = y_true - pred_adj, indexed by k."""
    files = sorted(glob.glob(f"results/resid_ab/{CELL}/{SUB}/chunk_*.csv"))
    df = (
        pd.concat([pd.read_csv(f) for f in files]).drop_duplicates("k").sort_values("k")
    )
    tw = int(cache["cell"]["train_win"])
    n_oos = int(cache["cell"]["n"]) - tw
    assert len(df) == n_oos, f"d8 chunks incomplete {len(df)}/{n_oos}"
    k = df["k"].to_numpy()
    assert np.array_equal(k, np.arange(n_oos)), "k not contiguous 0..n_oos-1"
    r2 = df["y_true"].to_numpy() - df["pred_adj"].to_numpy()
    return k, r2


def _describe_main(scores: np.ndarray) -> dict:
    """Auto-describe a univariate shape from its ordered per-bin scores."""
    s = np.asarray(scores, dtype=float)
    if s.size < 2:
        return {"form": "flat/degenerate"}
    idx = np.arange(s.size)
    rs = pd.Series(s).rank().to_numpy()
    ri = pd.Series(idx).rank().to_numpy()
    spearman = (
        float(np.corrcoef(rs, ri)[0, 1]) if rs.std() > 0 and ri.std() > 0 else 0.0
    )
    d = np.abs(np.diff(s))
    half = max(1, s.size // 2)
    frac_first = float(d[:half].sum() / (d.sum() + 1e-12)) if d.sum() > 0 else 0.0
    if abs(spearman) > 0.8:
        form = "monotone_up" if spearman > 0 else "monotone_down"
    elif frac_first > 0.7:
        form = "saturating (most change at LOW end of feature)"
    elif frac_first < 0.3:
        form = "threshold/late (most change at HIGH end of feature)"
    else:
        form = "non-monotone"
    return {
        "form": form,
        "spearman_score_vs_bin": round(spearman, 3),
        "score_range": float(s.max() - s.min()),
        "score_min": float(s.min()),
        "score_max": float(s.max()),
        "sign_low_end": "neg" if s[0] < 0 else "pos",
        "sign_high_end": "neg" if s[-1] < 0 else "pos",
        "score_at_low": float(s[0]),
        "score_at_high": float(s[-1]),
    }


def _describe_interaction(scores2d: np.ndarray) -> dict:
    """Auto-describe a pairwise surface: sign pattern at the 4 corners + range."""
    z = np.asarray(scores2d, dtype=float)
    if z.ndim != 2 or z.size == 0:
        return {"form": "degenerate"}
    corners = {
        "lo_lo": float(z[0, 0]),
        "lo_hi": float(z[0, -1]),
        "hi_lo": float(z[-1, 0]),
        "hi_hi": float(z[-1, -1]),
    }
    return {
        "score_range": float(z.max() - z.min()),
        "score_min": float(z.min()),
        "score_max": float(z.max()),
        "corners_left_by_right": corners,
    }


def main() -> None:
    cache = load_cache(CELL)
    tw = int(cache["cell"]["train_win"])
    Xs = cache["Xs"]
    feats = cache["feats"]
    masks = cache["masks"]
    hour_idx = feats.index("hour")

    k, r2 = _load_d8_leftover(cache)
    rows = tw + k
    hour = np.asarray(Xs[rows, hour_idx])
    sel = (hour >= 16) & (hour <= 19)
    rows_h = rows[sel]
    y_target = r2[sel]

    # masks UNION across blocks: faithful global analogue of resid_regime's per-block survivors.
    col_mask = masks.any(axis=0)
    cols = np.where(col_mask)[0]
    col_names = [feats[int(j)] for j in cols]
    X = pd.DataFrame(np.asarray(Xs[rows_h][:, cols]), columns=col_names)

    print(
        f"PROBE rows_h16-19={X.shape[0]} cols(union survivors)={X.shape[1]} fitting EBM...",
        flush=True,
    )
    ebm = _tree_factory("ebm", CFG)()
    ebm.fit(X, y_target)

    pred = np.asarray(ebm.predict(X))
    ss_res = float(np.sum((y_target - pred) ** 2))
    ss_tot = float(np.sum((y_target - y_target.mean()) ** 2))
    insample_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    print(
        f"PROBE in-sample R2 of EBM on h16-19 leftover = {insample_r2:.4f}", flush=True
    )

    g = ebm.explain_global()
    importances = np.asarray(ebm.term_importances(), dtype=float)
    term_feats = list(ebm.term_features_)
    term_names = list(ebm.term_names_)

    main_terms = [i for i, tf in enumerate(term_feats) if len(tf) == 1]
    int_terms = [i for i, tf in enumerate(term_feats) if len(tf) == 2]
    main_sorted = sorted(main_terms, key=lambda i: importances[i], reverse=True)[
        :N_TOP_MAIN
    ]
    int_sorted = sorted(int_terms, key=lambda i: importances[i], reverse=True)[
        :N_TOP_INT
    ]

    main_out = []
    for i in main_sorted:
        data = g.data(i)
        fname = col_names[int(term_feats[i][0])]
        scores = np.asarray(data["scores"], dtype=float)
        rec = {
            "feature": fname,
            "term_name": term_names[i],
            "importance": float(importances[i]),
            "bin_edges": data.get("names"),
            "scores": scores,
            "lower_bounds": data.get("lower_bounds"),
            "upper_bounds": data.get("upper_bounds"),
            "describe": _describe_main(scores),
        }
        main_out.append(rec)

    int_out = []
    for i in int_sorted:
        data = g.data(i)
        f0 = col_names[int(term_feats[i][0])]
        f1 = col_names[int(term_feats[i][1])]
        z = np.asarray(data["scores"], dtype=float)
        rec = {
            "term_name": term_names[i],
            "left_feature": f0,
            "right_feature": f1,
            "importance": float(importances[i]),
            "left_bin_edges": data.get("left_names"),
            "right_bin_edges": data.get("right_names"),
            "scores": z,
            "describe": _describe_interaction(z),
        }
        int_out.append(rec)

    # overall importance ranking for context
    order = np.argsort(importances)[::-1]
    ranking = [
        {
            "term": term_names[int(i)],
            "importance": float(importances[int(i)]),
            "arity": len(term_feats[int(i)]),
        }
        for i in order[:30]
    ]

    out = {
        "_label": "IN-SAMPLE PROBE (single global EBM on OOS h16-19 leftover) -- NOT a forecast",
        "cell": CELL,
        "global_sub": SUB,
        "leftover_target": "r2 = y_true - pred_adj (adj space) after Hero-A d8 global stage",
        "column_set": "masks UNION across blocks (survivors in >=1 cadence block)",
        "ebm_cfg": CFG,
        "n_rows_h16_19": int(X.shape[0]),
        "n_features_union": int(X.shape[1]),
        "ebm_intercept": float(ebm.intercept_),
        "insample_r2_on_leftover": insample_r2,
        "n_terms_total": len(term_names),
        "n_main_terms": len(main_terms),
        "n_interaction_terms": len(int_terms),
        "top_importance_ranking": ranking,
        "main_effects": main_out,
        "interactions": int_out,
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, default=_json_default)
    print(f"WROTE {OUT}", flush=True)
    print("\nTOP MAIN EFFECTS (feature | importance | form | sign lo->hi):", flush=True)
    for r in main_out:
        dsc = r["describe"]
        print(
            f"  {r['feature']:<28} {r['importance']:.5f}  {dsc.get('form', '?'):<40} "
            f"{dsc.get('sign_low_end', '?')}->{dsc.get('sign_high_end', '?')}",
            flush=True,
        )
    print("\nTOP INTERACTIONS (left x right | importance | range):", flush=True)
    for r in int_out:
        print(
            f"  {r['left_feature']:<22} x {r['right_feature']:<22} {r['importance']:.5f}  "
            f"range={r['describe'].get('score_range', float('nan')):.4g}",
            flush=True,
        )


if __name__ == "__main__":
    main()
