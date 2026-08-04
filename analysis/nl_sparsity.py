"""Is the NONLINEAR / INTERACTION alpha locally sparse? (the part the linear test left open)

`alpha_manifestation.py` refuted sparse-rotating alpha in the **linear marginal** channel:
OOS accuracy rose monotonically with the number of features and the daily top-k set churned
~2%. That says nothing about interactions — and the intraday-regime study put ~38% of the
tuned EBM's edge in genuinely nonlinear structure. So the same question is re-asked here in
the pairwise-product space, where the prior is the opposite (thousands of candidate products,
most of them noise, so *pooled* sparsity is almost guaranteed).

The trap this module is built to avoid: **"sparse beats dense" in a p >> n space is not
evidence of local sparsity.** With ~9k candidate products and 12k training rows, top-k
selection wins on variance reduction alone, whichever k is best, whether or not the active
set moves. Distinguishing the two claims needs three separate measurements:

1. **Does the interaction channel add anything at all?** ``base`` (133 linear columns) vs
   ``base + top-k products``. If nothing is added, sparsity of it is moot.
2. **Is it sparse *pooled*?** Where the k-ladder peaks. A peak at small k = pooled sparsity
   (contrast the linear channel, which rose monotonically to all 246).
3. **Is it sparse *locally* — i.e. does the active set rotate?** The decisive arm pair:
   **dynamic** selection (reselect from the trailing window at every monthly refit) vs
   **static** selection (choose once from the first window, frozen forever), with
   *coefficients refit identically* in both. Only selection differs, so a dynamic win is
   rotation and nothing else. Reported alongside the month-to-month churn of the selected
   set and the within-month participation ratio against a circular-shift null.

**The design matrix must be clipped, and this is not cosmetic.** The panel is already
rolling-robust-scaled, but re-standardizing by a *training-window* sd reproduces the repo's
Part-3 "divide by a transiently-degenerate scale" disease: window-standardized test values
reach **1131 sigma**, and products of those square the damage. Unclipped, the linear base arm
scores R2 **-0.635** (vs +0.021 clipped) and the pair ICs that drive selection are pure
outlier artifacts. Every arm therefore standardizes on the training window and clips to
**+-4** — applied to train and test alike, and to base columns and products alike, so no arm
gets an advantage. On a rolling-robust-scaled feature +-4 is already extreme; the ladder is
flat in the clip (R2 +0.021 / +0.020 / +0.019 / +0.018 at +-4 / 6 / 8 / 10).

All pairwise product ICs at every refit come from three rolled window matrices rather than a
materialized product matrix: with ``Z`` the window's standardized base features,

    G = ZᵀZ,   A = Zᵀ diag(e) Z,   Q = (Z∘Z)ᵀ(Z∘Z)

give, for every product ``P_ij = Z_i Z_j``, ``E[P] = G_ij/n``, ``E[P²] = Q_ij/n`` and
``cov(P, e) = A_ij/n − E[P]·ē`` — so the full ~9k-column IC vector costs three matmuls, and
only the selected k columns are ever materialized.

Usage
-----
    python analysis/nl_sparsity.py --stage ladder    # arms 1-3: add / pooled-sparse / rotating
    python analysis/nl_sparsity.py --stage local     # within-month PR + IC persistence vs null
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.alpha_manifestation import TW, _blocks, _gini, _pr, _shifts, dm_test  # noqa: E402
from analysis.alpha_panel import CACHE_DIR, load_panel  # noqa: E402
from analysis.wf import r2_oos  # noqa: E402
from src.features.transforms.target import PERIODS_PER_DAY  # noqa: E402

REFIT = 21 * PERIODS_PER_DAY  # monthly reselect + refit
RIDGE_ALPHA = 3000.0  # as in the linear study; columns are window-standardized
CLIP = 4.0  # design-matrix clip in window-sd units (see module docstring — load-bearing)
EXOG_WINDOWS = (1, 25, 625)  # fast / intraday / slow, not all 6 (keeps the pair space ~9k)
KS = (10, 25, 50, 100, 200, 400)
OUT = "results/alpha_manifestation"


def _p(name: str) -> str:
    return os.path.join(CACHE_DIR, name)


def base_columns(p) -> tuple[np.ndarray, list[str]]:
    """The linear base whose pairwise products form the candidate interaction space."""
    cols = list(p.cols("har"))
    cols += [
        j
        for j in p.cols("value")
        if p.window[j] in EXOG_WINDOWS
    ]
    cols += [p.names.index(c) for c in ("is_open", "is_close", "is_overnight", "hour")]
    cols_arr = np.asarray(cols, dtype=np.int64)
    return cols_arr, [p.names[j] for j in cols_arr]


def _pair_ic(Z: np.ndarray, e: np.ndarray) -> np.ndarray:
    """(p, p) IC of every product ``Z_i * Z_j`` with ``e``, via three window matmuls."""
    n = len(Z)
    G = Z.T @ Z
    A = Z.T @ (Z * e[:, None])
    Z2 = Z * Z
    Q = Z2.T @ Z2
    mean_p = G / n
    var_p = np.maximum(Q / n - mean_p**2, 1e-12)
    cov = A / n - mean_p * e.mean()
    sd_e = e.std()
    return cov / (np.sqrt(var_p) * (sd_e if sd_e > 0 else 1.0))


def _upper(pb: int) -> tuple[np.ndarray, np.ndarray]:
    """Row/col indices of the upper triangle incl. diagonal (squares are interactions too)."""
    return np.triu_indices(pb)


def _products(Z: np.ndarray, ii: np.ndarray, jj: np.ndarray) -> np.ndarray:
    return Z[:, ii] * Z[:, jj]


def _std_clip(
    tr: np.ndarray, te: np.ndarray, clip: float = CLIP
) -> tuple[np.ndarray, np.ndarray]:
    """Standardize on the training window, clip both sides to ``+-clip`` window sds.

    Causal (train-window statistics only) and applied identically to train and test, so the
    fit sees the same bounded distribution it predicts on. Without the clip a single
    regime-shifted column detonates the forecast — see the module docstring.
    """
    mu, sd = tr.mean(0), tr.std(0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    return np.clip((tr - mu) / sd, -clip, clip), np.clip((te - mu) / sd, -clip, clip)


def _fit_predict(
    Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, alpha: float = RIDGE_ALPHA
) -> np.ndarray:
    """Plain ridge. Columns are expected pre-standardized/clipped by :func:`_std_clip`."""
    g = Xtr.T @ Xtr + alpha * np.eye(Xtr.shape[1])
    b = np.linalg.solve(g, Xtr.T @ (ytr - ytr.mean()))
    return Xte @ b + ytr.mean()


def stage_ladder() -> None:  # noqa: C901
    os.makedirs(OUT, exist_ok=True)
    p = load_panel()
    e = np.load(_p("har_resid.npz"))["e"]
    bcols, bnames = base_columns(p)
    X = np.ascontiguousarray(p.X[TW:, bcols], dtype=np.float64)
    n = len(e)
    assert len(X) == n, (len(X), n)
    pb = X.shape[1]
    ii, jj = _upper(pb)
    n_pairs = len(ii)
    print(f"base {pb} columns -> {n_pairs} candidate products (incl. squares)", flush=True)

    arms = ["base"] + [f"dyn{k}" for k in KS] + [f"stat{k}" for k in KS]
    preds = {a: np.full(n - TW, np.nan) for a in arms}
    sel_hist: dict[int, list[np.ndarray]] = {k: [] for k in KS}
    static_sel: dict[int, np.ndarray] = {}
    first = True

    for t0 in range(TW, n, REFIT):
        tr = slice(t0 - TW, t0)
        t1 = min(t0 + REFIT, n)
        Ztr, Zte = _std_clip(X[tr], X[t0:t1])
        ytr = e[tr]
        out = slice(t0 - TW, t1 - TW)

        preds["base"][out] = _fit_predict(Ztr, ytr, Zte)

        ic = _pair_ic(Ztr, ytr)[ii, jj]
        rank = np.argsort(-np.abs(np.nan_to_num(ic)))
        for k in KS:
            dyn = rank[:k]
            sel_hist[k].append(np.sort(dyn))
            if first:
                static_sel[k] = dyn.copy()
            for tag, sel in (("dyn", dyn), ("stat", static_sel[k])):
                Ptr, Pte = _std_clip(
                    _products(Ztr, ii[sel], jj[sel]), _products(Zte, ii[sel], jj[sel])
                )
                preds[f"{tag}{k}"][out] = _fit_predict(
                    np.hstack([Ztr, Ptr]), ytr, np.hstack([Zte, Pte])
                )
        first = False

    y = e[TW:]
    rows = []
    cb, rb = float(np.corrcoef(preds["base"], y)[0, 1]), r2_oos(y, preds["base"])
    print(f"\n  base (linear, {pb} cols): corr {cb:+.4f}  R2 {rb:+.5f}", flush=True)
    rows.append({"arm": "base", "k": 0, "corr": cb, "r2": rb, "dr2_vs_base": 0.0,
                 "dm_t_vs_base": 0.0, "dm_t_dyn_vs_stat": np.nan, "churn": np.nan})
    for k in KS:
        d, s = preds[f"dyn{k}"], preds[f"stat{k}"]
        cd, rd = float(np.corrcoef(d, y)[0, 1]), r2_oos(y, d)
        cs, rs = float(np.corrcoef(s, y)[0, 1]), r2_oos(y, s)
        churn = float(
            np.mean(
                [
                    1.0 - len(np.intersect1d(a, b)) / k
                    for a, b in zip(sel_hist[k][:-1], sel_hist[k][1:])
                ]
            )
        )
        print(
            f"  k={k:3d}  dynamic corr {cd:+.4f} R2 {rd:+.5f} (dR2 {rd - rb:+.5f}, DM-t {dm_test(y, d, preds['base']):+5.2f})"
            f" | static R2 {rs:+.5f} (dR2 {rs - rb:+.5f})"
            f" | dyn-vs-stat DM-t {dm_test(y, d, s):+5.2f} | churn {churn:.3f}",
            flush=True,
        )
        rows.append({"arm": "dynamic", "k": k, "corr": cd, "r2": rd, "dr2_vs_base": rd - rb,
                     "dm_t_vs_base": dm_test(y, d, preds["base"]),
                     "dm_t_dyn_vs_stat": dm_test(y, d, s), "churn": churn})
        rows.append({"arm": "static", "k": k, "corr": cs, "r2": rs, "dr2_vs_base": rs - rb,
                     "dm_t_vs_base": dm_test(y, s, preds["base"]),
                     "dm_t_dyn_vs_stat": np.nan, "churn": 0.0})
    pd.DataFrame(rows).to_csv(f"{OUT}/nl_sparsity_ladder.csv", index=False)

    # which products does the selector keep, and how concentrated is the pooled IC?
    best = sel_hist[100]
    freq = pd.Series(np.concatenate(best)).value_counts()
    top = [
        {"pair": f"{bnames[ii[q]]} x {bnames[jj[q]]}", "months_selected": int(c),
         "of_months": len(best)}
        for q, c in freq.head(15).items()
    ]
    pd.DataFrame(top).to_csv(f"{OUT}/nl_top_pairs.csv", index=False)
    print("\n  most persistently selected products (top-100 arm):", flush=True)
    for r in top[:10]:
        print(f"    {r['months_selected']:3d}/{r['of_months']} months  {r['pair']}", flush=True)
    print(f"\nwrote {OUT}/nl_sparsity_ladder.csv + nl_top_pairs.csv")


def stage_local() -> None:
    """Within-month concentration + persistence of the *interaction* IC vector vs null."""
    os.makedirs(OUT, exist_ok=True)
    p = load_panel()
    e = np.load(_p("har_resid.npz"))["e"][TW:]
    bcols, bnames = base_columns(p)
    X = np.ascontiguousarray(p.X[2 * TW :, bcols], dtype=np.float64)
    ts = pd.Series(pd.to_datetime(p.t[2 * TW :]))
    n = len(e)
    assert len(X) == n
    pb = X.shape[1]
    ii, jj = _upper(pb)
    blk = _blocks(ts)
    nb = blk.max() + 1

    # restrict to the 1000 strongest products pooled, so the PR is over a comparable
    # candidate set rather than 9k mostly-dead columns
    Zall, _ = _std_clip(X, X[:1])
    ic_pool_full = _pair_ic(Zall, e)[ii, jj]
    keep = np.argsort(-np.abs(np.nan_to_num(ic_pool_full)))[:1000]
    ic_pool = ic_pool_full[keep]

    def monthly(y: np.ndarray) -> np.ndarray:
        M = np.full((nb, len(keep)), np.nan)
        for b in range(nb):
            m = blk == b
            if m.sum() < 200:
                continue
            Zb, _ = _std_clip(X[m], X[m][:1])
            M[b] = _pair_ic(Zb, y[m])[ii, jj][keep]
        return M

    M = monthly(e)
    pr_month = float(np.nanmean([_pr(M[b]) for b in range(nb)]))
    D = M - np.nanmean(M, axis=0)
    sp = float(
        np.nanmean(
            [pd.Series(D[b]).corr(pd.Series(D[b + 1]), method="spearman") for b in range(nb - 1)]
        )
    )
    pr_null, sp_null = [], []
    for sh in _shifts(n)[:8]:
        Mn = monthly(np.roll(e, sh))
        pr_null.append(float(np.nanmean([_pr(Mn[b]) for b in range(nb)])))
        Dn = Mn - np.nanmean(Mn, axis=0)
        sp_null.append(
            float(
                np.nanmean(
                    [
                        pd.Series(Dn[b]).corr(pd.Series(Dn[b + 1]), method="spearman")
                        for b in range(nb - 1)
                    ]
                )
            )
        )
    print(f"interaction IC vector over the 1000 strongest products, {nb} months")
    print(f"  participation ratio: within-month {pr_month:.1f}  pooled {_pr(ic_pool):.1f}  "
          f"null {np.mean(pr_null):.1f}  (of 1000)")
    print(f"  pooled |IC|: mean {np.abs(ic_pool).mean():.4f}  max {np.abs(ic_pool).max():.4f}  "
          f"Gini {_gini(np.abs(ic_pool)):.3f}")
    print(f"  month-to-month Spearman of the time-varying part: {sp:+.4f}  "
          f"(null {np.mean(sp_null):+.4f})")
    pd.DataFrame(
        {
            "pair": [f"{bnames[ii[q]]} x {bnames[jj[q]]}" for q in keep],
            "ic_pooled": ic_pool,
            "ic_month_sd": np.nanstd(M, axis=0),
        }
    ).to_csv(f"{OUT}/nl_pair_ic.csv", index=False)
    pd.DataFrame(
        [{"pr_within_month": pr_month, "pr_pooled": _pr(ic_pool), "pr_null": np.mean(pr_null),
          "spearman_persistence": sp, "spearman_null": np.mean(sp_null), "n_months": nb}]
    ).to_csv(f"{OUT}/nl_local_sparsity.csv", index=False)
    print(f"wrote {OUT}/nl_pair_ic.csv + nl_local_sparsity.csv")


def stage_vsfull() -> None:
    """Do the interaction terms still add on top of the FULL linear model?

    ``stage_ladder`` measures the interaction gain over a 133-column base (the columns whose
    products form the candidate space). That base is smaller than the production linear set,
    so part of the gain could be information the full linear model already has. This arm
    re-runs the winner against all 516 linear exog columns (246 value + 270
    availability/occurrence), same monthly refit, same clip, so the only difference is the
    100 frozen product columns.
    """
    os.makedirs(OUT, exist_ok=True)
    p = load_panel()
    e = np.load(_p("har_resid.npz"))["e"]
    full = np.concatenate([p.cols("har"), p.cols("value"), p.cols("indicator")])
    bcols, bnames = base_columns(p)
    XF = np.ascontiguousarray(p.X[TW:, full], dtype=np.float64)
    XB = np.ascontiguousarray(p.X[TW:, bcols], dtype=np.float64)
    n = len(e)
    ii, jj = _upper(XB.shape[1])
    print(f"full linear base: {XF.shape[1]} cols; products drawn from {XB.shape[1]}", flush=True)

    lin = np.full(n - TW, np.nan)
    aug = np.full(n - TW, np.nan)
    sel: np.ndarray | None = None
    for t0 in range(TW, n, REFIT):
        tr = slice(t0 - TW, t0)
        t1 = min(t0 + REFIT, n)
        out = slice(t0 - TW, t1 - TW)
        Ftr, Fte = _std_clip(XF[tr], XF[t0:t1])
        Btr, Bte = _std_clip(XB[tr], XB[t0:t1])
        ytr = e[tr]
        lin[out] = _fit_predict(Ftr, ytr, Fte)
        if sel is None:  # freeze the selection on the first window (static arm)
            sel = np.argsort(-np.abs(np.nan_to_num(_pair_ic(Btr, ytr)[ii, jj])))[:100]
        Ptr, Pte = _std_clip(
            _products(Btr, ii[sel], jj[sel]), _products(Bte, ii[sel], jj[sel])
        )
        aug[out] = _fit_predict(np.hstack([Ftr, Ptr]), ytr, np.hstack([Fte, Pte]))

    y = e[TW:]
    cl, rl = float(np.corrcoef(lin, y)[0, 1]), r2_oos(y, lin)
    ca, ra = float(np.corrcoef(aug, y)[0, 1]), r2_oos(y, aug)
    t = dm_test(y, aug, lin)
    print(f"  full linear (516 cols)      : corr {cl:+.4f}  R2 {rl:+.5f}")
    print(f"  + 100 frozen products       : corr {ca:+.4f}  R2 {ra:+.5f}  "
          f"dR2 {ra - rl:+.5f}  ({100 * (ra - rl) / rl:+.0f}%)  DM-t {t:+.2f}")
    pd.DataFrame(
        [{"arm": "full_linear", "corr": cl, "r2": rl, "dr2": 0.0, "dm_t": 0.0},
         {"arm": "full_linear+100_frozen_products", "corr": ca, "r2": ra, "dr2": ra - rl,
          "dm_t": t}]
    ).to_csv(f"{OUT}/nl_vs_full_linear.csv", index=False)
    print(f"wrote {OUT}/nl_vs_full_linear.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["ladder", "local", "vsfull"], required=True)
    a = ap.parse_args()
    {"ladder": stage_ladder, "local": stage_local, "vsfull": stage_vsfull}[a.stage]()
