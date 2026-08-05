"""A heat equation on the exog interaction graph: Laplacian-regularised alpha.

§7 found the interaction alpha lives on a **graph**: nodes are features, edges are pairwise
products, edge weight is the product's |IC| with the HAR residual — a non-negative symmetric
function over the interaction graph. It is sparse (~25-100 live edges of 8,911), static across 18
years, and **hub-dominated**: 9 of the top 10 edges touch ``sumret``.

Given a graph, the natural prior is the heat equation on it. With ``L = D - W`` the graph
Laplacian, running heat flow for time ``t`` is the smoothing operator ``exp(-tL)``, and its
quadratic form is what appears in a penalty:

    minimise  ||y - X b||^2  +  lambda_1 ||b||^2  +  lambda_2 * b' L b

The ``b' L b`` term equals ``sum_{(a,b) in E} w_ab (b_a - b_b)^2`` — it penalises *differences
between coefficients of graph-adjacent terms* rather than their magnitudes. That is exactly the
right instrument for dense-but-weak alpha with known structure: plain ridge shrinks every
coefficient toward zero independently, whereas Laplacian regularisation lets related terms borrow
strength from each other. (Standard in genomics as network-constrained regularisation, where the
problem shape is identical: many tiny effects, no sparsity, but a known adjacency.) The solve is
one line — ``(X'X + lambda_1 I + lambda_2 L) b = X'y``.

**Two graphs, two hypotheses**, because "the interaction graph" admits two readings:

* ``node`` — L on the *feature* graph, penalising differences between the linear coefficients of
  features that interact strongly with each other.
* ``edge`` — L on the **line graph**: the products themselves are the nodes, and two products are
  adjacent when they *share a parent feature*. The penalty then says "products sharing a parent
  should have similar coefficients", which is precisely the hub structure §7 measured. This is the
  faithful reading and the one the finding predicts should work.

**Protocol — pre-registered, with a real holdout.** This is a *new* model family, so for the first
time in this study a clean decision is available (§10's audit explains why nothing else here had
one left). Fixed in advance:

* Grid: ``lambda_2 / lambda_1`` in ``{0.1, 1, 10, 100}``, both graph variants. Nothing else varies.
* The graph is **frozen on the first training window** (2006), matching §7's finding that structure
  is static, so no graph re-estimation can leak.
* **Search period: rows through 2020-12-31.** The single best configuration by search-period OOS R²
  is carried forward. **Holdout: 2021-01-01 onward, scored once.** The full search grid is reported
  regardless, so the reader sees what was chosen from.
* Benchmark: the §11.1 winner — the same feature set with a plain *diagonal* product penalty. The
  question is strictly "does graph smoothing beat plain shrinkage of the same terms", not "do
  interactions help", which §11.1 already answered.

Honest limit: the feature set and the decision to look at interactions at all were informed by
full-sample work, so the holdout is clean for the *graph/lambda* decision and not a virgin test of
the interaction channel itself.

Usage
-----
    python analysis/heat_graph.py --stage spectrum   # descriptive: what does the graph look like?
    python analysis/heat_graph.py --stage fit        # pre-registered search + holdout decision
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.alpha_manifestation import TW, dm_test  # noqa: E402
from analysis.alpha_panel import CACHE_DIR  # noqa: E402
from analysis.nl_sparsity import REFIT, RIDGE_ALPHA, _pair_ic, _products, _upper, base_columns  # noqa: E402
from analysis.wf import r2_oos  # noqa: E402

K_EDGES = 100  # §7's frozen edge count
LAMBDA_RATIOS = (0.1, 1.0, 10.0, 100.0)  # pre-registered, nothing else varies
PROD_PENALTY = 3e4  # the §11.1 winner's diagonal product penalty, held fixed
SPLIT = "2021-01-01"  # search <= 2020, holdout from here, scored once
VARIANTS = ("node", "edge")
OUT = "results/alpha_manifestation"
FIXED_CACHE = os.path.join(CACHE_DIR, "fixed")  # the §8-fixed panel


def _panel():
    os.environ["ALPHA_PANEL_CACHE"] = FIXED_CACHE
    import importlib

    import analysis.alpha_panel as ap

    importlib.reload(ap)
    return ap.load_panel()


def _graph(Ztr: np.ndarray, ytr: np.ndarray, ii: np.ndarray, jj: np.ndarray):
    """Freeze the graph: top-K edges by |IC|, with their weights and parent indices."""
    ic = np.abs(np.nan_to_num(_pair_ic(Ztr, ytr)[ii, jj]))
    sel = np.argsort(-ic)[:K_EDGES]
    return sel, ic[sel], ii[sel], jj[sel]


def _laplacian_node(p: int, w: np.ndarray, pi: np.ndarray, pj: np.ndarray) -> np.ndarray:
    """L = D - W on the FEATURE graph, from the frozen weighted edge list."""
    W = np.zeros((p, p))
    for k in range(len(w)):
        if pi[k] != pj[k]:  # a squared term is a self-loop and carries no difference penalty
            W[pi[k], pj[k]] += w[k]
            W[pj[k], pi[k]] += w[k]
    L = np.diag(W.sum(1)) - W
    tr = np.trace(L)
    return L * (len(L) / tr) if tr > 0 else L  # normalise so lambda_2 is comparable


def _laplacian_edge(pi: np.ndarray, pj: np.ndarray) -> np.ndarray:
    """L on the LINE graph: products are adjacent when they share a parent feature."""
    k = len(pi)
    A = np.zeros((k, k))
    for a in range(k):
        for b in range(a + 1, k):
            if len({pi[a], pj[a]} & {pi[b], pj[b]}) > 0:
                A[a, b] = A[b, a] = 1.0
    L = np.diag(A.sum(1)) - A
    tr = np.trace(L)
    return L * (k / tr) if tr > 0 else L


def stage_spectrum() -> None:
    """Descriptive only — no forecast is scored, so this costs no inferential budget."""
    os.makedirs(OUT, exist_ok=True)
    p = _panel()
    e = np.load(os.path.join(CACHE_DIR, "har_resid.npz"))["e"]
    bc, bn = base_columns(p)
    X = np.ascontiguousarray(p.X[TW:, bc], dtype=np.float64)
    ii, jj = _upper(X.shape[1])
    mu = X[:TW].mean(0)
    sel, w, pi, pj = _graph(X[:TW] - mu, e[:TW], ii, jj)

    deg = pd.Series(np.concatenate([pi, pj])).value_counts()
    print(f"frozen graph: {K_EDGES} edges over {X.shape[1]} nodes, {len(deg)} nodes touched")
    print("\n  node degree (top 8) — the hub structure §7 reported:")
    for node, d in deg.head(8).items():
        print(f"    {bn[node]:38s} degree {d}")
    print(f"\n  degree concentration: top node holds {deg.iloc[0] / (2 * K_EDGES):.1%} of edge ends")

    Ln = _laplacian_node(X.shape[1], w, pi, pj)
    Le = _laplacian_edge(pi, pj)
    for name, L in (("feature graph", Ln), ("line graph", Le)):
        ev = np.linalg.eigvalsh(L)
        ev = ev[ev > 1e-9]
        pr = float(ev.sum() ** 2 / (ev**2).sum()) if ev.size else np.nan
        print(
            f"\n  {name}: {len(ev)} non-zero eigenvalues, lambda_max {ev.max():.2f}, "
            f"participation ratio {pr:.1f} of {len(L)}"
        )
        print(f"    -> {'rich multi-scale structure' if pr > 0.25 * len(L) else 'concentrated / star-like: heat smoothing degenerates toward a hub average'}")
    pd.DataFrame({"node": [bn[i] for i in deg.index], "degree": deg.to_numpy()}).to_csv(
        f"{OUT}/heat_graph_degrees.csv", index=False
    )
    print(f"\nwrote {OUT}/heat_graph_degrees.csv")


def stage_fit() -> None:  # noqa: C901
    os.makedirs(OUT, exist_ok=True)
    p = _panel()
    e = np.load(os.path.join(CACHE_DIR, "har_resid.npz"))["e"]
    ts = pd.Series(pd.to_datetime(p.t[TW:]))
    bc, _ = base_columns(p)
    X = np.ascontiguousarray(p.X[TW:, bc], dtype=np.float64)
    n, pb = len(e), X.shape[1]
    ii, jj = _upper(pb)

    mu0 = X[:TW].mean(0)
    sel, w, pi, pj = _graph(X[:TW] - mu0, e[:TW], ii, jj)
    Ln = _laplacian_node(pb, w, pi, pj)
    Le = _laplacian_edge(pi, pj)

    arms = ["diagonal"] + [f"{v}_r{r}" for v in VARIANTS for r in LAMBDA_RATIOS]
    preds = {a: np.full(n - TW, np.nan) for a in arms}

    def floored(Ptr, Pte):
        sd = Ptr.std(0)
        sd = np.maximum(sd, 0.1 * np.median(sd[sd > 0]) if (sd > 0).any() else 1.0)
        return Ptr / sd, Pte / sd

    for t0 in range(TW, n, REFIT):
        tr = slice(t0 - TW, t0)
        t1 = min(t0 + REFIT, n)
        out = slice(t0 - TW, t1 - TW)
        mu = X[tr].mean(0)
        Ztr, Zte = X[tr] - mu, X[t0:t1] - mu
        ytr = e[tr]
        Ptr, Pte = floored(_products(Ztr, pi, pj), _products(Zte, pi, pj))
        Atr, Ate = np.hstack([Ztr, Ptr]), np.hstack([Zte, Pte])
        G = Atr.T @ Atr
        rhs = Atr.T @ (ytr - ytr.mean())
        diag = np.concatenate([np.full(pb, RIDGE_ALPHA), np.full(K_EDGES, PROD_PENALTY)])

        def solve(extra: np.ndarray | None) -> np.ndarray:
            M = G + np.diag(diag)
            if extra is not None:
                M = M + extra
            b = np.linalg.solve(M, rhs)
            return Ate @ b + ytr.mean()

        preds["diagonal"][out] = solve(None)
        for r in LAMBDA_RATIOS:
            blk = np.zeros_like(G)
            blk[:pb, :pb] = r * RIDGE_ALPHA * Ln
            preds[f"node_r{r}"][out] = solve(blk)
            blk = np.zeros_like(G)
            blk[pb:, pb:] = r * PROD_PENALTY * Le
            preds[f"edge_r{r}"][out] = solve(blk)

    y = e[TW:]
    tsx = ts.iloc[TW:].reset_index(drop=True)
    search = (tsx < SPLIT).to_numpy()
    hold = ~search
    print(f"  search rows {int(search.sum())} (through 2020), holdout rows {int(hold.sum())} (from {SPLIT})\n")
    rows = []
    for a in arms:
        rows.append(
            {
                "arm": a,
                "search_r2": r2_oos(y[search], preds[a][search]),
                "holdout_r2": r2_oos(y[hold], preds[a][hold]),
            }
        )
    d = pd.DataFrame(rows)
    base_s = d.loc[d.arm == "diagonal", "search_r2"].iloc[0]
    base_h = d.loc[d.arm == "diagonal", "holdout_r2"].iloc[0]
    d["search_gain"] = d.search_r2 - base_s
    d["holdout_gain"] = d.holdout_r2 - base_h
    print("  full pre-registered grid (search period is what the choice is made on):")
    print(d.round(5).to_string(index=False))

    cand = d[d.arm != "diagonal"]
    winner = cand.loc[cand.search_r2.idxmax(), "arm"]
    t_hold = dm_test(y[hold], preds[winner][hold], preds["diagonal"][hold])
    t_search = dm_test(y[search], preds[winner][search], preds["diagonal"][search])
    print(f"\n  chosen on the SEARCH period only: {winner}")
    print(f"    search  : ΔR² {d.loc[d.arm == winner, 'search_gain'].iloc[0]:+.5f}  DM-t {t_search:+.2f}")
    print(f"    HOLDOUT : ΔR² {d.loc[d.arm == winner, 'holdout_gain'].iloc[0]:+.5f}  DM-t {t_hold:+.2f}"
          "   <- the one number this protocol licenses")
    d.insert(0, "winner_by_search", winner)
    d["holdout_dm_t_winner"] = t_hold
    d.to_csv(f"{OUT}/heat_graph_fit.csv", index=False)
    print(f"wrote {OUT}/heat_graph_fit.csv")


# ---------------------------------------------------------------------------
# Stage 3 — the corrected test: a properly estimated graph, and heat flow on the
#           FEATURES rather than only on the coefficients
# ---------------------------------------------------------------------------

PERSIST_K = 100  # top-K per window, for the persistence-weighted graph
RATIOS2 = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3)  # re-centred: stage `fit` declined from its smallest
DIFFUSION_T = (0.1, 0.3, 1.0, 3.0)


def _persistence_graph(X: np.ndarray, e: np.ndarray, ii, jj, end: int):
    """Interaction graph averaged over every refit window in the SEARCH period.

    Stage ``fit`` froze the graph on a single 2006 window, and its hub disagreed with the
    persistence-weighted hub §7 reported — i.e. a one-window graph is a high-variance estimate, and
    heat flow on a mis-estimated graph tests nothing. This averages |IC| across ~167 windows, which
    is the low-variance estimator, and takes the top edges by *mean* strength. Uses only rows before
    ``end`` (the search/holdout split), so the holdout stays untouched.
    """
    acc = np.zeros(len(ii))
    hits = np.zeros(len(ii))
    n_win = 0
    for t0 in range(TW, end, REFIT):
        tr = slice(t0 - TW, t0)
        Ztr = X[tr] - X[tr].mean(0)
        ic = np.abs(np.nan_to_num(_pair_ic(Ztr, e[tr])[ii, jj]))
        acc += ic
        hits[np.argsort(-ic)[:PERSIST_K]] += 1.0
        n_win += 1
    acc /= max(n_win, 1)
    hits /= max(n_win, 1)
    sel = np.argsort(-acc)[:K_EDGES]
    return sel, acc[sel], hits[sel], ii[sel], jj[sel], n_win


def _laplacian_from_W(W: np.ndarray) -> np.ndarray:
    L = np.diag(W.sum(1)) - W
    tr = np.trace(L)
    return L * (len(L) / tr) if tr > 0 else L


def _cov_graph(X: np.ndarray, end: int, keep: float = 0.15) -> np.ndarray:
    """SIMILARITY adjacency: |corr| between features, sparsified to its strongest ``keep`` share.

    This is the adjacency the Laplacian coefficient prior actually assumes — "adjacent coefficients
    should be close" is justified for *correlated* features, not for *interacting* ones. Stage
    ``fit``'s conceptual mismatch, corrected. Search-period rows only.
    """
    Z = X[:end]
    C = np.abs(np.corrcoef(Z, rowvar=False))
    np.fill_diagonal(C, 0.0)
    C = np.nan_to_num(C)
    thr = np.quantile(C[np.triu_indices_from(C, 1)], 1.0 - keep)
    return np.where(C >= thr, C, 0.0)


def _weighted_line_laplacian(w: np.ndarray, pi: np.ndarray, pj: np.ndarray) -> np.ndarray:
    """Line graph with adjacency *weighted* by shared-parent strength, not 0/1.

    Stage ``fit`` used unweighted share-a-parent adjacency over 100 products drawn from 50 nodes,
    which is near-complete by construction (98 non-zero eigenvalues of 100) and therefore collapses
    to shrink-toward-a-common-mean. Weighting by the geometric mean of the two products' edge
    strengths, and only for shared parents, gives a graph with actual contrast.
    """
    k = len(pi)
    A = np.zeros((k, k))
    for a in range(k):
        for b in range(a + 1, k):
            shared = len({pi[a], pj[a]} & {pi[b], pj[b]})
            if shared:
                A[a, b] = A[b, a] = shared * np.sqrt(w[a] * w[b])
    return _laplacian_from_W(A)


def stage_fit2() -> None:  # noqa: C901
    """The corrected test. Four fixes over stage ``fit``, then one holdout score.

    1. **Graph properly estimated** — |IC| averaged over every search-period window instead of one
       frozen 2006 draw (:func:`_persistence_graph`).
    2. **Similarity graph added** — ``|corr|`` adjacency, which is the prior's actual assumption
       (:func:`_cov_graph`), alongside the interaction graph.
    3. **Grid re-centred** — stage ``fit``'s node arm declined monotonically from its *smallest*
       ratio (0.1), so that grid never bracketed the optimum. Now 0.001 to 0.3.
    4. **Heat flow on the FEATURES, not only the coefficients** — ``Z <- Z (I + tL)^-1``, a graph
       convolution / resolvent form of ``exp(-tL)``. This is the literal heat equation on the graph
       applied to the signal, which stage ``fit`` never tested. Implemented via ``S' G S`` so it
       costs no extra Gram.

    Also adds a *weak* baseline (plain ridge, no products) next to the strong one (the §11.1
    diagonal winner), so a graph-aware estimator can be seen to recover part of the interaction gain
    even if it cannot beat the best available shrinkage.

    **Holdout accounting, stated plainly: this is the second evaluation of the 2021-2024 block.**
    The winner is chosen on the search period via an internal 2007-2016 / 2017-2020 validation split
    so the holdout is touched exactly once more, and a reader should discount accordingly — two
    looks is not one look.
    """
    os.makedirs(OUT, exist_ok=True)
    p = _panel()
    e = np.load(os.path.join(CACHE_DIR, "har_resid.npz"))["e"]
    ts = pd.Series(pd.to_datetime(p.t[TW:]))
    bc, bn = base_columns(p)
    X = np.ascontiguousarray(p.X[TW:, bc], dtype=np.float64)
    n, pb = len(e), X.shape[1]
    ii, jj = _upper(pb)
    end_search = int((ts < SPLIT).sum())

    sel, w, hits, pi, pj, n_win = _persistence_graph(X, e, ii, jj, end_search)
    deg = pd.Series(np.concatenate([pi, pj])).value_counts()
    print(f"  persistence graph over {n_win} search windows; top nodes by degree:")
    for node, d in deg.head(5).items():
        print(f"    {bn[node]:38s} degree {d}")
    print(f"  mean top-{PERSIST_K} membership of the chosen edges: {hits.mean():.2f}\n")

    Wint = np.zeros((pb, pb))
    for k in range(len(w)):
        if pi[k] != pj[k]:
            Wint[pi[k], pj[k]] += w[k]
            Wint[pj[k], pi[k]] += w[k]
    L_int = _laplacian_from_W(Wint)
    L_cov = _laplacian_from_W(_cov_graph(X, end_search))
    L_edge = _weighted_line_laplacian(w, pi, pj)
    for nm, L in (("interaction(node)", L_int), ("similarity(cov)", L_cov), ("line(weighted)", L_edge)):
        ev = np.linalg.eigvalsh(L)
        ev = ev[ev > 1e-9]
        print(f"  {nm:20s} non-zero eig {len(ev):3d}  PR {ev.sum() ** 2 / (ev ** 2).sum():6.1f} of {len(L)}")

    arms: dict[str, np.ndarray] = {}

    def put(name, out, val):
        arms.setdefault(name, np.full(n - TW, np.nan))[out] = val

    diag_full = np.concatenate([np.full(pb, RIDGE_ALPHA), np.full(K_EDGES, PROD_PENALTY)])
    S_cache = {
        (g, t): np.linalg.inv(np.eye(pb) + t * (L_int if g == "int" else L_cov))
        for g in ("int", "cov")
        for t in DIFFUSION_T
    }

    def floored(Ptr, Pte):
        sd = Ptr.std(0)
        sd = np.maximum(sd, 0.1 * np.median(sd[sd > 0]) if (sd > 0).any() else 1.0)
        return Ptr / sd, Pte / sd

    for t0 in range(TW, n, REFIT):
        tr = slice(t0 - TW, t0)
        t1 = min(t0 + REFIT, n)
        out = slice(t0 - TW, t1 - TW)
        mu = X[tr].mean(0)
        Ztr, Zte = X[tr] - mu, X[t0:t1] - mu
        ytr = e[tr]
        yc = ytr - ytr.mean()
        Ptr, Pte = floored(_products(Ztr, pi, pj), _products(Zte, pi, pj))
        Atr, Ate = np.hstack([Ztr, Ptr]), np.hstack([Zte, Pte])
        G = Atr.T @ Atr
        rhs = Atr.T @ yc

        def solve(extra, Gm=G, r=rhs, Xte=Ate, dg=diag_full):
            b = np.linalg.solve(Gm + np.diag(dg) + (0.0 if extra is None else extra), r)
            return Xte @ b + ytr.mean()

        put("diagonal(strong base)", out, solve(None))
        # weak baseline: plain ridge on the base features only, no products
        Gz = G[:pb, :pb]
        bz = np.linalg.solve(Gz + RIDGE_ALPHA * np.eye(pb), rhs[:pb])
        put("plain_ridge_no_products", out, Zte @ bz + ytr.mean())

        for r in RATIOS2:
            for nm, L in (("nodeInt", L_int), ("nodeCov", L_cov)):
                blk = np.zeros_like(G)
                blk[:pb, :pb] = r * RIDGE_ALPHA * L
                put(f"{nm}_r{r}", out, solve(blk))
            blk = np.zeros_like(G)
            blk[pb:, pb:] = r * PROD_PENALTY * L_edge
            put(f"edgeW_r{r}", out, solve(blk))

        # heat flow on the FEATURES: Z <- Z S, so Gram -> S' Gzz S and cross -> S' Gzp
        for g in ("int", "cov"):
            for t in DIFFUSION_T:
                S = S_cache[(g, t)]
                Gd = G.copy()
                Gd[:pb, :pb] = S.T @ G[:pb, :pb] @ S
                Gd[:pb, pb:] = S.T @ G[:pb, pb:]
                Gd[pb:, :pb] = Gd[:pb, pb:].T
                rd = rhs.copy()
                rd[:pb] = S.T @ rhs[:pb]
                Xte = np.hstack([Zte @ S, Pte])
                put(f"heat{g}_t{t}", out, solve(None, Gd, rd, Xte))

    y = e[TW:]
    tsx = ts.iloc[TW:].reset_index(drop=True)
    val_start = int((tsx < "2017-01-01").sum())
    search = (tsx < SPLIT).to_numpy()
    hold = ~search
    inner_tr = np.zeros(len(y), bool)
    inner_tr[:val_start] = True
    inner_va = search & ~inner_tr

    base = arms["diagonal(strong base)"]
    rows = []
    for a, v in arms.items():
        rows.append(
            {
                "arm": a,
                "val_r2": r2_oos(y[inner_va], v[inner_va]),
                "search_r2": r2_oos(y[search], v[search]),
                "holdout_r2": r2_oos(y[hold], v[hold]),
            }
        )
    d = pd.DataFrame(rows)
    b = d[d.arm == "diagonal(strong base)"].iloc[0]
    for c in ("val", "search", "holdout"):
        d[f"{c}_gain"] = d[f"{c}_r2"] - b[f"{c}_r2"]
    print("\n  full grid (the choice is made on val_r2 = 2017-2020 only):")
    print(d.sort_values("val_r2", ascending=False).round(5).to_string(index=False))

    cand = d[~d.arm.isin(["diagonal(strong base)", "plain_ridge_no_products"])]
    win = cand.loc[cand.val_r2.idxmax(), "arm"]
    t_h = dm_test(y[hold], arms[win][hold], base[hold])
    print(f"\n  chosen on the internal validation block only: {win}")
    print(f"    validation: ΔR² {d.loc[d.arm == win, 'val_gain'].iloc[0]:+.5f}")
    print(f"    HOLDOUT   : ΔR² {d.loc[d.arm == win, 'holdout_gain'].iloc[0]:+.5f}  DM-t {t_h:+.2f}")
    print("    (second evaluation of the 2021-2024 block -- discount accordingly)")
    d.insert(0, "winner_by_validation", win)
    d["holdout_dm_t_winner"] = t_h
    d.to_csv(f"{OUT}/heat_graph_fit2.csv", index=False)
    print(f"wrote {OUT}/heat_graph_fit2.csv")


if __name__ == "__main__":
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--stage", choices=["spectrum", "fit", "fit2"], required=True)
    a = ap_.parse_args()
    {"spectrum": stage_spectrum, "fit": stage_fit, "fit2": stage_fit2}[a.stage]()
