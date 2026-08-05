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


if __name__ == "__main__":
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--stage", choices=["spectrum", "fit"], required=True)
    a = ap_.parse_args()
    {"spectrum": stage_spectrum, "fit": stage_fit}[a.stage]()
