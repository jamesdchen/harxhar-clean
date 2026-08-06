"""§27: maximum mining of Cucuringu — the full battery, pre-registered in the writeup FIRST.

Every instrument in his program that a single-market panel can support, run against the study's
three stable objects (frozen frame, factor-pair IC map +0.62, lead-lag flow +0.79):

``hermitian``
    C1. His directed-graph clustering embedding H = iA on the flow, plus the Hodge decomposition
    the ranking program implicitly assumes: gradient part (a potential — strict lead-lag
    hierarchy) vs curl part (genuine rotation). Gate: split-half aligned phase concordance >= 0.5.

``sync``
    C2. SyncRank — global lead-lag ordering by angular synchronization (top eigenvector of
    exp(i*pi*A/max|A|)). Gate: split-half Kendall tau >= 0.5, against the naive row-sum baseline.

``sponge``
    C3. SPONGE signed clustering of the off-diagonal IC map (a signed weighted graph on 20
    factor nodes), k = 2, 3, 4. Gate: split-half ARI above the 95th pct of a 200-draw
    edge-shuffle null.

``cpd``
    C4. Dynamic-network change-point detection — the rot detector done properly: quarterly
    trailing-2y map AND flow sequences, correlation-similarity matrices, scan statistic = mean
    similarity between adjacent 4-quarter windows. Gate (descriptive): minima must coincide with
    independently known events (2018-02, 2020-03, 2023-24), no tuning.

``rmt``
    C5. Marchenko-Pastur edge of the first-window base correlation with an
    autocorrelation-adjusted effective sample size (bar data is dependent; raw q lies). Where the
    frame and the §23 carriers sit relative to the noise edge.

``flow``
    C6. The only scored arm: flow-predicted products. Ghat_j(t) = sum_i A[i,j] G_i(t-1d) with A
    from the OPPOSITE half (no shared estimation); products of predicted scores; IC map with the
    daily HAR residual. Gates in order: (i) split-half map replication >= +0.30, (ii) alignment
    with the actual contemporaneous product map > 0. Pre-registered expectation: FAIL.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.linalg import eigh as geigh
from scipy.stats import kendalltau, spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.alpha_manifestation import TW  # noqa: E402
from analysis.alpha_panel import load_panel  # noqa: E402
from analysis.map_monitor import QPOOL, _frame_and_scores, _map  # noqa: E402
from analysis.nl_sparsity import base_columns  # noqa: E402
from analysis.synthesis import _p  # noqa: E402
from src.features.transforms.target import PERIODS_PER_DAY  # noqa: E402

OUT = "results/alpha_manifestation"
CARRIERS = [3, 9, 11, 14, 16, 19]  # §23 signal carriers (mid-spectrum)


# ---------------------------------------------------------------- shared machinery

def _daily(G: np.ndarray, e: np.ndarray, ts: pd.Series):
    """Last-bar factor scores and within-day mean residual, one row per day."""
    day_last = np.flatnonzero(ts.dt.date.ne(ts.dt.date.shift(-1)).to_numpy())
    day_id = ts.dt.date.astype(str).to_numpy()
    ed = pd.Series(e).groupby(day_id).mean().to_numpy()
    return G[day_last], ed, ts.iloc[day_last].reset_index(drop=True)


def _antisym(g: np.ndarray) -> np.ndarray:
    a, b = g[:-1], g[1:]
    a = (a - a.mean(0)) / (a.std(0) + 1e-12)
    b = (b - b.mean(0)) / (b.std(0) + 1e-12)
    C1 = (a.T @ b) / len(a)  # C1[i,j] = corr(G_i today, G_j tomorrow): i leads j
    return (C1 - C1.T) / 2.0


def _sym_ic_map(G: np.ndarray, e: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The factor-pair IC map as a symmetric QPOOL x QPOOL matrix (diagonal = squares)."""
    qi, qj = np.triu_indices(QPOOL)
    m = _map(G[:, qi] * G[:, qj], e, mask)
    M = np.zeros((QPOOL, QPOOL))
    M[qi, qj] = m
    M[qj, qi] = m
    return M


def _phase_concordance(t1: np.ndarray, t2: np.ndarray) -> float:
    """Circular concordance of two phase vectors, maximized over the two gauge freedoms of a
    Hermitian eigenvector (global rotation; complex conjugation = flow-direction sign)."""
    best = -1.0
    for s2 in (t2, -t2):
        d = t1 - s2
        best = max(best, float(np.hypot(np.cos(d).mean(), np.sin(d).mean())))
    return best


def _top_phases(A: np.ndarray) -> np.ndarray:
    lam, V = np.linalg.eigh(1j * A)
    v = V[:, np.argmax(np.abs(lam))]
    return np.angle(v)


# ---------------------------------------------------------------- C1: Hermitian + Hodge

def stage_hermitian() -> None:
    G, e, ts = _frame_and_scores()
    Gd, _, _ = _daily(G, e, ts)
    A = _antisym(Gd)
    h = len(Gd) // 2
    A1, A2 = _antisym(Gd[:h]), _antisym(Gd[h:])

    def hodge(a: np.ndarray):
        r = a.mean(1)
        grad = r[:, None] - r[None, :]
        return r, float((grad**2).sum() / (a**2).sum())

    r, gf = hodge(A)
    _, gf1 = hodge(A1)
    _, gf2 = hodge(A2)
    print(f"Hodge decomposition of the flow (complete graph, closed form):")
    print(f"  gradient (hierarchy) fraction: {gf:.3f}   halves: {gf1:.3f} / {gf2:.3f}")
    print(f"  curl (rotation) fraction:      {1 - gf:.3f}")

    th = _top_phases(A)
    conc = _phase_concordance(_top_phases(A1), _top_phases(A2))
    lam = np.linalg.eigvalsh(1j * A)
    lead_share = float(np.max(np.abs(lam)) ** 2 / (lam**2).sum() * 2)  # +/- pair
    print(f"\nHermitian embedding H = iA:")
    print(f"  top eigenpair energy share: {lead_share:.3f}")
    print(f"  split-half phase concordance: {conc:+.3f}   (gate >= +0.5)")
    rs, _ = spearmanr(th, r)
    # phases live on a circle; if gradient-dominant they occupy an arc and rank-correlate
    print(f"  spearman(phase, net leadingness): {rs:+.3f}")
    km = KMeans(3, n_init=10, random_state=0).fit(np.c_[np.cos(th), np.sin(th)])
    print("\n  phase ordering (leaders first) with 3-cluster labels:")
    ref = th[np.argmax(A.mean(1))]  # unwrap around the strongest source
    order = np.argsort(np.angle(np.exp(1j * (th - ref))))
    for a in order:
        tag = " <- carrier" if a in CARRIERS else ""
        print(f"    PC{a:2d}: phase {np.angle(np.exp(1j * (th[a] - ref))):+.2f}  "
              f"netlead {A.mean(1)[a]*QPOOL:+.3f}  cluster {km.labels_[a]}{tag}")
    print(f"\n  gate: {'PASS' if conc >= 0.5 else 'FAIL'}")


# ---------------------------------------------------------------- C2: SyncRank

def _syncrank(A: np.ndarray) -> np.ndarray:
    th = np.pi * A / (np.abs(A).max() + 1e-12)
    lam, V = np.linalg.eigh(np.exp(1j * th))
    v = V[:, np.argmax(lam)]
    p = np.angle(v)
    # fix the direction gauge with the (ambiguity-free) row-sum ordering of the SAME matrix
    if spearmanr(p, A.mean(1))[0] < 0:
        p = -p
    return p


def stage_sync() -> None:
    G, e, ts = _frame_and_scores()
    Gd, _, _ = _daily(G, e, ts)
    A = _antisym(Gd)
    h = len(Gd) // 2
    A1, A2 = _antisym(Gd[:h]), _antisym(Gd[h:])
    p1, p2 = _syncrank(A1), _syncrank(A2)
    t_sync, _ = kendalltau(p1, p2)
    t_naive, _ = kendalltau(A1.mean(1), A2.mean(1))
    t_agree, _ = kendalltau(_syncrank(A), A.mean(1))
    print("SyncRank (angular synchronization) vs naive row-sum leadingness:")
    print(f"  split-half Kendall tau, SyncRank ordering: {t_sync:+.3f}   (gate >= 0.5)")
    print(f"  split-half Kendall tau, row-sum ordering:  {t_naive:+.3f}")
    print(f"  full-sample agreement between the two:     {t_agree:+.3f}")
    print(f"  gate: {'PASS' if t_sync >= 0.5 else 'FAIL'}")


# ---------------------------------------------------------------- C3: SPONGE

def _sponge(M: np.ndarray, k: int) -> np.ndarray:
    Ap, Am = np.clip(M, 0, None), np.clip(-M, 0, None)
    np.fill_diagonal(Ap, 0.0)
    np.fill_diagonal(Am, 0.0)
    Dp, Dm = np.diag(Ap.sum(1)), np.diag(Am.sum(1))
    lam, V = geigh(Dp - Ap + Dm, Dm - Am + Dp)  # (L+ + D-) x = lam (L- + D+) x, tau=1
    emb = V[:, np.argsort(lam)[:k]]
    return KMeans(k, n_init=10, random_state=0).fit(emb).labels_


def stage_sponge() -> None:
    rng = np.random.default_rng(0)
    G, e, ts = _frame_and_scores()
    n = len(e)
    M = _sym_ic_map(G, e, np.ones(n, bool))
    M1 = _sym_ic_map(G, e, np.arange(n) < n // 2)
    M2 = _sym_ic_map(G, e, np.arange(n) >= n // 2)
    iu = np.triu_indices(QPOOL, k=1)

    def shuffled(m: np.ndarray) -> np.ndarray:
        s = np.zeros_like(m)
        s[iu] = rng.permutation(m[iu])
        return s + s.T

    print("SPONGE on the signed factor-pair IC map (off-diagonal, tau=1):")
    for k in (2, 3, 4):
        ari = adjusted_rand_score(_sponge(M1, k), _sponge(M2, k))
        null = np.array([adjusted_rand_score(_sponge(shuffled(M1), k), _sponge(shuffled(M2), k))
                         for _ in range(200)])
        p95 = float(np.quantile(null, 0.95))
        ok = ari > p95
        print(f"  k={k}: split-half ARI {ari:+.3f}   null p95 {p95:+.3f}  "
              f"null mean {null.mean():+.3f}   {'PASS' if ok else 'FAIL'}")
        if k == 2:
            lab = _sponge(M, k)
            for c in range(k):
                mem = [f"PC{a}" + ("*" if a in CARRIERS else "") for a in np.flatnonzero(lab == c)]
                print(f"    full-sample cluster {c}: {' '.join(mem)}   (* = §23 carrier)")


# ---------------------------------------------------------------- C4: change points

def stage_cpd() -> None:
    os.makedirs(OUT, exist_ok=True)
    G, e, ts = _frame_and_scores()
    n = len(e)
    trail = 2 * 252 * PERIODS_PER_DAY
    step = 3 * 21 * PERIODS_PER_DAY
    qi, qj = np.triu_indices(QPOOL)
    Q = G[:, qi] * G[:, qj]
    day = ts.dt.date.to_numpy()
    maps, flows, when = [], [], []
    for end in range(trail, n, step):
        mask = np.zeros(n, bool)
        mask[end - trail : end] = True
        maps.append(_map(Q, e, mask))
        gd = G[mask][np.flatnonzero(pd.Series(day[mask]).ne(pd.Series(day[mask]).shift(-1)))]
        flows.append(_antisym(gd)[np.triu_indices(QPOOL, k=1)])
        when.append(str(ts.iloc[end - 1].date()))
    w = 4  # quarters per side of the scan window

    def scan(vs: list[np.ndarray], name: str) -> pd.DataFrame:
        S = np.corrcoef(np.array(vs))
        s = np.array([S[t - w : t, t : t + w].mean() for t in range(w, len(vs) - w)])
        d = pd.DataFrame({"asof": when[w : len(vs) - w], "adjacency_sim": s})
        thr = s.mean() - 1.5 * s.std()
        print(f"\n{name}: scan statistic (mean cross-window similarity), "
              f"mean {s.mean():+.3f} sd {s.std():.3f}")
        print("  candidate breaks (local minima below mean - 1.5 sd):")
        for t in range(1, len(s) - 1):
            if s[t] < thr and s[t] <= s[t - 1] and s[t] <= s[t + 1]:
                print(f"    {d['asof'].iloc[t]}  sim {s[t]:+.3f}")
        d["kind"] = name
        return d

    d1 = scan(maps, "map")
    d2 = scan(flows, "flow")
    pd.concat([d1, d2]).to_csv(f"{OUT}/cucuringu_cpd.csv", index=False)
    print(f"\nwrote {OUT}/cucuringu_cpd.csv")


# ---------------------------------------------------------------- C5: RMT / MP edge

def stage_rmt() -> None:
    p = load_panel()
    bc, _ = base_columns(p)
    X = np.ascontiguousarray(p.X[TW : 2 * TW, bc], dtype=np.float64)
    sd0 = X.std(0)
    live = sd0 > 1e-8
    Z = (X[:, live] - X[:, live].mean(0)) / X[:, live].std(0)
    lam = np.sort(np.linalg.eigvalsh(np.corrcoef(Z, rowvar=False)))[::-1]
    N, T = Z.shape[1], Z.shape[0]

    # dependence-adjusted effective T: integrated autocorrelation time of the factor scores
    lam_f, V = np.linalg.eigh(np.corrcoef(Z, rowvar=False))
    S = Z @ V[:, np.argsort(lam_f)[::-1][:QPOOL]]
    taus = []
    for j in range(QPOOL):
        s = S[:, j] - S[:, j].mean()
        ac = np.correlate(s, s, "full")[len(s) - 1 :] / (s @ s)
        tau, L = 1.0, 1
        while L < 5 * PERIODS_PER_DAY and ac[L] > 0.05:
            tau += 2 * ac[L]
            L += 1
        taus.append(tau)
    t_eff = T / float(np.median(taus))
    for label, q in (("raw (iid) q", N / T), ("dependence-adjusted q", N / t_eff)):
        edge = (1 + np.sqrt(q)) ** 2
        above = int((lam > edge).sum())
        print(f"{label} = {q:.4f}: MP edge {edge:.3f}, eigenvalues above: {above} / {N}")
    print(f"  (median integrated autocorrelation time of factor scores: "
          f"{np.median(taus):.0f} bars; T_eff = {t_eff:.0f})")
    edge = (1 + np.sqrt(N / t_eff)) ** 2
    print(f"\nframe (QPOOL={QPOOL}) vs adjusted edge {edge:.2f}:")
    for a in range(QPOOL):
        tag = " <- carrier" if a in CARRIERS else ""
        print(f"  PC{a:2d}: eigenvalue {lam[a]:7.2f}  "
              f"{'above' if lam[a] > edge else 'BELOW'}{tag}")


# ---------------------------------------------------------------- C6: the scored arm

def stage_flow() -> None:
    G, e, ts = _frame_and_scores()
    Gd, ed, _ = _daily(G, e, ts)
    h = len(Gd) // 2
    A1, A2 = _antisym(Gd[:h]), _antisym(Gd[h:])
    qi, qj = np.triu_indices(QPOOL)

    def half_map(g: np.ndarray, res: np.ndarray, A_other: np.ndarray) -> np.ndarray:
        ghat = g[:-1] @ A_other  # Ghat_j(t) = sum_i A[i,j] G_i(t-1)
        q = ghat[:, qi] * ghat[:, qj]
        return _map(q, res[1:], np.ones(len(q), bool))

    m1 = half_map(Gd[:h], ed[:h], A2)   # cross-half A: no shared estimation
    m2 = half_map(Gd[h:], ed[h:], A1)
    rep = float(np.corrcoef(m1, m2)[0, 1])
    actual = _map(Gd[:, qi] * Gd[:, qj], ed, np.ones(len(Gd), bool))
    align = float(np.corrcoef((m1 + m2) / 2, actual)[0, 1])
    print("C6 flow-predicted products (the only scored arm; pre-registered expectation: FAIL):")
    print(f"  gate (i)  split-half replication of the flow-predicted map: {rep:+.3f}  "
          f"(need >= +0.30) -> {'PASS' if rep >= 0.30 else 'FAIL'}")
    print(f"  gate (ii) alignment with the actual contemporaneous map:    {align:+.3f}  "
          f"(need > 0)     -> {'PASS' if align > 0 else 'FAIL'}")
    print(f"  mean |IC|: flow-predicted {np.abs((m1 + m2) / 2).mean():.4f}   "
          f"actual {np.abs(actual).mean():.4f}")
    if rep >= 0.30 and align > 0:
        print("  both gates pass -> a walk-forward arm may now be pre-registered (§27).")
    else:
        print("  gates not met -> no walk-forward arm; the flow stays descriptive on this panel.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["hermitian", "sync", "sponge", "cpd", "rmt", "flow"],
                    required=True)
    a = ap.parse_args()
    {"hermitian": stage_hermitian, "sync": stage_sync, "sponge": stage_sponge,
     "cpd": stage_cpd, "rmt": stage_rmt, "flow": stage_flow}[a.stage]()
