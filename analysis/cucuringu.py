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


# ---------------------------------------------------------------- §28: the phase of the cycle

def _rot_plane(A: np.ndarray):
    """The dominant rotation plane: real/imag parts of the top eigenvector of iA."""
    lam, V = np.linalg.eigh(1j * A)
    v = V[:, np.argmax(lam)]
    return np.real(v), np.imag(v)


def _wrap(a: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * a))


def stage_phase() -> None:
    """§28 K + F: kinematics of the cycle, and the phase-forecast skill ladder."""
    G, e, ts = _frame_and_scores()
    Gd, _, _ = _daily(G, e, ts)
    A = _antisym(Gd)
    u, w = _rot_plane(A)
    phi = np.arctan2(Gd @ w, Gd @ u)
    d = _wrap(np.diff(phi))
    mv = np.exp(1j * d).mean()
    omega, R = float(np.angle(mv)), float(np.abs(mv))
    sigma = float(np.sqrt(-2 * np.log(max(R, 1e-12))))
    print("K. kinematics (shared plane = top Hermitian eigenpair of the full-sample flow):")
    print(f"  daily drift omega {omega:+.4f} rad/d   circular sd {sigma:.3f} rad/d   R {R:.3f}")
    print(f"  implied period {2 * np.pi / abs(omega):.0f} days   "
          f"drift-to-noise {abs(omega) / sigma:.3f}")
    print(f"  diffusion-limited horizon (phase error ~ pi/2, iid increments): "
          f"{(np.pi / 2 / sigma) ** 2:.1f} days")
    h = len(Gd) // 2
    om_h = {}
    for name, sl in (("h1", slice(0, h)), ("h2", slice(h, None))):
        m = np.exp(1j * _wrap(np.diff(phi[sl]))).mean()
        om_h[name] = float(np.angle(m))
        print(f"  {name}: omega {om_h[name]:+.4f}  R {np.abs(m):.3f}")
    agree = np.sign(om_h["h1"]) == np.sign(om_h["h2"])
    print(f"  rotation direction agrees across halves: {bool(agree)}")

    print("\nF. phase-forecast skill, S(h) = mean cos(err | omega from OTHER half) - no-drift null:")
    H = [1, 2, 5, 10, 21, 42, 63]
    gate5 = []
    for name, sl, om in (("h1", slice(0, h), om_h["h2"]), ("h2", slice(h, None), om_h["h1"])):
        ph = phi[sl]
        row = []
        for hh in H:
            err = _wrap(ph[hh:] - ph[:-hh] - om * hh)
            null = _wrap(ph[hh:] - ph[:-hh])
            row.append(float(np.cos(err).mean() - np.cos(null).mean()))
        gate5.append(row[H.index(5)])
        print(f"  {name}: " + "  ".join(f"h={hh}: {s:+.3f}" for hh, s in zip(H, row)))
    ok = all(s > 0 for s in gate5)
    print(f"  gate (skill > 0 at h = 5 in both halves): {'PASS' if ok else 'FAIL'}")


def stage_phasex() -> None:
    """§28 E1 + E2: does anything that pays depend on the phase?"""
    rng = np.random.default_rng(1)
    G, e, ts = _frame_and_scores()
    Gd, ed, _ = _daily(G, e, ts)
    A = _antisym(Gd)
    u, w = _rot_plane(A)
    phi = np.arctan2(Gd @ w, Gd @ u)
    nd = len(Gd)
    hd = nd // 2

    def ic(a: np.ndarray, b: np.ndarray) -> float:
        a = (a - a.mean()) / (a.std() + 1e-12)
        b = (b - b.mean()) / (b.std() + 1e-12)
        return float((a * b).mean())

    z = (ed - ed.mean()) / ed.std()
    harm = {"cos": np.cos(phi), "sin": np.sin(phi),
            "cos2": np.cos(2 * phi), "sin2": np.sin(2 * phi)}
    halves = {"h1": slice(0, hd), "h2": slice(hd, None)}
    null = []
    for _ in range(200):
        zz = np.roll(z, int(rng.integers(63, nd - 63)))
        null.append(max(abs(ic(hv[sl], zz[sl])) for hv in harm.values()
                        for sl in halves.values()))
    p95 = float(np.quantile(null, 0.95))
    print(f"E1. phase harmonics vs daily HAR residual (shift-null p95 of max|IC|: {p95:.4f}):")
    any_pass = False
    for name, hv in harm.items():
        i1, i2 = ic(hv[:hd], z[:hd]), ic(hv[hd:], z[hd:])
        hit = np.sign(i1) == np.sign(i2) and abs(i1) > p95 and abs(i2) > p95
        any_pass |= hit
        print(f"  {name:5s}: h1 {i1:+.4f}  h2 {i2:+.4f}   {'PASS' if hit else 'fail'}")
    print(f"  E1 gate: {'PASS' if any_pass else 'FAIL'}")

    # E2: realized map-channel gain, profiled over 8 phase bins (map from the OTHER half)
    qi, qj = np.triu_indices(QPOOL)
    Q = G[:, qi] * G[:, qj]
    day_codes = pd.factorize(ts.dt.date.to_numpy())[0]
    bmask1 = day_codes < hd
    m1, m2 = _map(Q, e, bmask1), _map(Q, e, ~bmask1)
    bins = np.clip(np.floor((phi + np.pi) / (2 * np.pi / 8)).astype(int), 0, 7)
    profs, amps, amp_p95 = {}, {}, {}
    for name, bm, mo, bsl in (("h1", bmask1, m2, slice(0, hd)),
                              ("h2", ~bmask1, m1, slice(hd, None))):
        Qs = (Q[bm] - Q[bm].mean(0)) / (Q[bm].std(0) + 1e-12)
        ez = (e[bm] - e[bm].mean()) / e[bm].std()
        cd = pd.Series((Qs @ mo) * ez).groupby(day_codes[bm]).mean().to_numpy()
        bb = bins[bsl]
        prof = np.array([cd[bb == k].mean() for k in range(8)])
        profs[name], amps[name] = prof, float(prof.var())
        nv = []
        for _ in range(200):
            cs = np.roll(cd, int(rng.integers(63, len(cd) - 63)))
            nv.append(np.array([cs[bb == k].mean() for k in range(8)]).var())
        amp_p95[name] = float(np.quantile(nv, 0.95))
    pc = float(np.corrcoef(profs["h1"], profs["h2"])[0, 1])
    print(f"\nE2. map-channel realized gain over 8 phase bins (map from other half):")
    print("  bin centers (rad): " + "  ".join(f"{-np.pi + (k + .5) * np.pi / 4:+.2f}"
                                              for k in range(8)))
    for name in ("h1", "h2"):
        print(f"  {name}: " + "  ".join(f"{v:+.4f}" for v in profs[name]) +
              f"   amp {amps[name]:.6f} (null p95 {amp_p95[name]:.6f})")
    amp_ok = all(amps[n] > amp_p95[n] for n in ("h1", "h2"))
    print(f"  split-half profile corr: {pc:+.3f}   (gate >= +0.5)")
    print(f"  E2 gate: {'PASS' if pc >= 0.5 and amp_ok else 'FAIL'}"
          f"   (profile corr {'ok' if pc >= 0.5 else 'fail'}, "
          f"amplitude {'ok' if amp_ok else 'fail'})")


def stage_phasecheck() -> None:
    """§28.1: the falsification battery a double gate-pass owes before any claim."""
    rng = np.random.default_rng(2)
    G, e, ts = _frame_and_scores()
    Gd, ed, _ = _daily(G, e, ts)
    A = _antisym(Gd)
    u, w = _rot_plane(A)
    phi = np.arctan2(Gd @ w, Gd @ u)
    nd = len(Gd)
    hd = nd // 2
    phl = np.roll(phi, 1)[1:]  # phi(t-1), implementable
    day_codes = pd.factorize(ts.dt.date.to_numpy())[0]

    def ic(a: np.ndarray, b: np.ndarray) -> float:
        m = np.isfinite(a) & np.isfinite(b)
        a, b = a[m], b[m]
        a = (a - a.mean()) / (a.std() + 1e-12)
        b = (b - b.mean()) / (b.std() + 1e-12)
        return float((a * b).mean())

    def e1(res_daily: np.ndarray, label: str) -> None:
        z = (res_daily - np.nanmean(res_daily)) / np.nanstd(res_daily)
        z = z[1:]
        n1 = hd - 1
        harm = {"cos": np.cos(phl), "sin": np.sin(phl),
                "cos2": np.cos(2 * phl), "sin2": np.sin(2 * phl)}
        null = []
        for _ in range(200):
            zz = np.roll(z, int(rng.integers(63, len(z) - 63)))
            null.append(max(abs(ic(hv[s], zz[s])) for hv in harm.values()
                            for s in (slice(0, n1), slice(n1, None))))
        p95 = float(np.quantile(null, 0.95))
        print(f"{label} (lagged phase; shift-null p95 {p95:.4f}):")
        any_pass = False
        for name, hv in harm.items():
            i1, i2 = ic(hv[:n1], z[:n1]), ic(hv[n1:], z[n1:])
            hit = np.sign(i1) == np.sign(i2) and abs(i1) > p95 and abs(i2) > p95
            any_pass |= hit
            print(f"  {name:5s}: h1 {i1:+.4f}  h2 {i2:+.4f}   {'PASS' if hit else 'fail'}")
        print(f"  gate: {'PASS' if any_pass else 'FAIL'}")

    # (a) E1 at lag 1 vs HAR residual
    e1(ed, "(a) E1, HAR residual")

    # (a) E2 at lag 1, plus (b) the amplitude confound
    qi, qj = np.triu_indices(QPOOL)
    Q = G[:, qi] * G[:, qj]
    bmask1 = day_codes < hd
    m1, m2 = _map(Q, e, bmask1), _map(Q, e, ~bmask1)
    bins_l = np.clip(np.floor((np.roll(phi, 1) + np.pi) / (2 * np.pi / 8)).astype(int), 0, 7)
    e2d_all = pd.Series(e**2).groupby(day_codes).mean().to_numpy()
    profs, parts, amp, amp95, pamp, pamp95, e2profs = {}, {}, {}, {}, {}, {}, {}
    for name, bm, mo, bsl in (("h1", bmask1, m2, slice(1, hd)),
                              ("h2", ~bmask1, m1, slice(hd, None))):
        Qs = (Q[bm] - Q[bm].mean(0)) / (Q[bm].std(0) + 1e-12)
        ez = (e[bm] - e[bm].mean()) / e[bm].std()
        cd = pd.Series((Qs @ mo) * ez).groupby(day_codes[bm]).mean().to_numpy()
        if name == "h1":
            cd = cd[1:]  # drop day 0: its lagged phase wraps from the end
        bb = bins_l[bsl]
        e2d = e2d_all[bsl]
        beta = np.polyfit(e2d, cd, 1)
        part = cd - np.polyval(beta, e2d)

        def prof_of(v: np.ndarray) -> np.ndarray:
            return np.array([v[bb == k].mean() for k in range(8)])

        profs[name], parts[name] = prof_of(cd), prof_of(part)
        e2profs[name] = prof_of(e2d)
        amp[name], pamp[name] = float(profs[name].var()), float(parts[name].var())
        nv, pv = [], []
        for _ in range(200):
            off = int(rng.integers(63, len(cd) - 63))
            nv.append(prof_of(np.roll(cd, off)).var())
            pv.append(prof_of(np.roll(part, off)).var())
        amp95[name], pamp95[name] = float(np.quantile(nv, 0.95)), float(np.quantile(pv, 0.95))
    pc = float(np.corrcoef(profs["h1"], profs["h2"])[0, 1])
    ppc = float(np.corrcoef(parts["h1"], parts["h2"])[0, 1])
    print(f"\n(a) E2, lagged phase: split-half profile corr {pc:+.3f}")
    for name in ("h1", "h2"):
        print(f"  {name} gain profile: " + "  ".join(f"{v:+.4f}" for v in profs[name]) +
              f"   amp {amp[name]:.6f} (null p95 {amp95[name]:.6f})")
    a_ok = pc >= 0.5 and all(amp[n] > amp95[n] for n in ("h1", "h2"))
    print(f"  gate: {'PASS' if a_ok else 'FAIL'}")
    print("\n(b) amplitude confound:")
    for name in ("h1", "h2"):
        ce = float(np.corrcoef(profs[name], e2profs[name])[0, 1])
        print(f"  {name}: corr(gain profile, e^2 profile) {ce:+.3f}   "
              f"partial amp {pamp[name]:.6f} (null p95 {pamp95[name]:.6f}) "
              f"{'survives' if pamp[name] > pamp95[name] else 'DIES'}")
    print(f"  partial-profile split-half corr {ppc:+.3f}")
    b_ok = ppc >= 0.5 and all(pamp[n] > pamp95[n] for n in ("h1", "h2"))
    print(f"  verdict: {'phase structure beyond amplitude' if b_ok else 'AMPLITUDE CONFOUND'}")

    # (c) increment vs the final 679-column per-bar model
    fin = np.load(_p("final_onestage.npz"))["yhat_bar"]
    hr = np.load(_p("har_resid.npz"))
    y = hr["e"] + hr["pred"]
    ef = (y - fin)[TW:]
    ok = np.isfinite(ef)
    efd = pd.Series(np.where(ok, ef, np.nan)).groupby(day_codes).mean().to_numpy()
    good = np.isfinite(efd)
    print(f"\n(c) vs FINAL model residual ({good.sum()} of {nd} days usable):")
    e1(efd, "(c) E1, final-model residual")


def _causal_phase() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Daily phase of the cycle, built causally: flow on a trailing 504-day window of daily
    scores, refreshed quarterly, top eigenplane gauge-chained across refreshes, NaN in warmup.
    Returns (phi_day, day_codes for the G span, number of days)."""
    G, e, ts = _frame_and_scores()
    day_codes = pd.factorize(ts.dt.date.to_numpy())[0]
    day_last = np.flatnonzero(ts.dt.date.ne(ts.dt.date.shift(-1)).to_numpy())
    Gd = G[day_last]
    nd = len(Gd)
    TRAIL_D, REFRESH = 504, 63
    phi = np.full(nd, np.nan)
    v_prev = None
    for start in range(TRAIL_D, nd, REFRESH):
        A = _antisym(Gd[start - TRAIL_D : start])
        lam, V = np.linalg.eigh(1j * A)
        v = V[:, np.argmax(lam)]
        if v_prev is not None:
            v = v * np.exp(-1j * np.angle(np.vdot(v_prev, v)))
        v_prev = v
        seg = Gd[start : min(start + REFRESH, nd)]
        phi[start : min(start + REFRESH, nd)] = np.angle(
            seg @ np.real(v) + 1j * (seg @ np.imag(v)))
    return phi, day_codes, nd


def stage_phasewalk() -> None:
    """§28.2: the pre-registered feature arm — {cos, sin} of the causal lagged phase added to
    the §22 one-stage design at the exog penalty, daily refit, vs the cached 679-column twin."""
    from analysis.minimal_model import (CACHE, HOLDOUT, _qlike_series, _require_fixed_cache,
                                        _upper, dm_test, r2_oos)
    from analysis.wf import walk_forward
    _require_fixed_cache()

    p = load_panel()
    sig = dict(np.load(_p(CACHE)))
    frozen = sig["frozen"]
    har_cols = np.concatenate([p.cols("har"), p.cols("calendar"), p.cols("regime")])
    lin_cols = np.concatenate([p.cols("value"), p.cols("indicator")])
    bc, _ = base_columns(p)
    XH = np.ascontiguousarray(p.X[:, har_cols], dtype=np.float64)
    XL = np.ascontiguousarray(p.X[:, lin_cols], dtype=np.float64)
    XB = np.ascontiguousarray(p.X[:, bc], dtype=np.float64)
    XS = np.load(_p("xsec_features.npz"))["F"].astype(np.float64)
    ii, jj = _upper(XB.shape[1])
    P = XB[:, ii[frozen]] * XB[:, jj[frozen]]
    B = 250 * PERIODS_PER_DAY
    sd = pd.DataFrame(P).rolling(B, min_periods=1000).std().shift(1)
    med = np.nanmedian(sd.to_numpy(), axis=1, keepdims=True)
    sdv = np.maximum(sd.to_numpy(), 0.1 * np.where(np.isfinite(med), med, 1.0))
    P = P / pd.DataFrame(sdv).bfill().to_numpy()

    phi, day_codes, nd = _causal_phase()
    ph_lag = np.full(nd, np.nan)
    ph_lag[1:] = phi[:-1]
    n_all = p.X.shape[0]
    PH = np.zeros((n_all, 2))
    PH[2 * TW :, 0] = np.nan_to_num(np.cos(ph_lag))[day_codes]
    PH[2 * TW :, 1] = np.nan_to_num(np.sin(ph_lag))[day_codes]
    active_days = int(np.isfinite(ph_lag).sum())
    print(f"causal phase: {active_days} of {nd} days active "
          f"(504d warmup + quarterly refresh, gauge-chained)", flush=True)

    ALPHA = 3000.0
    X = np.hstack([XH * np.sqrt(ALPHA / 1.0), XL, XS, P * np.sqrt(ALPHA / 3e4), PH])
    print(f"phase arm: {X.shape[1]} cols (679 + 2 phase@3e3), daily refit", flush=True)
    yhat = walk_forward(X, p.y, TW, alpha=ALPHA, refit_every=PERIODS_PER_DAY)

    ref = np.load(_p("final_onestage.npz"))["yhat_daily"]
    y_adj = p.y[2 * TW :]
    baseline = p.baseline[2 * TW :]
    ts = pd.Series(pd.to_datetime(p.t[2 * TW :]))
    late = (ts >= HOLDOUT).to_numpy()
    act = PH[2 * TW :, 0] != 0.0
    yc = y_adj - y_adj.mean()
    f_new, f_ref = yhat[TW:], ref[TW:]
    q_new = _qlike_series(f_new, y_adj, baseline)
    q_ref = _qlike_series(f_ref, y_adj, baseline)
    d = q_ref - q_new
    print(f"\n  679-col twin : R2 {r2_oos(yc, f_ref - y_adj.mean()):+.5f}  "
          f"QLIKE {np.nanmean(q_ref):.5f}")
    print(f"  +phase (681) : R2 {r2_oos(yc, f_new - y_adj.mean()):+.5f}  "
          f"QLIKE {np.nanmean(q_new):.5f}")
    from analysis.minimal_model import _hac_mean_t
    print(f"\n  QLIKE DM (+phase vs twin), ACTIVE span ({act.sum()} bars): "
          f"{_hac_mean_t(d[act]):+.2f}   <- the gate (>= +2.0)")
    print(f"  QLIKE DM full span: {_hac_mean_t(d):+.2f}   2020+: {_hac_mean_t(d[late]):+.2f}")
    print(f"  sqrt DM active: {dm_test(y_adj[act], f_new[act], f_ref[act]):+.2f}")
    g = _hac_mean_t(d[act])
    print(f"\n  gate: {'PASS' if g >= 2.0 else 'FAIL'}")


def _causal_intraday_phase() -> tuple[np.ndarray, np.ndarray]:
    """§30.1: per-bar phase and radius of the intraday rotation plane, built causally —
    bar-lag flow on a trailing 504-day window, refreshed quarterly, gauge-chained (the §28.2
    recipe at bar resolution; no clock subtraction — measured share -0.003). NaN in warmup."""
    G, e, ts = _frame_and_scores()
    n = len(G)
    TRAIL, REFRESH = 504 * PERIODS_PER_DAY, 63 * PERIODS_PER_DAY

    def bar_flow(g: np.ndarray) -> np.ndarray:
        a, b = g[:-1], g[1:]
        a = (a - a.mean(0)) / (a.std(0) + 1e-12)
        b = (b - b.mean(0)) / (b.std(0) + 1e-12)
        C = (a.T @ b) / len(a)
        return (C - C.T) / 2.0

    phi = np.full(n, np.nan)
    rad = np.full(n, np.nan)
    v_prev = None
    for start in range(TRAIL, n, REFRESH):
        lam, V = np.linalg.eigh(1j * bar_flow(G[start - TRAIL : start]))
        v = V[:, np.argmax(lam)]
        if v_prev is not None:
            v = v * np.exp(-1j * np.angle(np.vdot(v_prev, v)))
        v_prev = v
        end = min(start + REFRESH, n)
        z = G[start:end] @ np.real(v) + 1j * (G[start:end] @ np.imag(v))
        phi[start:end] = np.angle(z)
        rad[start:end] = np.abs(z)
    return phi, rad


def stage_intraflow() -> None:
    """§30: the bar-lag lead-lag flow with the diurnal clock subtracted."""
    rng = np.random.default_rng(3)
    G, e, ts = _frame_and_scores()
    slot = (ts.dt.hour * 2 + ts.dt.minute // 30).to_numpy()
    Gd = G.copy()
    for s in np.unique(slot):
        m = slot == s
        Gd[m] = (G[m] - G[m].mean(0)) / (G[m].std(0) + 1e-12)
    n = len(Gd)
    iu = np.triu_indices(QPOOL, k=1)

    def flow(g: np.ndarray, k: int) -> np.ndarray:
        a, b = g[:-k], g[k:]
        a = (a - a.mean(0)) / (a.std(0) + 1e-12)
        b = (b - b.mean(0)) / (b.std(0) + 1e-12)
        C = (a.T @ b) / len(a)
        return (C - C.T) / 2.0

    h = n // 2
    daily_plane = None
    for k in (1, 2, 4, 8):
        A = flow(Gd, k)
        A1, A2 = flow(Gd[:h], k), flow(Gd[h:], k)
        raw_rep = float(np.corrcoef(A1[iu], A2[iu])[0, 1])
        Ac = np.mean([flow(Gd, k + 48 * int(m)) for m in rng.integers(20, 61, 12)], axis=0)
        clock_share = float(np.corrcoef(A[iu], Ac[iu])[0, 1])
        D1, D2 = A1 - Ac, A2 - Ac
        dyn_rep = float(np.corrcoef(D1[iu], D2[iu])[0, 1])
        print(f"k={k} bars: raw split-half {raw_rep:+.3f}   corr with clock {clock_share:+.3f}"
              f"   DYNAMIC split-half {dyn_rep:+.3f} {'PASS' if dyn_rep >= 0.5 else 'fail'}")
        if k == 1:
            D = A - Ac
            lam = np.linalg.eigvalsh(1j * D)
            print(f"  dynamic flow top-plane energy: "
                  f"{np.max(np.abs(lam))**2 / (lam**2).sum() * 2:.3f}")
            u, w = _rot_plane(D)
            v_intra = u + 1j * w
            day_last = np.flatnonzero(ts.dt.date.ne(ts.dt.date.shift(-1)).to_numpy())
            ud, wd = _rot_plane(_antisym(G[day_last]))
            v_day = ud + 1j * wd
            print(f"  overlap with DAILY circulation plane: "
                  f"{abs(np.vdot(v_day, v_intra)) / (np.linalg.norm(v_day) * np.linalg.norm(v_intra)):.3f}")
            hr = ts.dt.hour.to_numpy()
            rth = (hr >= 10) & (hr <= 16)
            Ar, Ao = flow(Gd[rth], 1), flow(Gd[~rth], 1)
            print(f"  RTH vs overnight lag-1 flow similarity: "
                  f"{np.corrcoef((Ar - Ac)[iu], (Ao - Ac)[iu])[0, 1]:+.3f}")
            top = np.argsort(-np.abs(D[iu]))[:5]
            print("  strongest dynamic edges (i leads j):")
            for t in top:
                i, j = iu[0][t], iu[1][t]
                print(f"    PC{i:2d} -> PC{j:2d}: {D[i, j]:+.4f}  h1 {D1[i, j]:+.4f}  "
                      f"h2 {D2[i, j]:+.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["hermitian", "sync", "sponge", "cpd", "rmt", "flow",
                                        "phase", "phasex", "phasecheck", "phasewalk",
                                        "intraflow"],
                    required=True)
    a = ap.parse_args()
    {"hermitian": stage_hermitian, "sync": stage_sync, "sponge": stage_sponge,
     "cpd": stage_cpd, "rmt": stage_rmt, "flow": stage_flow,
     "phase": stage_phase, "phasex": stage_phasex, "phasecheck": stage_phasecheck,
     "phasewalk": stage_phasewalk, "intraflow": stage_intraflow}[a.stage]()
