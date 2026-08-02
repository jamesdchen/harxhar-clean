"""Analog retrieval under a SUPERVISED metric.

The paper's headline negative result on the k-NN arm is that the Laplacian
eigenmap step hurts in all eight matched cells: retrieval in the ambient view
space beats retrieval in the embedded space, and the stated cause is Nystrom
extension error -- new points cannot be placed on the learned manifold
accurately enough.

That diagnosis leaves the interesting question unanswered, because the
embedding is wrong on TWO axes at once:

  1. WRONG DOMAIN. Laplacian eigenvectors are functions on the observation
     set. A test bar is not in that set, so it has to be extended in, and the
     extension error is the measured failure. The fitted operator's singular
     vectors instead live on the DESIGN axes -- channels and lags -- which are
     fixed, known a priori, finite, and require no extension at all.

  2. WRONG SUPERVISION. The eigenmap is built from X alone. In this panel the
     leading variance directions are nearly orthogonal to the predictable
     part of y (PC1 = 96.2% of X-variance, 0.02% of explained y; the signal
     sits in PCs 6-20 at 0.18% of variance). Any unsupervised compression
     therefore spends its budget on the wrong directions.

This script separates the two. Every arm uses the SAME anchor, the SAME
candidate pool, the SAME evaluation rows and the SAME neighbourhood sizes;
only the retrieval coordinate changes:

    ambient        the composed path+state view itself (d = 0 passthrough)
    ambient_path   the trailing-residual path alone (the paper's W-dim view)
    lap_d6         Laplacian eigenmaps, d = 6, kNN-Nystrom extension
    pca_d6         top-6 principal components -- linear AND unsupervised,
                   so it isolates "extension error" from "unsupervised"
    op_d6          SUPERVISED: [path score] + sigma_k (u_k' Z v_k), k = 1..5,
                   from the train-only fitted exogenous operator
    op_d12         the same with k = 1..11
    op_scores_d5   the five operator scores with no path score

If op_d6 beats ambient, the paper's negative result is about UNSUPERVISED
geometry, not about geometry. If it does not, the negative result is about
retrieval geometry as such, which is the stronger claim.

Protocol. Single 60/40 split of the rebuilt results/b2_mmap panel; the anchor,
the operator, the PCA frame, the eigenmap and every scaling constant are
estimated on training rows only. QLIKE is computed on raw-scale forecasts with
Duan smearing, exactly as the production drivers do, and matched-row Diebold-
Mariano tests are run against the ambient arm at each neighbourhood size.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference import dm_test, holm  # noqa: E402

D = os.path.join(ROOT, "results", "b2_mmap")
OUT_DIR = os.path.join(ROOT, "writeup", "stats")

TRAIN_FRAC = 0.60
POOL_STRIDE = 3          # candidate pool = every 3rd training row
TEST_STRIDE = 4          # evaluation rows = every 4th test row
KS = (25, 100, 4000, 8000)
KMAX = max(KS)
W_LIST = (12, 24)
LAP_ANCHORS = 8000       # rows the eigenmap itself is solved on
LAP_KGRAPH = 20
CHUNK = 512


# ---------------------------------------------------------------- panel

def load_panel():
    X = np.load(os.path.join(D, "X.npy"), mmap_mode="r")
    y = np.load(os.path.join(D, "y.npy"))
    b = np.load(os.path.join(D, "b.npy"))
    names = [str(s) for s in np.load(os.path.join(D, "names.npy"),
                                     allow_pickle=True)]
    chan: dict[str, list] = {}
    for j, nm in enumerate(names):
        m = re.match(r"^adj_(.+)_ma_(\d+)$", nm)
        if m:
            chan.setdefault(m.group(1), []).append((int(m.group(2)), j))
    for c in chan:
        chan[c].sort()
    labels = [c for c in chan if len(chan[c]) == 12]
    rungs = np.array([r for r, _ in chan[labels[0]]], float)
    ladder_cols = np.array([j for c in labels for _, j in chan[c]], int)
    har_cols = np.array([j for j, nm in enumerate(names)
                         if re.match(r"^har_ma_\d+$", nm)], int)
    return X, y, b, names, labels, rungs, ladder_cols, har_cols


def qlike_per_bar(pred_raw, true_raw):
    r = true_raw / pred_raw
    return r - np.log(r) - 1.0


# ---------------------------------------------------------------- retrieval

def knn_predict(Cpool, Ctest, epool, ks, chunk=CHUNK, tag=""):
    """Mean of the k nearest pool residuals, for every k in `ks` at once.

    One argpartition at KMAX per test chunk; the k-variants are prefix means of
    the distance-sorted neighbour list, so extra neighbourhood sizes are free.
    """
    P = np.asarray(Cpool, np.float32)
    Q = np.asarray(Ctest, np.float32)
    e = np.asarray(epool, np.float64)
    pn = (P * P).sum(1)
    kmax = min(max(ks), len(P))
    out = {k: np.empty(len(Q)) for k in ks}
    t0 = time.time()
    for s in range(0, len(Q), chunk):
        q = Q[s:s + chunk]
        d2 = pn[None, :] - 2.0 * (q @ P.T)          # ||q||^2 dropped: constant
        part = np.argpartition(d2, kmax - 1, axis=1)[:, :kmax]
        rows = np.arange(len(q))[:, None]
        order = np.argsort(np.take_along_axis(d2, part, 1), axis=1)
        nb = np.take_along_axis(part, order, 1)
        cs = np.cumsum(e[nb], axis=1)
        for k in ks:
            kk = min(k, kmax)
            out[k][s:s + len(q)] = cs[:, kk - 1] / kk
        del d2, part, order, nb, cs
    if tag:
        print(f"      [{tag}] retrieval {time.time() - t0:.0f}s", flush=True)
    return out


def laplacian_coords(Ppool, Ptest, d, rng):
    """Eigenmap on a pool subsample, kNN-Nystrom extension for everything else.

    Both pool and test points are extended the same way, so the arm is not
    handicapped by pool points having exact embeddings that test points lack.
    """
    from sklearn.manifold import SpectralEmbedding
    from sklearn.neighbors import NearestNeighbors

    m = min(LAP_ANCHORS, len(Ppool))
    anc = rng.choice(len(Ppool), m, replace=False)
    A = np.asarray(Ppool[anc], np.float64)
    se = SpectralEmbedding(n_components=d, affinity="nearest_neighbors",
                           n_neighbors=LAP_KGRAPH, random_state=0, n_jobs=1)
    phi = se.fit_transform(A)
    nn = NearestNeighbors(n_neighbors=LAP_KGRAPH).fit(A)

    def extend(Z):
        out = np.empty((len(Z), d))
        for s in range(0, len(Z), 4096):
            dist, idx = nn.kneighbors(np.asarray(Z[s:s + 4096], np.float64))
            sig = np.median(dist, axis=1, keepdims=True) + 1e-12
            w = np.exp(-(dist ** 2) / (2 * sig ** 2))
            w /= np.clip(w.sum(1, keepdims=True), 1e-12, None)
            out[s:s + 4096] = np.einsum("mk,mkd->md", w, phi[idx])
        return out

    return extend(Ppool), extend(Ptest)


# ---------------------------------------------------------------- main

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    from src.evaluation.metrics import apply_duan_smearing

    t00 = time.time()
    X, y, b, names, labels, rungs, ladder_cols, har_cols = load_panel()
    n = X.shape[0]
    cut = int(n * TRAIN_FRAC)
    print(f"panel {n:,} x {X.shape[1]}   train {cut:,}   "
          f"{len(labels)} channels x {len(rungs)} rungs")

    # ---- anchor: OLS on the HAR backbone, training rows only
    Xh = np.asarray(X[:, har_cols], np.float64)
    Atr = np.hstack([np.ones((cut, 1)), Xh[:cut]])
    cf = np.linalg.solve(Atr.T @ Atr, Atr.T @ y[:cut])
    anchor = np.hstack([np.ones((n, 1)), Xh]) @ cf
    e = y - anchor
    print(f"anchor: HAR-{len(har_cols)} OLS, train R2 = "
          f"{1 - e[:cut].var() / y[:cut].var():.4f}")

    # ---- exogenous operator on the anchor residual, training rows only
    Z = np.asarray(X[:, ladder_cols], np.float64)        # n x (41*12)
    mu_z, sd_z = Z[:cut].mean(0), Z[:cut].std(0)
    sd_z[sd_z < 1e-12] = 1.0
    Zs = (Z - mu_z) / sd_z
    del Z
    G = Zs[:cut].T @ Zs[:cut] + np.eye(Zs.shape[1])
    w_ex = np.linalg.solve(G, Zs[:cut].T @ e[:cut])
    Km = w_ex.reshape(len(labels), len(rungs))
    U_, sv, Vt = np.linalg.svd(Km, full_matrices=False)
    print(f"operator: 41x12 ridge on the anchor residual; "
          f"sv[:5] = {np.round(sv[:5], 4).tolist()}")

    # supervised scores: sigma_k * u_k' Z v_k, k = 1..11
    KOP = min(11, len(sv))
    S = np.empty((n, KOP))
    Zt = Zs.reshape(n, len(labels), len(rungs))
    for k in range(KOP):
        S[:, k] = sv[k] * np.einsum("c,ncr,r->n", U_[:, k], Zt, Vt[k])
    del Zt

    rng = np.random.default_rng(0)
    results = {}
    per_bar_store = {}

    for W in W_LIST:
        print("\n" + "=" * 78)
        print(f"W = {W}  trailing-residual path")
        print("=" * 78)
        lo = max(W, 1)
        pool_idx = np.arange(lo, cut, POOL_STRIDE)
        test_idx = np.arange(cut, n, TEST_STRIDE)
        # true_raw = y^2 * b, so a bar with y == 0 has an undefined QLIKE; drop
        # those rows ONCE, from a rule that does not look at any arm, so every
        # metric is scored on an identical row set.
        test_idx = test_idx[y[test_idx] != 0.0]
        print(f"  pool {len(pool_idx):,}   evaluation rows {len(test_idx):,}")

        # path views (uncentred: level gaps are the regime-analog signal)
        offs = np.arange(-W, 0)
        Vp_pool = e[pool_idx[:, None] + offs[None, :]]
        Vp_test = e[test_idx[:, None] + offs[None, :]]

        # supervised path direction, training rows only
        bp = np.linalg.lstsq(np.hstack([np.ones((len(pool_idx), 1)), Vp_pool]),
                             e[pool_idx], rcond=None)[0]
        sp_pool = Vp_pool @ bp[1:]
        sp_test = Vp_test @ bp[1:]

        # ambient composed view: per-column train standardisation, one global
        # sqrt(D) so weight falls per COORDINATE (the production convention)
        ps = e[lo:cut].std()
        zs_pool = Zs[pool_idx]
        zs_test = Zs[test_idx]
        Dtot = W + zs_pool.shape[1]
        g = np.sqrt(Dtot)
        Cpool_amb = np.hstack([Vp_pool / (ps * g), zs_pool / g])
        Ctest_amb = np.hstack([Vp_test / (ps * g), zs_test / g])

        coords = {
            "ambient": (Cpool_amb, Ctest_amb),
            "ambient_path": (Vp_pool / ps, Vp_test / ps),
            "op_d6": (np.column_stack([sp_pool, S[pool_idx, :5]]),
                      np.column_stack([sp_test, S[test_idx, :5]])),
            "op_d12": (np.column_stack([sp_pool, S[pool_idx, :11]]),
                       np.column_stack([sp_test, S[test_idx, :11]])),
            "op_scores_d5": (S[pool_idx, :5], S[test_idx, :5]),
        }

        # unsupervised linear control: top-6 PCs of the ambient view
        Cm = Cpool_amb.mean(0)
        _, _, Vp = np.linalg.svd(Cpool_amb[::4] - Cm, full_matrices=False)
        Vp6 = Vp[:6].T
        coords["pca_d6"] = ((Cpool_amb - Cm) @ Vp6, (Ctest_amb - Cm) @ Vp6)

        # the arm the paper measured
        t0 = time.time()
        lp, lt = laplacian_coords(Cpool_amb, Ctest_amb, 6, rng)
        coords["lap_d6"] = (lp, lt)
        print(f"  eigenmap fitted on {min(LAP_ANCHORS, len(pool_idx)):,} anchors "
              f"({time.time() - t0:.0f}s)")

        ytrue = y[test_idx]
        btrue = b[test_idx]
        anc_te = anchor[test_idx]
        ep = e[pool_idx]

        # reference: the anchor with no retrieval at all
        pr, tr = apply_duan_smearing(anc_te, ytrue, btrue)
        l_anchor = qlike_per_bar(pr, tr)
        print(f"\n  {'metric':<14}{'d':>4}" +
              "".join(f"{'k=' + str(k):>11}" for k in KS))
        print(f"  {'anchor only':<14}{'-':>4}" +
              "".join(f"{l_anchor.mean():>11.5f}" for _ in KS))

        wres = {"anchor_qlike": float(l_anchor.mean()),
                "n_pool": int(len(pool_idx)), "n_test": int(len(test_idx))}
        losses = {}
        for name, (Cp, Ct) in coords.items():
            preds = knn_predict(Cp, Ct, ep, KS,
                                tag=name if name == "ambient" else "")
            row = {}
            for k in KS:
                f = anc_te + preds[k]
                pr, tr = apply_duan_smearing(f, ytrue, btrue)
                lb = qlike_per_bar(pr, tr)
                losses[(name, k)] = lb
                row[str(k)] = float(lb.mean())
            print(f"  {name:<14}{Cp.shape[1]:>4}" +
                  "".join(f"{row[str(k)]:>11.5f}" for k in KS))
            wres[name] = {"d": int(Cp.shape[1]), "qlike": row}

        # ---- matched-row DM against the ambient arm
        print(f"\n  Diebold-Mariano versus the ambient arm (same rows, same k):")
        print(f"  {'metric':<14}" +
              "".join(f"{'k=' + str(k):>18}" for k in KS))
        pvals = {}
        dmres = {}
        for name in coords:
            if name == "ambient":
                continue
            cells = []
            for k in KS:
                r = dm_test(losses[(name, k)], losses[("ambient", k)], h=1)
                cells.append(f"{r['mean_diff']:+.5f} (t{r['t']:+.1f})")
                pvals[f"{name}@{k}"] = r["p"]
                dmres[f"{name}@{k}"] = {"mean_diff": r["mean_diff"],
                                        "t": r["t"], "p": r["p"]}
            print(f"  {name:<14}" + "".join(f"{c:>18}" for c in cells))
        print("  (negative = the arm BEATS ambient retrieval)")

        hm = holm(pvals)
        wres["dm_vs_ambient"] = dmres
        wres["holm"] = {k: {"adjusted_p": v[0], "reject": bool(v[1])}
                        for k, v in hm.items()}

        best_k = min(KS, key=lambda k: losses[("ambient", k)].mean())
        sup = losses[("op_d6", best_k)].mean()
        amb = losses[("ambient", best_k)].mean()
        lap = losses[("lap_d6", best_k)].mean()
        wres["verdict"] = {
            "best_k_for_ambient": best_k,
            "ambient": float(amb), "op_d6": float(sup), "lap_d6": float(lap),
            "supervised_beats_ambient": bool(sup < amb),
            "supervised_beats_laplacian": bool(sup < lap),
            "delta_sup_minus_ambient": float(sup - amb),
            "delta_lap_minus_ambient": float(lap - amb)}
        print(f"\n  At the ambient arm's own best k = {best_k}: "
              f"ambient {amb:.5f}, eigenmap {lap:.5f} ({lap - amb:+.5f}), "
              f"supervised {sup:.5f} ({sup - amb:+.5f})")
        results[f"W{W}"] = wres
        per_bar_store[f"W{W}"] = {f"{nm}_k{k}": losses[(nm, k)]
                                  for (nm, k) in losses}
        per_bar_store[f"W{W}"]["anchor"] = l_anchor
        del coords, Cpool_amb, Ctest_amb, losses

    out = {"protocol": {"train_frac": TRAIN_FRAC, "pool_stride": POOL_STRIDE,
                        "test_stride": TEST_STRIDE, "ks": list(KS),
                        "windows": list(W_LIST),
                        "lap_anchors": LAP_ANCHORS, "lap_kgraph": LAP_KGRAPH,
                        "operator_singular_values": sv[:5].tolist()},
           "results": results}
    with open(os.path.join(OUT_DIR, "supervised_metric_knn.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    np.savez_compressed(os.path.join(ROOT, "results",
                                     "supervised_metric_knn_losses.npz"),
                        **{f"{w}__{k}": v for w, dd in per_bar_store.items()
                           for k, v in dd.items()})
    print(f"\nwrote supervised_metric_knn.json  ({time.time() - t00:.0f}s)")


if __name__ == "__main__":
    main()
