"""The exogenous operator as a 3-tensor, plus signed structure in the loadings.

The operator is channels x lags x sessions (41 x 12 x 6). Earlier work analysed
it as six separate matrices and compared frames by cosine similarity. Two
things follow from treating it jointly:

  1. STATIC GEOMETRY AS A MODEL, NOT AN EYEBALL. A CP/PARAFAC decomposition
     T ~ sum_r lambda_r (a_r x b_r x c_r) with session-varying weights c_r on
     FIXED channel factors a_r and lag factors b_r *is* the hypothesis "the
     geometry is static and only amplitudes move". Fitting it and comparing to
     a full Tucker model, and to six independent per-session SVDs, turns the
     claim into a model comparison at matched parameter count.

  2. SIGNED STRUCTURE. Roughly half the cross-kernel cells are negative, so the
     channel-similarity matrix built from the loadings is a SIGNED graph.
     Ordinary spectral clustering on |A| discards the sign, which is exactly the
     information distinguishing channels that co-move from channels that
     oppose. Signed-graph clustering in the SPONGE family (Cucuringu et al.)
     is built for this case; we compare it against the unsigned baseline.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT_DIR = os.path.join(ROOT, "writeup", "stats")
SESSIONS = ["overnight", "early", "open", "midday", "close", "after"]


# ---------------------------------------------------------------- tensor ops

def unfold(T, mode):
    return np.moveaxis(T, mode, 0).reshape(T.shape[mode], -1)


def khatri_rao(A, B):
    r = A.shape[1]
    return (A[:, None, :] * B[None, :, :]).reshape(-1, r)


def cp_als(T, R, iters=600, seed=0, tol=1e-11):
    """CP/PARAFAC by alternating least squares."""
    rng = np.random.default_rng(seed)
    F = [rng.standard_normal((d, R)) for d in T.shape]
    nrm = np.linalg.norm(T)
    prev = np.inf
    for _ in range(iters):
        for m in range(3):
            others = [F[i] for i in range(3) if i != m]
            KR = khatri_rao(others[0], others[1]) if m == 0 else \
                 khatri_rao(others[0], others[1])
            # correct Khatri-Rao ordering for the chosen unfolding
            idx = [i for i in range(3) if i != m]
            KR = khatri_rao(F[idx[0]], F[idx[1]])
            G = (F[idx[0]].T @ F[idx[0]]) * (F[idx[1]].T @ F[idx[1]])
            F[m] = np.linalg.solve(G + 1e-10 * np.eye(R),
                                   (unfold(T, m) @ KR).T).T
        rec = reconstruct_cp(F)
        err = np.linalg.norm(T - rec) / nrm
        if abs(prev - err) < tol:
            break
        prev = err
    return F, float(err)


def reconstruct_cp(F):
    A, B, C = F
    R = A.shape[1]
    out = np.zeros((A.shape[0], B.shape[0], C.shape[0]))
    for r in range(R):
        out += A[:, r][:, None, None] * B[:, r][None, :, None] * C[:, r][None, None, :]
    return out


def hosvd(T, ranks):
    """Tucker via higher-order SVD (truncated), then core projection."""
    Us = []
    for m in range(3):
        U, _, _ = np.linalg.svd(unfold(T, m), full_matrices=False)
        Us.append(U[:, :ranks[m]])
    core = T.copy()
    for m in range(3):
        core = np.moveaxis(np.tensordot(Us[m].T, np.moveaxis(core, m, 0), axes=(1, 0)), 0, m)
    rec = core.copy()
    for m in range(3):
        rec = np.moveaxis(np.tensordot(Us[m], np.moveaxis(rec, m, 0), axes=(1, 0)), 0, m)
    return Us, core, float(np.linalg.norm(T - rec) / np.linalg.norm(T))


def per_session_svd_err(T, r):
    """Six independent rank-r SVDs -- the analysis actually used."""
    num = 0.0
    for s in range(T.shape[2]):
        M = T[:, :, s]
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        rec = (U[:, :r] * S[:r]) @ Vt[:r]
        num += np.linalg.norm(M - rec) ** 2
    return float(np.sqrt(num) / np.linalg.norm(T))


# ---------------------------------------------------------------- signed graph

def sponge(Apos, Aneg, k, tau=1.0):
    """SPONGE-style signed clustering: generalised eigenproblem
    (L_pos + tau * D_neg) x = lambda (L_neg + tau * D_pos) x, smallest eigvecs."""
    dp, dn = Apos.sum(1), Aneg.sum(1)
    Lp, Ln = np.diag(dp) - Apos, np.diag(dn) - Aneg
    Aм = Lp + tau * np.diag(dn)
    Bм = Ln + tau * np.diag(dp) + 1e-9 * np.eye(len(dp))
    import scipy.linalg as sla
    vals, vecs = sla.eigh(Aм, Bм)
    return vecs[:, :k]


def kmeans(X, k, seed=0, iters=200):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)]
    lab = np.zeros(len(X), int)
    for _ in range(iters):
        d = ((X[:, None, :] - C[None]) ** 2).sum(-1)
        new = d.argmin(1)
        if (new == lab).all():
            break
        lab = new
        for j in range(k):
            if (lab == j).any():
                C[j] = X[lab == j].mean(0)
    return lab


def signed_cut_ratio(A, lab):
    """Fraction of within-cluster edge weight that is NEGATIVE (lower is better)
    plus fraction of between-cluster weight that is POSITIVE."""
    same = lab[:, None] == lab[None, :]
    np.fill_diagonal(same, False)
    win, bet = A[same], A[~same]
    bad = (np.abs(win[win < 0]).sum() + bet[bet > 0].sum())
    tot = np.abs(A).sum() - np.abs(np.diag(A)).sum()
    return float(bad / max(tot, 1e-12))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    T = np.stack([np.load(os.path.join(ROOT, "results", f"sesskern_{s}.npy"))
                  for s in SESSIONS], axis=2)
    print(f"operator tensor: {T.shape}  (channels x lags x sessions)")
    out = {"shape": list(T.shape), "sessions": SESSIONS}

    print("\n" + "=" * 74)
    print("STATIC GEOMETRY AS A MODEL COMPARISON")
    print("=" * 74)
    print(f"{'model':34s}{'params':>8}{'rel.err':>10}  interpretation")
    rows = []
    for R in (1, 2, 3, 4, 5, 6, 8):
        F, err = cp_als(T, R)
        p = R * sum(T.shape)
        rows.append(("CP rank %d" % R, p, err))
        print(f"{'CP rank %d' % R:34s}{p:>8}{err:>10.4f}  "
              f"{'fixed geometry, session weights' if R <= 6 else ''}")
        out[f"cp_r{R}"] = {"params": p, "rel_err": err,
                           "session_weights": F[2].tolist()}
    for r in (3, 5):
        rk = (r, r, min(6, T.shape[2]))
        _, core, err = hosvd(T, rk)
        p = rk[0] * T.shape[0] + rk[1] * T.shape[1] + rk[2] * T.shape[2] + np.prod(rk)
        print(f"{'Tucker (%d,%d,%d)' % rk:34s}{int(p):>8}{err:>10.4f}  "
              f"full mode interaction")
        out[f"tucker_{r}"] = {"params": int(p), "rel_err": err}
    for r in (3, 5):
        err = per_session_svd_err(T, r)
        p = 6 * r * (T.shape[0] + T.shape[1])
        print(f"{'6x independent rank-%d SVD' % r:34s}{p:>8}{err:>10.4f}  "
              f"the analysis actually used")
        out[f"persession_r{r}"] = {"params": p, "rel_err": err}

    cp5 = out["cp_r5"]["rel_err"]
    ps5 = out["persession_r5"]["rel_err"]
    print(f"\n  CP-5 uses {out['cp_r5']['params']} params for rel.err {cp5:.4f};")
    print(f"  six independent rank-5 SVDs use {out['persession_r5']['params']} "
          f"for {ps5:.4f}.")
    print(f"  ratio of params {out['persession_r5']['params']/out['cp_r5']['params']:.1f}x, "
          f"error penalty {cp5-ps5:+.4f}")
    print("  -> " + ("STATIC GEOMETRY SUPPORTED: near-equal fit at a fraction of the "
                     "parameters" if cp5 - ps5 < 0.10 else
                     "static geometry NOT supported: CP pays a large error penalty"))
    out["static_geometry_verdict"] = {
        "cp5_err": cp5, "persession5_err": ps5, "delta": cp5 - ps5,
        "param_ratio": out["persession_r5"]["params"] / out["cp_r5"]["params"],
        "supported": bool(cp5 - ps5 < 0.10)}

    print("\n  CP-5 session weights (rows = component, cols = session):")
    C = np.array(out["cp_r5"]["session_weights"])
    print("    " + "  ".join(f"{s[:5]:>7}" for s in SESSIONS))
    for r in range(C.shape[1]):
        print(f"  c{r+1}" + "  ".join(f"{v:>7.3f}" for v in C[:, r]))

    # ------------------------------------------------------------------
    print("\n" + "=" * 74)
    print("SIGNED STRUCTURE IN THE CHANNEL LOADINGS")
    print("=" * 74)
    M = T.reshape(T.shape[0], -1)                       # channels x (lags*sessions)
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    L = U[:, :5] * S[:5]
    A = L @ L.T                                          # signed channel similarity
    np.fill_diagonal(A, 0.0)
    frac_neg = float((A < 0).mean())
    print(f"  channel-similarity matrix: {frac_neg:.1%} of off-diagonal entries "
          f"are NEGATIVE")
    Apos, Aneg = np.maximum(A, 0), np.maximum(-A, 0)
    res = {"frac_negative": frac_neg}
    print(f"\n{'k':>3}{'unsigned |A|':>15}{'SPONGE signed':>16}  winner")
    for k in (2, 3, 4, 5):
        Uu, _, _ = np.linalg.svd(np.abs(A))
        lab_u = kmeans(Uu[:, :k], k)
        lab_s = kmeans(sponge(Apos, Aneg, k), k)
        cu, cs = signed_cut_ratio(A, lab_u), signed_cut_ratio(A, lab_s)
        print(f"{k:>3}{cu:>15.4f}{cs:>16.4f}  "
              f"{'SPONGE' if cs < cu else 'unsigned'}")
        res[f"k{k}"] = {"unsigned_badcut": cu, "sponge_badcut": cs,
                        "sponge_better": bool(cs < cu)}
    out["signed_clustering"] = res
    nwin = sum(1 for k in (2, 3, 4, 5) if res[f"k{k}"]["sponge_better"])
    print(f"\n  SPONGE beats the unsigned baseline on {nwin}/4 cluster counts")
    print("  -> the sign carries clustering information that |A| discards"
          if nwin >= 3 else
          "  -> sign structure does not help clustering here")

    with open(os.path.join(OUT_DIR, "tensor_operator.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote tensor_operator.json")


if __name__ == "__main__":
    main()
