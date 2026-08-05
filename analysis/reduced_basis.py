"""Is the right trial space a subspace or a manifold? The reduced-basis test.

The basis horse race left one fact unexplained. Every LINEAR trial space of
dimension 12 landed in a narrow band -- boxcar b2 0.26492, bands 0.26492,
piecewise-linear 0.26466, cubic B-spline 0.26466, Chebyshev 0.26472 -- while a
single MIDAS Beta weight function, K = 1 with two nonlinear shape parameters,
reached 0.26340. One nonlinear degree of freedom beat twelve linear ones.

In approximation-theoretic language, that is the signature of a small NONLINEAR
width with a large Kolmogorov width: the solution sits on a low-dimensional
MANIFOLD M = {phi_theta} that no low-dimensional SUBSPACE approximates well.

The reduced-basis method is the standard response. If the manifold is the right
object, then the right linear space is not a generic polynomial or spline space
but the span of well-chosen SNAPSHOTS drawn from the manifold itself, selected
greedily. This script measures three things:

  1. THE WIDTH GAP. Singular value decay of the snapshot matrix, i.e. how fast
     the best possible linear space converges on M. If it decays slowly, no
     linear basis of any construction can be blamed.

  2. POD AND GREEDY REDUCED BASES. Linear spaces of dimension m = 1..8 drawn
     from M by proper orthogonal decomposition and by a train-only greedy
     search, scored on the same out-of-sample QLIKE as everything else.

  3. THE NONLINEAR CEILING. A two-component MIDAS mixture -- four shape
     parameters fitted jointly, coefficients profiled out -- which is the
     manifold used at dimension 2 in the nonlinear sense.

The question the design answers: does an m-dimensional space drawn FROM the
manifold beat a generic m-dimensional space, and does it close the gap to the
nonlinear fit? If the greedy basis needs many more dimensions than the
nonlinear fit needs parameters, the manifold reading is confirmed and the
paper's basis conclusion should be stated as a statement about widths.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from basis_horserace import (  # noqa: E402
    BURN, TRAIN_FRAC, U, best_over_lambda, featurize, fit_eval, valid_qlike,
    w_boxcar, w_bspline, w_midas,
)

OUT_DIR = os.path.join(ROOT, "writeup", "stats")
MDIMS = (1, 2, 3, 4, 6, 8)


def beta_grid(n1=16, n2=16):
    th1 = np.exp(np.linspace(np.log(0.02), np.log(6.0), n1))
    th2 = np.exp(np.linspace(np.log(0.5), np.log(400.0), n2))
    thetas, rows = [], []
    for a in th1:
        for c in th2:
            B, _ = w_midas(a, c)
            if not np.isfinite(B).all() or np.abs(B).sum() <= 0:
                continue
            thetas.append((float(a), float(c)))
            rows.append(B[0])
    return np.asarray(rows), thetas


def gram_solve(G, c, cols, ridge=1e-10):
    S = np.ix_(cols, cols)
    A = G[S] + ridge * np.eye(len(cols))
    return np.linalg.solve(A, c[cols])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    from src.data.loading import load_raw_data
    from src.features.transforms import target as T
    t0 = time.time()

    df = load_raw_data("data", allow_missing=True)
    T.WINSOR_LOWER_Q, T.WINSOR_UPPER_Q = 0.05, 0.95
    tgt, base = T.robust_transform(df, "RV", use_diurnal=True,
                                   allow_missing=True, is_target=True)
    y = tgt.to_numpy(np.float64)
    bl = base.to_numpy(np.float64)
    rv = df["RV"].to_numpy(np.float64)
    ok = np.isfinite(y) & np.isfinite(bl) & np.isfinite(rv) & (rv > 0)
    ok[:BURN] = False
    idx = np.flatnonzero(ok)
    ysafe = np.where(np.isfinite(y), y, 0.0)
    cut = int(len(idx) * TRAIN_FRAC)
    yv, blv, rvv = y[idx], bl[idx], rv[idx]
    print(f"panel {len(idx):,} rows, train {cut:,}  ({time.time()-t0:.0f}s)")
    out = {}

    # ---------------------------------------------------------- the manifold
    Bs, thetas = beta_grid()
    print(f"\nsnapshot set: {Bs.shape[0]} Beta weight functions on the "
          f"{U}-lag grid")

    # ---- 1. width gap: how well ANY linear space of dim m covers M
    # column 0 of a basis row is the u = 0 slot, which every kernel leaves at
    # zero; the manifold lives on columns 1..U, so the POD is taken there.
    Bk = Bs[:, 1:]
    Bn = Bk / np.maximum(np.linalg.norm(Bk, axis=1, keepdims=True), 1e-300)
    sv = np.linalg.svd(Bn, compute_uv=False)
    en = np.cumsum(sv ** 2) / np.sum(sv ** 2)
    print("\n" + "=" * 74)
    print("1. KOLMOGOROV WIDTH OF THE MIDAS MANIFOLD (snapshot POD spectrum)")
    print("=" * 74)
    print(f"  {'m':>3}{'sigma_m/sigma_1':>18}{'energy captured':>18}"
          f"{'worst-case rel. err':>22}")
    wid = {}
    for m in (1, 2, 3, 4, 6, 8, 12, 16):
        if m > len(sv):
            break
        err = float(np.sqrt(max(0.0, 1.0 - en[m - 1])))
        print(f"  {m:>3}{sv[m-1]/sv[0]:>18.3e}{en[m-1]:>18.6f}{err:>22.4f}")
        wid[str(m)] = {"sv_ratio": float(sv[m - 1] / sv[0]),
                       "energy": float(en[m - 1]), "rms_rel_err": err}
    out["pod_spectrum"] = wid
    print("  (energy is over the L2-normalised snapshot set: the m-term POD "
          "space's\n   mean squared relative error on the manifold is "
          "1 - energy)")

    # ---- features for every snapshot, once
    print(f"\nfeaturising {Bs.shape[0]} snapshots ...", flush=True)
    Phi = np.empty((len(idx), Bs.shape[0]), np.float32)
    for k in range(Bs.shape[0]):
        Phi[:, k] = featurize(ysafe, Bs[k:k + 1])[idx, 0]
    print(f"  Phi {Phi.shape} float32  ({time.time()-t0:.0f}s)", flush=True)

    vcut = int(cut * 0.75)
    FITr, VALr = np.arange(vcut), np.arange(vcut, cut)

    # ---- 2a. greedy reduced basis: selected on the validation tail
    #
    # The greedy criterion is the same loss the bases are reported under,
    # measured on held-out rows. An earlier version used training SSE, which
    # picks a systematically different (too short-memory) kernel: see the
    # theta trajectories in analysis/selection_criterion.py.
    print("\n" + "=" * 74)
    print("2. REDUCED BASES DRAWN FROM THE MANIFOLD")
    print("=" * 74)
    Pfit = Phi[FITr].astype(np.float64)
    Pval = Phi[VALr].astype(np.float64)
    yfit, yval = yv[FITr], yv[VALr]
    blval, rvval = blv[VALr], rvv[VALr]
    Afit1 = np.ones((len(FITr), 1))
    Aval1 = np.ones((len(VALr), 1))

    def greedy_score(cols):
        A = np.hstack([Afit1, Pfit[:, cols]])
        cf, *_ = np.linalg.lstsq(A, yfit, rcond=None)
        r_ = yfit - A @ cf
        sig2 = float(r_ @ r_) / len(r_)
        p = ((np.hstack([Aval1, Pval[:, cols]]) @ cf) ** 2 + sig2) * blval
        m = (rvval > 0) & (p > 0) & np.isfinite(p)
        q = rvval[m] / p[m]
        return float(np.mean(q - np.log(q) - 1.0))

    sel: list[int] = []
    greedy_traj = []
    for m in range(max(MDIMS)):
        bestj, bestq = None, np.inf
        for j in range(Phi.shape[1]):
            if j in sel:
                continue
            try:
                q = greedy_score(sel + [j])
            except np.linalg.LinAlgError:
                continue
            if np.isfinite(q) and q < bestq:
                bestj, bestq = j, q
        if bestj is None:
            break
        sel.append(bestj)
        greedy_traj.append({"m": m + 1, "theta": thetas[bestj],
                            "valid_qlike": float(bestq)})
    del Pfit, Pval
    print("  greedy snapshot picks (validation QLIKE):")
    for g in greedy_traj:
        print(f"    m={g['m']}  theta=({g['theta'][0]:.3f}, "
              f"{g['theta'][1]:.3f})   valid QLIKE={g['valid_qlike']:.5f}")
    out["greedy_picks"] = greedy_traj

    # ---- 2b. POD basis functions, featurised directly
    _, _, Vt = np.linalg.svd(Bn, full_matrices=False)

    print(f"\n  {'m':>3}{'POD span':>14}{'greedy span':>14}"
          f"{'generic bspline':>18}")
    rows = {}
    for m in MDIMS:
        Bp = np.zeros((m, U + 1))
        Bp[:, 1:] = Vt[:m]
        Fp = featurize(ysafe, Bp)[idx]
        qp, _ = best_over_lambda(Fp, yv, blv, rvv, cut)
        del Fp
        Fg = Phi[:, sel[:m]].astype(np.float64)
        qg, _ = best_over_lambda(Fg, yv, blv, rvv, cut)
        del Fg
        Bb, _ = w_bspline(max(m, 4))
        Fb = featurize(ysafe, Bb)[idx]
        qb, _ = best_over_lambda(Fb, yv, blv, rvv, cut)
        del Fb
        print(f"  {m:>3}{qp:>14.5f}{qg:>14.5f}{qb:>18.5f}")
        rows[str(m)] = {"pod": qp, "greedy": qg, "bspline": qb,
                        "bspline_K": int(Bb.shape[0])}
    out["reduced_bases"] = rows

    # ---- 3. the nonlinear ceiling: one- and two-component MIDAS
    print("\n" + "=" * 74)
    print("3. THE NONLINEAR CEILING (shape parameters fitted, not spanned)")
    print("=" * 74)
    # Shape parameters are selected by QLIKE on a validation tail held out of
    # the TRAINING portion -- the same loss the result is reported under, on
    # rows the search has not seen. Selecting against the reported
    # out-of-sample QLIKE is an oracle; selecting on training SSE is a
    # different objective on a different scale and costs more than the oracle
    # bias it was meant to remove. See analysis/selection_criterion.py.
    from scipy.optimize import minimize

    def _mix_feats(v):
        th = np.exp(v).reshape(-1, 2)
        B = np.vstack([w_midas(a, b)[0][0] for a, b in th])
        return th, featurize(ysafe, B)[idx]

    def mix_obj(v):
        _, F = _mix_feats(v)
        q = valid_qlike(F, yv, blv, rvv, FITr, VALr)
        del F
        return q if np.isfinite(q) else 1e6

    def score(v):
        th, F = _mix_feats(v)
        q, lam = best_over_lambda(F, yv, blv, rvv, cut)
        del F
        return th, float(q), lam

    r1 = minimize(mix_obj, np.log([1.0, 3.0]), method="Nelder-Mead",
                  options={"maxiter": 80, "xatol": 1e-3, "fatol": 1e-3})
    th1, q1, _ = score(r1.x)
    print(f"  1 component  ({2} shape params + 1 coef)   QLIKE {q1:.5f}"
          f"   theta=({th1[0,0]:.3f}, {th1[0,1]:.3f})")
    x0 = np.log(np.array([th1[0, 0], th1[0, 1], 1.0, 40.0]))
    r2 = minimize(mix_obj, x0, method="Nelder-Mead",
                  options={"maxiter": 250, "xatol": 1e-3, "fatol": 1e-3})
    th2, q2, _ = score(r2.x)
    print(f"  2 components ({4} shape params + 2 coefs)  QLIKE {q2:.5f}"
          f"   theta=({th2[0,0]:.3f}, {th2[0,1]:.3f}), "
          f"({th2[1,0]:.3f}, {th2[1,1]:.3f})")
    print("  (shape parameters selected by QLIKE on the held-out validation "
          "tail)")
    out["nonlinear"] = {
        "m1": {"qlike": q1, "theta": th1.tolist(), "free_params": 3},
        "m2": {"qlike": q2, "theta": th2.tolist(), "free_params": 6},
        "theta_selected_on": "training SSE"}

    # ---- reference incumbent
    Bx, _ = w_boxcar()
    Fx = featurize(ysafe, Bx)[idx]
    qx, lx = best_over_lambda(Fx, yv, blv, rvv, cut)
    qx0, _, _ = fit_eval(Fx, yv, blv, rvv, cut, 0.0)
    del Fx
    print(f"\n  incumbent boxcar b2 ladder (K=12)          QLIKE {qx:.5f}")
    out["boxcar_b2"] = {"K": 12, "qlike": qx, "lambda": lx, "ols_qlike": qx0}

    # ---- verdict
    ceiling = float(min(q1, q2))
    reach = None
    for m in MDIMS:
        if min(rows[str(m)]["pod"], rows[str(m)]["greedy"]) <= ceiling:
            reach = m
            break
    best_lin = min(min(v["pod"], v["greedy"]) for v in rows.values())
    print("\n" + "=" * 74)
    print("VERDICT")
    print("=" * 74)
    print(f"  nonlinear fit, 1 component       {ceiling:.5f} (3 free params)")
    print(f"  best manifold-drawn linear space {best_lin:.5f} "
          f"(up to m = {max(MDIMS)})")
    print(f"  incumbent 12-rung ladder         {qx:.5f}")
    spread = max([qx, ceiling, best_lin]) - min([qx, ceiling, best_lin])
    print(f"\n  full spread across every trial space tested: {spread:.5f}")
    print("\n  Note on what this can and cannot show. A 1-component MIDAS fit"
          "\n  IS a one-dimensional linear span drawn from this manifold, so"
          "\n  'a linear space from the manifold cannot reach the nonlinear"
          "\n  fit' is not a statement about approximation widths -- the span"
          "\n  at the same theta reaches it by construction, and any gap is"
          "\n  the selection rule. Kolmogorov width versus nonlinear width is"
          "\n  a claim about covering a FAMILY of solutions with one fixed"
          "\n  space; this panel has one target and one split, hence one"
          "\n  solution, so the distinction has nothing to bite on here."
          "\n  Testing it needs a real parameter axis -- session bins,"
          "\n  regimes, rolling windows, assets -- and a space held fixed"
          "\n  across it.")
    if spread < 0.005:
        print(f"\n  -> what the run DOES establish: every trial space tested,"
              f"\n     linear or nonlinear, generic or manifold-drawn, lands"
              f"\n     within {spread:.5f} QLIKE. The trial space is a"
              f"\n     NON-LEVER, consistent with the span-preserving"
              f"\n     reparameterisation result.")
    out["verdict"] = {"nonlinear_1component": ceiling,
                      "best_linear_from_manifold": float(best_lin),
                      "incumbent": qx,
                      "linear_dim_reaching_nonlinear": reach,
                      "full_spread": float(spread),
                      "trial_space_is_a_nonlever": bool(spread < 0.005),
                      "width_claim_not_testable_here": True}

    with open(os.path.join(OUT_DIR, "reduced_basis.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote reduced_basis.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
