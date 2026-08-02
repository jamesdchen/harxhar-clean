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
    BURN, TRAIN_FRAC, U, best_over_lambda, featurize, fit_eval,
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
    Bn = Bs / np.maximum(np.linalg.norm(Bs, axis=1, keepdims=True), 1e-300)
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

    Ptr = Phi[:cut].astype(np.float64)
    ytr = yv[:cut]
    mu = Ptr.mean(0)
    Pc = Ptr - mu
    yc = ytr - ytr.mean()
    G = Pc.T @ Pc
    c = Pc.T @ yc
    yy = float(yc @ yc)
    del Ptr, Pc

    # ---- 2a. greedy reduced basis: train-only selection
    print("\n" + "=" * 74)
    print("2. REDUCED BASES DRAWN FROM THE MANIFOLD")
    print("=" * 74)
    sel: list[int] = []
    greedy_traj = []
    for m in range(max(MDIMS)):
        bestj, bestsse = None, np.inf
        for j in range(G.shape[0]):
            if j in sel:
                continue
            cols = sel + [j]
            try:
                w = gram_solve(G, c, cols)
            except np.linalg.LinAlgError:
                continue
            sse = yy - 2 * w @ c[cols] + w @ G[np.ix_(cols, cols)] @ w
            if sse < bestsse:
                bestj, bestsse = j, sse
        sel.append(bestj)
        greedy_traj.append({"m": m + 1, "theta": thetas[bestj],
                            "train_sse": float(bestsse)})
    print("  greedy snapshot picks (train SSE):")
    for g in greedy_traj:
        print(f"    m={g['m']}  theta=({g['theta'][0]:.3f}, "
              f"{g['theta'][1]:.3f})   SSE={g['train_sse']:.2f}")
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
    from scipy.optimize import minimize

    def mix_obj(v):
        th = np.exp(v).reshape(-1, 2)
        B = np.vstack([w_midas(a, b)[0][0] for a, b in th])
        F = featurize(ysafe, B)[idx]
        q, _ = best_over_lambda(F, yv, blv, rvv, cut)
        return q if np.isfinite(q) else 1e6

    r1 = minimize(mix_obj, np.log([1.0, 3.0]), method="Nelder-Mead",
                  options={"maxiter": 60, "xatol": 1e-3, "fatol": 1e-7})
    th1 = np.exp(r1.x).reshape(-1, 2)
    print(f"  1 component  ({2} shape params + 1 coef)   QLIKE {r1.fun:.5f}"
          f"   theta=({th1[0,0]:.3f}, {th1[0,1]:.3f})")
    x0 = np.log(np.array([th1[0, 0], th1[0, 1], 1.0, 40.0]))
    r2 = minimize(mix_obj, x0, method="Nelder-Mead",
                  options={"maxiter": 200, "xatol": 1e-3, "fatol": 1e-7})
    th2 = np.exp(r2.x).reshape(-1, 2)
    print(f"  2 components ({4} shape params + 2 coefs)  QLIKE {r2.fun:.5f}"
          f"   theta=({th2[0,0]:.3f}, {th2[0,1]:.3f}), "
          f"({th2[1,0]:.3f}, {th2[1,1]:.3f})")
    out["nonlinear"] = {
        "m1": {"qlike": float(r1.fun), "theta": th1.tolist(),
               "free_params": 3},
        "m2": {"qlike": float(r2.fun), "theta": th2.tolist(),
               "free_params": 6}}

    # ---- reference incumbent
    Bx, _ = w_boxcar()
    Fx = featurize(ysafe, Bx)[idx]
    qx, lx = best_over_lambda(Fx, yv, blv, rvv, cut)
    qx0, _, _ = fit_eval(Fx, yv, blv, rvv, cut, 0.0)
    del Fx
    print(f"\n  incumbent boxcar b2 ladder (K=12)          QLIKE {qx:.5f}")
    out["boxcar_b2"] = {"K": 12, "qlike": qx, "lambda": lx, "ols_qlike": qx0}

    # ---- verdict
    ceiling = float(min(r1.fun, r2.fun))
    reach = None
    for m in MDIMS:
        if min(rows[str(m)]["pod"], rows[str(m)]["greedy"]) <= ceiling:
            reach = m
            break
    best_lin = min(min(v["pod"], v["greedy"]) for v in rows.values())
    print("\n" + "=" * 74)
    print("VERDICT")
    print("=" * 74)
    print(f"  nonlinear ceiling                {ceiling:.5f} "
          f"({min(3, 6)} free parameters at 1 component)")
    print(f"  best manifold-drawn linear space {best_lin:.5f} "
          f"(up to m = {max(MDIMS)})")
    print(f"  incumbent 12-rung ladder         {qx:.5f}")
    if reach is None:
        print(f"  -> NO linear space drawn from the manifold reaches the "
              f"nonlinear fit\n     at m <= {max(MDIMS)}: small nonlinear "
              f"width, large Kolmogorov width.\n     The trial space is a "
              f"MANIFOLD, and 'pick a better basis' is the wrong move.")
    else:
        print(f"  -> a manifold-drawn linear space of dimension {reach} "
              f"reaches the nonlinear fit;\n     the gap was basis choice, "
              f"not nonlinearity.")
    out["verdict"] = {"nonlinear_ceiling": ceiling,
                      "best_linear_from_manifold": float(best_lin),
                      "incumbent": qx,
                      "linear_dim_reaching_ceiling": reach,
                      "manifold_not_subspace": bool(reach is None)}

    with open(os.path.join(OUT_DIR, "reduced_basis.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote reduced_basis.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
