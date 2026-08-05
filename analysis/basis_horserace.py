"""Trial-space and penalty horse race for the lag profile.

Céa's lemma says least squares on a fixed basis IS the Gamma-orthogonal
projection, so all approximation error is the trial space. This tests every
trial space and penalty proposed:

  TRIAL SPACES (all are linear functionals of the past, applied by FFT
  convolution so that full-support bases cost the same as boxcars)
    boxcar_b2   nested rolling means at rungs 1,2,...,2048     -- P0, the incumbent
    bands       band_k = boxcar_k - boxcar_{k-1}               -- span-preserving
                reparameterisation; MUST give an identical OLS fit, which is the
                numerical check on the claim that it is not a model lever
    colloc      Gauss-Legendre nodes in log-lag, POINT lags    -- true collocation
    p1          continuous piecewise-linear hats in log-lag    -- P1 elements
    bspline     cubic B-splines in log-lag                     -- higher order
    chebyshev   Chebyshev polynomials in log-lag               -- spectral Galerkin
    midas       Beta weight function, 2 shape parameters       -- parametric

  PENALTIES (on the boxcar basis, matched otherwise)
    ridge       lambda * ||w||^2                     -- shrink toward zero
    rough       lambda * ||D2 phi_w||^2              -- shrink toward SMOOTH,
                where phi_w is the implied per-lag kernel on a log grid and D2 is
                a second-difference operator. This is the live descendant of the
                banded reparameterisation: the basis change was a non-lever, but
                the penalty it gestured at is not.

Scored by out-of-sample QLIKE on the raw variance scale, 60/40 split, identical
rows for every arm.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT_DIR = os.path.join(ROOT, "writeup", "stats")
U = 2048
BURN = 3125
TRAIN_FRAC = 0.60
LAMBDAS = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]


# ---------------------------------------------------------------- bases

def w_boxcar(rungs=None):
    rungs = rungs or [2 ** i for i in range(12)]
    B = np.zeros((len(rungs), U + 1))
    for k, r in enumerate(rungs):
        B[k, 1:r + 1] = 1.0 / r
    return B, [f"ma{r}" for r in rungs]


def w_bands():
    B, _ = w_boxcar()
    D = B.copy()
    D[1:] = B[1:] - B[:-1]
    return D, [f"band{k}" for k in range(len(B))]


def w_colloc(n):
    """Gauss-Legendre nodes in s = log u; features are POINT lags at those nodes."""
    s, _ = np.polynomial.legendre.leggauss(n)
    smax = np.log(U)
    nodes = np.unique(np.clip(np.round(np.exp((s + 1) / 2 * smax)), 1, U).astype(int))
    B = np.zeros((len(nodes), U + 1))
    for k, u in enumerate(nodes):
        B[k, u] = 1.0
    return B, [f"lag{u}" for u in nodes]


def _logx():
    u = np.arange(1, U + 1)
    return np.log(u) / np.log(U)          # in [0, 1]


def w_p1(K):
    """Continuous piecewise-linear hat functions on a uniform grid in log-lag."""
    x = _logx()
    knots = np.linspace(0, 1, K)
    h = knots[1] - knots[0]
    B = np.zeros((K, U + 1))
    for k, c in enumerate(knots):
        B[k, 1:] = np.maximum(0.0, 1.0 - np.abs(x - c) / h)
    return B, [f"p1_{k}" for k in range(K)]


def w_bspline(K, deg=3):
    """Cubic B-spline basis in log-lag via Cox-de Boor."""
    x = _logx()
    nint = max(K - deg, 1)
    inner = np.linspace(0, 1, nint + 1)[1:-1]
    t = np.concatenate([np.zeros(deg + 1), inner, np.ones(deg + 1)])
    nb = len(t) - deg - 1

    def basis(i, d, xx):
        if d == 0:
            return ((xx >= t[i]) & (xx < t[i + 1])).astype(float)
        a = np.zeros_like(xx)
        if t[i + d] > t[i]:
            a += (xx - t[i]) / (t[i + d] - t[i]) * basis(i, d - 1, xx)
        if t[i + d + 1] > t[i + 1]:
            a += (t[i + d + 1] - xx) / (t[i + d + 1] - t[i + 1]) * basis(i + 1, d - 1, xx)
        return a

    B = np.zeros((nb, U + 1))
    for i in range(nb):
        B[i, 1:] = basis(i, deg, x)
    B[-1, U] = 1.0                                   # close the right endpoint
    return B, [f"bs_{i}" for i in range(nb)]


def w_cheb(K):
    """Chebyshev polynomials in log-lag: the spectral Galerkin trial space."""
    x = 2 * _logx() - 1
    B = np.zeros((K, U + 1))
    for k in range(K):
        c = np.zeros(k + 1)
        c[k] = 1.0
        B[k, 1:] = np.polynomial.chebyshev.chebval(x, c)
    return B, [f"T{k}" for k in range(K)]


def w_midas(th1, th2):
    """Beta weight function; two shape parameters, one scale coefficient."""
    z = (np.arange(1, U + 1) - 0.5) / U
    with np.errstate(divide="ignore", invalid="ignore"):
        v = z ** (th1 - 1.0) * (1.0 - z) ** (th2 - 1.0)
    v = np.where(np.isfinite(v), v, 0.0)
    s = v.sum()
    if s <= 0:
        v = np.ones(U)
        s = float(U)
    B = np.zeros((1, U + 1))
    B[0, 1:] = v / s
    return B, ["midas"]


# ---------------------------------------------------------------- machinery

def featurize(y, B):
    """feature_k[t] = sum_u B[k,u] y[t-u], by overlap-add FFT convolution."""
    from scipy.signal import oaconvolve
    n = len(y)
    out = np.empty((n, B.shape[0]))
    for k in range(B.shape[0]):
        out[:, k] = oaconvolve(y, B[k])[:n]
    return out


def implied_kernel_map(B):
    """phi(u) = sum_k w_k B[k,u]; returns the (U x K) map from w to phi."""
    return B[:, 1:].T


def d2_operator(m):
    D = np.zeros((m - 2, m))
    for i in range(m - 2):
        D[i, i], D[i, i + 1], D[i, i + 2] = 1.0, -2.0, 1.0
    return D


def fit_eval(F, y, bl, rv, cut, lam, P=None):
    """Penalised least squares; P is the penalty matrix (None -> ridge)."""
    Ftr, ytr = F[:cut], y[:cut]
    mu, sd = Ftr.mean(0), Ftr.std(0)
    sd[sd < 1e-12] = 1.0
    Z = np.hstack([np.ones((len(ytr), 1)), (Ftr - mu) / sd])
    k = Z.shape[1]
    R = np.zeros((k, k))
    if P is None:
        R[1:, 1:] = np.eye(k - 1)
    else:
        R[1:, 1:] = P
    beta = np.linalg.solve(Z.T @ Z + lam * R, Z.T @ ytr)
    resid = ytr - Z @ beta
    sig2 = float(resid @ resid) / len(resid)
    Zte = np.hstack([np.ones((len(y) - cut, 1)), (F[cut:] - mu) / sd])
    pred = (Zte @ beta) ** 2 + sig2
    pred = pred * bl[cut:]
    t, p = rv[cut:], pred
    m = (t > 0) & (p > 0) & np.isfinite(t) & np.isfinite(p)
    r = t[m] / p[m]
    return float(np.mean(r - np.log(r) - 1.0)), beta, (mu, sd)


def valid_qlike(F, y, bl, rv, fit_rows, val_rows):
    """Fit OLS on fit_rows, score QLIKE on val_rows.

    The correct criterion for selecting a shape parameter: the SAME loss the
    result is reported under, on rows the selection has not touched. Training
    SSE is not a substitute -- it is squared error on the winsorised,
    diurnal-divided square-root target, a different objective on a different
    scale, and selecting under it costs +0.00157 here with no selection bias
    involved at all (analysis/selection_criterion.py).
    """
    A = np.hstack([np.ones((len(fit_rows), 1)), F[fit_rows]])
    cf, *_ = np.linalg.lstsq(A, y[fit_rows], rcond=None)
    r = y[fit_rows] - A @ cf
    sig2 = float(r @ r) / len(fit_rows)
    Av = np.hstack([np.ones((len(val_rows), 1)), F[val_rows]])
    p = ((Av @ cf) ** 2 + sig2) * bl[val_rows]
    t = rv[val_rows]
    m = (t > 0) & (p > 0) & np.isfinite(t) & np.isfinite(p)
    q = t[m] / p[m]
    return float(np.mean(q - np.log(q) - 1.0))


def best_over_lambda(F, y, bl, rv, cut, P=None):
    best = (np.inf, None)
    for lam in LAMBDAS:
        try:
            q, _, _ = fit_eval(F, y, bl, rv, cut, lam, P)
        except np.linalg.LinAlgError:
            continue
        if np.isfinite(q) and q < best[0]:
            best = (q, lam)
    return best


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
    cutpos = int(len(idx) * TRAIN_FRAC)
    print(f"panel {len(idx):,} rows, train {cutpos:,}  ({time.time()-t0:.0f}s)")

    SPECS = [("boxcar_b2", *w_boxcar()), ("bands", *w_bands())]
    for n in (4, 6, 8, 12):
        SPECS.append((f"colloc_gl{n}", *w_colloc(n)))
    for K in (6, 12):
        SPECS.append((f"p1_{K}", *w_p1(K)))
        SPECS.append((f"bspline_{K}", *w_bspline(K)))
    for K in (4, 6, 8, 12):
        SPECS.append((f"cheb_{K}", *w_cheb(K)))

    res = {}
    print(f"\n{'basis':16s}{'K':>4}{'lambda*':>10}{'OOS QLIKE':>12}")
    for name, B, cols in SPECS:
        F = featurize(ysafe, B)[idx]
        q, lam = best_over_lambda(F, y[idx], bl[idx], rv[idx], cutpos)
        res[name] = {"K": int(B.shape[0]), "lambda": lam, "oos_qlike": q,
                     "cols": cols}
        print(f"{name:16s}{B.shape[0]:>4}{lam:>10g}{q:>12.5f}")
        del F

    # ---- MIDAS: profile over the two Beta shape parameters ----
    #
    # The shape parameters are selected by QLIKE on a validation tail carved
    # out of the TRAINING portion: the same loss the row is reported under, on
    # rows the search has not seen. Two earlier versions got this wrong in
    # opposite directions -- first profiling theta against the reported
    # out-of-sample QLIKE (an oracle, optimism +0.00017 here), then
    # over-correcting to training SSE, which is a different objective on a
    # different scale and costs +0.00157 with no selection bias at all. See
    # analysis/selection_criterion.py for the decomposition.
    from scipy.optimize import minimize

    vcut = int(cutpos * 0.75)
    FITr, VALr = np.arange(vcut), np.arange(vcut, cutpos)

    def _midas_feats(v):
        th = np.exp(v)
        B, _ = w_midas(th[0], th[1])
        return th, featurize(ysafe, B)[idx]

    def midas_obj_valid(v):
        _, F = _midas_feats(v)
        q = valid_qlike(F, y[idx], bl[idx], rv[idx], FITr, VALr)
        del F
        return q if np.isfinite(q) else 1e6

    def midas_obj_oracle(v):
        _, F = _midas_feats(v)
        q, _ = best_over_lambda(F, y[idx], bl[idx], rv[idx], cutpos)
        del F
        return q if np.isfinite(q) else 1e6

    rt = minimize(midas_obj_valid, np.log([1.0, 3.0]), method="Nelder-Mead",
                  options={"maxiter": 90, "xatol": 1e-3, "fatol": 1e-6})
    tht, Ft = _midas_feats(rt.x)
    qt, lamt = best_over_lambda(Ft, y[idx], bl[idx], rv[idx], cutpos)
    del Ft
    res["midas_beta"] = {"K": 1, "theta1": float(tht[0]), "theta2": float(tht[1]),
                         "oos_qlike": float(qt), "lambda": lamt,
                         "theta_selected_on": "validation QLIKE (held-out "
                                              "tail of the training portion)"}
    print(f"{'midas_beta':16s}{1:>4}{lamt:>10g}{qt:>12.5f}   "
          f"theta=({tht[0]:.3f}, {tht[1]:.3f})  [theta by VALIDATION QLIKE]")

    ro = minimize(midas_obj_oracle, np.log([1.0, 3.0]), method="Nelder-Mead",
                  options={"maxiter": 80, "xatol": 1e-3, "fatol": 1e-7})
    tho = np.exp(ro.x)
    res["midas_beta_oracle"] = {"K": 1, "theta1": float(tho[0]),
                                "theta2": float(tho[1]),
                                "oos_qlike": float(ro.fun),
                                "theta_selected_on": "OOS QLIKE (oracle)",
                                "optimism": float(qt - ro.fun)}
    print(f"{'midas_beta*':16s}{1:>4}{'--':>10}{ro.fun:>12.5f}   "
          f"theta=({tho[0]:.3f}, {tho[1]:.3f})  [ORACLE: theta by OOS QLIKE, "
          f"optimism {qt - ro.fun:+.5f}]")

    # ---- span check: bands must reproduce boxcar exactly at lambda = 0 ----
    Fb = featurize(ysafe, w_boxcar()[0])[idx]
    Fd = featurize(ysafe, w_bands()[0])[idx]
    qb, _, _ = fit_eval(Fb, y[idx], bl[idx], rv[idx], cutpos, 0.0)
    qd, _, _ = fit_eval(Fd, y[idx], bl[idx], rv[idx], cutpos, 0.0)
    print(f"\n=== span check (OLS, lambda=0) ===")
    print(f"  boxcar {qb:.9f}   bands {qd:.9f}   |diff| {abs(qb-qd):.2e}")
    print("  -> span-preserving reparameterisation is a NON-LEVER under OLS"
          if abs(qb - qd) < 1e-9 else "  -> UNEXPECTED: spans differ")
    res["_span_check"] = {"boxcar_ols": qb, "bands_ols": qd,
                          "abs_diff": abs(qb - qd),
                          "identical": bool(abs(qb - qd) < 1e-9)}

    # ---- penalty: ridge vs roughness, on the boxcar basis ----
    print(f"\n=== penalty on the boxcar basis (same trial space) ===")
    Bx, _ = w_boxcar()
    L = implied_kernel_map(Bx)                      # (U, K): w -> phi
    sel = np.unique(np.round(np.exp(np.linspace(0, np.log(U), 64))).astype(int)) - 1
    Lg = L[sel]                                     # phi on a log grid
    D2 = d2_operator(len(sel))
    A = D2 @ Lg
    P_rough = A.T @ A
    P_rough = P_rough / max(np.trace(P_rough) / P_rough.shape[0], 1e-12)  # comparable scale
    qr, lr = best_over_lambda(Fb, y[idx], bl[idx], rv[idx], cutpos, None)
    qs, ls = best_over_lambda(Fb, y[idx], bl[idx], rv[idx], cutpos, P_rough)
    print(f"  ridge      lambda*={lr:<8g} OOS QLIKE {qr:.5f}")
    print(f"  roughness  lambda*={ls:<8g} OOS QLIKE {qs:.5f}")
    print(f"  roughness - ridge = {qs - qr:+.5f} "
          f"({'roughness WINS' if qs < qr else 'ridge wins'})")
    res["_penalty"] = {"ridge_qlike": qr, "ridge_lambda": lr,
                       "rough_qlike": qs, "rough_lambda": ls,
                       "delta": qs - qr}

    # roughness penalty on the best higher-order basis too
    bestho = min((k for k in res if k.startswith(("cheb", "bspline", "p1"))),
                 key=lambda k: res[k]["oos_qlike"])
    print(f"  (best higher-order trial space was {bestho} at "
          f"{res[bestho]['oos_qlike']:.5f})")

    with open(os.path.join(OUT_DIR, "basis_horserace.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nwrote basis_horserace.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
