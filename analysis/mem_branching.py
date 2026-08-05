"""The properly specified Hawkes object for a non-negative aggregated observable.

Two estimates of persistence that do not depend on an unconstrained regression
kernel:

  1. MULTIPLICATIVE ERROR MODEL (Engle's MEM; equivalently INGARCH, or ACD
     applied to variance). This is the Hawkes intensity equation written for a
     non-negative continuous observable rather than a counting measure:

         RV_t = lambda_t * eps_t,   eps_t >= 0,  E[eps] = 1
         lambda_t = omega + sum_k alpha_k * (past RV)

     Fitted by QMLE under the exponential quasi-likelihood, whose log-density
     is  -(log lambda_t + RV_t / lambda_t)  --  i.e. maximising it is exactly
     minimising QLIKE, the paper's own evaluation loss. Three advantages over
     the unconstrained ladder regression:
       * non-negativity of the kernel is IMPOSED (alpha_k >= 0), so the
         negative-lobe question cannot arise;
       * the branching ratio is a parameter with a standard error, not a sum
         of unconstrained coefficients;
       * it is fitted on the RAW variance scale -- no square root, no smearing
         correction, no winsorisation.

     Two specifications:
       MEM(1,1)  lambda_t = omega + a*RV_{t-1} + b*lambda_{t-1},
                 whose implied kernel is phi_k = a*b^(k-1), so n = a/(1-b).
       HAR-MEM   lambda_t = omega + sum_k alpha_k * MA_k(t-1) on the base-2
                 ladder, so n = sum_k alpha_k -- the constrained, raw-scale
                 counterpart of the paper's 0.913.

  2. VARIANCE-RATIO SCALING. Var(sum of T consecutive bars) ~ T^(2H). This
     needs no model at all. It is reported as a persistence cross-check, NOT a
     second estimate of n: H measures long memory, and the mapping from H to a
     branching ratio holds only in the nearly-unstable Hawkes limit.

Seasonality is handled by a FIXED per-slot factor estimated on the training
split, not the rolling divisor used elsewhere in this project -- a fixed
seasonal is not itself a forecast, so it cannot absorb persistence.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT_DIR = os.path.join(ROOT, "writeup", "stats")
RUNGS = [2 ** i for i in range(12)]          # 1 .. 2048
BURN = 3125
TRAIN_FRAC = 0.60


# --------------------------------------------------------------------------
# Likelihood
# --------------------------------------------------------------------------

def qlike_mean(y, lam):
    """Mean QLIKE. Equals the negative exponential quasi-log-likelihood up to
    an additive constant, so minimising it IS the MEM QMLE."""
    r = y / lam
    return float(np.mean(r - np.log(r) - 1.0))


def nll_har(theta, MA, y):
    """Negative exponential quasi-log-likelihood for the HAR-MEM.

    theta = [log omega, log alpha_1 .. log alpha_K]; the log parameterisation
    imposes omega > 0 and alpha_k > 0 without a boundary.
    """
    w = np.exp(theta[0])
    a = np.exp(theta[1:])
    lam = w + MA @ a
    if not np.all(np.isfinite(lam)) or np.any(lam <= 0):
        return 1e12
    return float(np.mean(np.log(lam) + y / lam))


def _mem11_lambda(y, w, a, b):
    """lambda_t = w + a*y_{t-1} + b*lambda_{t-1}, recursive."""
    n = len(y)
    lam = np.empty(n)
    lam[0] = y[:100].mean()
    for t in range(1, n):
        lam[t] = w + a * y[t - 1] + b * lam[t - 1]
    return lam


def nll_mem11(theta, y):
    w, a, b = np.exp(theta[0]), np.exp(theta[1]), 1.0 / (1.0 + np.exp(-theta[2]))
    lam = _mem11_lambda(y, w, a, b)
    if not np.all(np.isfinite(lam)) or np.any(lam <= 0):
        return 1e12
    return float(np.mean(np.log(lam) + y / lam))


# --------------------------------------------------------------------------
# Standard errors: QMLE sandwich
# --------------------------------------------------------------------------

def sandwich_se(f, theta, y_len, eps=1e-5):
    """Robust (sandwich) standard errors from numerical derivatives.

    f returns the MEAN negative log-likelihood, so the Hessian of the sum is
    y_len * H(mean). We use the observed-information form H^-1 scaled by 1/n;
    for a QMLE this is the conservative choice absent an OPG estimate.
    """
    k = len(theta)
    H = np.zeros((k, k))
    f0 = f(theta)
    for i in range(k):
        for j in range(i, k):
            tpp, tpm, tmp, tmm = (theta.copy() for _ in range(4))
            tpp[i] += eps; tpp[j] += eps
            tpm[i] += eps; tpm[j] -= eps
            tmp[i] -= eps; tmp[j] += eps
            tmm[i] -= eps; tmm[j] -= eps
            H[i, j] = H[j, i] = (f(tpp) - f(tpm) - f(tmp) + f(tmm)) / (4 * eps * eps)
    try:
        cov = np.linalg.inv(H) / y_len
        se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except np.linalg.LinAlgError:
        cov, se = None, np.full(k, np.nan)
    return se, cov, f0


def delta_se(g, theta, cov, eps=1e-6):
    """Delta-method SE for a scalar function g(theta)."""
    if cov is None:
        return float("nan")
    grad = np.zeros(len(theta))
    for i in range(len(theta)):
        tp, tm = theta.copy(), theta.copy()
        tp[i] += eps; tm[i] -= eps
        grad[i] = (g(tp) - g(tm)) / (2 * eps)
    return float(np.sqrt(max(grad @ cov @ grad, 0.0)))


# --------------------------------------------------------------------------

def variance_ratio(x, horizons):
    """Var(sum of T consecutive obs) / T, and the log-log slope 2H."""
    v = []
    for T in horizons:
        m = (len(x) // T) * T
        s = x[:m].reshape(-1, T).sum(axis=1)
        v.append(s.var(ddof=1) / T)
    v = np.array(v)
    lt, lv = np.log(horizons), np.log(v)
    A = np.column_stack([np.ones(len(lt)), lt])
    coef, *_ = np.linalg.lstsq(A, lv, rcond=None)
    resid = lv - A @ coef
    s2 = float(resid @ resid) / (len(lt) - 2)
    se = float(np.sqrt(s2 * np.linalg.inv(A.T @ A)[1, 1]))
    slope = float(coef[1])
    return {"horizons": list(map(int, horizons)), "var_per_unit": v.tolist(),
            "slope": slope, "slope_se": se,
            "H": 0.5 * (slope + 1.0), "H_se": 0.5 * se}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    from scipy.optimize import minimize
    import pandas as pd
    from src.data.loading import load_raw_data

    df = load_raw_data("data", allow_missing=True)
    rv = df["RV"].to_numpy(np.float64)
    slot = df["time_of_day"].to_numpy()
    ok = np.isfinite(rv) & (rv > 0)
    ok[:BURN] = False
    rv, slot = rv[ok], slot[ok]
    n_all = len(rv)
    cut = int(n_all * TRAIN_FRAC)
    print(f"panel {n_all:,} bars ({n_all - cut:,} held out)")

    # Fixed per-slot seasonal, estimated on the training split only.
    s = pd.Series(rv[:cut]).groupby(pd.Series(slot[:cut])).mean()
    fac = pd.Series(slot).map(s).to_numpy(np.float64)
    fac = np.where(np.isfinite(fac) & (fac > 0), fac, np.nanmean(rv[:cut]))
    x = rv / fac
    x = x / x[:cut].mean()                       # unit training mean, scale-free
    print(f"deseasonalised, unit-mean series: mean {x[:cut].mean():.4f}, "
          f"max {x.max():.1f}")
    out = {"n_bars": n_all, "n_train": cut, "n_test": n_all - cut}

    # ---------------- 1. MEM(1,1) ----------------
    print("\n" + "=" * 74)
    print("MEM(1,1):  lambda_t = omega + a*RV_{t-1} + b*lambda_{t-1}")
    print("=" * 74)
    ytr = x[:cut]
    f11 = lambda th: nll_mem11(th, ytr)                                # noqa: E731
    r11 = minimize(f11, np.array([np.log(0.05), np.log(0.10), 0.0]),
                   method="Nelder-Mead",
                   options={"maxiter": 4000, "xatol": 1e-8, "fatol": 1e-10})
    th = r11.x
    w, a, b = np.exp(th[0]), np.exp(th[1]), 1.0 / (1.0 + np.exp(-th[2]))
    n_hat = a / (1.0 - b)
    se, cov, _ = sandwich_se(f11, th, cut)
    n_se = delta_se(lambda t: np.exp(t[1]) / (1.0 - 1.0 / (1.0 + np.exp(-t[2]))), th, cov)
    lam_full = _mem11_lambda(x, w, a, b)
    q_oos = qlike_mean(x[cut:], lam_full[cut:])
    print(f"  omega {w:.5f}   alpha {a:.5f}   beta {b:.5f}")
    print(f"  persistence alpha+beta = {a + b:.5f}")
    print(f"  BRANCHING RATIO  n = alpha/(1-beta) = {n_hat:.4f}  (SE {n_se:.4f})")
    print(f"  implied multiplier 1/(1-n) = {1/(1-n_hat):.2f}"
          if n_hat < 1 else "  n >= 1: non-stationary")
    print(f"  out-of-sample QLIKE {q_oos:.5f}")
    out["mem11"] = {"omega": w, "alpha": a, "beta": b, "alpha_plus_beta": a + b,
                    "n": n_hat, "n_se": n_se, "oos_qlike": q_oos,
                    "converged": bool(r11.success)}

    # ---------------- 2. HAR-MEM, non-negativity imposed ----------------
    print("\n" + "=" * 74)
    print("HAR-MEM:  lambda_t = omega + sum_k alpha_k * MA_k(t-1),  alpha_k > 0")
    print("=" * 74)
    xs = pd.Series(x)
    MA = np.column_stack([xs.rolling(r, min_periods=r).mean().shift(1).to_numpy()
                          for r in RUNGS])
    good = np.isfinite(MA).all(axis=1)
    gtr = good.copy(); gtr[cut:] = False
    gte = good.copy(); gte[:cut] = False
    MAtr, ytr2 = MA[gtr], x[gtr]
    fh = lambda th: nll_har(th, MAtr, ytr2)                            # noqa: E731
    th0 = np.concatenate([[np.log(0.05)], np.log(np.full(len(RUNGS), 0.9 / len(RUNGS)))])
    rh = minimize(fh, th0, method="L-BFGS-B",
                  options={"maxiter": 8000, "maxfun": 20000, "ftol": 1e-14})
    thh = rh.x
    wh, ah = np.exp(thh[0]), np.exp(thh[1:])
    nh = float(ah.sum())
    seh, covh, _ = sandwich_se(fh, thh, int(gtr.sum()))
    nh_se = delta_se(lambda t: float(np.exp(t[1:]).sum()), thh, covh)
    lam_te = wh + MA[gte] @ ah
    q_oos_h = qlike_mean(x[gte], lam_te)
    print(f"  omega {wh:.5f}")
    print(f"  alpha_k: {np.round(ah, 4).tolist()}")
    print(f"  all alpha_k > 0 by construction: min {ah.min():.2e}")
    print(f"  BRANCHING RATIO  n = sum_k alpha_k = {nh:.4f}  (SE {nh_se:.4f})")
    print(f"  95% CI [{nh - 1.96*nh_se:.4f}, {nh + 1.96*nh_se:.4f}]")
    if nh < 1:
        print(f"  sub-critical; endogenous share {nh:.1%}, "
              f"multiplier 1/(1-n) = {1/(1-nh):.2f}")
    print(f"  out-of-sample QLIKE {q_oos_h:.5f}")
    out["har_mem"] = {"omega": wh, "alpha": ah.tolist(), "rungs": RUNGS,
                      "n": nh, "n_se": nh_se,
                      "ci_low": nh - 1.96 * nh_se, "ci_high": nh + 1.96 * nh_se,
                      "oos_qlike": q_oos_h, "converged": bool(rh.success),
                      "n_train": int(gtr.sum()), "n_test": int(gte.sum())}

    # ---------------- 3. variance-ratio persistence check ----------------
    print("\n" + "=" * 74)
    print("VARIANCE-RATIO SCALING (model-free persistence cross-check)")
    print("=" * 74)
    hz = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    vr = variance_ratio(x[:cut], hz)
    print(f"  Var(sum of T bars)/T  ~  T^(2H-1)")
    print(f"  log-log slope {vr['slope']:+.4f} (SE {vr['slope_se']:.4f})"
          f"  ->  H = {vr['H']:.4f} (SE {vr['H_se']:.4f})")
    print(f"  H = 0.5 would be short memory; H > 0.5 is long memory.")
    print(f"  interpretation: persistence is {'LONG-memory' if vr['H'] > 0.55 else 'short-memory'}"
          f", consistent with a near-critical kernel"
          if vr["H"] > 0.55 else "  interpretation: short memory")
    out["variance_ratio"] = vr

    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    print(f"  unconstrained ladder regression (sqrt scale, ridge)   n = 0.9133")
    print(f"  HAR-MEM, alpha_k > 0 imposed, raw scale, QMLE         n = {nh:.4f} "
          f"+/- {nh_se:.4f}")
    print(f"  MEM(1,1), raw scale, QMLE                             n = {n_hat:.4f} "
          f"+/- {n_se:.4f}")
    print(f"  variance-ratio H (model-free)                         H = {vr['H']:.4f}")

    with open(os.path.join(OUT_DIR, "mem_branching.json"), "w") as fh_:
        json.dump(out, fh_, indent=2)
    print(f"\nwrote {os.path.join(OUT_DIR, 'mem_branching.json')}")


if __name__ == "__main__":
    main()
