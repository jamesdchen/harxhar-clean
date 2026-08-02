"""Frequency-domain duality, and the data-matrix / coefficient-matrix duality.

Two tests that need no new modelling machinery.

  1. LOCAL WHITTLE. The MEM put the branching ratio at 0.9917 with a 95%
     interval whose upper end touches 1.0001 -- i.e. the constrained estimator
     cannot separate "near-critical" from "critical". The frequency domain
     settles this differently: a near-unit-root / long-memory process has
     spectral density f(lambda) ~ lambda^(-2d) near zero, and the local Whittle
     estimator of d uses only the m lowest Fourier frequencies, so it is
     semiparametric in the short-run dynamics. d = 1/2 is the stationarity
     boundary. Estimating d with a standard error tests criticality on an axis
     the time-domain fit cannot.

  2. GRAM SPECTRUM VERSUS SIGNAL. X'X (coefficient space) and XX' (data space)
     share nonzero eigenvalues, so "the covariance concentrates but the signal
     is elsewhere" has a dual reading in OBSERVATION space: a few typical market
     states dominate the Gram spectrum, but y is not aligned with them. This
     projects y onto the leading principal directions and compares, component by
     component, the share of X-variance against the share of explained y.
     If the predictable variation sits in low-variance directions, that is the
     dense-but-weak fact stated in its dual form.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT_DIR = os.path.join(ROOT, "writeup", "stats")
BURN = 3125


# ---------------------------------------------------------------- Whittle

def local_whittle(x, m):
    """Robinson's local Whittle estimator of the memory parameter d.

    R(d) = log( (1/m) sum_j lambda_j^(2d) I_j ) - (2d/m) sum_j log lambda_j
    minimised over d; asymptotic SE is 1/(2 sqrt(m)).
    """
    x = np.asarray(x, float)
    x = x - x.mean()
    n = len(x)
    I = np.abs(np.fft.rfft(x)) ** 2 / (2 * np.pi * n)
    lam = 2 * np.pi * np.arange(len(I)) / n
    I, lam = I[1:m + 1], lam[1:m + 1]
    loglam = np.log(lam)

    def R(d):
        g = np.mean(lam ** (2 * d) * I)
        return np.log(max(g, 1e-300)) - 2 * d * loglam.mean()

    grid = np.linspace(-0.49, 1.49, 401)
    vals = np.array([R(d) for d in grid])
    d0 = grid[np.argmin(vals)]
    # golden refine
    lo, hi = d0 - 0.01, d0 + 0.01
    for _ in range(80):
        a, b = lo + 0.382 * (hi - lo), lo + 0.618 * (hi - lo)
        if R(a) < R(b):
            hi = b
        else:
            lo = a
    d = 0.5 * (lo + hi)
    return float(d), float(1.0 / (2 * np.sqrt(m)))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    import pandas as pd
    from src.data.loading import load_raw_data
    out = {}

    # ------------------------------------------------ local Whittle
    print("=" * 74)
    print("LOCAL WHITTLE: the memory parameter d")
    print("=" * 74)
    df = load_raw_data("data", allow_missing=True)
    rv = df["RV"].to_numpy(np.float64)
    slot = df["time_of_day"].to_numpy()
    ok = np.isfinite(rv) & (rv > 0)
    ok[:BURN] = False
    rv, slot = rv[ok], slot[ok]
    s = pd.Series(rv).groupby(pd.Series(slot)).mean()
    x = rv / pd.Series(slot).map(s).to_numpy()          # fixed seasonal removed
    n = len(x)
    print(f"  deseasonalised RV, n = {n:,}")

    res = {}
    for series_name, ser in (("RV", x), ("log RV", np.log(np.maximum(x, 1e-300)))):
        print(f"\n  {series_name}:")
        print(f"    {'m':>8}{'m/n':>8}{'d':>9}{'SE':>8}  95% CI          verdict")
        rr = {}
        for frac in (0.02, 0.05, 0.10, 0.20):
            m = int(n ** 0.5) if frac is None else int(n * frac)
            m = min(m, n // 2 - 1)
            d, se = local_whittle(ser, m)
            lo, hi = d - 1.96 * se, d + 1.96 * se
            verdict = ("nonstationary (d>=0.5)" if lo > 0.5 else
                       "stationary long memory" if hi < 0.5 else
                       "straddles the d=0.5 boundary")
            print(f"    {m:>8}{frac:>8.2f}{d:>9.4f}{se:>8.4f}  "
                  f"[{lo:.3f}, {hi:.3f}]  {verdict}")
            rr[f"m_{frac}"] = {"m": m, "d": d, "se": se, "ci": [lo, hi],
                               "verdict": verdict}
        res[series_name] = rr
    out["local_whittle"] = res

    dl = res["log RV"]["m_0.05"]["d"]
    print(f"\n  Reading: d = {dl:.3f} on log RV at m/n = 0.05.")
    print(f"    d in (0, 0.5)  -> stationary long memory")
    print(f"    d >= 0.5       -> nonstationary; the MEM's n -> 1 would be the")
    print(f"                      time-domain image of the same fact")

    # ------------------------------------------------ Gram spectrum vs signal
    print("\n" + "=" * 74)
    print("GRAM SPECTRUM VERSUS SIGNAL (the data/coefficient duality)")
    print("=" * 74)
    D = os.path.join(ROOT, "results", "b2_mmap")
    if not os.path.exists(os.path.join(D, "X.npy")):
        print("  b2_mmap not present; skipping")
    else:
        X = np.load(os.path.join(D, "X.npy"), mmap_mode="r")
        y = np.load(os.path.join(D, "y.npy"))
        sub = np.arange(0, X.shape[0], 4)               # every 4th row is plenty
        Xs = np.asarray(X[sub], dtype=np.float64)
        ys = y[sub]
        Xs -= Xs.mean(0)
        ys = ys - ys.mean()
        print(f"  subsample {Xs.shape}")
        C = Xs.T @ Xs / len(Xs)
        ev, V = np.linalg.eigh(C)
        order = np.argsort(ev)[::-1]
        ev, V = ev[order], V[:, order]
        Z = Xs @ V
        zvar = Z.var(0)
        r2 = np.array([float(np.corrcoef(Z[:, j], ys)[0, 1] ** 2)
                       for j in range(Z.shape[1])])
        varshare = zvar / zvar.sum()
        r2share = r2 / max(r2.sum(), 1e-300)
        print(f"\n  {'PC block':<14}{'var share':>12}{'R2 share':>12}"
              f"{'ratio':>10}")
        blocks = [(0, 1), (1, 5), (5, 20), (20, 50), (50, 200),
                  (200, Z.shape[1])]
        bl = []
        for a, b in blocks:
            vs, rs = varshare[a:b].sum(), r2share[a:b].sum()
            print(f"  {'PC %d-%d' % (a + 1, b):<14}{vs:>12.4f}{rs:>12.4f}"
                  f"{rs / max(vs, 1e-12):>10.2f}")
            bl.append({"block": [a + 1, b], "var_share": float(vs),
                       "r2_share": float(rs), "ratio": float(rs / max(vs, 1e-12))})
        top1 = varshare[0]
        print(f"\n  PC1 carries {top1:.1%} of X-variance and {r2share[0]:.1%} of "
              f"explained y.")
        k50 = int(np.searchsorted(np.cumsum(varshare), 0.50) + 1)
        r2_in_k50 = r2share[:k50].sum()
        print(f"  The first {k50} PCs hold 50% of X-variance and "
              f"{r2_in_k50:.1%} of explained y.")
        print("  -> " + ("signal is NOT in the leading variance directions "
                         "(dense-but-weak, dual form)" if r2_in_k50 < 0.5 else
                         "signal IS concentrated with the variance"))
        out["gram_vs_signal"] = {
            "blocks": bl, "pc1_var_share": float(top1),
            "pc1_r2_share": float(r2share[0]), "k_for_50pct_var": k50,
            "r2_share_in_those": float(r2_in_k50),
            "signal_off_leading_pcs": bool(r2_in_k50 < 0.5)}

    with open(os.path.join(OUT_DIR, "frequency_and_gram.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote frequency_and_gram.json")


if __name__ == "__main__":
    main()
