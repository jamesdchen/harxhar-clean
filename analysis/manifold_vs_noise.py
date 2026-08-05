"""Is the refit subspace SWINGING because the structure is a curved manifold,
or because two of its three directions are re-estimated noise?

Both stories predict "refit bases disagree with each other". They differ on the
FINE structure of the disagreement, and that is what this measures.

  MANIFOLD (dim 3, curved).  The local tangent space rotates smoothly as the
  data walks along it. Then:
    (M1) principal angles grow with the time-lag between two refits -- adjacent
         windows share almost the same tangent, distant ones do not;
    (M2) the rotation is a function of local STATE, so it should be at least
         partly predictable from the window's own vol level;
    (M3) destroying the y-signal while keeping the ladder covariance intact
         must destroy the rotation structure, because a tangent space to the
         regression surface cannot exist without a regression.

  NOISE (1 real direction + 2 amplified low-variance contrasts).  C^-1 rescales
  the seed by 1/lambda, so directions carrying <1% of the ladder's variance get
  their sampling error handed back as basis vectors. Then:
    (N1) angles are FLAT in lag -- independent draws have no autocorrelation;
    (N2) no state dependence;
    (N3) the y-shuffled placebo reproduces the rotation, because the rotation
         is driven by C's tail and the sampling error in xy, not by signal.

The placebo permutes y WITHIN each training window. C is untouched by
construction (it never sees y); only the xy cross-moment is destroyed. So any
angle structure that survives the shuffle is not about the regression surface.

PLS (seed = xy, no inversion) is carried as the control: it never touches the
tail, so under either story it should be the stable arm.
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

from ladder_refit import (RUNGS, W, build_target, gram_schmidt,  # noqa: E402
                          load_raw_data)

CACHE = os.environ.get("RANK_CACHE", "b2_mmap_warm")
CAD = 6000
RANK = 3
NPERM = 20
SEED = 20260803
OUT = os.path.join(ROOT, "writeup", "stats", "manifold_vs_noise.json")


def basis(Htr, ytr, r, kind):
    """Byte-for-byte the seed logic of ladder_refit.sup_basis."""
    ok = np.isfinite(Htr).all(1) & np.isfinite(ytr)
    Htr, ytr = Htr[ok], ytr[ok]
    C = np.cov(Htr, rowvar=False) + 1e-12 * np.eye(Htr.shape[1])
    xy = (Htr - Htr.mean(0)).T @ (ytr - ytr.mean()) / len(ytr)
    if kind == "pls":
        seed = xy
    elif kind == "ridge":
        lam = float(np.trace(C)) / C.shape[0] * 0.1
        seed = np.linalg.solve(C + lam * np.eye(C.shape[0]), xy)
    else:
        seed = np.linalg.solve(C, xy)
    fam = [seed, C @ seed]
    for _ in range(r):
        fam.append(C @ fam[-1])
    return np.column_stack(gram_schmidt(fam)[:r])


def angles(G1, G2):
    """All RANK principal angles, ascending, in degrees."""
    sv = np.linalg.svd(G1.T @ G2, compute_uv=False)
    return np.degrees(np.arccos(np.clip(sv, -1.0, 1.0)))


def lag_profile(Gs):
    """mean principal angle by |i-j|, per direction and pooled."""
    m = len(Gs)
    acc = {}
    for i in range(m):
        for j in range(i + 1, m):
            acc.setdefault(j - i, []).append(angles(Gs[i], Gs[j]))
    return {L: np.mean(np.array(v), axis=0) for L, v in sorted(acc.items())}


def main():
    t0 = time.time()
    D = os.path.join(ROOT, "results", CACHE)
    n = len(np.load(os.path.join(D, "y.npy")))
    raw = load_raw_data(os.path.join(ROOT, "data"), allow_missing=True).dropna(
        subset=["RV"]).reset_index(drop=True)
    off = len(raw) - n
    y99, _ = build_target(raw, 0.01, 0.99)
    y = y99[off:off + n]
    s = pd.Series(y ** 2)
    H = np.column_stack([
        np.sqrt(np.maximum(
            s.rolling(L, min_periods=L).mean().shift(1).to_numpy(), 0.0))
        for L in RUNGS])
    bounds = list(range(W, n, CAD)) + [n]
    wins = [(bounds[i] - W, bounds[i]) for i in range(len(bounds) - 1)]
    print(f"cache {CACHE}: {n:,} rows, {len(wins)} refit windows, "
          f"rank {RANK}", flush=True)

    out = {"cache": CACHE, "n": int(n), "cad": CAD, "rank": RANK,
           "n_windows": len(wins), "nperm": NPERM}

    # ---- real bases, per estimator
    real = {}
    for kind in ("sup", "ridge", "pls"):
        Gs = [basis(H[a:b], y[a:b], RANK, kind) for a, b in wins]
        real[kind] = Gs
        prof = lag_profile(Gs)
        Ls = sorted(prof)
        out[f"{kind}_lag_profile"] = {str(L): prof[L].tolist() for L in Ls}
        print(f"\n  {kind}: mean principal angle (deg) by refit lag")
        print(f"    {'lag':>5}{'dir1':>9}{'dir2':>9}{'dir3':>9}{'pooled':>9}")
        for L in Ls:
            if L in (1, 2, 4, 8, 16, Ls[-1]):
                a = prof[L]
                print(f"    {L:>5}{a[0]:>9.2f}{a[1]:>9.2f}{a[2]:>9.2f}"
                      f"{a.mean():>9.2f}")
        # M1/N1 discriminator: slope of pooled angle in log-lag
        x = np.log(np.array(Ls, float))
        pooled = np.array([prof[L].mean() for L in Ls])
        A = np.column_stack([np.ones_like(x), x])
        beta = np.linalg.lstsq(A, pooled, rcond=None)[0]
        rho = float(np.corrcoef(x, pooled)[0, 1])
        out[f"{kind}_lag_slope_deg_per_ln_lag"] = float(beta[1])
        out[f"{kind}_lag_corr"] = rho
        out[f"{kind}_lag1_pooled"] = float(prof[Ls[0]].mean())
        out[f"{kind}_lagmax_pooled"] = float(prof[Ls[-1]].mean())
        print(f"    slope {beta[1]:+.2f} deg per ln(lag)   corr(lag, angle) "
              f"{rho:+.3f}   [manifold: strongly +, noise: ~0]")

    # ---- M3/N3: y-shuffled placebo, C held exactly fixed
    print(f"\n  placebo: y permuted within each window, {NPERM} draws "
          f"({time.time()-t0:.0f}s)", flush=True)
    rng = np.random.default_rng(SEED)
    for kind in ("sup", "pls"):
        lag1, lagmax, allang = [], [], []
        for _ in range(NPERM):
            Gs = []
            for a, b in wins:
                yy = y[a:b].copy()
                fin = np.isfinite(yy)
                idx = np.flatnonzero(fin)
                yy[idx] = yy[rng.permutation(idx)]
                Gs.append(basis(H[a:b], yy, RANK, kind))
            prof = lag_profile(Gs)
            Ls = sorted(prof)
            lag1.append(prof[Ls[0]].mean())
            lagmax.append(prof[Ls[-1]].mean())
            allang.append(np.mean([prof[L] for L in Ls], axis=0))
        A = np.array(allang)
        out[f"{kind}_placebo_lag1"] = [float(np.mean(lag1)),
                                       float(np.std(lag1))]
        out[f"{kind}_placebo_mean_per_dir"] = A.mean(0).tolist()
        rl = np.mean([np.array(v) for v in
                      lag_profile(real[kind]).values()], axis=0)
        out[f"{kind}_real_mean_per_dir"] = rl.tolist()
        print(f"    {kind:>6} real   mean angle per dir "
              f"[{rl[0]:6.2f} {rl[1]:6.2f} {rl[2]:6.2f}]")
        print(f"    {kind:>6} shuffled                   "
              f"[{A.mean(0)[0]:6.2f} {A.mean(0)[1]:6.2f} "
              f"{A.mean(0)[2]:6.2f}]  +/- "
              f"[{A.std(0)[0]:.2f} {A.std(0)[1]:.2f} {A.std(0)[2]:.2f}]")
        z = (rl - A.mean(0)) / np.maximum(A.std(0), 1e-9)
        out[f"{kind}_placebo_z_per_dir"] = z.tolist()
        print(f"    {kind:>6} z(real vs shuffled)        "
              f"[{z[0]:+6.2f} {z[1]:+6.2f} {z[2]:+6.2f}]"
              f"   [manifold: real << shuffled]")

    # ---- M2/N2: is the rotation a function of local state?
    print(f"\n  state dependence ({time.time()-t0:.0f}s)", flush=True)
    lvl = np.array([np.nanmean(H[a:b, 0]) for a, b in wins])
    for kind in ("sup", "pls"):
        Gs = real[kind]
        d1 = np.array([G[:, 0] for G in Gs])
        # angle of each refit's direction-k to the GRAND direction, vs level
        rows = []
        for k in range(RANK):
            Vk = np.array([G[:, k] for G in Gs])
            u, _, _ = np.linalg.svd(Vk.T, full_matrices=False)
            g = u[:, 0]
            ang = np.degrees(np.arccos(np.clip(np.abs(Vk @ g), 0, 1)))
            rows.append(float(np.corrcoef(lvl, ang)[0, 1]))
        out[f"{kind}_state_corr_per_dir"] = rows
        print(f"    {kind:>6} corr(window vol level, angle-to-grand-dir) = "
              f"[{rows[0]:+.3f} {rows[1]:+.3f} {rows[2]:+.3f}]"
              f"   [manifold: |corr| large]")
        del d1

    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {os.path.relpath(OUT, ROOT)}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
