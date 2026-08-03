"""Does the rank-3 win survive refitting the basis, or does it need the freeze?

The headline result -- three supervised coefficients beating twelve rungs by
0.00059 -- used a projection estimated once on rows [0, W) and applied through
2024. Seventeen years stale at the end of the sample, and the largest
assumption in that design.

Freezing could be doing either of two things. It might be a harmless
simplification, in which case refitting changes nothing and the result is
stronger for it. Or the freeze might BE the result: this project has measured
estimated directions losing badly to imposed ones, with an in-window operator
SVD at 0.17112 against an imposed rung^-1 at 0.14878, because the estimated
leading direction flips -- consecutive cosine similarity averaging 0.893 but
bottoming out at 0.002. A basis refit into that instability can be worse than
a stale one, so this is not a formality.

Cadences swept: frozen, then refit every 24,000 bars (one training-window
turnover), 6,000 (about a quarter), and 1,000 (this project's usual refit
grid). All causal -- at a refit boundary the basis is rebuilt from the
TRAILING window [t-W, t) only, never from anything the arm will be scored on.

Changing the basis changes the design, so the Sherman-Morrison chain is
restarted at each boundary rather than carried across it, which would silently
mix inverses of different matrices.

Reported alongside the losses: the principal angles between consecutive bases.
A subspace that swings between refits is a different object from one that
drifts, and the loss alone will not distinguish them.
"""

from __future__ import annotations
import json, os, sys, time
import numpy as np
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import dm_test                       # noqa: E402
import src.features.transforms.target as TGT        # noqa: E402
from src.data.loading import load_raw_data          # noqa: E402

OUT_DIR = os.path.join(ROOT, "writeup", "stats")
CACHE = os.environ.get("LR_CACHE", "b2_mmap_warm")
W = 24000
REFRESH = 5000
NBLOCK = 8
RUNGS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
RANK = 3
CADENCES = [0, 24000, 6000, 1000]      # 0 = frozen, the published arm


def qlike_bar(rv, p):
    r = rv / p
    return r - np.log(r) - 1.0


def gram_schmidt(vs):
    out = []
    for v in vs:
        v = np.asarray(v, float).copy()
        for u in out:
            v -= (v @ u) * u
        nv = np.linalg.norm(v)
        if nv > 1e-8:
            out.append(v / nv)
    return out


def sup_basis(Htr, ytr, r):
    """Krylov/PLS subspace K_r(C, C^-1 xy) from one training window."""
    ok = np.isfinite(Htr).all(1) & np.isfinite(ytr)
    Htr, ytr = Htr[ok], ytr[ok]
    C = np.cov(Htr, rowvar=False) + 1e-12 * np.eye(Htr.shape[1])
    xy = (Htr - Htr.mean(0)).T @ (ytr - ytr.mean()) / len(ytr)
    d0 = np.linalg.solve(C, xy)
    fam = [d0, C @ d0]
    for _ in range(r):
        fam.append(C @ fam[-1])
    return np.column_stack(gram_schmidt(fam)[:r])


def build_target(df, lo, hi):
    o_lo, o_hi, o_w = TGT.WINSOR_LOWER_Q, TGT.WINSOR_UPPER_Q, TGT.rolling_winsorize
    try:
        if lo is None:
            TGT.rolling_winsorize = lambda s, **k: s
        else:
            TGT.WINSOR_LOWER_Q, TGT.WINSOR_UPPER_Q = lo, hi
        y, b = TGT.robust_transform(df, "RV", use_transform=True, use_diurnal=True,
                                    winsor_window=240, is_target=True)
    finally:
        TGT.WINSOR_LOWER_Q, TGT.WINSOR_UPPER_Q = o_lo, o_hi
        TGT.rolling_winsorize = o_w
    return y.to_numpy(np.float64), b.to_numpy(np.float64)


def walk(F, y, b, W, lo, hi, refresh=REFRESH):
    """Per-bar OLS by rank-2 inverse updates over [lo, hi)."""
    A = np.hstack([np.ones((len(F), 1)), F])
    pp = A.shape[1]
    out = np.full(hi - lo, np.nan)
    G = A[lo - W:lo].T @ A[lo - W:lo] + 1e-10 * np.eye(pp)
    c = A[lo - W:lo].T @ y[lo - W:lo]
    Ainv = np.linalg.inv(G)
    drift = 0.0
    for t in range(lo, hi):
        cf = Ainv @ c
        rr = y[t - W:t] - A[t - W:t] @ cf
        out[t - lo] = ((A[t] @ cf) ** 2 + float(rr @ rr) / W) * b[t]
        u = A[t - W]; v = A[t]
        Au = Ainv @ u
        Ainv += np.outer(Au, Au) / (1.0 - float(u @ Au))
        Av = Ainv @ v
        Ainv -= np.outer(Av, Av) / (1.0 + float(v @ Av))
        c = c - u * y[t - W] + v * y[t]
        if (t - lo + 1) % refresh == 0 and t + 1 < hi:
            s0 = t + 1 - W
            Ge = A[s0:s0 + W].T @ A[s0:s0 + W] + 1e-10 * np.eye(pp)
            Ae = np.linalg.inv(Ge)
            drift = max(drift, float(np.max(np.abs(Ainv - Ae))
                                     / max(1e-12, np.max(np.abs(Ae)))))
            Ainv, c = Ae, A[s0:s0 + W].T @ y[s0:s0 + W]
    return out, drift


def main():
    t0 = time.time()
    D = os.path.join(ROOT, "results", CACHE)
    Xc = np.load(os.path.join(D, "X.npy"), mmap_mode="r")
    names = [str(s) for s in np.load(os.path.join(D, "names.npy"), allow_pickle=True)]
    n = len(np.load(os.path.join(D, "y.npy")))
    CALN = ["DOW_0", "DOW_1", "DOW_2", "DOW_3", "DOW_4", "hour",
            "is_overnight", "is_open", "is_close"]
    cal = np.asarray(Xc[:, [names.index(c) for c in CALN if c in names]], np.float64)
    del Xc
    raw = load_raw_data(os.path.join(ROOT, "data"), allow_missing=True).dropna(
        subset=["RV"]).reset_index(drop=True)
    off = len(raw) - n
    y99, b99 = build_target(raw, 0.01, 0.99)
    y595, _ = build_target(raw, 0.05, 0.95)
    y = y99[off:off + n]; b = b99[off:off + n]
    truth = y595[off:off + n] ** 2 * b        # battery footing
    s = pd.Series(y ** 2)
    H = np.column_stack([
        np.sqrt(np.maximum(s.rolling(L, min_periods=L).mean().shift(1).to_numpy(), 0.0))
        for L in RUNGS])
    print(f"cache {CACHE}: {n:,} rows, clipped 5/95 truth, per-bar OLS", flush=True)

    preds, meta = {}, {}
    F0 = np.column_stack([H, cal])
    pr, dr = walk(np.where(np.isfinite(F0), F0, 0.0), y, b, W, W, n)
    pr[~np.isfinite(F0).all(1)[W:]] = np.nan
    preds["base-2 (r=12)"] = pr; meta["base-2 (r=12)"] = {"drift": dr, "nbasis": 0}
    print(f"    {'base-2 (r=12)':>22} drift {dr:.1e} ({time.time()-t0:.0f}s)", flush=True)

    for CAD in CADENCES:
        nm = "sup r=3 frozen" if CAD == 0 else f"sup r=3 refit/{CAD}"
        bounds = [W, n] if CAD == 0 else list(range(W, n, CAD)) + [n]
        out = np.full(n - W, np.nan); dr = 0.0; Gs = []
        for i in range(len(bounds) - 1):
            lo, hi = bounds[i], bounds[i + 1]
            Gm = sup_basis(H[lo - W:lo], y[lo - W:lo], RANK)
            Gs.append(Gm)
            F = np.column_stack([H @ Gm, cal])
            ok = np.isfinite(F).all(1)
            seg, d = walk(np.where(ok[:, None], F, 0.0), y, b, W, lo, hi)
            seg[~ok[lo:hi]] = np.nan
            out[lo - W:hi - W] = seg
            dr = max(dr, d)
        # principal angles between consecutive bases
        ang = []
        for i in range(1, len(Gs)):
            sv = np.linalg.svd(Gs[i - 1].T @ Gs[i], compute_uv=False)
            ang.append(float(np.degrees(np.arccos(np.clip(sv.min(), -1, 1)))))
        preds[nm] = out
        meta[nm] = {"drift": dr, "nbasis": len(Gs),
                    "max_angle_deg": max(ang) if ang else 0.0,
                    "mean_angle_deg": float(np.mean(ang)) if ang else 0.0}
        print(f"    {nm:>22} bases={len(Gs):<4d} drift {dr:.1e} "
              f"max principal angle {meta[nm]['max_angle_deg']:6.2f} deg "
              f"({time.time()-t0:.0f}s)", flush=True)

    tr = truth[W:]
    good = (tr > 0) & np.isfinite(tr)
    for k in preds: good &= np.isfinite(preds[k]) & (preds[k] > 0)
    idx = np.flatnonzero(good)
    L = {k: qlike_bar(tr[idx], preds[k][idx]) for k in preds}
    Q = {k: float(L[k].mean()) for k in L}
    REF = "base-2 (r=12)"
    print(f"\n  scored {len(idx):,} bars\n")
    print(f"  {'arm':>22}{'bases':>7}{'QLIKE':>11}{'vs base-2':>12}{'t':>8}"
          f"{'max angle':>11}")
    out = {"cache": CACHE, "n": int(len(idx)), "qlike": Q, "meta": meta, "dm": {}}
    nb = len(idx) // NBLOCK
    for nm in preds:
        m = meta[nm]
        if nm == REF:
            print(f"  {nm:>22}{'--':>7}{Q[nm]:>11.5f}{'--':>12}{'--':>8}{'--':>11}")
            continue
        r = dm_test(L[nm], L[REF], h=1)
        d = L[nm] - L[REF]
        em = [float(d[i*nb:(i+1)*nb if i < NBLOCK-1 else len(idx)].mean())
              for i in range(NBLOCK)]
        out["dm"][nm] = {"diff": float(r["mean_diff"]), "t": float(r["t"]),
                         "p": float(r["p"]), "eras_better": int(sum(v < 0 for v in em)),
                         "era_means": em}
        print(f"  {nm:>22}{m['nbasis']:>7}{Q[nm]:>11.5f}{r['mean_diff']:>+12.5f}"
              f"{r['t']:>8.2f}{m['max_angle_deg']:>11.2f}")
    print(f"\n  eras better than base-2 (of {NBLOCK}):")
    for nm in preds:
        if nm == REF: continue
        print(f"    {nm:>22}: {out['dm'][nm]['eras_better']}/{NBLOCK}")
    best = min((v, k) for k, v in Q.items())[1]
    print(f"\n  best: {best}")
    print(f"  a large principal angle between consecutive bases means the")
    print(f"  subspace SWINGS rather than drifts, which the loss alone hides.")
    with open(os.path.join(OUT_DIR, "ladder_refit.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote ladder_refit.json ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
