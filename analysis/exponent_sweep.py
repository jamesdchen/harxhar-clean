"""Sweep and fit the amplitude exponent properly, on corrected data.

The claim "the usable range is roughly [-1, -0.5] and the exponent is a design
choice needing validation" rested on three grid points (0.5, 1.0, 1.5),
measured in the PENALTY context rather than the feature context, on the cache
that is now known to have carried an imputation defect worth 6.37 QLIKE on the
affected bars. None of those three footings survives, so the quantity has never
actually been measured.

It also matters more now than when it was asserted. On corrected data plain
ridge on the 492 raw columns reaches 0.13177 against 0.13320 for the
53-column amplitude design -- a gap of 0.0014. Whether a properly chosen
exponent closes it decides whether the compact model is worth anything.

Four arms, all on the full walk-forward over the corrected cache:

  fixed beta        a_c = sum_k u_k^-beta Z_{c,k}, swept finely over beta,
                    which gives the actual curve rather than three points and
                    shows whether the optimum is sharp or flat
  fitted beta       beta chosen inside each training window by profiling the
                    in-window fit, so the exponent is estimated causally
                    rather than assumed
  fitted shape      a shared 12-vector g estimated on each training window by
                    least squares -- 12 parameters instead of one, and the
                    honest "let the data choose the shape" arm. Whether this
                    beats a fixed exponent is the same estimated-versus-imposed
                    question the operator work asked, now on corrected data
                    where its earlier answer no longer stands
  references        the backbone, and plain ridge on the 492 raw columns

Per-era optima are reported alongside the pooled one: an exponent that is
optimal on the panel but wanders across eras is a fitted constant, not a
structural one, and should be described as such.
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

from inference import dm_test  # noqa: E402

D = os.path.join(ROOT, "results", "b2_mmap_fix")
OUT_DIR = os.path.join(ROOT, "writeup", "stats")
W, CAD = 24000, 1000
LAM_AMP, LAM_RAW = 1e2, 1e4
BETAS = np.round(np.arange(-0.5, 3.01, 0.1), 2)
NBLOCK = 8


def qlike_bar(rv, p):
    r = rv / p
    return r - np.log(r) - 1.0


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    import pandas as pd
    from src.data.loading import load_raw_data
    t0 = time.time()

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
    lad = np.array([j for c in labels for _, j in chan[c]], int)
    har = np.array([j for j, nm in enumerate(names)
                    if re.match(r"^har_ma_\d+$", nm)], int)
    C, R, nh = len(labels), 12, len(har)
    n = len(y)
    rvv = y ** 2 * b
    Fh = np.asarray(X[:, har], np.float64)
    Fl = np.asarray(X[:, lad], np.float64).reshape(n, C, R)
    print(f"corrected cache {X.shape}: {C} channels x {R} rungs, "
          f"{len(BETAS)} exponents")

    def amp_walk(gvec, lam=LAM_AMP):
        """Rolling walk with backbone + 41 amplitudes under a fixed weight."""
        Fa = Fl @ gvec
        Fd = np.hstack([Fh, Fa])
        pp = Fd.shape[1]
        P = np.zeros((1 + pp, 1 + pp))
        P[1 + nh:, 1 + nh:] = np.eye(C)
        out = np.full(n - W, np.nan)
        for s in range(W, n, CAD):
            e = min(s + CAD, n)
            A = np.hstack([np.ones((W, 1)), Fd[s - W:s]])
            cf = np.linalg.solve(A.T @ A + lam * P + 1e-8 * np.eye(1 + pp),
                                 A.T @ y[s - W:s])
            rr = y[s - W:s] - A @ cf
            Ae = np.hstack([np.ones((e - s, 1)), Fd[s:e]])
            out[s - W:e - W] = ((Ae @ cf) ** 2 + float(rr @ rr) / W) * b[s:e]
            del A, Ae
        return out

    rv_w = rvv[W:]
    base_ok = rv_w > 0

    # ---------------------------------------------------- 1. the sweep
    print("\n" + "=" * 74)
    print("1. FIXED EXPONENT, SWEPT")
    print("=" * 74)
    curve, preds_by_beta = [], {}
    for beta in BETAS:
        g = rungs ** (-beta)
        g = g / np.linalg.norm(g)
        p_ = amp_walk(g)
        m = base_ok & np.isfinite(p_) & (p_ > 0)
        q = float(np.mean(qlike_bar(rv_w[m], p_[m])))
        curve.append({"beta": float(beta), "qlike": q})
        preds_by_beta[float(beta)] = p_
        print(f"  beta = {beta:>5.2f}   QLIKE {q:.5f}")
    best = min(curve, key=lambda r: r["qlike"])
    qs = np.array([r["qlike"] for r in curve])
    within = [r["beta"] for r in curve if r["qlike"] <= best["qlike"] + 0.0005]
    print(f"\n  optimum beta = {best['beta']:.2f} at {best['qlike']:.5f}")
    print(f"  within 0.0005 of it: beta in [{min(within):.2f}, "
          f"{max(within):.2f}]  ({len(within)}/{len(curve)} grid points)")
    print(f"  spread across the whole sweep: {qs.max() - qs.min():.5f}")

    # ---------------------------------------------------- 2/3. fitted variants
    print("\n" + "=" * 74)
    print("2. FITTED EXPONENT, AND A FITTED 12-VECTOR SHAPE")
    print("=" * 74)

    def fitted_walk(mode):
        out = np.full(n - W, np.nan)
        picks = []
        for s in range(W, n, CAD):
            e = min(s + CAD, n)
            yt = y[s - W:s]
            if mode == "beta":
                bestg, bestsse = None, np.inf
                for beta in BETAS:
                    g = rungs ** (-beta)
                    g = g / np.linalg.norm(g)
                    A = np.hstack([np.ones((W, 1)), Fh[s - W:s],
                                   Fl[s - W:s] @ g])
                    cf, *_ = np.linalg.lstsq(A, yt, rcond=None)
                    rr = yt - A @ cf
                    sse = float(rr @ rr)
                    if sse < bestsse:
                        bestg, bestsse = g, sse
                    del A
                g = bestg
                picks.append(float(BETAS[
                    int(np.argmin([np.abs(rungs ** (-bb)
                                          / np.linalg.norm(rungs ** (-bb))
                                          - g).sum() for bb in BETAS]))]))
            else:
                # shared 12-vector: regress y on the pooled channel ladders
                Z = Fl[s - W:s].reshape(W * C, R)
                rep = np.repeat(yt[:, None], C, axis=1).reshape(-1)
                gg, *_ = np.linalg.lstsq(Z, rep, rcond=None)
                nrm = np.linalg.norm(gg)
                g = gg / nrm if nrm > 0 else np.ones(R) / np.sqrt(R)
            Fd_tr = np.hstack([np.ones((W, 1)), Fh[s - W:s], Fl[s - W:s] @ g])
            pp = Fd_tr.shape[1]
            P = np.zeros((pp, pp))
            P[1 + nh:, 1 + nh:] = np.eye(C)
            cf = np.linalg.solve(Fd_tr.T @ Fd_tr + LAM_AMP * P
                                 + 1e-8 * np.eye(pp), Fd_tr.T @ yt)
            rr = yt - Fd_tr @ cf
            Ae = np.hstack([np.ones((e - s, 1)), Fh[s:e], Fl[s:e] @ g])
            out[s - W:e - W] = ((Ae @ cf) ** 2 + float(rr @ rr) / W) * b[s:e]
            del Fd_tr, Ae
        return out, picks

    res = {}
    p_fb, picks = fitted_walk("beta")
    p_fs, _ = fitted_walk("shape")

    # ---------------------------------------------------- references
    def raw_walk():
        Fd = np.hstack([Fh, Fl.reshape(n, C * R)])
        pp = Fd.shape[1]
        P = np.zeros((1 + pp, 1 + pp))
        for c in range(C):
            s0 = 1 + nh + c * R
            P[s0:s0 + R, s0:s0 + R] = np.eye(R)
        out = np.full(n - W, np.nan)
        for s in range(W, n, CAD):
            e = min(s + CAD, n)
            A = np.hstack([np.ones((W, 1)), Fd[s - W:s]])
            cf = np.linalg.solve(A.T @ A + LAM_RAW * P + 1e-8 * np.eye(1 + pp),
                                 A.T @ y[s - W:s])
            rr = y[s - W:s] - A @ cf
            Ae = np.hstack([np.ones((e - s, 1)), Fd[s:e]])
            out[s - W:e - W] = ((Ae @ cf) ** 2 + float(rr @ rr) / W) * b[s:e]
            del A, Ae
        return out

    def bb_walk():
        out = np.full(n - W, np.nan)
        for s in range(W, n, CAD):
            e = min(s + CAD, n)
            A = np.hstack([np.ones((W, 1)), Fh[s - W:s]])
            cf = np.linalg.solve(A.T @ A + 1e-8 * np.eye(1 + nh),
                                 A.T @ y[s - W:s])
            rr = y[s - W:s] - A @ cf
            Ae = np.hstack([np.ones((e - s, 1)), Fh[s:e]])
            out[s - W:e - W] = ((Ae @ cf) ** 2 + float(rr @ rr) / W) * b[s:e]
        return out

    arms = {"fixed beta*": preds_by_beta[best["beta"]],
            "fixed beta=1": preds_by_beta[1.0],
            "fitted beta": p_fb, "fitted 12-vector": p_fs,
            "ridge on 492 raw": raw_walk(), "backbone": bb_walk()}
    ok = base_ok.copy()
    for p_ in arms.values():
        ok &= np.isfinite(p_) & (p_ > 0)
    gi = np.flatnonzero(ok)
    print(f"\n  scored {len(gi):,} bars")
    print(f"  {'arm':>20}{'QLIKE':>11}")
    losses = {}
    for k, p_ in arms.items():
        losses[k] = qlike_bar(rv_w[gi], p_[gi])
        q = float(np.mean(losses[k]))
        res[k] = q
        print(f"  {k:>20}{q:>11.5f}")
    if picks:
        print(f"\n  fitted beta picked: median {np.median(picks):.2f}, "
              f"range [{min(picks):.2f}, {max(picks):.2f}] across "
              f"{len(picks)} refits")

    print(f"\n  DM against ridge on 492 raw:")
    dms = {}
    for k in arms:
        if k == "ridge on 492 raw":
            continue
        r = dm_test(losses[k], losses["ridge on 492 raw"], h=1)
        print(f"    {k:>20}{r['mean_diff']:>+11.5f}  t {r['t']:>+6.2f}  "
              f"p {r['p']:.3g}")
        dms[k] = {"diff": r["mean_diff"], "t": r["t"], "p": r["p"]}

    # ---------------------------------------------------- per-era optimum
    print("\n" + "=" * 74)
    print("3. IS THE EXPONENT STABLE ACROSS ERAS?")
    print("=" * 74)
    blocks = np.array_split(np.arange(len(gi)), NBLOCK)
    raw = load_raw_data("data", allow_missing=True).dropna(
        subset=["RV"]).reset_index(drop=True)
    tt = pd.to_datetime(raw["t"]).iloc[3125:3125 + n].reset_index(drop=True)
    dt = tt.iloc[W:].reset_index(drop=True).iloc[gi].reset_index(drop=True)
    del raw
    print(f"  {'blk':>4}{'period':>22}{'best beta':>11}{'QLIKE':>10}")
    eras = []
    for bi, blk in enumerate(blocks):
        qq = [(float(bt), float(np.mean(qlike_bar(
            rv_w[gi][blk], preds_by_beta[float(bt)][gi][blk]))))
            for bt in BETAS]
        bb = min(qq, key=lambda z: z[1])
        per = f"{dt.iloc[blk[0]].date()}..{dt.iloc[blk[-1]].date()}"
        print(f"  {bi+1:>4}{per:>22}{bb[0]:>11.2f}{bb[1]:>10.5f}")
        eras.append({"block": bi + 1, "period": per, "best_beta": bb[0],
                     "qlike": bb[1]})
    bs = [e["best_beta"] for e in eras]
    stable = max(bs) - min(bs) <= 0.5
    print(f"\n  per-era optima span [{min(bs):.2f}, {max(bs):.2f}]")
    print("  -> " + ("the exponent is STABLE across eras: a structural "
                     "constant" if stable else
                     "the exponent WANDERS across eras: it is a fitted "
                     "constant, not a structural one, and should be described "
                     "as such"))

    out = {"curve": curve, "best": best, "flat_range": [min(within),
                                                        max(within)],
           "sweep_spread": float(qs.max() - qs.min()), "arms": res,
           "dm_vs_raw": dms, "eras": eras,
           "fitted_beta_picks": {"median": float(np.median(picks)) if picks
                                 else None,
                                 "min": float(min(picks)) if picks else None,
                                 "max": float(max(picks)) if picks else None},
           "exponent_stable": bool(stable)}
    with open(os.path.join(OUT_DIR, "exponent_sweep.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote exponent_sweep.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
