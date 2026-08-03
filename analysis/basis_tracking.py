"""If the rank-3 subspace really drifts, how fast should we track it?

manifold_drift_test.py asks whether there IS drift. This asks the only question
that matters for the paper: is the drift EXPLOITABLE. The two come apart --
a subspace can genuinely rotate and still be untrackable if the rotation is
slower than the estimation error is large.

One knob: an exponential halflife on the moments the basis is built from.
At each refit boundary the basis is rebuilt from EWMA moments over all prior
data, decayed with halflife h (bars). h -> 0 recovers the fully-local refit
that lost by +0.00023; h -> inf recovers an expanding-window estimate that uses
everything and never forgets. Sweeping h traces the bias-variance curve between
them, and the argmin IS the drift timescale if one exists.

Three readings, and they are mutually exclusive:

  INTERIOR ARGMIN. Real drift, real estimation error, and a finite optimal
  memory. The subspace rotates on a measurable timescale and tracking it at
  that rate beats both extremes. This is the manifold/drift reading, and it
  would be the first thing in this study to make the compression exploitable.

  ARGMIN AT h = inf. No exploitable drift. More data always helps and the best
  estimate of the subspace is the one that forgets nothing. Whatever the
  permutation test says about rotation, it is not worth tracking.

  FROZEN(first W) BEATS h = inf. The published win was a lucky draw. Both arms
  are stable, but frozen sees only rows [0, 24000) -- 2005-2007 -- while h=inf
  sees everything up to t. An estimator using strictly more data cannot be
  beaten by one using strictly less unless the less-informed one got lucky.

That third arm is the control the original design lacked: it separates "freezing
helps" from "THIS freeze helped".

All moments are causal: at boundary t only rows < t enter.
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

from inference import dm_test                                    # noqa: E402
from ladder_refit import (CACHE, NBLOCK, RANK, RUNGS, W,          # noqa: E402
                          build_target, gram_schmidt, load_raw_data,
                          qlike_bar, walk)

CAD = 6000
HALFLIVES = [6000, 12000, 24000, 48000, 96000, 192000, None]   # None = inf
KIND = os.environ.get("BT_KIND", "sup")
OUT = os.path.join(ROOT, "writeup", "stats", "basis_tracking.json")


def seed_basis(C, xy, r, kind):
    """Same Krylov construction as ladder_refit.sup_basis, moments supplied."""
    C = C + 1e-12 * np.eye(C.shape[0])
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


class EWMoments:
    """Causal exponentially-weighted first and second moments of (H, y)."""

    def __init__(self, p, halflife):
        self.rho = 1.0 if halflife is None else 0.5 ** (CAD / halflife)
        self.n = 0.0
        self.sH = np.zeros(p)
        self.SHH = np.zeros((p, p))
        self.sy = 0.0
        self.sHy = np.zeros(p)

    def add(self, Hb, yb):
        ok = np.isfinite(Hb).all(1) & np.isfinite(yb)
        Hb, yb = Hb[ok], yb[ok]
        if not len(Hb):
            return
        r = self.rho
        self.n = r * self.n + len(Hb)
        self.sH = r * self.sH + Hb.sum(0)
        self.SHH = r * self.SHH + Hb.T @ Hb
        self.sy = r * self.sy + yb.sum()
        self.sHy = r * self.sHy + Hb.T @ yb

    def moments(self):
        mu = self.sH / self.n
        my = self.sy / self.n
        C = self.SHH / self.n - np.outer(mu, mu)
        xy = self.sHy / self.n - mu * my
        return C, xy


def main():
    t0 = time.time()
    D = os.path.join(ROOT, "results", CACHE)
    Xc = np.load(os.path.join(D, "X.npy"), mmap_mode="r")
    names = [str(s) for s in np.load(os.path.join(D, "names.npy"),
                                     allow_pickle=True)]
    n = len(np.load(os.path.join(D, "y.npy")))
    CALN = ["DOW_0", "DOW_1", "DOW_2", "DOW_3", "DOW_4", "hour",
            "is_overnight", "is_open", "is_close"]
    cal = np.asarray(Xc[:, [names.index(c) for c in CALN if c in names]],
                     np.float64)
    del Xc
    raw = load_raw_data(os.path.join(ROOT, "data"), allow_missing=True).dropna(
        subset=["RV"]).reset_index(drop=True)
    off = len(raw) - n
    y99, b99 = build_target(raw, 0.01, 0.99)
    y595, _ = build_target(raw, 0.05, 0.95)
    y = y99[off:off + n]
    b = b99[off:off + n]
    truth = y595[off:off + n] ** 2 * b                # battery footing
    s = pd.Series(y ** 2)
    H = np.column_stack([
        np.sqrt(np.maximum(
            s.rolling(L, min_periods=L).mean().shift(1).to_numpy(), 0.0))
        for L in RUNGS])
    bounds = list(range(W, n, CAD)) + [n]
    print(f"cache {CACHE}: {n:,} rows, kind={KIND}, {len(bounds)-1} boundaries",
          flush=True)

    preds, meta = {}, {}

    # ---- reference arms: the full ladder, and the published freeze
    F0 = np.column_stack([H, cal])
    pr, _ = walk(np.where(np.isfinite(F0), F0, 0.0), y, b, W, W, n)
    pr[~np.isfinite(F0).all(1)[W:]] = np.nan
    preds["base-2 (r=12)"] = pr
    print(f"    {'base-2 (r=12)':>26} ({time.time()-t0:.0f}s)", flush=True)

    def run_basis(name, basis_at):
        out = np.full(n - W, np.nan)
        Gs = []
        for i in range(len(bounds) - 1):
            lo, hi = bounds[i], bounds[i + 1]
            Gm = basis_at(i, lo)
            Gs.append(Gm)
            F = np.column_stack([H @ Gm, cal])
            ok = np.isfinite(F).all(1)
            seg, _ = walk(np.where(ok[:, None], F, 0.0), y, b, W, lo, hi)
            seg[~ok[lo:hi]] = np.nan
            out[lo - W:hi - W] = seg
        preds[name] = out
        ang = []
        for i in range(1, len(Gs)):
            sv = np.linalg.svd(Gs[i - 1].T @ Gs[i], compute_uv=False)
            ang.append(np.degrees(np.arccos(np.clip(sv, -1, 1))))
        A = np.array(ang) if ang else np.zeros((1, RANK))
        meta[name] = {"angles_mean_per_dim": A.mean(0).tolist(),
                      "angles_max_per_dim": A.max(0).tolist()}
        print(f"    {name:>26} consec angles "
              f"[{A.mean(0)[0]:5.2f} {A.mean(0)[1]:5.2f} {A.mean(0)[2]:5.2f}]"
              f"  ({time.time()-t0:.0f}s)", flush=True)

    ok0 = np.isfinite(H[:W]).all(1) & np.isfinite(y[:W])
    H0, y0 = H[:W][ok0], y[:W][ok0]
    C0 = np.cov(H0, rowvar=False)
    xy0 = (H0 - H0.mean(0)).T @ (y0 - y0.mean()) / len(y0)
    G_frozen = seed_basis(C0, xy0, RANK, KIND)
    run_basis(f"{KIND} frozen(first W)", lambda i, lo: G_frozen)

    # ---- the halflife sweep
    for hl in HALFLIVES:
        tag = f"{KIND} h={'inf' if hl is None else hl}"
        M = EWMoments(H.shape[1], hl)
        cache = {}

        def basis_at(i, lo, M=M, cache=cache):
            if i not in cache:
                a = bounds[i - 1] if i else 0
                M.add(H[a:lo], y[a:lo])
                C, xy = M.moments()
                cache[i] = seed_basis(C, xy, RANK, KIND)
            return cache[i]

        run_basis(tag, basis_at)

    # ---- score, common support
    tw = truth[W:]
    good = tw > 0
    for k in preds:
        good &= np.isfinite(preds[k]) & (preds[k] > 0)
    idx = np.flatnonzero(good)
    L = {k: qlike_bar(tw[idx], preds[k][idx]) for k in preds}
    Q = {k: float(L[k].mean()) for k in L}
    print(f"\n  scored {len(idx):,} bars (clipped 5/95 truth)\n")
    ref = "base-2 (r=12)"
    print(f"  {'arm':>26}{'QLIKE':>10}{'d vs r=12':>12}{'t':>8}"
          f"{'eras':>7}")
    for k in sorted(Q, key=lambda k: Q[k]):
        if k == ref:
            print(f"  {k:>26}{Q[k]:>10.5f}{'--':>12}{'':>8}{'':>7}")
            continue
        d = dm_test(L[k], L[ref], h=1)
        e = np.array_split(L[k] - L[ref], NBLOCK)
        nb = int(sum(1 for x in e if x.mean() < 0))
        print(f"  {k:>26}{Q[k]:>10.5f}{d['mean_diff']:>+12.5f}"
              f"{d['t']:>+8.2f}{nb:>4}/{NBLOCK}")

    out = {"cache": CACHE, "kind": KIND, "cad": CAD, "n_scored": int(len(idx)),
           "halflives": [str(h) for h in HALFLIVES], "qlike": Q, "meta": meta}
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {os.path.relpath(OUT, ROOT)}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
