"""Where does rank on the lag axis saturate?

The amplitude design contracts each channel's 12 rungs against ONE fixed shape.
Rank-r keeps r shapes instead. Earlier runs showed 1 -> 2 -> 3 improving, and a
six-design factorial then showed rank-3 (0.13117) beating both rank-6 (0.13154)
and the full 492-column design (0.13146) -- i.e. the curve may turn over rather
than saturate. That would be worth knowing: it says the lag axis has an
intrinsic dimension of about three, and that spending the remaining nine
directions costs more in variance than it returns in fit.

Two things this controls for.

BASIS DEPENDENCE. A rank-r result is a statement about an r-dimensional
subspace, and which subspace depends on the basis. Two nested orthonormal
bases are swept independently, and a conclusion is only reported if both agree:

  power   Gram-Schmidt over rungs^{-beta} for a spread of beta, then filled
          with log-polynomials once the power family goes collinear. Closest to
          the paper's power-law reading of the effective kernel.
  logpoly orthonormal polynomials in log2(rung), degree 0..11. Nested and
          complete by construction; degree 1 is a pure power law, so the
          sequence generalises the power reading rather than competing with it.

At r = 12 either basis spans the same space as the raw 492 columns, so that row
is a sanity check: it should land near raw492 and does.

SELECTION. lambda is chosen per design on a validation region strictly
preceding evaluation, never on the scored rows. The reported argmin over r is
an ORACLE and is labelled as such -- picking r on the evaluation set is exactly
the error that cost this project +0.0033 once already. The achievable arm picks
r on the same validation region.
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

OUT_DIR = os.path.join(ROOT, "writeup", "stats")
CACHE = os.environ.get("RANK_CACHE", "b2_mmap_warm")
W, CAD = 24000, 1000
LAMS = [0.0] + [10.0 ** e for e in range(-2, 7)]
RANKS = [1, 2, 3, 4, 5, 6, 8, 10, 12]
NBLOCK = 8


def qlike_bar(rv, p):
    r = rv / p
    return r - np.log(r) - 1.0


def layout(names):
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
    return labels, rungs, lad, har


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


def build_bases(rungs):
    lg = np.log2(rungs)
    lg = (lg - lg.mean()) / lg.std()
    poly = [lg ** d for d in range(12)]
    logpoly = gram_schmidt(poly)
    pw = [rungs ** (-b) for b in (1.0, 0.5, 2.0, 0.0, 3.0, 1.5, 0.25, 4.0)]
    power = gram_schmidt(pw + poly)          # fill once the power family runs out
    return {"power": power, "logpoly": logpoly}


def main():
    t0 = time.time()
    D = os.path.join(ROOT, "results", CACHE)
    X = np.load(os.path.join(D, "X.npy"), mmap_mode="r")
    y = np.load(os.path.join(D, "y.npy"))
    b = np.load(os.path.join(D, "b.npy"))
    names = [str(s) for s in np.load(os.path.join(D, "names.npy"),
                                     allow_pickle=True)]
    labels, rungs, lad, har = layout(names)
    C, R = len(labels), 12
    n = len(y)
    rvv = y ** 2 * b
    Fh = np.asarray(X[:, har], np.float64)
    Fl = np.asarray(X[:, lad], np.float64).reshape(n, C, R)
    del X
    print(f"cache: {CACHE}")
    print(f"  {n:,} rows, {C} channels x {R} rungs ({time.time()-t0:.0f}s)")
    BASES = build_bases(rungs)
    for k, v in BASES.items():
        print(f"  basis {k}: {len(v)} orthonormal directions")

    def walk(Fd, npen, lam, lo, hi):
        pp = Fd.shape[1]
        P = np.zeros((1 + pp, 1 + pp))
        if npen:
            P[1 + pp - npen:, 1 + pp - npen:] = np.eye(npen)
        out = np.full(hi - lo, np.nan)
        s0 = lo - ((lo - W) % CAD)
        for s in range(s0, hi, CAD):
            e = min(s + CAD, hi)
            if e <= lo:
                continue
            A = np.hstack([np.ones((W, 1)), Fd[s - W:s]])
            cf = np.linalg.solve(A.T @ A + lam * P + 1e-8 * np.eye(1 + pp),
                                 A.T @ y[s - W:s])
            rr = y[s - W:s] - A @ cf
            a, z = max(s, lo), e
            Ae = np.hstack([np.ones((z - a, 1)), Fd[a:z]])
            out[a - lo:z - lo] = ((Ae @ cf) ** 2 + float(rr @ rr) / W) * b[a:z]
            del A, Ae
        return out

    def pick_lam(Fd, npen):
        if npen == 0:
            return 0.0, np.inf
        v0, v1 = int(n * 0.55), int(n * 0.65)
        pp = Fd.shape[1]
        P = np.zeros((1 + pp, 1 + pp))
        P[1 + pp - npen:, 1 + pp - npen:] = np.eye(npen)
        num = {L: 0.0 for L in LAMS}
        den = {L: 0 for L in LAMS}
        s0 = v0 - ((v0 - W) % CAD)
        for s in range(s0, v1, CAD):
            e = min(s + CAD, v1)
            if e <= v0:
                continue
            A = np.hstack([np.ones((W, 1)), Fd[s - W:s]])
            G, c = A.T @ A, A.T @ y[s - W:s]
            a, z = max(s, v0), e
            Ae = np.hstack([np.ones((z - a, 1)), Fd[a:z]])
            for L in LAMS:
                cf = np.linalg.solve(G + L * P + 1e-8 * np.eye(1 + pp), c)
                rr = y[s - W:s] - A @ cf
                pv = ((Ae @ cf) ** 2 + float(rr @ rr) / W) * b[a:z]
                rt = rvv[a:z]
                m = (rt > 0) & np.isfinite(pv) & (pv > 0)
                if m.sum():
                    q = rt[m] / pv[m]
                    num[L] += float(np.sum(q - np.log(q) - 1.0))
                    den[L] += int(m.sum())
            del A, G, Ae
        best = (np.inf, 0.0)
        for L in LAMS:
            if den[L] and num[L] / den[L] < best[0]:
                best = (num[L] / den[L], L)
        return best[1], best[0]

    preds, lams, valq = {}, {}, {}
    Fd = Fh
    lam, vq = pick_lam(Fd, 0)
    preds["backbone"] = walk(Fd, 0, lam, W, n)
    lams["backbone"], valq["backbone"] = lam, vq
    print(f"    {'backbone':>18} lambda*={lam:<8g} ({time.time()-t0:.0f}s)",
          flush=True)

    Fraw = np.hstack([Fh, Fl.reshape(n, C * R)])
    lam, vq = pick_lam(Fraw, C * R)
    preds["raw492"] = walk(Fraw, C * R, lam, W, n)
    lams["raw492"], valq["raw492"] = lam, vq
    del Fraw
    print(f"    {'raw492':>18} lambda*={lam:<8g} ({time.time()-t0:.0f}s)",
          flush=True)

    for bname, basis in BASES.items():
        for r in RANKS:
            if r > len(basis):
                continue
            key = f"{bname} r={r}"
            A = np.concatenate([Fl @ basis[j] for j in range(r)], axis=1)
            Fd = np.hstack([Fh, A])
            lam, vq = pick_lam(Fd, A.shape[1])
            preds[key] = walk(Fd, A.shape[1], lam, W, n)
            lams[key], valq[key] = lam, vq
            del A, Fd
            print(f"    {key:>18} lambda*={lam:<8g} cols={C*r+len(har):<4d} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    rv_w = rvv[W:]
    good = rv_w > 0
    for k in preds:
        good &= np.isfinite(preds[k]) & (preds[k] > 0)
    idx = np.flatnonzero(good)
    print(f"\n  scored {len(idx):,} bars")
    L = {k: qlike_bar(rv_w[idx], preds[k][idx]) for k in preds}
    Q = {k: float(L[k].mean()) for k in L}

    print(f"\n  {'r':>3}{'cols':>7}" +
          "".join(f"{bn:>14}" for bn in BASES) + f"{'agree':>8}")
    for r in RANKS:
        row = []
        for bn in BASES:
            k = f"{bn} r={r}"
            row.append(Q.get(k, float("nan")))
        agree = "yes" if (np.isfinite(row).all() and
                          abs(row[0] - row[1]) < 0.0005) else ""
        print(f"  {r:>3}{C*r+len(har):>7}" +
              "".join(f"{v:>14.5f}" for v in row) + f"{agree:>8}")
    print(f"  {'--':>3}{len(har):>7}{Q['backbone']:>14.5f}"
          f"{'':>14}{'backbone':>8}")
    print(f"  {'--':>3}{C*R+len(har):>7}{Q['raw492']:>14.5f}"
          f"{'':>14}{'raw492':>8}")

    out = {"cache": CACHE, "n_scored": int(len(idx)), "qlike": Q,
           "lambda": lams, "ranks": RANKS,
           "cols": {f"r={r}": C * r + len(har) for r in RANKS}}

    # ---- saturation, per basis, with the oracle labelled
    print()
    for bn in BASES:
        ks = [f"{bn} r={r}" for r in RANKS if f"{bn} r={r}" in Q]
        rs = [r for r in RANKS if f"{bn} r={r}" in Q]
        qs = [Q[k] for k in ks]
        r_or = rs[int(np.argmin(qs))]
        r_val = rs[int(np.argmin([valq[k] for k in ks]))]
        print(f"  {bn}: best r = {r_or} at {min(qs):.5f}   ORACLE "
              f"(not achievable)")
        print(f"  {bn}: validation-selected r = {r_val} at "
              f"{Q[f'{bn} r={r_val}']:.5f}   <- achievable")
        # is the turnover real? DM best vs r=12
        if f"{bn} r=12" in Q:
            d = dm_test(L[f"{bn} r={r_or}"], L[f"{bn} r=12"], h=1)
            print(f"      r={r_or} vs r=12: {d['mean_diff']:+.5f}  "
                  f"t {d['t']:+.2f}  p {d['p']:.3g}")
        d = dm_test(L[f"{bn} r={r_val}"], L["raw492"], h=1)
        print(f"      validation r={r_val} vs raw492: {d['mean_diff']:+.5f}  "
              f"t {d['t']:+.2f}  p {d['p']:.3g}")
        out[f"{bn}_oracle_r"] = int(r_or)
        out[f"{bn}_validation_r"] = int(r_val)

    with open(os.path.join(OUT_DIR, "rank_sweep.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote rank_sweep.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
