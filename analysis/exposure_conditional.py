"""The exposure-conditional restatement of H2, and a direct test of dilution.

H2 as pre-registered -- "the cache main effect exceeds the design main effect"
-- was rejected (0.00067 against 0.00311) and stays rejected. This is a
SEPARATE, POST-HOC experiment. Nothing here rehabilitates H2; it tests the
mechanism we claimed explains H2's failure, which is a different question and
is labelled as exploratory throughout.

THE CLAIM UNDER TEST. We asserted that the pooled cache main effect is diluted
because five of the six designs in the factorial cannot respond to the cache
factor: the backbone uses no exogenous columns at all, and the amplitude
designs average each channel's twelve rungs into one number, attenuating a
corrupted channel about twelvefold. If that is right, a design's sensitivity to
the data version should rise with how much of the lag axis it keeps.

THE CIRCULARITY THIS AVOIDS. Defining "exposed" as "responds to the cache" and
then measuring the cache effect among exposed designs is guaranteed to produce
a large number and would prove nothing. Exposure here is a STRUCTURAL property
fixed before any loss is computed: a design that projects each channel's 12
rungs onto r fixed shapes has exposure r/12, independent of any data. The
backbone is 0, rank-1 is 1/12, raw 492 is 1. The ordering is known in advance,
so the regression of cache sensitivity on exposure is a real prediction.

THE ESTIMANDS. The pre-registered comparison also failed a like-for-like test:
it set a two-level difference against a six-level range, and ranges grow with
level count. Here both sides are the same statistic on the same design set, and
three statistics are reported because a pooled mean is a poor instrument for a
failure in which 27 bars carry 47% of the damage:

    T1  pooled mean QLIKE
    T2  era-8 mean QLIKE          (the era where the defect binds)
    T3  worst-decile mean QLIKE   (bars ranked by realized variance)

T2 and T3 select bars by time and by the target respectively, so the row set is
identical across arms and no arm's own loss influences which bars it is scored
on.
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
CACHES = [("none", "b2_mmap"), ("composed", "b2_mmap_warm")]
W, CAD = 24000, 1000
LAMS = [0.0] + [10.0 ** e for e in range(-2, 7)]
NBLOCK = 8
TAIL_FRAC = 0.10

# Designs ordered by structural exposure = lag dimensions retained per channel,
# divided by 12. Fixed here, before anything is run.
DESIGNS = [
    ("backbone", 0),
    ("amp rank-1", 1),
    ("amp rank-2", 2),
    ("amp rank-3", 3),
    ("amp rank-6", 6),
    ("raw492", 12),
]
# The a priori cut for "exposed": a design must keep at least half the lag
# axis. Stated before running, and it selects rank-6 and raw492.
EXPOSED_CUT = 0.5


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


def run_cache(tag, path):
    D = os.path.join(ROOT, "results", path)
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

    def gv(beta):
        g = rungs ** (-beta)
        return g / np.linalg.norm(g)

    # Six fixed shapes spanning the plausible decay range, Gram-Schmidt'd so a
    # rank-r design spans an r-dimensional subspace of the lag axis rather than
    # r nearly-collinear copies of the same direction. Fixed, never estimated.
    RAW = [gv(bb) for bb in (1.0, 0.5, 2.0, 0.0, 3.0, 1.5)]
    SHAPES = []
    for v in RAW:
        for u in SHAPES:
            v = v - (v @ u) * u
        nv = np.linalg.norm(v)
        if nv > 1e-8:
            SHAPES.append(v / nv)

    def design_of(name, r):
        if r == 0:
            return Fh, 0
        if r == 12:
            return np.hstack([Fh, Fl.reshape(n, C * R)]), C * R
        A = np.concatenate([Fl @ SHAPES[j] for j in range(r)], axis=1)
        return np.hstack([Fh, A]), A.shape[1]

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
            return 0.0
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
        return best[1]

    preds, lams = {}, {}
    for nm, r in DESIGNS:
        Fd, npen = design_of(nm, r)
        lam = pick_lam(Fd, npen)
        preds[nm] = walk(Fd, npen, lam, W, n)
        lams[nm] = lam
        del Fd
        print(f"    {nm:>14} lambda*={lam:<10g} ({time.time()-T0:.0f}s)",
              flush=True)
    return preds, rvv[W:], n - W


def main():
    global T0
    T0 = time.time()
    print("EXPOSURE-CONDITIONAL H2 -- post-hoc, exploratory.")
    print("H2 as pre-registered remains REJECTED (0.00067 vs 0.00311).\n")
    P, rv_ref = {}, None
    for tag, path in CACHES:
        if not os.path.exists(os.path.join(ROOT, "results", path, "X.npy")):
            print(f"  {tag}: cache absent, skipped")
            continue
        print(f"  running {tag} ...", flush=True)
        pr, rvw, _ = run_cache(tag, path)
        P[tag] = pr
        rv_ref = rvw if rv_ref is None else rv_ref

    tags = list(P)
    # rows common to every cell, chosen once from a rule no arm influences
    good = rv_ref > 0
    for t in tags:
        for nm, _ in DESIGNS:
            good &= np.isfinite(P[t][nm]) & (P[t][nm] > 0)
    idx = np.flatnonzero(good)
    print(f"\n  scored {len(idx):,} bars common to every cell")

    L = {(t, nm): qlike_bar(rv_ref[idx], P[t][nm][idx])
         for t in tags for nm, _ in DESIGNS}

    # ---- the three arm-independent estimands
    nb = len(idx) // NBLOCK
    era8 = np.arange(7 * nb, len(idx))
    tail = np.argsort(rv_ref[idx])[-int(TAIL_FRAC * len(idx)):]
    EST = {"T1 pooled": np.arange(len(idx)),
           "T2 era-8": era8,
           "T3 worst decile by RV": tail}

    out = {"note": "post-hoc; H2 as pre-registered remains rejected",
           "n_scored": int(len(idx)),
           "exposure": {nm: r / 12.0 for nm, r in DESIGNS},
           "exposed_cut": EXPOSED_CUT, "estimands": {}}

    for ename, rows in EST.items():
        print(f"\n{'=' * 84}\n{ename}   ({len(rows):,} bars)\n{'=' * 84}")
        cell = {(t, nm): float(L[(t, nm)][rows].mean())
                for t in tags for nm, _ in DESIGNS}
        print(f"  {'design':>14}{'exposure':>10}" +
              "".join(f"{t:>12}" for t in tags) + f"{'cache eff':>12}")
        eff = {}
        for nm, r in DESIGNS:
            e = r / 12.0
            ce = (cell[("none", nm)] - cell[("composed", nm)]
                  if len(tags) > 1 else float("nan"))
            eff[nm] = ce
            print(f"  {nm:>14}{e:>10.3f}" +
                  "".join(f"{cell[(t, nm)]:>12.5f}" for t in tags) +
                  f"{ce:>+12.5f}")

        expd = [nm for nm, r in DESIGNS if r / 12.0 >= EXPOSED_CUT]
        cache_eff = float(np.mean([eff[nm] for nm in expd]))
        design_rng = float(max(cell[("composed", nm)] for nm in expd) -
                           min(cell[("composed", nm)] for nm in expd))
        holds = cache_eff > design_rng
        print(f"\n  exposed designs (exposure >= {EXPOSED_CUT}): {expd}")
        print(f"    cache effect  (mean over exposed)      = {cache_eff:+.5f}")
        print(f"    design range  (over the same designs)  = {design_rng:+.5f}")
        print(f"    -> exposure-conditional H2 {'HOLDS' if holds else 'FAILS'}"
              f" on {ename}")

        # mechanism: does sensitivity rise with exposure?
        xs = np.array([r / 12.0 for _, r in DESIGNS])
        ys = np.array([eff[nm] for nm, _ in DESIGNS])
        from scipy.stats import spearmanr
        rho = spearmanr(xs, ys).correlation
        print(f"    spearman(exposure, cache effect) = {rho:+.3f}"
              f"   (dilution predicts a positive, monotone relation)")

        out["estimands"][ename] = {
            "n_rows": int(len(rows)),
            "cells": {f"{t}|{nm}": cell[(t, nm)]
                      for t in tags for nm, _ in DESIGNS},
            "cache_effect_per_design": eff,
            "exposed": expd,
            "cache_effect_exposed": cache_eff,
            "design_range_exposed": design_rng,
            "holds": bool(holds),
            "spearman_exposure_vs_effect": float(rho),
        }

    # ---- DM on the headline exposed contrast, pooled
    if len(tags) > 1:
        r = dm_test(L[("composed", "raw492")], L[("none", "raw492")], h=1)
        out["dm_raw492_composed_vs_none"] = {
            "diff": float(r["mean_diff"]), "t": float(r["t"]),
            "p": float(r["p"])}
        print(f"\n  DM raw492 composed vs none (pooled): "
              f"{r['mean_diff']:+.5f}  t {r['t']:+.2f}  p {r['p']:.3g}")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "exposure_conditional.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote exposure_conditional.json  ({time.time()-T0:.0f}s)")
    print("\nREMINDER: this is exploratory. The pre-registered H2 is rejected;")
    print("the restatement must be fixed in advance on a FUTURE panel to count.")


if __name__ == "__main__":
    main()
