"""Factorial over cache version x design, replacing the one-factor-at-a-time sweeps.

Two problems with how this investigation has been running experiments.

FIRST, THE GRID WAS WASTEFUL AND THE SELECTION LOOP WORSE. The exponent sweep
evaluated 36 levels of a response that turned out to be flat over
beta in [0.2, 3.0] -- 29 of 36 points within 0.0005 of the best. Five levels
characterise that, and the rest bought nothing. Worse, lambda selection called
a walker that traversed the ENTIRE panel and then scored a 24-window validation
slice, ten times over: about 13 TFLOP spent choosing one hyperparameter. Both
are fixed here. The selection walk visits only the windows overlapping the
validation region, and within each window the Gram is built once and solved for
every lambda, so the penalty grid is nearly free.

SECOND, AND MORE IMPORTANTLY, ONE-FACTOR-AT-A-TIME CANNOT SEE THE INTERACTION
THAT HAS ALREADY REVERSED A CONCLUSION HERE. The stripped-down design beat the
elaborate one on the pre-fix cache and lost to plain ridge on the corrected
one. That is a design x data-version interaction, and it cost two full paper
rewrites to discover by running one factor at a time and rebuilding everything
in between. A factorial surfaces it directly, so the data version is promoted
from a nuisance we keep re-running to a first-class factor.

    A  cache      none / indicator-only / both fixes
    B  design     backbone, amplitude at beta in {0.5, 1, 2}, amplitude with
                  beta chosen per window on a held-out tail, raw 492 columns
    blocks        eight eras, which give the stability read rather than a
                  single pooled number

Reading the cell means gives the main effect of each factor and, in the
difference of differences, the interaction. The quantity that matters is not
"which design wins" but "does the winner depend on the cache", because the
answer so far is yes.
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
CACHES = [("none", "b2_mmap"), ("indicator", "b2_mmap_indonly"),
          ("both", "b2_mmap_fix")]
W, CAD = 24000, 1000
LAMS = [0.0] + [10.0 ** e for e in range(-2, 7)]
BETAS_FIT = [0.5, 1.0, 1.5, 2.0, 3.0]      # flat response: 5 levels, not 36
VAL_FRAC = 0.25
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


def run_cache(tag, path):
    D = os.path.join(ROOT, "results", path)
    X = np.load(os.path.join(D, "X.npy"), mmap_mode="r")
    y = np.load(os.path.join(D, "y.npy"))
    b = np.load(os.path.join(D, "b.npy"))
    names = [str(s) for s in np.load(os.path.join(D, "names.npy"),
                                     allow_pickle=True)]
    labels, rungs, lad, har = layout(names)
    C, R, nh = len(labels), 12, len(har)
    n = len(y)
    rvv = y ** 2 * b
    Fh = np.asarray(X[:, har], np.float64)
    Fl = np.asarray(X[:, lad], np.float64).reshape(n, C, R)
    del X

    def gvec(beta):
        g = rungs ** (-beta)
        return g / np.linalg.norm(g)

    def design_of(name):
        """(feature matrix, n penalised trailing columns)."""
        if name == "backbone":
            return Fh, 0
        if name == "raw492":
            return np.hstack([Fh, Fl.reshape(n, C * R)]), C * R
        beta = float(name.split("=")[1])
        return np.hstack([Fh, Fl @ gvec(beta)]), C

    def walk(Fd, npen, lam, lo, hi):
        """Predictions over [lo, hi). One Gram per window."""
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
        """Select lambda on the validation region ONLY, one Gram per window.

        The previous implementation walked the whole panel per lambda and then
        scored a slice, doing ~9x the necessary work ten times over. Here each
        validation window's Gram is built once and reused for every candidate.
        """
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
            G = A.T @ A
            c = A.T @ y[s - W:s]
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
            if den[L]:
                q = num[L] / den[L]
                if q < best[0]:
                    best = (q, L)
        return best[1]

    def walk_fitted_beta(lam):
        """beta chosen inside each training window on a held-out tail."""
        WV = int(W * VAL_FRAC)
        WF = W - WV
        out = np.full(n - W, np.nan)
        picks = []
        for s in range(W, n, CAD):
            e = min(s + CAD, n)
            f0, v0 = s - W, s - WV
            best = (np.inf, None, None)
            for beta in BETAS_FIT:
                g = gvec(beta)
                Af = np.hstack([np.ones((WF, 1)), Fh[f0:v0], Fl[f0:v0] @ g])
                pp = Af.shape[1]
                P = np.zeros((pp, pp))
                P[pp - C:, pp - C:] = np.eye(C)
                cf = np.linalg.solve(Af.T @ Af + lam * P + 1e-8 * np.eye(pp),
                                     Af.T @ y[f0:v0])
                rr = y[f0:v0] - Af @ cf
                Av = np.hstack([np.ones((WV, 1)), Fh[v0:s], Fl[v0:s] @ g])
                pv = ((Av @ cf) ** 2 + float(rr @ rr) / WF) * b[v0:s]
                rt = rvv[v0:s]
                m = (rt > 0) & np.isfinite(pv) & (pv > 0)
                sc = np.inf
                if m.sum():
                    q = rt[m] / pv[m]
                    sc = float(np.mean(q - np.log(q) - 1.0))
                del Af, Av
                if sc < best[0]:
                    best = (sc, g, beta)
            g, beta = best[1], best[2]
            picks.append(beta)
            Fd = np.hstack([np.ones((W, 1)), Fh[s - W:s], Fl[s - W:s] @ g])
            pp = Fd.shape[1]
            P = np.zeros((pp, pp))
            P[pp - C:, pp - C:] = np.eye(C)
            cf = np.linalg.solve(Fd.T @ Fd + lam * P + 1e-8 * np.eye(pp),
                                 Fd.T @ y[s - W:s])
            rr = y[s - W:s] - Fd @ cf
            Ae = np.hstack([np.ones((e - s, 1)), Fh[s:e], Fl[s:e] @ g])
            out[s - W:e - W] = ((Ae @ cf) ** 2 + float(rr @ rr) / W) * b[s:e]
            del Fd, Ae
        return out, picks

    DESIGNS = ["backbone", "amp b=0.5", "amp b=1.0", "amp b=2.0", "raw492"]
    preds, lams = {}, {}
    for dn in DESIGNS:
        Fd, npen = design_of(dn)
        lam = pick_lam(Fd, npen)
        lams[dn] = lam
        preds[dn] = walk(Fd, npen, lam, W, n)
        del Fd
    lam_amp = lams["amp b=1.0"]
    preds["amp b fitted"], picks = walk_fitted_beta(lam_amp)
    lams["amp b fitted"] = lam_amp
    return {"y": y, "b": b, "rvv": rvv, "n": n, "preds": preds,
            "lams": lams, "picks": picks}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    avail = [(t, p) for t, p in CACHES
             if os.path.exists(os.path.join(ROOT, "results", p, "X.npy"))]
    print(f"caches present: {[t for t, _ in avail]}")
    runs = {}
    for tag, path in avail:
        print(f"  running {tag} ...", flush=True)
        runs[tag] = run_cache(tag, path)
        print(f"    done ({time.time()-t0:.0f}s)", flush=True)

    ref = runs[avail[0][0]]
    n = ref["n"]
    rv_w = ref["rvv"][W:]
    good = rv_w > 0
    for r in runs.values():
        for p_ in r["preds"].values():
            good &= np.isfinite(p_) & (p_ > 0)
    gi = np.flatnonzero(good)
    print(f"\n  scored {len(gi):,} bars common to every cell")

    designs = list(ref["preds"])
    print("\n" + "=" * 84)
    print("CELL MEANS: cache (rows) x design (columns)")
    print("=" * 84)
    print(f"  {'cache':>12}" + "".join(f"{d:>14}" for d in designs))
    cells = {}
    for tag in runs:
        row = []
        for d in designs:
            q = float(np.mean(qlike_bar(rv_w[gi], runs[tag]["preds"][d][gi])))
            cells[(tag, d)] = q
            row.append(q)
        print(f"  {tag:>12}" + "".join(f"{v:>14.5f}" for v in row))

    print(f"\n  {'lambda*':>12}" + "".join(
        f"{ref['lams'][d]:>14g}" for d in designs))
    if ref["picks"]:
        pk = runs[avail[-1][0]]["picks"]
        print(f"  fitted beta picks: median {np.median(pk):.2f}, "
              f"range [{min(pk):.1f}, {max(pk):.1f}]")

    print("\n" + "=" * 84)
    print("MAIN EFFECTS AND THE INTERACTION")
    print("=" * 84)
    print("  design main effect (mean over caches):")
    for d in designs:
        m = np.mean([cells[(t, d)] for t in runs])
        print(f"    {d:>14}{m:>12.5f}")
    print("  cache main effect (mean over designs):")
    for t in runs:
        m = np.mean([cells[(t, d)] for d in designs])
        print(f"    {t:>14}{m:>12.5f}")

    print("\n  best design per cache -- the interaction, stated directly:")
    flips = set()
    for t in runs:
        bd = min(designs, key=lambda d: cells[(t, d)])
        flips.add(bd)
        print(f"    {t:>14}  ->  {bd}  ({cells[(t, bd)]:.5f})")
    print("  -> " + ("THE WINNER DEPENDS ON THE CACHE: a design x data-version "
                     "interaction,\n     which one-factor-at-a-time could not "
                     "have shown" if len(flips) > 1 else
                     "the same design wins on every cache: no interaction, and "
                     "the design\n     conclusion is robust to the data fixes"))

    out = {"cells": {f"{t}|{d}": v for (t, d), v in cells.items()},
           "designs": designs, "caches": list(runs),
           "lambdas": {d: ref["lams"][d] for d in designs},
           "best_per_cache": {t: min(designs, key=lambda d: cells[(t, d)])
                              for t in runs},
           "interaction": len(flips) > 1, "n_scored": int(len(gi))}

    if len(runs) > 1:
        print("\n" + "=" * 84)
        print("PER-ERA, BEST TWO DESIGNS ON THE MOST CORRECTED CACHE")
        print("=" * 84)
        tag = avail[-1][0]
        order = sorted(designs, key=lambda d: cells[(tag, d)])[:2]
        blocks = np.array_split(np.arange(len(gi)), NBLOCK)
        print(f"  {'blk':>4}" + "".join(f"{d:>16}" for d in order))
        eras = []
        for bi, blk in enumerate(blocks):
            vals = [float(np.mean(qlike_bar(
                rv_w[gi][blk], runs[tag]["preds"][d][gi][blk]))) for d in order]
            print(f"  {bi+1:>4}" + "".join(f"{v:>16.5f}" for v in vals))
            eras.append({"block": bi + 1,
                         **{d: v for d, v in zip(order, vals)}})
        out["eras"] = {"cache": tag, "designs": order, "rows": eras}
        r = dm_test(qlike_bar(rv_w[gi], runs[tag]["preds"][order[0]][gi]),
                    qlike_bar(rv_w[gi], runs[tag]["preds"][order[1]][gi]), h=1)
        print(f"\n  {order[0]} vs {order[1]} on {tag}: "
              f"{r['mean_diff']:+.5f}  t {r['t']:+.2f}  p {r['p']:.3g}")
        out["headline_dm"] = {"a": order[0], "b": order[1], "cache": tag,
                              "diff": r["mean_diff"], "t": r["t"],
                              "p": r["p"]}

    with open(os.path.join(OUT_DIR, "factorial_design.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote factorial_design.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
