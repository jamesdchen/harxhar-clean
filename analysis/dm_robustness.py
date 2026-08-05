"""Is the pooled cache effect on raw492 (-0.00472, t -2.95) a reliable inference?

Two things prompted this and only one of them is a real question.

THE FALSE ALARM. A note in PROTOCOL.md described `|t| > 3` as "the bar this
protocol uses elsewhere". It is not. That threshold appears exactly once, as
the pre-registered falsification condition for one specific comparison, and
was never a house convention. Flagging a different result for failing an
invented rule is itself an error and is corrected in the protocol.

THE REAL QUESTION. The differential is heavy-tailed and clustered in time --
essentially all of it lives in one era, and within that era in a handful of
days. A Diebold-Mariano statistic assumes the mean of the differential is
approximately normal, which is exactly what concentration and dependence break.
n = 218,909 is no defence: the EFFECTIVE sample size for a series whose signal
sits in a few hundred bars is far smaller than its length.

So rather than argue about a threshold, this measures whether the inference
survives being done four ways:

  1  DM across a range of HAC bandwidths, to see if the rule-of-thumb
     bandwidth is doing the work
  2  a stationary block bootstrap, which needs neither normality nor a
     parametric variance estimate
  3  an era-level paired test on 8 block means -- tiny n, but each block is
     nearly independent, so it asks whether the effect is a general property
     or one era's
  4  a leave-one-era-out sweep, which answers "does this survive deleting the
     era that carries it"

Concentration diagnostics are reported alongside, because a mean differential
whose top 0.1% of bars carry most of its mass is a different object from one
that is spread evenly, whatever its t-statistic says.
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

from inference import (auto_lag, dm_test, newey_west_lrv,  # noqa: E402
                       stationary_bootstrap_indices)

OUT_DIR = os.path.join(ROOT, "writeup", "stats")
CACHES = [("none", "b2_mmap"), ("composed", "b2_mmap_warm")]
# MODE=cache  : raw492 on two caches (the original question)
# MODE=design : r=3 power basis vs raw492, both on the clean cache
MODE = os.environ.get("DMR_MODE", "cache")
DESIGN_CACHE = os.environ.get("DMR_CACHE", "b2_mmap_warm")
W, CAD = 24000, 1000
LAMS = [0.0] + [10.0 ** e for e in range(-2, 7)]
NBLOCK = 8
BOOT = 4000
SEED = 20240803


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
    lad = np.array([j for c in labels for _, j in chan[c]], int)
    har = np.array([j for j, nm in enumerate(names)
                    if re.match(r"^har_ma_\d+$", nm)], int)
    return labels, lad, har


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


def design_losses(path, which):
    """which='raw492' or 'r3' -- both on the SAME cache, so no October effect."""
    D = os.path.join(ROOT, "results", path)
    X = np.load(os.path.join(D, "X.npy"), mmap_mode="r")
    y = np.load(os.path.join(D, "y.npy"))
    b = np.load(os.path.join(D, "b.npy"))
    names = [str(s) for s in np.load(os.path.join(D, "names.npy"),
                                     allow_pickle=True)]
    chan = {}
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
    n = len(y)
    rvv = y ** 2 * b
    Fh = np.asarray(X[:, har], np.float64)
    Fl = np.asarray(X[:, lad], np.float64).reshape(n, len(labels), 12)
    del X
    if which == "raw492":
        Fd = np.hstack([Fh, Fl.reshape(n, len(labels) * 12)])
    else:
        SH = gram_schmidt([rungs ** -1.0, rungs ** -0.5, rungs ** -2.0])
        A = np.concatenate([Fl @ SH[j] for j in range(3)], axis=1)
        Fd = np.hstack([Fh, A])
    npen = Fd.shape[1] - len(har)
    return _fit(Fd, npen, y, b, rvv, n), rvv[W:], Fd.shape[1]


def _fit(Fd, npen, y, b, rvv, n):
    pp = Fd.shape[1]
    P = np.zeros((1 + pp, 1 + pp))
    P[1 + pp - npen:, 1 + pp - npen:] = np.eye(npen)
    v0, v1 = int(n * 0.55), int(n * 0.65)
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
    lam = min((num[L] / den[L], L) for L in LAMS if den[L])[1]
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
    print(f"      lambda*={lam:g}", flush=True)
    return out


def raw492_losses(path):
    D = os.path.join(ROOT, "results", path)
    X = np.load(os.path.join(D, "X.npy"), mmap_mode="r")
    y = np.load(os.path.join(D, "y.npy"))
    b = np.load(os.path.join(D, "b.npy"))
    names = [str(s) for s in np.load(os.path.join(D, "names.npy"),
                                     allow_pickle=True)]
    labels, lad, har = layout(names)
    n = len(y)
    rvv = y ** 2 * b
    Fd = np.hstack([np.asarray(X[:, har], np.float64),
                    np.asarray(X[:, lad], np.float64)])
    del X
    npen = len(lad)
    pp = Fd.shape[1]
    P = np.zeros((1 + pp, 1 + pp))
    P[1 + pp - npen:, 1 + pp - npen:] = np.eye(npen)

    # lambda on a validation region strictly preceding evaluation
    v0, v1 = int(n * 0.55), int(n * 0.65)
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
    lam = min((num[L] / den[L], L) for L in LAMS if den[L])[1]

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
    return out, rvv[W:], lam


def main():
    t0 = time.time()
    print("Is the raw492 cache effect a reliable inference?\n")
    pr, rv_ref, lams = {}, None, {}
    if MODE == "design":
        print(f"MODE=design: r=3 power basis vs raw492, both on {DESIGN_CACHE}")
        for tag, which in (("none", "raw492"), ("composed", "r3")):
            print(f"  building {which} ...", flush=True)
            pr[tag], rvw, ncol = design_losses(DESIGN_CACHE, which)
            rv_ref = rvw if rv_ref is None else rv_ref
            lams[tag] = ncol
            print(f"  {which:>9}: {ncol} cols  ({time.time()-t0:.0f}s)",
                  flush=True)
    else:
        for tag, path in CACHES:
            pr[tag], rvw, lams[tag] = raw492_losses(path)
            rv_ref = rvw if rv_ref is None else rv_ref
            print(f"  {tag:>9}: lambda*={lams[tag]:g}  ({time.time()-t0:.0f}s)",
                  flush=True)

    good = rv_ref > 0
    for t in pr:
        good &= np.isfinite(pr[t]) & (pr[t] > 0)
    idx = np.flatnonzero(good)
    L = {t: qlike_bar(rv_ref[idx], pr[t][idx]) for t in pr}
    d = L["composed"] - L["none"]          # negative favours composed
    n = d.size
    print(f"\n  {n:,} bars, mean differential {d.mean():+.6f}")

    out = {"n": int(n), "mean_diff": float(d.mean()),
           "lambda": {k: float(v) for k, v in lams.items()}}

    # ---- concentration
    print(f"\n{'=' * 72}\nCONCENTRATION\n{'=' * 72}")
    o = np.argsort(d)          # most negative first
    tot = d.sum()
    conc = {}
    for frac in (0.001, 0.01, 0.05, 0.10):
        k = max(1, int(frac * n))
        share = float(d[o[:k]].sum() / tot)
        conc[f"top_{frac:g}"] = share
        print(f"  most-favourable {frac * 100:5.1f}% of bars ({k:>6,}) "
              f"carry {share * 100:6.1f}% of the total differential")
    pos = float((d > 0).mean())
    print(f"  bars where composed is WORSE: {pos * 100:.1f}%")
    print(f"  median per-bar differential : {np.median(d):+.7f}")
    out["concentration"] = conc
    out["frac_bars_worse"] = pos
    out["median_diff"] = float(np.median(d))

    # ---- 1. DM across bandwidths
    print(f"\n{'=' * 72}\n1. DM ACROSS HAC BANDWIDTHS\n{'=' * 72}")
    ar = auto_lag(n)
    bw = sorted({0, 1, 10, 100, ar, 1000, 5000})
    dmres = {}
    for L_ in bw:
        r = dm_test(L["composed"], L["none"], h=1, lag=L_)
        dmres[str(L_)] = {"t": float(r["t"]), "p": float(r["p"])}
        mark = "  <- rule of thumb" if L_ == ar else ""
        print(f"  lag {L_:>5}: t {r['t']:+7.2f}  p {r['p']:.4g}{mark}")
    out["dm_by_bandwidth"] = dmres

    # ---- 2. stationary block bootstrap
    print(f"\n{'=' * 72}\n2. STATIONARY BLOCK BOOTSTRAP ({BOOT} reps)\n{'=' * 72}")
    rng = np.random.default_rng(SEED)
    boots = {}
    for blk in (100.0, 1000.0, 5000.0):
        I = stationary_bootstrap_indices(n, BOOT, blk, rng)
        means = d[I].mean(axis=1)
        lo, hi = np.percentile(means, (2.5, 97.5))
        pb = float(2 * min((means >= 0).mean(), (means <= 0).mean()))
        boots[str(int(blk))] = {"lo": float(lo), "hi": float(hi), "p": pb}
        print(f"  block {int(blk):>5}: 95% CI [{lo:+.6f}, {hi:+.6f}]  "
              f"p {pb:.4g}  {'excludes 0' if hi < 0 or lo > 0 else 'INCLUDES 0'}")
        del I, means
    out["bootstrap"] = boots

    # ---- 3. era-level paired test
    print(f"\n{'=' * 72}\n3. ERA-LEVEL (8 blocks, nearly independent)\n{'=' * 72}")
    nb = n // NBLOCK
    em = np.array([d[i * nb:(i + 1) * nb if i < NBLOCK - 1 else n].mean()
                   for i in range(NBLOCK)])
    for i, v in enumerate(em, 1):
        print(f"    era {i}: {v:+.6f}  {'composed better' if v < 0 else 'WORSE'}")
    tt = em.mean() / (em.std(ddof=1) / np.sqrt(NBLOCK))
    nneg = int((em < 0).sum())
    # exact sign test, two-sided
    from math import comb
    pv = sum(comb(NBLOCK, k) for k in range(nneg, NBLOCK + 1)) / 2 ** NBLOCK
    print(f"\n  mean over eras {em.mean():+.6f}, paired t({NBLOCK-1}) = {tt:+.2f}")
    print(f"  composed better in {nneg}/{NBLOCK} eras, sign test p = "
          f"{min(1.0, 2 * pv):.4g}")
    out["era_means"] = [float(v) for v in em]
    out["era_t"] = float(tt)
    out["eras_favouring_composed"] = nneg

    # ---- 4. leave-one-era-out
    print(f"\n{'=' * 72}\n4. LEAVE-ONE-ERA-OUT\n{'=' * 72}")
    loo = {}
    for i in range(NBLOCK):
        a, z = i * nb, ((i + 1) * nb if i < NBLOCK - 1 else n)
        keep = np.ones(n, bool)
        keep[a:z] = False
        r = dm_test(L["composed"][keep], L["none"][keep], h=1)
        loo[str(i + 1)] = {"diff": float(r["mean_diff"]), "t": float(r["t"])}
        print(f"  drop era {i + 1}: diff {r['mean_diff']:+.6f}  t {r['t']:+6.2f}")
    out["leave_one_era_out"] = loo

    with open(os.path.join(OUT_DIR, f"dm_robustness_{MODE}.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote dm_robustness.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
