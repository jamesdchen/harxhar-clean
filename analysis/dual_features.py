"""Does the dual/operator work yield an ESTIMATOR, or only diagnostics?

Two findings from the dual analysis were reported and never acted on.

FIRST, the effective-degrees-of-freedom accounting showed plain ridge spending
about 72 of its ~95 degrees of freedom in principal directions carrying 0.02%
of the explained target. That is a prescription, not just an observation: it
says shrink those directions harder. The estimator it implies -- a spectral
penalty whose strength rises with principal index -- has never been fitted.

SECOND, a supervised six-dimensional projection onto the operator's singular
triplets beat the full 516-dimensional ambient view in a retrieval metric by
0.00540 (t -8.9) on corrected data, having been an exact null on corrupted
data. It was never tried as a REGRESSION feature set.

Arms, all on the clean cache, lambda chosen on a validation region strictly
preceding evaluation:

    ridge-504    12 HAR + 492 exogenous, identity penalty      (reference)
    pc-k         12 HAR + top-k principal directions of the exogenous block,
                 SVD refit per window on training rows only, k in {5,20,50}
    spectral-a   12 HAR + all 492, penalty diag(s_j^-2a) in principal
                 coordinates, a in {0.5, 1.0}
    op-d6        12 HAR + 5 supervised operator scores, the operator refit
                 per window on training rows only

A linear model on a strict subspace cannot beat ridge on the full space by
fit alone, so any gain here is a VARIANCE effect -- which is exactly what the
df accounting predicts is available. Given that every design contrast measured
on this panel has so far tied, the honest prior is another tie; this is run
because the mechanism is specific, not because a win is expected.
"""
from __future__ import annotations
import json, os, re, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import dm_test  # noqa: E402
OUT_DIR = os.path.join(ROOT, "writeup", "stats")
CACHE = os.environ.get("DF_CACHE", "b2_mmap_warm")
W, CAD = 24000, 1000
LAMS = [0.0] + [10.0 ** e for e in range(-2, 7)]
NBLOCK = 8

def qlike_bar(rv, p):
    r = rv / p
    return r - np.log(r) - 1.0

def main():
    t0 = time.time()
    D = os.path.join(ROOT, "results", CACHE)
    X = np.load(os.path.join(D, "X.npy"), mmap_mode="r")
    y = np.load(os.path.join(D, "y.npy")); b = np.load(os.path.join(D, "b.npy"))
    names = [str(s) for s in np.load(os.path.join(D, "names.npy"), allow_pickle=True)]
    chan = {}
    for j, nm in enumerate(names):
        m = re.match(r"^adj_(.+)_ma_(\d+)$", nm)
        if m: chan.setdefault(m.group(1), []).append((int(m.group(2)), j))
    for c in chan: chan[c].sort()
    labels = [c for c in chan if len(chan[c]) == 12]
    lad = np.array([j for c in labels for _, j in chan[c]], int)
    har = np.array([j for j, nm in enumerate(names) if re.match(r"^har_ma_\d+$", nm)], int)
    n = len(y); rvv = y ** 2 * b
    Fh = np.asarray(X[:, har], np.float64)
    Fl = np.asarray(X[:, lad], np.float64)
    del X
    C, R, nh = len(labels), 12, len(har)
    print(f"cache {CACHE}: {n:,} rows, {C}x{R} exog, {nh} backbone ({time.time()-t0:.0f}s)", flush=True)

    def solve(G, c, P, lam, pp):
        return np.linalg.solve(G + lam * P + 1e-8 * np.eye(1 + pp), c)

    def run(kind, param, lo, hi, lam=None):
        """Walk. If lam is None, return validation QLIKE instead of predictions."""
        val = lam is None
        if val: lo, hi = int(n * 0.55), int(n * 0.65)
        out = np.full(hi - lo, np.nan)
        num = {L: 0.0 for L in LAMS}; den = {L: 0 for L in LAMS}
        s0 = lo - ((lo - W) % CAD)
        for s in range(s0, hi, CAD):
            e = min(s + CAD, hi)
            if e <= lo: continue
            Ztr = Fl[s - W:s]
            if kind == "ridge":
                Btr, Bte = Ztr, Fl[max(s, lo):e]
            elif kind in ("pc", "spectral"):
                mu = Ztr.mean(0)
                U, sv, Vt = np.linalg.svd(Ztr - mu, full_matrices=False)
                k = param if kind == "pc" else len(sv)
                Vk = Vt[:k].T
                Btr = (Ztr - mu) @ Vk; Bte = (Fl[max(s, lo):e] - mu) @ Vk
            else:  # op: supervised operator scores
                Atr = np.hstack([np.ones((W, 1)), Fh[s - W:s]])
                cf0 = np.linalg.lstsq(Atr, y[s - W:s], rcond=None)[0]
                res = y[s - W:s] - Atr @ cf0
                Zt = Ztr.reshape(W, C, R)
                M = np.einsum("t,tcr->cr", res, Zt) / W
                Uu, _, Vv = np.linalg.svd(M, full_matrices=False)
                k = param
                Btr = np.column_stack([np.einsum("tcr,c,r->t", Zt, Uu[:, j], Vv[j]) for j in range(k)])
                Zte = Fl[max(s, lo):e].reshape(e - max(s, lo), C, R)
                Bte = np.column_stack([np.einsum("tcr,c,r->t", Zte, Uu[:, j], Vv[j]) for j in range(k)])
            Fdtr = np.hstack([np.ones((W, 1)), Fh[s - W:s], Btr])
            a, z = max(s, lo), e
            Fdte = np.hstack([np.ones((z - a, 1)), Fh[a:z], Bte])
            pp = Fdtr.shape[1] - 1
            npen = Btr.shape[1]
            P = np.zeros((1 + pp, 1 + pp))
            if kind == "spectral":
                w = (sv / sv[0]) ** (-2.0 * param)
                P[1 + pp - npen:, 1 + pp - npen:] = np.diag(w)
            else:
                P[1 + pp - npen:, 1 + pp - npen:] = np.eye(npen)
            G, c = Fdtr.T @ Fdtr, Fdtr.T @ y[s - W:s]
            for L in (LAMS if val else [lam]):
                cf = solve(G, c, P, L, pp)
                rr = y[s - W:s] - Fdtr @ cf
                pv = ((Fdte @ cf) ** 2 + float(rr @ rr) / W) * b[a:z]
                if val:
                    rt = rvv[a:z]; m = (rt > 0) & np.isfinite(pv) & (pv > 0)
                    if m.sum():
                        q = rt[m] / pv[m]
                        num[L] += float(np.sum(q - np.log(q) - 1.0)); den[L] += int(m.sum())
                else:
                    out[a - lo:z - lo] = pv
        if val:
            return min((num[L] / den[L], L) for L in LAMS if den[L])[1]
        return out

    ARMS = [("ridge-504", "ridge", None), ("pc-5", "pc", 5), ("pc-20", "pc", 20),
            ("pc-50", "pc", 50), ("spectral-0.5", "spectral", 0.5),
            ("spectral-1.0", "spectral", 1.0), ("op-d5", "op", 5)]
    preds = {}
    for nm, kind, prm in ARMS:
        lam = run(kind, prm, None, None)
        preds[nm] = run(kind, prm, W, n, lam=lam)
        print(f"    {nm:>14} lambda*={lam:<9g} ({time.time()-t0:.0f}s)", flush=True)

    rv_w = rvv[W:]; good = rv_w > 0
    for k in preds: good &= np.isfinite(preds[k]) & (preds[k] > 0)
    idx = np.flatnonzero(good)
    L = {k: qlike_bar(rv_w[idx], preds[k][idx]) for k in preds}
    Q = {k: float(L[k].mean()) for k in L}
    print(f"\n  scored {len(idx):,} bars\n")
    print(f"  {'arm':>14}{'QLIKE':>11}{'vs ridge-504':>14}{'t':>8}")
    out = {"cache": CACHE, "n": int(len(idx)), "qlike": Q, "dm": {}}
    for nm, _, _ in ARMS:
        if nm == "ridge-504":
            print(f"  {nm:>14}{Q[nm]:>11.5f}{'--':>14}{'--':>8}"); continue
        r = dm_test(L[nm], L["ridge-504"], h=1)
        out["dm"][nm] = {"diff": float(r["mean_diff"]), "t": float(r["t"]), "p": float(r["p"])}
        print(f"  {nm:>14}{Q[nm]:>11.5f}{r['mean_diff']:>+14.5f}{r['t']:>8.2f}")
    nb = len(idx) // NBLOCK
    best = min((v, k) for k, v in Q.items())[1]
    print(f"\n  best arm: {best}")
    if best != "ridge-504":
        d = L[best] - L["ridge-504"]
        em = [float(d[i*nb:(i+1)*nb if i < NBLOCK-1 else len(idx)].mean()) for i in range(NBLOCK)]
        print(f"  era means vs ridge-504: {['%+.5f' % v for v in em]}")
        print(f"  better in {sum(1 for v in em if v < 0)}/{NBLOCK} eras")
        out["best_era_means"] = em
    with open(os.path.join(OUT_DIR, "dual_features.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote dual_features.json ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
