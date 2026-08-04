"""What IS the one column, and where does its variance actually come from?

rank1_placebo established that a single column ties the twelve-rung ladder by
three independent routes, but with a wide spread across which window the
direction was estimated on (sd 0.00147, range -0.00161 to +0.00315). Two things
follow that the loss table cannot answer.

WHICH COLUMN. Direction 1 is a weight vector over the twelve rung-MAs, so the
column is a specific weighted blend of sqrt-RV at horizons 1..2048. Printed
here as raw loadings, as the power-law exponent that best matches it, and as
the horizon at which cumulative weight passes half -- the last being the
interpretable summary, since it says what averaging window the ladder is really
asking for.

WHERE THE VARIANCE IS. At rank 1 the design carries one estimated column plus
nine calendar dummies and an intercept on a 24,000-row window, so COEFFICIENT
sampling variance is negligible -- and it is barely less negligible at rank 12.
That matters, because it means the rank-1-ties-rank-12 result cannot be a
bias-variance trade in the coefficients. Two candidate mechanisms are separated:

  RANK. The twelve rungs are nested MAs of one series and carry far less than
  twelve dimensions of information, so columns 2-12 add nothing to estimate.
  Measured by the eigenvalue spectrum and effective rank of the ladder.

  DIRECTION VARIANCE. What variance remains at rank 1 sits entirely in the
  DIRECTION, which is estimated from a finite window and is not shared with the
  coefficient path. Measured by the angular spread of the nine estimated d1
  vectors and by whether that spread predicts realised QLIKE.

If direction variance dominates, the control is not shrinkage on coefficients
-- it is removing the estimation step, which is exactly what an imposed shape
does at a cost of +0.00034.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ladder_refit import (CACHE, RUNGS, W, build_target,  # noqa: E402
                          load_raw_data)
from basis_tracking import seed_basis                     # noqa: E402
from imposed_longmem import consensus                     # noqa: E402

RG = np.array(RUNGS, float)
OUT = os.path.join(ROOT, "writeup", "stats", "rank1_anatomy.json")


def ang(u, v):
    c = abs(float(u @ v) / (np.linalg.norm(u) * np.linalg.norm(v)))
    return float(np.degrees(np.arccos(min(1.0, c))))


def main():
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

    Gs, Cs = [], []
    for k in range(9):
        a0, a1 = k * W, (k + 1) * W
        ok = np.isfinite(H[a0:a1]).all(1) & np.isfinite(y[a0:a1])
        Ht, yt = H[a0:a1][ok], y[a0:a1][ok]
        C = np.cov(Ht, rowvar=False)
        xy = (Ht - Ht.mean(0)).T @ (yt - yt.mean()) / len(yt)
        Gs.append(seed_basis(C, xy, 3, "sup"))
        Cs.append(C)
    GC = consensus(Gs[1:])
    d = [g[:, 0] * np.sign(g[np.argmax(np.abs(g[:, 0])), 0]) for g in Gs]
    dc = GC[:, 0] * np.sign(GC[np.argmax(np.abs(GC[:, 0])), 0])
    out = {"cache": CACHE, "rungs": RUNGS}

    # ---------- WHICH COLUMN
    print("=" * 74)
    print("WHICH COLUMN -- direction-1 weights over the twelve rung-MAs")
    print("=" * 74)
    print("  rung   " + " ".join(f"{r:>7d}" for r in RUNGS))
    print(f"  {'consensus':<7}" + " ".join(f"{x:>7.3f}" for x in dc))
    print(f"  {'k=0':<7}" + " ".join(f"{x:>7.3f}" for x in d[0]))
    lo = np.log(RG)

    def fit_alpha(v):
        """least-squares power-law exponent through the positive weights."""
        m = v > 1e-6
        if m.sum() < 3:
            return float("nan")
        A = np.column_stack([np.ones(m.sum()), lo[m]])
        return float(-np.linalg.lstsq(A, np.log(v[m]), rcond=None)[0][1])

    def half(v):
        w = np.maximum(v, 0.0)
        cw = np.cumsum(w) / w.sum()
        return int(RUNGS[int(np.searchsorted(cw, 0.5))])

    print()
    print(f"  {'basis':>12}{'alpha (r^-a)':>15}{'half-weight rung':>19}"
          f"{'w(1..4)':>10}{'w(256..2048)':>14}")
    rows = {}
    for nm, v in [("consensus", dc)] + [(f"k={k}", d[k]) for k in range(9)]:
        w = np.maximum(v, 0.0)
        w = w / w.sum()
        rows[nm] = {"alpha": fit_alpha(v), "half": half(v),
                    "short": float(w[:3].sum()), "long": float(w[8:].sum()),
                    "loadings": v.tolist()}
        print(f"  {nm:>12}{rows[nm]['alpha']:>15.3f}{rows[nm]['half']:>19d}"
              f"{rows[nm]['short']:>10.3f}{rows[nm]['long']:>14.3f}")
    for a in (0.25, 0.5, 1.0):
        v = RG ** -a
        vn = v / v.sum()
        print(f"  {'imposed r^-'+str(a):>12}{a:>15.3f}{half(v):>19d}"
              f"{vn[:3].sum():>10.3f}{vn[8:].sum():>14.3f}")
    out["direction1"] = rows
    print(f"\n  angle(consensus d1, r^-0.25) = {ang(dc, RG**-0.25):.1f} deg   "
          f"vs r^-1 = {ang(dc, RG**-1.0):.1f} deg")
    out["ang_consensus_to_rm025"] = ang(dc, RG ** -0.25)
    out["ang_consensus_to_rm1"] = ang(dc, RG ** -1.0)

    # ---------- WHERE THE VARIANCE IS
    print()
    print("=" * 74)
    print("WHERE THE VARIANCE IS")
    print("=" * 74)
    Cm = np.mean(Cs, axis=0)
    w = np.sort(np.linalg.eigvalsh(Cm))[::-1]
    share = w / w.sum()
    eff = float(np.exp(-np.sum(share * np.log(share))))
    print(f"\n  ladder spectrum (mean over windows), 12 rungs")
    print(f"    var share  " + " ".join(f"{x:>6.3f}" for x in share))
    print(f"    effective rank (entropy) = {eff:.2f} of 12; "
          f"top direction = {share[0]:.1%}")
    out["eff_rank"] = eff
    out["var_share"] = share.tolist()

    # coefficient sampling variance, rank 1 vs rank 12, same window
    print(f"\n  coefficient sampling variance on W={W:,} rows "
          f"(10 cal+intercept cols in both)")
    for nm, Gm in [("rank 1 (consensus d1)", dc.reshape(-1, 1)),
                   ("rank 12 (raw rungs)", np.eye(12))]:
        tr = []
        for k in range(9):
            a0, a1 = k * W, (k + 1) * W
            ok = np.isfinite(H[a0:a1]).all(1)
            F = H[a0:a1][ok] @ Gm
            A = np.column_stack([np.ones(len(F)), F])
            G = A.T @ A
            tr.append(float(np.trace(np.linalg.inv(G))))
        print(f"    {nm:>24}  tr((A'A)^-1) = {np.mean(tr):.3e}   "
              f"p = {Gm.shape[1]+1}")
        out[f"trAinv_{Gm.shape[1]}"] = float(np.mean(tr))

    # direction variance: angular spread of the nine estimates
    print(f"\n  DIRECTION spread across the nine estimation windows")
    P = [ang(d[i], d[j]) for i in range(9) for j in range(i + 1, 9)]
    print(f"    pairwise angle between d1 vectors: mean {np.mean(P):.1f} deg, "
          f"min {min(P):.1f}, max {max(P):.1f}")
    out["d1_pairwise_angle"] = {"mean": float(np.mean(P)),
                                "min": float(min(P)), "max": float(max(P))}
    ac = [ang(d[k], dc) for k in range(9)]
    try:
        R = json.load(open(os.path.join(ROOT, "writeup", "stats",
                                        "rank1_placebo.json")))
        dd = [R["null"][f"freeze k={k} d1"]["diff"] for k in range(9)]
        print(f"\n    {'k':>3}{'angle to consensus':>21}{'d vs r=12':>12}")
        for k in range(9):
            print(f"    {k:>3}{ac[k]:>21.1f}{dd[k]:>+12.5f}")
        rho = float(np.corrcoef(ac, dd)[0, 1])
        print(f"\n    corr(angle to consensus, QLIKE diff) = {rho:+.3f}")
        out["corr_angle_qlike"] = rho
        out["angle_to_consensus"] = ac
    except (OSError, KeyError) as e:
        print(f"    (rank1_placebo.json unavailable: {e})")

    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
