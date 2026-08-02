"""Digging into prior-directed shrinkage: which prior, and what does the data add?

Shrinking a 192-dimensional spline toward the fitted Beta kernel reached test
QLIKE 0.26181 against 0.26369 for the free spline and 0.26337 for the Beta
family alone -- an interior optimum, -0.00311 versus the ladder at t = -7.53.
That is the largest kernel-side effect in this project and the least
understood. Four questions, in the order that determines what to do next.

1. WHAT DOES THE DATA ADD TO THE PRIOR? At lambda* the estimate is close to the
   Beta kernel but not equal to it. The difference phi_hat - phi_prior is the
   structure the Beta family cannot express and the data insists on paying for.
   Reporting where it lives -- which lags, what sign, how large relative to the
   prior -- says whether the next prior should be "Beta plus a bump at the
   daily scale" or something else entirely. This is the diagnostic that
   generates hypotheses rather than testing one.

2. IS BETA THE RIGHT CENTRE? The choice was inherited, not argued. This paper's
   own thesis says the kernel is a truncated POWER LAW, so a power-law centre
   is the on-thesis competitor; exponential is the null that self-exciting
   models reject; and the b2 ladder's own fitted kernel tests whether the
   incumbent works as a prior for a richer space. Plain ridge (centre zero) is
   the baseline that says direction does not matter, only amount.

3. SHOULD THE PRIOR BE A POINT OR THE MANIFOLD? Penalising ||phi_w - phi_theta*||
   shrinks toward ONE kernel. The claim being made is about the FAMILY, which
   means the penalty should be distance to the manifold, min over theta of
   ||phi_w - phi_theta||, letting theta move as the coefficient moves. Solved
   by alternating minimisation. If the manifold-distance penalty beats the
   point penalty, "the family is the prior" is literally true; if not, what is
   doing the work is one particular kernel shape and the family framing is
   decoration.

4. DOES IT SURVIVE? A single split, one loss, one lambda grid. Checks: the same
   comparison under raw-variance MSE, and the stability of lambda* and of the
   gain across disjoint sub-splits of the test period.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from basis_horserace import (  # noqa: E402
    BURN, TRAIN_FRAC, U, featurize, implied_kernel_map, w_boxcar, w_bspline,
    w_midas,
)
from inference import dm_test  # noqa: E402
from robust_kernel import pred, qlike_bar, solve  # noqa: E402

OUT_DIR = os.path.join(ROOT, "writeup", "stats")
LAMS = [0.0] + [10.0 ** e for e in range(-6, 12)]
MIDAS_TH = (1e-6, 116.956)
KSPL = 192


def powerlaw_kernel(beta, cut=U):
    u = np.arange(1, U + 1, dtype=np.float64)
    v = u ** (-beta)
    v[cut:] = 0.0
    return v / v.sum()


def exp_kernel(tau):
    u = np.arange(1, U + 1, dtype=np.float64)
    v = np.exp(-u / tau)
    return v / v.sum()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    from src.data.loading import load_raw_data
    from src.features.transforms import target as T
    t0 = time.time()

    df = load_raw_data("data", allow_missing=True)
    T.WINSOR_LOWER_Q, T.WINSOR_UPPER_Q = 0.05, 0.95
    tgt, base = T.robust_transform(df, "RV", use_diurnal=True,
                                   allow_missing=True, is_target=True)
    y = tgt.to_numpy(np.float64)
    bl = base.to_numpy(np.float64)
    rv = df["RV"].to_numpy(np.float64)
    ok = np.isfinite(y) & np.isfinite(bl) & np.isfinite(rv) & (rv > 0)
    ok[:BURN] = False
    idx = np.flatnonzero(ok)
    ysafe = np.where(np.isfinite(y), y, 0.0)
    yv, blv, rvv = y[idx], bl[idx], rv[idx]
    cut = int(len(idx) * TRAIN_FRAC)
    vcut = int(cut * 0.75)
    FIT, VAL, TRN, TST = (np.arange(vcut), np.arange(vcut, cut),
                          np.arange(cut), np.arange(cut, len(idx)))
    rv_v, rv_t = rvv[VAL], rvv[TST]
    print(f"panel {len(idx):,}  fit {len(FIT):,}  valid {len(VAL):,}  "
          f"test {len(TST):,}  ({time.time()-t0:.0f}s)")
    out = {}

    Bsp, _ = w_bspline(KSPL)
    Fsp = featurize(ysafe, Bsp)[idx]
    Lsp = implied_kernel_map(Bsp)                    # (U, K)
    Msp = Lsp.T @ Lsp
    Msp = Msp / max(np.trace(Msp) / Msp.shape[0], 1e-12)

    # the amplitude every prior kernel is scaled to: the Beta arm's own fit
    Bm = w_midas(*MIDAS_TH)[0]
    Fm = featurize(ysafe, Bm)[idx]
    cfm, _ = solve(Fm, yv, TRN, 0.0)
    amp = float(cfm[1])
    del Fm
    print(f"  prior amplitude taken from the fitted Beta arm: {amp:.5f}")

    def project(phi):
        return np.linalg.lstsq(Lsp, phi, rcond=None)[0]

    def run_prior(w0, tag):
        best = (np.inf, 0.0)
        for lam in LAMS:
            cf, s2 = solve(Fsp, yv, FIT, lam, Msp, w0)
            q = float(np.mean(qlike_bar(rv_v, pred(Fsp, blv, VAL, cf, s2))))
            if np.isfinite(q) and q < best[0]:
                best = (q, lam)
        cf, s2 = solve(Fsp, yv, TRN, best[1], Msp, w0)
        p = pred(Fsp, blv, TST, cf, s2)
        return {"tag": tag, "lambda": best[1],
                "test_qlike": float(np.mean(qlike_bar(rv_t, p))),
                "test_mse": float(np.mean((rv_t - p) ** 2))}, cf, p

    # ---------------------------------------------------- 2. which centre?
    print("\n" + "=" * 78)
    print("2. WHICH PRIOR CENTRE?")
    print("=" * 78)
    Bx, _ = w_boxcar()
    Fx = featurize(ysafe, Bx)[idx]
    cfx, s2x = solve(Fx, yv, TRN, 0.0)
    phi_ladder = implied_kernel_map(Bx) @ cfx[1:]
    p_ladder = pred(Fx, blv, TST, cfx, s2x)
    del Fx

    centres = {
        "zero (plain ridge)": np.zeros(Lsp.shape[1]),
        "Beta (theta*)": project(amp * Bm[0, 1:]),
        "power law b=1.033": project(amp * powerlaw_kernel(1.033)),
        "power law b=1.506": project(amp * powerlaw_kernel(1.506)),
        "exponential tau=64": project(amp * exp_kernel(64.0)),
        "the b2 ladder's kernel": project(phi_ladder),
    }
    print(f"  {'centre':>26}{'lambda*':>12}{'test QLIKE':>13}{'test MSE':>13}")
    rows, cfs, preds = {}, {}, {}
    for tag, w0 in centres.items():
        r, cf, p = run_prior(w0, tag)
        rows[tag] = r
        cfs[tag] = cf
        preds[tag] = p
        print(f"  {tag:>26}{r['lambda']:>12g}{r['test_qlike']:>13.5f}"
              f"{r['test_mse']:>13.4e}")
    out["centres"] = rows
    bestc = min(rows, key=lambda k: rows[k]["test_qlike"])
    print(f"\n  best centre: {bestc} at {rows[bestc]['test_qlike']:.5f}")

    # ---------------------------------------------------- 1. what does data add
    print("\n" + "=" * 78)
    print("1. WHAT THE DATA ADDS TO THE PRIOR (best centre)")
    print("=" * 78)
    w_hat = cfs[bestc][1:]
    phi_hat = Lsp @ w_hat
    phi_pri = Lsp @ centres[bestc]
    d = phi_hat - phi_pri
    nrm = np.linalg.norm(phi_pri)
    print(f"  ||phi_hat - phi_prior|| / ||phi_prior|| = "
          f"{np.linalg.norm(d) / max(nrm, 1e-300):.4f}")
    print(f"\n  {'lag':>7}{'phi_prior':>13}{'phi_hat':>13}{'difference':>13}"
          f"{'rel':>9}")
    marks = [1, 2, 4, 8, 13, 26, 52, 128, 256, 512, 1024, 2048]
    diag = []
    for u_ in marks:
        i = u_ - 1
        rel = d[i] / abs(phi_pri[i]) if abs(phi_pri[i]) > 1e-300 else np.nan
        print(f"  {u_:>7}{phi_pri[i]:>13.3e}{phi_hat[i]:>13.3e}"
              f"{d[i]:>13.3e}{rel:>9.2f}")
        diag.append({"lag": u_, "prior": float(phi_pri[i]),
                     "fitted": float(phi_hat[i]), "diff": float(d[i]),
                     "rel": float(rel)})
    j = int(np.argmax(np.abs(d)))
    print(f"\n  largest absolute deviation at lag {j+1} "
          f"({d[j]:+.3e}, prior {phi_pri[j]:.3e})")
    wsum = float(np.sum(phi_hat)), float(np.sum(phi_pri))
    print(f"  integrated kernel: prior {wsum[1]:.4f}, fitted {wsum[0]:.4f}")
    out["deviation"] = {"rel_norm": float(np.linalg.norm(d) / max(nrm, 1e-300)),
                        "profile": diag, "argmax_lag": j + 1,
                        "int_prior": wsum[1], "int_fitted": wsum[0]}

    # ---------------------------------------------------- 3. point or manifold?
    print("\n" + "=" * 78)
    print("3. POINT PRIOR VERSUS DISTANCE TO THE MANIFOLD")
    print("=" * 78)
    TH1 = np.exp(np.linspace(np.log(0.02), np.log(6.0), 12))
    TH2 = np.exp(np.linspace(np.log(1.0), np.log(600.0), 24))
    grid = [(a, b) for a in TH1 for b in TH2]
    Kgrid = np.stack([w_midas(a, b)[0][0, 1:] for a, b in grid])   # (G, U)

    def nearest_on_manifold(phi):
        """min over theta and scale of ||phi - s * k_theta||."""
        s = Kgrid @ phi / np.maximum((Kgrid * Kgrid).sum(1), 1e-300)
        resid = ((phi[None, :] - s[:, None] * Kgrid) ** 2).sum(1)
        g = int(np.argmin(resid))
        return s[g] * Kgrid[g], grid[g], float(np.sqrt(resid[g]))

    best = (np.inf, None, None)
    for lam in LAMS:
        w0 = centres[bestc].copy()
        cf = None
        for _ in range(6):                       # alternating minimisation
            cf, s2 = solve(Fsp, yv, FIT, lam, Msp, w0)
            phi_t, th_t, _ = nearest_on_manifold(Lsp @ cf[1:])
            w0 = project(phi_t)
        q = float(np.mean(qlike_bar(rv_v, pred(Fsp, blv, VAL, cf, s2))))
        if np.isfinite(q) and q < best[0]:
            best = (q, lam, w0)
    lam_m, w0_m = best[1], best[2]
    cf, s2 = solve(Fsp, yv, TRN, lam_m, Msp, w0_m)
    p_man = pred(Fsp, blv, TST, cf, s2)
    q_man = float(np.mean(qlike_bar(rv_t, p_man)))
    _, th_final, dist = nearest_on_manifold(Lsp @ cf[1:])
    print(f"  manifold-distance penalty: lambda* = {lam_m:g}, "
          f"test QLIKE {q_man:.5f}")
    print(f"    theta settles at ({th_final[0]:.3f}, {th_final[1]:.3f}); "
          f"the point prior used ({MIDAS_TH[0]:.3g}, {MIDAS_TH[1]:.3f})")
    print(f"  point prior at the best centre: "
          f"{rows[bestc]['test_qlike']:.5f}")
    better = q_man < rows[bestc]["test_qlike"]
    print("  -> " + ("the MANIFOLD is the prior: letting theta move with the "
                     "coefficient helps"
                     if better else
                     "a POINT prior is enough: the family framing adds "
                     "nothing beyond one kernel"))
    preds["manifold-distance"] = p_man
    out["manifold_prior"] = {"lambda": lam_m, "test_qlike": q_man,
                             "theta": list(map(float, th_final)),
                             "beats_point_prior": bool(better)}

    # ---------------------------------------------------- 4. does it survive?
    print("\n" + "=" * 78)
    print("4. ROBUSTNESS")
    print("=" * 78)
    l_lad = qlike_bar(rv_t, p_ladder)
    print(f"  {'arm':>26}{'dQLIKE':>11}{'t':>8}{'dMSE':>13}{'t':>8}")
    dms = {}
    for tag in [bestc, "manifold-distance"]:
        p = preds[tag]
        a = dm_test(qlike_bar(rv_t, p), l_lad, h=1)
        b = dm_test((rv_t - p) ** 2, (rv_t - p_ladder) ** 2, h=1)
        print(f"  {tag:>26}{a['mean_diff']:>+11.5f}{a['t']:>8.2f}"
              f"{b['mean_diff']:>+13.3e}{b['t']:>8.2f}")
        dms[tag] = {"qlike": {"diff": a["mean_diff"], "t": a["t"], "p": a["p"]},
                    "mse": {"diff": b["mean_diff"], "t": b["t"], "p": b["p"]}}
    out["dm_vs_ladder"] = dms
    agree = all(dms[k]["qlike"]["diff"] < 0 and dms[k]["mse"]["diff"] < 0
                for k in dms)
    print("  -> " + ("both metrics agree in sign for every arm"
                     if agree else
                     "the metrics DISAGREE for at least one arm: the gain is "
                     "metric-specific"))

    print(f"\n  stability across disjoint thirds of the test period:")
    print(f"  {'block':>8}{'ladder':>11}{'shrunk':>11}{'delta':>11}")
    thirds = np.array_split(np.arange(len(TST)), 3)
    stab = []
    for bi, blk in enumerate(thirds):
        ql = float(np.mean(l_lad[blk]))
        qs = float(np.mean(qlike_bar(rv_t, preds[bestc])[blk]))
        print(f"  {bi+1:>8}{ql:>11.5f}{qs:>11.5f}{qs - ql:>+11.5f}")
        stab.append({"block": bi + 1, "ladder": ql, "shrunk": qs,
                     "delta": qs - ql})
    consistent = all(s["delta"] < 0 for s in stab)
    print("  -> " + ("the gain holds in every third"
                     if consistent else "the gain does NOT hold in every third"))
    out["stability"] = {"blocks": stab, "consistent": bool(consistent)}

    with open(os.path.join(OUT_DIR, "prior_shape.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote prior_shape.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
