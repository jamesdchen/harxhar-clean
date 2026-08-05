"""Quadratics of the top feature principal components — the frame reading's constructive test.

The random-vs-IC probe established the structure of the product dictionary: the 133 base columns
have effective rank ~18 (top-5 eigenvalues = 41% of trace), so the 8,911 products are redundant
probes of a factor-quadratic space; uniform-random 100 capture ~56% of the IC-selected gain (huge
seed variance: 26/86/56%), so probes are region-interchangeable but SNR-heterogeneous.

Random-features theory then says the *proper* importance sampling for a decaying spectrum is by
ridge leverage score, with two requirements the first-window IC selection does not meet: the
weights come from X alone (no target → no selection bias, no OOS budget spent validating the
choice), and sampled features are reweighted by ``1/sqrt(m p_i)`` so the kernel estimate stays
unbiased. On *standardized* original features, though, per-product leverage is nearly flat
(second moment of a product of standardized vars is ``1 + 2 rho²`` ∈ [1,3]) — the uniform-random
arm effectively IS flat-leverage sampling, which is why it got half. The importance ordering that
matters lives in the PC basis, where the kernel operator's spectrum is ``lambda_a lambda_b`` —
steep — and its deterministic limit is simply: **take the quadratics of the top-q PCs**.

So this runs base + IC-100 + PC-quadratics at q = 6 / 10 / 15 (21 / 55 / 120 columns), identical
frozen monthly protocol (projection frozen on the first training window; per-window centering;
floored product scale; blockwise penalty). Pre-registered readings:

* pcq ≈ IC-100's +0.0069 at a third of the columns → the frame reading is complete; the signal
  sits in the top factor-quadratics and supervised selection was only ever recovering leverage.
* pcq ≈ random's ~half → the signal's location *within* the factor-quadratic space requires
  supervision (IC carries target information leverage cannot), and the clean parameterization
  fails. Either answer settles what the IC selector was actually buying.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.alpha_manifestation import TW, dm_test  # noqa: E402
from analysis.alpha_panel import load_panel  # noqa: E402
from analysis.nl_sparsity import REFIT, _pair_ic, _products, _upper, base_columns  # noqa: E402
from analysis.synthesis import ALPHA_PROD, _blockwise_ridge, _floored_scale, _p  # noqa: E402
from analysis.wf import r2_oos  # noqa: E402

OUT = "results/alpha_manifestation"
QS = (6, 10, 15)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    p = load_panel()
    e_full = np.load(_p("har_resid.npz"))["e"]
    bc, _ = base_columns(p)
    X = np.ascontiguousarray(p.X[TW:, bc], dtype=np.float64)
    n, pb = X.shape
    y = e_full[TW:]
    ii, jj = _upper(pb)

    # frozen first-window PCA (causal: nothing after row TW is touched). Columns that are
    # constant in the first window (features not yet live in 2005-06: stocktwits, vvix, vix3m)
    # carry no first-window information and are excluded from the projection with zero weight.
    mu0, sd0 = X[:TW].mean(0), X[:TW].std(0)
    live = sd0 > 1e-8
    print(f"first-window live columns: {int(live.sum())} / {len(live)}", flush=True)
    sd0 = np.where(live, sd0, 1.0)
    lam_l, V_l = np.linalg.eigh(
        np.corrcoef(((X[:TW] - mu0) / sd0)[:, live], rowvar=False)
    )
    order = np.argsort(lam_l)[::-1]
    lam = lam_l[order]
    V = np.zeros((X.shape[1], len(lam)))
    V[live] = V_l[:, order]
    print(f"first-window spectrum: top-{max(QS)} shares "
          + " ".join(f"{v:.3f}" for v in lam[: max(QS)] / lam.sum()), flush=True)

    def walk(product_fn, n_prod, post_w=None):
        base = np.full(n - TW, np.nan)
        aug = np.full(n - TW, np.nan)
        for t0 in range(TW, n, REFIT):
            tr = slice(t0 - TW, t0)
            t1 = min(t0 + REFIT, n)
            out = slice(t0 - TW, t1 - TW)
            mu = X[tr].mean(0)
            Ztr, Zte = X[tr] - mu, X[t0:t1] - mu
            ytr = e_full[tr]
            base[out] = _blockwise_ridge(Ztr, ytr, Zte, pb, 3000.0, 3000.0)
            Ptr, Pte = _floored_scale(product_fn(Ztr), product_fn(Zte))
            if post_w is not None:  # graded penalty: applied AFTER the sd normalization
                Ptr = Ptr * post_w
                Pte = Pte * post_w
            aug[out] = _blockwise_ridge(
                np.hstack([Ztr, Ptr]), ytr, np.hstack([Zte, Pte]), pb, 3000.0, ALPHA_PROD
            )
        return base, aug

    # reference arm: the frozen IC-100 (§16.3's stat100), recomputed in-loop for exactness
    ic = np.abs(np.nan_to_num(_pair_ic(X[:TW] - mu0, e_full[:TW])[ii, jj]))
    sel = np.argsort(-ic)[:100]
    base, aug_ic = walk(lambda Z: _products(Z, ii[sel], jj[sel]), 100)
    rb = r2_oos(y, base)
    ri = r2_oos(y, aug_ic)
    print(f"\nbase {rb:+.5f}   IC-100 dR2 {ri - rb:+.5f}  DM {dm_test(y, aug_ic, base):+.2f}",
          flush=True)

    rows = [{"arm": "ic100", "n_cols": 100, "dr2": ri - rb,
             "dm_vs_base": dm_test(y, aug_ic, base), "dm_vs_ic": 0.0}]
    for q in QS:
        W = V[:, :q] / sd0[:, None]  # project raw-centered rows: (X - mu) @ W
        qi, qj = np.triu_indices(q)

        def pc_products(Z, W=W, qi=qi, qj=qj):
            G = Z @ W
            return G[:, qi] * G[:, qj]

        _, aug = walk(pc_products, len(qi))
        r = r2_oos(y, aug)
        print(f"PC-{q:2d} quadratics ({len(qi):3d} cols): dR2 {r - rb:+.5f}  "
              f"DM vs base {dm_test(y, aug, base):+.2f}  vs IC-100 {dm_test(y, aug, aug_ic):+.2f}  "
              f"fraction of IC gain {100 * (r - rb) / (ri - rb):.0f}%", flush=True)
        rows.append({"arm": f"pc{q}", "n_cols": int(len(qi)), "dr2": r - rb,
                     "dm_vs_base": dm_test(y, aug, base), "dm_vs_ic": dm_test(y, aug, aug_ic)})
    # --- soft thresholding: all pairs of a generous top-20 pool, penalty graded by the spectrum.
    # Column weight (lambda_a lambda_b)^(gamma/2) => effective per-column penalty
    # alpha_prod / (lambda_a lambda_b)^gamma: gamma = 0 is a flat prior over the pool (pure
    # breadth), gamma = 1 is the kernel-natural decay, gamma = 1/2 the compromise. Weights are
    # normalized to median 1 so the effective overall penalty stays comparable across arms.
    # Pre-registered readings: if the hard cutoff's exclusion was the q=10 trap's cause, the flat
    # pool recovers PC-15's level or better; if eigenvalue decay helps beyond that, gamma > 0
    # beats gamma = 0; if the mid-spectrum misalignment dominates, gamma = 1 UNDER-performs
    # gamma = 0 (it discounts the signal-bearing 11-15 below the noisy 7-10).
    # NOTE the first implementation applied the weight to the RAW product column, which
    # _floored_scale's per-window sd normalization cancels exactly — all three gammas came back
    # bit-identical (each at 92% of IC), which is itself the diagnosis. The weight must multiply
    # the column AFTER the sd normalization, where it sets the effective per-column penalty
    # alpha / w^2. ``walk`` therefore takes an optional post-scale weight.
    QPOOL = 20
    qi, qj = np.triu_indices(QPOOL)
    lw = lam[:QPOOL]
    for gamma in (0.0, 0.5, 1.0):
        w = (lw[qi] * lw[qj]) ** (gamma / 2.0)
        w = w / np.median(w)
        Wp = V[:, :QPOOL] / sd0[:, None]

        def soft_products(Z, Wp=Wp, qi=qi, qj=qj):
            G = Z @ Wp
            return G[:, qi] * G[:, qj]

        _, aug = walk(soft_products, len(qi), post_w=w)
        r = r2_oos(y, aug)
        print(f"soft gamma={gamma:.1f} (pool 20, {len(qi)} cols): dR2 {r - rb:+.5f}  "
              f"DM vs base {dm_test(y, aug, base):+.2f}  vs IC-100 {dm_test(y, aug, aug_ic):+.2f}  "
              f"fraction {100 * (r - rb) / (ri - rb):.0f}%", flush=True)
        rows.append({"arm": f"soft_g{gamma}", "n_cols": int(len(qi)), "dr2": r - rb,
                     "dm_vs_base": dm_test(y, aug, base), "dm_vs_ic": dm_test(y, aug, aug_ic)})

    import pandas as pd

    pd.DataFrame(rows).to_csv(f"{OUT}/pc_quadratics.csv", index=False)
    print(f"wrote {OUT}/pc_quadratics.csv")


if __name__ == "__main__":
    main()
