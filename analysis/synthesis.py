"""The §15.3 action list: do the two confirmed gains add, and does the one nominated
state-dependent term survive a fresher split?

Three stages here, all on the **§8-fixed panel** (the one where ``voldemand``'s transform is
repaired), because every earlier interaction number was measured either with ``voldemand``
dropped or behind a +-4 clip that existed to survive it.

``prep``
    Rebuild the two cached intermediates — the HAR walk-forward residual ``e`` and the seven
    per-bucket walk-forward signals ``S`` — against the fixed panel, so the rest of this module
    is not silently reading pre-fix artifacts. Written into the *fixed* cache directory, so run
    this module with ``ALPHA_PANEL_CACHE`` pointing there.

``interactions`` (§15.3 item 5)
    Re-run the interaction study on the fixed panel with **no clip anywhere** and ``voldemand``
    *included*. §7.3 had to drop it and had to pick a product-block penalty by grid; the fix
    should make the column usable, and the grid is re-reported rather than re-chosen.

``additivity`` (§15.3 item 1)
    The two gains that survived Romano-Wolf — the vol-regime 50/50 blend (+0.0024) and the ~100
    frozen interaction products (+0.0067) — have never been run in the same model. This stage
    puts both on one base and one row set so the arithmetic is exact.

``sentiment`` (§15.3 item 2)
    §14.3 nominated ``sentiment x vol-state`` out of seven candidates, on a sample that already
    included 2021-24 (a block scored twice before). Here the *selection* is redone on <=2019 only
    and the winner is scored, 1 df, on 2020+.

Pre-registration
----------------
Written before any of it was run, because the whole point of §10 was that this study has spent
its OOS budget and cannot settle anything by bake-off:

* ``interactions`` — expectation: ``stat100`` still beats base and ``dyn100`` still does not beat
  ``stat100`` (selection does not rotate). The fix should make the result *less* penalty-sensitive,
  since a large part of §7.3's sensitivity was one column with |z| in the hundreds.
* ``additivity`` — expectation **sub-additive**. The frozen product set is drawn from a space that
  contains ``har_ma_w x exog`` terms, which is vol-state conditioning in a different
  parameterization; the two gains should overlap. The discriminating statistic is therefore *not*
  the ratio (full redundancy already predicts 0.0067/0.0091 = 0.74, too close to additive to
  separate) but the **DM-t of the combined model against the better single arm**. Declared
  threshold: |t| > 2. Prediction: positive but under 2.
* ``sentiment`` — expectation: the <=2019 search picks ``sentiment`` (it was the only |t| > 2 in
  §14.3 and positive in both halves of the search sample), and the 2020+ score is positive but
  small. One candidate, one df, one-sided DM at 1.645 because §14.3 fixed the sign.

Nothing here selects a specification by OOS score. ``interactions`` reports a pre-existing grid,
``additivity`` has no free parameter, and ``sentiment`` freezes its choice on data disjoint from
the block it is scored on.

Usage
-----
    C=/tmp/claude-0/-home-user-harxhar-clean/8d6e56a6-934c-5a06-a38d-8fa3ada92453/scratchpad/fixed
    ALPHA_PANEL_CACHE=$C python analysis/synthesis.py --stage prep
    ALPHA_PANEL_CACHE=$C python analysis/synthesis.py --stage interactions
    ALPHA_PANEL_CACHE=$C python analysis/synthesis.py --stage additivity
    ALPHA_PANEL_CACHE=$C python analysis/synthesis.py --stage sentiment
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.alpha_manifestation import (  # noqa: E402
    BUCKETS,
    COMBINE_WINDOW_DAYS,
    REFIT_EVERY,
    RIDGE_ALPHA,
    TW,
    _bin_walk_forward,
    _vol_regime,
    dm_test,
)
from analysis.alpha_panel import CACHE_DIR, load_panel  # noqa: E402
from analysis.nl_sparsity import REFIT, _pair_ic, _products, _upper, base_columns  # noqa: E402
from analysis.wf import r2_oos, walk_forward  # noqa: E402
from src.features.transforms.target import PERIODS_PER_DAY  # noqa: E402

OUT = "results/alpha_manifestation"

# The settled §7.3 configuration, restated as constants so ``additivity`` cannot drift from
# ``interactions``: 100 products, chosen once on the first training window and frozen forever,
# scaled by a *floored* window sd, and penalized separately from the linear block.
N_PROD = 100
ALPHA_PROD = 3e4  # the §7.3 grid point where the frozen-product gain was +0.0067 / DM-t 2.71
HOLDOUT = "2020-01-01"  # §15.3 item 2: selection <= 2019, score 2020+


def _p(name: str) -> str:
    return os.path.join(CACHE_DIR, name)


def _require_fixed_cache() -> None:
    """Guard against running the fixed-panel stages against the pre-fix cache by accident."""
    if os.path.basename(os.path.normpath(CACHE_DIR)) != "fixed":
        raise SystemExit(
            "refusing to run: ALPHA_PANEL_CACHE must point at the .../fixed cache so these "
            f"stages read the §8-fixed panel and its own intermediates (got {CACHE_DIR})"
        )


# ---------------------------------------------------------------------------
# prep — the residual and the bucket signals, rebuilt on the fixed panel
# ---------------------------------------------------------------------------


def stage_prep() -> None:
    _require_fixed_cache()
    p = load_panel()
    print(f"fixed panel: X {p.X.shape}  |X| max {np.abs(p.X).max():.1f}", flush=True)

    har = np.concatenate([p.cols("har"), p.cols("calendar"), p.cols("regime")])
    pred = walk_forward(p.X[:, har].astype(np.float64), p.y, TW, alpha=1.0, refit_every=1)
    e = p.y[TW:] - pred
    np.savez_compressed(_p("har_resid.npz"), e=e, pred=pred, tw=TW)
    yc = p.y[TW:] - p.y[TW:].mean()
    print(
        f"HAR baseline: {len(har)} cols, R2 {r2_oos(yc, pred - p.y[TW:].mean()):.4f}, "
        f"resid std {e.std():.4f}, n_oos {len(e)}",
        flush=True,
    )

    sig: dict[str, np.ndarray] = {}
    for b in BUCKETS + ["all"]:
        cols = (
            np.concatenate([p.cols("value"), p.cols("indicator")])
            if b == "all"
            else np.concatenate([p.cols("value", b), p.cols("indicator", b)])
        )
        s = walk_forward(
            p.X[TW:, cols].astype(np.float64), e, TW, alpha=RIDGE_ALPHA, refit_every=REFIT_EVERY
        )
        sig[b] = s
        print(
            f"{b:12s} p={len(cols):4d}  OOS corr {np.corrcoef(s, e[TW:])[0, 1]:+.4f}  "
            f"R2 {r2_oos(e[TW:], s):+.5f}",
            flush=True,
        )
    np.savez_compressed(_p("bucket_signals.npz"), **sig)
    print(f"wrote {_p('har_resid.npz')} + {_p('bucket_signals.npz')}")


# ---------------------------------------------------------------------------
# the interaction block, as one reusable walk-forward
# ---------------------------------------------------------------------------


def _floored_scale(Ptr: np.ndarray, Pte: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scale a product block by its window sd, floored at 10% of the block's median sd.

    The floor is the repo's own cure for the Part-3 degenerate-divisor disease: a product whose
    window sd momentarily collapses would otherwise be multiplied into the hundreds.
    """
    sd = Ptr.std(0)
    sd = np.maximum(sd, 0.1 * np.median(sd[sd > 0]) if (sd > 0).any() else 1.0)
    return Ptr / sd, Pte / sd


def _blockwise_ridge(
    Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, n_base: int, a_base: float, a_prod: float
) -> np.ndarray:
    pen = np.concatenate(
        [np.full(n_base, a_base), np.full(Xtr.shape[1] - n_base, a_prod)]
    )
    b = np.linalg.solve(Xtr.T @ Xtr + np.diag(pen), Xtr.T @ (ytr - ytr.mean()))
    return Xte @ b + ytr.mean()


def _interaction_walk(
    X: np.ndarray,
    e: np.ndarray,
    alphas: tuple[float, ...],
    dynamic: bool = False,
) -> tuple[np.ndarray, dict[float, np.ndarray], list[np.ndarray]]:
    """Monthly-refit base and product-augmented predictions, no clipping anywhere.

    ``X`` is already the production rolling-robust-scaled panel, so each window only needs
    centering; the products then get :func:`_floored_scale` and their own ridge penalty.
    Returns ``(base_pred, {alpha: aug_pred}, selection_history)``.
    """
    n = len(e)
    ii, jj = _upper(X.shape[1])
    base = np.full(n - TW, np.nan)
    aug = {a: np.full(n - TW, np.nan) for a in alphas}
    hist: list[np.ndarray] = []
    frozen: np.ndarray | None = None
    for t0 in range(TW, n, REFIT):
        tr = slice(t0 - TW, t0)
        t1 = min(t0 + REFIT, n)
        out = slice(t0 - TW, t1 - TW)
        mu = X[tr].mean(0)
        Ztr, Zte = X[tr] - mu, X[t0:t1] - mu
        ytr = e[tr]
        nb = Ztr.shape[1]
        base[out] = _blockwise_ridge(Ztr, ytr, Zte, nb, RIDGE_ALPHA, RIDGE_ALPHA)
        sel = np.argsort(-np.abs(np.nan_to_num(_pair_ic(Ztr, ytr)[ii, jj])))[:N_PROD]
        hist.append(np.sort(sel))
        if frozen is None:
            frozen = sel.copy()
        use = sel if dynamic else frozen
        Ptr, Pte = _floored_scale(
            _products(Ztr, ii[use], jj[use]), _products(Zte, ii[use], jj[use])
        )
        Atr, Ate = np.hstack([Ztr, Ptr]), np.hstack([Zte, Pte])
        for a in alphas:
            aug[a][out] = _blockwise_ridge(Atr, ytr, Ate, nb, RIDGE_ALPHA, a)
    return base, aug, hist


# ---------------------------------------------------------------------------
# item 5 — the interaction study on the fixed panel, no clip
# ---------------------------------------------------------------------------


def stage_interactions() -> None:
    """§15.3 item 5. §7's interaction study, redone where the reason for its patches is gone.

    §7.3 had to drop ``voldemand`` (a 2023 0DTE level break putting |z| in the hundreds) and
    §7.1's ``+-4`` clip existed to survive exactly that class of column. §8 fixed the transform,
    so the study can now run on the full base with no clip and nothing dropped. Same grid, same
    frozen-vs-dynamic contrast, so the numbers are comparable to ``nl_group_penalty.csv``.
    """
    _require_fixed_cache()
    os.makedirs(OUT, exist_ok=True)
    p = load_panel()
    e = np.load(_p("har_resid.npz"))["e"]
    bc, bn = base_columns(p)
    X = np.ascontiguousarray(p.X[TW:, bc], dtype=np.float64)
    vd = [nm for nm in bn if "voldemand" in nm]
    print(
        f"base {X.shape[1]} cols incl. {len(vd)} voldemand, max |X| {np.abs(X).max():.1f}, no clip",
        flush=True,
    )

    grid = (3e3, 3e4, 3e5, 3e6)
    base, stat, hist = _interaction_walk(X, e, grid, dynamic=False)
    _, dyn, _ = _interaction_walk(X, e, grid, dynamic=True)
    y = e[TW:]
    rb = r2_oos(y, base)
    churn = float(
        np.mean([1.0 - len(np.intersect1d(a, b)) / N_PROD for a, b in zip(hist[:-1], hist[1:])])
    )
    print(f"  base (no products): R2 {rb:+.5f}   monthly selection churn {churn:.3f}", flush=True)
    rows = [{"alpha_prod": np.nan, "arm": "base", "r2": rb, "dr2": 0.0, "dm_t_vs_base": 0.0}]
    for a in grid:
        rd, rs = r2_oos(y, dyn[a]), r2_oos(y, stat[a])
        td, tsq = dm_test(y, dyn[a], base), dm_test(y, stat[a], base)
        tds = dm_test(y, dyn[a], stat[a])
        print(
            f"  alpha_prod {a:8.0f}: stat{N_PROD} {rs:+.5f} (dR2 {rs - rb:+.5f} DM-t {tsq:+5.2f}) | "
            f"dyn{N_PROD} {rd:+.5f} (dR2 {rd - rb:+.5f} DM-t {td:+5.2f}) | dyn-vs-stat DM-t {tds:+5.2f}",
            flush=True,
        )
        rows += [
            {"alpha_prod": a, "arm": f"stat{N_PROD}", "r2": rs, "dr2": rs - rb,
             "dm_t_vs_base": tsq, "dm_t_dyn_vs_stat": tds, "churn": churn},
            {"alpha_prod": a, "arm": f"dyn{N_PROD}", "r2": rd, "dr2": rd - rb,
             "dm_t_vs_base": td, "dm_t_dyn_vs_stat": tds, "churn": churn},
        ]
    pd.DataFrame(rows).to_csv(f"{OUT}/fixed_interactions.csv", index=False)
    print(f"wrote {OUT}/fixed_interactions.csv")


# ---------------------------------------------------------------------------
# item 1 — do the two confirmed gains add?
# ---------------------------------------------------------------------------


def _prod_signal(p, e: np.ndarray) -> np.ndarray:
    """The *incremental* interaction forecast: augmented minus base, at ``ALPHA_PROD``.

    Taking the difference rather than the augmented level is what makes the additivity arithmetic
    interpretable — the base part of the augmented model overlaps heavily with the bucket signals,
    and only the increment is the thing §7 established.
    """
    bc, _ = base_columns(p)
    X = np.ascontiguousarray(p.X[TW:, bc], dtype=np.float64)
    base, aug, _ = _interaction_walk(X, e, (ALPHA_PROD,), dynamic=False)
    return aug[ALPHA_PROD] - base


def _combiner(S: np.ndarray, y: np.ndarray, cw: int) -> np.ndarray:
    """Pooled second-stage combiner, NaN over its own warm-up."""
    out = np.full(len(y), np.nan)
    out[cw:] = walk_forward(S, y, cw, alpha=1.0, refit_every=REFIT_EVERY)
    return out


def stage_additivity() -> None:  # noqa: C901
    """§15.3 item 1. Both confirmed gains, on one base and one row set.

    Four arms, all predicting the HAR residual and all scored on the intersection of rows every
    arm can predict:

    * ``M0``   pooled combiner over the 7 bucket signals — the base both gains were measured
      against (the vol-regime blend directly, the product gain after rescaling to this base).
    * ``M_A``  50/50 blend of ``M0`` with a vol-regime-conditional combiner (the §6 gain).
    * ``M_B``  pooled combiner over the 7 signals **plus** the frozen-product increment (the §7 gain).
    * ``M_AB`` both: 50/50 blend of the 8-signal pooled combiner with its vol-regime-conditional twin.

    The headline is not the ratio ``dR2(AB) / (dR2(A) + dR2(B))``: complete redundancy already
    predicts 0.74 of it, which is not far enough from 1.0 to decide anything. The test is the
    **Diebold-Mariano t of AB against whichever single arm is better** — i.e. does the second gain
    still pay once the first is in the model. Pre-registered threshold |t| > 2.
    """
    _require_fixed_cache()
    os.makedirs(OUT, exist_ok=True)
    p = load_panel()
    e_full = np.load(_p("har_resid.npz"))["e"]
    e = e_full[TW:]
    sig = dict(np.load(_p("bucket_signals.npz")))
    ts = pd.Series(pd.to_datetime(p.t[TW + TW :]))
    S = np.column_stack([sig[b] for b in BUCKETS])
    cw = COMBINE_WINDOW_DAYS * PERIODS_PER_DAY

    sp = _prod_signal(p, e_full)
    print(
        f"product increment signal: sd {sp.std():.5f}  corr with e {np.corrcoef(sp, e)[0, 1]:+.4f}",
        flush=True,
    )
    S8 = np.column_stack([S, sp])

    rows = []
    for K in (2, 3, 5):
        m0 = _combiner(S, e, cw)
        mb = _combiner(S8, e, cw)
        bins = _vol_regime(p, ts, K)
        ca = _bin_walk_forward(S, e, bins, K, COMBINE_WINDOW_DAYS)
        cb = _bin_walk_forward(S8, e, bins, K, COMBINE_WINDOW_DAYS)
        ok = np.isfinite(m0) & np.isfinite(mb) & np.isfinite(ca) & np.isfinite(cb)
        ma = 0.5 * (m0 + ca)
        mab = 0.5 * (mb + cb)
        y = e[ok]
        r0 = r2_oos(y, m0[ok])
        rA, rB, rAB = r2_oos(y, ma[ok]), r2_oos(y, mb[ok]), r2_oos(y, mab[ok])
        dA, dB, dAB = rA - r0, rB - r0, rAB - r0
        best, best_name = (ma, "A") if dA > dB else (mb, "B")
        t_best = dm_test(y, mab[ok], best[ok])
        ratio = dAB / (dA + dB) if (dA + dB) != 0 else np.nan
        print(f"\nK={K}  n={int(ok.sum())}")
        print(f"  M0  pooled 7-signal combiner      R2 {r0:+.5f}")
        print(f"  M_A  + vol-regime 50/50 blend     R2 {rA:+.5f}  dR2 {dA:+.5f}  "
              f"DM-t vs M0 {dm_test(y, ma[ok], m0[ok]):+5.2f}")
        print(f"  M_B  + frozen-product increment   R2 {rB:+.5f}  dR2 {dB:+.5f}  "
              f"DM-t vs M0 {dm_test(y, mb[ok], m0[ok]):+5.2f}")
        print(f"  M_AB both                         R2 {rAB:+.5f}  dR2 {dAB:+.5f}  "
              f"DM-t vs M0 {dm_test(y, mab[ok], m0[ok]):+5.2f}")
        print(f"  additivity: dR2(AB) {dAB:+.5f} vs dR2(A)+dR2(B) {dA + dB:+.5f}  "
              f"ratio {ratio:.2f}")
        print(f"  PRE-REGISTERED TEST  DM-t of AB vs the better single arm (M_{best_name}): "
              f"{t_best:+.2f}  ({'adds' if abs(t_best) > 2 and t_best > 0 else 'does not add'} "
              f"at |t|>2)")
        rows.append(
            {"K": K, "n": int(ok.sum()), "r2_M0": r0, "r2_A": rA, "r2_B": rB, "r2_AB": rAB,
             "dr2_A": dA, "dr2_B": dB, "dr2_AB": dAB, "sum_dr2": dA + dB, "ratio": ratio,
             "dm_t_A_vs_M0": dm_test(y, ma[ok], m0[ok]),
             "dm_t_B_vs_M0": dm_test(y, mb[ok], m0[ok]),
             "dm_t_AB_vs_M0": dm_test(y, mab[ok], m0[ok]),
             "better_single_arm": best_name, "dm_t_AB_vs_best": t_best}
        )
        # secondary: the same arithmetic restricted to the 2020+ block, which no arm's
        # construction looked at for tuning (the frozen selection is from the first window).
        late = ok & (ts >= HOLDOUT).to_numpy()
        yl = e[late]
        r0l = r2_oos(yl, m0[late])
        rows[-1].update(
            {"n_2020plus": int(late.sum()), "dr2_A_2020plus": r2_oos(yl, ma[late]) - r0l,
             "dr2_B_2020plus": r2_oos(yl, mb[late]) - r0l,
             "dr2_AB_2020plus": r2_oos(yl, mab[late]) - r0l,
             "dm_t_AB_vs_best_2020plus": dm_test(yl, mab[late], best[late])}
        )
        print(f"  2020+ only (n={int(late.sum())}): dR2 A {rows[-1]['dr2_A_2020plus']:+.5f}  "
              f"B {rows[-1]['dr2_B_2020plus']:+.5f}  AB {rows[-1]['dr2_AB_2020plus']:+.5f}  "
              f"DM-t AB vs best {rows[-1]['dm_t_AB_vs_best_2020plus']:+.2f}")
    pd.DataFrame(rows).to_csv(f"{OUT}/additivity.csv", index=False)
    print(f"\nwrote {OUT}/additivity.csv")


# ---------------------------------------------------------------------------
# item 2 — the nominated state term, on a fresher split
# ---------------------------------------------------------------------------


def _state(p) -> np.ndarray:
    """The §14.3 continuous causal vol state: robust-scaled ``har_ma_25``, clipped to +-4."""
    z = p.X[TW + TW :, p.names.index("har_ma_25")].astype(np.float64)
    iqr = np.percentile(z, 75) - np.percentile(z, 25)
    return np.clip((z - np.median(z)) / (iqr + 1e-12), -4, 4)


def stage_sentiment() -> None:
    """§15.3 item 2. Re-select the state term on <=2019, score it on 2020+, 1 df.

    §14.3 found ``gamma`` significant jointly (chi2(7) = 28.1, p = 2e-4) with ``sentiment`` the
    only component clearing |t| = 2 — but it fit through 2024, and the 2021-24 block had already
    been scored twice. So the *selection* is redone here on rows before 2020 only: fit the same
    14-parameter interaction regression, take the single largest |HAC-t| component, and then score
    that one interaction as a 1-df addition to the pooled combiner on 2020+ and nothing else.

    Two ways this can fail, and both are reported rather than hidden: the <=2019 search can
    nominate a different bucket (in which case §14.3's nomination was sample-specific), or it can
    nominate ``sentiment`` and the 2020+ score can be flat or negative.
    """
    _require_fixed_cache()
    os.makedirs(OUT, exist_ok=True)
    from scipy.stats import chi2

    p = load_panel()
    e = np.load(_p("har_resid.npz"))["e"][TW:]
    sig = dict(np.load(_p("bucket_signals.npz")))
    ts = pd.Series(pd.to_datetime(p.t[TW + TW :]))
    S = np.column_stack([sig[b] for b in BUCKETS])
    z = _state(p)
    q = S.shape[1]
    search = (ts < HOLDOUT).to_numpy()
    print(f"selection rows (<{HOLDOUT}): {int(search.sum())}   holdout rows: {int((~search).sum())}")

    D = np.hstack([S[search], S[search] * z[search][:, None]])
    D -= D.mean(0)
    yc = e[search] - e[search].mean()
    G = D.T @ D + np.eye(2 * q)
    b = np.linalg.solve(G, D.T @ yc)
    u = yc - D @ b
    lags = 480
    W = D * u[:, None]
    Om = W.T @ W
    for L in range(1, lags + 1):
        A = W[L:].T @ W[:-L]
        Om += (1.0 - L / (lags + 1.0)) * (A + A.T)
    Gi = np.linalg.inv(G)
    V = Gi @ Om @ Gi
    gamma, Vg = b[q:], V[q:, q:]
    tg = gamma / np.sqrt(np.maximum(np.diag(Vg), 1e-24))
    wald = float(gamma @ np.linalg.solve(Vg, gamma))
    print("\n  SELECTION on <=2019 (14-parameter bar-resolution fit)")
    print("  bucket          gamma   HAC-t")
    for k, nm in enumerate(BUCKETS):
        print(f"    {nm:12s} {gamma[k]:+.4f}  {tg[k]:+6.2f}")
    print(f"  joint HAC Wald gamma = 0: chi2({q}) = {wald:.1f}, p = {chi2.sf(wald, q):.2e}")
    pick = int(np.argmax(np.abs(tg)))
    agree = BUCKETS[pick] == "sentiment"
    print(f"\n  <=2019 nominates: {BUCKETS[pick]} (HAC-t {tg[pick]:+.2f})   "
          f"{'AGREES' if agree else 'DISAGREES'} with §14.3's sentiment")

    cw = COMBINE_WINDOW_DAYS * PERIODS_PER_DAY
    base = _combiner(S, e, cw)
    extra = (S[:, pick] * z)[:, None]
    aug = _combiner(np.hstack([S, extra]), e, cw)
    rows = []
    for tag, m in (
        ("2020+", np.isfinite(base) & np.isfinite(aug) & (ts >= HOLDOUT).to_numpy()),
        ("<=2019", np.isfinite(base) & np.isfinite(aug) & search),
    ):
        y = e[m]
        rb, ra = r2_oos(y, base[m]), r2_oos(y, aug[m])
        t = dm_test(y, aug[m], base[m])
        print(f"\n  {tag:7s} n={int(m.sum())}  pooled R2 {rb:+.5f}  "
              f"+{BUCKETS[pick]}xz R2 {ra:+.5f}  dR2 {ra - rb:+.5f}  DM-t {t:+.2f}"
              + ("   (one-sided 1.645: PASS)" if t > 1.645 else "   (one-sided 1.645: fail)"
                 if tag == "2020+" else ""))
        rows.append({"block": tag, "n": int(m.sum()), "picked": BUCKETS[pick],
                     "agrees_with_14_3": agree, "hac_t_selection": tg[pick],
                     "r2_pooled": rb, "r2_aug": ra, "dr2": ra - rb, "dm_t": t})
    pd.DataFrame(rows).to_csv(f"{OUT}/state_term_freshsplit.csv", index=False)
    print(f"\nwrote {OUT}/state_term_freshsplit.csv")


def stage_additivity2() -> None:
    """§15.3 item 1, second architecture — because the first one was not a fair test of gain B.

    ``additivity`` collapsed the 100 frozen product columns into a **single** signal
    (``aug - base``) and gave it one coefficient in the second-stage combiner. That destroys most of
    what §7 measured: the +0.0069 comes from 100 columns carrying their *own* ridge penalty, which
    lets the fit shrink the useless ones away individually. Collapsed to one signal with one
    coefficient it cannot, and it scored dR2 **-0.0096** against the pooled combiner over the full
    sample while still being additive and significant on 2020+ — a pattern that says "the wrapper is
    wrong", not "the gain is fake".

    So: leave both estimators exactly as they were measured and ask the question that "do the gains
    add" actually means — **does using both beat using either?** Three forecasts of the same
    residual, on the rows all three can predict:

    * ``f_A``  0.5 x pooled 7-signal combiner + 0.5 x its vol-regime-conditional twin (§6)
    * ``f_B``  133 linear columns + 100 frozen products at ``ALPHA_PROD``, monthly refit (§7)
    * ``f_AB`` the equal-weight average of the two

    Equal weight is pre-specified, not fitted: fitting a combination weight on the same sample would
    be one more specification chosen on exhausted data, and the point of the test is whether the two
    error series are jointly reducible at all, which a 50/50 average already answers. The
    pre-registered statistic is unchanged — DM-t of ``f_AB`` against the better single arm, |t| > 2 —
    and the pre-registered expectation is unchanged: positive but under 2, because both gains draw on
    a ``har x exog`` interaction structure and should overlap.
    """
    _require_fixed_cache()
    os.makedirs(OUT, exist_ok=True)
    p = load_panel()
    e_full = np.load(_p("har_resid.npz"))["e"]
    e = e_full[TW:]
    sig = dict(np.load(_p("bucket_signals.npz")))
    ts = pd.Series(pd.to_datetime(p.t[TW + TW :]))
    S = np.column_stack([sig[b] for b in BUCKETS])
    cw = COMBINE_WINDOW_DAYS * PERIODS_PER_DAY

    bc, _ = base_columns(p)
    Xd = np.ascontiguousarray(p.X[TW:, bc], dtype=np.float64)
    lin, aug, _ = _interaction_walk(Xd, e_full, (ALPHA_PROD,), dynamic=False)
    fB = aug[ALPHA_PROD]
    m0 = _combiner(S, e, cw)
    print(
        f"f_B (133 linear + {N_PROD} frozen products, alpha_prod {ALPHA_PROD:.0f}): "
        f"R2 {r2_oos(e[np.isfinite(fB)], fB[np.isfinite(fB)]):+.5f}   "
        f"linear-only {r2_oos(e[np.isfinite(lin)], lin[np.isfinite(lin)]):+.5f}",
        flush=True,
    )

    rows = []
    for K in (2, 3, 5):
        bins = _vol_regime(p, ts, K)
        cond = _bin_walk_forward(S, e, bins, K, COMBINE_WINDOW_DAYS)
        ok = np.isfinite(m0) & np.isfinite(cond) & np.isfinite(fB)
        fA = 0.5 * (m0 + cond)
        fAB = 0.5 * (fA + fB)
        y = e[ok]
        r0 = r2_oos(y, m0[ok])
        rA, rB, rAB = r2_oos(y, fA[ok]), r2_oos(y, fB[ok]), r2_oos(y, fAB[ok])
        dA, dB, dAB = rA - r0, rB - r0, rAB - r0
        best, bname = (fA, "A") if dA > dB else (fB, "B")
        t_best = dm_test(y, fAB[ok], best[ok])
        print(f"\nK={K}  n={int(ok.sum())}")
        print(f"  M0   pooled combiner            R2 {r0:+.5f}")
        print(f"  f_A  vol-regime blend (§6)      R2 {rA:+.5f}  dR2 {dA:+.5f}  "
              f"DM-t vs M0 {dm_test(y, fA[ok], m0[ok]):+5.2f}")
        print(f"  f_B  linear+products (§7)       R2 {rB:+.5f}  dR2 {dB:+.5f}  "
              f"DM-t vs M0 {dm_test(y, fB[ok], m0[ok]):+5.2f}")
        print(f"  f_AB 50/50 of the two           R2 {rAB:+.5f}  dR2 {dAB:+.5f}  "
              f"DM-t vs M0 {dm_test(y, fAB[ok], m0[ok]):+5.2f}")
        print(f"  additivity: dR2(AB) {dAB:+.5f} vs dR2(A)+dR2(B) {dA + dB:+.5f}"
              f"  ratio {dAB / (dA + dB) if (dA + dB) else float('nan'):.2f}")
        print(f"  PRE-REGISTERED TEST  DM-t of AB vs the better single arm (f_{bname}): "
              f"{t_best:+.2f}  ({'ADDS' if t_best > 2 else 'does not add'} at |t|>2)")
        row = {"K": K, "n": int(ok.sum()), "r2_M0": r0, "r2_A": rA, "r2_B": rB, "r2_AB": rAB,
               "dr2_A": dA, "dr2_B": dB, "dr2_AB": dAB, "sum_dr2": dA + dB,
               "ratio": dAB / (dA + dB) if (dA + dB) else np.nan,
               "dm_t_A_vs_M0": dm_test(y, fA[ok], m0[ok]),
               "dm_t_B_vs_M0": dm_test(y, fB[ok], m0[ok]),
               "dm_t_AB_vs_M0": dm_test(y, fAB[ok], m0[ok]),
               "better_single_arm": bname, "dm_t_AB_vs_best": t_best}
        late = ok & (ts >= HOLDOUT).to_numpy()
        yl = e[late]
        r0l = r2_oos(yl, m0[late])
        row.update({"n_2020plus": int(late.sum()),
                    "dr2_A_2020plus": r2_oos(yl, fA[late]) - r0l,
                    "dr2_B_2020plus": r2_oos(yl, fB[late]) - r0l,
                    "dr2_AB_2020plus": r2_oos(yl, fAB[late]) - r0l,
                    "dm_t_AB_vs_best_2020plus": dm_test(yl, fAB[late], best[late])})
        print(f"  2020+ only (n={int(late.sum())}): dR2 A {row['dr2_A_2020plus']:+.5f}  "
              f"B {row['dr2_B_2020plus']:+.5f}  AB {row['dr2_AB_2020plus']:+.5f}  "
              f"DM-t AB vs best {row['dm_t_AB_vs_best_2020plus']:+.2f}")
        rows.append(row)
    pd.DataFrame(rows).to_csv(f"{OUT}/additivity_combination.csv", index=False)
    print(f"\nwrote {OUT}/additivity_combination.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stage",
        choices=["prep", "interactions", "additivity", "additivity2", "sentiment"],
        required=True,
    )
    a = ap.parse_args()
    {
        "prep": stage_prep,
        "interactions": stage_interactions,
        "additivity": stage_additivity,
        "additivity2": stage_additivity2,
        "sentiment": stage_sentiment,
    }[a.stage]()
