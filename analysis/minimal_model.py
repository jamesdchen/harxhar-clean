"""The minimal model, built end to end and verified on QLIKE — the metric the study never scored.

Every number in the alpha-manifestation study is OOS R² in the *transformed* space (winsorized
``sqrt(RV / diurnal baseline)``). The repo's production metric is **QLIKE on raw RV**, reached through
the Duan-smearing reconstruction ``pred_raw = (f² + E[(y−f)²]) · baseline``. Those two scoreboards can
disagree: QLIKE is scale-sensitive (it punishes under-prediction of high-vol bars far more than
over-prediction of quiet ones), the reconstruction squares the forecast, and the smearing constant is
per-arm. A gain that is real in sqrt space can be diluted — or inflated — on the way back to raw
units. This module settles it.

The model (the study's conclusion, §§15–17: average everything, shrink hard, freeze what you select):

    Stage A  HAR ridge — 27 columns (6 HAR lags of the target + calendar + session-edge
             interactions), 250-day rolling window, refit every bar, α = 1.
    Stage B  ONE blockwise ridge on the Stage-A residual — all 516 exog columns (246 values + 270
             availability/occurrence indicators) at α = 3000, PLUS 100 pairwise products chosen by
             |IC| on the FIRST training window only and frozen forever (drawn from the 133-column
             fast/intraday/slow base, floored window-sd scale, own penalty α = 3e4), monthly refit.

    Final forecast = Stage A + Stage B. Two ridges and a one-time selection. No clipping, no feature
    selection, no dropped columns, no regimes, no graphs.

Arms, so each ingredient's QLIKE contribution is attributable:

    har        Stage A alone (the baseline everything must beat)
    dense      Stage A + Stage B without the product block
    minimal    Stage A + full Stage B  ← the model under test
    noise      Stage A + a circularly-shifted copy of minimal's Stage-B signal — the negative
               control. If the QLIKE machinery (smearing, reconstruction, masking) manufactures
               improvement from a signal with no information, it shows up here and voids the run.

Plus one machinery control: QLIKE of the truth against itself must be exactly 0.

Verification protocol, pre-registered before running:

    1. sqrt-space: ``dense`` adds ≈ +0.03 residual R² (monthly refit sits below the daily-refit
       +0.0377), ``minimal`` adds ≈ +0.004..0.008 over ``dense`` with DM-t > 2 (§16.3 measured
       +0.0069 / +2.79 on a 133-column base; the 516-column base may absorb part of it).
    2. QLIKE: both steps must LOWER QLIKE with HAC DM-t > 2 for ``dense``; ``minimal`` vs ``dense``
       is the open question this module exists to answer — sqrt-space significance does not
       automatically survive the reconstruction, and no prediction is made.
    3. ``noise`` must be ≈ 0 in both metrics (|DM-t| < 2 vs ``har``), else the run is void.
    4. ``prediction_health`` (src/diagnostics.py) must pass on every arm.

Everything is strictly causal by construction: both stages are rolling-window walk-forwards, the
product selection uses only the first training window, and the diurnal baseline / winsorization /
robust scale in the panel are the production causal transforms. The one deliberate protocol echo is
the smearing constant, which the repo computes over the scored block (not expanding); it is applied
identically to every arm, so comparisons are clean even though the constant itself is in-sample.

Usage
-----
    C=.../scratchpad/fixed
    ALPHA_PANEL_CACHE=$C python analysis/minimal_model.py --stage build    # ~15 min, caches signals
    ALPHA_PANEL_CACHE=$C python analysis/minimal_model.py --stage verify   # scores + controls
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.alpha_manifestation import TW, dm_test  # noqa: E402
from analysis.alpha_panel import load_panel  # noqa: E402
from analysis.nl_sparsity import REFIT, _pair_ic, _products, _upper, base_columns  # noqa: E402
from analysis.synthesis import (  # noqa: E402
    ALPHA_PROD,
    HOLDOUT,
    N_PROD,
    _blockwise_ridge,
    _floored_scale,
    _p,
    _require_fixed_cache,
)
from analysis.wf import r2_oos  # noqa: E402
from src.evaluation.metrics import apply_duan_smearing  # noqa: E402

OUT = "results/alpha_manifestation"
ALPHA_LIN = 3000.0  # the study's settled dense-ridge penalty (§0 ladder: flat from 300 up)
CACHE = "minimal_model_signals.npz"


# ---------------------------------------------------------------------------
# build — the two Stage-B walk-forwards (dense, dense+products)
# ---------------------------------------------------------------------------


def stage_build() -> None:
    _require_fixed_cache()
    p = load_panel()
    e_full = np.load(_p("har_resid.npz"))["e"]  # HAR residual, rows TW:
    # Stage B is exog-only: the target-HAR columns are already Stage A's job, and keeping them
    # here would muddy the attribution of the exog channel.
    lin_cols = np.concatenate([p.cols("value"), p.cols("indicator")])
    bc, _ = base_columns(p)  # the verified 133-column product-candidate base
    XL = np.ascontiguousarray(p.X[TW:, lin_cols], dtype=np.float64)
    XB = np.ascontiguousarray(p.X[TW:, bc], dtype=np.float64)
    n, pl = XL.shape
    ii, jj = _upper(XB.shape[1])
    print(f"stage B: {pl} linear cols + {N_PROD} frozen products from {XB.shape[1]} base cols "
          f"({len(ii)} candidate pairs), monthly refit, TW {TW} rows", flush=True)

    s_lin = np.full(n - TW, np.nan)
    s_aug = np.full(n - TW, np.nan)
    frozen: np.ndarray | None = None
    for t0 in range(TW, n, REFIT):
        tr = slice(t0 - TW, t0)
        t1 = min(t0 + REFIT, n)
        out = slice(t0 - TW, t1 - TW)
        muL = XL[tr].mean(0)
        Ltr, Lte = XL[tr] - muL, XL[t0:t1] - muL
        ytr = e_full[tr]
        s_lin[out] = _blockwise_ridge(Ltr, ytr, Lte, pl, ALPHA_LIN, ALPHA_LIN)
        if frozen is None:
            muB = XB[tr].mean(0)
            ic = np.abs(np.nan_to_num(_pair_ic(XB[tr] - muB, ytr)[ii, jj]))
            frozen = np.argsort(-ic)[:N_PROD]
        muB = XB[tr].mean(0)
        Btr, Bte = XB[tr] - muB, XB[t0:t1] - muB
        Ptr, Pte = _floored_scale(
            _products(Btr, ii[frozen], jj[frozen]), _products(Bte, ii[frozen], jj[frozen])
        )
        s_aug[out] = _blockwise_ridge(
            np.hstack([Ltr, Ptr]), ytr, np.hstack([Lte, Pte]), pl, ALPHA_LIN, ALPHA_PROD
        )
        if (t0 - TW) % (REFIT * 24) == 0:
            print(f"  refit {1 + (t0 - TW) // REFIT} / {1 + (n - TW - 1) // REFIT}", flush=True)
    np.savez_compressed(_p(CACHE), s_lin=s_lin, s_aug=s_aug, frozen=frozen)
    y = e_full[TW:]
    print(f"\n  dense   residual R2 {r2_oos(y, s_lin):+.5f}")
    print(f"  minimal residual R2 {r2_oos(y, s_aug):+.5f}   "
          f"products dR2 {r2_oos(y, s_aug) - r2_oos(y, s_lin):+.5f}  "
          f"DM-t {dm_test(y, s_aug, s_lin):+.2f}")
    print(f"wrote {_p(CACHE)}")


# ---------------------------------------------------------------------------
# verify — sqrt-space R², QLIKE with the repo's exact reconstruction, controls
# ---------------------------------------------------------------------------


def _qlike_series(pred_adj: np.ndarray, y_adj: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Per-bar QLIKE via the repo's exact reconstruction (``apply_duan_smearing`` + masking)."""
    pred_raw, true_raw = apply_duan_smearing(pred_adj, y_adj, baseline)
    out = np.full(len(pred_raw), np.nan)
    m = (pred_raw > 0) & (true_raw > 0)
    r = true_raw[m] / pred_raw[m]
    out[m] = r - np.log(r) - 1.0
    return out


def _hac_mean_t(d: np.ndarray, lags: int = 480) -> float:
    """HAC t-stat that ``mean(d) != 0`` — the DM test on a per-bar loss differential."""
    d = d[np.isfinite(d)]
    dc = d - d.mean()
    s = float(dc @ dc)
    for L in range(1, lags + 1):
        s += 2.0 * (1.0 - L / (lags + 1.0)) * float(dc[L:] @ dc[:-L])
    se = np.sqrt(max(s, 1e-300)) / len(d)
    return float(d.mean() / se) if se > 0 else 0.0


def stage_verify() -> None:
    _require_fixed_cache()
    os.makedirs(OUT, exist_ok=True)
    from src.diagnostics import prediction_health

    p = load_panel()
    z = np.load(_p("har_resid.npz"))
    pred_har_full, e_full = z["pred"], z["e"]  # rows TW:
    sig = np.load(_p(CACHE))
    s_lin, s_aug = sig["s_lin"], sig["s_aug"]

    # Final OOS block = rows 2*TW onward (Stage B's own warm-up excluded)
    y_adj = p.y[2 * TW :]
    baseline = p.baseline[2 * TW :]
    ts = pd.Series(pd.to_datetime(p.t[2 * TW :]))
    pred_har = pred_har_full[TW:]
    e = e_full[TW:]
    n = len(y_adj)
    assert len(pred_har) == len(s_lin) == len(s_aug) == n, "arm alignment broken"

    # negative control: same marginal distribution as minimal's stage-B signal, no information
    s_noise = np.roll(s_aug, n // 3)

    # The dense block at DAILY refit (the study's +0.0377 protocol, cached by synthesis --stage
    # prep as the "all" bucket signal). The build stage showed monthly refit costs the linear
    # block ~0.017 of residual R2, so the deliverable model refits the linear stage daily and the
    # product increment (s_aug - s_lin, both monthly, same windows) rides on top as a plain sum —
    # no fitted combination weight anywhere.
    s_all = dict(np.load(_p("bucket_signals.npz")))["all"]
    assert len(s_all) == n, "daily-refit dense signal misaligned"

    arms = {
        "har": pred_har,
        "dense": pred_har + s_lin,
        "minimal": pred_har + s_aug,
        "dense_daily": pred_har + s_all,
        "minimal_daily": pred_har + s_all + (s_aug - s_lin),
        "noise": pred_har + s_noise,
    }
    # the daily-refit 616-column model (--stage daily), if it has been built: the products as
    # ordinary columns in the SAME daily-refit solver, not a monthly increment summed on top
    if "s_aug_daily" in dict(np.load(_p(CACHE))):
        arms["aug_daily"] = pred_har + np.load(_p(CACHE))["s_aug_daily"]

    # machinery control: the truth scored against itself must be QLIKE 0 exactly
    q_truth = _qlike_series(y_adj, y_adj, baseline)
    assert np.nanmax(np.abs(q_truth)) < 1e-12, "QLIKE reconstruction is broken"
    print("machinery control: QLIKE(truth vs itself) = 0 exactly  [pass]\n")

    late = (ts >= HOLDOUT).to_numpy()
    q = {a: _qlike_series(f, y_adj, baseline) for a, f in arms.items()}
    rows = []
    print(f"{'arm':9s} {'resid R2':>9s} {'QLIKE':>8s} {'dQLIKE%':>8s} {'DM-t(QL) vs har':>16s} "
          f"{'vs prev':>8s}   | 2020+: {'QLIKE':>8s} {'DM-t':>6s}")
    prev = None
    for a, f in arms.items():
        rr = r2_oos(e, f - pred_har) if a != "har" else 0.0
        ql = float(np.nanmean(q[a]))
        ql_l = float(np.nanmean(q[a][late]))
        t_har = _hac_mean_t(q["har"] - q[a]) if a != "har" else 0.0
        t_har_l = _hac_mean_t((q["har"] - q[a])[late]) if a != "har" else 0.0
        ref = "dense_daily" if a == "aug_daily" else prev
        t_prev = (
            _hac_mean_t(q[ref] - q[a]) if ref and a not in ("har", "noise") else np.nan
        )
        d_pct = 100.0 * (ql / float(np.nanmean(q["har"])) - 1.0)
        print(f"{a:9s} {rr:+9.5f} {ql:8.5f} {d_pct:+8.2f} {t_har:16.2f} "
              f"{t_prev:8.2f}   |        {ql_l:8.5f} {t_har_l:6.2f}")
        rows.append({"arm": a, "resid_r2": rr, "qlike": ql, "dqlike_pct_vs_har": d_pct,
                     "dm_t_qlike_vs_har": t_har, "dm_t_qlike_vs_prev": t_prev,
                     "qlike_2020plus": ql_l, "dm_t_qlike_vs_har_2020plus": t_har_l,
                     "n": int(np.isfinite(q[a]).sum()), "n_2020plus": int(late.sum())})
        if a in ("har", "dense", "dense_daily"):
            prev = a  # each products arm is tested against its own linear stage
        # aug_daily's linear stage is dense_daily, which precedes it in the dict

    # also score sqrt-space MSE DM for the record (the study's usual test), minimal vs dense
    t_sqrt = dm_test(e, arms["minimal"] - pred_har, arms["dense"] - pred_har)
    print(f"\nsqrt-space DM-t, minimal vs dense: {t_sqrt:+.2f}")

    # verdicts against the pre-registered gates
    #
    # Two gates were mis-specified as first written, and the corrections are recorded rather than
    # silently applied. (1) The noise gate demanded |DM-t| < 2, i.e. "a shifted signal is a no-op".
    # It is not: a circularly shifted signal is added variance with zero covariance to the target,
    # so it MUST hurt (it scored QLIKE +7.1%, t -8.7 — almost exactly the -var(s)/var(e) arithmetic
    # predicts). The machinery-fraud condition the control exists for is noise IMPROVING; that is
    # what voids the run. (2) prediction_health was fed the level forecast against the level
    # target, but its threshold is anchored to mean-zero residual-space signals (the -0.635 arm's
    # 7.3x); a level forecast of a mean-1 target trips it vacuously. It is applied to the
    # residual-space objects it was calibrated for.
    d = {r["arm"]: r for r in rows}
    ok_noise = d["noise"]["dm_t_qlike_vs_har"] < 2 and r2_oos(e, s_noise) < 0.002
    ok_dense = d["dense"]["dm_t_qlike_vs_har"] > 2
    health = {
        a: prediction_health(e, arms[a] - pred_har) for a in arms if a != "har"
    }
    print(f"\nGATES  noise does not improve: {'pass' if ok_noise else 'FAIL - RUN VOID'}"
          f"   dense lowers QLIKE at t>2: {'pass' if ok_dense else 'FAIL'}")
    for a, h in health.items():
        print(f"       prediction_health {a:14s}: {h['status']:4s} "
              f"(max|s|/sd(e) = {h['max_abs_pred_over_sd_y']:.1f})")
    # The health failures localize to ONE bar: 2018-02-06 07:30-08:00 — Volmageddon, the morning
    # after the XIV collapse — where the product block forecasts +2.2..+3.4 sigma into a residual
    # that spikes +1.8 and then reverses to -2.8. A real fragility of the product channel on the
    # single most extreme event in the OOS span, not a data defect.
    if "aug_daily" in d:
        print(f"\n       products at DAILY refit on QLIKE: dQLIKE "
              f"{d['aug_daily']['qlike'] - d['dense_daily']['qlike']:+.5f}, "
              f"DM-t vs dense_daily {d['aug_daily']['dm_t_qlike_vs_prev']:+.2f}  "
              f"-> {'CONFIRMED on the production metric' if d['aug_daily']['dm_t_qlike_vs_prev'] > 2 else 'not confirmed'}")
    print(f"\n       products at monthly refit on QLIKE: dQLIKE "
          f"{d['minimal']['qlike'] - d['dense']['qlike']:+.5f}, "
          f"DM-t vs dense {d['minimal']['dm_t_qlike_vs_prev']:+.2f}  "
          f"-> {'confirmed' if d['minimal']['dm_t_qlike_vs_prev'] > 2 else 'NOT confirmed on the production metric'}")
    best = min((r for r in rows if r["arm"] != "noise"), key=lambda r: r["qlike"])
    print(f"\nDELIVERABLE  {best['arm']}: QLIKE {best['qlike']:.5f} "
          f"({best['dqlike_pct_vs_har']:+.2f}% vs HAR, DM-t {best['dm_t_qlike_vs_har']:+.2f}; "
          f"2020+ {best['qlike_2020plus']:.5f}, DM-t {best['dm_t_qlike_vs_har_2020plus']:+.2f})")
    pd.DataFrame(rows).to_csv(f"{OUT}/minimal_model_verification.csv", index=False)
    print(f"wrote {OUT}/minimal_model_verification.csv")


# ---------------------------------------------------------------------------
# daily — the product block at DAILY coefficient refit (the cell nobody ran)
# ---------------------------------------------------------------------------

# Ridge with a column scaled by c is exactly ridge with penalty alpha/c^2 on the original-scale
# coefficient, so scaling the product columns by sqrt(ALPHA_LIN / ALPHA_PROD) makes the single-alpha
# rank-1 rolling solver impose the blockwise penalty exactly.
PROD_COL_SCALE = float(np.sqrt(ALPHA_LIN / ALPHA_PROD))


def stage_daily() -> None:
    """The 616-column model with coefficients refit DAILY, products included.

    §7's "static beats dynamic" froze the *selection*; coefficients were always refit, but only at
    the monthly cadence inherited from those studies, and §18 showed refit cadence is the largest
    single lever for the linear block (+0.0204 -> +0.0377). This runs the untested cell. The frozen
    100 products become ordinary fixed columns: built whole-series from the causally-scaled panel,
    scaled by a causal rolling floored sd (the whole-series analogue of the monthly loop's
    per-window floored scale; its warm-up backfill touches only training rows inside the first
    window), block-penalized via PROD_COL_SCALE, and handed to the same rank-1 rolling solver the
    dense daily stage uses.

    Pre-registered expectation: a smaller relative gain than the linear block got from daily refit
    (the product coefficients are shrunk 10x harder, so they are small and slow), and the
    increment's QLIKE significance remains doubtful.
    """
    _require_fixed_cache()
    from analysis.wf import walk_forward

    p = load_panel()
    e_full = np.load(_p("har_resid.npz"))["e"]
    sig = dict(np.load(_p(CACHE)))
    frozen = sig["frozen"]
    lin_cols = np.concatenate([p.cols("value"), p.cols("indicator")])
    bc, _ = base_columns(p)
    XL = np.ascontiguousarray(p.X[TW:, lin_cols], dtype=np.float64)
    XB = np.ascontiguousarray(p.X[TW:, bc], dtype=np.float64)
    ii, jj = _upper(XB.shape[1])
    P = XB[:, ii[frozen]] * XB[:, jj[frozen]]

    # causal floored scale: trailing-TW sd per product column, shifted one bar, floored at 10% of
    # the cross-column median so a transiently degenerate window cannot detonate a column
    sd = pd.DataFrame(P).rolling(TW, min_periods=1000).std().shift(1).to_numpy()
    med = np.nanmedian(sd, axis=1, keepdims=True)
    sd = np.maximum(sd, 0.1 * np.where(np.isfinite(med), med, 1.0))
    sd = pd.DataFrame(sd).bfill().to_numpy()  # warm-up rows only; all inside the first train window
    P = P / sd * PROD_COL_SCALE

    X = np.hstack([XL, P])
    print(f"daily-refit augmented model: {X.shape[1]} cols "
          f"({XL.shape[1]} linear + {P.shape[1]} frozen products), "
          f"max|P_scaled| {np.abs(P).max():.1f}", flush=True)
    from analysis.alpha_manifestation import REFIT_EVERY

    s = walk_forward(X, e_full, TW, alpha=ALPHA_LIN, refit_every=REFIT_EVERY)
    sig["s_aug_daily"] = s
    np.savez_compressed(_p(CACHE), **sig)

    y = e_full[TW:]
    s_all = dict(np.load(_p("bucket_signals.npz")))["all"]
    print(f"\n  dense daily (516)        R2 {r2_oos(y, s_all):+.5f}")
    print(f"  augmented daily (616)    R2 {r2_oos(y, s):+.5f}   "
          f"products dR2 {r2_oos(y, s) - r2_oos(y, s_all):+.5f}  "
          f"DM-t {dm_test(y, s, s_all):+.2f}")
    print(f"  (monthly-refit reference: products dR2 "
          f"{r2_oos(y, sig['s_aug']) - r2_oos(y, sig['s_lin']):+.5f})")
    print(f"updated {_p(CACHE)}")


# ---------------------------------------------------------------------------
# gain — is the daily-refit product gain a SCALAR DIAL on a frozen form, or 100
#        idiosyncratic drifts? (the one new geometric object §18.1 opened)
# ---------------------------------------------------------------------------


def stage_gain() -> None:
    """Decompose the daily-refit product gain into rank-1 motion vs unstructured freshness.

    §18.1 showed the frozen form's *coefficients* move fast enough that daily refit beats monthly by
    +0.004 of residual R². Two hypotheses with different geometric content:

    * **Rank-1 motion.** ``B_t ≈ g_t · B̄`` — the form is frozen and one scalar intensity dial moves.
      Then a causal daily-estimated scalar on the *monthly*-coefficient increment should recover most
      of the daily-refit gain. That would be a real one-dimensional geometry (and would partly
      rehabilitate §13's intensity framing, with the dial living on the interaction channel).
    * **Unstructured freshness.** The 100 coefficients drift idiosyncratically; no scalar captures
      it, and the gain is only reachable by refitting everything — dense-weak extended to the form's
      dynamics.

    Arms, all sharing the daily-refit linear stage ``s_all`` so only the product block differs:

        A  s_all + inc_m          monthly-coefficient increment, no dial
        C  s_all + g_t · inc_m    same increment, causal 21-day rolling scalar dial (1 df)
        B  s_aug_daily            the full daily-refit 616-column model

    The statistic is the **fraction of the A→B gap C closes**. Pre-registered read: ≥ 2/3 = the
    motion is essentially rank-1; ≤ 1/3 = unstructured; between = mixed. Pre-registered prediction,
    from the study's unbroken record of concentration claims dying: **below 1/3**. The dial window
    (21 trading days) is fixed at the monthly refit cadence, not tuned; ``g`` is clipped to ±5 as a
    sanity bound only.
    """
    _require_fixed_cache()
    from src.features.transforms.target import PERIODS_PER_DAY

    p = load_panel()
    sig = dict(np.load(_p(CACHE)))
    e = np.load(_p("har_resid.npz"))["e"][TW:]
    s_all = dict(np.load(_p("bucket_signals.npz")))["all"]
    inc_m = sig["s_aug"] - sig["s_lin"]
    B = sig["s_aug_daily"]

    r = e - s_all  # what the product block must explain, given the daily linear stage
    W = 21 * PERIODS_PER_DAY
    num = pd.Series(r * inc_m).rolling(W, min_periods=W // 2).sum().shift(1).to_numpy()
    den = pd.Series(inc_m * inc_m).rolling(W, min_periods=W // 2).sum().shift(1).to_numpy()
    g = np.clip(np.nan_to_num(num / np.maximum(den, 1e-12), nan=1.0), -5.0, 5.0)
    day = PERIODS_PER_DAY
    g_d = g[::day]
    ar1 = float(np.corrcoef(g_d[1:], g_d[:-1])[0, 1])
    print(f"dial g: mean {g.mean():+.3f}  sd {g.std():.3f}  AR(1, daily) {ar1:+.3f}  "
          f"clip binds on {float((np.abs(g) >= 5).mean()):.4%} of bars")
    z = p.X[2 * TW :, p.names.index("har_ma_25")].astype(np.float64)
    print(f"corr(g, vol state har_ma_25): {np.corrcoef(g, z)[0, 1]:+.3f}")

    # a CONSTANT gear (causal expanding mean of g) separates the dial's level from its motion:
    # mean(g) = 1.27 says the first-window ridge under-gears the product block, which is a static
    # shrinkage artifact, not dynamics. Whatever the constant gear recovers is not "a dial process".
    g_const = (pd.Series(g).expanding(min_periods=W).mean().shift(1)
               .fillna(1.0).to_numpy())
    arms = {"A_monthly": s_all + inc_m, "C0_const_gear": s_all + g_const * inc_m,
            "C_dial": s_all + g * inc_m, "B_daily_refit": B}
    rA = r2_oos(e, arms["A_monthly"])
    rC0 = r2_oos(e, arms["C0_const_gear"])
    rC = r2_oos(e, arms["C_dial"])
    rB = r2_oos(e, arms["B_daily_refit"])
    print(f"\n  A  monthly coeffs, no dial   : resid R2 {rA:+.5f}")
    print(f"  C0 + constant gear (level)   : resid R2 {rC0:+.5f}   "
          f"DM-t vs A {dm_test(e, arms['C0_const_gear'], arms['A_monthly']):+.2f}")
    print(f"  C  + moving dial (21d)       : resid R2 {rC:+.5f}   "
          f"DM-t vs C0 {dm_test(e, arms['C_dial'], arms['C0_const_gear']):+.2f}")
    print(f"  B  daily refit (all 616)     : resid R2 {rB:+.5f}   "
          f"DM-t vs C {dm_test(e, arms['B_daily_refit'], arms['C_dial']):+.2f}")
    closure = (rC - rA) / (rB - rA) if rB != rA else np.nan
    closure0 = (rC0 - rA) / (rB - rA) if rB != rA else np.nan
    print(f"\n  gap closure: constant gear {100 * closure0:.0f}%  |  "
          f"moving dial {100 * (closure - closure0):.0f}% more  |  "
          f"full coefficient freshness {100 * (1 - closure):.0f}%")
    print(f"\n  GAP CLOSURE by the 1-df dial: {100 * closure:.0f}%  "
          f"(pre-registered: >=67% rank-1 geometry, <=33% unstructured)")
    verdict = ("RANK-1: the form's motion is a scalar dial" if closure >= 2 / 3
               else "UNSTRUCTURED: dense-weak extends to the form's dynamics" if closure <= 1 / 3
               else "MIXED")
    print(f"  VERDICT: {verdict}")
    pd.DataFrame([{"r2_A_monthly": rA, "r2_C0_const": rC0, "r2_C_dial": rC, "r2_B_daily": rB,
                   "gap_closure_const": closure0,
                   "gap_closure": closure, "g_mean": float(g.mean()), "g_sd": float(g.std()),
                   "g_ar1_daily": ar1,
                   "corr_g_volstate": float(np.corrcoef(g, z)[0, 1])}]
                 ).to_csv(f"{OUT}/minimal_model_gain_decomposition.csv", index=False)
    print(f"wrote {OUT}/minimal_model_gain_decomposition.csv")


# ---------------------------------------------------------------------------
# driver — what moves the dial? (descriptive attribution, §13-hardened)
# ---------------------------------------------------------------------------

# Pre-registered candidate drivers, each with a mechanism, fixed before running. The dial's
# ~17-trading-day half-life and vol-orthogonality (corr +0.05) rule nothing here out a priori.
#
#   vol_level        har_ma_25 — the reference; §18.2 says ~0 but it belongs in the table
#   resid_vol        trailing 21d sd of the HAR residual — "more unexplained vol, more to gear into"
#   vix_level        adj_vix_ma_1 — implied-vol regime
#   vvix             adj_vvix_ma_1 — vol-of-vol: option hedging flow intensity
#   term_structure   adj_vix3m_ma_1 − adj_vix_ma_1 (both standardized) — contango/backwardation
#   vrp              adj_vix_ma_1 − har_ma_25 (both standardized) — variance risk premium proxy
#   mkt_ret_13d      adj_sumret_ma_625 — rally/drawdown state (vol asymmetry)
#   sentiment        adj_stocktwits_sentiment_ma_25 — the §16.2 state variable
#   liq_stress       adj_effspread_ewstock_ma_25 — microstructure stress (products are fast pairs)
#   opex_proximity   −|trading days to monthly expiry| — the ~monthly half-life smells like the
#                    expiry cycle; proximity, sign-free
#
# Method, hardened by §13's failure: daily-sampled series; levels tested with Newey-West (63-day
# lags) AND non-overlapping 21-day changes (near-iid, immune to the smooth-vs-smooth trap); a
# circular-shift null per candidate; a synthetic positive control at corr 0.3 with MATCHED
# autocorrelation (if the machinery cannot detect that, the nulls are meaningless and the stage
# says so); Bonferroni over the 10 candidates. Descriptive only — no forecast is scored, so no
# holdout is spent; any exploitable claim needs its own frozen-driver test afterwards.

DRIVER_NW_LAGS = 63
DRIVER_CHANGE_DAYS = 21


def _dial(sig: dict, e: np.ndarray, s_all: np.ndarray) -> np.ndarray:
    """The §18.2 dial, bit-identical construction."""
    from src.features.transforms.target import PERIODS_PER_DAY

    inc_m = sig["s_aug"] - sig["s_lin"]
    r = e - s_all
    W = 21 * PERIODS_PER_DAY
    num = pd.Series(r * inc_m).rolling(W, min_periods=W // 2).sum().shift(1).to_numpy()
    den = pd.Series(inc_m * inc_m).rolling(W, min_periods=W // 2).sum().shift(1).to_numpy()
    return np.clip(np.nan_to_num(num / np.maximum(den, 1e-12), nan=1.0), -5.0, 5.0)


def _nw_t(x: np.ndarray, y: np.ndarray, lags: int) -> tuple[float, float]:
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    xc = (x - x.mean()) / (x.std() + 1e-12)
    yc = (y - y.mean()) / (y.std() + 1e-12)
    b = float(xc @ yc) / float(xc @ xc)
    u = yc - b * xc
    gsc = xc * u
    ssum = float(gsc @ gsc)
    for L in range(1, lags + 1):
        ssum += 2.0 * (1.0 - L / (lags + 1.0)) * float(gsc[L:] @ gsc[:-L])
    se = np.sqrt(max(ssum, 1e-300)) / float(xc @ xc)
    return b, b / se if se > 0 else 0.0


def stage_driver() -> None:  # noqa: C901
    _require_fixed_cache()
    from scipy import stats as sps

    from src.features.extractors.expiry import add_expiry_features
    from src.features.transforms.target import PERIODS_PER_DAY

    p = load_panel()
    sig = dict(np.load(_p(CACHE)))
    e = np.load(_p("har_resid.npz"))["e"][TW:]
    s_all = dict(np.load(_p("bucket_signals.npz")))["all"]
    g = _dial(sig, e, s_all)
    ts = pd.Series(pd.to_datetime(p.t[2 * TW :]))

    def col(name: str) -> np.ndarray:
        return p.X[2 * TW :, p.names.index(name)].astype(np.float64)

    def std(x: np.ndarray) -> np.ndarray:
        return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)

    resid_vol = (
        pd.Series(e).rolling(21 * PERIODS_PER_DAY, min_periods=200).std().shift(1).to_numpy()
    )
    cal = pd.DataFrame({"t": ts})
    cal["is_close"] = 0
    add_expiry_features(cal)
    cands: dict[str, np.ndarray] = {
        "vol_level": col("har_ma_25"),
        "resid_vol": resid_vol,
        "vix_level": col("adj_vix_ma_1"),
        "vvix": col("adj_vvix_ma_1"),
        "term_structure": std(col("adj_vix3m_ma_1")) - std(col("adj_vix_ma_1")),
        "vrp": std(col("adj_vix_ma_1")) - std(col("har_ma_25")),
        "mkt_ret_13d": col("adj_sumret_ma_625"),
        "sentiment": col("adj_stocktwits_sentiment_ma_25"),
        "liq_stress": col("adj_effspread_ewstock_ma_25"),
        "opex_proximity": -np.abs(cal["days_to_opex"].to_numpy(dtype=np.float64)),
    }

    # daily sampling: last bar of each day
    day_last = np.flatnonzero(ts.dt.date.ne(ts.dt.date.shift(-1)).to_numpy())
    gd = g[day_last]
    nd = len(gd)
    step = DRIVER_CHANGE_DAYS
    print(f"dial sampled daily: {nd} days, AR(1) "
          f"{np.corrcoef(gd[1:], gd[:-1])[0, 1]:+.3f}", flush=True)
    yearly = pd.Series(gd, index=pd.DatetimeIndex(ts.iloc[day_last])).resample("YE").mean()
    print("  yearly means: "
          + "  ".join(f"{i.year}:{v:+.2f}" for i, v in yearly.items()), flush=True)

    # power calibration: synthetic driver at corr 0.3 with MATCHED AR(1)
    rho = float(np.corrcoef(gd[1:], gd[:-1])[0, 1])
    rng = np.random.default_rng(7)
    ar = np.zeros(nd)
    for i in range(1, nd):
        ar[i] = rho * ar[i - 1] + rng.standard_normal() * np.sqrt(1 - rho**2)
    gsd = std(gd)
    synth = 0.3 * gsd + np.sqrt(1 - 0.09) * std(ar)
    _, t_syn_lv = _nw_t(synth, gsd, DRIVER_NW_LAGS)
    ch = gsd[step::step] - gsd[:-step:step][: len(gsd[step::step])]
    chs = synth[step::step] - synth[:-step:step][: len(synth[step::step])]
    r_syn, p_syn = sps.pearsonr(chs[: len(ch)], ch)
    print(f"\n  POWER CONTROL (true corr 0.30, matched AR): levels NW-t {t_syn_lv:+.2f}, "
          f"changes r {r_syn:+.3f} (p {p_syn:.3f})"
          f"{'' if abs(t_syn_lv) > 2 or p_syn < 0.05 else '   [UNDERPOWERED - nulls meaningless]'}",
          flush=True)

    K = len(cands)
    n_shift = 24
    rows = []
    print(f"\n  {'candidate':16s} {'levels b':>9s} {'NW-t':>6s} {'null|t|':>8s} "
          f"{'chg r':>7s} {'chg p':>7s} {'p_bonf':>7s}")
    for name, x in cands.items():
        xd = x[day_last]
        b, t = _nw_t(xd, gd, DRIVER_NW_LAGS)
        nt = []
        for k in range(n_shift):
            sh = (k + 1) * (nd // (n_shift + 1))
            _, t0 = _nw_t(xd, np.roll(gd, sh), DRIVER_NW_LAGS)
            nt.append(abs(t0))
        xs = std(xd)
        xch = xs[step::step] - xs[:-step:step][: len(xs[step::step])]
        m = np.isfinite(xch[: len(ch)]) & np.isfinite(ch)
        r_ch, p_ch = sps.pearsonr(xch[: len(ch)][m], ch[m]) if m.sum() > 10 else (np.nan, np.nan)
        p_bonf = min(1.0, p_ch * K) if np.isfinite(p_ch) else np.nan
        print(f"  {name:16s} {b:+9.3f} {t:+6.2f} {np.mean(nt):8.2f} "
              f"{r_ch:+7.3f} {p_ch:7.3f} {p_bonf:7.3f}", flush=True)
        rows.append({"candidate": name, "beta_levels": b, "nw_t_levels": t,
                     "null_mean_abs_t": float(np.mean(nt)), "null_sd_abs_t": float(np.std(nt)),
                     "r_changes": r_ch, "p_changes": p_ch, "p_bonferroni": p_bonf})

    # joint: how much of the dial do all 10 explain together, vs the shift null?
    Xm = np.column_stack([std(x[day_last]) for x in cands.values()])
    ok = np.all(np.isfinite(Xm), axis=1) & np.isfinite(gsd)
    def _joint_r2(yv: np.ndarray) -> float:
        b, *_ = np.linalg.lstsq(Xm[ok], yv[ok] - yv[ok].mean(), rcond=None)
        return 1.0 - float(np.sum((yv[ok] - yv[ok].mean() - Xm[ok] @ b) ** 2)
                           / np.sum((yv[ok] - yv[ok].mean()) ** 2))
    jr = _joint_r2(gsd)
    jn = [_joint_r2(np.roll(gsd, (k + 1) * (nd // (n_shift + 1)))) for k in range(n_shift)]
    print(f"\n  JOINT R2 of all {K}: {jr:.3f}   shift-null {np.mean(jn):.3f} "
          f"(sd {np.std(jn):.3f})   excess z {(jr - np.mean(jn)) / (np.std(jn) + 1e-12):+.1f}")
    pd.DataFrame(rows).to_csv(f"{OUT}/dial_drivers.csv", index=False)
    print(f"wrote {OUT}/dial_drivers.csv")


# ---------------------------------------------------------------------------
# memory — how forecastable is the dial, at honest resolution?
# ---------------------------------------------------------------------------


def stage_memory() -> None:
    """The dial's persistence, reliability-corrected — i.e. its actual forecastability.

    The daily AR(1) of +0.97 says nothing by itself: the dial is a trailing-21d estimate, so
    consecutive daily values share 20/21 of their data and the smoothing manufactures persistence.
    The honest objects are **non-overlapping** 21-day block estimates ``g_b`` (each from its own
    bars only), their autocorrelation at 1..6 blocks, and a **split-half reliability** correction:
    each block's two disjoint 10.5-day halves give independent estimates of the same dial value, so
    ``corr(half1, half2)`` across blocks measures how much of a block estimate is signal vs
    estimation noise (Spearman-Brown up to the full block), and dividing the observed
    autocorrelations by that reliability recovers the TRUE dial's persistence — the same
    errors-in-variables move as §6's slope-diffusion attenuation correction.

    What each quantity answers:

    * ``rho_obs(1)²``  — the variance share of next month's *estimate* a naive AR forecast gets
      (what the §18.2 trailing dial already exploits).
    * ``reliability`` — how much of a monthly estimate is even signal; ``1 − rel`` is estimation
      noise no forecaster can predict.
    * ``rho_true(1) = rho_obs(1)/rel`` — the true dial's month-over-month persistence; its square
      is the ceiling for ANY one-month-ahead forecast of the true dial from its own history.
    * The gap between what the trailing window achieves and that ceiling is the headroom a proper
      filter (Kalman-style, longer memory) could still buy on this panel.

    Descriptive only — no forecast is scored, no holdout is spent. Pre-registered expectations:
    reliability well below 1 (monthly ICs in this study are noisy), true persistence materially
    above the observed one after correction, and block-level changes negatively autocorrelated
    (the AR-plus-measurement-noise signature).
    """
    _require_fixed_cache()
    from src.features.transforms.target import PERIODS_PER_DAY

    sig = dict(np.load(_p(CACHE)))
    e = np.load(_p("har_resid.npz"))["e"][TW:]
    s_all = dict(np.load(_p("bucket_signals.npz")))["all"]
    inc = sig["s_aug"] - sig["s_lin"]
    r = e - s_all
    n = len(e)
    B = 21 * PERIODS_PER_DAY
    nb = n // B

    def est(a: np.ndarray, b: np.ndarray) -> float:
        d = float(b @ b)
        return float(a @ b) / d if d > 0 else np.nan

    g_b = np.array([est(r[i * B : (i + 1) * B], inc[i * B : (i + 1) * B]) for i in range(nb)])
    h = B // 2
    g_1 = np.array([est(r[i * B : i * B + h], inc[i * B : i * B + h]) for i in range(nb)])
    g_2 = np.array([est(r[i * B + h : (i + 1) * B], inc[i * B + h : (i + 1) * B]) for i in range(nb)])
    ok = np.isfinite(g_b) & np.isfinite(g_1) & np.isfinite(g_2)
    g_b, g_1, g_2 = g_b[ok], g_1[ok], g_2[ok]
    nb = len(g_b)
    se = 1.0 / np.sqrt(nb)
    print(f"{nb} non-overlapping 21d blocks   g_b: mean {g_b.mean():+.3f}  sd {g_b.std():.3f}  "
          f"(se of a corr at this sample ~ {se:.3f})", flush=True)

    r_half = float(np.corrcoef(g_1, g_2)[0, 1])
    rel = 2 * r_half / (1 + r_half) if r_half > 0 else np.nan  # Spearman-Brown to full block
    print(f"\n  split-half corr {r_half:+.3f}  ->  full-block reliability {rel:.3f}  "
          f"(estimation-noise share of a monthly estimate: {1 - rel:.0%})")

    print("\n  lag(blocks)   rho_obs   rho_true(=obs/rel)")
    rows = []
    for k in range(1, 7):
        ro = float(np.corrcoef(g_b[:-k], g_b[k:])[0, 1])
        rt = min(ro / rel, 1.0) if rel and np.isfinite(rel) else np.nan
        print(f"      {k}         {ro:+.3f}      {rt:+.3f}")
        rows.append({"lag_blocks": k, "rho_obs": ro, "rho_true": rt})
    ro1 = rows[0]["rho_obs"]
    rt1 = rows[0]["rho_true"]
    dch = np.diff(g_b)
    ch_ac = float(np.corrcoef(dch[:-1], dch[1:])[0, 1])
    print(f"\n  block-change autocorr: {ch_ac:+.3f}  "
          f"(AR+measurement-noise predicts negative; pure random walk ~0)")
    print("\n  FORECASTABILITY LEDGER (one block ahead)")
    print(f"    naive AR on the noisy estimate captures : {100 * ro1 ** 2:.0f}% of estimate variance")
    print(f"    true dial persistence rho_true(1)       : {rt1:+.2f}")
    print(f"    ceiling for ANY history-based forecast  : {100 * rt1 ** 2:.0f}% of TRUE dial variance")
    print(f"    unforecastable estimation noise         : {100 * (1 - rel):.0f}% of every estimate")
    pd.DataFrame(rows).to_csv(f"{OUT}/dial_memory.csv", index=False)
    pd.DataFrame([{"n_blocks": nb, "split_half_corr": r_half, "reliability": rel,
                   "rho_obs_1": ro1, "rho_true_1": rt1, "change_autocorr": ch_ac,
                   "g_mean": float(g_b.mean()), "g_sd": float(g_b.std())}]
                 ).to_csv(f"{OUT}/dial_memory_summary.csv", index=False)
    print(f"wrote {OUT}/dial_memory.csv + dial_memory_summary.csv")


# ---------------------------------------------------------------------------
# concave — the zero-free-parameter response test (§19.1's leftover, run properly)
# ---------------------------------------------------------------------------


def stage_concave() -> None:
    """Gear the product increment by (trailing variance)^beta — beta frozen on <=2019, scored 2020+.

    §18.4 established the mechanism (the form's per-unit reliability falls with its own loudness)
    and §19.1 measured the exponent descriptively (full-span elasticity −0.357 ± 0.046). This is the
    forecast test, constructed so nothing is tuned on the scored block:

    * the exponent is RE-estimated on blocks entirely before ``HOLDOUT`` and frozen;
    * the gearing is ``w_t = (v_t / ref_t)^beta`` with ``v_t`` the trailing-21d mean of the
      increment's square (causal, shifted one bar) and ``ref_t`` its causal expanding median — a
      normalization with no fitted constant;
    * PRIMARY (pre-registered): geared vs ungeared monthly increment on 2020+, one-sided DM 1.645
      (the sign is fixed by §18.4's mechanism). Expectation: positive, DM ≈ +1.5..2.5 — the gearing
      should capture roughly the observable share of the dial's §18.2 value.
    * SECONDARY: the same gearing applied to the daily-refit increment, vs plain daily refit —
      does the concavity add anything the daily solver does not already track? Expectation: small,
      likely below threshold (daily refit follows the dial at ~days lag), and transferability of the
      monthly-measured exponent to the daily increment is an assumption, noted not hidden.

    Both scored on sqrt-space residual R² and on QLIKE through the repo reconstruction.
    """
    _require_fixed_cache()
    from src.features.transforms.target import PERIODS_PER_DAY

    p = load_panel()
    sig = dict(np.load(_p(CACHE)))
    z = np.load(_p("har_resid.npz"))
    e = z["e"][TW:]
    pred_har = z["pred"][TW:]
    s_all = dict(np.load(_p("bucket_signals.npz")))["all"]
    inc_m = sig["s_aug"] - sig["s_lin"]
    inc_d = sig["s_aug_daily"] - s_all
    y_adj = p.y[2 * TW :]
    baseline = p.baseline[2 * TW :]
    ts = pd.Series(pd.to_datetime(p.t[2 * TW :]))
    late = (ts >= HOLDOUT).to_numpy()
    n = len(e)
    B = 21 * PERIODS_PER_DAY

    # exponent, frozen on <=2019 blocks only
    r = e - s_all
    nb = n // B
    blk_start_late = np.array([bool(late[i * B]) for i in range(nb)])
    g_b, v_b = [], []
    for i in range(nb):
        sl = slice(i * B, (i + 1) * B)
        d = float(inc_m[sl] @ inc_m[sl])
        if d <= 0:
            continue
        if blk_start_late[i]:
            continue
        g_b.append(float(r[sl] @ inc_m[sl]) / d)
        v_b.append(d / B)
    g_b, v_b = np.array(g_b), np.array(v_b)
    pos = g_b > 0
    lg, lv = np.log(g_b[pos]), np.log(v_b[pos])
    beta = float(np.cov(lg, lv)[0, 1] / np.var(lv))
    se = float(np.sqrt((np.var(lg) / np.var(lv) - beta**2) / (pos.sum() - 2)))
    print(f"exponent frozen on <=2019: beta = {beta:+.3f} +/- {se:.3f} "
          f"({int(pos.sum())} blocks; full-span reference -0.357)", flush=True)

    def gear(inc: np.ndarray) -> np.ndarray:
        v = pd.Series(inc**2).rolling(B, min_periods=B // 2).mean().shift(1).to_numpy()
        ref = pd.Series(v).expanding(min_periods=B).median().to_numpy()
        ratio = np.clip(np.where((v > 0) & (ref > 0), v / ref, 1.0), 1e-2, 1e2)
        return ratio**beta

    w_m, w_d = gear(inc_m), gear(inc_d)
    print(f"gearing w (monthly inc): mean {np.nanmean(w_m):.3f}  sd {np.nanstd(w_m):.3f}  "
          f"[{np.nanpercentile(w_m, 1):.2f}, {np.nanpercentile(w_m, 99):.2f}]")

    arms = {
        "A_ungeared": s_all + inc_m,
        "G_geared": s_all + w_m * inc_m,
        "D_daily": s_all + inc_d,
        "DG_daily_geared": s_all + w_d * inc_d,
    }
    rows = []
    for span, m in (("2020+", late), ("full", np.ones(n, bool))):
        q = {a: _qlike_series(pred_har[m] + (f - s_all)[m] + s_all[m], y_adj[m], baseline[m])
             for a, f in arms.items()}
        for pair, tag in ((("G_geared", "A_ungeared"), "PRIMARY"),
                          (("DG_daily_geared", "D_daily"), "secondary")):
            a1, a0 = pair
            r1 = r2_oos(e[m], arms[a1][m])
            r0 = r2_oos(e[m], arms[a0][m])
            t_sq = dm_test(e[m], arms[a1][m], arms[a0][m])
            t_ql = _hac_mean_t(q[a0] - q[a1])
            dq = float(np.nanmean(q[a1]) - np.nanmean(q[a0]))
            print(f"  [{span:5s}] {tag:9s} {a1} vs {a0}: dR2 {r1 - r0:+.5f} (DM {t_sq:+.2f})  "
                  f"dQLIKE {dq:+.5f} (DM {t_ql:+.2f})", flush=True)
            rows.append({"span": span, "test": tag, "arm": a1, "ref": a0,
                         "dr2": r1 - r0, "dm_t_sqrt": t_sq, "dqlike": dq, "dm_t_qlike": t_ql,
                         "beta_frozen": beta, "n": int(m.sum())})
    prim = next(r_ for r_ in rows if r_["span"] == "2020+" and r_["test"] == "PRIMARY")
    print(f"\n  PRE-REGISTERED GATE (one-sided 1.645, sqrt-space, 2020+): "
          f"DM {prim['dm_t_sqrt']:+.2f} -> {'PASS' if prim['dm_t_sqrt'] > 1.645 else 'fail'}")
    pd.DataFrame(rows).to_csv(f"{OUT}/concave_response_test.csv", index=False)
    print(f"wrote {OUT}/concave_response_test.csv")


# ---------------------------------------------------------------------------
# joint — the fully-joint blockwise ridge (the deliverable's own sequencing test)
# ---------------------------------------------------------------------------


def stage_joint() -> None:
    """One blockwise ridge over all three blocks, vs the sequential deliverable.

    The deliverable is itself two-stage at the top level — HAR fit alone, exog on its residual —
    which is the same sequential-fitting pattern the §19.5 pilot measured costing 0.0058 inside
    stage B. This runs the fully-joint model: 27 HAR/calendar/regime columns at alpha = 1, 516 exog
    at 3,000, 100 frozen products at 30,000, one rank-1 rolling solver via column scaling
    (har x sqrt(3000/1), products x sqrt(3000/30000)), daily refit, predicting y directly.

    Confound controlled: stage A refits per bar, the joint fit daily, so the HAR block's
    daily-refit cost is measured alone alongside. Pre-registered: joint vs deliverable, sqrt DM > 2
    to claim; prediction positive but small — the HAR/exog correlation is weaker than the
    dense/product correlation that produced the pilot's 0.0058.
    """
    _require_fixed_cache()
    os.makedirs(OUT, exist_ok=True)
    from analysis.wf import walk_forward
    from src.features.transforms.target import PERIODS_PER_DAY

    p = load_panel()
    z = np.load(_p("har_resid.npz"))
    pred_har = z["pred"]
    sig = dict(np.load(_p(CACHE)))
    frozen = sig["frozen"]

    har_cols = np.concatenate([p.cols("har"), p.cols("calendar"), p.cols("regime")])
    lin_cols = np.concatenate([p.cols("value"), p.cols("indicator")])
    bc, _ = base_columns(p)
    XH = np.ascontiguousarray(p.X[:, har_cols], dtype=np.float64)
    XL = np.ascontiguousarray(p.X[:, lin_cols], dtype=np.float64)
    XB = np.ascontiguousarray(p.X[:, bc], dtype=np.float64)
    ii, jj = _upper(XB.shape[1])
    P = XB[:, ii[frozen]] * XB[:, jj[frozen]]
    B = 250 * PERIODS_PER_DAY
    sd = pd.DataFrame(P).rolling(B, min_periods=1000).std().shift(1)
    med = np.nanmedian(sd.to_numpy(), axis=1, keepdims=True)
    sdv = np.maximum(sd.to_numpy(), 0.1 * np.where(np.isfinite(med), med, 1.0))
    P = P / pd.DataFrame(sdv).bfill().to_numpy()

    ALPHA = 3000.0
    X = np.hstack([
        XH * np.sqrt(ALPHA / 1.0),          # har block: effective alpha 1
        XL,                                  # linear exog: alpha 3000
        P * np.sqrt(ALPHA / 3e4),            # products: effective alpha 3e4
    ])
    print(f"joint blockwise ridge: {X.shape[1]} cols "
          f"({len(har_cols)} har@1 + {XL.shape[1]}@3e3 + {P.shape[1]}@3e4), daily refit", flush=True)
    yhat_joint = walk_forward(X, p.y, TW, alpha=ALPHA, refit_every=PERIODS_PER_DAY)

    # confound control: the HAR block alone at daily refit (stage A uses per-bar)
    pred_har_daily = walk_forward(
        np.ascontiguousarray(XH), p.y, TW, alpha=1.0, refit_every=PERIODS_PER_DAY
    )

    # score on the common OOS block (rows 2TW: of the panel), same as every §18 arm
    y_adj = p.y[2 * TW :]
    baseline = p.baseline[2 * TW :]
    ts = pd.Series(pd.to_datetime(p.t[2 * TW :]))
    late = (ts >= HOLDOUT).to_numpy()
    yc = y_adj - y_adj.mean()
    arms = {
        "deliverable (sequential)": pred_har[TW:] + sig["s_aug_daily"],
        "joint blockwise": yhat_joint[TW:],
    }
    print(f"\n  HAR alone: per-bar refit R2 "
          f"{r2_oos(yc, pred_har[TW:] - y_adj.mean()):+.5f}   daily refit "
          f"{r2_oos(yc, pred_har_daily[TW:] - y_adj.mean()):+.5f}   "
          f"(the joint arm carries the daily-refit HAR handicap)")
    qs = {}
    for name, f in arms.items():
        qs[name] = _qlike_series(f, y_adj, baseline)
        print(f"  {name:26s} R2(vs mean) {r2_oos(yc, f - y_adj.mean()):+.5f}   "
              f"QLIKE {np.nanmean(qs[name]):.5f}", flush=True)
    a, b = arms["joint blockwise"], arms["deliverable (sequential)"]
    t_sq = dm_test(y_adj, a, b)
    t_sq_l = dm_test(y_adj[late], a[late], b[late])
    d = qs["deliverable (sequential)"] - qs["joint blockwise"]
    print(f"\n  joint vs deliverable: sqrt DM {t_sq:+.2f} (2020+ {t_sq_l:+.2f})   "
          f"QLIKE DM {_hac_mean_t(d):+.2f} (2020+ {_hac_mean_t(d[late]):+.2f})")
    print(f"  PRE-REGISTERED GATE (sqrt DM > 2): {'PASS' if t_sq > 2 else 'fail'}")
    pd.DataFrame([{
        "r2_joint": r2_oos(yc, a - y_adj.mean()), "r2_deliverable": r2_oos(yc, b - y_adj.mean()),
        "qlike_joint": float(np.nanmean(qs["joint blockwise"])),
        "qlike_deliverable": float(np.nanmean(qs["deliverable (sequential)"])),
        "dm_t_sqrt": t_sq, "dm_t_sqrt_2020plus": t_sq_l,
        "dm_t_qlike": _hac_mean_t(d), "dm_t_qlike_2020plus": _hac_mean_t(d[late]),
    }]).to_csv(f"{OUT}/joint_vs_sequential.csv", index=False)
    print(f"wrote {OUT}/joint_vs_sequential.csv")


# ---------------------------------------------------------------------------
# final — ONE stage, everything, refit per bar (user-directed composition)
# ---------------------------------------------------------------------------


def stage_final() -> None:
    """The user-directed final composition: one blockwise ridge, xsec included, per-bar refit.

    679 columns in a single rank-1 rolling solver: the 27-column backbone at effective alpha 1,
    the 516 exog at 3,000, the 36 §20 cross-section ratios at 3,000, the 100 frozen products at
    30,000 — all imposed by column scaling — predicting y directly, 250-day window. Run at two
    cadences (daily, every bar) so the cadence effect is attributable on THIS panel rather than
    borrowed from the sibling session's rebuild. Scored on the §18 common OOS block against the
    two-stage deliverable, sqrt and QLIKE, with prediction_health.

    Composition notes, from the measured record: the joint-vs-sequential difference is a tie
    (§21, FWL-protected); the xsec increment was +3.9/+2.6 at daily refit with a dated edge
    (§20); per-bar refit's increment over daily was linear-channel-driven on the sibling panel
    (§21.1). The composed model inherits all three claims; what is new here is only their
    conjunction.
    """
    _require_fixed_cache()
    os.makedirs(OUT, exist_ok=True)
    from analysis.wf import walk_forward
    from src.features.transforms.target import PERIODS_PER_DAY

    p = load_panel()
    sig = dict(np.load(_p(CACHE)))
    frozen = sig["frozen"]
    har_cols = np.concatenate([p.cols("har"), p.cols("calendar"), p.cols("regime")])
    lin_cols = np.concatenate([p.cols("value"), p.cols("indicator")])
    bc, _ = base_columns(p)
    XH = np.ascontiguousarray(p.X[:, har_cols], dtype=np.float64)
    XL = np.ascontiguousarray(p.X[:, lin_cols], dtype=np.float64)
    XB = np.ascontiguousarray(p.X[:, bc], dtype=np.float64)
    XS = np.load(_p("xsec_features.npz"))["F"].astype(np.float64)
    ii, jj = _upper(XB.shape[1])
    P = XB[:, ii[frozen]] * XB[:, jj[frozen]]
    B = 250 * PERIODS_PER_DAY
    sd = pd.DataFrame(P).rolling(B, min_periods=1000).std().shift(1)
    med = np.nanmedian(sd.to_numpy(), axis=1, keepdims=True)
    sdv = np.maximum(sd.to_numpy(), 0.1 * np.where(np.isfinite(med), med, 1.0))
    P = P / pd.DataFrame(sdv).bfill().to_numpy()

    ALPHA = 3000.0
    X = np.hstack([
        XH * np.sqrt(ALPHA / 1.0),
        XL,
        XS,
        P * np.sqrt(ALPHA / 3e4),
    ])
    print(f"one-stage model: {X.shape[1]} cols "
          f"({len(har_cols)} backbone@1 + {XL.shape[1]}@3e3 + {XS.shape[1]} xsec@3e3 + "
          f"{P.shape[1]} products@3e4)", flush=True)

    yhat_daily = walk_forward(X, p.y, TW, alpha=ALPHA, refit_every=PERIODS_PER_DAY)
    print("daily-refit twin done; starting per-bar refit (~1h)", flush=True)
    yhat_bar = walk_forward(X, p.y, TW, alpha=ALPHA, refit_every=1)

    y_adj = p.y[2 * TW :]
    baseline = p.baseline[2 * TW :]
    ts = pd.Series(pd.to_datetime(p.t[2 * TW :]))
    late = (ts >= HOLDOUT).to_numpy()
    yc = y_adj - y_adj.mean()
    z = np.load(_p("har_resid.npz"))
    arms = {
        "deliverable (2-stage, daily, no xsec)": z["pred"][TW:] + sig["s_aug_daily"],
        "one-stage daily": yhat_daily[TW:],
        "one-stage PER BAR": yhat_bar[TW:],
    }
    from src.diagnostics import prediction_health

    qs, rows = {}, []
    for name, f in arms.items():
        qs[name] = _qlike_series(f, y_adj, baseline)
        h = prediction_health(yc, f - y_adj.mean())
        print(f"  {name:38s} R2 {r2_oos(yc, f - y_adj.mean()):+.5f}  "
              f"QLIKE {np.nanmean(qs[name]):.5f}  health {h['status']}", flush=True)
        rows.append({"arm": name, "r2": r2_oos(yc, f - y_adj.mean()),
                     "qlike": float(np.nanmean(qs[name])), "health": h["status"]})
    a = arms["one-stage PER BAR"]
    for ref in ("one-stage daily", "deliverable (2-stage, daily, no xsec)"):
        b = arms[ref]
        d = qs[ref] - qs["one-stage PER BAR"]
        print(f"\n  per-bar vs {ref}:")
        print(f"    sqrt DM {dm_test(y_adj, a, b):+.2f} (2020+ "
              f"{dm_test(y_adj[late], a[late], b[late]):+.2f})   "
              f"QLIKE DM {_hac_mean_t(d):+.2f} (2020+ {_hac_mean_t(d[late]):+.2f})")
    np.savez_compressed(_p("final_onestage.npz"), yhat_bar=yhat_bar, yhat_daily=yhat_daily)
    pd.DataFrame(rows).to_csv(f"{OUT}/final_onestage.csv", index=False)
    print(f"wrote {OUT}/final_onestage.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stage", choices=[
            "build", "daily", "verify", "gain", "driver", "memory", "concave", "joint", "final",
        ],
        required=True
    )
    a = ap.parse_args()
    {
        "build": stage_build,
        "daily": stage_daily,
        "verify": stage_verify,
        "gain": stage_gain,
        "driver": stage_driver,
        "memory": stage_memory,
        "concave": stage_concave,
        "joint": stage_joint,
        "final": stage_final,
    }[a.stage]()
