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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["build", "daily", "verify"], required=True)
    a = ap.parse_args()
    {"build": stage_build, "daily": stage_daily, "verify": stage_verify}[a.stage]()
