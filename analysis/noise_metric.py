"""Cashing the principle: the noise's metric, pulled back onto two more levels of the stack.

§19.1 formalized what the whole study kept finding — the signal has no geometry, the noise does, and
every win was a re-expression of some quantity in units of its own local uncertainty (target: QLIKE
= Fisher–Rao on the variance cone; clock: per-slot division; features: diagonal whitening; the
form's output: the dial). This module applies the same move at the two remaining cheap levels:

``smear``
    **The reconstruction.** ``apply_duan_smearing`` adds one GLOBAL error-variance constant:
    ``pred_raw = (f² + E[(y−f)²]) · baseline``. But the sqrt-space error variance is strongly
    conditional — overnight slots and RTH slots do not share a residual distribution (§16.4 was
    entirely about that fact). The metric version conditions the smear on the clock:
    ``smear_s(slot)``, a causal expanding mean of squared error per 30-min slot, shifted one day.
    48 groups, no fitted parameters, and both variants are made CAUSAL (expanding) so the only
    difference is the conditioning. This is calibration, not alpha: the conditional mean of raw RV
    given an unbiased sqrt-space forecast simply *is* the conditionally-smeared reconstruction.

``wls``
    **The estimation loss.** Stage B minimizes unweighted sqrt-space MSE, but the production loss is
    QLIKE ≈ 2·(y−f)²/f² to second order — each bar's error should be weighted by the inverse squared
    *level*. The sqrt chart flattens part of the variance structure; the loss metric is log, so the
    residual misalignment is a per-bar weight ``w_t ∝ 1/pred_HAR(t)²`` (causal — the HAR forecast is
    the best available causal level estimate). Implemented exactly in the same rank-1 rolling solver
    by row scaling: WLS(w) = OLS on ``(√w·x, √w·y)``, with a ``√w`` constant column carrying the
    intercept and predictions un-scaled by ``√w_t``. The weight normalization is fixed causally on
    the first training window, so the effective ridge penalty is unchanged.

Pre-registered gates and predictions, written before running:

* ``smear`` — claim at QLIKE DM > 2 vs the causal-global smear, same forecast. Prediction: PASSES
  for the HAR arm (slot-conditional error variance is large and §16.4-documented), and for the
  deliverable arm as well.
* ``wls`` — claim at QLIKE DM > 2 vs the unweighted deliverable. Prediction: positive but possibly
  under threshold — the sqrt transform already absorbs much of the metric misalignment; sqrt-space
  R² is expected to WORSEN slightly (it is a different metric — that is the point, and a worsening
  there with a QLIKE gain would be the principle's cleanest possible signature).

Usage
-----
    C=.../scratchpad/fixed
    ALPHA_PANEL_CACHE=$C python analysis/noise_metric.py --stage smear
    ALPHA_PANEL_CACHE=$C python analysis/noise_metric.py --stage wls
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
from analysis.minimal_model import CACHE, _hac_mean_t, _qlike_series  # noqa: E402
from analysis.nl_sparsity import _upper, base_columns  # noqa: E402
from analysis.synthesis import HOLDOUT, _p, _require_fixed_cache  # noqa: E402
from analysis.wf import r2_oos  # noqa: E402
from src.features.transforms.target import PERIODS_PER_DAY  # noqa: E402

OUT = "results/alpha_manifestation"


def _load_common():
    p = load_panel()
    z = np.load(_p("har_resid.npz"))
    e = z["e"][TW:]
    pred_har = z["pred"][TW:]
    sig = dict(np.load(_p(CACHE)))
    s_all = dict(np.load(_p("bucket_signals.npz")))["all"]
    y_adj = p.y[2 * TW :]
    baseline = p.baseline[2 * TW :]
    ts = pd.Series(pd.to_datetime(p.t[2 * TW :]))
    return p, e, pred_har, sig, s_all, y_adj, baseline, ts


def _qlike(pred_raw: np.ndarray, true_raw: np.ndarray) -> np.ndarray:
    out = np.full(len(pred_raw), np.nan)
    m = (pred_raw > 0) & (true_raw > 0)
    r = true_raw[m] / pred_raw[m]
    out[m] = r - np.log(r) - 1.0
    return out


def stage_smear() -> None:
    """Causal per-slot smearing vs causal global smearing, same forecasts."""
    _require_fixed_cache()
    os.makedirs(OUT, exist_ok=True)
    _, e, pred_har, sig, s_all, y_adj, baseline, ts = _load_common()
    true_raw = (y_adj**2) * baseline
    slot = (ts.dt.hour * 2 + ts.dt.minute // 30).to_numpy()
    late = (ts >= HOLDOUT).to_numpy()

    arms = {
        "har": pred_har,
        "deliverable": pred_har + sig["s_aug_daily"],
    }
    rows = []
    for name, f in arms.items():
        err2 = (y_adj - f) ** 2
        # causal global: expanding mean of squared error, shifted one bar
        sm_g = pd.Series(err2).expanding(min_periods=480).mean().shift(1).to_numpy()
        # causal per-slot: expanding mean within slot, shifted one in-slot observation
        sm_s = (
            pd.Series(err2)
            .groupby(slot)
            .transform(lambda g: g.expanding(min_periods=60).mean().shift(1))
            .to_numpy()
        )
        sm_s = np.where(np.isfinite(sm_s), sm_s, sm_g)  # slot warm-up falls back to global
        q_g = _qlike((f**2 + np.nan_to_num(sm_g, nan=np.nanmean(err2))) * baseline, true_raw)
        q_s = _qlike((f**2 + sm_s) * baseline, true_raw)
        d = q_g - q_s
        t = _hac_mean_t(d)
        t_l = _hac_mean_t(d[late])
        print(f"  {name:12s} QLIKE global {np.nanmean(q_g):.5f} -> per-slot {np.nanmean(q_s):.5f}  "
              f"d {np.nanmean(q_g) - np.nanmean(q_s):+.5f}  DM {t:+.2f}   (2020+ DM {t_l:+.2f})",
              flush=True)
        rows.append({"arm": name, "qlike_global": float(np.nanmean(q_g)),
                     "qlike_perslot": float(np.nanmean(q_s)), "dm_t": t, "dm_t_2020plus": t_l})
    pd.DataFrame(rows).to_csv(f"{OUT}/noise_metric_smear.csv", index=False)
    print(f"wrote {OUT}/noise_metric_smear.csv")


def stage_wls() -> None:
    """The 616-column daily-refit stage B, refit under QLIKE-aligned per-bar weights."""
    _require_fixed_cache()
    os.makedirs(OUT, exist_ok=True)
    from src.models.rolling_least_squares import RollingLeastSquares

    p, e, pred_har, sig, s_all, y_adj, baseline, ts = _load_common()
    e_full = np.load(_p("har_resid.npz"))["e"]
    pred_har_full = np.load(_p("har_resid.npz"))["pred"]
    frozen = sig["frozen"]

    # QLIKE-aligned weights: 1 / level^2, level = the causal HAR forecast, clipped at its own
    # causal 1st percentile so a stray near-zero forecast cannot dominate a window
    lvl = pred_har_full.copy()
    lo = np.nanpercentile(lvl[: TW], 1)
    w = 1.0 / np.clip(lvl, max(lo, 1e-3), None) ** 2
    w = w / w[:TW].mean()  # causal normalization: effective ridge penalty unchanged
    sw = np.sqrt(w)

    lin_cols = np.concatenate([p.cols("value"), p.cols("indicator")])
    bc, _ = base_columns(p)
    XL = np.ascontiguousarray(p.X[TW:, lin_cols], dtype=np.float64)
    XB = np.ascontiguousarray(p.X[TW:, bc], dtype=np.float64)
    ii, jj = _upper(XB.shape[1])
    P = XB[:, ii[frozen]] * XB[:, jj[frozen]]
    B = 250 * PERIODS_PER_DAY
    sd = pd.DataFrame(P).rolling(B, min_periods=1000).std().shift(1)
    med = np.nanmedian(sd.to_numpy(), axis=1, keepdims=True)
    sdv = np.maximum(sd.to_numpy(), 0.1 * np.where(np.isfinite(med), med, 1.0))
    sdv = pd.DataFrame(sdv).bfill().to_numpy()
    P = P / sdv * np.sqrt(3000.0 / 3e4)
    X = np.hstack([XL, P, np.ones((len(XL), 1))])  # explicit constant column, weighted below

    Xw = X * sw[:, None]
    yw = e_full * sw
    n = len(Xw)
    solver = RollingLeastSquares(alpha=3000.0, fit_intercept=False)
    solver.init_window(Xw[:TW], yw[:TW])
    solver.solve()
    out = np.empty(n - TW)
    refit = PERIODS_PER_DAY
    for i in range(n - TW):
        t = TW + i
        if i and i % refit == 0:
            solver.solve()
        out[i] = solver.predict_one(Xw[t]) / sw[t]
        solver.roll(Xw[t], yw[t], Xw[t - TW], yw[t - TW])
    s_wls = out

    y = e_full[TW:]
    s_ref = sig["s_aug_daily"]
    late = (ts >= HOLDOUT).to_numpy()
    true_raw = (y_adj**2) * baseline

    def q(f_sig: np.ndarray) -> np.ndarray:
        f = pred_har + f_sig
        smear = float(np.mean((y_adj - f) ** 2))
        return _qlike((f**2 + smear) * baseline, true_raw)

    q_ref, q_wls = q(s_ref), q(s_wls)
    print(f"  sqrt-space resid R2: unweighted {r2_oos(y, s_ref):+.5f}  "
          f"QLIKE-weighted {r2_oos(y, s_wls):+.5f}  DM {dm_test(y, s_wls, s_ref):+.2f}")
    t = _hac_mean_t(q_ref - q_wls)
    t_l = _hac_mean_t((q_ref - q_wls)[late])
    print(f"  QLIKE: unweighted {np.nanmean(q_ref):.5f}  weighted {np.nanmean(q_wls):.5f}  "
          f"d {np.nanmean(q_ref) - np.nanmean(q_wls):+.5f}  DM {t:+.2f}   (2020+ DM {t_l:+.2f})")
    print(f"  PRE-REGISTERED GATE (QLIKE DM > 2): {'PASS' if t > 2 else 'fail'}")
    pd.DataFrame([{"r2_unweighted": r2_oos(y, s_ref), "r2_wls": r2_oos(y, s_wls),
                   "dm_t_sqrt": dm_test(y, s_wls, s_ref),
                   "qlike_unweighted": float(np.nanmean(q_ref)),
                   "qlike_wls": float(np.nanmean(q_wls)),
                   "dm_t_qlike": t, "dm_t_qlike_2020plus": t_l}]
                 ).to_csv(f"{OUT}/noise_metric_wls.csv", index=False)
    print(f"wrote {OUT}/noise_metric_wls.csv")


# ---------------------------------------------------------------------------
# kernel — item 4's pilot: does a MATCHED kernel beat the flat 250d window?
# ---------------------------------------------------------------------------

# Halflife pinned by §18.4's measured dial memory (trackable at ~days-to-weeks, gone at a month),
# NOT tuned: 21 trading days. 63d reported as a secondary observation, labeled as such.
KERNEL_HALFLIFE_DAYS = 21
KERNEL_HALFLIFE_SECONDARY = 63


def stage_kernel() -> None:
    """Two-stage pilot for per-coefficient precision updating (the §19.3 program's item 4).

    The 250-day flat window was inherited by the product block from the linear stage's tuning and
    never tested — the same class of untested inheritance as the monthly refit cadence, whose
    correction produced §18.1. This pilot isolates KERNEL SHAPE with everything else identical:

        stage 1 (fixed)   the dense daily 250d ridge, ``s_all`` (the deliverable's linear stage)
        stage 2, arm A    the 100 frozen products fit on the stage-1 residual with a FLAT 250d
                          rolling window, daily refit
        stage 2, arm B    identical, but the Gram is EXPONENTIALLY forgotten with halflife 21d
                          (pinned by §18.4), ridge penalty scaled by the effective-sample ratio so
                          shrinkage per observation is identical

    PRIMARY gate: B vs A, sqrt DM > 2 (QLIKE reported). Both are also compared to the one-stage
    deliverable so the two-stage architecture cost is visible rather than confounded. If B clears
    its gate, the full diagonal-Kalman (per-coefficient halflives) is justified; if not, item 4
    ends at the price of a pilot. Prediction, written first: +0.001..0.003, a coin flip against
    the gate — §19.2 says daily refit already harvests the trackable motion, the §18.1 precedent
    says untested inheritances keep paying.
    """
    _require_fixed_cache()
    os.makedirs(OUT, exist_ok=True)
    p, e, pred_har, sig, s_all, y_adj, baseline, ts = _load_common()
    frozen = sig["frozen"]
    bc, _ = base_columns(p)
    XB = np.ascontiguousarray(p.X[TW:, bc], dtype=np.float64)
    ii, jj = _upper(XB.shape[1])
    P = XB[:, ii[frozen]] * XB[:, jj[frozen]]
    B = 250 * PERIODS_PER_DAY
    sd = pd.DataFrame(P).rolling(B, min_periods=1000).std().shift(1)
    med = np.nanmedian(sd.to_numpy(), axis=1, keepdims=True)
    sdv = np.maximum(sd.to_numpy(), 0.1 * np.where(np.isfinite(med), med, 1.0))
    P = P / pd.DataFrame(sdv).bfill().to_numpy()
    P = np.hstack([P, np.ones((len(P), 1))])  # explicit intercept column

    # stage-1 residual on panel rows 2TW: (where s_all lives)
    r = e - s_all
    Pr = P[TW:]
    n, q = Pr.shape
    alpha_flat = 3e4

    def flat_arm() -> np.ndarray:
        from analysis.wf import walk_forward as wf

        return wf(Pr, r, TW, alpha=alpha_flat, refit_every=PERIODS_PER_DAY)

    def ewma_arm(hl_days: int) -> np.ndarray:
        lam = 0.5 ** (1.0 / (hl_days * PERIODS_PER_DAY))
        wsum = 1.0 / (1.0 - lam)
        a_eff = alpha_flat * (wsum / TW)  # penalty per effective observation held constant
        G = np.zeros((q, q))
        c = np.zeros(q)
        b = np.zeros(q)
        out = np.full(n - TW, np.nan)
        for t in range(n):
            if t >= TW:
                if t % PERIODS_PER_DAY == 0:
                    b = np.linalg.solve(G + a_eff * np.eye(q), c)
                out[t - TW] = float(Pr[t] @ b)  # predict BEFORE seeing (x_t, r_t)
            G *= lam
            c *= lam
            G += np.outer(Pr[t], Pr[t])
            c += Pr[t] * r[t]
        return out

    s_flat = flat_arm()
    s_e21 = ewma_arm(KERNEL_HALFLIFE_DAYS)
    s_e63 = ewma_arm(KERNEL_HALFLIFE_SECONDARY)

    # common scored rows: both stage-2 arms' warm-ups excluded
    m = np.zeros(len(e), dtype=bool)
    m[2 * TW :] = True
    late = (ts >= HOLDOUT).to_numpy() & m
    s_del = sig["s_aug_daily"]  # one-stage deliverable, for architecture context
    arms = {
        "A_flat250": s_all[TW:] + np.nan_to_num(s_flat),
        "B_ewma21": s_all[TW:] + np.nan_to_num(s_e21[: len(s_flat)]),
        "C_ewma63": s_all[TW:] + np.nan_to_num(s_e63[: len(s_flat)]),
        "deliverable": s_del[TW:],
    }
    y = e[TW:]
    ml = late[TW:]
    rows = []
    qs = {}
    for name, f in arms.items():
        qs[name] = _qlike_series(pred_har[TW:] + (f - s_all[TW:]) + s_all[TW:],
                                 y_adj[TW:], baseline[TW:])
        rows.append({"arm": name, "r2": r2_oos(y, f)})
        print(f"  {name:12s} resid R2 {r2_oos(y, f):+.5f}   QLIKE {np.nanmean(qs[name]):.5f}",
              flush=True)
    tBA = dm_test(y, arms["B_ewma21"], arms["A_flat250"])
    tBA_l = dm_test(y[ml], arms["B_ewma21"][ml], arms["A_flat250"][ml])
    tql = _hac_mean_t(qs["A_flat250"] - qs["B_ewma21"])
    print(f"\n  PRIMARY  B_ewma21 vs A_flat250: dR2 "
          f"{r2_oos(y, arms['B_ewma21']) - r2_oos(y, arms['A_flat250']):+.5f}  "
          f"sqrt DM {tBA:+.2f} (2020+ {tBA_l:+.2f})  QLIKE DM {tql:+.2f}")
    print(f"  GATE (sqrt DM > 2): {'PASS -> build the full diagonal Kalman' if tBA > 2 else 'fail -> item 4 closes'}")
    tCA = dm_test(y, arms["C_ewma63"], arms["A_flat250"])
    print(f"  secondary C_ewma63 vs A: DM {tCA:+.2f}   "
          f"deliverable vs A (architecture cost): DM {dm_test(y, arms['deliverable'], arms['A_flat250']):+.2f}")
    pd.DataFrame(rows).to_csv(f"{OUT}/noise_metric_kernel.csv", index=False)
    print(f"wrote {OUT}/noise_metric_kernel.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["smear", "wls", "kernel"], required=True)
    a = ap.parse_args()
    {"smear": stage_smear, "wls": stage_wls, "kernel": stage_kernel}[a.stage]()
