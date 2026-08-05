"""Stage-B refit cadence below daily, and saturation as tail insurance — the two §19 leftovers.

Two open cells from the study, run on the §8-fixed panel with the study's own machinery:

1. **Intraday refit cadence.** §18/§18.1 found refit cadence to be the single largest lever for
   Stage B (monthly -> daily: dense +0.0204 -> +0.0377, products carried to QLIKE only at daily),
   and the cadence axis was measured at exactly two points. Nobody ran faster-than-daily. This
   module runs the deliverable's 616-column design through the SAME rank-1 rolling solver at
   refit_every in {1008 (monthly), 48 (daily, cached reference), 24, 8, 1 (every bar)}, so the
   whole curve comes from one code path and every pairwise comparison is bar-for-bar.

   Pre-registered expectations, fixed before running: §18.4 found the dial's trackable motion
   decays within days and has no measured sub-daily structure, so the prior is DIMINISHING
   RETURNS — refit-1 at most marginally better than refit-48, well short of the monthly->daily
   step. The gate for claiming an intraday-refit gain is the study's usual one: QLIKE DM-t > 2
   (HAC, 480 lags) for the faster arm against refit-48, on the full span. The dense-only
   516-column design is run at refit-1 alongside, so any gain attributes to the linear or the
   product channel. No prediction is made for the split.

2. **The joint group-ridge arm.** The deliverable is ONE backfitting pass on the blockwise
   (group-ridge) objective: Stage A gets undiluted first claim on the variance it shares with
   the exog block, because Stage B only ever sees A's residual. The converged alternative is a
   single solve on [backbone | exog | products] with the SAME per-block penalties (1 / 3000 /
   3e4), imposed exactly by column scaling (a column scaled by c has effective penalty alpha/c^2).
   Same solver, same panel, same cadence as the two-stage reference — the only thing that changes
   is how shared variance is split between the blocks.

   Pre-registered expectation: joint is a wash or slightly WORSE — several exog columns are
   themselves volatility measures, and the study's marginal-vs-partial kernel result says letting
   them compete with the backbone for persistence redistributes mass away from it. Directional
   only; no gate — whichever way it lands, the number is the answer to "does one ridge work".

3. **Saturation (the "tail insurance" cell).** §19 identified the response saturation a tree's
   bounded leaves provide as the one playbook verb the two-ridge model lacks, and §18.1's
   Volmageddon bar (product block forecasting +2.2..+3.4 sigma into a +1.8 -> -2.8 reversal) as
   the cost of not having it. §19.2's concave *gearing* failed its gate as an average-gain
   claim — but insurance is not an average-gain claim. This stage bounds the product increment
   with a causal tanh clamp,

       inc_sat = k * s_t * tanh(inc / (k * s_t)),   s_t = trailing-21d sd of inc, shifted 1 bar,

   which is exactly a leaf's saturation in closed form: identity for |inc| << k*s_t, hard bound
   at k*s_t. Primary k = 3 (fixed a priori: the health gate is stated in sigmas and the shouts
   are 2-3+ sigma); k in {2, 4} reported as sensitivity, not selected over.

   Pre-registered verdict rule, fixed before running: saturation SHIPS if (a) the average QLIKE
   cost is indistinguishable from zero (|DM-t| < 2 vs unsaturated, full span) and (b) the
   worst-bar amplitude max|s|/sd(e) strictly falls. It is REJECTED if the average cost is
   significant. It is NOT required to win on average — that is the point of insurance.

Usage (cache must be the .../fixed dir):

    C=.../fixed
    ALPHA_PANEL_CACHE=$C python analysis/refit_cadence.py --stage walk --design aug   --refit 1
    ALPHA_PANEL_CACHE=$C python analysis/refit_cadence.py --stage walk --design aug   --refit 8
    ALPHA_PANEL_CACHE=$C python analysis/refit_cadence.py --stage walk --design aug   --refit 24
    ALPHA_PANEL_CACHE=$C python analysis/refit_cadence.py --stage walk --design aug   --refit 1008
    ALPHA_PANEL_CACHE=$C python analysis/refit_cadence.py --stage walk --design dense --refit 1
    ALPHA_PANEL_CACHE=$C python analysis/refit_cadence.py --stage walk --design joint --refit 48
    ALPHA_PANEL_CACHE=$C python analysis/refit_cadence.py --stage score
    ALPHA_PANEL_CACHE=$C python analysis/refit_cadence.py --stage saturate
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
from analysis.minimal_model import (  # noqa: E402
    ALPHA_LIN,
    CACHE,
    PROD_COL_SCALE,
    _hac_mean_t,
    _qlike_series,
)
from analysis.nl_sparsity import _products, _upper, base_columns  # noqa: E402
from analysis.synthesis import HOLDOUT, _p, _require_fixed_cache  # noqa: E402
from analysis.wf import r2_oos, walk_forward  # noqa: E402

OUT = "results/alpha_manifestation"
SIG = "refit_cadence_signals.npz"
CADENCES = (1008, 48, 24, 8, 1)  # monthly, daily (cached), half-day, hourly, every bar
SAT_K = (2.0, 3.0, 4.0)  # tanh bound in trailing sigmas; 3 is primary, fixed a priori
VOLMAGEDDON = "2018-02-06"


BACKBONE_COL_SCALE = float(np.sqrt(ALPHA_LIN / 1.0))  # solver alpha 3000 -> effective alpha 1


def _designs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(X_aug, X_dense, X_backbone, e_full), rows TW: — product block bit-identical to
    minimal_model.stage_daily."""
    p = load_panel()
    e_full = np.load(_p("har_resid.npz"))["e"]
    frozen = np.load(_p(CACHE))["frozen"]
    lin_cols = np.concatenate([p.cols("value"), p.cols("indicator")])
    har_cols = np.concatenate([p.cols("har"), p.cols("calendar"), p.cols("regime")])
    bc, _ = base_columns(p)
    XL = np.ascontiguousarray(p.X[TW:, lin_cols], dtype=np.float64)
    XH = np.ascontiguousarray(p.X[TW:, har_cols], dtype=np.float64)
    XB = np.ascontiguousarray(p.X[TW:, bc], dtype=np.float64)
    ii, jj = _upper(XB.shape[1])
    P = XB[:, ii[frozen]] * XB[:, jj[frozen]]
    sd = pd.DataFrame(P).rolling(TW, min_periods=1000).std().shift(1).to_numpy()
    med = np.nanmedian(sd, axis=1, keepdims=True)
    sd = np.maximum(sd, 0.1 * np.where(np.isfinite(med), med, 1.0))
    sd = pd.DataFrame(sd).bfill().to_numpy()
    P = P / sd * PROD_COL_SCALE
    return np.hstack([XL, P]), XL, XH, e_full


def stage_walk(design: str, refit: int) -> None:
    _require_fixed_cache()
    X_aug, X_dense, XH, e_full = _designs()
    if design == "joint":
        # the group-ridge fixed point: one solve, per-block penalties via column scaling,
        # target = adj_RV itself (there is no residual stage to hand anything to)
        p = load_panel()
        X = np.hstack([XH * BACKBONE_COL_SCALE, X_aug])
        y_t = p.y[TW:]
    else:
        X = X_aug if design == "aug" else X_dense
        y_t = e_full
    key = f"{design}_r{refit}"
    print(f"walk {key}: {X.shape[1]} cols, refit_every={refit}, n={len(X)}", flush=True)
    s = walk_forward(X, y_t, TW, alpha=ALPHA_LIN, refit_every=refit)
    # one file per arm: walks run as parallel processes and must not clobber a shared npz
    np.savez_compressed(_p(f"refit_cadence_{key}.npz"), s=s)
    pred_har = np.load(_p("har_resid.npz"))["pred"][TW:]
    resid_sig = s - pred_har if design == "joint" else s
    print(f"  {key}: resid R2 {r2_oos(e_full[TW:], resid_sig):+.5f}   "
          f"-> refit_cadence_{key}.npz", flush=True)


def _load_arms() -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """All cadence signals (cached refit-48 arms included), HAR pred, sqrt residual."""
    import glob

    z = np.load(_p("har_resid.npz"))
    mm = dict(np.load(_p(CACHE)))
    sig = {
        os.path.basename(f)[len("refit_cadence_"):-len(".npz")]: np.load(f)["s"]
        for f in glob.glob(_p("refit_cadence_*.npz"))
    }
    sig["aug_r48"] = mm["s_aug_daily"]  # §18.1's arm, same solver + design
    sig["dense_r48"] = dict(np.load(_p("bucket_signals.npz")))["all"]
    return sig, z["pred"][TW:], z["e"][TW:]


def stage_score() -> None:
    _require_fixed_cache()
    os.makedirs(OUT, exist_ok=True)
    p = load_panel()
    sig, pred_har, e = _load_arms()
    y_adj = p.y[2 * TW :]
    baseline = p.baseline[2 * TW :]
    late = (pd.Series(pd.to_datetime(p.t[2 * TW :])) >= HOLDOUT).to_numpy()

    # joint arms predict the level y directly; two-stage arms are residual signals on pred_har
    pred = {k: (s if k.startswith("joint") else pred_har + s) for k, s in sig.items()}
    q = {k: _qlike_series(f, y_adj, baseline) for k, f in pred.items()}
    ref = "aug_r48"
    rows = []
    print(f"{'arm':12s} {'refit':>6s} {'resid R2':>9s} {'QLIKE':>8s} "
          f"{'dQL vs r48':>11s} {'DM-t':>6s} | 2020+ DM-t")
    for design in ("aug", "dense", "joint"):
        for r in CADENCES:
            k = f"{design}_r{r}"
            if k not in sig:
                continue
            base = q["dense_r48"] if design == "dense" else q[ref]
            dq = float(np.nanmean(q[k]) - np.nanmean(base))
            t = _hac_mean_t(base - q[k]) if k not in (ref, "dense_r48") else 0.0
            t_l = _hac_mean_t((base - q[k])[late]) if k not in (ref, "dense_r48") else 0.0
            rr = r2_oos(e, pred[k] - pred_har)
            print(f"{k:12s} {r:6d} {rr:+9.5f} {float(np.nanmean(q[k])):8.5f} "
                  f"{dq:+11.5f} {t:6.2f} | {t_l:6.2f}", flush=True)
            rows.append({"arm": k, "design": design, "refit_every": r,
                         "resid_r2": rr, "qlike": float(np.nanmean(q[k])),
                         "dqlike_vs_r48": dq, "dm_t_vs_r48": t, "dm_t_vs_r48_2020plus": t_l})
    if "aug_r1" in sig and "dense_r1" in sig:
        # attribution: how much of any refit-1 gain is the product block vs the linear block
        d_lin = float(np.nanmean(q["dense_r1"]) - np.nanmean(q["dense_r48"]))
        d_aug = float(np.nanmean(q["aug_r1"]) - np.nanmean(q[ref]))
        print(f"\nattribution of the refit-1 step: linear channel {d_lin:+.5f}, "
              f"whole model {d_aug:+.5f}, product-channel share {d_aug - d_lin:+.5f}")
        t_p = _hac_mean_t((q["dense_r1"] - q["dense_r48"]) - (q["aug_r1"] - q[ref]))
        print(f"product-increment cadence effect DM-t: {t_p:+.2f}")
    gate = next((r_ for r_ in rows if r_["arm"] == "aug_r1"), None)
    if gate:
        print(f"\nPRE-REGISTERED GATE (aug refit-1 vs refit-48, QLIKE DM-t > 2): "
              f"{gate['dm_t_vs_r48']:+.2f} -> {'PASS' if gate['dm_t_vs_r48'] > 2 else 'fail'}")
    pd.DataFrame(rows).to_csv(f"{OUT}/refit_cadence.csv", index=False)
    print(f"wrote {OUT}/refit_cadence.csv")


def stage_saturate() -> None:
    _require_fixed_cache()
    os.makedirs(OUT, exist_ok=True)
    from src.features.transforms.target import PERIODS_PER_DAY

    p = load_panel()
    sig, pred_har, e = _load_arms()
    y_adj = p.y[2 * TW :]
    baseline = p.baseline[2 * TW :]
    ts = pd.Series(pd.to_datetime(p.t[2 * TW :]))
    late = (ts >= HOLDOUT).to_numpy()
    vol_bar = (ts.dt.strftime("%Y-%m-%d") == VOLMAGEDDON).to_numpy()

    # best-cadence arm if the sweep has run, else the §18.1 daily deliverable
    key = "aug_r1" if "aug_r1" in sig else "aug_r48"
    lin_key = "dense_r1" if key == "aug_r1" else "dense_r48"
    inc = sig[key] - sig[lin_key]  # the product increment — the channel that shouts
    W = 21 * PERIODS_PER_DAY
    s_t = pd.Series(inc).rolling(W, min_periods=W // 2).std().shift(1).bfill().to_numpy()
    sd_e = float(e.std())

    q0 = _qlike_series(pred_har + sig[key], y_adj, baseline)
    print(f"saturating the product increment of {key} (trailing sd, tanh bound)\n"
          f"{'arm':12s} {'QLIKE':>8s} {'dQL':>9s} {'DM-t':>6s} {'2020+ t':>8s} "
          f"{'max|inc|/sd':>12s} {'volmageddon dQL':>16s}")
    rows = []
    amp0 = float(np.nanmax(np.abs(inc)) / sd_e)
    print(f"{'unsaturated':12s} {float(np.nanmean(q0)):8.5f} {'':>9s} {'':>6s} {'':>8s} "
          f"{amp0:12.2f} {'':>16s}")
    for k in SAT_K:
        cap = k * s_t
        inc_s = cap * np.tanh(inc / np.maximum(cap, 1e-12))
        qs = _qlike_series(pred_har + sig[lin_key] + inc_s, y_adj, baseline)
        d = float(np.nanmean(qs) - np.nanmean(q0))
        t = _hac_mean_t(q0 - qs)  # >0 means saturation improves
        t_l = _hac_mean_t((q0 - qs)[late])
        amp = float(np.nanmax(np.abs(inc_s)) / sd_e)
        dv = float(np.nanmean(qs[vol_bar]) - np.nanmean(q0[vol_bar]))
        binds = float((np.abs(inc) > cap).mean())
        print(f"tanh k={k:.0f}     {float(np.nanmean(qs)):8.5f} {d:+9.5f} {t:6.2f} {t_l:8.2f} "
              f"{amp:12.2f} {dv:+16.4f}", flush=True)
        rows.append({"k": k, "qlike": float(np.nanmean(qs)), "dqlike": d, "dm_t": t,
                     "dm_t_2020plus": t_l, "max_inc_over_sd": amp,
                     "max_inc_over_sd_unsat": amp0, "binds_frac": binds,
                     "volmageddon_dqlike": dv, "base_arm": key})
    prim = next(r_ for r_ in rows if r_["k"] == 3.0)
    ok_avg = abs(prim["dm_t"]) < 2 or prim["dm_t"] > 0
    ok_amp = prim["max_inc_over_sd"] < amp0
    print(f"\nPRE-REGISTERED VERDICT (k=3): avg cost nil-or-better: "
          f"{'pass' if ok_avg else 'FAIL'} (DM-t {prim['dm_t']:+.2f}); "
          f"worst-bar amplitude falls: {'pass' if ok_amp else 'FAIL'} "
          f"({amp0:.2f} -> {prim['max_inc_over_sd']:.2f}) "
          f"-> {'SHIP as tail insurance' if ok_avg and ok_amp else 'REJECT'}")
    pd.DataFrame(rows).to_csv(f"{OUT}/saturation_insurance.csv", index=False)
    print(f"wrote {OUT}/saturation_insurance.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["walk", "score", "saturate"], required=True)
    ap.add_argument("--design", choices=["aug", "dense", "joint"], default="aug")
    ap.add_argument("--refit", type=int, default=1)
    a = ap.parse_args()
    if a.stage == "walk":
        stage_walk(a.design, a.refit)
    elif a.stage == "score":
        stage_score()
    else:
        stage_saturate()
