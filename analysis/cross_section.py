"""The cross-section that was baked in all along: relational features between market and stocks.

The panel's target is the MARKET's realized variance, and the exog set contains the cross-section's
marginal aggregates (``sumret2_{ew,vw}stock``, ``sumbipow_*``, ``sumpret2_*``). What the linear span
of those columns cannot reach is the *relational* objects between the market and its constituents —
ratios, which in the scaled-sqrt feature space are not linear combinations of anything present. In
SPD-cone language (§19.4): the marginals give the trace of the return covariance matrix; the target
is (approximately) its top eigenvalue; their RATIO is the spectral concentration — realized average
correlation — the first genuinely cross-sectional coordinate of the covariance object.

Six families, each computed as a ratio of same-window rolling sums (a window-realized ratio is the
standard estimator; a bar-level ratio of two noisy variances is not), at the six HAR windows,
shifted one bar so causality matches ``generate_har_features``:

    rc_w    log( MA_w(RV) / MA_w(sumret2_vwstock) )   realized avg correlation / spectral share
    sz_w    log( MA_w(sumret2_ewstock) / MA_w(sumret2_vwstock) )   size tilt of cross-sectional vol
    jm_w    1 − MA_w(sumbipow) / MA_w(RV)             market jump share (bipower = continuous part)
    js_w    1 − MA_w(sumbipow_ewstock) / MA_w(sumret2_ewstock)   average-stock jump share
    um_w    MA_w(sumpret2) / MA_w(RV)                 market upside-semivariance share*
    us_w    MA_w(sumpret2_ewstock) / MA_w(sumret2_ewstock)       stock upside share*

    * semantics of ``pret2`` assumed from naming (positive-return semivariance); the build prints
      the empirical range of the ratio — inside [0, 1] is consistent with that reading, and the
      feature is a well-defined bounded ratio regardless.

36 columns. Mechanisms are literature-standard, stated before scoring: correlation spikes when
diversification fails and predicts vol beyond the level (dispersion/correlation-risk literature);
jump share separates continuous from discontinuous vol dynamics with different persistence;
semivariance asymmetry is the leverage effect's realized form; the size tilt is the breadth of a
vol episode. None of these is linearly spanned by the existing 516 columns, and only crude products
of them are reachable by the §7 channel.

Pre-registration: PRIMARY gate is the block's increment over the daily-refit dense ridge, sqrt-space
DM > 2, with QLIKE reported alongside; SECONDARY is the increment over the full deliverable
(dense + products). Expectation: positive but honestly uncertain against the gate — differences of
scaled sqrt columns approximate log-ratios locally, so part of this information is already inside
the dense ridge; +0.001..0.004 residual R² would match how every other real channel here has paid.

Usage
-----
    C=.../scratchpad/fixed
    ALPHA_PANEL_CACHE=$C python analysis/cross_section.py --stage build
    ALPHA_PANEL_CACHE=$C python analysis/cross_section.py --stage score
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
from analysis.minimal_model import CACHE, PROD_COL_SCALE, _hac_mean_t, _qlike_series  # noqa: E402
from analysis.nl_sparsity import _upper, base_columns  # noqa: E402
from analysis.synthesis import HOLDOUT, _p, _require_fixed_cache  # noqa: E402
from analysis.wf import r2_oos, walk_forward  # noqa: E402
from src.data.loading import ALL_FEATURES, load_raw_data  # noqa: E402
from src.features.extractors.har import resolve_har_lags  # noqa: E402
from src.features.transforms.scaling import rolling_robust_scale  # noqa: E402
from src.features.transforms.target import PERIODS_PER_DAY  # noqa: E402

OUT = "results/alpha_manifestation"
WINDOWS = (1, 5, 25, 125, 625, 3125)
XSEC_CACHE = "xsec_features.npz"
EPS = 1e-12


def stage_build() -> None:
    _require_fixed_cache()
    df = load_raw_data("data", allow_missing=True)
    cols = [c for c in ALL_FEATURES if c in df.columns]
    df[cols] = df[cols].ffill()
    df = df.dropna(subset=["RV"]).reset_index(drop=True)

    def ma(col: str, w: int) -> np.ndarray:
        return df[col].rolling(w, min_periods=max(1, w // 2)).mean().shift(1).to_numpy()

    names, feats = [], []
    ranges = {}
    for w in WINDOWS:
        rv, vw, ew = ma("RV", w), ma("sumret2_vwstock", w), ma("sumret2_ewstock", w)
        bp, bps = ma("sumbipow", w), ma("sumbipow_ewstock", w)
        pm, ps = ma("sumpret2", w), ma("sumpret2_ewstock", w)
        fam = {
            f"rc_{w}": np.log(np.clip(rv, EPS, None) / np.clip(vw, EPS, None)),
            f"sz_{w}": np.log(np.clip(ew, EPS, None) / np.clip(vw, EPS, None)),
            f"jm_{w}": 1.0 - np.clip(bp / np.clip(rv, EPS, None), 0.0, 2.0),
            f"js_{w}": 1.0 - np.clip(bps / np.clip(ew, EPS, None), 0.0, 2.0),
            f"um_{w}": np.clip(pm / np.clip(rv, EPS, None), 0.0, 2.0),
            f"us_{w}": np.clip(ps / np.clip(ew, EPS, None), 0.0, 2.0),
        }
        for k, v in fam.items():
            names.append(k)
            feats.append(v)
            if w == 25:
                fin = v[np.isfinite(v)]
                ranges[k] = (float(np.nanpercentile(fin, 1)), float(np.nanpercentile(fin, 99)))
    F = np.column_stack(feats)
    print("p1..p99 at w=25 (semantics check; upside shares in [0,1] = semivariance reading holds):")
    for k, (a, b) in ranges.items():
        print(f"  {k:8s} [{a:+.3f}, {b:+.3f}]")

    F = F[resolve_har_lags()[-1] :]
    p = load_panel()
    assert len(F) == len(p.X), f"alignment broken: {len(F)} vs {len(p.X)}"
    F = np.nan_to_num(F, nan=0.0)
    Fs = rolling_robust_scale(
        np.ascontiguousarray(F, dtype=np.float64), 250 * PERIODS_PER_DAY
    )
    print(f"scaled block: {Fs.shape}, max|z| {np.abs(Fs).max():.1f}, "
          f"p99.9 {np.percentile(np.abs(Fs), 99.9):.1f}")
    np.savez_compressed(_p(XSEC_CACHE), F=Fs.astype(np.float32), names=np.array(names))
    print(f"wrote {_p(XSEC_CACHE)}")


def stage_score() -> None:
    _require_fixed_cache()
    os.makedirs(OUT, exist_ok=True)
    p = load_panel()
    z = np.load(_p("har_resid.npz"))
    e_full, pred_har = z["e"], z["pred"][TW:]
    e = e_full[TW:]
    sig = dict(np.load(_p(CACHE)))
    s_all = dict(np.load(_p("bucket_signals.npz")))["all"]
    xz = np.load(_p(XSEC_CACHE))
    F = xz["F"].astype(np.float64)[TW:]
    y_adj = p.y[2 * TW :]
    baseline = p.baseline[2 * TW :]
    ts = pd.Series(pd.to_datetime(p.t[2 * TW :]))
    late = (ts >= HOLDOUT).to_numpy()

    lin_cols = np.concatenate([p.cols("value"), p.cols("indicator")])
    XL = np.ascontiguousarray(p.X[TW:, lin_cols], dtype=np.float64)
    frozen = sig["frozen"]
    bc, _ = base_columns(p)
    XB = np.ascontiguousarray(p.X[TW:, bc], dtype=np.float64)
    ii, jj = _upper(XB.shape[1])
    P = XB[:, ii[frozen]] * XB[:, jj[frozen]]
    B = 250 * PERIODS_PER_DAY
    sd = pd.DataFrame(P).rolling(B, min_periods=1000).std().shift(1)
    med = np.nanmedian(sd.to_numpy(), axis=1, keepdims=True)
    sdv = np.maximum(sd.to_numpy(), 0.1 * np.where(np.isfinite(med), med, 1.0))
    P = P / pd.DataFrame(sdv).bfill().to_numpy() * PROD_COL_SCALE

    print(f"scoring: dense 516 + xsec {F.shape[1]} (and + products {P.shape[1]})", flush=True)
    s_dx = walk_forward(np.hstack([XL, F]), e_full, TW, alpha=3000.0,
                        refit_every=PERIODS_PER_DAY)
    s_full = walk_forward(np.hstack([XL, P, F]), e_full, TW, alpha=3000.0,
                          refit_every=PERIODS_PER_DAY)
    s_aug = sig["s_aug_daily"]

    rows = []
    print(f"\n  {'arm':24s} {'resid R2':>9s} {'dR2 vs ref':>11s} {'DM sqrt':>8s} "
          f"{'dQLIKE':>9s} {'DM QL':>6s} {'2020+ DM':>9s}")
    for name, s, ref, refname in (
        ("dense+xsec vs dense", s_dx, s_all, "dense_daily"),
        ("full+xsec vs deliverable", s_full, s_aug, "aug_daily"),
    ):
        q1 = _qlike_series(pred_har + s, y_adj, baseline)
        q0 = _qlike_series(pred_har + ref, y_adj, baseline)
        d = q0 - q1
        r1, r0 = r2_oos(e, s), r2_oos(e, ref)
        t_sq = dm_test(e, s, ref)
        t_ql, t_ql_l = _hac_mean_t(d), _hac_mean_t(d[late])
        t_sq_l = dm_test(e[late], s[late], ref[late])
        print(f"  {name:24s} {r1:+9.5f} {r1 - r0:+11.5f} {t_sq:+8.2f} "
              f"{float(np.nanmean(q1) - np.nanmean(q0)):+9.5f} {t_ql:+6.2f} "
              f"{t_sq_l:+5.2f}/{t_ql_l:+5.2f}")
        rows.append({"arm": name, "ref": refname, "r2": r1, "dr2": r1 - r0,
                     "dm_t_sqrt": t_sq, "dm_t_sqrt_2020plus": t_sq_l,
                     "dqlike": float(np.nanmean(q1) - np.nanmean(q0)),
                     "dm_t_qlike": t_ql, "dm_t_qlike_2020plus": t_ql_l})
    prim = rows[0]
    print(f"\n  PRE-REGISTERED GATE (xsec block over dense, sqrt DM > 2): "
          f"{prim['dm_t_sqrt']:+.2f} -> {'PASS' if prim['dm_t_sqrt'] > 2 else 'fail'}")
    pd.DataFrame(rows).to_csv(f"{OUT}/cross_section_test.csv", index=False)
    print(f"wrote {OUT}/cross_section_test.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["build", "score"], required=True)
    a = ap.parse_args()
    {"build": stage_build, "score": stage_score}[a.stage]()
