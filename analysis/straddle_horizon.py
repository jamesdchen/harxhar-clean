"""§29: intraday multi-horizon ladder for the straddle product — pre-registered in the writeup.

Scope per the user's ruling (re-confirmed 2026-08-06): INTRADAY straddles, H < 48 bars. This
module inherits the msweep-2026-08-01 conventions rather than re-deriving them — most
importantly the HONEST LABEL convention: at horizon H the training window is [t-H-W, t-H), so
every label's forward window is realized by decision time (their overlap leak invalidated a
whole table; `walk_forward_embargo` below is that lesson as code).

Targets: y_H = sqrt(sum RV / sum baseline) over the next H bars, from the panel's own winsorized
reconstruction (RV_bar = y^2 * baseline). QLIKE at horizon uses the exact Duan reconstruction
with the integrated baseline. HAC lags 2H + 480 everywhere (overlapping targets).

``ladder --hb {4,8,16}``
    (a) backbone (27 cols, alpha=1) and (c) the one-stage 679 (§22 penalties, recorded caveat:
    no per-horizon retuning, so the top rung is a floor under the msweep alpha-law). At H=8 only,
    (d) +2 causal phase columns as a CONTROL — a daily-scale cycle should not pay inside the day.

``cadence``
    (e) H=8 only: the 679 design refit per bar vs daily — the study's biggest h=1 lever (+3.56),
    never tested at product horizons (msweep solved daily blocks). Gate: QLIKE DM >= +2.0.
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
from analysis.cucuringu import _causal_phase  # noqa: E402
from analysis.minimal_model import (  # noqa: E402
    CACHE, HOLDOUT, _hac_mean_t, _qlike_series, _require_fixed_cache, _upper,
)
from analysis.nl_sparsity import base_columns  # noqa: E402
from analysis.synthesis import _p  # noqa: E402
from analysis.wf import r2_oos  # noqa: E402
from src.features.transforms.target import PERIODS_PER_DAY  # noqa: E402
from src.models.rolling_least_squares import RollingLeastSquares  # noqa: E402

OUT = "results/alpha_manifestation"


def _y_horizon(p, hb: int) -> tuple[np.ndarray, np.ndarray]:
    """Integrated-vol target over the next ``hb`` bars and its integrated baseline."""
    rv = p.y**2 * p.baseline
    crv = np.concatenate([[0.0], np.cumsum(rv)])
    cb = np.concatenate([[0.0], np.cumsum(p.baseline)])
    n = len(rv)
    yh = np.full(n, np.nan)
    Bh = np.full(n, np.nan)
    yh[: n - hb + 1] = np.sqrt((crv[hb:] - crv[:-hb]) / (cb[hb:] - cb[:-hb]))
    Bh[: n - hb + 1] = cb[hb:] - cb[:-hb]
    return yh, Bh


def walk_forward_embargo(X: np.ndarray, y: np.ndarray, train_win: int, embargo: int,
                         alpha: float, refit_every: int) -> np.ndarray:
    """Rolling ridge with honest labels: prediction at t trains on rows [t-embargo-W, t-embargo)
    so every training label's forward window closes by decision time. Returns predictions
    aligned like ``walk_forward`` (length n - train_win, NaN before train_win + embargo)."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    n = len(X)
    solver = RollingLeastSquares(alpha=alpha, fit_intercept=True)
    solver.init_window(X[:train_win], y[:train_win])
    solver.solve()
    out = np.full(n - train_win, np.nan)
    for step, t in enumerate(range(train_win + embargo, n)):
        j = t - embargo  # newest row whose label window [j, j+H) has closed
        solver.roll(X[j], y[j], X[j - train_win], y[j - train_win])
        if step % refit_every == 0:
            solver.solve()
        out[t - train_win] = solver.predict_one(X[t])
    return out


def _build_design() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The §22 one-stage 679-column design (scaled), the raw backbone, and the phase pair."""
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
    X679 = np.hstack([XH * np.sqrt(ALPHA / 1.0), XL, XS, P * np.sqrt(ALPHA / 3e4)])

    phi, day_codes, nd = _causal_phase()
    ph_lag = np.full(nd, np.nan)
    ph_lag[1:] = phi[:-1]
    PH = np.zeros((p.X.shape[0], 2))
    PH[2 * TW :, 0] = np.nan_to_num(np.cos(ph_lag))[day_codes]
    PH[2 * TW :, 1] = np.nan_to_num(np.sin(ph_lag))[day_codes]
    return X679, XH, PH


def _score(name: str, pred: np.ndarray, yh: np.ndarray, Bh: np.ndarray,
           qs: dict, late: np.ndarray) -> None:
    m = np.isfinite(pred) & np.isfinite(yh)
    q = np.full(len(yh), np.nan)
    q[m] = _qlike_series(pred[m], yh[m], Bh[m])
    qs[name] = q
    yc = yh - np.nanmean(yh)
    print(f"  {name:22s} R2 {r2_oos(yc[m], pred[m] - np.nanmean(yh)):+.5f}  "
          f"QLIKE {np.nanmean(q):.5f}  (2020+ {np.nanmean(q[late]):.5f})  "
          f"[{m.sum()} bars]", flush=True)


def stage_ladder(hb: int) -> None:
    _require_fixed_cache()
    os.makedirs(OUT, exist_ok=True)
    p = load_panel()
    yh, Bh = _y_horizon(p, hb)
    X679, XH, PH = _build_design()
    lags = 2 * hb + 480
    print(f"H={hb} bars: honest-label walks (embargo {hb}), 250d window, daily refit, "
          f"HAC lags {lags}", flush=True)

    preds = {}
    preds["backbone"] = walk_forward_embargo(XH, yh, TW, hb, 1.0, PERIODS_PER_DAY)
    print("  backbone done", flush=True)
    preds["one-stage 679"] = walk_forward_embargo(X679, yh, TW, hb, 3000.0, PERIODS_PER_DAY)
    print("  679 done", flush=True)
    if hb == 8:
        preds["679 + phase"] = walk_forward_embargo(
            np.hstack([X679, PH]), yh, TW, hb, 3000.0, PERIODS_PER_DAY)
        print("  phase arm done", flush=True)

    yh_o, Bh_o = yh[2 * TW :], Bh[2 * TW :]
    ts = pd.Series(pd.to_datetime(p.t[2 * TW :]))
    late = (ts >= HOLDOUT).to_numpy()
    qs = {}
    for name, pr in preds.items():
        _score(name, pr[TW:], yh_o, Bh_o, qs, late)

    d = qs["backbone"] - qs["one-stage 679"]
    print(f"\n  679 vs backbone: QLIKE DM {_hac_mean_t(d, lags):+.2f} "
          f"(2020+ {_hac_mean_t(d[late], lags):+.2f})", flush=True)
    if hb == 8:
        act = PH[2 * TW :, 0] != 0.0
        dp = qs["one-stage 679"] - qs["679 + phase"]
        g = _hac_mean_t(dp[act], lags)
        print(f"  phase control (active span): QLIKE DM {g:+.2f}  "
              f"-> {'suspicious PASS — falsify' if g >= 2.0 else 'no-add, as expected'}",
              flush=True)
    np.savez_compressed(_p(f"straddle_ladder_h{hb}.npz"),
                        **{k.replace(" ", "_"): v for k, v in preds.items()})
    print(f"cached predictions -> straddle_ladder_h{hb}.npz")


def stage_cadence() -> None:
    _require_fixed_cache()
    p = load_panel()
    hb = 8
    yh, Bh = _y_horizon(p, hb)
    X679, _, _ = _build_design()
    lags = 2 * hb + 480
    print(f"H={hb}: 679 design refit PER BAR (honest labels) vs the daily-refit twin", flush=True)
    pb = walk_forward_embargo(X679, yh, TW, hb, 3000.0, 1)
    ref = np.load(_p(f"straddle_ladder_h{hb}.npz"))["one-stage_679"]
    yh_o, Bh_o = yh[2 * TW :], Bh[2 * TW :]
    ts = pd.Series(pd.to_datetime(p.t[2 * TW :]))
    late = (ts >= HOLDOUT).to_numpy()
    qs = {}
    _score("679 daily refit", ref[TW:], yh_o, Bh_o, qs, late)
    _score("679 PER-BAR refit", pb[TW:], yh_o, Bh_o, qs, late)
    d = qs["679 daily refit"] - qs["679 PER-BAR refit"]
    g = _hac_mean_t(d, lags)
    m = np.isfinite(pb[TW:]) & np.isfinite(yh_o)
    print(f"\n  per-bar vs daily at H=8: QLIKE DM {g:+.2f} (2020+ {_hac_mean_t(d[late], lags):+.2f})"
          f"   sqrt DM {dm_test(yh_o[m], pb[TW:][m], ref[TW:][m]):+.2f}")
    print(f"  gate (>= +2.0): {'PASS' if g >= 2.0 else 'FAIL'}")
    np.savez_compressed(_p("straddle_cadence_h8.npz"), perbar=pb)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["ladder", "cadence"], required=True)
    ap.add_argument("--hb", type=int, default=8, choices=[4, 8, 16])
    a = ap.parse_args()
    if a.stage == "ladder":
        stage_ladder(a.hb)
    else:
        stage_cadence()
