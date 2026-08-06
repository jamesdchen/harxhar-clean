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


def _build_design(frozen_override=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The §22 one-stage 679-column design (scaled), the raw backbone, and the phase pair."""
    p = load_panel()
    sig = dict(np.load(_p(CACHE)))
    frozen = sig["frozen"] if frozen_override is None else frozen_override
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
    if hb == 13:
        hour = ts.dt.hour.to_numpy()
        minute = ts.dt.minute.to_numpy()
        open_rows = (hour == 10) & (minute == 0)
        print(f"  §29.2 readout — same clean fixed-13 walk, 10:00 rows only "
              f"({open_rows.sum()}): QLIKE DM {_hac_mean_t(d[open_rows], 63):+.2f} "
              f"(2020+ {_hac_mean_t(d[open_rows & late], 63):+.2f})", flush=True)
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


def stage_eod() -> None:
    """§29.1: remaining variance to the close — the 0DTE object, decided anywhere in the day."""
    _require_fixed_cache()
    os.makedirs(OUT, exist_ok=True)
    p = load_panel()
    ts = pd.Series(pd.to_datetime(p.t))
    n = len(ts)
    rv = p.y**2 * p.baseline
    crv = np.concatenate([[0.0], np.cumsum(rv)])
    cb = np.concatenate([[0.0], np.cumsum(p.baseline)])
    close_rows = np.flatnonzero((ts.dt.hour == 16) & (ts.dt.minute == 0).to_numpy())
    pos = np.searchsorted(close_rows, np.arange(n), side="left")
    has = pos < len(close_rows)
    c = np.where(has, close_rows[np.minimum(pos, len(close_rows) - 1)], 0)
    hb = np.where(has, c - np.arange(n) + 1, 0)
    v = crv[c + 1] - crv[np.arange(n)]
    b = cb[c + 1] - cb[np.arange(n)]
    ok = has & (b > 0)
    yh = np.full(n, np.nan)
    Bh = np.full(n, np.nan)
    yh[ok] = np.sqrt(v[ok] / b[ok])
    Bh[ok] = b[ok]
    EMB = 64  # covers the longest overnight-to-next-close label window
    print(f"EOD target: every row labels to its NEXT 16:00 close ({ok.sum()} defined; horizon "
          f"1..{hb[ok].max()} bars); embargo {EMB}", flush=True)
    # the last <EMB rows have no close: fill for the solver only; scoring masks on finite yh
    y_train = pd.Series(yh).ffill().to_numpy()

    X679, XH, _ = _build_design()
    preds = {}
    preds["backbone"] = walk_forward_embargo(XH, y_train, TW, EMB, 1.0, PERIODS_PER_DAY)
    print("  backbone done", flush=True)
    preds["one-stage 679"] = walk_forward_embargo(X679, y_train, TW, EMB, 3000.0,
                                                  PERIODS_PER_DAY)
    print("  679 done", flush=True)

    yh_o, Bh_o = yh[2 * TW :], Bh[2 * TW :]
    hb_o = hb[2 * TW :]
    hour = ts.dt.hour.to_numpy()[2 * TW :]
    late = (ts[2 * TW :] >= HOLDOUT).to_numpy()
    rth = (hour >= 10) & (hour <= 16)
    slices = {
        "all rows": np.isfinite(yh_o),
        "RTH pooled (h 2-13)": np.isfinite(yh_o) & rth & (hb_o <= 13) & (hb_o >= 2),
        "OPEN decision (h=13)": np.isfinite(yh_o) & (hb_o == 13) & (hour == 10),
    }
    qs = {}
    for name, pr in preds.items():
        f = pr[TW:]
        m0 = np.isfinite(f) & np.isfinite(yh_o)
        q = np.full(len(yh_o), np.nan)
        q[m0] = _qlike_series(f[m0], yh_o[m0], Bh_o[m0])
        qs[name] = q
        print(f"\n  {name}:")
        for sl, m in slices.items():
            mm = m & m0
            print(f"    {sl:22s} QLIKE {np.nanmean(q[mm]):.5f}  "
                  f"(2020+ {np.nanmean(q[mm & late]):.5f})  [{mm.sum()} rows]", flush=True)
    d = qs["backbone"] - qs["one-stage 679"]
    print("\n  679 vs backbone, QLIKE DM:")
    for sl, m in slices.items():
        lags = 480 if sl != "OPEN decision (h=13)" else 63  # open slice: 1/day, no overlap
        print(f"    {sl:22s} {_hac_mean_t(d[m], lags):+.2f} "
              f"(2020+ {_hac_mean_t(d[m & late], lags):+.2f})")
    g = _hac_mean_t(d[slices["OPEN decision (h=13)"]], 63)
    print(f"\n  gate (open slice >= +2.0): {'PASS' if g >= 2.0 else 'FAIL'}")
    np.savez_compressed(_p("straddle_eod.npz"),
                        **{k.replace(" ", "_"): v for k, v in preds.items()},
                        y_eod=yh, B_eod=Bh, h_bars=hb)
    print("cached -> straddle_eod.npz")


def stage_overnight() -> None:
    """Tier-1 support (spec: om_0dte_atopen_export_spec.md): integrated RV from each row to the
    NEXT session open (the 10:00-labeled row, exclusive) — at the 16:30-labeled row this is
    exactly the overnight window [16:00 -> 09:30 next], the piece the close-to-close implied
    carries that the §29.1 session object does not."""
    _require_fixed_cache()
    p = load_panel()
    ts = pd.Series(pd.to_datetime(p.t))
    n = len(ts)
    rv = p.y**2 * p.baseline
    crv = np.concatenate([[0.0], np.cumsum(rv)])
    cb = np.concatenate([[0.0], np.cumsum(p.baseline)])
    open_rows = np.flatnonzero((ts.dt.hour == 10) & (ts.dt.minute == 0).to_numpy())
    pos = np.searchsorted(open_rows, np.arange(n), side="right")
    has = pos < len(open_rows)
    o = np.where(has, open_rows[np.minimum(pos, len(open_rows) - 1)], 0)
    v = crv[o] - crv[np.arange(n)]
    b = cb[o] - cb[np.arange(n)]
    ok = has & (b > 0)
    yh = np.full(n, np.nan)
    Bh = np.full(n, np.nan)
    yh[ok] = np.sqrt(v[ok] / b[ok])
    Bh[ok] = b[ok]
    EMB = 64
    print(f"overnight target: {ok.sum()} rows label to next session open; embargo {EMB}",
          flush=True)
    y_train = pd.Series(yh).ffill().to_numpy()
    X679, XH, _ = _build_design()
    preds = {}
    preds["backbone"] = walk_forward_embargo(XH, y_train, TW, EMB, 1.0, PERIODS_PER_DAY)
    print("  backbone done", flush=True)
    preds["one-stage 679"] = walk_forward_embargo(X679, y_train, TW, EMB, 3000.0,
                                                  PERIODS_PER_DAY)
    print("  679 done", flush=True)
    np.savez_compressed(_p("straddle_overnight.npz"),
                        **{k.replace(" ", "_"): v for k, v in preds.items()},
                        y_on=yh, B_on=Bh)
    print("cached -> straddle_overnight.npz")


def stage_flowarm() -> None:
    """§30.1: the intraday-rotation state {cos phi, sin phi, log r}, causal, one-bar lag, as
    three columns on the 679 design at H=8. Gate: QLIKE DM >= +2.0 vs the cached 679 twin."""
    from analysis.cucuringu import _causal_intraday_phase
    _require_fixed_cache()
    p = load_panel()
    hb = 8
    yh, Bh = _y_horizon(p, hb)
    X679, _, _ = _build_design()
    phi, rad = _causal_intraday_phase()  # aligned to panel rows [2*TW, n)
    n_all = p.X.shape[0]
    F = np.zeros((n_all, 3))
    good = np.isfinite(phi)
    F[2 * TW + 1 :, 0] = np.where(good, np.cos(phi), 0.0)[:-1]
    F[2 * TW + 1 :, 1] = np.where(good, np.sin(phi), 0.0)[:-1]
    F[2 * TW + 1 :, 2] = np.where(good, np.log(rad + 1e-9), 0.0)[:-1]
    print(f"flow-state columns active on {good.sum()} bars (of {len(phi)})", flush=True)
    pred = walk_forward_embargo(np.hstack([X679, F]), yh, TW, hb, 3000.0, PERIODS_PER_DAY)
    ref = np.load(_p(f"straddle_ladder_h{hb}.npz"))["one-stage_679"]
    yh_o, Bh_o = yh[2 * TW :], Bh[2 * TW :]
    ts = pd.Series(pd.to_datetime(p.t[2 * TW :]))
    late = (ts >= HOLDOUT).to_numpy()
    act = F[2 * TW :, 0] != 0.0
    lags = 2 * hb + 480
    qs = {}
    _score("679 twin", ref[TW:], yh_o, Bh_o, qs, late)
    _score("679 + flow state", pred[TW:], yh_o, Bh_o, qs, late)
    d = qs["679 twin"] - qs["679 + flow state"]
    g = _hac_mean_t(d[act], lags)
    print(f"\n  QLIKE DM (+flow state vs twin), active span: {g:+.2f}   "
          f"full {_hac_mean_t(d, lags):+.2f}   2020+ {_hac_mean_t(d[late], lags):+.2f}")
    print(f"  gate (>= +2.0): {'PASS — falsification battery owed' if g >= 2.0 else 'FAIL'}")
    np.savez_compressed(_p("straddle_flowarm_h8.npz"), pred=pred)


def stage_redraw() -> None:
    """§32: re-draw the frozen product block on current data (the rot detector's demanded
    action). Same selector as the original block (|IC| via _pair_ic), trailing 3y ending
    2022-10-31; evaluated 2022-11+ walk-forward, old vs new block, fast engine both arms."""
    from analysis.nl_sparsity import _pair_ic
    from analysis.wf import walk_forward_embargo_blocked
    _require_fixed_cache()
    p = load_panel()
    ts = pd.Series(pd.to_datetime(p.t))
    e_full = np.load(_p("har_resid.npz"))["e"]  # rows TW:
    bc, _ = base_columns(p)
    XB = np.ascontiguousarray(p.X[:, bc], dtype=np.float64)
    ii, jj = _upper(XB.shape[1])
    sel = ((ts >= "2019-11-01") & (ts < "2022-11-01")).to_numpy()[TW:]
    Zw = XB[TW:][sel]
    Zw = (Zw - Zw.mean(0)) / (Zw.std(0) + 1e-12)
    ic = _pair_ic(Zw, e_full[sel])[ii, jj]
    new_frozen = np.argsort(-np.abs(ic))[:100]
    old_frozen = dict(np.load(_p(CACHE)))["frozen"]
    overlap = len(set(new_frozen.tolist()) & set(old_frozen.tolist()))
    print(f"re-draw: selected 100 products on 2019-11..2022-10; overlap with 2005-vintage "
          f"block: {overlap}/100", flush=True)

    day_codes = pd.factorize(ts.dt.normalize())[0]
    era = (ts >= "2022-11-01").to_numpy()
    yh8, Bh8 = _y_horizon(p, 8)
    for tname, yt, Bt, lags in (("1-bar", p.y, p.baseline, 480), ("H=8", yh8, Bh8, 496)):
        qs = {}
        for bname, fz in (("old block", None), ("NEW block", new_frozen)):
            X679, _, _ = _build_design(fz)
            pred = walk_forward_embargo_blocked(X679, yt, day_codes, 250, 1, 3000.0)
            m = np.isfinite(pred) & np.isfinite(yt)
            q = np.full(len(yt), np.nan)
            q[m] = _qlike_series(pred[m], yt[m], Bt[m])
            qs[bname] = q
            print(f"  {tname} {bname}: QLIKE all {np.nanmean(q):.5f}   "
                  f"2022-11+ {np.nanmean(q[era]):.5f}", flush=True)
        d = qs["old block"] - qs["NEW block"]
        g = _hac_mean_t(d[era], lags)
        print(f"  {tname}: NEW vs old QLIKE DM, 2022-11+ era: {g:+.2f}   "
              f"(full span, reference only: {_hac_mean_t(d, lags):+.2f})")
        print(f"  gate (>= +2.0): {'PASS' if g >= 2.0 else 'FAIL'}", flush=True)
    np.savez_compressed(_p("redraw_block.npz"), new_frozen=new_frozen, ic=ic[new_frozen])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["ladder", "cadence", "eod", "overnight", "flowarm", "redraw"],
                    required=True)
    ap.add_argument("--hb", type=int, default=8, choices=[4, 8, 13, 16])
    a = ap.parse_args()
    if a.stage == "ladder":
        stage_ladder(a.hb)
    elif a.stage == "cadence":
        stage_cadence()
    elif a.stage == "eod":
        stage_eod()
    elif a.stage == "overnight":
        stage_overnight()
    elif a.stage == "flowarm":
        stage_flowarm()
    else:
        stage_redraw()
