"""Consolidated fast-engine rerun of the in-flight §29–§30 arms (user: "can we speed this up?").

Replaces three killed slow-engine jobs with one sequential pass on the fast engines, ~40 min
total vs hours. Engine discipline: every comparison is same-engine both arms — the cadence
contrast uses the Sherman–Morrison dual-cadence walk (exact match to the rank-1 engine, max
diff 1e-11, validated); the feature arms use the day-blocked engine with their twins recomputed
on it.

Order:
  1. §29 cadence at H=8   — per-bar vs daily, one SM pass, engine-clean contrast.
  2. §29.2 H=13 rung      — blocked engine; pooled + 10:00-slice readouts.
  3. §30.1 flow-state arm — blocked engine; {cos, sin, log r} of the causal intraday plane.
  4. §30.2 transmission   — blocked engine; 20 antisymmetric lag-1 transmission columns.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.alpha_manifestation import TW  # noqa: E402
from analysis.alpha_panel import load_panel  # noqa: E402
from analysis.cucuringu import _causal_intraday_phase  # noqa: E402
from analysis.map_monitor import _frame_and_scores  # noqa: E402
from analysis.minimal_model import HOLDOUT, _hac_mean_t, _qlike_series  # noqa: E402
from analysis.straddle_horizon import _build_design, _y_horizon  # noqa: E402
from analysis.synthesis import _p  # noqa: E402
from analysis.wf import (  # noqa: E402
    walk_forward_embargo_blocked, walk_forward_embargo_dualcadence,
)
from src.features.transforms.target import PERIODS_PER_DAY  # noqa: E402


def _q(pred_full: np.ndarray, yt: np.ndarray, Bt: np.ndarray) -> np.ndarray:
    m = np.isfinite(pred_full) & np.isfinite(yt)
    q = np.full(len(yt), np.nan)
    q[m] = _qlike_series(pred_full[m], yt[m], Bt[m])
    return q


def _roll_scale(F: np.ndarray) -> np.ndarray:
    sd = pd.DataFrame(F).rolling(250 * PERIODS_PER_DAY, min_periods=1000).std().shift(1)
    med = np.nanmedian(sd.to_numpy(), axis=1, keepdims=True)
    sdv = np.maximum(sd.to_numpy(), 0.1 * np.where(np.isfinite(med), med, 1.0))
    return F / pd.DataFrame(sdv).bfill().to_numpy()


def main() -> None:
    p = load_panel()
    ts = pd.Series(pd.to_datetime(p.t))
    day_codes = pd.factorize(ts.dt.normalize())[0]
    late = (ts >= HOLDOUT).to_numpy()
    X679, XH, _ = _build_design()
    n_all = len(ts)

    # ---- 1. cadence at H=8, engine-clean dual pass -------------------------------------
    yh8, Bh8 = _y_horizon(p, 8)
    pb, dy = walk_forward_embargo_dualcadence(X679, yh8, TW, 8, 3000.0)
    f_pb = np.full(n_all, np.nan)
    f_dy = np.full(n_all, np.nan)
    f_pb[TW:], f_dy[TW:] = pb, dy
    q_pb, q_dy = _q(f_pb, yh8, Bh8), _q(f_dy, yh8, Bh8)
    d = q_dy - q_pb
    print(f"[1] §29 cadence at H=8 (SM dual pass, identical windows):", flush=True)
    print(f"    daily  QLIKE {np.nanmean(q_dy):.5f}   per-bar QLIKE {np.nanmean(q_pb):.5f}")
    g = _hac_mean_t(d[np.isfinite(d)], 496)
    print(f"    per-bar vs daily QLIKE DM {g:+.2f}  "
          f"(2020+ {_hac_mean_t(d[late & np.isfinite(d)], 496):+.2f})   "
          f"gate >= +2.0: {'PASS' if g >= 2.0 else 'FAIL'}", flush=True)
    np.savez_compressed(_p("cadence_h8_fast.npz"), perbar=pb, daily=dy)

    # ---- 2. H=13 rung, blocked engine ---------------------------------------------------
    yh13, Bh13 = _y_horizon(p, 13)
    f_bb = walk_forward_embargo_blocked(XH, yh13, day_codes, 250, 1, 1.0)
    f_679 = walk_forward_embargo_blocked(X679, yh13, day_codes, 250, 1, 3000.0)
    q_bb, q_679 = _q(f_bb, yh13, Bh13), _q(f_679, yh13, Bh13)
    d = q_bb - q_679
    md = np.isfinite(d)
    open_rows = ((ts.dt.hour == 10) & (ts.dt.minute == 0)).to_numpy()
    print(f"\n[2] §29.2 fixed H=13 rung (blocked engine):", flush=True)
    print(f"    backbone {np.nanmean(q_bb):.5f}   679 {np.nanmean(q_679):.5f}")
    print(f"    pooled DM {_hac_mean_t(d[md], 506):+.2f} "
          f"(2020+ {_hac_mean_t(d[md & late], 506):+.2f})")
    print(f"    10:00-slice DM {_hac_mean_t(d[md & open_rows], 63):+.2f} "
          f"(2020+ {_hac_mean_t(d[md & open_rows & late], 63):+.2f})", flush=True)

    # ---- 3. §30.1 flow-state arm at H=8, blocked twin recomputed ------------------------
    f_twin = walk_forward_embargo_blocked(X679, yh8, day_codes, 250, 1, 3000.0)
    phi, rad = _causal_intraday_phase()  # rows 2TW:
    F3 = np.zeros((n_all, 3))
    good = np.isfinite(phi)
    F3[2 * TW + 1 :, 0] = np.where(good, np.cos(phi), 0.0)[:-1]
    F3[2 * TW + 1 :, 1] = np.where(good, np.sin(phi), 0.0)[:-1]
    F3[2 * TW + 1 :, 2] = np.where(good, np.log(rad + 1e-9), 0.0)[:-1]
    f_arm = walk_forward_embargo_blocked(np.hstack([X679, F3]), yh8, day_codes, 250, 1, 3000.0)
    q_twin, q_arm = _q(f_twin, yh8, Bh8), _q(f_arm, yh8, Bh8)
    d = q_twin - q_arm
    act = (F3[:, 0] != 0.0) & np.isfinite(d)
    g = _hac_mean_t(d[act], 496)
    print(f"\n[3] §30.1 flow-state arm at H=8 (blocked engine, twin recomputed):", flush=True)
    print(f"    twin {np.nanmean(q_twin):.5f}   +flow state {np.nanmean(q_arm):.5f}")
    print(f"    QLIKE DM (active span) {g:+.2f}   2020+ {_hac_mean_t(d[act & late], 496):+.2f}"
          f"   gate >= +2.0: {'PASS — falsify' if g >= 2.0 else 'FAIL'}", flush=True)

    # ---- 4. §30.2 lagged transmission arm at H=8 ----------------------------------------
    G20, _, ts_g = _frame_and_scores()
    ng = len(G20)
    TRAIL, REFRESH = 504 * PERIODS_PER_DAY, 63 * PERIODS_PER_DAY
    Ghat = np.zeros((ng, G20.shape[1]))
    for start in range(TRAIL, ng, REFRESH):
        a, b = G20[start - TRAIL : start - 1], G20[start - TRAIL + 1 : start]
        az = (a - a.mean(0)) / (a.std(0) + 1e-12)
        bz = (b - b.mean(0)) / (b.std(0) + 1e-12)
        C = (az.T @ bz) / len(az)
        D = (C - C.T) / 2.0
        end = min(start + REFRESH, ng)
        Ghat[start:end] = G20[start - 1 : end - 1] @ D
    F20 = np.zeros((n_all, G20.shape[1]))
    F20[2 * TW :] = np.nan_to_num(Ghat)
    F20 = _roll_scale(F20)
    F20[~np.isfinite(F20)] = 0.0
    f_tr = walk_forward_embargo_blocked(np.hstack([X679, F20]), yh8, day_codes, 250, 1, 3000.0)
    q_tr = _q(f_tr, yh8, Bh8)
    d = q_twin - q_tr
    act = (np.abs(F20).sum(1) != 0.0) & np.isfinite(d)
    g = _hac_mean_t(d[act], 496)
    print(f"\n[4] §30.2 lagged-transmission arm at H=8 (20 antisym columns):", flush=True)
    print(f"    twin {np.nanmean(q_twin):.5f}   +transmission {np.nanmean(q_tr):.5f}")
    print(f"    QLIKE DM (active span) {g:+.2f}   2020+ {_hac_mean_t(d[act & late], 496):+.2f}"
          f"   gate >= +2.0: {'PASS — falsify' if g >= 2.0 else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
