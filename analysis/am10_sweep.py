"""AM-10: the Romano-Wolf step-down over the §27-§34 claim family — the §24 discipline applied
to everything that passed a single gate since.

Family (each a QLIKE loss-differential vs its own twin, SAME engine within each pair):
  cadence_h8        per-bar vs daily (SM dual-pass cache — engine-exact by construction)
  open_slice_h13    679 vs backbone at fixed H=13, 10:00 rows only (blocked engine)
  smear_means       means-only conditional smear vs trailing-constant (h=1 deliverable)
  transmission_h8   679+20 antisym lag columns vs 679 twin (blocked engine)
  alpha_law_h16     alpha=3e3*16 vs 3e3 (blocked engine)
  phase_h1          679+2 daily-phase columns vs 679 (blocked engine, daily refit, 1-bar target)

Claims live on different supports, so the joint null uses a common-time-axis circular block
bootstrap (1,008-bar months, shared indices across claims) with per-claim nan-support means,
studentized by each claim's own HAC se — cross-claim correlation preserved, per-claim scale
respected. A3 (amplitude-label uplift) is excluded: different loss, and its monetization died
in the §34.2 ablation.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.alpha_law import _blocks  # noqa: E402
from analysis.alpha_manifestation import TW  # noqa: E402
from analysis.alpha_panel import load_panel  # noqa: E402
from analysis.cucuringu import _causal_phase  # noqa: E402
from analysis.map_monitor import _frame_and_scores  # noqa: E402
from analysis.minimal_model import _qlike_series  # noqa: E402
from analysis.multiplicity import BLOCK, N_BOOT  # noqa: E402
from analysis.straddle_horizon import _y_horizon  # noqa: E402
from analysis.synthesis import _p  # noqa: E402
from analysis.wf import walk_forward_embargo_blocked  # noqa: E402
from src.features.transforms.target import PERIODS_PER_DAY  # noqa: E402


def _q(pred_full, yt, Bt):
    m = np.isfinite(pred_full) & np.isfinite(yt)
    q = np.full(len(yt), np.nan)
    q[m] = _qlike_series(pred_full[m], yt[m], Bt[m])
    return q


def build() -> tuple[np.ndarray, list[str]]:
    p = load_panel()
    ts = pd.Series(pd.to_datetime(p.t))
    n = len(ts)
    day_codes = pd.factorize(ts.dt.normalize())[0]
    XH, XL, XS, P = _blocks(p)
    A = 3000.0
    X679 = np.hstack([XH * np.sqrt(A), XL, XS, P * np.sqrt(0.1)])
    ds, names = [], []

    # 1. cadence_h8 (cached SM dual pass)
    yh8, Bh8 = _y_horizon(p, 8)
    z = np.load(_p("cadence_h8_fast.npz"))
    f_pb, f_dy = np.full(n, np.nan), np.full(n, np.nan)
    f_pb[TW:], f_dy[TW:] = z["perbar"], z["daily"]
    ds.append(_q(f_dy, yh8, Bh8) - _q(f_pb, yh8, Bh8))
    names.append("cadence_h8")
    print("cadence_h8 built", flush=True)

    # 2. open_slice_h13
    yh13, Bh13 = _y_horizon(p, 13)
    f_bb = walk_forward_embargo_blocked(XH, yh13, day_codes, 250, 1, 1.0)
    f_679_13 = walk_forward_embargo_blocked(X679, yh13, day_codes, 250, 1, A)
    d13 = _q(f_bb, yh13, Bh13) - _q(f_679_13, yh13, Bh13)
    open_rows = ((ts.dt.hour == 10) & (ts.dt.minute == 0)).to_numpy()
    d13[~open_rows] = np.nan
    ds.append(d13)
    names.append("open_slice_h13")
    print("open_slice_h13 built", flush=True)

    # 3. smear_means (recompute directly on the h=1 deliverable)
    f1 = np.full(n, np.nan)
    f1[TW:] = np.load(_p("final_onestage.npz"))["yhat_bar"]
    y, B = p.y, p.baseline
    e2 = (y - f1) ** 2
    day = ts.dt.normalize()
    a_by_day = pd.Series(e2).groupby(day.values).mean()
    la_full = np.log(a_by_day + 1e-12)
    Xm = pd.DataFrame({"l1": la_full.shift(1), "m5": la_full.shift(1).rolling(5).mean(),
                       "m21": la_full.shift(1).rolling(21).mean()})
    nd = len(la_full)
    ahat = pd.Series(np.nan, index=la_full.index)
    Xv = Xm.to_numpy()
    lav = la_full.to_numpy()
    for start in range(3 * 252 + 63, nd, 63):
        m = np.isfinite(Xv).all(1) & np.isfinite(lav)
        tr = np.flatnonzero(m & (np.arange(nd) < start))
        if len(tr) < 200:
            continue
        b = np.linalg.lstsq(np.c_[np.ones(len(tr)), Xv[tr]], lav[tr], rcond=None)[0]
        seg = np.arange(start, min(start + 63, nd))
        ok = np.isfinite(Xv[seg]).all(1)
        ahat.iloc[seg[ok]] = np.exp(np.c_[np.ones(ok.sum()), Xv[seg][ok]] @ b)
    slot = (ts.dt.hour * 2 + ts.dt.minute // 30).to_numpy()
    rel = pd.Series(e2) / pd.Series(e2).groupby(day_codes).transform("mean")
    ss = np.ones(n)
    for s in np.unique(slot):
        m = slot == s
        cs = pd.Series(np.where(m, rel, np.nan)).expanding(min_periods=100).mean().shift(48)
        ss[m] = cs.to_numpy()[m]
    ss = np.where(np.isfinite(ss), ss, 1.0)
    sm_cond = day.map(dict(zip(la_full.index, ahat))).to_numpy(dtype=float) * ss
    sm_trail = pd.Series(e2).rolling(250 * 48, min_periods=5000).mean().shift(1).to_numpy()
    m0 = np.isfinite(f1) & np.isfinite(y) & (B > 0)
    dsm = np.full(n, np.nan)
    for_both = m0 & np.isfinite(sm_cond) & (sm_cond > 0) & np.isfinite(sm_trail) & (sm_trail > 0)
    tr_raw = y**2 * B
    for sm, sign in ((sm_trail, +1.0), (sm_cond, -1.0)):
        pr = (f1**2 + sm) * B
        r = tr_raw[for_both] / pr[for_both]
        qv = r - np.log(r) - 1.0
        dsm[for_both] = (0.0 if sign > 0 else dsm[for_both])
        if sign > 0:
            dsm[for_both] = qv
        else:
            dsm[for_both] = dsm[for_both] - qv
    ds.append(dsm)
    names.append("smear_means")
    print("smear_means built", flush=True)

    # 4. transmission_h8
    f_twin8 = walk_forward_embargo_blocked(X679, yh8, day_codes, 250, 1, A)
    G20, _, _ = _frame_and_scores()
    ng = len(G20)
    TRAIL, REFRESH = 504 * PERIODS_PER_DAY, 63 * PERIODS_PER_DAY
    Ghat = np.zeros((ng, G20.shape[1]))
    for start in range(TRAIL, ng, REFRESH):
        a, b = G20[start - TRAIL : start - 1], G20[start - TRAIL + 1 : start]
        az = (a - a.mean(0)) / (a.std(0) + 1e-12)
        bz = (b - b.mean(0)) / (b.std(0) + 1e-12)
        C = (az.T @ bz) / len(az)
        D = (C - C.T) / 2.0
        Ghat[start : min(start + REFRESH, ng)] = G20[start - 1 : min(start + REFRESH, ng) - 1] @ D
    F20 = np.zeros((n, G20.shape[1]))
    F20[2 * TW :] = np.nan_to_num(Ghat)
    sd = pd.DataFrame(F20).rolling(250 * PERIODS_PER_DAY, min_periods=1000).std().shift(1)
    med = np.nanmedian(sd.to_numpy(), axis=1, keepdims=True)
    F20 = F20 / pd.DataFrame(np.maximum(sd.to_numpy(),
                                        0.1 * np.where(np.isfinite(med), med, 1.0))).bfill().to_numpy()
    F20[~np.isfinite(F20)] = 0.0
    f_tr = walk_forward_embargo_blocked(np.hstack([X679, F20]), yh8, day_codes, 250, 1, A)
    dtr = _q(f_twin8, yh8, Bh8) - _q(f_tr, yh8, Bh8)
    dtr[np.abs(F20).sum(1) == 0.0] = np.nan
    ds.append(dtr)
    names.append("transmission_h8")
    print("transmission_h8 built", flush=True)

    # 5. alpha_law_h16
    yh16, Bh16 = _y_horizon(p, 16)
    f_un = walk_forward_embargo_blocked(X679, yh16, day_codes, 250, 1, A)
    A16 = A * 16
    X16 = np.hstack([XH * np.sqrt(A16), XL, XS, P * np.sqrt(0.1)])
    f_law = walk_forward_embargo_blocked(X16, yh16, day_codes, 250, 1, A16)
    ds.append(_q(f_un, yh16, Bh16) - _q(f_law, yh16, Bh16))
    names.append("alpha_law_h16")
    print("alpha_law_h16 built", flush=True)

    # 6. phase_h1
    f_679_1 = walk_forward_embargo_blocked(X679, p.y, day_codes, 250, 1, A)
    phi, dcodes_g, ndays = _causal_phase()
    ph_lag = np.full(ndays, np.nan)
    ph_lag[1:] = phi[:-1]
    PH = np.zeros((n, 2))
    PH[2 * TW :, 0] = np.nan_to_num(np.cos(ph_lag))[dcodes_g]
    PH[2 * TW :, 1] = np.nan_to_num(np.sin(ph_lag))[dcodes_g]
    f_ph = walk_forward_embargo_blocked(np.hstack([X679, PH]), p.y, day_codes, 250, 1, A)
    dph = _q(f_679_1, p.y, p.baseline) - _q(f_ph, p.y, p.baseline)
    dph[PH[:, 0] == 0.0] = np.nan
    ds.append(dph)
    names.append("phase_h1")
    print("phase_h1 built", flush=True)

    D = np.column_stack(ds)
    np.savez_compressed(_p("am10_lossdiffs.npz"), D=D, names=np.array(names))
    return D, names


def sweep(D: np.ndarray, names: list[str],
          csv_path: str = "results/alpha_manifestation/am10_sweep.csv",
          title: str = "AM-10 Romano-Wolf step-down over the §27-§34 family") -> None:
    n, J = D.shape
    fin = np.isfinite(D)
    mu = np.array([D[fin[:, j], j].mean() for j in range(J)])

    def hac_se(x):
        x = x[np.isfinite(x)]
        xc = x - x.mean()
        s = xc @ xc
        for L in range(1, 481):
            s += 2.0 * (1.0 - L / 481.0) * (xc[L:] @ xc[:-L])
        return np.sqrt(max(s, 1e-300)) / len(x)

    se = np.array([hac_se(D[:, j]) for j in range(J)])
    t = mu / se
    Dc = D - mu  # recentre within support
    rng = np.random.default_rng(0)
    boot_t = np.empty((N_BOOT, J))
    for b in range(N_BOOT):
        starts = rng.integers(0, n, size=int(np.ceil(n / BLOCK)))
        idx = ((starts[:, None] + np.arange(BLOCK)[None, :]).ravel() % n)[:n]
        Db = Dc[idx]
        with np.errstate(invalid="ignore"):
            boot_t[b] = np.nanmean(Db, axis=0) / se
    order = np.argsort(-t)
    adj = np.ones(J)
    remaining = list(order)
    prev = 0.0
    while remaining:
        j = remaining[0]
        m = np.array(remaining)
        pval = float(np.mean(np.nanmax(boot_t[:, m], axis=1) >= t[j]))
        prev = max(prev, pval)
        adj[j] = prev
        remaining = remaining[1:]
    tab = pd.DataFrame({"claim": names, "mean_dQLIKE_1e4": mu * 1e4, "dm_t": t,
                        "rw_adj_p": adj}).sort_values("dm_t", ascending=False)
    print(f"\n{title} ({N_BOOT} draws, {BLOCK}-bar blocks, joint time axis):")
    print(tab.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    tab.to_csv(csv_path, index=False)
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    if os.path.exists(_p("am10_lossdiffs.npz")):
        z = np.load(_p("am10_lossdiffs.npz"), allow_pickle=True)
        D, names = z["D"], list(z["names"])
        print("loaded cached loss diffs", flush=True)
    else:
        D, names = build()
    sweep(D, names)
