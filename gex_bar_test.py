"""Fork B, DONE CORRECTLY: per-BAR gamma-regime-conditioned momentum/reversion.

The daily open->close arm (gex_regime_test A) was too coarse. The dealer-hedging effect lives bar-to-bar:
in a LONG-gamma regime dealers sell rallies / buy dips -> bar returns MEAN-REVERT; in SHORT-gamma they
chase -> bar returns TREND. That makes the return autocorrelation REGIME-DEPENDENT and SIGN-FLIPPING, so
UNCONDITIONALLY it averages to ~0 -- which is precisely the signed-channel null (commit 5881635). The
null never excluded a conditional edge; it's the signature of a regime mixture. The -0.30 validity
(regime controls the intraday RANGE) says the action is in the bar structure. So test it there.

Design: lag the daily GEX regime one trading day (causal), broadcast it to every 30-min SESSION bar of the
day (regime is a slow state, constant intraday), predict the NEXT bar's signed return. Two things:
  1. DIAGNOSTIC (the user's hypothesis, direct): corr(r_t, r_{t+1}) computed SEPARATELY in long- vs
     short-gamma regimes. Opposite signs => the autocorrelation flips with gamma => the unconditional null
     is a mixture, and a regime-conditioned strategy has content.
  2. GATED arms: candidate = recent-bar-return x regime (does the momentum/reversion coefficient flip by
     regime?), scored through feature_cv's gate (fwd-return MSE) with the net-of-1-tick sign-strategy P&L
     as the economic readout (bar strategies are turnover-heavy; the conditioning must clear costs).

Writes results/gex_bar/summary.json.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from flow_dir_pilot import cov_t_hac, oos_readout, score_dir
from gex_regime_test import lag_regime
from src.evaluation.feature_cv import purged_walk_forward, significance_gate
from vrp_screen_worker import build

OUT = "results/gex_bar"
LAGS = (
    1,
    2,
    3,
    6,
)  # recent bar returns (30-min bars): the momentum/reversion predictors


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    df, *_ = build()
    t = pd.to_datetime(df["t"])
    r = df["sumret"].to_numpy(float)
    fab = df["sumret_ind"].to_numpy(float) > 0
    hour = t.dt.hour.to_numpy()
    day = t.dt.normalize()
    sess = (hour >= 9) & (hour <= 15)

    # daily GEX regime, lagged one trading day, broadcast to every bar of the day (slow state, causal)
    gex = pd.read_parquet("results/gex/gex_daily.parquet")
    uniq = pd.DatetimeIndex(sorted(day.unique()))
    reg = lag_regime(gex, uniq)
    regime_sign = reg["regime_sign"].reindex(day).to_numpy()
    z_gex = reg["z_gex"].reindex(day).to_numpy()
    z_flip = reg["z_flip"].reindex(day).to_numpy()

    s = pd.Series(r)
    mom = np.column_stack([s.rolling(L, min_periods=1).sum().to_numpy() for L in LAGS])
    r_fwd = np.roll(r, -1)
    same = np.roll(day.to_numpy(), -1) == day.to_numpy()
    ok = (
        sess
        & np.roll(sess, -1)
        & same
        & ~fab
        & ~np.roll(fab, -1)
        & np.isfinite(regime_sign)
        & np.isfinite(r_fwd)
    )
    ok[-1] = False
    idx = np.where(ok)[0]
    y = r_fwd[idx]
    rt = mom[idx, 0]  # r_t
    reg_s = regime_sign[idx]
    zg = z_gex[idx]
    Xmom = mom[idx]

    # ---- DIAGNOSTIC: does the bar autocorrelation flip sign across gamma regimes? ----
    diag = {}
    for lab, m in (
        ("short_gamma", reg_s < 0),
        ("long_gamma", reg_s > 0),
        ("all", np.ones(len(idx), bool)),
    ):
        if m.sum() > 100:
            ac = float(np.corrcoef(rt[m], y[m])[0, 1])
            t_ac = cov_t_hac(rt[m] - rt[m].mean(), y[m])  # HAC t on the lag-1 autocov
            diag[lab] = {
                "n": int(m.sum()),
                "autocorr_r_t_r_next": ac,
                "autocov_t_hac5": t_ac,
            }
    print(
        "[autocorr by regime]  "
        + "   ".join(
            f"{k}:{v['autocorr_r_t_r_next']:+.4f}(t{v['autocov_t_hac5']:+.1f},n{v['n']})"
            for k, v in diag.items()
        ),
        flush=True,
    )

    # ---- GATED arms: is the regime-conditioned momentum/reversion tradable? ----
    folds = purged_walk_forward(len(idx), n_folds=5, embargo=0.01)
    res: dict = {"n_bars": int(len(idx)), "diagnostic": diag, "arms": {}}
    base_rd = oos_readout(Xmom, None, y, folds)  # unconditional momentum baseline
    res["uncond_mom_readout"] = base_rd
    for name, cand in (
        ("regime_sign_x_mom", (Xmom * reg_s[:, None])),
        ("z_gex_x_mom", (Xmom * zg[:, None])),
        ("both", np.column_stack([Xmom * reg_s[:, None], Xmom * zg[:, None]])),
    ):
        sc = score_dir(Xmom, cand, y, folds)
        passed, comp = significance_gate(sc, k=2.0)
        rd = oos_readout(Xmom, cand, y, folds)
        res["arms"][name] = {"gate_pass": bool(passed), **comp, "readout": rd}
        print(
            f"[{name}] GATE={'PASS' if passed else 'fail'} mean_dMSE={comp['mean']:+.2e} "
            f"repl={comp['repl_frac']:.2f} beats_plac={comp['beats_placebo']} | "
            f"corr={rd['oos_corr']:+.4f} cov_t={rd['cov_t_hac5']:+.2f} net={rd['net_sign_mean_bp']:+.3f}bp",
            flush=True,
        )
    print(
        f"[uncond mom] corr={base_rd['oos_corr']:+.4f} cov_t={base_rd['cov_t_hac5']:+.2f} "
        f"net={base_rd['net_sign_mean_bp']:+.3f}bp n={base_rd['n']}",
        flush=True,
    )
    with open(f"{OUT}/summary.json", "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
