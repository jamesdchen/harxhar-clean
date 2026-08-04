"""Deployment test + P(pass): does the Fork-A GEX RV-forecast gain (+0.022 OOS R^2) sharpen the TRADEABLE
VRP edge, and what does P(pass) look like?

The VRP strategy sells implied variance when the predicted premium prem = iv_bar - RV_hat^2 is rich (top of
its trailing distribution); its realized payoff is strike*iv_bar - rv on selected bars. A SHARPER RV
forecast => better selection of genuinely-rich bars => higher P&L. GEX improves RV_hat (gex_rv_test), so
this asks whether that translates into VRP Sharpe and, via eval_sim.simulate_topstep, into P(pass).

Baseline RV_hat = ridge(HAR + best6 survivor bundles). +GEX = same X plus the lagged daily GEX regime
(z_gex, z_flip, |z_gex|) broadcast to every bar. Deployed config: sqrt target, alpha=1, tw=48000, refit=1
every-bar, q=0.5, session 9-15. Reports daily VRP Sharpe (last1000 + common-tail) and a P(pass) curve over
the daily-risk fraction (Topstep 50K rules, block-bootstrap paths), for strike 1.0 and the x0.8 cliff, with
vs without GEX. Writes results/gex_vrp/summary.json.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (repo root + siblings on sys.path)

import json
import os

import numpy as np
import pandas as pd

from eval_sim.adp import simulate_topstep
from eval_sim.config import EvalConfig
from gex_regime_test import lag_regime
from src.models.ridge import fit_predict_ridge
from vrp_screen_worker import build

OUT = "results/gex_vrp"
BEST6 = [
    "sumabsret",
    "sumvolume",
    "vd_voldemand_all_open_only",
    "vd_vvix",
    "sn_stocktwits_sentcount",
    "sn_stocktwits_attention",
]
CFG = EvalConfig(initial=50_000.0, profit_target=3_000.0, max_drawdown=2_000.0)
LEVERS = [
    0.005,
    0.01,
    0.02,
    0.03,
    0.05,
    0.08,
]  # daily-risk fraction of the account (sizing knob)


def forecast(df, har_cols, HD, bundles, gex_extra=None):
    feat = har_cols + [c for b in BEST6 for c in bundles[b]]
    blocks = [df[feat].fillna(0).to_numpy(float), HD]
    if gex_extra is not None:
        blocks.append(gex_extra)
    X = np.column_stack(blocks)
    y = df["sq"].to_numpy(float)
    p = fit_predict_ridge(
        X, y, train_win_periods=48_000, hyperparams=dict(alpha=1.0, _refit_frequency=1)
    )
    pred = np.full(len(df), np.nan)
    pred[len(df) - len(p) :] = p
    return pred


def vrp_daily(pred, iv_bar, rv, hour, date, strike):
    """Daily VRP payoff series (variance units): select bars with rich predicted premium, realize
    strike*iv - rv. Returns (days, daily_pnl)."""
    prem = iv_bar - np.where(np.isfinite(pred), pred, np.nan) ** 2
    sess = (hour >= 9) & (hour <= 15)
    mm = np.isfinite(prem) & np.isfinite(iv_bar - rv) & sess
    q = pd.Series(prem[mm]).rolling(40 * 252).median().shift(1).to_numpy()
    sel = prem[mm] > np.nan_to_num(q, nan=np.inf)
    B = np.where(sel, strike * iv_bar[mm] - rv[mm], 0.0)
    g = pd.DataFrame({"d": date[mm], "p": B}).groupby("d")["p"].sum()
    return g.index.to_numpy(), g.to_numpy()


def sharpe(x):
    return float(x.mean() / x.std()) if x.std() > 0 else float("nan")


def block_boot(rets, n_paths, T, block=10, seed=0):
    rng = np.random.default_rng(seed)
    n = len(rets)
    nb = int(np.ceil(T / block))
    out = np.empty((n_paths, T))
    for i in range(n_paths):
        starts = rng.integers(0, n - block, nb)
        out[i] = np.concatenate([rets[s : s + block] for s in starts])[:T]
    return out


def p_pass_curve(daily_pnl, n_paths=20_000):
    """P(pass) vs daily-risk fraction w: normalize the daily P&L to unit std (so w*W = daily $ risk),
    block-bootstrap paths, run the Topstep 50K barrier (DLL $1k). Sharpe-preserving; w is the sizing."""
    sd = daily_pnl.std()
    if not (sd > 0):
        return {str(w): 0.0 for w in LEVERS}
    unit = daily_pnl / sd  # std 1, mean = Sharpe
    paths = block_boot(unit, n_paths, 250)
    out = {}
    for w in LEVERS:
        # policy leverage so that a +1 std day moves the account by w*W; DLL caps the day at -$1k
        oc, _, _ = simulate_topstep(paths, CFG, float(w), daily_loss_limit=1_000.0)
        out[str(w)] = float((oc == 1).mean())
    return out


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    df, raw, bundles, har_cols, HD, rv, iv_bar, hour, date = build()

    # lagged daily GEX regime broadcast to every bar (causal; pre-1996 / missing -> 0)
    day = pd.to_datetime(df["t"]).dt.normalize()
    reg = lag_regime(
        pd.read_parquet("results/gex/gex_daily.parquet"),
        pd.DatetimeIndex(sorted(day.unique())),
    )
    def bcast(c):
        return np.nan_to_num(reg[c].reindex(day).to_numpy(), nan=0.0)

    gex_extra = np.column_stack(
        [bcast("z_gex"), bcast("z_flip"), np.abs(bcast("z_gex"))]
    )

    pred_base = forecast(df, har_cols, HD, bundles)
    pred_gex = forecast(df, har_cols, HD, bundles, gex_extra=gex_extra)

    common = date >= date[168_000] if len(date) > 168_000 else np.zeros(len(date), bool)
    res: dict = {"config": "Topstep50k(DD2k) best6", "arms": {}}
    for arm, pred in (("baseline", pred_base), ("plus_gex", pred_gex)):
        res["arms"][arm] = {}
        for tag, strike in (("k1.0", 1.0), ("k0.8", 0.8)):
            days, pnl = vrp_daily(pred, iv_bar, rv, hour, date, strike)
            last = pnl[-1000:]
            cmask = (
                np.isin(days, np.unique(date[common]))
                if common.any()
                else np.ones(len(days), bool)
            )
            res["arms"][arm][tag] = {
                "sharpe_last1000": sharpe(last),
                "sharpe_common": sharpe(pnl[cmask]),
                "n_days": int(len(pnl)),
                "p_pass_last1000": p_pass_curve(last),
            }
            r = res["arms"][arm][tag]
            best_w = max(r["p_pass_last1000"], key=lambda k: r["p_pass_last1000"][k])
            print(
                f"[{arm} {tag}] Sh_last1k={r['sharpe_last1000']:+.3f} Sh_common={r['sharpe_common']:+.3f} "
                f"| P(pass) peak={r['p_pass_last1000'][best_w]:.3f} @w={best_w}",
                flush=True,
            )
    with open(f"{OUT}/summary.json", "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
