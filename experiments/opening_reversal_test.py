"""The one real signal from the fair-value work: the OPENING REVERSAL, tested clean, gated, net-of-cost.

fairvalue_test's phase scan found the early move off the 9:30 open reverts hard (corr(dev, next) = -0.121
at bar k1) -- the documented opening-reversal, and the strongest intraday directional signal all session.
Unlike the every-bar strategies it is ONCE PER DAY (low turnover), so the 1-tick cost that killed them is
small here. Test: at bar k=ENTRY_K each day, dev = cumulative return since the open; bet the REST-of-session
move reverts it (target = post-entry return, feature = -dev). Gate + net-of-2-tick daily Sharpe, plain and
x the lagged GEX long-gamma regime. PRE-COMMIT: clears the gate net-of-cost => a real prop-legal edge worth
1-min NQ data; fails => JJ's edge doesn't survive our resolution, buy 1-min to test his execution or drop it.
Writes results/opening_reversal/summary.json.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (repo root + siblings on sys.path)

import json
import os

import numpy as np
import pandas as pd

from flow_dir_pilot import score_dir
from gex_regime_test import lag_regime
from src.evaluation.feature_cv import purged_walk_forward, significance_gate
from vrp_screen_worker import build

ENTRY_K = 1  # enter after the first 2 bars (~10:30), where the reversion signal peaked
TICK = 0.25 / 6000.0  # one ES tick as a return fraction; a round trip = 2 ticks


def daily_readout(dev, ret_rest, regime, use_regime):
    """Once-per-day opening-reversal P&L: position = -sign(dev) (bet reversion), or gated to long-gamma
    days; held to the close; net of a 2-tick round trip. Pooled over purged folds (fit sign on train)."""
    sig = -np.sign(dev)
    if use_regime:
        sig = np.where(regime > 0, sig, 0.0)  # trade only long-gamma days
    gross = sig * ret_rest
    net = gross - np.abs(sig) * 2 * TICK
    return {
        "sharpe_net": float(net.mean() / net.std()) if net.std() > 0 else float("nan"),
        "t_net": float(net.mean() / (net.std() / np.sqrt(len(net))))
        if net.std() > 0
        else float("nan"),
        "mean_net_bp": float(net.mean() * 1e4),
        "corr_dev_restret": float(np.corrcoef(dev, ret_rest)[0, 1]),
        "trades": int((sig != 0).sum()),
    }


def main() -> None:
    os.makedirs("results/opening_reversal", exist_ok=True)
    df, *_ = build()
    t = pd.to_datetime(df["t"])
    hour = t.dt.hour.to_numpy()
    day = t.dt.normalize()
    sess = ((hour >= 9) & (hour <= 15)) & (df["sumret_ind"].to_numpy(float) <= 0)
    sd = pd.DataFrame({"day": day.to_numpy(), "r": df["sumret"].to_numpy(float)})[
        sess
    ].copy()
    sd["k"] = sd.groupby("day").cumcount()
    sd["cum"] = sd.groupby("day")["r"].cumsum()

    # one row/day: deviation-from-open at entry, and the rest-of-session return (the reversion leg)
    piv = sd.pivot_table(index="day", columns="k", values="cum")
    last = piv.ffill(axis=1).iloc[:, -1]
    have = piv[ENTRY_K].notna() & last.notna()
    dev = piv.loc[have, ENTRY_K].to_numpy()
    ret_rest = (
        last[have] - piv.loc[have, ENTRY_K]
    ).to_numpy()  # move after entry (toward/away from open)
    days = pd.DatetimeIndex(piv.index[have])

    reg = lag_regime(pd.read_parquet("results/gex/gex_daily.parquet"), days)
    regime = np.nan_to_num(reg["regime_sign"].to_numpy(), nan=0.0)

    print(
        f"[opening reversal] entry k={ENTRY_K}, n_days={len(dev)}, "
        f"corr(dev, rest-of-day ret)={np.corrcoef(dev, ret_rest)[0, 1]:+.4f}",
        flush=True,
    )

    # GATE the reversion feature (-dev predicts ret_rest) on daily obs
    folds = purged_walk_forward(len(dev), n_folds=5, embargo=0.01)
    base = np.ones((len(dev), 1)) * 1e-9  # ~intercept-only base
    sc = score_dir(base, (-dev).reshape(-1, 1), ret_rest, folds)
    passed, comp = significance_gate(sc, k=2.0)

    res = {
        "n_days": int(len(dev)),
        "entry_k": ENTRY_K,
        "gate_pass": bool(passed),
        "gate": comp,
        "all_days": daily_readout(dev, ret_rest, regime, False),
        "longgamma_only": daily_readout(dev, ret_rest, regime, True),
    }
    a, g = res["all_days"], res["longgamma_only"]
    print(
        f"[GATE] {'PASS' if passed else 'fail'} repl={comp['repl_frac']:.2f} "
        f"foldrepl={comp['fold_repl_frac']:.2f} beats_plac={comp['beats_placebo']}",
        flush=True,
    )
    print(
        f"[all days]      net Sharpe/day={a['sharpe_net']:+.3f} t={a['t_net']:+.2f} "
        f"mean={a['mean_net_bp']:+.3f}bp trades={a['trades']}",
        flush=True,
    )
    print(
        f"[long-gamma only] net Sharpe/day={g['sharpe_net']:+.3f} t={g['t_net']:+.2f} "
        f"mean={g['mean_net_bp']:+.3f}bp trades={g['trades']}",
        flush=True,
    )
    with open("results/opening_reversal/summary.json", "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
