"""Gated intraday test of the dealer-gamma REGIME as a conditioning state on ES direction.

Consumes results/gex/gex_daily.parquet (from gex.py on an OptionMetrics extract), LAGS it one trading
day (the causal contract: regime for day d is built from day d-1's settled OI/surface), merges onto our
intraday bars, and asks -- through the SAME gate that killed our vol levers (purged walk-forward +
circular-shift placebo + replication, objective = directional MSE, reused from flow_dir_pilot /
feature_cv) -- whether the regime carries prop-legal (intraday, flat-by-close) directional value:

  A momentum x regime : does regime x first-bar-return predict the LAST-bar return? (Gao-Han-Li-Zhou
                        intraday momentum CONDITIONED on the gamma switch: trend in short-gamma, fade in
                        long-gamma).
  B gap x regime      : does regime decide whether the post-open session REVERSES or EXTENDS the overnight
                        gap / open move? (the tug-of-war resolved by gamma; the overnight signal as an
                        opening fair-value anchor -- see the 'overnight->open' analysis).
  C pin reversion     : does signed distance-to-pin predict the day return (mean-reversion toward the
                        max-gamma-OI magnet)? sign-free, needs no dealer-sign assumption.

Plus a VALIDITY check (not an alpha test): lagged long-gamma regime should NEGATIVELY correlate with the
day's realized range -- if it doesn't on real data, the GEX build is suspect, stop before trusting A/B/C.

Self-validation (run BEFORE spending WRDS time):
  --selftest : fully synthetic panel with a PLANTED regime x momentum effect -> asserts arm A PASSES
               (proves the gate has POWER to detect a real gamma effect, so a future null is a real null).
  --dry-run  : real bars + synthetic (random) regime -> runs the whole merge/lag/gate pipeline end to end
               (proves the plumbing; expects nulls).
Real: needs results/gex/gex_daily.parquet. Writes results/gex_regime/summary.json.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from flow_dir_pilot import oos_readout, score_dir, zscore_causal
from src.evaluation.feature_cv import purged_walk_forward, significance_gate
from vrp_screen_worker import build

OUT = "results/gex_regime"
REGIME_COLS = [
    "regime_sign",
    "z_gex",
    "z_flip",
    "z_pin",
    "z_charm",
    "z_vanna",
    "z_0dte",
]


def load_daily_panel() -> pd.DataFrame:
    """Bars -> per-trading-day directional aggregates (fabricated ffill-0 bars dropped, causal)."""
    df, *_ = build()
    t = pd.to_datetime(df["t"])
    d = pd.DataFrame(
        {
            "day": t.dt.normalize().to_numpy(),
            "hour": t.dt.hour.to_numpy(),
            "r": df["sumret"].to_numpy(float),
            "sess": ((t.dt.hour >= 9) & (t.dt.hour <= 15)).to_numpy(),
            "fab": (df["sumret_ind"].to_numpy(float) > 0),
        }
    )
    d = d[~d["fab"]]
    gs = d[d["sess"]].groupby("day")["r"]
    panel = pd.DataFrame(
        {
            "open_ret": gs.first(),
            "close_ret": gs.last(),
            "day_ret": gs.sum(),
            "nb": gs.size(),
        }
    ).sort_index()
    panel = panel[panel["nb"] >= 3]
    panel["post_open_ret"] = panel["day_ret"] - panel["open_ret"]
    pre = d[d["hour"] < 9].groupby("day")["r"].sum().reindex(panel.index).fillna(0.0)
    post = d[d["hour"] >= 16].groupby("day")["r"].sum().reindex(panel.index).fillna(0.0)
    panel["overnight_ret"] = (
        post.shift(1).fillna(0.0) + pre
    )  # prior evening + this pre-open
    return panel


def lag_regime(gex: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Align an EOD gex_daily table to the panel dates and LAG one trading day (ffill within reindex so a
    holiday mismatch still uses the most recent PRIOR EOD chain, then shift(1) = causal)."""
    g = gex.copy()
    g.index = pd.to_datetime(g.index).normalize()
    g = g.reindex(index).ffill().shift(1)
    out = pd.DataFrame(index=index)
    out["regime_sign"] = np.sign(g["gex_level"])
    out["z_gex"] = zscore_causal(g["gex_level"].to_numpy(float))
    out["z_flip"] = zscore_causal(g["dist_to_flip"].to_numpy(float))
    out["z_pin"] = zscore_causal(g["dist_to_pin"].to_numpy(float))
    out["z_charm"] = zscore_causal(
        g.get("net_charm", pd.Series(0.0, index=index)).to_numpy(float)
    )
    out["z_vanna"] = zscore_causal(
        g.get("net_vanna", pd.Series(0.0, index=index)).to_numpy(float)
    )
    out["z_0dte"] = zscore_causal(
        g.get("gex_0dte", pd.Series(0.0, index=index)).to_numpy(float)
    )
    return out


def synth_regime(index: pd.DatetimeIndex, seed: int = 0) -> pd.DataFrame:
    """Random but persistent regime (dry-run plumbing check; carries NO real relation to the panel)."""
    rng = np.random.default_rng(seed)
    n = len(index)
    ar = np.zeros(n)
    for i in range(1, n):
        ar[i] = 0.97 * ar[i - 1] + rng.standard_normal()
    out = pd.DataFrame(index=index)
    out["regime_sign"] = np.sign(ar)
    out["z_gex"] = zscore_causal(ar)
    for c in ("z_flip", "z_pin", "z_charm", "z_vanna", "z_0dte"):
        out[c] = zscore_causal(
            np.roll(ar, rng.integers(5, 50)) + rng.standard_normal(n)
        )
    return out


ARMS = [
    (
        "A_momentum_x_regime",
        "close_ret",
        ["open_ret"],
        lambda p, g: np.column_stack(
            [p["open_ret"] * g["regime_sign"], p["open_ret"] * g["z_flip"]]
        ),
    ),
    (
        "B_gap_x_regime",
        "post_open_ret",
        ["overnight_ret", "open_ret"],
        lambda p, g: np.column_stack(
            [p["overnight_ret"] * g["regime_sign"], p["open_ret"] * g["regime_sign"]]
        ),
    ),
    (
        "C_pin_reversion",
        "post_open_ret",
        ["open_ret"],
        lambda p, g: g["z_pin"].to_numpy(float).reshape(-1, 1),
    ),
]


def run_arms(panel: pd.DataFrame, regime: pd.DataFrame) -> dict:
    both = (
        panel.join(regime, how="inner")
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=list(panel.columns) + REGIME_COLS)
    )
    folds = purged_walk_forward(len(both), n_folds=5, embargo=0.01)
    # validity: long-gamma regime should suppress the realized range (negative corr)
    val = float(np.corrcoef(both["regime_sign"], both["day_ret"].abs())[0, 1])
    res: dict = {
        "n_days": int(len(both)),
        "n_folds": len(folds),
        "validity_regime_vs_absret_corr": val,
        "arms": {},
    }
    for name, tgt, base_cols, mk_cand in ARMS:
        y = both[tgt].to_numpy(float)
        Xb = both[base_cols].to_numpy(float)
        cand = np.asarray(mk_cand(both, both), dtype=float)
        sc = score_dir(Xb, cand, y, folds)
        passed, comp = significance_gate(sc, k=2.0)
        rd = oos_readout(Xb, cand, y, folds)
        res["arms"][name] = {
            "target": tgt,
            "gate_pass": bool(passed),
            **comp,
            "readout": rd,
        }
        print(
            f"[{name}] GATE={'PASS' if passed else 'fail'} mean_dMSE={comp['mean']:+.3e} "
            f"ci0={comp['ci_excludes_0']} repl={comp['repl_frac']:.2f} "
            f"beats_plac={comp['beats_placebo']} | corr={rd['oos_corr']:+.4f} "
            f"cov_t={rd['cov_t_hac5']:+.2f} net={rd['net_sign_mean_bp']:+.3f}bp",
            flush=True,
        )
    print(
        f"[validity] corr(regime_sign, |day_ret|) = {val:+.4f}  (expect < 0 if GEX real)",
        flush=True,
    )
    return res


def _selftest() -> None:
    """Fully synthetic panel with a PLANTED regime x open-return effect in close_ret -> arm A must PASS."""
    rng = np.random.default_rng(1)
    n = 4000
    idx = pd.bdate_range("2005-01-03", periods=n)
    ar = np.zeros(n)
    for i in range(1, n):
        ar[i] = 0.97 * ar[i - 1] + rng.standard_normal()
    regime_sign = np.sign(ar)
    open_ret = rng.standard_normal(n) * 0.004
    overnight = rng.standard_normal(n) * 0.003
    close_noise = rng.standard_normal(n) * 0.004
    close_ret = (
        0.35 * regime_sign * open_ret + close_noise
    )  # PLANTED: regime flips the momentum sign
    panel = pd.DataFrame(
        {
            "open_ret": open_ret,
            "close_ret": close_ret,
            "overnight_ret": overnight,
            "day_ret": open_ret + close_ret,
            "post_open_ret": close_ret,
            "nb": 13,
        },
        index=idx,
    )
    regime = pd.DataFrame(
        {
            "regime_sign": regime_sign,
            "z_gex": zscore_causal(ar),
            "z_flip": zscore_causal(ar),
            "z_pin": zscore_causal(rng.standard_normal(n)),
            "z_charm": 0.0,
            "z_vanna": 0.0,
            "z_0dte": 0.0,
        },
        index=idx,
    )
    res = run_arms(panel, regime)
    a = res["arms"]["A_momentum_x_regime"]
    print(
        f"\nSELFTEST: arm A gate_pass={a['gate_pass']} (planted effect must be detected)"
    )
    assert a["gate_pass"], (
        "GATE LACKS POWER: failed to detect a planted regime x momentum effect"
    )
    print("SELFTEST PASS -- the gate detects a real gamma-regime effect.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--gex", default="results/gex/gex_daily.parquet")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
        return
    os.makedirs(OUT, exist_ok=True)
    panel = load_daily_panel()
    if a.dry_run:
        print(f"DRY-RUN: {len(panel)} panel days, synthetic regime (expect nulls)")
        regime = synth_regime(panel.index)
    else:
        if not os.path.exists(a.gex):
            raise SystemExit(
                f"missing {a.gex} -- run gex.py on the OptionMetrics extract first "
                f"(or use --dry-run / --selftest)"
            )
        regime = lag_regime(pd.read_parquet(a.gex), panel.index)
    res = run_arms(panel, regime)
    tag = "dryrun" if a.dry_run else "real"
    with open(f"{OUT}/summary_{tag}.json", "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
