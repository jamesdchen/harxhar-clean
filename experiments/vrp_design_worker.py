"""VRP DESIGN-choice factorial (CARC, runs concurrently with the Hoffman2 column screen): 2^5 over the
model-config axes the hand-rolled-vs-research deflation exposed, at both bundle extremes. Determines the
config stage-2 should run at.

Factors (16 fits; refit=1 EVERY BAR always; selection quantile evaluated within each fit):
  A target    : sqrt | log with CAUSAL multiplicative debias (expanding mean of rv/exp(pred), shifted)
  B window    : 48000 (1000d) | 144000 (3000d)
  C alpha     : 1 | 30
  E bundles   : none (RV-HAR + hour dummies only) | all 38 on
Responses per cell x quantile {0.5, 0.7}: full / last1000 Sharpe, + x0.8-strike variants.
One SLURM task = one fit. Writes results/vrp_design/cell_{tid:03d}.json.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (repo root + siblings on sys.path)

import itertools
import json
import os

import numpy as np
import pandas as pd

from src.models.ridge import fit_predict_ridge
from vrp_screen_worker import build

OUT = "results/vrp_design"
GRID = list(
    itertools.product(("sq", "log"), (48_000, 144_000), (1.0, 30.0), (0, 1))
)  # refit=1 ALWAYS (every-bar, per standing instruction)


def main() -> None:
    tid = int(
        os.environ.get("SLURM_ARRAY_TASK_ID", os.environ.get("SGE_TASK_ID", 1)) or 0
    )
    if "SGE_TASK_ID" in os.environ and "SLURM_ARRAY_TASK_ID" not in os.environ:
        tid -= 1
    A, B, C, E = GRID[tid]
    os.makedirs(OUT, exist_ok=True)

    df, raw, bundles, har_cols, HD, rv, iv_bar, hour, date = build()
    if A == "log":
        y = np.log(rv + 1e-16)
        ladder_src = pd.Series(y, index=df.index)
        # log-target ladder: rebuild MAs of log-RV for the CORE (bundle ladders stay as built on levels)
        core = np.column_stack(
            [
                ladder_src.rolling(lag, min_periods=1)
                .mean()
                .shift(1)
                .fillna(0)
                .to_numpy()
                for lag in (1, 5, 25, 125, 625, 3125)
            ]
        )
    else:
        y = df["sq"].to_numpy(float)
        core = df[har_cols].fillna(0).to_numpy(float)
    feat = [core, HD] + (
        [
            np.column_stack(
                [df[c].fillna(0).to_numpy(float) for b in raw for c in bundles[b]]
            )
        ]
        if E
        else []
    )
    X = np.column_stack(feat)
    p = fit_predict_ridge(
        X, y, train_win_periods=B, hyperparams=dict(alpha=C, _refit_frequency=1)
    )
    pred = np.full(len(df), np.nan)
    pred[len(df) - len(p) :] = p

    if A == "log":
        e = np.exp(pred)
        ratio = (
            pd.Series(rv / np.where(e > 0, e, np.nan))
            .expanding()
            .mean()
            .shift(1)
            .to_numpy()
        )
        rvhat = e * np.nan_to_num(ratio, nan=1.0)
    else:
        rvhat = np.where(np.isfinite(pred), pred, np.nan) ** 2
    prem = iv_bar - rvhat
    vrp = iv_bar - rv
    sess = (hour >= 9) & (hour <= 15)
    mm = np.isfinite(prem) & np.isfinite(vrp) & sess
    res = {"tid": tid, "cfg": dict(target=A, window=B, alpha=C, refit=1, bundles=E)}
    for qq in (0.5, 0.7):
        thr = pd.Series(prem[mm]).rolling(40 * 252).quantile(qq).shift(1).to_numpy()
        sel = prem[mm] > np.nan_to_num(thr, nan=np.inf)
        for tag, strike in (("", 1.0), ("_hc", 0.8)):
            Bp = np.where(sel, strike * iv_bar[mm] - rv[mm], 0.0)
            dd = (
                pd.DataFrame({"d": date[mm], "p": Bp})
                .groupby("d")["p"]
                .sum()
                .to_numpy()
            )
            res[f"q{qq}_sh_full{tag}"] = float(dd.mean() / dd.std())
            res[f"q{qq}_sh_last1000{tag}"] = float(dd[-1000:].mean() / dd[-1000:].std())
    with open(f"{OUT}/cell_{tid:03d}.json", "w") as fh:
        json.dump(res, fh)
    print(f"tid={tid} cfg={res['cfg']} q0.5_full={res['q0.5_sh_full']:+.4f}")


if __name__ == "__main__":
    main()
