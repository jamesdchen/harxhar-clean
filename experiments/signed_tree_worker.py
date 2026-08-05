"""Signed-return channel, NONLINEAR pass: the ridge null (commit 5881635) does not exclude a
tree-shaped E[r|F] -- the vol edge itself is ~36% higher-order interactions -- so the same two cells
(core = HAR-only, all8 = sparse-flow superset) are re-run through the repo's LightGBM walk-forward
harness on the signed bar-return target. Pre-registered: DEFAULT_LGBM_PARAMS (repo default, no
tuning), tw=48000, refit every 2500 bars (trees cannot refit every bar; cadence is a compute
constraint, stated). Scored on the same bar-hour window panel as signed_channel_worker (Bonferroni
threshold ~2.9). Writes results/signed_channel/{core,all8}_tree.json."""

from __future__ import annotations

import _bootstrap  # noqa: F401  (repo root + siblings on sys.path)

import json
import os

import numpy as np

from signed_channel_worker import corr_block
from src.models.lightgbm import DEFAULT_LGBM_PARAMS, fit_predict_lgbm
from vrp_pnl_worker import SURV
from vrp_screen_worker import build

OUT = "results/signed_channel"


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    df, raw, bundles, har_cols, HD, rv, iv_bar, hour, date = build()
    y = df["sumret"].to_numpy(float)
    fabricated = df["sumret_ind"].to_numpy(float) > 0
    windows = {
        "session9_15": (hour >= 9) & (hour <= 15),
        "open9": hour == 9,
        "mid10_14": (hour >= 10) & (hour <= 14),
        "close15": hour == 15,
        "afterhours16_19": (hour >= 16) & (hour <= 19),
        "overnight20_8": (hour >= 20) | (hour <= 8),
        "all24": np.ones_like(hour, dtype=bool),
    }

    for name, cols in (
        ("core", har_cols),
        ("all8", har_cols + [c for b in SURV for c in bundles[b]]),
    ):
        X = np.column_stack([df[cols].fillna(0).to_numpy(float), HD])
        p = fit_predict_lgbm(
            X,
            y,
            train_win_periods=48_000,
            hyperparams=dict(DEFAULT_LGBM_PARAMS, _refit_frequency=2_500),
        )
        pred = np.full(len(df), np.nan)
        pred[len(df) - len(p) :] = p
        base = np.isfinite(pred) & np.isfinite(y) & ~fabricated

        res: dict = {"cell": f"{name}_tree", "windows": {}}
        for wname, wmask in windows.items():
            mm = base & wmask
            if mm.sum() >= 1000:
                res["windows"][wname] = corr_block(pred[mm], y[mm])
        with open(f"{OUT}/{name}_tree.json", "w") as fh:
            json.dump(res, fh, indent=1)
        wl = "  ".join(
            f"{k}:{v['oos_corr']:+.4f}(t{v['t_cov_hac5']:+.1f})"
            for k, v in res["windows"].items()
        )
        print(f"[{name}_tree] {wl}", flush=True)


if __name__ == "__main__":
    main()
